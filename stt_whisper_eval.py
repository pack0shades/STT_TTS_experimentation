"""Evaluate Whisper (optionally with a LoRA adapter) on a JSONL dataset.

Expected JSONL format:
  {"audio": "path/to.wav", "text": "reference transcript", ...}

Example:
  /home/pragay/WWAI/.venv/bin/python stt_whisper_eval.py \
    --data-jsonl datasets/sales_synth_melotts/train.jsonl \
    --model openai/whisper-small \
    --language en --task transcribe \
    --max-samples 50

With LoRA adapter saved by stt_whisper_lora_finetune.py:
  /home/pragay/WWAI/.venv/bin/python stt_whisper_eval.py \
    --data-jsonl datasets/sales_synth_melotts/train.jsonl \
    --model openai/whisper-small \
    --adapter-dir outputs/whisper-small-lora-quick \
    --language en --task transcribe \
    --max-samples 200
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import tts_common as common

logger = logging.getLogger(__name__)


def _require(pkg: str, hint: str) -> None:
    raise RuntimeError(f"Missing dependency '{pkg}'. Install: {hint}")


def _resolve_audio_path(p: str, *, audio_root: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((audio_root / pp).resolve())


def main(argv: list[str] | None = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--data-jsonl", type=str, required=True)
    p.add_argument("--audio-root", type=str, default=".")

    p.add_argument("--model", type=str, default="openai/whisper-small")
    p.add_argument("--adapter-dir", type=str, default="", help="Optional PEFT adapter dir")

    p.add_argument("--language", type=str, default="en")
    p.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])

    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--generation-max-length", type=int, default=225)
    p.add_argument("--save-preds", type=str, default="", help="Optional JSONL output with predictions")

    args = p.parse_args(argv)

    try:
        import torch
    except Exception:
        _require("torch", "Install torch for your CUDA/CPU setup")

    try:
        from datasets import Audio, load_dataset
    except Exception:
        _require(
            "datasets",
            "uv pip install -U datasets --python /home/pragay/WWAI/.venv/bin/python",
        )

    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception:
        _require(
            "transformers",
            "uv pip install -U transformers accelerate --python /home/pragay/WWAI/.venv/bin/python",
        )

    try:
        from jiwer import cer, wer
    except Exception:
        _require("jiwer", "uv pip install -U jiwer --python /home/pragay/WWAI/.venv/bin/python")

    adapter_dir = Path(args.adapter_dir).resolve() if str(args.adapter_dir).strip() else None

    device: str
    if str(args.device) == "cpu":
        device = "cpu"
    elif str(args.device) == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_root = Path(args.audio_root).resolve()

    ds = load_dataset("json", data_files={"data": str(Path(args.data_jsonl).resolve())})

    def _fix_paths(batch: dict[str, Any]) -> dict[str, Any]:
        batch["audio"] = _resolve_audio_path(batch["audio"], audio_root=audio_root)
        return batch

    ds = ds.map(_fix_paths)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    if int(args.max_samples) > 0 and len(ds["data"]) > int(args.max_samples):
        ds["data"] = ds["data"].shuffle(seed=0).select(range(int(args.max_samples)))

    processor = WhisperProcessor.from_pretrained(str(adapter_dir) if adapter_dir else args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # Decoder prompt for correct language/task
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=str(args.language), task=str(args.task))
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except Exception:
            _require("peft", "uv pip install -U peft --python /home/pragay/WWAI/.venv/bin/python")
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.to(device)
    model.eval()

    use_amp = bool(args.fp16) and device == "cuda"

    preds: list[str] = []
    refs: list[str] = []

    save_path = Path(args.save_preds).resolve() if str(args.save_preds).strip() else None
    f_out = open(save_path, "w", encoding="utf-8") if save_path else None

    try:
        bs = max(1, int(args.batch_size))
        items = ds["data"]

        for start in range(0, len(items), bs):
            chunk = items.select(range(start, min(start + bs, len(items))))
            wavs = [np.asarray(a["array"], dtype=np.float32) for a in chunk["audio"]]
            # ensure mono
            wavs = [w.mean(axis=1) if w.ndim > 1 else w for w in wavs]

            inputs = processor.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else contextlib.nullcontext()
                )
                with ctx:
                    generated_ids = model.generate(
                        input_features,
                        max_length=int(args.generation_max_length),
                    )

            pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred_t, ref_t, audio_p in zip(pred_texts, chunk["text"], chunk["audio"]):
                pred_t = str(pred_t).strip()
                ref_t = str(ref_t).strip()
                preds.append(pred_t)
                refs.append(ref_t)
                if f_out is not None:
                    f_out.write(
                        json.dumps(
                            {
                                "audio": audio_p.get("path"),
                                "ref": ref_t,
                                "pred": pred_t,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if f_out is not None:
            f_out.close()

    metrics = {
        "n": len(refs),
        "wer": float(wer(refs, preds)) if refs else None,
        "cer": float(cer(refs, preds)) if refs else None,
        "model": str(args.model),
        "adapter_dir": str(adapter_dir) if adapter_dir else "",
        "device": device,
    }

    common.log_json(logger, {"event": "eval_done", **metrics})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

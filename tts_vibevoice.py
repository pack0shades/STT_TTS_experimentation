"""VibeVoice realtime TTS runner (local).

This wraps the VibeVoice community/official-style Python package with the same
small CLI shape used by the other TTS comparison scripts in this repo.

Example:
  python tts_vibevoice.py --text "Hello" --speaker-name Emma --no-play --wav-out out.wav
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_SR = 24000


@dataclass
class VibeVoiceHandle:
    processor: object
    model: object
    voice_prefill: object
    sr: int
    device: str
    speaker_name: str


def _candidate_vibevoice_repos(explicit_repo: str = "") -> list[Path]:
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if explicit_repo:
        candidates.append(Path(explicit_repo).expanduser())
    candidates.extend(
        [
            here / "VibeVoice",
            here.parent / "VibeVoice",
            Path("/home/pragay/WWAI/VibeVoice"),
        ]
    )
    return candidates


def _ensure_vibevoice_on_sys_path(explicit_repo: str = "") -> list[str]:
    checked: list[str] = []
    for candidate in _candidate_vibevoice_repos(explicit_repo):
        checked.append(str(candidate))
        if (candidate / "vibevoice").is_dir():
            p = str(candidate)
            if p not in sys.path:
                sys.path.insert(0, p)
            break
    return checked


def _import_vibevoice(explicit_repo: str = "") -> tuple[object, object]:
    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # type: ignore
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (  # type: ignore
            VibeVoiceStreamingProcessor,
        )

        return VibeVoiceStreamingProcessor, VibeVoiceStreamingForConditionalGenerationInference
    except Exception as first_error:
        checked = _ensure_vibevoice_on_sys_path(explicit_repo)
        for key in list(sys.modules):
            if key == "vibevoice" or key.startswith("vibevoice."):
                sys.modules.pop(key, None)
        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # type: ignore
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (  # type: ignore
                VibeVoiceStreamingProcessor,
            )

            return VibeVoiceStreamingProcessor, VibeVoiceStreamingForConditionalGenerationInference
        except Exception as e:
            raise RuntimeError(
                "VibeVoice is not importable. Install the VibeVoice repo/package "
                "(for example: git clone https://github.com/vibevoice-community/VibeVoice.git "
                "then `uv pip install -e ./VibeVoice`) or pass --vibevoice-repo. "
                f"Checked: {checked}."
            ) from (first_error or e)


def _pick_device(*, use_gpu: bool) -> str:
    if use_gpu:
        return "cuda"
    return "cpu"


def _load_dtype_and_attention(device: str) -> tuple[object, str]:
    import torch

    if device == "cuda":
        return torch.bfloat16, "flash_attention_2"
    if device == "mps":
        return torch.float32, "sdpa"
    return torch.float32, "sdpa"


def _default_voices_dirs(vibevoice_repo: str = "") -> list[Path]:
    here = Path(__file__).resolve().parent
    dirs: list[Path] = []
    for repo in _candidate_vibevoice_repos(vibevoice_repo):
        dirs.append(repo / "demo" / "voices" / "streaming_model")
    dirs.extend(
        [
            here / "demo" / "voices" / "streaming_model",
            here / "voices" / "streaming_model",
        ]
    )
    return dirs


def _scan_voice_presets(voices_dir: Path) -> dict[str, Path]:
    if not voices_dir.exists():
        return {}

    presets: dict[str, Path] = {}
    for pt in sorted(voices_dir.glob("*.pt")):
        name = pt.stem
        presets[name] = pt
        alias = name
        if "_" in alias:
            alias = alias.split("_")[0]
        if "-" in alias:
            alias = alias.split("-")[-1]
        presets.setdefault(alias, pt)
    return presets


def _resolve_voice_preset(
    *,
    speaker_name: str,
    voice_preset: str,
    voices_dir: str,
    vibevoice_repo: str,
) -> Path:
    if voice_preset:
        p = Path(voice_preset).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Voice preset not found: {p}")
        return p

    dirs = [Path(voices_dir).expanduser()] if voices_dir else _default_voices_dirs(vibevoice_repo)
    checked: list[str] = []
    presets: dict[str, Path] = {}
    for d in dirs:
        checked.append(str(d))
        presets.update(_scan_voice_presets(d))

    if not presets:
        raise RuntimeError(
            "No VibeVoice streaming voice presets were found. Pass --voice-preset /path/to/voice.pt "
            "or --voices-dir pointing at VibeVoice/demo/voices/streaming_model. "
            f"Checked: {checked}."
        )

    if speaker_name in presets:
        return presets[speaker_name]

    wanted = speaker_name.lower()
    for name, path in presets.items():
        if wanted in name.lower() or name.lower() in wanted:
            return path

    default = sorted(set(presets.values()), key=lambda p: str(p))[0]
    logger.warning("No preset matched speaker '%s'; using %s", speaker_name, default)
    return default


def _load_vibevoice(
    *,
    use_gpu: bool,
    model_path: str,
    speaker_name: str,
    voice_preset: str = "",
    voices_dir: str = "",
    vibevoice_repo: str = "",
    ddpm_steps: int = 5,
) -> tuple[VibeVoiceHandle, float]:
    import torch

    processor_cls, model_cls = _import_vibevoice(vibevoice_repo)

    device = _pick_device(use_gpu=use_gpu)
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    load_dtype, attn_impl = _load_dtype_and_attention(device)

    t0 = perf_counter()
    processor = processor_cls.from_pretrained(model_path)
    try:
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map=device,
            attn_implementation=attn_impl,
        )
    except Exception:
        if attn_impl != "flash_attention_2":
            raise
        logger.warning("flash_attention_2 load failed; retrying VibeVoice with sdpa attention")
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map=device,
            attn_implementation="sdpa",
        )
    model.eval()
    if hasattr(model, "set_ddpm_inference_steps"):
        model.set_ddpm_inference_steps(num_steps=int(ddpm_steps))

    preset_path = _resolve_voice_preset(
        speaker_name=speaker_name,
        voice_preset=voice_preset,
        voices_dir=voices_dir,
        vibevoice_repo=vibevoice_repo,
    )
    voice_prefill = torch.load(str(preset_path), map_location=device, weights_only=False)
    init_s = perf_counter() - t0

    common.log_json(
        logger,
        {
            "event": "vibevoice_loaded",
            "model_path": model_path,
            "device": device,
            "speaker_name": speaker_name,
            "voice_preset": str(preset_path),
            "sr": DEFAULT_SR,
            "ddpm_steps": int(ddpm_steps),
            "init_s": round(float(init_s), 3),
        },
    )
    return (
        VibeVoiceHandle(
            processor=processor,
            model=model,
            voice_prefill=voice_prefill,
            sr=DEFAULT_SR,
            device=device,
            speaker_name=speaker_name,
        ),
        float(init_s),
    )


def _tensor_audio_to_numpy(audio: object) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(audio):
            arr = audio.detach().float().cpu().numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)
    except Exception:
        arr = np.asarray(audio, dtype=np.float32)
    return common.to_mono_float32(arr)


def synthesize_vibevoice(
    *,
    handle: VibeVoiceHandle,
    text: str,
    cfg_scale: float = 1.5,
) -> tuple[np.ndarray, int, float]:
    import torch

    full_script = str(text).replace("\u2019", "'").strip()
    if not full_script:
        return np.zeros((0,), dtype=np.float32), handle.sr, 0.0

    inputs = handle.processor.process_input_with_cached_prompt(
        text=full_script,
        cached_prompt=handle.voice_prefill,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(handle.device)

    t0 = perf_counter()
    with torch.inference_mode():
        outputs = handle.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=float(cfg_scale),
            tokenizer=handle.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(handle.voice_prefill),
        )
    gen_s = perf_counter() - t0

    speech_outputs = getattr(outputs, "speech_outputs", None)
    if not speech_outputs or speech_outputs[0] is None:
        raise RuntimeError("VibeVoice did not return speech output")

    return _tensor_audio_to_numpy(speech_outputs[0]), int(handle.sr), float(gen_s)


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--model-path", type=str, default=os.environ.get("VIBEVOICE_MODEL_PATH", DEFAULT_MODEL_PATH))
    p.add_argument("--vibevoice-repo", type=str, default=os.environ.get("VIBEVOICE_REPO", ""))
    p.add_argument("--speaker-name", type=str, default="Emma")
    p.add_argument("--voice-preset", type=str, default="", help="Path to a VibeVoice streaming .pt voice preset")
    p.add_argument("--voices-dir", type=str, default="", help="Directory containing streaming_model .pt voice presets")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--ddpm-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="Accepted for CLI compatibility; VibeVoice ignores it")
    p.add_argument("--gpu", action="store_true", help="Force GPU (CUDA) if available")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0)
    args = p.parse_args(argv)

    if float(args.speed) != 1.0:
        logger.warning("--speed is not exposed by the VibeVoice realtime API and will be ignored")

    if args.seed:
        import torch

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    if bool(args.cpu):
        use_gpu = False
    elif bool(args.gpu):
        use_gpu = True
    else:
        use_gpu = common.gpu_is_available()

    common.log_json(logger, {"device": "cuda" if use_gpu else "cpu"})

    handle, init_s = _load_vibevoice(
        use_gpu=use_gpu,
        model_path=str(args.model_path),
        speaker_name=str(args.speaker_name),
        voice_preset=str(args.voice_preset),
        voices_dir=str(args.voices_dir),
        vibevoice_repo=str(args.vibevoice_repo),
        ddpm_steps=int(args.ddpm_steps),
    )

    audio, sr, gen_s = synthesize_vibevoice(handle=handle, text=str(args.text), cfg_scale=float(args.cfg_scale))
    metrics = common.compute_metrics(gen_s, audio, sr)

    payload = {
        "model": "vibevoice-realtime-0.5b",
        "streaming": False,
        "samplerate": sr,
        "init_s": round(float(init_s), 3),
        "gen_s": round(metrics.gen_s, 3),
        "ttfc_s": round(metrics.gen_s, 3),
        "ttft_s": round(metrics.gen_s, 3),
        "audio_s": round(metrics.audio_s, 3),
        "rtf": round(metrics.rtf, 3),
        "speaker_name": str(args.speaker_name),
        "cfg_scale": float(args.cfg_scale),
        "ddpm_steps": int(args.ddpm_steps),
    }
    common.log_json(logger, payload)

    marks = []
    if bool(args.print_marks) or not args.no_play:
        phonemes = common.text_to_phonemes_g2p_en(args.text)
        marks = common.build_fallback_marks(phonemes, metrics.audio_s, jitter_s=float(args.jitter), seed=0)

    if args.wav_out:
        common.save_wav_pcm16(args.wav_out, audio, sr)
        common.log_json(logger, {"wav_out": args.wav_out})

    if args.print_marks:
        for m in marks:
            common.log_json(logger, {"t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme})

    if not args.no_play:
        common.play_audio_with_marks(audio, sr, marks, chunk_size=int(args.chunk), logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

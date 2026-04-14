"""Coqui-TTS VITS runner with phoneme->viseme marks.

This is the cleaned-up version of your current `VITS.py`, but parameterized and
structured for side-by-side model comparisons.

Example:
  python tts_vits_coqui.py --text "Hello world" --play --print-marks
"""

from __future__ import annotations

import argparse
import logging
import re
from time import perf_counter
from typing import Optional

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)


def _infer_sr(tts_obj, fallback: int = 22050) -> int:
    # Coqui internals vary; try to get the true output SR if available.
    for attr_path in (
        ("synthesizer", "output_sample_rate"),
        ("synthesizer", "output_sample_rate"),
    ):
        obj = tts_obj
        ok = True
        for a in attr_path:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and isinstance(obj, (int, float)) and obj:
            return int(obj)
    return int(fallback)


def _load_coqui_tts(model_name: str, use_gpu: bool) -> tuple[object, int, float]:
    from TTS.api import TTS  # type: ignore

    t0 = perf_counter()
    tts = TTS(model_name=model_name, gpu=bool(use_gpu))
    init_s = perf_counter() - t0
    sr = _infer_sr(tts, fallback=22050)
    return tts, int(sr), float(init_s)


def _split_text_streaming(text: str, *, max_chars: int = 200) -> list[str]:
    """Best-effort chunking for pseudo-streaming.

    Coqui VITS does not expose a native incremental audio generator, so we
    synthesize text pieces sequentially and stream playback per piece.
    """

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on sentence-ish boundaries first
    parts = re.split(r"(?<=[\.!\?;:])\s+", text)
    parts = [p.strip() for p in parts if p and p.strip()]

    # Then enforce max_chars
    out: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            # fallback: split by words
            cur: list[str] = []
            cur_len = 0
            for w in p.split(" "):
                if not w:
                    continue
                add = (1 if cur else 0) + len(w)
                if cur and (cur_len + add) > max_chars:
                    out.append(" ".join(cur))
                    cur = [w]
                    cur_len = len(w)
                else:
                    cur.append(w)
                    cur_len += add
            if cur:
                out.append(" ".join(cur))
    return out


def _synthesize_piecewise(
    *,
    tts: object,
    sr: int,
    text_pieces: list[str],
    play: bool,
    gap_ms: int,
    emit_marks: bool,
    marks_offset_s: float,
) -> tuple[np.ndarray, float, float]:
    """Return (audio, ttft_s, gen_s).

    Note: In --stream mode, writing to a sounddevice OutputStream is blocking and
    effectively "real-time". For benchmarking/model comparisons, we treat gen_s
    as *synthesis/inference time only* (excluding playback blocking).
    """

    if not text_pieces:
        return np.zeros((0,), dtype=np.float32), 0.0, 0.0

    gap = np.zeros((int(sr * (gap_ms / 1000.0)),), dtype=np.float32) if gap_ms > 0 else None

    stream = None
    if play:
        try:
            import sounddevice as sd  # type: ignore

            stream = sd.OutputStream(samplerate=int(sr), channels=1, dtype="float32")
            stream.start()
        except Exception as e:
            raise RuntimeError(
                "sounddevice is required for streaming playback. Install it (pip install sounddevice)."
            ) from e

    pieces_audio: list[np.ndarray] = []
    wall_t0 = perf_counter()
    ttft_s: Optional[float] = None

    synth_s = 0.0

    audio_pos_s = float(marks_offset_s)

    try:
        for i, piece in enumerate(text_pieces):
            t_piece0 = perf_counter()
            audio_piece = getattr(tts, "tts")(piece)
            synth_s += float(perf_counter() - t_piece0)
            audio_piece = common.to_mono_float32(np.array(audio_piece, dtype=np.float32))

            if ttft_s is None:
                ttft_s = perf_counter() - wall_t0

            pieces_audio.append(audio_piece)

            if emit_marks and audio_piece.size:
                try:
                    ph = common.text_to_phonemes_g2p_en(piece)
                    local = common.build_fallback_marks(ph, float(audio_piece.size) / float(sr), seed=0)
                    for m in common.offset_marks(local, audio_pos_s):
                        common.log_json(
                            logger,
                            {"event": "mark", "t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme},
                        )
                except Exception as e:
                    logger.warning("Failed to emit marks for chunk (%s)", e)

            audio_pos_s += float(audio_piece.size) / float(sr) if audio_piece.size else 0.0
            if gap is not None and i != (len(text_pieces) - 1):
                pieces_audio.append(gap)
                audio_pos_s += float(gap.size) / float(sr) if gap.size else 0.0

            if stream is not None and audio_piece.size:
                stream.write(audio_piece)
                if gap is not None and i != (len(text_pieces) - 1) and gap.size:
                    stream.write(gap)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    wall_s = perf_counter() - wall_t0
    audio = np.concatenate(pieces_audio) if pieces_audio else np.zeros((0,), dtype=np.float32)
    # Return gen_s as synthesis-only time for apples-to-apples comparisons.
    return audio, float(ttft_s or wall_s), float(synth_s)


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--model", type=str, default="tts_models/en/ljspeech/vits")
    p.add_argument("--gpu", action="store_true", help="Force GPU (CUDA) if available")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--stream", action="store_true", help="Pseudo-stream by chunking text and synthesizing sequentially")
    p.add_argument("--stream-max-chars", type=int, default=200, help="Max chars per synthesized chunk in --stream mode")
    p.add_argument("--stream-gap-ms", type=int, default=0, help="Silence gap between streamed chunks")
    p.add_argument("--stream-marks", action="store_true", help="Emit phoneme/viseme marks during --stream mode")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0, help="Seconds of jitter for fallback scheduling")
    args = p.parse_args(argv)

    if bool(args.cpu):
        use_gpu = False
    elif bool(args.gpu):
        use_gpu = True
    else:
        # default: use GPU when available
        use_gpu = common.gpu_is_available()

    common.log_json(logger, {"device": "cuda" if use_gpu else "cpu"})

    # 1) Init (excluded from TTFT)
    tts, sr, init_s = _load_coqui_tts(args.model, use_gpu)

    # 2) Synthesize
    play_s: Optional[float] = None

    if args.stream:
        pieces = _split_text_streaming(args.text, max_chars=int(args.stream_max_chars))
        wall_t0 = perf_counter()
        audio, ttft_s, gen_s = _synthesize_piecewise(
            tts=tts,
            sr=sr,
            text_pieces=pieces,
            play=(not args.no_play),
            gap_ms=int(args.stream_gap_ms),
            emit_marks=bool(args.stream_marks),
            marks_offset_s=0.0,
        )
        wall_s = perf_counter() - wall_t0
        play_s = float(max(0.0, float(wall_s) - float(gen_s)))
    else:
        t0 = perf_counter()
        audio = getattr(tts, "tts")(args.text)
        gen_s = perf_counter() - t0
        ttft_s = float(gen_s)
        audio = common.to_mono_float32(np.array(audio, dtype=np.float32))

    metrics = common.compute_metrics(gen_s, audio, sr)
    payload = {
        "model": args.model,
        "streaming": bool(args.stream),
        "samplerate": sr,
        "init_s": round(float(init_s), 3),
        # gen_s is synthesis-only time in --stream mode; this keeps comparisons fair.
        "gen_s": round(metrics.gen_s, 3),
        "ttfc_s": round(float(ttft_s), 3),
        "ttft_s": round(float(ttft_s), 3),
        "audio_s": round(metrics.audio_s, 3),
        "rtf": round(metrics.rtf, 3),
    }
    if play_s is not None:
        payload["play_s"] = round(float(play_s), 3)
    common.log_json(logger, payload)

    # 2) Marks (fallback schedule)
    phonemes = common.text_to_phonemes_g2p_en(args.text)
    marks = common.build_fallback_marks(phonemes, metrics.audio_s, jitter_s=float(args.jitter), seed=0)

    # 3) Output
    if args.wav_out:
        common.save_wav_pcm16(args.wav_out, audio, sr)
        common.log_json(logger, {"wav_out": args.wav_out})

    if args.print_marks and args.no_play:
        # print marks once without realtime playback
        for m in marks:
            common.log_json(logger, {"t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme})

    if (not args.no_play) and (not args.stream):
        common.play_audio_with_marks(audio, sr, marks, chunk_size=int(args.chunk), logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

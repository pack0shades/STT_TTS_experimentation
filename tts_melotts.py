"""MeloTTS runner (local).

MeloTTS has a few different Python APIs depending on the repo/version you install.
This script tries to import common entry points, and otherwise tells you what to do.

Once you confirm which MeloTTS package/repo you installed, I can tighten this up
into a fully-working implementation.

Example:
  python tts_melotts.py --text "Hello" --no-play --wav-out out.wav
"""

from __future__ import annotations

import argparse
import logging
from time import perf_counter
from typing import Optional

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)


def _pick_speaker_id(tts_obj, speaker: str, speaker_id: int) -> int:
    if speaker_id >= 0:
        return int(speaker_id)

    # Try mapping if provided by MeloTTS
    spk2id = None
    try:
        spk2id = getattr(getattr(getattr(tts_obj, "hps"), "data"), "spk2id", None)
    except Exception:
        spk2id = None

    if isinstance(spk2id, dict) and spk2id:
        if speaker:
            if speaker in spk2id:
                return int(spk2id[speaker])
            raise RuntimeError(
                f"Unknown speaker '{speaker}'. Available: {sorted(spk2id.keys())[:20]}"
            )
        # default: first speaker (stable ordering)
        first_key = sorted(spk2id.keys())[0]
        return int(spk2id[first_key])

    # Fallback: assume single-speaker or id=0
    return 0


def _load_melotts(
    *,
    use_gpu: bool,
    language: str,
    speaker: str,
    speaker_id: int,
) -> tuple[object, int, int, float]:
    """Load MeloTTS once. Returns (tts, spk_id, sr, init_s)."""

    try:
        from melo.api import TTS  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MeloTTS not importable. Ensure the MeloTTS repo/package is installed/available (melo.api.TTS)."
        ) from e

    device = "cuda" if use_gpu else "cpu"
    t0 = perf_counter()
    tts = TTS(language=language, device=device)
    init_s = perf_counter() - t0

    spk_id = _pick_speaker_id(tts, speaker=speaker, speaker_id=speaker_id)
    sr = int(getattr(getattr(tts, "hps"), "data").sampling_rate)
    return tts, int(spk_id), int(sr), float(init_s)


def _split_melotts_text(text: str, language: str) -> list[str]:
    """Use MeloTTS' own sentence splitting (but without printing)."""

    try:
        from melo.split_utils import split_sentence  # type: ignore

        return [t for t in split_sentence(text, language_str=language) if t and str(t).strip()]
    except Exception:
        # fallback: very simple split
        return [t.strip() for t in text.split(".") if t.strip()]


def _synthesize_melotts_piecewise(
    *,
    tts: object,
    spk_id: int,
    sr: int,
    pieces: list[str],
    speed: float,
    quiet: bool,
    play: bool,
    gap_ms: int,
    emit_marks: bool,
    marks_offset_s: float,
    language: str,
) -> tuple[np.ndarray, float, float]:
    """Return (audio, ttft_s, gen_s).

    In --stream mode, writing to sounddevice is blocking (real-time). For
    benchmarking/model comparisons we report gen_s as *synthesis time only*
    (excluding playback blocking).
    """

    if not pieces:
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

    wall_t0 = perf_counter()
    ttft_s: Optional[float] = None
    audio_parts: list[np.ndarray] = []

    synth_s = 0.0

    audio_pos_s = float(marks_offset_s)

    try:
        for i, piece in enumerate(pieces):
            t_piece0 = perf_counter()
            audio_piece = getattr(tts, "tts_to_file")(
                piece,
                spk_id,
                output_path=None,
                speed=float(speed),
                quiet=bool(quiet),
            )
            synth_s += float(perf_counter() - t_piece0)
            audio_piece = common.to_mono_float32(np.array(audio_piece, dtype=np.float32))

            if ttft_s is None:
                ttft_s = perf_counter() - wall_t0

            audio_parts.append(audio_piece)

            if emit_marks and audio_piece.size:
                # Best-effort: use g2p_en only for English.
                if str(language).upper().startswith("EN"):
                    try:
                        ph = common.text_to_phonemes_g2p_en(piece)
                        local = common.build_fallback_marks(ph, float(audio_piece.size) / float(sr), seed=0)
                        for m in common.offset_marks(local, audio_pos_s):
                            common.log_json(
                                logger,
                                {"event": "mark", "t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme},
                            )
                    except Exception as e:
                        logger.warning("Failed to emit marks for sentence (%s)", e)
                else:
                    logger.warning("--stream-marks currently supports EN only (g2p_en). language=%s", language)

            audio_pos_s += float(audio_piece.size) / float(sr) if audio_piece.size else 0.0
            if gap is not None and i != (len(pieces) - 1):
                audio_parts.append(gap)
                audio_pos_s += float(gap.size) / float(sr) if gap.size else 0.0

            if stream is not None and audio_piece.size:
                stream.write(audio_piece)
                if gap is not None and i != (len(pieces) - 1) and gap.size:
                    stream.write(gap)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    wall_s = perf_counter() - wall_t0
    audio = np.concatenate(audio_parts) if audio_parts else np.zeros((0,), dtype=np.float32)
    return audio, float(ttft_s or wall_s), float(synth_s)


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--language", type=str, default="EN")
    p.add_argument("--speaker", type=str, default="", help="Speaker name (if model provides spk2id)")
    p.add_argument("--speaker-id", type=int, default=-1, help="Numeric speaker id override")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--quiet", action="store_true", help="Reduce MeloTTS console output")
    p.add_argument("--gpu", action="store_true", help="Force GPU (CUDA) if available")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-play", action="store_true")
    p.add_argument(
        "--stream",
        action="store_true",
        help="Pseudo-stream by splitting into sentences and synthesizing sequentially",
    )
    p.add_argument("--stream-gap-ms", type=int, default=50, help="Silence gap between streamed sentences")
    p.add_argument("--stream-marks", action="store_true", help="Emit phoneme/viseme marks during --stream mode (EN only)")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0)
    args = p.parse_args(argv)

    if bool(args.cpu):
        use_gpu = False
    elif bool(args.gpu):
        use_gpu = True
    else:
        use_gpu = common.gpu_is_available()

    common.log_json(logger, {"device": "cuda" if use_gpu else "cpu"})

    tts, spk_id, sr, init_s = _load_melotts(
        use_gpu=use_gpu,
        language=str(args.language),
        speaker=str(args.speaker),
        speaker_id=int(args.speaker_id),
    )

    play_s: Optional[float] = None

    if args.stream:
        pieces = _split_melotts_text(args.text, str(args.language))
        wall_t0 = perf_counter()
        audio, ttft_s, gen_s = _synthesize_melotts_piecewise(
            tts=tts,
            spk_id=spk_id,
            sr=sr,
            pieces=pieces,
            speed=float(args.speed),
            quiet=True if args.stream else bool(args.quiet),
            play=(not args.no_play),
            gap_ms=int(args.stream_gap_ms),
            emit_marks=bool(args.stream_marks),
            marks_offset_s=0.0,
            language=str(args.language),
        )
        wall_s = perf_counter() - wall_t0
        play_s = float(max(0.0, float(wall_s) - float(gen_s)))
    else:
        t0 = perf_counter()
        audio = getattr(tts, "tts_to_file")(
            args.text,
            spk_id,
            output_path=None,
            speed=float(args.speed),
            quiet=bool(args.quiet),
        )
        gen_s = perf_counter() - t0
        ttft_s = float(gen_s)
        audio = common.to_mono_float32(np.array(audio, dtype=np.float32))
    metrics = common.compute_metrics(gen_s, audio, sr)

    payload = {
        "model": "melotts",
        "streaming": bool(args.stream),
        "samplerate": sr,
        "init_s": round(float(init_s), 3),
        # gen_s is synthesis-only time in --stream mode.
        "gen_s": round(metrics.gen_s, 3),
        "ttfc_s": round(float(ttft_s), 3),
        "ttft_s": round(float(ttft_s), 3),
        "audio_s": round(metrics.audio_s, 3),
        "rtf": round(metrics.rtf, 3),
    }
    if play_s is not None:
        payload["play_s"] = round(float(play_s), 3)
    common.log_json(logger, payload)

    phonemes = common.text_to_phonemes_g2p_en(args.text)
    marks = common.build_fallback_marks(phonemes, metrics.audio_s, jitter_s=float(args.jitter), seed=0)

    if args.wav_out:
        common.save_wav_pcm16(args.wav_out, audio, sr)
        common.log_json(logger, {"wav_out": args.wav_out})

    if args.print_marks and args.no_play:
        for m in marks:
            common.log_json(logger, {"t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme})

    if (not args.no_play) and (not args.stream):
        common.play_audio_with_marks(audio, sr, marks, chunk_size=int(args.chunk), logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

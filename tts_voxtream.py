"""VoXtream2 runner (aligned with MeloTTS runner for comparison)."""

from __future__ import annotations

import argparse
import logging
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple
import numpy as np
import tts_common as common


logger = logging.getLogger(__name__)


def _maybe_add_voxtream_to_syspath(voxtream_repo_path: Path) -> None:
    """Allow importing the local `voxtream` package without requiring installation."""

    import sys

    p = str(voxtream_repo_path)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_voxtream_config(config_path: str) -> object:
    voxtream_repo_path = Path("/home/pragay/WWAI/voxtream")
    _maybe_add_voxtream_to_syspath(voxtream_repo_path)

    from voxtream.config import SpeechGeneratorConfig  # type: ignore

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SpeechGeneratorConfig(**data)


def _stream_voxtream_inprocess(
    *,
    text: str,
    prompt_audio: str,
    use_gpu: bool,
    spk_rate: Optional[float],
    config_path: str,
    play: bool,
    emit_marks: bool,
    prefill_ms: int,
) -> tuple[list[np.ndarray], int, float, float, float, float, float]:
    """Run Voxtream in-process and return frames + timings.

    Returns: (frames, sr, init_s, ttft_s, gen_s, play_write_s, gen_compute_s)
    """

    voxtream_repo_path = Path("/home/pragay/WWAI/voxtream")
    _maybe_add_voxtream_to_syspath(voxtream_repo_path)

    # Device control: Voxtream picks torch.cuda.is_available(); to force CPU, blank CUDA.
    # (We keep this consistent with the previous subprocess mode.)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from voxtream.generator import SpeechGenerator  # type: ignore
    from voxtream.utils.generator import (  # type: ignore
        interpolate_speaking_rate_params,
        set_seed,
    )

    # Config + generator init (excluded from TTFT)
    config = _load_voxtream_config(config_path)

    init_t0 = perf_counter()
    speech_generator = SpeechGenerator(config)
    init_s = perf_counter() - init_t0

    set_seed()

    duration_state = weight = cfg_gamma = None
    if spk_rate is not None:
        # Load speaking-rate config if present next to generator config.
        spk_cfg = str(Path(voxtream_repo_path) / "configs" / "speaking_rate.json")
        try:
            with open(spk_cfg, "r", encoding="utf-8") as f:
                speaking_rate_config = json.load(f)
            duration_state, weight, cfg_gamma = interpolate_speaking_rate_params(
                speaking_rate_config, float(spk_rate), logger=speech_generator.logger
            )
        except Exception as e:
            logger.warning("Failed to load speaking rate config (%s): %s", spk_cfg, e)
            duration_state = weight = cfg_gamma = None

    frames: list[np.ndarray] = []
    ttft_s: Optional[float] = None

    # Best-effort phonemes/visemes for streaming: online scheduler (no total duration needed)
    scheduler: Optional[common.OnlineMarkScheduler] = None
    if emit_marks:
        try:
            phonemes = common.text_to_phonemes_g2p_en(text)
            scheduler = common.OnlineMarkScheduler(phonemes)
        except Exception as e:
            logger.warning("Phoneme extraction unavailable; marks disabled (%s)", e)
            scheduler = None

    sr = int(getattr(config, "mimi_sr"))

    out_stream = None
    if play:
        try:
            import sounddevice  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sounddevice is required for streaming playback. Install it (pip install sounddevice)."
            ) from e

    # TTFT should measure model time-to-first-audio, not phoneme scheduling or audio device setup.
    # So we start the clock right before invoking the generator.
    gen_t0 = perf_counter()
    stream = speech_generator.generate_stream(
        prompt_audio_path=Path(prompt_audio),
        text=text,
        target_spk_rate_cnt=duration_state,
        spk_rate_weight=weight,
        cfg_gamma=cfg_gamma,
    )

    prefill_ms = int(max(0, prefill_ms))
    prefill_samples = int((prefill_ms / 1000.0) * float(sr))
    buffered: list[np.ndarray] = []
    buffered_samples = 0
    play_write_s = 0.0

    try:
        for audio_frame, _gen_time in stream:
            if ttft_s is None:
                ttft_s = perf_counter() - gen_t0
            af = np.asarray(audio_frame, dtype=np.float32)

            # Emit marks for this chunk duration (aligned to audio timeline)
            if scheduler is not None:
                chunk_dur_s = float(len(af)) / float(getattr(config, "mimi_sr")) if af.size else 0.0
                for m in scheduler.schedule_for_chunk(chunk_dur_s):
                    common.log_json(
                        logger,
                        {"event": "mark", "t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme},
                    )

            frames.append(af)

            if play and af.size:
                # Optional prefill to reduce underflows (audible gaps) when generation is bursty.
                if out_stream is None and prefill_samples > 0:
                    buffered.append(af)
                    buffered_samples += int(af.size)
                    if buffered_samples >= prefill_samples:
                        import sounddevice as sd  # type: ignore

                        out_stream = sd.OutputStream(samplerate=int(sr), channels=1, dtype="float32")
                        out_stream.start()
                        for b in buffered:
                            if b.size:
                                out_stream.write(b)
                        buffered.clear()
                else:
                    if out_stream is None:
                        import sounddevice as sd  # type: ignore

                        out_stream = sd.OutputStream(samplerate=int(sr), channels=1, dtype="float32")
                        out_stream.start()
                    t_write0 = perf_counter()
                    out_stream.write(af)
                    play_write_s += float(perf_counter() - t_write0)
    finally:
        if out_stream is not None:
            out_stream.stop()
            out_stream.close()

    gen_s = perf_counter() - gen_t0
    # Best-effort: estimate time spent blocked on audio device writes.
    # (Generator-side waiting time remains part of gen_s, which is what you care about for streaming.)
    gen_compute_s = float(max(0.0, float(gen_s) - float(play_write_s)))
    return frames, sr, float(init_s), float(ttft_s or gen_s), float(gen_s), float(play_write_s), float(gen_compute_s)


def _read_audio_file(path: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        audio, sr = sf.read(path, always_2d=False)
        audio = common.to_mono_float32(np.asarray(audio))
        return audio, int(sr)
    except ModuleNotFoundError:
        raise RuntimeError("Missing dependency: soundfile. Install it via pip.")


def synthesize_voxtream(
    text: str,
    prompt_audio: str,
    use_gpu: bool,
    spk_rate: Optional[float] = None,
) -> tuple[np.ndarray, int, float]:
    # Point directly to the folder you just cloned
    voxtream_repo_path = Path("/home/pragay/WWAI/voxtream")
    voxtream_bin = shutil.which("voxtream") or str(voxtream_repo_path / "venv/bin/voxtream")

    env = os.environ.copy()
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    with tempfile.TemporaryDirectory(prefix="tts_voxtream_") as td:
        out_path = os.path.join(td, "voxtream_out.wav")

        # We run the command INSIDE the VoXtream2 folder so it finds configs/ automatically
        cmd = [
            voxtream_bin,
            "--prompt-audio", os.path.abspath(prompt_audio),
            "--text", text,
            "--output", out_path,
        ]

        if spk_rate:
            cmd.extend(["--spk-rate", str(spk_rate)])

        t0 = time.time()
        # 'cwd' is the key: it runs the tool from the repo directory
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(voxtream_repo_path))
        gen_s = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"VoXtream failed: {proc.stderr}")

        audio, sr = _read_audio_file(out_path)
        return audio, sr, float(gen_s)


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--prompt-audio", type=str, default="prompt.wav", help="Voice prompt WAV")
    p.add_argument("--gpu", action="store_true", help="Force GPU")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--speed", type=float, default=1.0, help="Mapped to spk-rate")
    p.add_argument(
        "--stream",
        action="store_true",
        help="Use in-process Voxtream streaming generator (precise TTFT).",
    )
    p.add_argument(
        "--stream-marks",
        action="store_true",
        help="Emit phoneme/viseme marks while streaming (best-effort heuristic).",
    )
    p.add_argument(
        "--stream-prefill-ms",
        type=int,
        default=0,
        help="Buffer this many ms of audio before starting playback in --stream mode (reduces gaps; adds latency).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("/home/pragay/WWAI/voxtream") / "configs" / "generator.json"),
        help="Voxtream generator config JSON",
    )
    args = p.parse_args(argv)

    # Device selection logic identical to your MeloTTS script
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = common.gpu_is_available()

    common.log_json(logger, {"device": "cuda" if use_gpu else "cpu"})

    # Check for system dependencies
    if not os.path.exists(args.prompt_audio):
        logger.error("Prompt audio not found: %s", args.prompt_audio)
        return 1

    init_s = 0.0
    ttft_s = None
    stream_play_write_s: Optional[float] = None
    stream_gen_compute_s: Optional[float] = None

    if args.stream:
        frames, sr, init_s, ttft_s_val, gen_s, stream_play_write_s, stream_gen_compute_s = _stream_voxtream_inprocess(
            text=args.text,
            prompt_audio=args.prompt_audio,
            use_gpu=use_gpu,
            spk_rate=float(args.speed) if args.speed else None,
            config_path=str(args.config),
            play=(not args.no_play),
            emit_marks=bool(args.stream_marks),
            prefill_ms=int(args.stream_prefill_ms),
        )
        audio = common.to_mono_float32(np.concatenate(frames) if frames else np.zeros((0,), dtype=np.float32))
        ttft_s = float(ttft_s_val)
    else:
        audio, sr, gen_s = synthesize_voxtream(
            args.text,
            args.prompt_audio,
            use_gpu=use_gpu,
            spk_rate=args.speed  # Using speed arg to control rate
        )

    metrics = common.compute_metrics(gen_s, audio, sr)

    # Print block matches MeloTTS format exactly
    payload = {
        "model": "voxtream",
        "streaming": bool(args.stream),
        "samplerate": sr,
        "init_s": round(float(init_s), 3),
        "gen_s": round(metrics.gen_s, 3),
        "ttfc_s": round(float(ttft_s if ttft_s is not None else metrics.gen_s), 3),
        "ttft_s": round(float(ttft_s if ttft_s is not None else metrics.gen_s), 3),
        "audio_s": round(metrics.audio_s, 3),
        "rtf": round(metrics.rtf, 3),
    }
    if args.stream:
        # Expose compute vs playback-blocking time so you can compare to offline models.
        if stream_play_write_s is not None:
            payload["play_write_s"] = round(float(stream_play_write_s), 3)
        if stream_gen_compute_s is not None:
            payload["gen_compute_s"] = round(float(stream_gen_compute_s), 3)
    common.log_json(logger, payload)

    # Only compute fallback marks when we're going to use them.
    # In --stream mode, playback already happened inside _stream_voxtream_inprocess.
    marks = []
    if bool(args.print_marks) or ((not args.no_play) and (not args.stream)):
        phonemes = common.text_to_phonemes_g2p_en(args.text)
        marks = common.build_fallback_marks(phonemes, metrics.audio_s)

    if args.wav_out:
        common.save_wav_pcm16(args.wav_out, audio, sr)

    if args.print_marks:
        for m in marks:
            common.log_json(logger, {"t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme})

    # Avoid double-speaking: --stream mode already plays live (unless --no-play).
    if (not args.no_play) and (not args.stream):
        common.play_audio_with_marks(audio, sr, marks, logger=logger)

    return 0


if __name__ == "__main__":
    main()

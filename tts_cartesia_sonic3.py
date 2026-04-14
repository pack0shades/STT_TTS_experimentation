"""Cartesia Sonic (cloud) TTS runner.

Cartesia is cloud-only; their API can return timestamp alignment and supports IPA.
Because API schemas evolve, this file is a *template*:
- It saves/playbacks returned audio.
- It can consume timestamp marks if you paste/adjust the parsing logic.

Requires:
- CARTESIA_API_KEY in environment (or .env)

Example:
  python tts_cartesia_sonic3.py --text "Hello" --voice "sonic-3" --wav-out out.wav --no-play
"""

from __future__ import annotations

import argparse
import logging
import os
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)


def _load_env_if_present() -> None:
    if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass


def _parse_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    # same logic as deepgram script
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    pcm = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    audio = (pcm.astype(np.float32) / 32767.0).clip(-1.0, 1.0)
    return audio, int(sr)


def cartesia_tts_request(
    text: str,
    *,
    api_key: str,
    endpoint: str,
    voice: str,
    http_stream: bool = True,
    http_chunk_size: int = 16384,
    timeout_s: int = 60,
) -> Tuple[bytes, common.StreamMetrics, List[common.Mark]]:
    """Call Cartesia TTS.

    This is intentionally conservative: you will likely need to adjust `endpoint`,
    headers, and payload to match your Cartesia account/docs.

    Returns (wav_bytes, marks).
    """

    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("requests is required for Cartesia API calls (pip install requests).") from e

    # TODO: update endpoint/payload to match Cartesia docs.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "voice": voice,
        # "timestamps": True,
        # "format": "wav",
    }

    t0 = perf_counter()
    resp = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout_s,
        stream=bool(http_stream),
    )
    resp.raise_for_status()

    # Many APIs return JSON with base64 audio + timestamps; others return raw audio.
    # Handle both.
    ct = resp.headers.get("content-type", "").lower()

    marks: List[common.Mark] = []

    if "application/json" in ct:
        data = resp.json()

        # TODO: adjust to actual Cartesia response shape.
        # Example pseudo-structure:
        #  audio_b64 = data["audio"]
        #  timestamps = data.get("timestamps", [])
        #  for t in timestamps: marks.append(common.Mark(t=t["time"], phoneme=t["token"], viseme=...))
        raise RuntimeError(
            "Cartesia returned JSON; update parsing to decode audio + timestamps. "
            "Dump: " + str(data)[:400]
        )

    if http_stream:
        wav_bytes, dl = common.consume_iterable_to_bytes(
            resp.iter_content(chunk_size=int(http_chunk_size)),
            start_t=t0,
        )
    else:
        wav_bytes = resp.content
        total_s = float(perf_counter() - t0)
        dl = common.StreamMetrics(ttfc_s=None, total_s=total_s, n_chunks=1, n_bytes=len(wav_bytes))

    return wav_bytes, dl, marks


def main(argv: Optional[list[str]] = None) -> int:
    _load_env_if_present()
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--voice", type=str, default="sonic-3")
    p.add_argument("--endpoint", type=str, default="https://api.cartesia.ai/tts")
    p.add_argument("--no-http-stream", action="store_true", help="Disable HTTP response streaming")
    p.add_argument("--http-chunk", type=int, default=16384, help="HTTP streaming chunk size (bytes)")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0)
    args = p.parse_args(argv)

    api_key = os.getenv("CARTESIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Missing CARTESIA_API_KEY. Create a .env with CARTESIA_API_KEY=... or export it in your shell."
        )

    wav_bytes, dl, marks = cartesia_tts_request(
        args.text,
        api_key=api_key,
        endpoint=args.endpoint,
        voice=args.voice,
        http_stream=not bool(args.no_http_stream),
        http_chunk_size=int(args.http_chunk),
    )

    gen_s = float(dl.total_s)

    audio, sr = _parse_wav_bytes(wav_bytes)
    metrics = common.compute_metrics(gen_s, audio, sr)

    common.log_json(
        logger,
        {
            "model": f"cartesia:{args.voice}",
            "streaming": (not bool(args.no_http_stream)),
            "samplerate": sr,
            "gen_s": round(metrics.gen_s, 3),
            "ttfc_s": None if dl.ttfc_s is None else round(float(dl.ttfc_s), 3),
            "ttft_s": None if dl.ttfc_s is None else round(float(dl.ttfc_s), 3),
            "http_stream": (not bool(args.no_http_stream)),
            "http_bytes": int(dl.n_bytes),
            "http_chunks": int(dl.n_chunks),
            "audio_s": round(metrics.audio_s, 3),
            "rtf": round(metrics.rtf, 3),
        },
    )

    if not marks:
        phonemes = common.text_to_phonemes_g2p_en(args.text)
        marks = common.build_fallback_marks(phonemes, metrics.audio_s, jitter_s=float(args.jitter), seed=0)

    if args.wav_out:
        common.save_wav_pcm16(args.wav_out, audio, sr)
        common.log_json(logger, {"wav_out": args.wav_out})

    if args.print_marks and args.no_play:
        for m in marks:
            common.log_json(logger, {"t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme})

    if not args.no_play:
        common.play_audio_with_marks(audio, sr, marks, chunk_size=int(args.chunk), logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

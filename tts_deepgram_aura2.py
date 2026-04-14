"""Deepgram Aura (cloud) TTS runner.

Notes:
- Deepgram models and query params change over time; this script is designed to be
  easy to tweak.
- If Deepgram returns "speech marks" (timestamps for visemes/phonemes), we print
  them. Otherwise we fall back to the shared heuristic schedule.

Requires:
- DEEPGRAM_API_KEY in environment (or .env)

Example:
  python tts_deepgram_aura2.py --text "Hello" --model "aura-2" --wav-out out.wav --no-play
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)


def _load_env_if_present() -> None:
    # Optional: load .env if python-dotenv is installed.
    if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass


def _parse_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Parse WAV bytes into mono float32 numpy using stdlib `wave`."""
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


def deepgram_speak_request(
    text: str,
    *,
    api_key: str,
    model: str,
    endpoint: str,
    extra_query: str = "",
    http_stream: bool = True,
    http_chunk_size: int = 16384,
    timeout_s: int = 60,
) -> Tuple[bytes, common.StreamMetrics, Optional[dict]]:
    """Send a TTS request.

    Returns (audio_bytes, metadata_json_or_none).

    By default we request WAV so we can parse with stdlib.
    """

    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("requests is required for Deepgram API calls (pip install requests).") from e

    # Common Deepgram Speak endpoint pattern (may need adjustment):
    #   https://api.deepgram.com/v1/speak?model=...&encoding=linear16&container=wav
    # If your account/model supports speech marks, Deepgram may return JSON metadata
    # via a different endpoint/param; keep this script flexible.
    url = f"{endpoint}?model={model}&encoding=linear16&container=wav"
    if extra_query:
        url = url + "&" + extra_query.lstrip("&")

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    payload = {"text": text}

    t0 = perf_counter()
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout_s,
        stream=bool(http_stream),
    )
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        meta = resp.json()
        # Some variants return audio as base64 within JSON; not handled by default.
        raise RuntimeError(
            "Deepgram returned JSON instead of WAV. Adjust this script to decode the response: "
            + json.dumps(meta)[:400]
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

    return wav_bytes, dl, None


def main(argv: Optional[list[str]] = None) -> int:
    _load_env_if_present()
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--model", type=str, default="aura-2")
    p.add_argument("--endpoint", type=str, default="https://api.deepgram.com/v1/speak")
    p.add_argument("--extra-query", type=str, default="")
    p.add_argument("--no-http-stream", action="store_true", help="Disable HTTP response streaming")
    p.add_argument("--http-chunk", type=int, default=16384, help="HTTP streaming chunk size (bytes)")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0)
    args = p.parse_args(argv)

    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Missing DEEPGRAM_API_KEY. Create a .env with DEEPGRAM_API_KEY=... or export it in your shell."
        )

    wav_bytes, dl, meta = deepgram_speak_request(
        args.text,
        api_key=api_key,
        model=args.model,
        endpoint=args.endpoint,
        extra_query=args.extra_query,
        http_stream=not bool(args.no_http_stream),
        http_chunk_size=int(args.http_chunk),
    )

    # For streaming HTTP downloads, dl.total_s is end-to-end (request start -> last byte)
    gen_s = float(dl.total_s)

    audio, sr = _parse_wav_bytes(wav_bytes)
    metrics = common.compute_metrics(gen_s, audio, sr)

    common.log_json(
        logger,
        {
            "model": f"deepgram:{args.model}",
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

    # Marks: if metadata contains speech marks, prefer them. Otherwise fallback.
    marks: List[common.Mark]

    marks = []
    if isinstance(meta, dict):
        # TODO: adapt if/when you have real Deepgram speech mark payloads.
        pass

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

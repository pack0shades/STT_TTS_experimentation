"""Shared utilities for TTS model comparisons.

Goals:
- Keep phoneme/viseme extraction consistent across models.
- Provide a common schedule generator when a model/API does not return timestamps.
- Provide simple playback + wav saving so each model script stays small.

This module intentionally avoids heavyweight deps (e.g., scipy). It uses only stdlib + numpy.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import wave
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional, Sequence

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a reasonable default logging setup.

    No-op if logging is already configured by the host application.
    """

    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_json(logger: logging.Logger, obj: object, *, level: int = logging.INFO) -> None:
    """Log a JSON-serialized object (best-effort)."""

    try:
        msg = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        msg = json.dumps(str(obj), ensure_ascii=False)
    logger.log(level, msg)


def offset_marks(marks: Sequence["Mark"], offset_s: float) -> List["Mark"]:
    """Return a new marks list with time shifted by offset_s."""

    out: List[Mark] = []
    for m in marks:
        out.append(Mark(t=float(m.t) + float(offset_s), phoneme=m.phoneme, viseme=m.viseme))
    return out


# ------------------------------
# Phonemes -> visemes
# ------------------------------

# ARPAbet-ish phonemes -> simple viseme buckets (customizable)
VISEME_MAP = {
    # lips closed
    "P": "PP",
    "B": "PP",
    "M": "PP",

    # teeth on lip
    "F": "FF",
    "V": "FF",

    # tongue between teeth
    "TH": "TH",
    "DH": "TH",

    # alveolar
    "T": "DD",
    "D": "DD",
    "S": "DD",
    "Z": "DD",

    # velar
    "K": "kk",
    "G": "kk",
    "NG": "kk",

    # affricates
    "CH": "CH",
    "JH": "CH",

    # fricatives
    "SH": "SS",
    "ZH": "SS",

    # nasal
    "N": "nn",

    # approximants
    "R": "RR",

    # vowels (VERY IMPORTANT)
    "AA": "aa",
    "AE": "aa",
    "AH": "aa",
    "AO": "oh",
    "OW": "oh",
    "UH": "ou",
    "UW": "ou",
    "EH": "E",
    "EY": "E",
    "IH": "ih",
    "IY": "ih",
}

VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}

MEDIUM = {"L", "R", "W", "Y", "N", "NG"}


def normalize_phoneme(p: str) -> str:
    """Strip stress digits and normalize."""
    return re.sub(r"\d", "", p).upper()


def safe_viseme(phoneme: str) -> str:
    phoneme = normalize_phoneme(phoneme)
    if phoneme == "PAUSE":
        return "sil"
    return VISEME_MAP.get(phoneme, "sil")


# ------------------------------
# G2P phoneme extraction
# ------------------------------


def text_to_phonemes_g2p_en(text: str) -> List[str]:
    """Return a phoneme token list using `g2p_en`.

    Keeps behavior similar to your current `VITS.py`:
    - punctuation -> "PAUSE"
    - spaces removed
    """

    try:
        from g2p_en import G2p  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "g2p_en is required for phoneme extraction. Install it (pip install g2p_en)."
        ) from e

    g2p = G2p()
    raw = g2p(text)

    cleaned: List[str] = []
    for p in raw:
        if p == " ":
            continue
        if not str(p).isalpha():
            cleaned.append("PAUSE")
        else:
            cleaned.append(str(p))
    return cleaned


# ------------------------------
# Scheduling (fallback when no timestamps)
# ------------------------------


@dataclass(frozen=True)
class Mark:
    t: float
    phoneme: str
    viseme: str


def _base_weight(phoneme: str) -> float:
    p = normalize_phoneme(phoneme)
    if p == "PAUSE":
        return 2.8
    if p in VOWELS:
        return 2.0
    if p in MEDIUM:
        return 1.4
    return 1.0


def build_fallback_marks(
    phonemes: Sequence[str],
    total_duration_s: float,
    *,
    jitter_s: float = 0.0,
    seed: Optional[int] = 0,
    min_dur_s: float = 0.03,
    max_dur_s: float = 0.25,
) -> List[Mark]:
    """Create a time schedule for phonemes scaled to exactly match audio duration.

    This is *not* forced alignment; it is a consistent heuristic so you can compare
    models apples-to-apples when no native timing is available.
    """

    if total_duration_s <= 0:
        return []

    if len(phonemes) == 0:
        return []

    rng = random.Random(seed)

    weights = np.array([_base_weight(p) for p in phonemes], dtype=np.float64)
    weights_sum = float(weights.sum())
    if weights_sum <= 0:
        weights = np.ones(len(phonemes), dtype=np.float64)
        weights_sum = float(weights.sum())

    durs = (weights / weights_sum) * float(total_duration_s)

    # clamp then renormalize (two-pass so we still sum to total_duration_s)
    durs = np.clip(durs, min_dur_s, max_dur_s)
    durs = durs * (float(total_duration_s) / float(durs.sum()))

    if jitter_s and jitter_s > 0:
        jitter = np.array([rng.uniform(-jitter_s, jitter_s) for _ in phonemes], dtype=np.float64)
        durs = np.clip(durs + jitter, min_dur_s, max_dur_s)
        durs = durs * (float(total_duration_s) / float(durs.sum()))

    marks: List[Mark] = []
    t = 0.0
    for p, dur in zip(phonemes, durs.tolist()):
        marks.append(Mark(t=t, phoneme=normalize_phoneme(p), viseme=safe_viseme(p)))
        t += float(dur)

    return marks


class OnlineMarkScheduler:
    """Online phoneme scheduler for streaming audio.

    Why this exists:
    - Many TTS libs don't provide forced-alignment timestamps during generation.
    - In streaming voice bots, you still want to emit viseme events with minimal lag.

    This scheduler consumes a phoneme list and assigns heuristic durations as audio
    arrives in chunks. It does *not* guarantee linguistic accuracy; it guarantees:
    - monotonic timestamps
    - no need to know total duration upfront
    - predictable, low-lag mark emission
    """

    def __init__(
        self,
        phonemes: Sequence[str],
        *,
        base_unit_s: float = 0.06,
        min_dur_s: float = 0.03,
        max_dur_s: float = 0.25,
    ) -> None:
        self._phonemes = [normalize_phoneme(p) for p in phonemes]
        self._i = 0
        self._t = 0.0
        self._base = float(base_unit_s)
        self._min = float(min_dur_s)
        self._max = float(max_dur_s)

    @property
    def done(self) -> bool:
        return self._i >= len(self._phonemes)

    @property
    def t(self) -> float:
        """Current audio timeline position (seconds)."""

        return float(self._t)

    def _next_dur(self, phoneme: str) -> float:
        # convert weight to duration; clamp for stability
        w = float(_base_weight(phoneme))
        d = w * self._base
        return float(min(self._max, max(self._min, d)))

    def schedule_for_chunk(self, chunk_dur_s: float) -> List[Mark]:
        """Return marks whose *start* falls within this chunk.

        Advances internal time by chunk_dur_s.
        """

        chunk_dur_s = float(max(0.0, chunk_dur_s))
        if chunk_dur_s <= 0:
            return []

        chunk_start = float(self._t)
        chunk_end = float(self._t + chunk_dur_s)

        marks: List[Mark] = []

        # Emit as many phonemes as can fit starting inside [chunk_start, chunk_end)
        while self._i < len(self._phonemes) and self._t < chunk_end:
            p = self._phonemes[self._i]
            # If this mark starts after chunk end, stop
            if self._t >= chunk_end:
                break

            # emit
            marks.append(Mark(t=float(self._t), phoneme=p, viseme=safe_viseme(p)))

            # advance by heuristic duration
            self._t += self._next_dur(p)
            self._i += 1

        # Ensure we advance at least the chunk duration in the audio timeline.
        # If our heuristic phoneme durations lag behind audio arrival, push time forward.
        self._t = float(max(self._t, chunk_end))

        # If we didn't emit any marks but time advanced, keep silence until next chunk.
        _ = chunk_start  # reserved for future debugging
        return marks


# ------------------------------
# Audio I/O
# ------------------------------


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        # (channels, samples) or (samples, channels)
        if audio.shape[0] in (1, 2) and audio.shape[0] < audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    return np.clip(audio, -1.0, 1.0)


def save_wav_pcm16(path: str, audio: np.ndarray, samplerate: int) -> None:
    """Save mono audio in 16-bit PCM WAV."""

    audio = to_mono_float32(audio)
    pcm = (audio * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(pcm.tobytes())


def play_audio_with_marks(
    audio: np.ndarray,
    samplerate: int,
    marks: Sequence[Mark],
    *,
    chunk_size: int = 1024,
    on_mark: Optional[Callable[[Mark], None]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Stream audio to the default sounddevice output and emit marks in real-time."""

    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sounddevice is required for playback. Install it (pip install sounddevice)."
        ) from e

    audio = to_mono_float32(audio)

    if logger is None:
        logger = logging.getLogger(__name__)

    mark_index = 0
    start_time = time.time()

    def emit_due_marks(now_s: float) -> None:
        nonlocal mark_index
        while mark_index < len(marks) and now_s >= float(marks[mark_index].t):
            m = marks[mark_index]
            if on_mark is not None:
                on_mark(m)
            else:
                log_json(
                    logger,
                    {
                        "time": round(now_s, 3),
                        "phoneme": m.phoneme,
                        "viseme": m.viseme,
                    },
                )
            mark_index += 1

    with sd.OutputStream(samplerate=int(samplerate), channels=1, dtype="float32") as stream:
        for i in range(0, len(audio), int(chunk_size)):
            chunk = audio[i:i + int(chunk_size)]
            stream.write(chunk)
            now = time.time() - start_time
            emit_due_marks(now)


# ------------------------------
# Metrics
# ------------------------------


@dataclass(frozen=True)
class Metrics:
    gen_s: float
    audio_s: float

    @property
    def rtf(self) -> float:
        """Real-time factor: generation time / audio duration."""
        if self.audio_s <= 0:
            return math.inf
        return self.gen_s / self.audio_s


def compute_metrics(gen_s: float, audio: np.ndarray, samplerate: int) -> Metrics:
    audio = np.asarray(audio)
    audio_s = float(audio.shape[-1]) / float(samplerate) if audio.size else 0.0
    return Metrics(gen_s=float(gen_s), audio_s=float(audio_s))


def gpu_is_available() -> bool:
    """Best-effort CUDA availability check.

    - Returns False if torch is not installed.
    - Returns torch.cuda.is_available() otherwise.

    This avoids forcing torch as a hard dependency for cloud-only users.
    """

    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ------------------------------
# Streaming helpers (TTFC/TTFT)
# ------------------------------


@dataclass(frozen=True)
class StreamMetrics:
    """Metrics for streamed payload consumption.

    ttfc_s is a good proxy for "time to first token" in audio streaming.
    """

    ttfc_s: Optional[float]
    total_s: float
    n_chunks: int
    n_bytes: int


def consume_iterable_to_bytes(chunks, *, start_t: Optional[float] = None) -> tuple[bytes, StreamMetrics]:
    """Consume an iterable of (byte) chunks, measuring TTFC + total time."""

    if start_t is None:
        start_t = perf_counter()

    first_t: Optional[float] = None
    buf = bytearray()
    n_chunks = 0

    for c in chunks:
        if not c:
            continue
        if first_t is None:
            first_t = perf_counter()
        n_chunks += 1
        buf.extend(c)

    end_t = perf_counter()
    ttfc_s = (first_t - start_t) if first_t is not None else None
    return bytes(buf), StreamMetrics(
        ttfc_s=ttfc_s,
        total_s=float(end_t - start_t),
        n_chunks=int(n_chunks),
        n_bytes=int(len(buf)),
    )

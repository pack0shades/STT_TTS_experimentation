"""OmniTalker runner (placeholder).

OmniTalker is typically an end-to-end audio+video/viseme pipeline.
Once you add the repo/weights, we can adapt it to output either:
- audio only, plus viseme state over time, or
- full video viseme state.

This script reserves a stable CLI entry point.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)


def synthesize_omnitalker(text: str) -> tuple[np.ndarray, int, float]:
    raise NotImplementedError


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--gpu", action="store_true", help="Force GPU (placeholder)")
    p.add_argument("--cpu", action="store_true", help="Force CPU (placeholder)")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0)
    _ = p.parse_args(argv)

    common.log_json(
        logger,
        {
            "model": "omnitalker",
            "status": "not_implemented",
            "streaming": False,
            "reason": (
                "Add the OmniTalker repo + weights and wire its inference entrypoint "
                "(often audio+video/visemes)."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

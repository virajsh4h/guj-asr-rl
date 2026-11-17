"""Text normalisation and utterance metrics.

The normaliser itself lives in `gu_normalizer.py` and is reused verbatim. Rewriting
it would silently invalidate every number already reported, so this module only
wraps it and pins the version that produced a row.
"""

from __future__ import annotations

from typing import Any

import gu_normalizer

NORMALIZER_VERSION = "gu-normalizer/1"


def normalize(text: str) -> str:
    return gu_normalizer.normalize_gujarati(text)


def utterance_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    return gu_normalizer.compute_utterance_metrics(reference, hypothesis)

from __future__ import annotations

from typing import Any

import gu_normalizer

NORMALIZER_VERSION = "gu-normalizer/1"


def normalize(text: str) -> str:
    return gu_normalizer.normalize_gujarati(text)


def utterance_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    return gu_normalizer.compute_utterance_metrics(reference, hypothesis)

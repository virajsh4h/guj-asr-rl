from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jiwer
import numpy as np

DEFAULT_REPS = 2000
DEFAULT_ALPHA = 0.05


@dataclass
class Counts:
    edits: np.ndarray
    lengths: np.ndarray
    units: str

    @property
    def n(self) -> int:
        return int(self.edits.size)

    @property
    def rate(self) -> float:
        total = float(self.lengths.sum())
        if total <= 0:
            return 0.0
        return 100.0 * float(self.edits.sum()) / total


def _char_edits(ref: str, hyp: str) -> tuple[int, int]:
    out = jiwer.process_characters(ref, hyp)
    edits = int(out.substitutions) + int(out.deletions) + int(out.insertions)
    return edits, len(ref)


def _word_edits(ref: str, hyp: str) -> tuple[int, int]:
    out = jiwer.process_words(ref, hyp)
    edits = int(out.substitutions) + int(out.deletions) + int(out.insertions)
    return edits, len(ref.split())


def counts(references: Sequence[str], hypotheses: Sequence[str], unit: str = "char") -> Counts:
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references against {len(hypotheses)} hypotheses")
    fn = _char_edits if unit == "char" else _word_edits
    edits: list[int] = []
    lengths: list[int] = []
    for ref, hyp in zip(references, hypotheses):
        e, n = fn(ref or "", hyp or "")
        if n == 0:
            continue
        edits.append(e)
        lengths.append(n)
    return Counts(np.asarray(edits, dtype=np.int64), np.asarray(lengths, dtype=np.int64), unit)


def corpus_rate(refs: Sequence[str], hyps: Sequence[str], unit: str = "char") -> float:
    return counts(refs, hyps, unit).rate


def bootstrap_ci(
    data: Counts,
    reps: int = DEFAULT_REPS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    if data.n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.n, size=(reps, data.n))
    edits = data.edits[idx].sum(axis=1)
    lengths = data.lengths[idx].sum(axis=1)
    rates = 100.0 * edits / np.where(lengths == 0, 1, lengths)
    lo, hi = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class PairedResult:
    delta: float
    lo: float
    hi: float
    p_value: float
    reps: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def paired_bootstrap(
    left: Counts,
    right: Counts,
    reps: int = DEFAULT_REPS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> PairedResult:
    if left.n != right.n:
        raise ValueError(f"paired test needs equal n: {left.n} vs {right.n}")
    if left.n == 0:
        return PairedResult(0.0, 0.0, 0.0, 1.0, reps)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, left.n, size=(reps, left.n))
    base = left.lengths[idx].sum(axis=1)
    base = np.where(base == 0, 1, base)
    left_rates = 100.0 * left.edits[idx].sum(axis=1) / base
    right_rates = 100.0 * right.edits[idx].sum(axis=1) / base
    deltas = left_rates - right_rates
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    below = float(np.mean(deltas <= 0))
    above = float(np.mean(deltas >= 0))
    p = min(1.0, 2.0 * min(below, above))
    return PairedResult(
        delta=float(left.rate - right.rate),
        lo=float(lo),
        hi=float(hi),
        p_value=p,
        reps=reps,
    )


def macro(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap_macro(
    per_dataset: Sequence[Counts],
    reps: int = DEFAULT_REPS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    if not per_dataset:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    rates = []
    for data in per_dataset:
        if data.n == 0:
            continue
        idx = rng.integers(0, data.n, size=(reps, data.n))
        edits = data.edits[idx].sum(axis=1)
        lengths = np.maximum(data.lengths[idx].sum(axis=1), 1)
        rates.append(100.0 * edits / lengths)
    if not rates:
        return 0.0, 0.0
    stacked = np.mean(np.vstack(rates), axis=0)
    lo, hi = np.quantile(stacked, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

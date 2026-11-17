from __future__ import annotations

import importlib
from typing import Sequence

import numpy as np


class Adapter:
    name = "adapter"
    batch_mode = "serial"

    def transcribe_batch(
        self,
        audio_arrays: Sequence[np.ndarray],
        row_indices: Sequence[int],
        batch_size: int,
    ) -> list:
        raise NotImplementedError


def module():
    return importlib.import_module("run_baselines_optimized")


def build(model_key: str) -> Adapter:
    return module().build_adapter(model_key)


class ScriptedAdapter(Adapter):

    def __init__(self, name: str = "scripted", fail_on: set[int] | None = None, echo: str = "ok"):
        self.name = name
        self.batch_mode = "scripted"
        self.fail_on = fail_on or set()
        self.echo = echo
        self.calls = 0

    def transcribe_batch(self, audio_arrays, row_indices, batch_size):
        self.calls += 1
        out: list = []
        for idx, audio in zip(row_indices, audio_arrays):
            if idx in self.fail_on:
                out.append(RuntimeError(f"scripted failure at {idx}"))
            else:
                out.append(f"{self.echo} {float(np.asarray(audio).sum()):.2f}")
        return out

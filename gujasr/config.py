from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from .normalize import NORMALIZER_VERSION

RESULT_FIELDS = (
    "batch_size", "max_samples", "bucket_multiplier", "torch_threads", "seed",
)


@dataclass
class Config:
    models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    db_path: str = "results/harness.sqlite"
    results_dir: str = "results/harness"
    batch_size: int = 32
    decode_workers: int = 8
    prefetch: int = 32
    bucket_multiplier: int = 4
    max_samples: int = 0
    torch_threads: int = 4
    seed: int = 0
    bootstrap_reps: int = 2000

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def scoring(self) -> dict[str, Any]:
        payload = {k: getattr(self, k) for k in RESULT_FIELDS}
        payload["normalizer_version"] = NORMALIZER_VERSION
        return payload

    def config_hash(self) -> str:
        blob = json.dumps(self.scoring(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def run_key(self, model_key: str, dataset: str) -> str:
        blob = json.dumps(
            {
                "model": model_key,
                "dataset": dataset,
                "config": self.config_hash(),
                "code": code_sha(),
                "normalizer": NORMALIZER_VERSION,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:20]


def code_sha() -> str:
    override = os.environ.get("GUJASR_CODE_SHA")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def collect_provenance() -> dict[str, Any]:
    gpu = None
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(props.total_memory / 1024 ** 3, 1),
            "bf16": bool(torch.cuda.is_bf16_supported()),
        }
    return {
        "code_sha": code_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "datasets", "jiwer", "numpy", "accelerate")
        },
        "gpu": gpu,
        "normalizer_version": NORMALIZER_VERSION,
    }

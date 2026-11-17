"""Evaluation loop.

Same decode path as the original bench, with three differences that change what the
numbers mean rather than what the model does: rows are bound to a run identity, every
inference is timed at batch level and amortised per utterance, and failures are recorded
instead of dropped.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import manifests, store
from .adapters import Adapter
from .config import Config, collect_provenance
from .normalize import utterance_metrics


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _default_loader(dataset_key: str):
    from run_baselines_optimized import load_gujarati_dataset

    return load_gujarati_dataset(dataset_key)


def _default_iterator():
    from run_baselines_optimized import iter_preprocessed_items

    return iter_preprocessed_items


def _chunks(items: Iterable[dict], size: int, multiplier: int) -> Iterable[list[dict]]:
    bucket: list[dict] = []
    window = max(size * multiplier, size)
    for item in items:
        bucket.append(item)
        if len(bucket) >= window:
            bucket.sort(key=lambda x: x["duration_s"])
            for offset in range(0, len(bucket), size):
                yield bucket[offset:offset + size]
            bucket = []
    if bucket:
        bucket.sort(key=lambda x: x["duration_s"])
        for offset in range(0, len(bucket), size):
            yield bucket[offset:offset + size]


def _row(run_key: str, model_key: str, model_name: str, item: dict, output: Any,
         latency: float, rtf: float) -> dict:
    failed = isinstance(output, Exception)
    hypothesis = "" if failed else str(output or "").strip()
    metrics = utterance_metrics(item["reference"], hypothesis)
    return {
        "run_key": run_key,
        "dataset": item["dataset"],
        "split": item["split"],
        "sample_id": item["sample_id"],
        "row_index": item["row_index"],
        "model_key": model_key,
        "model_name": model_name,
        "audio_duration_s": item["duration_s"],
        "latency_s": round(latency, 4),
        "rtf": round(rtf, 5),
        "reference_raw": item["reference"],
        "hypothesis_raw": hypothesis,
        "reference_norm": metrics["ref_norm"],
        "hypothesis_norm": metrics["hyp_norm"],
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "capped_cer": metrics["capped_cer"],
        "script_drift": metrics["script_drift"],
        "drift_ratio": metrics["drift_ratio"],
        "loop_flag": metrics["loop_flag"],
        "failure_tag": repr(output) if failed else "",
    }


def evaluate(
    conn,
    cfg: Config,
    adapter: Adapter,
    dataset_key: str,
    model_key: str,
    loader: Callable[[str], tuple] | None = None,
    iterator: Callable[..., Iterable[dict]] | None = None,
) -> dict[str, Any]:
    spec = manifests.DATASETS[dataset_key]
    model_name = manifests.model_name(model_key)
    run_key = cfg.run_key(model_key, dataset_key)
    loader = loader or _default_loader
    iterator = iterator or _default_iterator()

    ds, split, audio_column, text_column = loader(dataset_key)
    done = store.completed_ids(conn, run_key)
    available = len(ds) if hasattr(ds, "__len__") else spec["expected_rows"]
    expected = min(spec["expected_rows"], available)
    if cfg.max_samples > 0:
        expected = min(expected, cfg.max_samples)

    store.start_run(
        conn,
        {
            "run_key": run_key,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_key": model_key,
            "model_name": model_name,
            "dataset": dataset_key,
            "split": split,
            "config_hash": cfg.config_hash(),
            "code_sha": collect_provenance()["code_sha"],
            "normalizer_version": cfg.scoring()["normalizer_version"],
            "manifest_json": json.dumps(
                {
                    "dataset": manifests.dataset_manifest(dataset_key),
                    "model": manifests.MODELS[model_key],
                    "config": cfg.scoring(),
                },
                ensure_ascii=False,
            ),
            "status": "running",
        },
    )

    if len(done) >= expected:
        store.set_status(conn, run_key, "complete")
        _log(f"[{model_name}/{dataset_key}] complete already: {len(done)}/{expected}")
        return {
            "run_key": run_key, "model_key": model_key, "model_name": model_name,
            "dataset": dataset_key, "split": split, "rows": len(done),
            "expected": expected, "failures": 0, "coverage": round(len(done) / expected, 4),
            "audio_seconds": 0.0, "inference_seconds": 0.0, "wall_seconds": 0.0,
            "rtf": 0.0, "samples_per_second": 0.0, "peak_vram_gb": 0.0,
        }

    items = iterator(
        ds, dataset_key, split, audio_column, text_column,
        cfg.max_samples, done, cfg.decode_workers, cfg.prefetch,
    )

    written = failures = 0
    audio_seconds = 0.0
    infer_seconds = 0.0
    started = time.perf_counter()
    for chunk in _chunks(items, cfg.batch_size, cfg.bucket_multiplier):
        arrays = [item["audio_array"] for item in chunk]
        indices = [item["row_index"] for item in chunk]
        _sync()
        t0 = time.perf_counter()
        outputs = adapter.transcribe_batch(arrays, indices, cfg.batch_size)
        _sync()
        elapsed = time.perf_counter() - t0
        if len(outputs) != len(chunk):
            raise RuntimeError(f"adapter returned {len(outputs)} outputs for {len(chunk)} samples")
        infer_seconds += elapsed
        audio_seconds += sum(item["duration_s"] for item in chunk)
        per_item = elapsed / max(len(chunk), 1)
        rows = []
        for item, output in zip(chunk, outputs):
            rtf = elapsed / max(sum(i["duration_s"] for i in chunk), 1e-6)
            rows.append(_row(run_key, model_key, model_name, item, output, per_item, rtf))
        written += store.write_rows(conn, rows)
        failures += sum(1 for row in rows if row["failure_tag"])
        _log(f"  [{model_name}/{dataset_key}] {written}/{expected} utterances")

    wall = time.perf_counter() - started
    store.set_status(conn, run_key, "complete" if len(done) + written >= expected else "partial")
    summary = {
        "run_key": run_key,
        "model_key": model_key,
        "model_name": model_name,
        "dataset": dataset_key,
        "split": split,
        "rows": written,
        "expected": expected,
        "failures": failures,
        "coverage": round(written / expected, 4) if expected else 0.0,
        "audio_seconds": round(audio_seconds, 2),
        "inference_seconds": round(infer_seconds, 2),
        "wall_seconds": round(wall, 2),
        "rtf": round(infer_seconds / max(audio_seconds, 1e-6), 5),
        "samples_per_second": round(written / max(infer_seconds, 1e-9), 3),
        "peak_vram_gb": _peak_vram(),
    }
    _log(
        f"[{model_name}/{dataset_key}] {written}/{expected} rows, {failures} failures, "
        f"RTF {summary['rtf']:.4f}, {summary['samples_per_second']:.2f} utt/s"
    )
    return summary


def _peak_vram() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)


def run(cfg: Config, build_adapter: Callable[[str], Adapter] | None = None,
        loader=None, iterator=None) -> list[dict[str, Any]]:
    from .adapters import build as default_build

    build_adapter = build_adapter or default_build
    conn = store.connect(cfg.db_path)
    summaries: list[dict[str, Any]] = []
    try:
        for model_key in cfg.models:
            adapter = build_adapter(model_key)
            _reset_peak()
            for dataset_key in cfg.datasets:
                summaries.append(
                    evaluate(conn, cfg, adapter, dataset_key, model_key, loader, iterator)
                )
            del adapter
            _empty_cache()
    finally:
        conn.close()
    return summaries


def _reset_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

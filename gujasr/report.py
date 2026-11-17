from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import manifests, metrics, store

DATASET_ORDER = ("fleurs", "indicvoices", "kathbath")


@dataclass
class Cell:
    model_key: str
    model_name: str
    dataset: str
    n: int
    expected: int
    coverage: float
    wer: float
    cer: float
    cer_lo: float
    cer_hi: float
    capped_cer: float
    drift_pct: float
    loop_pct: float
    rtf: float
    utt_per_s: float

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Macro:
    model_key: str
    model_name: str
    n: int
    expected: int
    coverage: float
    wer: float
    cer: float
    cer_lo: float
    cer_hi: float
    per_dataset: dict[str, float]
    complete: bool


@dataclass
class Comparison:
    model_key: str
    model_name: str
    dataset: str
    reference: str
    delta: float
    lo: float
    hi: float
    p_value: float
    significant: bool


def _group(rows: Sequence[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model_key"], row["dataset"]), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda r: r["row_index"])
    return grouped


def _cell(model_key: str, dataset: str, items: Sequence[dict], reps: int, seed: int) -> Cell:
    refs = [i["reference_norm"] for i in items]
    hyps = [i["hypothesis_norm"] for i in items]
    chars = metrics.counts(refs, hyps, "char")
    words = metrics.counts(refs, hyps, "word")
    lo, hi = metrics.bootstrap_ci(chars, reps=reps, seed=seed)
    duration = sum(i["audio_duration_s"] or 0 for i in items)
    latency = sum(i["latency_s"] or 0 for i in items)
    return Cell(
        model_key=model_key,
        model_name=items[0]["model_name"],
        dataset=dataset,
        n=len(items),
        expected=manifests.DATASETS[dataset]["expected_rows"],
        coverage=round(len(items) / manifests.DATASETS[dataset]["expected_rows"], 4),
        wer=round(words.rate, 4),
        cer=round(chars.rate, 4),
        cer_lo=round(lo, 4),
        cer_hi=round(hi, 4),
        capped_cer=round(float(np.mean([i["capped_cer"] for i in items])), 4),
        drift_pct=round(100.0 * sum(int(i["script_drift"] or 0) for i in items) / len(items), 3),
        loop_pct=round(100.0 * sum(int(i["loop_flag"] or 0) for i in items) / len(items), 3),
        rtf=round(latency / duration, 5) if duration else 0.0,
        utt_per_s=round(len(items) / latency, 3) if latency else 0.0,
    )


def build(
    conn,
    cfg,
    run_keys: Sequence[str] | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    rows = store.fetch_predictions(conn, run_keys)
    grouped = _group(rows)
    datasets = [d for d in DATASET_ORDER if any(ds == d for (_, ds) in grouped)] or list(
        {ds for (_, ds) in grouped}
    )

    cells = [
        _cell(model_key, dataset, items, cfg.bootstrap_reps, cfg.seed)
        for (model_key, dataset), items in sorted(grouped.items())
    ]

    by_model: dict[str, dict[str, Cell]] = {}
    for cell in cells:
        by_model.setdefault(cell.model_key, {})[cell.dataset] = cell

    macros: list[Macro] = []
    for model_key, per in by_model.items():
        complete = all(d in per and per[d].coverage >= 0.99 for d in datasets)
        available = next(d for d in datasets if (model_key, d) in grouped)
        counts_by_dataset = [
            metrics.counts(
                [i["reference_norm"] for i in grouped[(model_key, d)]],
                [i["hypothesis_norm"] for i in grouped[(model_key, d)]],
                "char",
            )
            for d in datasets if (model_key, d) in grouped
        ]
        macro_cer = metrics.macro([per[d].cer for d in datasets if d in per])
        lo, hi = metrics.bootstrap_macro(counts_by_dataset, reps=cfg.bootstrap_reps, seed=cfg.seed)
        macros.append(
            Macro(
                model_key=model_key,
                model_name=grouped[(model_key, available)][0]["model_name"],
                n=sum(per[d].n for d in datasets if d in per),
                expected=sum(
                    manifests.DATASETS[d]["expected_rows"] for d in datasets if d in per
                ),
                coverage=round(float(np.mean([per[d].coverage for d in datasets if d in per])), 4),
                wer=round(metrics.macro([per[d].wer for d in datasets if d in per]), 4),
                cer=round(macro_cer, 4),
                cer_lo=round(lo, 4),
                cer_hi=round(hi, 4),
                per_dataset={d: per[d].cer for d in datasets if d in per},
                complete=complete,
            )
        )

    ranked = sorted((m for m in macros if m.complete), key=lambda m: m.cer)
    reference = reference or (ranked[0].model_key if ranked else None)

    comparisons: list[Comparison] = []
    if reference and reference in by_model:
        for dataset in datasets:
            if dataset not in by_model.get(reference, {}):
                continue
            ref_items = {i["sample_id"]: i for i in grouped[(reference, dataset)]}
            for model_key, per in by_model.items():
                if model_key == reference or dataset not in per:
                    continue
                items = [i for i in grouped[(model_key, dataset)] if i["sample_id"] in ref_items]
                if not items:
                    continue
                left = metrics.counts(
                    [i["reference_norm"] for i in items],
                    [i["hypothesis_norm"] for i in items],
                    "char",
                )
                right = metrics.counts(
                    [ref_items[i["sample_id"]]["reference_norm"] for i in items],
                    [ref_items[i["sample_id"]]["hypothesis_norm"] for i in items],
                    "char",
                )
                result = metrics.paired_bootstrap(left, right, reps=cfg.bootstrap_reps, seed=cfg.seed)
                comparisons.append(
                    Comparison(
                        model_key=model_key,
                        model_name=per[dataset].model_name,
                        dataset=dataset,
                        reference=reference,
                        delta=round(result.delta, 4),
                        lo=round(result.lo, 4),
                        hi=round(result.hi, 4),
                        p_value=round(result.p_value, 5),
                        significant=result.significant,
                    )
                )

    return {
        "generated_from": {
            "runs": sorted({c.model_key for c in cells}),
            "datasets": datasets,
            "bootstrap_reps": cfg.bootstrap_reps,
            "seed": cfg.seed,
        },
        "cells": [c.as_row() for c in cells],
        "macro": [asdict(m) for m in sorted(macros, key=lambda m: (not m.complete, m.cer))],
        "comparisons": [asdict(c) for c in comparisons],
        "reference": reference,
    }


def _pct(value: float) -> str:
    return f"{value:.2f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def markdown(board: dict[str, Any], title: str = "Gujarati ASR leaderboard") -> str:
    lines = [f"# {title}", ""]
    datasets = board["generated_from"]["datasets"]
    lines.append(
        f"Corpus CER over {', '.join(datasets)}. Intervals are 95% bootstrap over utterances "
        f"({board['generated_from']['bootstrap_reps']} resamples, seed {board['generated_from']['seed']})."
    )
    lines.append("")
    lines.append("## Three-dataset macro")
    lines.append("")
    rows = []
    for rank, macro in enumerate([m for m in board["macro"] if m["complete"]], start=1):
        rows.append([
            rank,
            macro["model_name"],
            _pct(macro["cer"]),
            f"{_pct(macro['cer_lo'])}–{_pct(macro['cer_hi'])}",
            _pct(macro["wer"]),
            f"{macro['n']}/{macro['expected']}",
        ])
    lines.append(_table(["#", "Model", "Macro CER", "95% CI", "Macro WER", "N"], rows))
    lines.append("")
    lines.append("## Per dataset")
    lines.append("")
    rows = []
    for cell in sorted(board["cells"], key=lambda c: (c["model_name"], c["dataset"])):
        rows.append([
            cell["model_name"],
            f"{cell['dataset']}",
            f"{cell['n']}/{cell['expected']}",
            _pct(cell["cer"]),
            f"{_pct(cell['cer_lo'])}–{_pct(cell['cer_hi'])}",
            _pct(cell["wer"]),
            f"{cell['drift_pct']:.1f}",
            f"{cell['loop_pct']:.1f}",
            f"{cell['rtf']:.4f}",
        ])
    lines.append(_table(
        ["Model", "Dataset", "N", "CER", "95% CI", "WER", "Drift %", "Loop %", "RTF"], rows
    ))
    if board["comparisons"]:
        lines += ["", f"## Paired against `{board['reference']}`", ""]
        rows = []
        for cmp in board["comparisons"]:
            mark = "yes" if cmp["significant"] else "no"
            p_text = "<0.001" if cmp["p_value"] < 0.001 else f"{cmp['p_value']:.3f}"
            rows.append([
                cmp["model_name"],
                cmp["dataset"],
                cmp["delta"],
                f"{cmp['lo']}–{cmp['hi']}",
                p_text,
                mark,
            ])
        lines.append(_table(["Model", "Dataset", "Δ CER", "95% CI", "p", "Significant"], rows))
        lines.append("")
        lines.append("Positive Δ means the model is worse than the reference on the same utterances.")
    partial = [m for m in board["macro"] if not m["complete"]]
    if partial:
        lines += ["", "## Partial coverage (not ranked)", ""]
        rows = [[m["model_name"], f"{m['n']}", f"{m['coverage'] * 100:.1f}%"] for m in partial]
        lines.append(_table(["Model", "N", "Coverage"], rows))
    return "\n".join(lines) + "\n"

HTML_CSS = """
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#16181d;max-width:980px;margin:48px auto;padding:0 20px}
h1{font-size:30px;letter-spacing:-.02em;margin-bottom:4px}
h2{font-size:19px;margin-top:38px;padding-bottom:6px;border-bottom:1px solid #e6e8ec}
.sub{color:#6b7280;margin:0 0 28px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{text-align:left;font-weight:600;color:#4b5563;border-bottom:1px solid #d1d5db;padding:8px 10px}
td{padding:7px 10px;border-bottom:1px solid #f0f1f3;font-variant-numeric:tabular-nums}
tr:hover td{background:#fafbfc}
.num{text-align:right}
code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12.5px}
"""


def render_html(board: dict[str, Any], title: str = "Gujarati ASR leaderboard") -> str:
    esc = html.escape
    datasets = board["generated_from"]["datasets"]
    out = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{esc(title)}</title>",
        f"<style>{HTML_CSS}</style>",
        f"<h1>{esc(title)}</h1>",
        f"<p class='sub'>Corpus CER over {esc(', '.join(datasets))}. 95% bootstrap interval over "
        f"utterances, {board['generated_from']['bootstrap_reps']} resamples.</p>",
        "<h2>Three-dataset macro</h2><table><tr><th>#</th><th>Model</th>"
        "<th class='num'>Macro CER</th><th class='num'>95% CI</th><th class='num'>Macro WER</th></tr>",
    ]
    for rank, macro in enumerate([m for m in board["macro"] if m["complete"]], start=1):
        out.append(
            f"<tr><td>{rank}</td><td>{esc(macro['model_name'])}</td>"
            f"<td class='num'>{macro['cer']:.2f}</td>"
            f"<td class='num'>{macro['cer_lo']:.2f}–{macro['cer_hi']:.2f}</td>"
            f"<td class='num'>{macro['wer']:.2f}</td></tr>"
        )
    out.append("</table><h2>Per dataset</h2><table><tr><th>Model</th><th>Dataset</th>"
               "<th class='num'>N</th><th class='num'>CER</th><th class='num'>95% CI</th>"
               "<th class='num'>WER</th><th class='num'>Loop %</th><th class='num'>RTF</th></tr>")
    for cell in board["cells"]:
        out.append(
            f"<tr><td>{esc(cell['model_name'])}</td><td>{cell['dataset']}</td>"
            f"<td class='num'>{cell['n']}</td><td class='num'>{cell['cer']:.2f}</td>"
            f"<td class='num'>{cell['cer_lo']:.2f}–{cell['cer_hi']:.2f}</td>"
            f"<td class='num'>{cell['wer']:.2f}</td><td class='num'>{cell['loop_pct']:.1f}</td>"
            f"<td class='num'>{cell['rtf']:.4f}</td></tr>"
        )
    out.append("</table>")
    if board["comparisons"]:
        out.append(f"<h2>Paired against <code>{esc(board['reference'])}</code></h2>"
                   "<table><tr><th>Model</th><th>Dataset</th><th class='num'>Δ CER</th>"
                   "<th class='num'>95% CI</th><th class='num'>p</th></tr>")
        for cmp in board["comparisons"]:
            out.append(
                f"<tr><td>{esc(cmp['model_name'])}</td><td>{cmp['dataset']}</td>"
                f"<td class='num'>{cmp['delta']:+.2f}</td>"
                f"<td class='num'>{cmp['lo']:+.2f}–{cmp['hi']:+.2f}</td>"
                f"<td class='num'>{cmp['p_value']:.4f}</td></tr>"
            )
        out.append("</table>")
    return "\n".join(out)


def write(board: dict[str, Any], out_dir: str | Path, title: str = "Gujarati ASR leaderboard") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    (out / "leaderboard.md").write_text(markdown(board, title), encoding="utf-8")
    (out / "report.html").write_text(render_html(board, title), encoding="utf-8")

    fields = list(Cell.__dataclass_fields__)
    with (out / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in board["cells"]:
            writer.writerow(cell)
    return {"dir": str(out), "files": ["leaderboard.json", "leaderboard.md", "leaderboard.csv", "report.html"]}

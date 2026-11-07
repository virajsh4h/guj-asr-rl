#!/usr/bin/env python3
"""simple gujarati asr inspector with decoupled batch list and sample viewer."""

from __future__ import annotations

import difflib
import html
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
from jiwer import cer as corpus_cer
from jiwer import wer as corpus_wer

from run_baselines_optimized import (
    DEFAULT_DB_PATH,
    TARGET_SR,
    _decode_dataset_row,
    load_gujarati_dataset,
)

# cache folder so audio plays immediately without re-decoding parquet files
WORKSPACE = Path(__file__).resolve().parent
AUDIO_CACHE_DIR = WORKSPACE / "results" / "ui_audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# keep dataset handles in memory
_DATASET_HANDLES: dict[str, tuple[Any, str, str, str]] = {}


def get_cached_wav(dataset_key: str, row_index: int, sample_id: str) -> str | None:
    # return cached wav if already saved on disk
    wav_path = AUDIO_CACHE_DIR / f"{sample_id}.wav"
    if wav_path.exists():
        return str(wav_path)

    # otherwise decode this single sample and write to disk cache
    try:
        if dataset_key not in _DATASET_HANDLES:
            _DATASET_HANDLES[dataset_key] = load_gujarati_dataset(dataset_key)
        ds, split_name, audio_col, text_col = _DATASET_HANDLES[dataset_key]
        item = _decode_dataset_row(ds, dataset_key, split_name, audio_col, text_col, int(row_index))
        if item is None:
            return None
        sf.write(str(wav_path), item["audio_array"], TARGET_SR, subtype="PCM_16")
        return str(wav_path)
    except Exception as exc:
        print(f"[audio cache error] {sample_id}: {exc!r}")
        return None


def make_color_diff_html(title: str, ref: str, hyp: str) -> str:
    # build word-by-word green (match), yellow (substitution), red (deletion/insertion) diff
    ref_words = (ref or "").split()
    hyp_words = (hyp or "").split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)

    tokens = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # green for exact match
            text = html.escape(" ".join(hyp_words[j1:j2]))
            tokens.append(
                f'<span style="background:#065f46;color:#ecfdf5;padding:3px 7px;border-radius:5px;margin:2px;display:inline-block;">{text}</span>'
            )
        elif tag == "replace":
            # yellow for substitution (gt -> pred)
            gt_txt = html.escape(" ".join(ref_words[i1:i2]))
            pred_txt = html.escape(" ".join(hyp_words[j1:j2]))
            tokens.append(
                f'<span style="background:#ca8a04;color:#fefce8;padding:3px 7px;border-radius:5px;margin:2px;display:inline-block;">'
                f'{gt_txt} &rarr; <b>{pred_txt}</b></span>'
            )
        elif tag == "delete":
            # red strikethrough for deleted ground truth words
            gt_txt = html.escape(" ".join(ref_words[i1:i2]))
            tokens.append(
                f'<span style="background:#991b1b;color:#fef2f2;padding:3px 7px;border-radius:5px;margin:2px;display:inline-block;text-decoration:line-through;">'
                f'{gt_txt}</span>'
            )
        elif tag == "insert":
            # red +word for inserted / extra words
            pred_txt = html.escape(" ".join(hyp_words[j1:j2]))
            tokens.append(
                f'<span style="background:#991b1b;color:#fef2f2;padding:3px 7px;border-radius:5px;margin:2px;display:inline-block;">'
                f'+{pred_txt}</span>'
            )

    diff_body = " ".join(tokens) if tokens else "<i>(empty prediction)</i>"
    return (
        f'<div style="background:#111827;border:1px solid #374151;border-radius:8px;padding:12px;margin-bottom:8px;font-size:17px;line-height:1.9;">'
        f'<div style="font-size:13px;color:#9ca3af;font-weight:700;margin-bottom:6px;">{html.escape(title)}</div>'
        f'<div>{diff_body}</div></div>'
    )


def format_metrics_line(label: str, n: int, cer: float, wer: float, capped: float, drift: float, loop: float, rtf: float) -> str:
    # format one clean line of tracked metrics
    return (
        f"<div style='margin:3px 0;'><b>{html.escape(label)} (N={n}):</b> &nbsp; "
        f"CER: <b>{cer:.2f}%</b> &nbsp;|&nbsp; "
        f"WER: <b>{wer:.2f}%</b> &nbsp;|&nbsp; "
        f"Capped CER: <b>{capped:.2f}%</b> &nbsp;|&nbsp; "
        f"Drift: <b>{drift:.1f}%</b> &nbsp;|&nbsp; "
        f"Loops: <b>{loop:.1f}%</b> &nbsp;|&nbsp; "
        f"Median RTF: <b>{rtf:.4f}</b></div>"
    )


def load_batch_list(dataset: str, sort_by: str, page_size: int, start_offset: int):
    # concern 1: load the side-by-side table of samples for both models at once
    order_map = {
        "Gemma Worst CER (High to Low)": "g.cer DESC, g.wer DESC",
        "Gemma Best CER (Low to High)": "g.cer ASC, g.wer ASC",
        "Indic-Transcribe Worst CER (High to Low)": "b.cer DESC, b.wer DESC",
        "Indic-Transcribe Best CER (Low to High)": "b.cer ASC, b.wer ASC",
        "Gemma Worst WER (High to Low)": "g.wer DESC, g.cer DESC",
        "Gemma Best WER (Low to High)": "g.wer ASC, g.cer ASC",
        "Indic-Transcribe Worst WER (High to Low)": "b.wer DESC, b.cer DESC",
        "Indic-Transcribe Best WER (Low to High)": "b.wer ASC, b.cer ASC",
        "Biggest Gap (Gemma Worse than Target)": "(g.cer - b.cer) DESC",
        "Sample Order (Row Index)": "g.row_index ASC",
    }
    order_sql = order_map.get(sort_by, "g.cer DESC, g.wer DESC")

    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    rows = conn.execute(
        f"""
        SELECT
            g.sample_id,
            g.row_index,
            ROUND(g.audio_duration_s, 2) AS dur_s,
            ROUND(g.cer, 2) AS gemma_cer,
            ROUND(b.cer, 2) AS target_cer,
            ROUND(g.wer, 2) AS gemma_wer,
            ROUND(b.wer, 2) AS target_wer,
            g.reference_norm AS gt,
            g.hypothesis_norm AS gemma_pred,
            b.hypothesis_norm AS target_pred,
            g.capped_cer, g.script_drift, g.loop_flag, g.rtf,
            b.capped_cer, b.script_drift, b.loop_flag, b.rtf
        FROM predictions g
        JOIN predictions b
          ON g.dataset = b.dataset
         AND g.split = b.split
         AND g.sample_id = b.sample_id
         AND b.model_name = 'indic-transcribe-core'
        WHERE g.dataset = ?
          AND g.model_name = 'gemma-4-E4B-sft'
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
        """,
        (dataset, int(page_size), int(start_offset)),
    ).fetchall()
    conn.close()

    if not rows:
        empty_df = pd.DataFrame(columns=[
            "sample_id", "dur_s", "gemma_cer", "target_cer",
            "gemma_wer", "target_wer", "ground_truth", "gemma_prediction", "target_prediction"
        ])
        return empty_df, "No samples found for this offset."

    # compute batch metrics for both models on these visible rows
    refs = [r[7] or "" for r in rows if r[7]]
    g_hyps = [r[8] or "" for r in rows if r[7]]
    b_hyps = [r[9] or "" for r in rows if r[7]]

    g_line = format_metrics_line(
        f"Gemma 4 E4B (Current Page on {dataset})",
        len(rows),
        100.0 * corpus_cer(refs, g_hyps) if refs else 0.0,
        100.0 * corpus_wer(refs, g_hyps) if refs else 0.0,
        float(np.mean([r[10] or 0.0 for r in rows])),
        100.0 * sum(int(r[11] or 0) for r in rows) / len(rows),
        100.0 * sum(int(r[12] or 0) for r in rows) / len(rows),
        float(median([r[13] or 0.0 for r in rows])),
    )
    b_line = format_metrics_line(
        f"Indic-Transcribe Target (Current Page on {dataset})",
        len(rows),
        100.0 * corpus_cer(refs, b_hyps) if refs else 0.0,
        100.0 * corpus_wer(refs, b_hyps) if refs else 0.0,
        float(np.mean([r[14] or 0.0 for r in rows])),
        100.0 * sum(int(r[15] or 0) for r in rows) / len(rows),
        100.0 * sum(int(r[16] or 0) for r in rows) / len(rows),
        float(median([r[17] or 0.0 for r in rows])),
    )
    batch_summary_html = (
        f"<div style='background:#1f2937;border:1px solid #374151;border-radius:8px;padding:10px 14px;color:#f9fafb;font-size:14px;'>"
        f"{g_line}{b_line}</div>"
    )

    # build dataframe showing gt and both predictions on the same row
    df = pd.DataFrame([
        {
            "sample_id": r[0],
            "dur_s": r[2],
            "gemma_cer": r[3],
            "target_cer": r[4],
            "gemma_wer": r[5],
            "target_wer": r[6],
            "ground_truth": r[7],
            "gemma_prediction": r[8],
            "target_prediction": r[9],
        }
        for r in rows
    ])
    return df, batch_summary_html


def inspect_locked_sample(sample_id: str, view_mode: str):
    # concern 2: inspect one locked sample_id across either or both models without touching the table
    if not sample_id or not str(sample_id).strip():
        return None, "", "", "Click any row in the table above to inspect a sample.", ""

    sample_id = str(sample_id).strip()
    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    rows = conn.execute(
        """
        SELECT dataset, row_index, audio_duration_s, model_name,
               reference_norm, hypothesis_norm, reference_raw, hypothesis_raw,
               cer, wer, capped_cer, script_drift, loop_flag, rtf
        FROM predictions
        WHERE sample_id = ?
          AND model_name IN ('gemma-4-E4B-sft', 'indic-transcribe-core')
        """,
        (sample_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return None, "", "", f"Sample {sample_id} not found.", ""

    ds, row_idx, dur = rows[0][0], rows[0][1], rows[0][2]
    ref_norm, ref_raw = rows[0][4], rows[0][6]
    wav_path = get_cached_wav(ds, row_idx, sample_id)

    by_model = {r[3]: r for r in rows}
    gt_text = f"{ref_norm}\n\n[raw]: {ref_raw}"

    legend = (
        '<div style="margin-bottom:8px;font-size:13px;color:#d1d5db;">'
        '<span style="background:#065f46;color:#ecfdf5;padding:2px 6px;border-radius:4px;margin-right:8px;">Green = Exact Match</span>'
        '<span style="background:#ca8a04;color:#fefce8;padding:2px 6px;border-radius:4px;margin-right:8px;">Yellow = Substitution (GT &rarr; Pred)</span>'
        '<span style="background:#991b1b;color:#fef2f2;padding:2px 6px;border-radius:4px;">Red = Deletion (strikethrough) or Insertion (+extra)</span>'
        '</div>'
    )

    # build view for selected model (or both stacked) without changing sample_id
    if view_mode == "Both Models Stacked":
        models_to_show = ["gemma-4-E4B-sft", "indic-transcribe-core"]
    else:
        models_to_show = [view_mode]

    pred_lines = []
    diff_blocks = [legend]
    metric_lines = []

    for m_name in models_to_show:
        if m_name not in by_model:
            continue
        r = by_model[m_name]
        hyp_norm, hyp_raw = r[5] or "", r[7] or ""
        cer, wer, capped, drift, loop, rtf = float(r[8] or 0), float(r[9] or 0), float(r[10] or 0), int(r[11] or 0), int(r[12] or 0), float(r[13] or 0)

        pred_lines.append(f"[{m_name}]: {hyp_norm}\n[raw]: {hyp_raw}")
        diff_blocks.append(
            make_color_diff_html(f"{m_name} (CER: {cer:.2f}% | WER: {wer:.2f}%)", ref_norm, hyp_norm)
        )
        metric_lines.append(
            format_metrics_line(f"{sample_id} ({dur:.2f}s) — {m_name}", 1, cer, wer, capped, 100.0 * drift, 100.0 * loop, rtf)
        )

    sample_metrics_html = (
        f"<div style='background:#1f2937;border:1px solid #10b981;border-radius:8px;padding:10px 14px;color:#f9fafb;font-size:14px;'>"
        f"{''.join(metric_lines)}</div>"
    )
    return wav_path, gt_text, "\n\n".join(pred_lines), "".join(diff_blocks), sample_metrics_html


def on_table_row_select(df: pd.DataFrame, view_mode: str, evt: gr.SelectData):
    # clicking any row locks its sample_id into active_sample_box and opens the inspector
    if df is None or df.empty or evt.index is None:
        return "", None, "", "", "", ""
    row_pos = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    sample_id = str(df.iloc[row_pos]["sample_id"])
    wav_path, gt_text, pred_text, diff_html, sample_metrics_html = inspect_locked_sample(sample_id, view_mode)
    return sample_id, wav_path, gt_text, pred_text, diff_html, sample_metrics_html


def on_initial_load(dataset: str, sort_by: str, page_size: int, start_offset: int, view_mode: str):
    # populate table and load the first row on initial startup only
    df, batch_html = load_batch_list(dataset, sort_by, page_size, start_offset)
    if df.empty:
        return df, batch_html, "", None, "", "", "", ""
    first_sid = str(df.iloc[0]["sample_id"])
    wav_path, gt_text, pred_text, diff_html, sample_metrics_html = inspect_locked_sample(first_sid, view_mode)
    return df, batch_html, first_sid, wav_path, gt_text, pred_text, diff_html, sample_metrics_html


def build_ui():
    with gr.Blocks(title="Gujarati ASR Inspector") as demo:
        gr.Markdown("## Gujarati ASR Inspector (Decoupled List & Sample Viewer)")

        # concern 1: pick dataset & sort metric (up to 20 visible rows with gt + both predictions)
        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=["fleurs", "indicvoices", "kathbath"],
                value="fleurs",
                label="1. Dataset",
            )
            sort_dd = gr.Dropdown(
                choices=[
                    "Gemma Worst CER (High to Low)",
                    "Gemma Best CER (Low to High)",
                    "Indic-Transcribe Worst CER (High to Low)",
                    "Indic-Transcribe Best CER (Low to High)",
                    "Gemma Worst WER (High to Low)",
                    "Gemma Best WER (Low to High)",
                    "Indic-Transcribe Worst WER (High to Low)",
                    "Indic-Transcribe Best WER (Low to High)",
                    "Biggest Gap (Gemma Worse than Target)",
                    "Sample Order (Row Index)",
                ],
                value="Gemma Worst CER (High to Low)",
                label="2. Sort Table By",
            )
            page_slider = gr.Slider(
                minimum=1, maximum=20, value=20, step=1,
                label="Rows Visible (Max 20)",
            )
            offset_num = gr.Number(
                value=0, minimum=0, step=20, precision=0,
                label="Start Row Offset (0, 20, 40...)",
            )
            refresh_btn = gr.Button("Update List", variant="primary")

        # batch metrics bar comparing both models on the current page
        batch_metrics_box = gr.HTML()

        # side-by-side table: click any row to lock that sample into the inspector below
        batch_table = gr.Dataframe(
            label="Step 1: Browse Samples (GT, Gemma & Indic-Transcribe on the same row — click any row to inspect below)",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("---")

        # concern 2: locked single-sample inspector (switching model here never resets the table or sample!)
        with gr.Row():
            active_sample_box = gr.Textbox(
                label="Locked Sample ID (Click a table row above or type ID)",
                value="fleurs_000409",
                scale=1,
            )
            view_mode_radio = gr.Radio(
                choices=["gemma-4-E4B-sft", "indic-transcribe-core", "Both Models Stacked"],
                value="Both Models Stacked",
                label="Step 2: Switch Model View on Locked Sample (Does NOT reset sample or table)",
                scale=2,
            )

        sample_metrics_box = gr.HTML()

        with gr.Row():
            with gr.Column(scale=1):
                audio_out = gr.Audio(label="Sample Audio (16kHz Cached)", type="filepath")
            with gr.Column(scale=2):
                gt_box = gr.Textbox(label="Ground Truth (GT)", lines=2, interactive=False)
                pred_box = gr.Textbox(label="Model Prediction(s)", lines=3, interactive=False)

        diff_box = gr.HTML(label="Word-Level Diff (Green = Match | Yellow = Substitution | Red = Deletion/Insertion)")

        # wire concern 1: changing dataset/sort/offset only updates the table & batch metrics
        table_inputs = [dataset_dd, sort_dd, page_slider, offset_num]
        table_outputs = [batch_table, batch_metrics_box]
        for ctrl in table_inputs:
            ctrl.change(fn=load_batch_list, inputs=table_inputs, outputs=table_outputs)
        refresh_btn.click(fn=load_batch_list, inputs=table_inputs, outputs=table_outputs)

        # wire concern 2: clicking a table row updates active_sample_box and opens the inspector
        batch_table.select(
            fn=on_table_row_select,
            inputs=[batch_table, view_mode_radio],
            outputs=[active_sample_box, audio_out, gt_box, pred_box, diff_box, sample_metrics_box],
        )

        # switching the radio button or editing active_sample_box ONLY refreshes the locked sample view
        view_mode_radio.change(
            fn=inspect_locked_sample,
            inputs=[active_sample_box, view_mode_radio],
            outputs=[audio_out, gt_box, pred_box, diff_html := diff_box, sample_metrics_box],
        )
        active_sample_box.submit(
            fn=inspect_locked_sample,
            inputs=[active_sample_box, view_mode_radio],
            outputs=[audio_out, gt_box, pred_box, diff_box, sample_metrics_box],
        )

        # load initial view once
        demo.load(
            fn=on_initial_load,
            inputs=[dataset_dd, sort_dd, page_slider, offset_num, view_mode_radio],
            outputs=[batch_table, batch_metrics_box, active_sample_box, audio_out, gt_box, pred_box, diff_box, sample_metrics_box],
        )

    return demo


if __name__ == "__main__":
    print("Launching sft version!")
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True)

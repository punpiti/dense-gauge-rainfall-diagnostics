#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_convective_thresholds.py

Apply rule-based convective thresholds to storm-level results.

Input
-----
all_storm_results.parquet or .csv

Expected columns
----------------
- status
- storm_id
- storm_start
- storm_end
- class_peak_pos_area_cells
- class_peak_pos_sum
- class_event_pos_sum
- class_n_strong_frames

Outputs
-------
- labeled full dataset
- convective storms only
- non-convective storms only
- summary csv

Example
-------
python apply_convective_thresholds.py \
  --input results/storm_convective_fast_test/all_storm_results.csv \
  --outdir results/storm_convective_fast/labeled \
  --area-threshold 500 \
  --event-sum-threshold 4000 \
  --peak-sum-threshold 1000 \
  --frames-threshold 10 \
  --mode strict
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="all_storm_results.parquet or .csv")
    ap.add_argument("--outdir", required=True, help="output directory")

    ap.add_argument("--status-filter", default="done", help="default: done")

    ap.add_argument("--area-threshold", type=float, default=500.0)
    ap.add_argument("--event-sum-threshold", type=float, default=4000.0)
    ap.add_argument("--peak-sum-threshold", type=float, default=1000.0)
    ap.add_argument("--frames-threshold", type=float, default=10.0)

    ap.add_argument(
        "--mode",
        choices=["strict", "score"],
        default="strict",
        help="strict: area AND (event_sum OR peak_sum OR frames), "
             "score: use number of passed conditions",
    )

    ap.add_argument(
        "--min-score",
        type=int,
        default=2,
        help="used only in score mode; convective if score >= min-score",
    )

    ap.add_argument(
        "--use-area-as-required",
        action="store_true",
        help="in score mode, require area threshold as mandatory in addition to min-score",
    )

    return ap.parse_args()


def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input)

    if "status" in df.columns and args.status_filter:
        df = df[df["status"].astype(str) == str(args.status_filter)].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering.")

    required = [
        "class_peak_pos_area_cells",
        "class_peak_pos_sum",
        "class_event_pos_sum",
        "class_n_strong_frames",
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    d = df.copy()

    d["class_peak_pos_area_cells"] = numeric(d, "class_peak_pos_area_cells")
    d["class_peak_pos_sum"] = numeric(d, "class_peak_pos_sum")
    d["class_event_pos_sum"] = numeric(d, "class_event_pos_sum")
    d["class_n_strong_frames"] = numeric(d, "class_n_strong_frames")

    d["passes_area"] = d["class_peak_pos_area_cells"] >= args.area_threshold
    d["passes_peak_sum"] = d["class_peak_pos_sum"] >= args.peak_sum_threshold
    d["passes_event_sum"] = d["class_event_pos_sum"] >= args.event_sum_threshold
    d["passes_frames"] = d["class_n_strong_frames"] >= args.frames_threshold

    d["convective_rule_score"] = (
        d["passes_area"].astype(int)
        + d["passes_peak_sum"].astype(int)
        + d["passes_event_sum"].astype(int)
        + d["passes_frames"].astype(int)
    )

    if args.mode == "strict":
        d["convective_rule_label"] = np.where(
            d["passes_area"] & (d["passes_peak_sum"] | d["passes_event_sum"] | d["passes_frames"]),
            "convective",
            "non_convective",
        )
    else:
        if args.use_area_as_required:
            d["convective_rule_label"] = np.where(
                d["passes_area"] & (d["convective_rule_score"] >= args.min_score),
                "convective",
                "non_convective",
            )
        else:
            d["convective_rule_label"] = np.where(
                d["convective_rule_score"] >= args.min_score,
                "convective",
                "non_convective",
            )

    # convenient subtype
    d["convective_rule_reason"] = ""
    d.loc[d["convective_rule_label"] == "convective", "convective_rule_reason"] = (
        "area=" + d["passes_area"].astype(str)
        + ";peak_sum=" + d["passes_peak_sum"].astype(str)
        + ";event_sum=" + d["passes_event_sum"].astype(str)
        + ";frames=" + d["passes_frames"].astype(str)
    )

    # outputs
    full_parquet = outdir / "all_storm_results_labeled.parquet"
    full_csv = outdir / "all_storm_results_labeled.csv"

    conv_parquet = outdir / "convective_storms.parquet"
    conv_csv = outdir / "convective_storms.csv"

    nonconv_parquet = outdir / "nonconvective_storms.parquet"
    nonconv_csv = outdir / "nonconvective_storms.csv"

    d.to_parquet(full_parquet, index=False)
    d.to_csv(full_csv, index=False)

    conv = d[d["convective_rule_label"] == "convective"].copy()
    nonconv = d[d["convective_rule_label"] == "non_convective"].copy()

    conv.to_parquet(conv_parquet, index=False)
    conv.to_csv(conv_csv, index=False)

    nonconv.to_parquet(nonconv_parquet, index=False)
    nonconv.to_csv(nonconv_csv, index=False)

    # summary
    summary = {
        "n_total": int(len(d)),
        "n_convective": int(len(conv)),
        "n_non_convective": int(len(nonconv)),
        "fraction_convective": float(len(conv) / len(d)) if len(d) else np.nan,
        "mode": args.mode,
        "area_threshold": args.area_threshold,
        "peak_sum_threshold": args.peak_sum_threshold,
        "event_sum_threshold": args.event_sum_threshold,
        "frames_threshold": args.frames_threshold,
        "min_score": args.min_score if args.mode == "score" else np.nan,
        "use_area_as_required": bool(args.use_area_as_required) if args.mode == "score" else np.nan,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(outdir / "label_summary.csv", index=False)
    summary_df.to_parquet(outdir / "label_summary.parquet", index=False)

    print("[INFO] labeling complete")
    for k, v in summary.items():
        print(f"{k:24s}: {v}")

    print("\nTop convective storms:")
    cols = [
        c for c in [
            "storm_id",
            "storm_start",
            "storm_end",
            "class_peak_pos_area_cells",
            "class_peak_pos_sum",
            "class_event_pos_sum",
            "class_n_strong_frames",
            "convective_rule_score",
            "convective_rule_reason",
        ] if c in conv.columns
    ]
    if len(conv):
        print(conv.sort_values(
            ["convective_rule_score", "class_event_pos_sum", "class_peak_pos_area_cells"],
            ascending=[False, False, False]
        )[cols].head(20).to_string(index=False))

    print(f"\n[SAVED] {full_parquet}")
    print(f"[SAVED] {full_csv}")
    print(f"[SAVED] {conv_parquet}")
    print(f"[SAVED] {conv_csv}")
    print(f"[SAVED] {nonconv_parquet}")
    print(f"[SAVED] {nonconv_csv}")


if __name__ == "__main__":
    main()
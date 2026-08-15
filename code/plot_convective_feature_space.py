#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_convective_feature_space.py

Visualize storm-level convective feature space.

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

Features
--------
- Scatter plots for important metric pairs
- Optional threshold lines
- Optional highlight of storms passing rule-based convective criteria
- Optional text labels for top-N storms
- Histograms for selected metrics
- Saves all figures to output directory

Example
-------
python plot_convective_feature_space.py \
  --input results/storm_convective_fast_test/all_storm_results.csv \
  --outdir results/storm_convective_fast/feature_space \
  --area-threshold 500 \
  --frames-threshold 10 \
  --peak-sum-threshold 1000 \
  --label-top-n 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCATTER_PAIRS = [
    ("class_peak_pos_area_cells", "class_n_strong_frames"),
    ("class_peak_pos_area_cells", "class_peak_pos_sum"),
    ("class_peak_pos_area_cells", "class_event_pos_sum"),
    ("class_peak_pos_sum", "class_event_pos_sum"),
    ("class_peak_pos_sum", "class_n_strong_frames"),
    ("class_event_pos_sum", "class_n_strong_frames"),
]

LABELS = {
    "class_peak_pos_area_cells": "Peak positive-residual area (1 km grid cells)",
    "class_peak_pos_sum": "Peak positive-residual sum (grid-cell sum)",
    "class_event_pos_sum": "Event-integrated positive residual (grid-cell-time sum)",
    "class_n_strong_frames": "Strong-residual frames (15-min frames)",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_storm_results.parquet or .csv")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--status-filter", default="done", help="default: done")
    ap.add_argument("--area-threshold", type=float, default=500.0)
    ap.add_argument("--frames-threshold", type=float, default=10.0)
    ap.add_argument("--peak-sum-threshold", type=float, default=1000.0)
    ap.add_argument("--event-sum-threshold", type=float, default=4000.0)
    ap.add_argument("--label-top-n", type=int, default=0, help="label top N storms by event_pos_sum")
    ap.add_argument("--log-event-sum", action="store_true")
    ap.add_argument("--log-peak-sum", action="store_true")
    ap.add_argument("--hist-bins", type=int, default=20)
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


def add_convective_flag(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["class_peak_pos_area_cells"] = numeric(out, "class_peak_pos_area_cells")
    out["class_n_strong_frames"] = numeric(out, "class_n_strong_frames")
    out["class_peak_pos_sum"] = numeric(out, "class_peak_pos_sum")
    out["class_event_pos_sum"] = numeric(out, "class_event_pos_sum")

    out["rule_convective"] = (
        (out["class_peak_pos_area_cells"] >= args.area_threshold) &
        (
            (out["class_n_strong_frames"] >= args.frames_threshold) |
            (out["class_peak_pos_sum"] >= args.peak_sum_threshold) |
            (out["class_event_pos_sum"] >= args.event_sum_threshold)
        )
    )
    return out


def scatter_plot(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_png: Path,
    args: argparse.Namespace,
    label_df: pd.DataFrame | None = None,
) -> None:
    d = df[[xcol, ycol, "rule_convective", "storm_id"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna()

    if len(d) == 0:
        return

    neg = d[~d["rule_convective"]]
    pos = d[d["rule_convective"]]

    plt.figure(figsize=(7, 5.5))
    if len(neg):
        plt.scatter(neg[xcol], neg[ycol], alpha=0.8, label="Non-convective")
    if len(pos):
        plt.scatter(pos[xcol], pos[ycol], alpha=0.95, label="Convective")

    if xcol == "class_peak_pos_area_cells":
        plt.axvline(args.area_threshold)
    if xcol == "class_n_strong_frames":
        plt.axvline(args.frames_threshold)
    if xcol == "class_peak_pos_sum":
        plt.axvline(args.peak_sum_threshold)
    if xcol == "class_event_pos_sum":
        plt.axvline(args.event_sum_threshold)

    if ycol == "class_peak_pos_area_cells":
        plt.axhline(args.area_threshold)
    if ycol == "class_n_strong_frames":
        plt.axhline(args.frames_threshold)
    if ycol == "class_peak_pos_sum":
        plt.axhline(args.peak_sum_threshold)
    if ycol == "class_event_pos_sum":
        plt.axhline(args.event_sum_threshold)

    if label_df is not None and len(label_df):
        tmp = d.merge(label_df[["storm_id"]], on="storm_id", how="inner")
        for _, r in tmp.iterrows():
            plt.annotate(str(r["storm_id"]), (r[xcol], r[ycol]), fontsize=8)

    if args.log_peak_sum:
        if xcol == "class_peak_pos_sum" and (d[xcol] > 0).all():
            plt.xscale("log")
        if ycol == "class_peak_pos_sum" and (d[ycol] > 0).all():
            plt.yscale("log")

    if args.log_event_sum:
        if xcol == "class_event_pos_sum" and (d[xcol] > 0).all():
            plt.xscale("log")
        if ycol == "class_event_pos_sum" and (d[ycol] > 0).all():
            plt.yscale("log")

    plt.xlabel(LABELS.get(xcol, xcol))
    plt.ylabel(LABELS.get(ycol, ycol))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def histogram_plot(df: pd.DataFrame, col: str, out_png: Path, bins: int = 20) -> None:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return

    plt.figure(figsize=(7, 4.5))
    plt.hist(s.to_numpy(), bins=bins)
    plt.xlabel(LABELS.get(col, col))
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_summary(df: pd.DataFrame, outdir: Path) -> None:
    cols = [
        "storm_id",
        "storm_start",
        "storm_end",
        "storm_duration_minutes",
        "n_member_events",
        "n_member_stations",
        "class_peak_pos_area_cells",
        "class_peak_pos_sum",
        "class_event_pos_sum",
        "class_n_strong_frames",
        "rule_convective",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].sort_values(
        ["rule_convective", "class_event_pos_sum"],
        ascending=[False, False]
    ).to_csv(outdir / "feature_space_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input)
    if "status" in df.columns and args.status_filter:
        df = df[df["status"].astype(str) == str(args.status_filter)].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering.")

    need_cols = [
        "class_peak_pos_area_cells",
        "class_peak_pos_sum",
        "class_event_pos_sum",
        "class_n_strong_frames",
    ]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    df = add_convective_flag(df, args)

    print(f"[INFO] storms plotted              : {len(df)}")
    print(f"[INFO] rule-convective storms     : {int(df['rule_convective'].sum())}")
    print(f"[INFO] area threshold             : {args.area_threshold}")
    print(f"[INFO] frames threshold           : {args.frames_threshold}")
    print(f"[INFO] peak sum threshold         : {args.peak_sum_threshold}")
    print(f"[INFO] event sum threshold        : {args.event_sum_threshold}")

    label_df = None
    if args.label_top_n > 0 and "storm_id" in df.columns:
        tmp = df.copy()
        tmp["class_event_pos_sum"] = pd.to_numeric(tmp["class_event_pos_sum"], errors="coerce")
        label_df = tmp.sort_values("class_event_pos_sum", ascending=False).head(args.label_top_n)

    for xcol, ycol in SCATTER_PAIRS:
        out_png = outdir / f"scatter_{xcol}_vs_{ycol}.png"
        scatter_plot(df, xcol, ycol, out_png, args, label_df=label_df)

    for col in need_cols:
        histogram_plot(df, col, outdir / f"hist_{col}.png", bins=args.hist_bins)

    save_summary(df, outdir)

    conv = df[df["rule_convective"]].copy()
    nonconv = df[~df["rule_convective"]].copy()

    conv.to_csv(outdir / "rule_convective_storms.csv", index=False)
    nonconv.to_csv(outdir / "rule_nonconvective_storms.csv", index=False)
    try:
        conv.to_parquet(outdir / "rule_convective_storms.parquet", index=False)
        nonconv.to_parquet(outdir / "rule_nonconvective_storms.parquet", index=False)
    except ImportError:
        print("[SKIP] parquet engine unavailable for rule storm exports")

    print(f"[SAVED] {outdir / 'feature_space_summary.csv'}")
    print(f"[SAVED] {outdir / 'rule_convective_storms.csv'}")
    print(f"[SAVED] {outdir / 'rule_nonconvective_storms.csv'}")


if __name__ == "__main__":
    main()

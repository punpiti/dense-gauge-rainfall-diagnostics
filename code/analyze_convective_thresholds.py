#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_convective_thresholds.py

Analyze storm-level convective classification metrics and suggest
data-driven thresholds using a simple knee/elbow method.

Input
-----
all_storm_results.parquet or all_storm_results.csv

Expected columns (some optional):
- status
- class_peak_pos_area_cells
- class_peak_pos_sum
- class_event_pos_sum
- class_n_strong_frames

Outputs
-------
1) Console summary
2) CSV/Parquet summary of metric statistics and suggested thresholds
3) Graphs:
   - histogram for each metric
   - sorted curve + knee for each metric
   - combined boxplot
   - scatter plots between important metric pairs

Example
-------
python analyze_convective_thresholds.py \
  --input results/storm_convective_fast/all_storm_results.parquet \
  --outdir results/storm_convective_fast/threshold_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "class_peak_pos_area_cells",
    "class_peak_pos_sum",
    "class_event_pos_sum",
    "class_n_strong_frames",
]

DEFAULT_SCATTER_PAIRS = [
    ("class_peak_pos_area_cells", "class_peak_pos_sum"),
    ("class_peak_pos_area_cells", "class_event_pos_sum"),
    ("class_peak_pos_area_cells", "class_n_strong_frames"),
    ("class_peak_pos_sum", "class_event_pos_sum"),
    ("class_peak_pos_sum", "class_n_strong_frames"),
    ("class_event_pos_sum", "class_n_strong_frames"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_storm_results.parquet or .csv")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="metrics to analyze",
    )
    ap.add_argument(
        "--status-filter",
        default="done",
        help="use only rows with this status; default=done",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="minimum non-null sample size required to analyze a metric",
    )
    ap.add_argument(
        "--save-filtered",
        action="store_true",
        help="save filtered input used for analysis",
    )
    ap.add_argument(
        "--hist-bins",
        type=int,
        default=20,
        help="number of bins for histograms",
    )
    ap.add_argument(
        "--log-x-hist",
        action="store_true",
        help="use log x-scale for histograms of positive-valued metrics",
    )
    ap.add_argument(
        "--log-y-sorted",
        action="store_true",
        help="use log y-scale for sorted curves of positive-valued metrics",
    )
    return ap.parse_args()


def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_knee_threshold(values: np.ndarray) -> tuple[float, int]:
    """
    Find knee on sorted ascending curve using max distance from the line
    connecting first and last points.
    """
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)

    if len(y) < 3:
        return float(y[0]), 0

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy)
    if norm == 0:
        return float(y[0]), 0

    distances = np.abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm
    knee_idx = int(np.argmax(distances))
    threshold = float(y[knee_idx])
    return threshold, knee_idx


def summarize_metric(series: pd.Series) -> dict:
    s = ensure_numeric(series).dropna().sort_values().reset_index(drop=True)
    arr = s.to_numpy(dtype=float)

    out = {
        "count": int(len(arr)),
        "min": np.nan,
        "p10": np.nan,
        "p25": np.nan,
        "median": np.nan,
        "p75": np.nan,
        "p90": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "knee_threshold": np.nan,
        "knee_index": np.nan,
        "n_at_or_above_knee": np.nan,
        "fraction_at_or_above_knee": np.nan,
    }

    if len(arr) == 0:
        return out

    out.update({
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    })

    thr, idx = compute_knee_threshold(arr)
    n_ge = int(np.sum(arr >= thr))

    out.update({
        "knee_threshold": float(thr),
        "knee_index": int(idx),
        "n_at_or_above_knee": int(n_ge),
        "fraction_at_or_above_knee": float(n_ge / len(arr)),
    })
    return out


def plot_sorted_metric(
    series: pd.Series,
    metric_name: str,
    out_png: Path,
    log_y: bool = False,
) -> Optional[dict]:
    s = ensure_numeric(series).dropna().sort_values().reset_index(drop=True)
    if len(s) < 3:
        return None

    arr = s.to_numpy(dtype=float)
    thr, idx = compute_knee_threshold(arr)

    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(len(arr)), arr)
    plt.axhline(thr)
    plt.axvline(idx)
    if log_y and np.all(arr > 0):
        plt.yscale("log")
    plt.xlabel("Sorted storm rank")
    plt.ylabel(metric_name)
    plt.title(f"Sorted curve and knee: {metric_name}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return {"knee_threshold": thr, "knee_index": idx}


def plot_histogram(
    series: pd.Series,
    metric_name: str,
    out_png: Path,
    bins: int = 20,
    log_x: bool = False,
) -> None:
    s = ensure_numeric(series).dropna()
    if len(s) == 0:
        return

    arr = s.to_numpy(dtype=float)
    thr, _ = compute_knee_threshold(np.sort(arr))

    plt.figure(figsize=(7, 4.5))
    plt.hist(arr, bins=bins)
    plt.axvline(thr)
    if log_x and np.all(arr > 0):
        plt.xscale("log")
    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.title(f"Histogram: {metric_name}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_boxplot(df: pd.DataFrame, metrics: List[str], out_png: Path) -> None:
    use_metrics = [m for m in metrics if m in df.columns]
    if not use_metrics:
        return

    data = []
    labels = []
    for m in use_metrics:
        s = ensure_numeric(df[m]).dropna()
        if len(s) == 0:
            continue
        data.append(s.to_numpy(dtype=float))
        labels.append(m)

    if not data:
        return

    plt.figure(figsize=(10, 5.5))
    plt.boxplot(data, labels=labels, vert=True)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Value")
    plt.title("Boxplot of convective metrics")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_scatter_pair(df: pd.DataFrame, xcol: str, ycol: str, out_png: Path) -> None:
    if xcol not in df.columns or ycol not in df.columns:
        return

    d = df[[xcol, ycol]].copy()
    d[xcol] = ensure_numeric(d[xcol])
    d[ycol] = ensure_numeric(d[ycol])
    d = d.dropna()

    if len(d) < 3:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(d[xcol].to_numpy(), d[ycol].to_numpy())
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def apply_single_threshold(df: pd.DataFrame, metric: str, thr: float) -> pd.DataFrame:
    d = df.copy()
    d[metric] = ensure_numeric(d[metric])
    return d[d[metric] >= thr].copy()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.input)

    if "status" in df.columns and args.status_filter:
        df = df[df["status"].astype(str) == str(args.status_filter)].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering.")

    if args.save_filtered:
        filtered_path = outdir / "filtered_input.parquet"
        df.to_parquet(filtered_path, index=False)
        print(f"[SAVED] filtered input: {filtered_path}")

    summary_rows = []

    print(f"[INFO] storms used for threshold analysis: {len(df):,}")

    # Per-metric stats + plots
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[WARN] metric not found, skip: {metric}")
            continue

        s = ensure_numeric(df[metric]).dropna()
        if len(s) < args.min_n:
            print(f"[WARN] metric has too few values, skip: {metric} (n={len(s)})")
            continue

        stats = summarize_metric(df[metric])
        stats["metric"] = metric

        thr = stats["knee_threshold"]
        dsel = apply_single_threshold(df, metric, thr)

        stats["selected_count_if_use_knee"] = int(len(dsel))
        stats["selected_fraction_if_use_knee"] = float(len(dsel) / len(df))

        summary_rows.append(stats)

        print(f"\n=== {metric} ===")
        for k in [
            "count", "min", "p10", "p25", "median", "p75", "p90", "max",
            "mean", "std", "knee_threshold", "knee_index",
            "selected_count_if_use_knee", "selected_fraction_if_use_knee",
        ]:
            print(f"{k:30s}: {stats[k]}")

        plot_sorted_metric(
            series=df[metric],
            metric_name=metric,
            out_png=outdir / f"{metric}_sorted_curve.png",
            log_y=args.log_y_sorted,
        )

        plot_histogram(
            series=df[metric],
            metric_name=metric,
            out_png=outdir / f"{metric}_histogram.png",
            bins=args.hist_bins,
            log_x=args.log_x_hist,
        )

        if "storm_id" in df.columns:
            dsel_out = dsel.copy().sort_values(metric, ascending=False).reset_index(drop=True)
            out_sel = outdir / f"{metric}_selected_by_knee.parquet"
            dsel_out.to_parquet(out_sel, index=False)

    if not summary_rows:
        raise RuntimeError("No analyzable metrics found.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            "metric",
            "count",
            "min",
            "p10",
            "p25",
            "median",
            "p75",
            "p90",
            "max",
            "mean",
            "std",
            "knee_threshold",
            "knee_index",
            "n_at_or_above_knee",
            "fraction_at_or_above_knee",
            "selected_count_if_use_knee",
            "selected_fraction_if_use_knee",
        ]
    ].sort_values("metric").reset_index(drop=True)

    summary_csv = outdir / "threshold_summary.csv"
    summary_parquet = outdir / "threshold_summary.parquet"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_parquet(summary_parquet, index=False)

    print(f"\n[SAVED] {summary_csv}")
    print(f"[SAVED] {summary_parquet}")

    # Combined boxplot
    plot_boxplot(df, args.metrics, outdir / "metrics_boxplot.png")

    # Scatter plots
    for xcol, ycol in DEFAULT_SCATTER_PAIRS:
        if xcol in args.metrics and ycol in args.metrics:
            plot_scatter_pair(df, xcol, ycol, outdir / f"scatter_{xcol}_vs_{ycol}.png")

    # Combined threshold selection
    available = [m for m in DEFAULT_METRICS if m in summary_df["metric"].tolist()]
    if available:
        thr_map = {
            row["metric"]: row["knee_threshold"]
            for _, row in summary_df.iterrows()
        }

        dcombo = df.copy()
        for m in available:
            dcombo[m] = ensure_numeric(dcombo[m])
            dcombo = dcombo[dcombo[m] >= thr_map[m]].copy()

        combo_csv = outdir / "selected_by_all_knee_thresholds.csv"
        combo_parquet = outdir / "selected_by_all_knee_thresholds.parquet"

        if not dcombo.empty:
            dcombo.to_csv(combo_csv, index=False)
            dcombo.to_parquet(combo_parquet, index=False)

        print("\n=== combined knee selection ===")
        print(f"metrics used: {available}")
        print(f"storms selected: {len(dcombo)} / {len(df)}")
        print(f"fraction selected: {len(dcombo)/len(df):.4f}")

        if not dcombo.empty:
            print("\nTop selected storms:")
            cols = [c for c in ["storm_id", "storm_start", "storm_end"] + available if c in dcombo.columns]
            print(dcombo[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
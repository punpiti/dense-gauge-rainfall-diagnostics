#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classify_convective_from_residual.py

Rule-based convective classification from optical-flow residual output.

Input
-----
Zarr/NetCDF produced by compute_optical_flow_xarray.py, containing:
- residual(time,y,x)   = R_next - R_adv
- R_adv(time,y,x)
Optionally:
- rainfall(time,y,x) from a separate grid zarr/netcdf can also be provided.

Main idea
---------
Positive residual indicates rainfall growth not explained by advection alone.
This script summarizes positive-residual behavior over an event and assigns:

- convective
- non_convective
- ambiguous

Outputs
-------
1) event-level summary parquet/csv
2) optional frame-level summary parquet/csv

Example
-------
python classify_convective_from_residual.py \
  --flow results/event_grids/event_station0_flow_farneb.zarr \
  --grid results/event_grids/event_station0_soft.zarr \
  --out-summary results/convective/event_station0_summary.parquet \
  --out-frames results/convective/event_station0_frames.parquet

Notes
-----
- This is a baseline classifier.
- Thresholds should be tuned after inspecting several events.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--flow", required=True, help="flow zarr/nc from compute_optical_flow_xarray.py")
    ap.add_argument("--grid", default="", help="optional rainfall grid zarr/nc from build_event_grid_xarray.py")

    ap.add_argument("--event-id", default="", help="optional event id to store in outputs")
    ap.add_argument("--source-stationid", default="", help="optional source station id to store in outputs")

    # Thresholds for frame-level signal extraction
    ap.add_argument("--rain-min", type=float, default=0.1,
                    help="minimum observed rainfall to define rainy cell")
    ap.add_argument("--pos-residual-min", type=float, default=0.2,
                    help="minimum positive residual to count as source-like growth")
    ap.add_argument("--strong-pos-residual", type=float, default=0.5,
                    help="strong positive residual threshold")
    ap.add_argument("--min-pos-area-cells", type=int, default=3,
                    help="minimum number of cells with positive residual in a frame")
    ap.add_argument("--min-rainy-area-cells", type=int, default=3,
                    help="minimum rainy cells in a frame for the frame to be meaningful")

    # Event-level classification thresholds
    ap.add_argument("--min-strong-frames", type=int, default=2,
                    help="minimum number of strong-positive frames to classify as convective")
    ap.add_argument("--min-peak-pos-area", type=int, default=5,
                    help="minimum peak positive-residual area (cells) to classify as convective")
    ap.add_argument("--min-peak-pos-sum", type=float, default=2.0,
                    help="minimum peak sum of positive residual to classify as convective")
    ap.add_argument("--min-event-pos-sum", type=float, default=5.0,
                    help="minimum event total positive residual sum to classify as convective")
    ap.add_argument("--max-strong-frames-for-nonconv", type=int, default=0,
                    help="if strong frames <= this and total positive signal is weak, classify as non_convective")
    ap.add_argument("--max-event-pos-sum-for-nonconv", type=float, default=1.0,
                    help="if total positive residual sum <= this and strong frames small, classify as non_convective")

    ap.add_argument("--out-summary", required=True, help="output event summary (.parquet or .csv)")
    ap.add_argument("--out-frames", default="", help="optional frame summary (.parquet or .csv)")

    return ap.parse_args()


def open_xarray_any(path_str: str) -> xr.Dataset:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix == ".zarr":
        return xr.open_zarr(p)
    if suffix in [".nc", ".netcdf"]:
        return xr.open_dataset(p)

    # fallback
    try:
        return xr.open_zarr(p)
    except Exception:
        return xr.open_dataset(p)


def safe_nanmax(a: np.ndarray) -> float:
    finite = np.isfinite(a)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmax(a))


def safe_nansum(a: np.ndarray) -> float:
    finite = np.isfinite(a)
    if not np.any(finite):
        return 0.0
    return float(np.nansum(a))


def safe_count(mask: np.ndarray) -> int:
    return int(np.sum(mask))


def derive_frame_rainfall(
    flow_ds: xr.Dataset,
    grid_ds: Optional[xr.Dataset],
) -> np.ndarray:
    """
    Return rainfall aligned to flow time dimension.

    flow time[k] corresponds to mapping from time[k] -> time[k+1].
    We align rainfall frame as observed rainfall at time[k+1] if grid is available.
    Otherwise fallback to R_adv + residual = R_next.
    """
    if grid_ds is not None and "rainfall" in grid_ds:
        Rg = grid_ds["rainfall"].values.astype(np.float32)
        # grid has T, flow has T-1
        if Rg.shape[0] >= 2:
            return Rg[1:].copy()

    if "R_adv" in flow_ds and "residual" in flow_ds:
        return (flow_ds["R_adv"].values + flow_ds["residual"].values).astype(np.float32)

    raise KeyError("Cannot derive rainfall. Provide --grid or ensure flow dataset has R_adv and residual.")


def classify_event(summary: dict, args: argparse.Namespace) -> str:
    """
    Rule-based event classification.
    """
    n_valid_frames = summary["n_valid_frames"]
    n_strong_frames = summary["n_strong_frames"]
    peak_pos_area = summary["peak_pos_area_cells"]
    peak_pos_sum = summary["peak_pos_sum"]
    event_pos_sum = summary["event_pos_sum"]

    if n_valid_frames == 0:
        return "ambiguous"

    if (
        n_strong_frames >= args.min_strong_frames
        and peak_pos_area >= args.min_peak_pos_area
        and peak_pos_sum >= args.min_peak_pos_sum
        and event_pos_sum >= args.min_event_pos_sum
    ):
        return "convective"

    if (
        n_strong_frames <= args.max_strong_frames_for_nonconv
        and event_pos_sum <= args.max_event_pos_sum_for_nonconv
    ):
        return "non_convective"

    return "ambiguous"


def main() -> None:
    args = parse_args()

    flow_ds = open_xarray_any(args.flow)
    grid_ds = open_xarray_any(args.grid) if args.grid else None

    required = ["residual", "R_adv"]
    missing = [v for v in required if v not in flow_ds]
    if missing:
        raise KeyError(f"Flow dataset missing variables: {missing}")

    residual = flow_ds["residual"].values.astype(np.float32)
    R_adv = flow_ds["R_adv"].values.astype(np.float32)
    rainfall = derive_frame_rainfall(flow_ds, grid_ds).astype(np.float32)

    times = pd.to_datetime(flow_ds["time"].values)
    T, H, W = residual.shape

    frame_rows = []

    for k in range(T):
        res = residual[k]
        radv = R_adv[k]
        robs = rainfall[k]

        # basic masks
        rainy_mask = np.isfinite(robs) & (robs >= args.rain_min)
        pos_mask = np.isfinite(res) & (res >= args.pos_residual_min)
        strong_pos_mask = np.isfinite(res) & (res >= args.strong_pos_residual)

        # positive residual over rainy area only
        pos_over_rainy_mask = rainy_mask & pos_mask
        strong_over_rainy_mask = rainy_mask & strong_pos_mask

        pos_residual = np.where(pos_over_rainy_mask, res, 0.0)
        strong_residual = np.where(strong_over_rainy_mask, res, 0.0)

        rainy_area_cells = safe_count(rainy_mask)
        pos_area_cells = safe_count(pos_over_rainy_mask)
        strong_pos_area_cells = safe_count(strong_over_rainy_mask)

        rainy_sum = safe_nansum(np.where(rainy_mask, robs, 0.0))
        adv_sum = safe_nansum(np.where(rainy_mask, radv, 0.0))
        pos_sum = safe_nansum(pos_residual)
        strong_pos_sum = safe_nansum(strong_residual)

        frame_max_rain = safe_nanmax(np.where(rainy_mask, robs, np.nan))
        frame_max_pos = safe_nanmax(np.where(pos_over_rainy_mask, res, np.nan))

        pos_fraction_of_rainy = (
            float(pos_area_cells) / float(rainy_area_cells)
            if rainy_area_cells > 0 else 0.0
        )

        pos_sum_fraction_of_rain = (
            float(pos_sum) / float(rainy_sum)
            if rainy_sum > 0 else 0.0
        )

        is_valid_frame = rainy_area_cells >= args.min_rainy_area_cells
        is_strong_frame = (
            is_valid_frame
            and strong_pos_area_cells >= args.min_pos_area_cells
            and strong_pos_sum > 0.0
        )

        frame_rows.append(
            {
                "frame_index": k,
                "time": times[k],
                "rainy_area_cells": rainy_area_cells,
                "pos_area_cells": pos_area_cells,
                "strong_pos_area_cells": strong_pos_area_cells,
                "rainy_sum": rainy_sum,
                "adv_sum": adv_sum,
                "pos_sum": pos_sum,
                "strong_pos_sum": strong_pos_sum,
                "frame_max_rain": frame_max_rain,
                "frame_max_pos_residual": frame_max_pos,
                "pos_fraction_of_rainy": pos_fraction_of_rainy,
                "pos_sum_fraction_of_rain": pos_sum_fraction_of_rain,
                "is_valid_frame": bool(is_valid_frame),
                "is_strong_frame": bool(is_strong_frame),
            }
        )

    frames_df = pd.DataFrame(frame_rows)

    valid_df = frames_df[frames_df["is_valid_frame"]].copy()
    strong_df = frames_df[frames_df["is_strong_frame"]].copy()

    summary = {
        "event_id": args.event_id if args.event_id else None,
        "source_stationid": args.source_stationid if args.source_stationid else None,
        "flow_path": str(Path(args.flow)),
        "grid_path": str(Path(args.grid)) if args.grid else None,
        "n_frames": int(len(frames_df)),
        "n_valid_frames": int(len(valid_df)),
        "n_strong_frames": int(len(strong_df)),
        "event_pos_sum": float(frames_df["pos_sum"].sum()),
        "event_strong_pos_sum": float(frames_df["strong_pos_sum"].sum()),
        "peak_pos_sum": float(frames_df["pos_sum"].max()) if len(frames_df) else 0.0,
        "peak_pos_area_cells": int(frames_df["pos_area_cells"].max()) if len(frames_df) else 0,
        "peak_strong_pos_area_cells": int(frames_df["strong_pos_area_cells"].max()) if len(frames_df) else 0,
        "peak_frame_max_pos_residual": float(frames_df["frame_max_pos_residual"].max()) if len(frames_df) else 0.0,
        "max_pos_fraction_of_rainy": float(frames_df["pos_fraction_of_rainy"].max()) if len(frames_df) else 0.0,
        "max_pos_sum_fraction_of_rain": float(frames_df["pos_sum_fraction_of_rain"].max()) if len(frames_df) else 0.0,
        "mean_pos_sum_valid_frames": float(valid_df["pos_sum"].mean()) if len(valid_df) else 0.0,
        "mean_pos_area_valid_frames": float(valid_df["pos_area_cells"].mean()) if len(valid_df) else 0.0,
        "time_start": pd.Timestamp(times.min()) if len(times) else None,
        "time_end": pd.Timestamp(times.max()) if len(times) else None,
    }

    label = classify_event(summary, args)
    summary["convective_label"] = label

    summary_df = pd.DataFrame([summary])

    # save
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    if out_summary.suffix.lower() == ".parquet":
        summary_df.to_parquet(out_summary, index=False)
    else:
        summary_df.to_csv(out_summary, index=False)

    if args.out_frames:
        out_frames = Path(args.out_frames)
        out_frames.parent.mkdir(parents=True, exist_ok=True)
        if out_frames.suffix.lower() == ".parquet":
            frames_df.to_parquet(out_frames, index=False)
        else:
            frames_df.to_csv(out_frames, index=False)

    print("[OK] classification complete")
    print(summary_df.T)

    if args.out_frames:
        print(f"[SAVED] frame summary : {args.out_frames}")
    print(f"[SAVED] event summary : {args.out_summary}")


if __name__ == "__main__":
    main()
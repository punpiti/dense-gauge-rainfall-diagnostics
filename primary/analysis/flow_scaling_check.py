#!/usr/bin/env python3
"""Optical-flow input-scaling check for the retained Bangkok storm sample.

Question: does the archived fixed 0--15 mm uint8 conversion make Farneback
under-estimate motion for weak rainfall, so that the optical-flow residual
collapses towards the no-advection difference?

Stages (run in order; each writes to flow_scaling_check/ and never touches
the archived products or the manuscript):

  0  Synthetic Gaussian control: known shift, no growth, weak and strong peak.
  A  Archived flows only: per-storm flow speed against rainfall intensity.
  B  Known-motion control on real fields: shift real frames by a known
     displacement (no growth) and measure motion recovery per scaling.
  C  All 157 storms recomputed with each scaling; the fixed15_trunc run must
     reproduce the archived residuals before any other variant is read.

Only the image passed to Farneback changes.  Flow parameters, masks, warping,
residual thresholds and the final Boolean rule are those of the retained
pipeline.  Warping and residuals always use rainfall in mm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from no_advection_ablation import (
    METRICS,
    PRODUCTS as DEFAULT_PRODUCTS,
    operational_label,
    read_zarr_v3_array,
    summarize,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "flow_scaling_check"
RAIN_THR = 0.1
CLIP_MAX = 15.0
SHIFTS = [(4, 2), (-3, 5), (6, -1)]  # (dx, dy) cells per 15 min; 1 cell = 1 km

_spec = importlib.util.spec_from_file_location(
    "flow", HERE.parent / "processing" / "compute_optical_flow_xarray.py"
)
flow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flow)


# ---------------------------------------------------------------- scalings
def _to_u8(a: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(a, 0.0, 255.0)).astype(np.uint8)


def scale_pair(rt: np.ndarray, rn: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the two uint8 images for one scaling; both use one transform."""
    if name == "fixed15_trunc":  # archived behaviour, kept bit-identical
        return flow.normalize_to_uint8(rt, CLIP_MAX), flow.normalize_to_uint8(rn, CLIP_MAX)
    rt = np.where(np.isfinite(rt), rt, 0.0)
    rn = np.where(np.isfinite(rn), rn, 0.0)
    if name == "fixed15_round":  # isolates the truncation effect
        f = lambda r: _to_u8(np.clip(r, 0.0, CLIP_MAX) / CLIP_MAX * 255.0)
    elif name == "log1p_fixed":  # fixed across storms, compresses the range
        f = lambda r: _to_u8(np.log1p(np.clip(r, 0.0, CLIP_MAX)) / np.log1p(CLIP_MAX) * 255.0)
    elif name == "pair_max":  # shared stretch for the pair only
        m = max(float(rt.max()), float(rn.max()), RAIN_THR)
        f = lambda r: _to_u8(np.clip(r, 0.0, m) / m * 255.0)
    else:
        raise ValueError(name)
    return f(rt), f(rn)


SCALINGS = ["fixed15_trunc", "fixed15_round", "log1p_fixed", "pair_max"]


def flow_and_residual(rt: np.ndarray, rn: np.ndarray, scaling: str):
    """Replicates compute_optical_flow_xarray.main for one pair."""
    rt = np.where(np.isfinite(rt), rt, 0.0).astype(np.float32)
    rn = np.where(np.isfinite(rn), rn, 0.0).astype(np.float32)
    mask = (rt >= RAIN_THR) | (rn >= RAIN_THR)
    p8, n8 = scale_pair(rt, rn, scaling)
    d = flow.compute_flow_farneback(flow.apply_mask_uint8(p8, mask), flow.apply_mask_uint8(n8, mask))
    d[~mask] = 0.0
    adv = flow.advect_forward_backward_warp(rt, d)
    return d, adv, (rn - adv).astype(np.float32), mask


def shift_field(f: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """G(y, x) = F(y - dy, x - dx) with zero fill."""
    g = np.zeros_like(f)
    H, W = f.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    g[yd, xd] = f[ys, xs]
    return g


def motion_recovery(f: np.ndarray, dx: int, dy: int, scaling: str) -> dict:
    g = shift_field(f, dx, dy)
    d, adv, _, _ = flow_and_residual(f, g, scaling)
    sel = g >= RAIN_THR
    if sel.sum() < 3:
        return {}
    ex, ey = d[..., 0][sel], d[..., 1][sel]
    epe = np.hypot(ex - dx, ey - dy)
    noadv = np.mean(np.abs(g - f)[sel | (f >= RAIN_THR)])
    union = sel | (f >= RAIN_THR)
    return {
        "est_dx_median": float(np.median(ex)),
        "est_dy_median": float(np.median(ey)),
        "epe_median": float(np.median(epe)),
        "speed_ratio": float(np.median(np.hypot(ex, ey)) / np.hypot(dx, dy)),
        "residual_mae_vs_noadv": float(np.mean(np.abs(g - adv)[union]) / noadv) if noadv > 0 else np.nan,
    }


# ---------------------------------------------------------------- stages
def stage0() -> pd.DataFrame:
    H, W = 57, 70
    yy, xx = np.mgrid[0:H, 0:W]
    rows = []
    for peak in [0.5, 1.0, 3.0, 10.0]:
        f = (peak * np.exp(-((xx - 30) ** 2 + (yy - 26) ** 2) / (2 * 4.0 ** 2))).astype(np.float32)
        for dx, dy in SHIFTS:
            for s in SCALINGS:
                rows.append({"peak_mm": peak, "dx": dx, "dy": dy, "scaling": s, **motion_recovery(f, dx, dy, s)})
    return pd.DataFrame(rows)


def completed_storms(products: Path) -> pd.DataFrame:
    src = pd.read_csv(products / "all_storm_results.csv")
    src = src.loc[src["status"].eq("done")].copy()
    if len(src) != 157:
        raise ValueError(f"Expected 157 completed storms, found {len(src)}")
    return src


def intensity_stats(r: np.ndarray) -> dict:
    rainy = r[np.isfinite(r) & (r >= RAIN_THR)]
    if rainy.size == 0:
        return {"rain_mean": 0.0, "rain_p90": 0.0, "rain_max": 0.0}
    return {"rain_mean": float(rainy.mean()), "rain_p90": float(np.percentile(rainy, 90)), "rain_max": float(rainy.max())}


def stageA(products: Path, storms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in storms["storm_id"]:
        rain = read_zarr_v3_array(products / "grids" / f"{sid}.zarr" / "rainfall")
        speed = read_zarr_v3_array(products / "flows" / f"{sid}.zarr" / "speed")
        r = np.where(np.isfinite(rain), rain, 0.0)
        mask = (r[:-1] >= RAIN_THR) | (r[1:] >= RAIN_THR)
        sp = speed[mask & np.isfinite(speed)]
        rows.append({
            "storm_id": sid,
            "speed_median_ms": float(np.median(sp)) if sp.size else np.nan,
            "speed_p90_ms": float(np.percentile(sp, 90)) if sp.size else np.nan,
            "frac_cells_below_0p5ms": float(np.mean(sp < 0.5)) if sp.size else np.nan,
            "rainy_cells_mean": float(mask.sum(axis=(1, 2)).mean()),
            **intensity_stats(rain),
        })
    return pd.DataFrame(rows)


def pick_frames(rain: np.ndarray, n: int = 3, min_cells: int = 30) -> list[int]:
    area = (np.nan_to_num(rain) >= RAIN_THR).sum(axis=(1, 2))
    ok = np.flatnonzero(area >= min_cells)
    if ok.size == 0:
        return []
    return sorted(set(ok[np.linspace(0, ok.size - 1, min(n, ok.size)).round().astype(int)].tolist()))


def stageB(products: Path, storms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in storms["storm_id"]:
        rain = read_zarr_v3_array(products / "grids" / f"{sid}.zarr" / "rainfall")
        for k in pick_frames(rain):
            f = np.where(np.isfinite(rain[k]), rain[k], 0.0).astype(np.float32)
            base = {"storm_id": sid, "frame": k, **intensity_stats(rain[k])}
            for dx, dy in SHIFTS:
                for s in SCALINGS:
                    rows.append({**base, "dx": dx, "dy": dy, "scaling": s, **motion_recovery(f, dx, dy, s)})
    return pd.DataFrame(rows)


def stageC(products: Path, storms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, repro = [], []
    for sid in storms["storm_id"]:
        rain = read_zarr_v3_array(products / "grids" / f"{sid}.zarr" / "rainfall")
        archived = read_zarr_v3_array(products / "flows" / f"{sid}.zarr" / "residual")
        r0 = np.where(np.isfinite(rain), rain, 0.0).astype(np.float32)
        observed = rain[1:]
        out = {"storm_id": sid, **intensity_stats(rain)}
        noadv = summarize(r0[1:] - r0[:-1], observed)
        out.update({f"no_advection_{m}": noadv[m] for m in METRICS})
        out["no_advection_label"] = operational_label(noadv)
        for s in SCALINGS:
            res, speeds, q = [], [], []
            for k in range(rain.shape[0] - 1):
                d, adv, rr, mask = flow_and_residual(rain[k], rain[k + 1], s)
                res.append(rr)
                if mask.any():
                    speeds.append(np.hypot(d[..., 0], d[..., 1])[mask])
                    base_mae = np.mean(np.abs(r0[k + 1] - r0[k])[mask])
                    if base_mae > 0:
                        q.append(1.0 - np.mean(np.abs(r0[k + 1] - adv)[mask]) / base_mae)
            res = np.stack(res)
            if s == "fixed15_trunc":
                repro.append({"storm_id": sid, "max_abs_diff_vs_archived": float(np.nanmax(np.abs(res - archived)))})
            summ = summarize(res, observed)
            out.update({f"{s}_{m}": summ[m] for m in METRICS})
            out[f"{s}_label"] = operational_label(summ)
            sp = np.concatenate(speeds) if speeds else np.array([np.nan])
            out[f"{s}_speed_median_cells"] = float(np.nanmedian(sp))
            out[f"{s}_Q_median"] = float(np.median(q)) if q else np.nan
        rows.append(out)
    return pd.DataFrame(rows), pd.DataFrame(repro)


def summarize_C(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SCALINGS:
        row = {"scaling": s,
               "speed_median_cells": float(df[f"{s}_speed_median_cells"].median()),
               "Q_median": float(df[f"{s}_Q_median"].median()),
               "n_high_residual": int(df[f"{s}_label"].eq("high_residual").sum()),
               "labels_changed_vs_archived": int((df[f"{s}_label"] != df["fixed15_trunc_label"]).sum()),
               "labels_equal_to_no_advection": int((df[f"{s}_label"] == df["no_advection_label"]).sum())}
        for m in METRICS:
            row[f"spearman_{m}_vs_noadv"] = float(df[f"{s}_{m}"].corr(df[f"no_advection_{m}"], method="spearman"))
        row["spearman_Q_vs_rain_p90"] = float(df[f"{s}_Q_median"].corr(df["rain_p90"], method="spearman"))
        rows.append(row)
    return pd.DataFrame(rows)


def _storm_task(task):
    stage, products, sid = task
    # Parallelize storms, with one OpenCV thread per process to avoid nested
    # thread pools. The numerical parameters and operations remain unchanged.
    flow.cv2.setNumThreads(1)
    storm = pd.DataFrame({"storm_id": [sid]})
    return {"A": stageA, "B": stageB, "C": stageC}[stage](products, storm)


def run_stage(stage: str, products: Path, storms: pd.DataFrame, workers: int):
    if workers == 1:
        return {"A": stageA, "B": stageB, "C": stageC}[stage](products, storms)
    tasks = [(stage, products, sid) for sid in storms["storm_id"]]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        results = list(tqdm(executor.map(_storm_task, tasks), total=len(tasks),
                            desc=f"Stage {stage}: storms ({workers} workers)"))
    # map preserves storm order, making serial/parallel outputs comparable.
    if stage == "C":
        return tuple(pd.concat([result[i] for result in results], ignore_index=True)
                     for i in range(2))
    return pd.concat(results, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "A", "B", "C", "all"], default="all")
    ap.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    ap.add_argument("--limit", type=int, default=0, help="first N storms only (smoke test)")
    ap.add_argument("--workers", type=int, default=1, help="parallel storm processes; OpenCV uses one thread per worker")
    ap.add_argument("--output", type=Path, default=OUTPUT, help="separate output directory for verification runs")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    stages = ["0", "A", "B", "C"] if args.stage == "all" else [args.stage]

    if "0" in stages:
        s0 = stage0()
        s0.to_csv(output / "stage0_gaussian_control.csv", index=False)
        print("Stage 0 (median over shifts)\n",
              s0.groupby(["peak_mm", "scaling"])[["speed_ratio", "epe_median", "residual_mae_vs_noadv"]].median().round(3))
    if not set(stages) & {"A", "B", "C"}:
        return

    storms = completed_storms(args.products)
    if args.limit:
        storms = storms.head(args.limit)
    if "A" in stages:
        a = run_stage("A", args.products, storms, args.workers)
        a.to_csv(output / "stageA_archived_speed.csv", index=False)
        print("Stage A Spearman (speed_median_ms vs intensity):",
              {c: round(a["speed_median_ms"].corr(a[c], method="spearman"), 3) for c in ["rain_mean", "rain_p90", "rain_max"]})
        print(a[["speed_median_ms", "frac_cells_below_0p5ms", "rain_p90"]].describe().round(3))
    if "B" in stages:
        b = run_stage("B", args.products, storms, args.workers)
        b.to_csv(output / "stageB_known_motion.csv", index=False)
        b["p90_bin"] = pd.cut(b["rain_p90"], [0, 0.5, 1, 2, 5, np.inf])
        print("Stage B median speed_ratio (1 = exact) by frame P90 intensity\n",
              b.pivot_table(index="p90_bin", columns="scaling", values="speed_ratio", aggfunc="median", observed=True).round(3))
        print(b.pivot_table(index="p90_bin", columns="scaling", values="epe_median", aggfunc="median", observed=True).round(3))
    if "C" in stages:
        c, repro = run_stage("C", args.products, storms, args.workers)
        repro.to_csv(output / "stageC_reproduction_check.csv", index=False)
        worst = float(repro["max_abs_diff_vs_archived"].max())
        print(f"Stage C reproduction: max |fixed15_trunc - archived residual| = {worst:.3g}")
        if worst > 1e-5:
            raise RuntimeError("Archived residuals not reproduced; stop and inspect before reading variants.")
        c.to_csv(output / "stageC_storm_level.csv", index=False)
        summ = summarize_C(c)
        summ.to_csv(output / "stageC_summary.csv", index=False)
        print(summ.T.to_string())
    (output / "run_info.json").write_text(json.dumps(
        {"stages": stages, "products": str(args.products), "limit": args.limit,
         "workers": args.workers, "scalings": SCALINGS, "shifts": SHIFTS,
         "opencv": flow.cv2.__version__}, indent=2) + "\n")


if __name__ == "__main__":
    main()

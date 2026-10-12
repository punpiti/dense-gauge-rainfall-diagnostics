#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_optical_flow_xarray.py

Compute optical flow (u,v) between consecutive rainfall grids in an xarray Dataset,
advect rainfall forward, and save outputs to Zarr/NetCDF.

Inputs:
  - Zarr (from build_event_grid_xarray.py): variable 'rainfall' dims (time,y,x)
    coords x,y are UTM meters (EPSG:32647) and time is datetime64[ns].

Outputs:
  - Zarr/NetCDF with:
      u(time-1,y,x) [m/s]
      v(time-1,y,x) [m/s]
      speed(time-1,y,x) [m/s]
      R_adv(time-1,y,x) [mm/15min]  -> advected estimate for time t+1
      residual(time-1,y,x) [mm/15min] -> R_next - R_adv
    where index k corresponds to mapping from time[k] -> time[k+1].

Usage:
  python compute_optical_flow_xarray.py \
    --in results/event_grids/event0_soft.zarr \
    --out results/event_grids/event0_flow_farneb.zarr \
    --method farneback \
    --rain-threshold 0.1 \
    --dt-min 15 \
    --overwrite

Notes:
  - Uses OpenCV (cv2). For TV-L1, opencv-contrib might be needed.
  - Flow computed in "pixel per frame" then scaled to m/s using grid spacing and dt.
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm

# OpenCV is required
import cv2


def dt_seconds_from_timecoord(times: np.ndarray, fallback_dt_min: float) -> float:
    if len(times) >= 2:
        dt = (np.datetime64(times[1]) - np.datetime64(times[0])) / np.timedelta64(1, "s")
        dt = float(dt)
        if dt > 0:
            return dt
    return float(fallback_dt_min) * 60.0


def grid_spacing_m(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # x,y are centers in meters
    if len(x) >= 2:
        dx = float(np.abs(x[1] - x[0]))
    else:
        dx = 1000.0
    if len(y) >= 2:
        dy = float(np.abs(y[1] - y[0]))
    else:
        dy = 1000.0
    return dx, dy


def normalize_to_uint8(img: np.ndarray, clip_max: float) -> np.ndarray:
    """
    Convert rainfall field to 8-bit for optical flow.
    - clip [0, clip_max]
    - scale to [0,255]
    """
    a = img.copy()
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a, 0.0, float(clip_max))
    if clip_max <= 0:
        clip_max = 1.0
    a = (a / float(clip_max) * 255.0).astype(np.uint8)
    return a


def apply_mask_uint8(img8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img8.copy()
    out[~mask] = 0
    return out


def compute_flow_farneback(prev8: np.ndarray, next8: np.ndarray,
                           pyr_scale=0.5, levels=3, winsize=25,
                           iterations=3, poly_n=5, poly_sigma=1.2, flags=0) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev8,
        next=next8,
        flow=None,
        pyr_scale=float(pyr_scale),
        levels=int(levels),
        winsize=int(winsize),
        iterations=int(iterations),
        poly_n=int(poly_n),
        poly_sigma=float(poly_sigma),
        flags=int(flags),
    )
    return flow  # shape (H,W,2), in pixels per frame (dx, dy)


def compute_flow_tvl1(prev8: np.ndarray, next8: np.ndarray,
                      tau=0.25, lambda_=0.15, theta=0.3, nscales=5,
                      warps=5, epsilon=0.01, iterations=300, scale_step=0.8, gamma=0.0) -> np.ndarray:
    """
    Requires cv2.optflow.DualTVL1OpticalFlow_create (opencv-contrib).
    """
    if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        raise RuntimeError("TV-L1 not available. Install opencv-contrib-python or build with contrib.")

    alg = cv2.optflow.DualTVL1OpticalFlow_create()
    alg.setTau(float(tau))
    alg.setLambda(float(lambda_))
    alg.setTheta(float(theta))
    alg.setScalesNumber(int(nscales))
    alg.setWarpingsNumber(int(warps))
    alg.setEpsilon(float(epsilon))
    alg.setInnerIterations(int(iterations))
    alg.setScaleStep(float(scale_step))
    alg.setGamma(float(gamma))

    flow = alg.calc(prev8, next8, None)
    return flow


def advect_forward_backward_warp(Rt: np.ndarray, flow_px: np.ndarray) -> np.ndarray:
    """
    Advect field Rt to t+1 using flow estimated from Rt->Rnext.
    Use backward warping: R_adv(x) = Rt(x - flow(x))
    """
    H, W = Rt.shape
    # Create base grid
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # Backward map
    map_x = xs - flow_px[..., 0].astype(np.float32)
    map_y = ys - flow_px[..., 1].astype(np.float32)

    a = Rt.copy()
    a[~np.isfinite(a)] = 0.0
    adv = cv2.remap(
        a.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return adv.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="inp", required=True, help="input .zarr or .nc (from build_event_grid_xarray.py)")
    ap.add_argument("--out", required=True, help="output .zarr or .nc")

    ap.add_argument("--method", default="farneback", choices=["farneback", "tvl1"])
    ap.add_argument("--dt-min", type=float, default=15.0, help="fallback dt in minutes if timecoord missing")
    ap.add_argument("--rain-threshold", type=float, default=0.1,
                    help="mask: consider signal where rainfall>=threshold (for flow stability)")
    ap.add_argument("--clip-max", type=float, default=15.0,
                    help="clip rainfall to this max before converting to 8-bit for flow")

    # Farneback params
    ap.add_argument("--fb-pyr-scale", type=float, default=0.5)
    ap.add_argument("--fb-levels", type=int, default=3)
    ap.add_argument("--fb-winsize", type=int, default=25)
    ap.add_argument("--fb-iterations", type=int, default=3)
    ap.add_argument("--fb-poly-n", type=int, default=5)
    ap.add_argument("--fb-poly-sigma", type=float, default=1.2)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chunks-time", type=int, default=1)

    args = ap.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.out)
    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")
        # remove zarr dir or file
        if out_path.is_dir():
            for p in sorted(out_path.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            out_path.rmdir()
        else:
            out_path.unlink()

    if in_path.suffix.lower() == ".zarr":
        ds = xr.open_zarr(in_path)
    elif in_path.suffix.lower() in [".nc", ".netcdf"]:
        ds = xr.open_dataset(in_path)
    else:
        raise ValueError("Unknown input suffix. Use .zarr or .nc")
    if "rainfall" not in ds:
        raise KeyError("Input dataset must contain variable 'rainfall'")

    R = ds["rainfall"]
    if not set(R.dims) == {"time", "y", "x"}:
        raise ValueError(f"rainfall dims must be (time,y,x), got {R.dims}")

    times = R["time"].values
    x = R["x"].values
    y = R["y"].values

    dt_s = dt_seconds_from_timecoord(times, args.dt_min)
    dx_m, dy_m = grid_spacing_m(x, y)

    # Flow will be computed on array index space:
    # flow_px[...,0] is pixels in x, flow_px[...,1] pixels in y per frame
    # Convert to m/s:
    # u = flow_px_x * dx_m / dt_s
    # v = flow_px_y * dy_m / dt_s
    print(f"[INFO] dt={dt_s:.1f}s | dx={dx_m:.1f}m dy={dy_m:.1f}m | method={args.method}")

    T = len(times)
    if T < 2:
        raise RuntimeError("Need at least 2 timestamps to compute flow.")

    H = R.sizes["y"]
    W = R.sizes["x"]

    u = np.zeros((T - 1, H, W), dtype=np.float32)
    v = np.zeros((T - 1, H, W), dtype=np.float32)
    speed = np.zeros((T - 1, H, W), dtype=np.float32)
    R_adv = np.zeros((T - 1, H, W), dtype=np.float32)
    residual = np.zeros((T - 1, H, W), dtype=np.float32)

    # Load into memory (small event); faster than repeated reads
    Rv = R.values.astype(np.float32)
    Rv[~np.isfinite(Rv)] = 0.0

    thr = float(args.rain_threshold)
    clip_max = float(args.clip_max)

    for k in tqdm(range(T - 1), desc="Compute optical flow"):
        Rt = Rv[k]
        Rn = Rv[k + 1]

        mask = (Rt >= thr) | (Rn >= thr)

        prev8 = normalize_to_uint8(Rt, clip_max=clip_max)
        next8 = normalize_to_uint8(Rn, clip_max=clip_max)

        # Mask weak-signal areas to stabilize flow
        prev8m = apply_mask_uint8(prev8, mask)
        next8m = apply_mask_uint8(next8, mask)

        if args.method == "farneback":
            flow_px = compute_flow_farneback(
                prev8m, next8m,
                pyr_scale=args.fb_pyr_scale,
                levels=args.fb_levels,
                winsize=args.fb_winsize,
                iterations=args.fb_iterations,
                poly_n=args.fb_poly_n,
                poly_sigma=args.fb_poly_sigma,
            )
        else:
            flow_px = compute_flow_tvl1(prev8m, next8m)

        # Zero flow where no signal at all
        flow_px[~mask] = 0.0

        # Advect Rt -> t+1
        adv = advect_forward_backward_warp(Rt, flow_px)

        # Convert to m/s
        uu = (flow_px[..., 0].astype(np.float32) * float(dx_m) / float(dt_s))
        vv = (flow_px[..., 1].astype(np.float32) * float(dy_m) / float(dt_s))

        u[k] = uu
        v[k] = vv
        speed[k] = np.sqrt(uu * uu + vv * vv).astype(np.float32)
        R_adv[k] = adv
        residual[k] = (Rn - adv).astype(np.float32)

    # Build output xarray Dataset
    t_out = ds["time"].values[:-1]  # time index for flow mapping time[k]->time[k+1]

    ds_out = xr.Dataset(
        data_vars={
            "u": (("time", "y", "x"), u, {"units": "m/s", "desc": "eastward component of optical flow"}),
            "v": (("time", "y", "x"), v, {"units": "m/s", "desc": "northward component of optical flow"}),
            "speed": (("time", "y", "x"), speed, {"units": "m/s", "desc": "flow speed"}),
            "R_adv": (("time", "y", "x"), R_adv, {"units": "mm/15min", "desc": "advected rainfall estimate for t+1"}),
            "residual": (("time", "y", "x"), residual, {"units": "mm/15min", "desc": "R_next - R_adv"}),
        },
        coords={
            "time": t_out,
            "y": y,
            "x": x,
        },
        attrs={
            "source_input": str(in_path),
            "method": str(args.method),
            "rain_threshold": float(args.rain_threshold),
            "clip_max": float(args.clip_max),
            "dt_seconds": float(dt_s),
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "note": "Variables at time[k] correspond to mapping from time[k] to time[k+1].",
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".zarr":
        chunks = {"time": int(args.chunks_time), "y": min(128, H), "x": min(128, W)}
        ds_out.chunk(chunks).to_zarr(out_path, mode="w")
        print(f"[OK] saved zarr: {out_path}")
    elif suffix in [".nc", ".netcdf"]:
        ds_out.to_netcdf(out_path)
        print(f"[OK] saved netcdf: {out_path}")
    else:
        raise ValueError("Unknown output suffix. Use .zarr or .nc")

    print(ds_out)


if __name__ == "__main__":
    main()

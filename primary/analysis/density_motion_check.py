#!/usr/bin/env python3
"""Local gauge support versus known-translation recovery on retained fields.

This is an association test on existing reconstructions, not a thinning
experiment or independent validation of real rainfall motion. Density is
attached to the source location of each translated rainfall cell.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Zstd
from pyproj import Transformer
from scipy.spatial.distance import cdist
from shapely.geometry import shape
from shapely.ops import transform, unary_union
from shapely import contains_xy
from tqdm import tqdm

from flow_scaling_check import (
    DEFAULT_PRODUCTS, SHIFTS, completed_storms, flow, flow_and_residual,
    pick_frames, read_zarr_v3_array, shift_field,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / "inputs"
SCALES = ["fixed15_trunc", "log1p_fixed", "pair_max"]
RAIN_EDGES = np.array([.1, .25, .5, 1., 2., 4., np.inf])
RAIN_LABELS = ["[0.1,0.25)", "[0.25,0.5)", "[0.5,1)", "[1,2)", "[2,4)", "[4,inf)"]
DENSITY_LABELS = ["sparse", "middle", "dense"]
CONFIG = None


def read_coordinate(path):
    meta = json.loads((path / "zarr.json").read_text())
    assert len(meta["shape"]) == 1
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == meta["shape"]
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "zstd"]
    assert meta["codecs"][0]["configuration"]["endian"] == "little"
    dtype = {"float64": "<f8", "int64": "<i8"}[meta["data_type"]]
    vals = np.frombuffer(Zstd().decode((path / "c" / "0").read_bytes()), dtype=dtype).copy()
    assert len(vals) == meta["shape"][0]
    if path.name == "time":
        units, origin = meta["attributes"]["units"].split(" since ", 1)
        unit = {"minutes": "min", "seconds": "s", "hours": "h", "days": "D",
                "milliseconds": "ms", "microseconds": "us", "nanoseconds": "ns"}[units]
        return pd.DatetimeIndex(pd.Timestamp(origin) + pd.to_timedelta(vals, unit=unit))
    return vals


def support_maps(cfg, available):
    indices = np.flatnonzero(available)
    near25 = cfg["distances"][:, indices] <= 2500
    near5 = cfg["distances"][:, indices] <= 5000
    count25, count5 = near25.sum(axis=1), near5.sum(axis=1)
    nearest = (cfg["distances"][:, indices].min(axis=1) / 1000
               if len(indices) else np.full(len(count25), np.nan))
    # Nearest-neighbour distance of each available station to the entire
    # available network; local moments use stations within 5 km of the cell.
    nn = np.full(len(indices), np.nan)
    if len(indices) >= 2:
        d = cfg["station_distances"][np.ix_(indices, indices)].copy()
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1) / 1000
    mean = np.full(len(count5), np.nan)
    variance = np.full(len(count5), np.nan)
    cv = np.full(len(count5), np.nan)
    if len(indices) >= 2:
        np.divide(near5 @ nn, count5, out=mean, where=count5 > 0)
        second = np.full(len(count5), np.nan)
        np.divide(near5 @ (nn ** 2), count5, out=second, where=count5 > 0)
        eligible = count5 >= 3
        variance[eligible] = np.maximum(0, second[eligible] - mean[eligible] ** 2)
        np.divide(np.sqrt(variance), mean, out=cv, where=eligible & (mean > 0))
    return {"count25": count25, "count5": count5, "nearest_km": nearest,
            "nn_mean_km": mean, "nn_variance_km2": variance, "nn_cv": cv}


def setup_geometry(products, first_sid, stations, boundary, border):
    st = pd.read_csv(stations)
    for key in ("id", "lat", "lon"):
        st[key] = pd.to_numeric(st[key], errors="coerce")
    st = st.dropna(subset=["id", "lat", "lon"])
    assert len(st) == st.id.nunique() == 130
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    sx, sy = tf.transform(st.lon.to_numpy(), st.lat.to_numpy())
    sxy = np.column_stack([sx, sy])
    gj = json.loads(boundary.read_text())
    polygon = unary_union([transform(tf.transform, shape(f["geometry"])) for f in gj["features"]])
    grid = products / "grids" / f"{first_sid}.zarr"
    x, y = read_coordinate(grid / "x"), read_coordinate(grid / "y")
    assert np.allclose(np.diff(x), 1000) and np.allclose(np.diff(y), 1000)
    xx, yy = np.meshgrid(x, y)
    iy, ix = np.indices(xx.shape)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    cfg = {"x": x, "y": y, "shape": xx.shape, "iy": iy, "ix": ix,
           "ids": st.id.astype(int).to_numpy(), "sxy": sxy,
           "distances": cdist(points, sxy), "station_distances": cdist(sxy, sxy),
           "inside": contains_xy(polygon, xx, yy), "border": border,
           "products": products}
    cfg["nominal"] = support_maps(cfg, np.ones(len(st), dtype=bool))
    table = pd.DataFrame({"iy": iy.ravel(), "ix": ix.ravel(),
                          "x_utm_m": xx.ravel(), "y_utm_m": yy.ravel(),
                          "inside_bangkok": cfg["inside"].ravel(), **cfg["nominal"]})
    return cfg, table


def init_worker(cfg):
    global CONFIG
    CONFIG = cfg
    flow.cv2.setNumThreads(1)


def origin_selection(cfg, f, dx, dy):
    iy, ix = cfg["iy"], cfg["ix"]
    height, width = cfg["shape"]
    margin = cfg["border"]
    valid = cfg["inside"] & np.isfinite(f) & (f >= .1)
    valid &= (iy >= margin) & (iy < height - margin) & (ix >= margin) & (ix < width - margin)
    valid &= (iy + dy >= margin) & (iy + dy < height - margin)
    valid &= (ix + dx >= margin) & (ix + dx < width - margin)
    return np.flatnonzero(valid.ravel())


def worker(sid):
    cfg = CONFIG
    grid = cfg["products"] / "grids" / f"{sid}.zarr"
    np.testing.assert_array_equal(read_coordinate(grid / "x"), cfg["x"])
    np.testing.assert_array_equal(read_coordinate(grid / "y"), cfg["y"])
    rain = read_zarr_v3_array(grid / "rainfall")
    times = read_coordinate(grid / "time")
    assert rain.shape[1:] == cfg["shape"] and len(rain) == len(times)
    attrs = json.loads((grid / "rainfall" / "zarr.json").read_text())["attributes"]
    assert attrs["anchor_stations"] and attrs["sigma_km"] == 2.5
    cache = cfg["products"] / "cache" / Path(attrs["source_cache"]).name
    obs = pd.read_parquet(cache, columns=["time", "stationid", "rainfall"])
    obs["time"] = pd.to_datetime(obs.time)
    obs["stationid"] = pd.to_numeric(obs.stationid.astype(str).str.replace(r"^[Ss]", "", regex=True), errors="coerce")
    obs["rainfall"] = pd.to_numeric(obs.rainfall, errors="coerce")
    np.testing.assert_array_equal(np.sort(obs.time.unique()), times.to_numpy())
    rows, frame_rows = [], []
    for k in pick_frames(rain):
        f = np.where(np.isfinite(rain[k]), rain[k], 0).astype(np.float32)
        current = obs.loc[obs.time.eq(times[k]) & np.isfinite(obs.rainfall)]
        available = np.isin(cfg["ids"], current.stationid.unique())
        live = support_maps(cfg, available)
        frame_rows.append({"storm_id": sid, "frame": k, "time": str(times[k]),
                           "available_geolocated_stations": int(available.sum())})
        gradient = np.hypot(*np.gradient(f)).ravel()
        for dx, dy in SHIFTS:
            selected = origin_selection(cfg, f, dx, dy)
            if not len(selected):
                continue
            iy = cfg["iy"].ravel()[selected]
            ix = cfg["ix"].ravel()[selected]
            dest_y, dest_x = iy + dy, ix + dx
            g = shift_field(f, dx, dy)
            np.testing.assert_array_equal(g[dest_y, dest_x], f.ravel()[selected])
            intensity = np.searchsorted(RAIN_EDGES, f.ravel()[selected], side="right") - 1
            assert (intensity >= 0).all() and (intensity < len(RAIN_LABELS)).all()
            for scaling in SCALES:
                d, _, _, _ = flow_and_residual(f, g, scaling)
                ux, uy = d[dest_y, dest_x, 0], d[dest_y, dest_x, 1]
                epe = np.hypot(ux - dx, uy - dy)
                ratio = np.hypot(ux, uy) / np.hypot(dx, dy)
                assert np.isfinite(epe).all() and np.isfinite(ratio).all()
                for basis, maps in (("nominal", cfg["nominal"]), ("available", live)):
                    for radius, count_key in ((2.5, "count25"), (5., "count5")):
                        counts = maps[count_key][selected]
                        # Fixed thresholds per unit area, matching <=2, 3--5,
                        # >=6 gauges at 2.5 km radius, for both radius choices.
                        density = counts / (np.pi * radius ** 2)
                        group = np.searchsorted([3 / (np.pi * 2.5 ** 2), 6 / (np.pi * 2.5 ** 2)], density, side="right")
                        for ibin in np.unique(intensity):
                            for dbin in np.unique(group):
                                take = (intensity == ibin) & (group == dbin)
                                if not take.any():
                                    continue
                                n = int(take.sum())
                                row = {"storm_id": sid, "frame": k, "dx": dx, "dy": dy,
                                       "scaling": scaling, "density_basis": basis, "radius_km": radius,
                                       "rain_bin": RAIN_LABELS[ibin], "density_group": DENSITY_LABELS[dbin],
                                       "n_cells": n, "epe_sum_cells": float(epe[take].astype(float).sum()),
                                       "speed_ratio_sum": float(ratio[take].astype(float).sum()),
                                       "epe_le_1cell_count": int((epe[take] <= 1).sum()),
                                       "rain_sum_mm": float(f.ravel()[selected][take].astype(float).sum()),
                                       "gradient_sum_mm_per_cell": float(gradient[selected][take].astype(float).sum()),
                                       "gauge_count_sum": int(counts[take].sum())}
                                for name in ("nn_mean_km", "nn_variance_km2", "nn_cv"):
                                    vals = maps[name][selected][take]
                                    ok = np.isfinite(vals)
                                    row[name + "_sum"] = float(vals[ok].sum())
                                    row[name + "_n"] = int(ok.sum())
                                rows.append(row)
    return rows, frame_rows


def summaries(cells, output):
    keys = ["scaling", "density_basis", "radius_km", "rain_bin", "density_group"]
    summary = cells.groupby(keys, observed=True).agg(
        cell_tests=("n_cells", "sum"), storms=("storm_id", "nunique"),
        epe_sum=("epe_sum_cells", "sum"), ratio_sum=("speed_ratio_sum", "sum"),
        epe_le_1=("epe_le_1cell_count", "sum"), rain_sum=("rain_sum_mm", "sum"),
        nn_var_sum=("nn_variance_km2_sum", "sum"), nn_var_n=("nn_variance_km2_n", "sum"),
        nn_mean_sum=("nn_mean_km_sum", "sum"), nn_mean_n=("nn_mean_km_n", "sum"),
        nn_cv_sum=("nn_cv_sum", "sum"), nn_cv_n=("nn_cv_n", "sum")).reset_index()
    summary["mean_epe_cells"] = summary.epe_sum / summary.cell_tests
    summary["mean_speed_ratio"] = summary.ratio_sum / summary.cell_tests
    summary["fraction_epe_le_1cell"] = summary.epe_le_1 / summary.cell_tests
    summary["mean_rain_mm"] = summary.rain_sum / summary.cell_tests
    for metric in ("var", "mean", "cv"):
        summary[f"mean_local_nn_{metric}"] = summary[f"nn_{metric}_sum"] / summary[f"nn_{metric}_n"].replace(0, np.nan)
    summary.to_csv(output / "density_intensity_summary.csv", index=False)

    # Pair dense/sparse regions within the same storm/frame/shift/intensity
    # stratum. Average contrasts within storm before storm-cluster bootstrap.
    paired_keys = ["storm_id", "frame", "dx", "dy", "scaling", "density_basis", "radius_km", "rain_bin"]
    eligible = cells.loc[cells.n_cells >= 5].copy()
    eligible["mean_epe"] = eligible.epe_sum_cells / eligible.n_cells
    wide = eligible.pivot(index=paired_keys, columns="density_group", values="mean_epe")
    wide = wide.reindex(columns=DENSITY_LABELS)
    paired = wide.dropna(subset=["sparse", "dense"]).reset_index()
    paired["dense_minus_sparse_epe"] = paired["dense"] - paired["sparse"]
    paired.to_csv(output / "paired_density_contrasts.csv", index=False)
    storm = paired.groupby(["storm_id", "scaling", "density_basis", "radius_km"], observed=True).dense_minus_sparse_epe.mean().reset_index()
    storm.to_csv(output / "storm_density_contrasts.csv", index=False)
    contrasts = []
    for key, group in storm.groupby(["scaling", "density_basis", "radius_km"], observed=True):
        values = group.dense_minus_sparse_epe.to_numpy()
        rng = np.random.default_rng(20261012)
        boot = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
        mask = (paired.scaling == key[0]) & (paired.density_basis == key[1]) & (paired.radius_km == key[2])
        contrasts.append({"scaling": key[0], "density_basis": key[1], "radius_km": key[2],
                          "matched_storms": len(values), "matched_strata": int(mask.sum()),
                          "mean_dense_minus_sparse_epe_cells": float(values.mean()),
                          "storm_bootstrap_ci_low": float(np.quantile(boot, .025)),
                          "storm_bootstrap_ci_high": float(np.quantile(boot, .975))})
    result = pd.DataFrame(contrasts)
    result.to_csv(output / "matched_density_summary.csv", index=False)
    print(result.to_string(index=False))
    return {"storm_frame_shift_cell_tests": int(cells.loc[
        cells.scaling.eq(SCALES[0]) & cells.density_basis.eq("nominal") & cells.radius_km.eq(2.5), "n_cells"].sum()),
            "group_rows": len(cells), "matched_contrasts": contrasts}


def self_test():
    # A source cell retains its own value and source-density index after a
    # known shift, even when its destination has a different gauge count.
    f = np.arange(35 * 40, dtype=np.float32).reshape(35, 40) + 1
    iy, ix = np.indices(f.shape)
    cfg = {"shape": f.shape, "iy": iy, "ix": ix,
           "inside": np.ones(f.shape, dtype=bool), "border": 2}
    for dx, dy in SHIFTS:
        take = origin_selection(cfg, f, dx, dy)
        sy, sx = iy.ravel()[take], ix.ravel()[take]
        shifted = shift_field(f, dx, dy)
        np.testing.assert_array_equal(shifted[sy + dy, sx + dx], f.ravel()[take])
    # Local spacing moments are evaluated on the chosen available network.
    tiny = {"distances": np.array([[0., 1000., 3000.]]),
            "station_distances": np.array([[0., 1000., 3000.], [1000., 0., 2000.], [3000., 2000., 0.]])}
    maps = support_maps(tiny, np.array([True, True, True]))
    np.testing.assert_allclose(maps["nn_mean_km"], [4 / 3])
    np.testing.assert_allclose(maps["nn_variance_km2"], [2 / 9])
    assert maps["count25"][0] == 2 and maps["count5"][0] == 3
    assert np.isnan(support_maps(tiny, np.array([True, True, False]))["nn_variance_km2"][0])
    print("Source/destination alignment and spacing-moment checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    ap.add_argument("--stations", type=Path, default=SOURCE / "station.csv")
    ap.add_argument("--boundary", type=Path, default=SOURCE / "bangkok_boundary.geojson")
    ap.add_argument("--output", type=Path, default=HERE / "density_motion_check")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--border", type=int, default=12, help="source and destination edge exclusion, cells")
    args = ap.parse_args()
    if args.workers < 1 or args.border < 0 or args.limit < 0:
        ap.error("workers >=1, border >=0 and limit >=0 required")
    self_test()
    storms = completed_storms(args.products)
    if args.limit:
        storms = storms.head(args.limit)
    cfg, geometry = setup_geometry(args.products, storms.storm_id.iloc[0], args.stations, args.boundary, args.border)
    args.output.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(args.output / "grid_gauge_geometry.csv", index=False)
    print("Gauge-count bins inside Bangkok (2.5 km radius):", pd.cut(
        geometry.loc[geometry.inside_bangkok, "count25"], [-1, 2, 5, np.inf]).value_counts(sort=False).to_dict())
    ids = storms.storm_id.tolist()
    if args.workers == 1:
        init_worker(cfg)
        results = [worker(sid) for sid in tqdm(ids, desc="Storms")]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ids)),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=(cfg,)) as executor:
            results = list(tqdm(executor.map(worker, ids), total=len(ids),
                                desc=f"Density recovery ({args.workers} workers)", mininterval=1))
    cells = pd.DataFrame([row for result in results for row in result[0]])
    frames = pd.DataFrame([row for result in results for row in result[1]])
    assert frames.storm_id.nunique() == len(ids) and not frames.duplicated(["storm_id", "frame"]).any()
    assert not cells.duplicated(["storm_id", "frame", "dx", "dy", "scaling", "density_basis", "radius_km", "rain_bin", "density_group"]).any()
    assert np.isfinite(cells.select_dtypes("number").to_numpy()).all()
    cells.to_csv(args.output / "cell_group_results.csv", index=False)
    frames.to_csv(args.output / "frame_availability.csv", index=False)
    report = summaries(cells, args.output)
    report.update({"storms": len(ids), "selected_frames": len(frames), "workers": args.workers,
                   "border_cells": args.border, "opencv": flow.cv2.__version__,
                   "density_thresholds_gauges_per_km2": [3 / (np.pi * 2.5 ** 2), 6 / (np.pi * 2.5 ** 2)],
                   "spacing_definition": "Nearest-neighbour distance in available or nominal network; local mean and population variance among stations within 5 km; variance/CV require >=3 local stations.",
                   "sampling": "Three selected frames per storm, three known integer shifts, three input scalings; source-origin density; exclude 12-cell borders by default.",
                   "inference_limit": "Association on existing reconstructed fields; intensity-stratified and storm-clustered, not causal density loss or independent motion truth.",
                   "checks_passed": True,
                   "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [Path(__file__), HERE / "flow_scaling_check.py", args.stations,
                               args.boundary, args.products / "all_storm_results.csv"]}})
    (args.output / "run_info.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Complete: {len(ids)} storms, {len(frames)} frames, {len(cells)} group rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_event_grid_xarray.py

สร้าง rainfall grid (time,y,x) จาก event cache parquet แล้วบันทึกเป็น xarray.Dataset
- อ่าน cache: results/event_cache/event_YYYYMMDD_HHMM_leadXX_lagXX.parquet
- อ่าน station locations: source/station.csv (id,nameShort,lat,lon)
- อ่านขอบเขตกทม.: source/bangkok_boundary.geojson (optional)
- สร้างกริด 1x1 km² ใน UTM47N
- สร้าง field ต่อเฟรม:
    - base grid: เฉพาะ cell ที่มี station (avg ถ้ามีหลาย station ใน cell)
    - optional smooth: Gaussian kernel smoothing + normalization (เติมกริดข้างเคียง)
    - optional anchor: overwrite cell ที่มี station ให้กลับเป็นค่าจริง (hard anchor)
    - optional threshold min_rain

Outputs:
- Zarr (recommended) หรือ NetCDF

Example:
  python build_event_grid_xarray.py \
    --cache results/event_cache/event_20141031_1830_lead45_lag45.parquet \
    --stations source/station.csv \
    --boundary source/bangkok_boundary.geojson \
    --grid-km 1.0 --pad-km 2.0 \
    --smooth --sigma-km 2.5 --cutoff-sigma 3.0 \
    --no-anchor-stations \
    --min-rain 0.1 \
    --out results/event_grids/event0_soft.zarr

Deps (conda/mamba):
  mamba install -c conda-forge numpy pandas pyproj xarray zarr tqdm
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from pyproj import Transformer


# --- CRS ---
TF_WGS84_TO_UTM47 = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)


# ---------- small helpers ----------
def normalize_stationid_to_int(series: pd.Series) -> pd.Series:
    """
    Convert stationid like 'S001' -> 1 (Int64). Also handles numeric/string mixed.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[Ss]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_station_locations(station_csv: str) -> pd.DataFrame:
    s = pd.read_csv(station_csv)
    req = ["id", "nameShort", "lat", "lon"]
    miss = [c for c in req if c not in s.columns]
    if miss:
        raise ValueError(f"station.csv missing columns: {miss} | columns={list(s.columns)}")

    s = s.rename(columns={"id": "stationid"})
    s["stationid"] = pd.to_numeric(s["stationid"], errors="coerce").astype("Int64")
    s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    s = s.dropna(subset=["stationid", "lat", "lon"]).copy()
    return s[["stationid", "nameShort", "lat", "lon"]]


def project_stations_to_utm(st: pd.DataFrame) -> pd.DataFrame:
    x, y = TF_WGS84_TO_UTM47.transform(st["lon"].to_numpy(), st["lat"].to_numpy())
    out = st.copy()
    out["x"] = np.asarray(x, dtype=float)
    out["y"] = np.asarray(y, dtype=float)
    return out


def infer_bounds_wgs84_from_stations(st: pd.DataFrame, pad_deg: float = 0.05) -> Tuple[float, float, float, float]:
    """
    Fallback domain bounds when no boundary geojson is available.
    """
    min_lon = float(st["lon"].min()) - pad_deg
    max_lon = float(st["lon"].max()) + pad_deg
    min_lat = float(st["lat"].min()) - pad_deg
    max_lat = float(st["lat"].max()) + pad_deg
    return (min_lon, min_lat, max_lon, max_lat)


def load_boundary_bounds_wgs84(geojson_path: str) -> Tuple[Tuple[float, float, float, float], dict]:
    """
    Return bounds_wgs84 = (min_lon, min_lat, max_lon, max_lat) and geojson dict (for metadata).
    Supports Polygon/MultiPolygon.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    lons = []
    lats = []

    feats = gj.get("features", [])
    if not feats:
        raise RuntimeError(f"No features in geojson: {geojson_path}")

    for feat in feats:
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        def collect_ring(ring):
            arr = np.array(ring, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                return
            lons.append(arr[:, 0])
            lats.append(arr[:, 1])

        if gtype == "Polygon":
            if coords and coords[0]:
                collect_ring(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly and poly[0]:
                    collect_ring(poly[0])

    if not lons:
        raise RuntimeError(f"No Polygon/MultiPolygon rings found in geojson: {geojson_path}")

    all_lon = np.concatenate(lons)
    all_lat = np.concatenate(lats)
    bounds = (float(all_lon.min()), float(all_lat.min()), float(all_lon.max()), float(all_lat.max()))
    return bounds, gj


def build_grid_edges_from_bounds(bounds_wgs84, cell_m: float, pad_m: float):
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    xmin, ymin = TF_WGS84_TO_UTM47.transform(min_lon, min_lat)
    xmax, ymax = TF_WGS84_TO_UTM47.transform(max_lon, max_lat)

    xmin -= pad_m
    xmax += pad_m
    ymin -= pad_m
    ymax += pad_m

    x_edges = np.arange(float(xmin), float(xmax) + cell_m, cell_m)
    y_edges = np.arange(float(ymin), float(ymax) + cell_m, cell_m)
    return x_edges, y_edges


def aggregate_to_grid(
    df_t: pd.DataFrame, stations_xy: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    """
    df_t columns: stationid(int), rainfall(float)
    stations_xy columns: stationid(int), x,y (UTM)
    Returns grid[ny, nx] with NaN for empty cells (only where stations exist in that cell)
    """
    merged = df_t.merge(stations_xy[["stationid", "x", "y"]], on="stationid", how="inner")
    ny = len(y_edges) - 1
    nx = len(x_edges) - 1
    if merged.empty:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    xs = merged["x"].to_numpy()
    ys = merged["y"].to_numpy()

    ix = np.searchsorted(x_edges, xs, side="right") - 1
    iy = np.searchsorted(y_edges, ys, side="right") - 1
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(ok):
        return np.full((ny, nx), np.nan, dtype=np.float32)

    merged = merged.loc[ok].copy()
    merged["ix"] = ix[ok]
    merged["iy"] = iy[ok]

    g = merged.groupby(["iy", "ix"])["rainfall"].mean()

    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    for (iy_, ix_), val in g.items():
        grid[int(iy_), int(ix_)] = float(val)
    return grid


def gaussian_smooth_grid_from_stations(
    df_t: pd.DataFrame,
    stations_xy: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_m: float,
    cutoff_sigma: float,
    anchor_grid: Optional[np.ndarray],
    only_influenced: bool,
) -> np.ndarray:
    """
    R(g) = sum_i w(d) r_i / sum_i w(d)
    w(d) = exp(-d^2/(2*sigma^2)), cutoff = cutoff_sigma*sigma
    """
    ny = len(y_edges) - 1
    nx = len(x_edges) - 1

    xc = (x_edges[:-1] + x_edges[1:]) / 2.0
    yc = (y_edges[:-1] + y_edges[1:]) / 2.0

    obs = df_t.merge(stations_xy[["stationid", "x", "y"]], on="stationid", how="inner")
    obs = obs.dropna(subset=["x", "y", "rainfall"])
    if obs.empty:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    xs = obs["x"].to_numpy(dtype=float)
    ys = obs["y"].to_numpy(dtype=float)
    rs = obs["rainfall"].to_numpy(dtype=float)

    num = np.zeros((ny, nx), dtype=np.float64)
    den = np.zeros((ny, nx), dtype=np.float64)

    cutoff_m = float(cutoff_sigma) * float(sigma_m)
    cutoff2 = cutoff_m * cutoff_m
    two_sigma2 = 2.0 * float(sigma_m) * float(sigma_m)

    for x0, y0, r0 in zip(xs, ys, rs):
        x_lo = x0 - cutoff_m
        x_hi = x0 + cutoff_m
        y_lo = y0 - cutoff_m
        y_hi = y0 + cutoff_m

        ix0 = np.searchsorted(xc, x_lo, side="left")
        ix1 = np.searchsorted(xc, x_hi, side="right")
        iy0 = np.searchsorted(yc, y_lo, side="left")
        iy1 = np.searchsorted(yc, y_hi, side="right")

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        X = xc[ix0:ix1]
        Y = yc[iy0:iy1]

        dx = (X[None, :] - x0)
        dy = (Y[:, None] - y0)
        d2 = dx * dx + dy * dy

        mask = d2 <= cutoff2
        if not np.any(mask):
            continue

        w = np.exp(-d2 / two_sigma2, dtype=np.float64)
        w[~mask] = 0.0

        num[iy0:iy1, ix0:ix1] += w * r0
        den[iy0:iy1, ix0:ix1] += w

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    ok = den > 0
    out[ok] = (num[ok] / den[ok]).astype(np.float32)

    if anchor_grid is not None:
        aok = ~np.isnan(anchor_grid)
        out[aok] = anchor_grid[aok]

    if only_influenced:
        # already NaN where den==0
        pass

    return out


def apply_min_rain(grid: np.ndarray, min_rain: Optional[float], to_nan: bool) -> np.ndarray:
    if min_rain is None:
        return grid
    thr = float(min_rain)
    out = grid.copy()
    mask = (~np.isnan(out)) & (out < thr)
    if to_nan:
        out[mask] = np.nan
    else:
        out[mask] = 0.0
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="event cache parquet file (from extract_event_window_df.py)")
    ap.add_argument("--stations", default="source/station.csv")
    ap.add_argument(
        "--boundary",
        default="source/bangkok_boundary.geojson",
        help="optional Bangkok boundary geojson; if missing, derive bounds from stations",
    )

    ap.add_argument("--grid-km", type=float, default=1.0)
    ap.add_argument("--pad-km", type=float, default=2.0)

    ap.add_argument("--only-flag-N", action="store_true", help="keep only rainfall_flag == 'N' if exists")

    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--sigma-km", type=float, default=2.5)
    ap.add_argument("--cutoff-sigma", type=float, default=3.0)
    ap.add_argument("--anchor-stations", dest="anchor_stations", action="store_true",
                    help="preserve station-cell values after smoothing (default)")
    ap.add_argument("--no-anchor-stations", dest="anchor_stations", action="store_false",
                    help="soft smoothing (do not anchor station-cells)")
    ap.set_defaults(anchor_stations=True)
    ap.add_argument("--only-influenced", action="store_true",
                    help="(when smoothing) keep NaN where no station within cutoff")

    ap.add_argument("--min-rain", type=float, default=None)
    ap.add_argument("--min-rain-to-nan", action="store_true")

    ap.add_argument("--out", required=True, help="output .zarr or .nc")
    ap.add_argument("--chunks-time", type=int, default=1, help="zarr chunk size along time")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    out_path = Path(args.out)
    if out_path.exists():
        if args.overwrite:
            if out_path.is_dir():
                # zarr directory
                for p in sorted(out_path.rglob("*"), reverse=True):
                    if p.is_file():
                        p.unlink()
                    else:
                        p.rmdir()
                out_path.rmdir()
            else:
                out_path.unlink()
        else:
            raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")

    # Load event cache
    df = pd.read_parquet(cache_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "stationid"])

    df["stationid"] = normalize_stationid_to_int(df["stationid"])
    df = df.dropna(subset=["stationid"]).copy()
    df["stationid"] = df["stationid"].astype("Int64")

    if args.only_flag_N and "rainfall_flag" in df.columns:
        df["rainfall_flag"] = df["rainfall_flag"].astype("string")
        df = df[df["rainfall_flag"] == "N"].copy()

    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

    times = sorted(df["time"].dropna().unique().tolist())
    if not times:
        raise RuntimeError("No valid timestamps in cache after parsing/filtering.")

    # Stations + boundary bounds
    st = load_station_locations(args.stations)
    st = st[st["stationid"].isin(df["stationid"].unique())].copy()
    st_xy = project_stations_to_utm(st)

    boundary_path = Path(args.boundary)
    if boundary_path.exists():
        bounds_wgs84, boundary_gj = load_boundary_bounds_wgs84(str(boundary_path))
        boundary_source = str(boundary_path)
    else:
        bounds_wgs84 = infer_bounds_wgs84_from_stations(st)
        boundary_gj = None
        boundary_source = ""
        print(f"[INFO] optional boundary not found: {boundary_path} ; deriving bounds from station extent")

    # Grid edges and centers (UTM meters)
    cell_m = float(args.grid_km) * 1000.0
    pad_m = float(args.pad_km) * 1000.0
    x_edges, y_edges = build_grid_edges_from_bounds(bounds_wgs84, cell_m=cell_m, pad_m=pad_m)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    ny = len(y_centers)
    nx = len(x_centers)

    sigma_m = float(args.sigma_km) * 1000.0

    # Build stack
    stack = np.full((len(times), ny, nx), np.nan, dtype=np.float32)

    for ti, t in enumerate(tqdm(times, desc="Build grid stack")):
        dft = df.loc[df["time"] == t, ["stationid", "rainfall"]].copy()
        dft["stationid"] = dft["stationid"].astype("Int64")
        dft["rainfall"] = pd.to_numeric(dft["rainfall"], errors="coerce")

        base_grid = aggregate_to_grid(dft, st_xy, x_edges, y_edges)

        if args.smooth:
            anchor = base_grid if args.anchor_stations else None
            grid = gaussian_smooth_grid_from_stations(
                df_t=dft,
                stations_xy=st_xy,
                x_edges=x_edges,
                y_edges=y_edges,
                sigma_m=sigma_m,
                cutoff_sigma=float(args.cutoff_sigma),
                anchor_grid=anchor,
                only_influenced=bool(args.only_influenced),
            )
        else:
            grid = base_grid

        grid = apply_min_rain(grid, args.min_rain, bool(args.min_rain_to_nan))
        stack[ti] = grid

    # Build xarray Dataset
    da = xr.DataArray(
        stack,
        coords={
            "time": pd.to_datetime(times),
            "y": y_centers,  # UTM northing (m)
            "x": x_centers,  # UTM easting (m)
        },
        dims=("time", "y", "x"),
        name="rainfall",
        attrs={
            "units": "mm/15min",
            "grid_cell_m": float(cell_m),
            "crs": "EPSG:32647",
            "smooth_enabled": bool(args.smooth),
            "sigma_km": float(args.sigma_km) if args.smooth else 0.0,
            "cutoff_sigma": float(args.cutoff_sigma) if args.smooth else 0.0,
            "anchor_stations": bool(args.anchor_stations) if args.smooth else False,
            "only_influenced": bool(args.only_influenced) if args.smooth else False,
            "min_rain": float(args.min_rain) if args.min_rain is not None else None,
            "min_rain_to_nan": bool(args.min_rain_to_nan) if args.min_rain is not None else False,
            "source_cache": str(cache_path),
        },
    )

    ds = xr.Dataset({"rainfall": da})

    # Add some metadata useful later
    ds.attrs["boundary_geojson_path"] = boundary_source
    ds.attrs["boundary_source"] = "geojson" if boundary_source else "stations_extent"
    ds.attrs["stations_csv_path"] = str(Path(args.stations))
    ds.attrs["n_stations_gps"] = int(len(st_xy))
    ds.attrs["time_start"] = str(pd.Timestamp(times[0]))
    ds.attrs["time_end"] = str(pd.Timestamp(times[-1]))

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".zarr":
        # chunking (time chunk small; spatial chunk moderate)
        chunks = {
            "time": int(args.chunks_time),
            "y": min(128, ny),
            "x": min(128, nx),
        }
        ds_chunked = ds.chunk(chunks)
        ds_chunked.to_zarr(out_path, mode="w")
        print(f"[OK] saved zarr: {out_path}")
    elif suffix in [".nc", ".netcdf"]:
        # netcdf can be big; ok for small event (12 frames)
        ds.to_netcdf(out_path)
        print(f"[OK] saved netcdf: {out_path}")
    else:
        raise ValueError("Unknown output suffix. Use .zarr or .nc")

    print("[INFO] dataset summary:")
    print(ds)


if __name__ == "__main__":
    main()

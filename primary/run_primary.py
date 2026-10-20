"""Portable primary pair-max and fixed-window finite-fill reconstruction.

Input rainfall remains provider-restricted. Supply reconstruction-matched
coordinates explicitly; no automatic coordinate update or event relinking.
"""
import argparse
import importlib.util
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"analysis"))
from flow_scaling_check import flow_and_residual, flow
from no_advection_ablation import METRICS, read_zarr_v3_array, summarize, operational_label
spec=importlib.util.spec_from_file_location("grid",HERE/"processing/build_event_grid_xarray.py")
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)


def reconstruct(cache,stations,boundary,sigma_km=2.5):
    df=cache.copy();df["time"]=pd.to_datetime(df.time,errors="coerce")
    df=df.dropna(subset=["time","stationid"])
    df["stationid"]=grid.normalize_stationid_to_int(df.stationid)
    df=df.dropna(subset=["stationid"]);df["rainfall"]=pd.to_numeric(df.rainfall,errors="coerce")
    st=grid.load_station_locations(str(stations));st=grid.project_stations_to_utm(st[st.stationid.isin(df.stationid.unique())])
    bounds,_=grid.load_boundary_bounds_wgs84(str(boundary));xe,ye=grid.build_grid_edges_from_bounds(bounds,1000.,2000.)
    fields=[]
    for t in sorted(df.time.unique().tolist()):
        dt=df.loc[df.time.eq(t),["stationid","rainfall"]]
        anchor=grid.aggregate_to_grid(dt,st,xe,ye)
        r=grid.gaussian_smooth_grid_from_stations(dt,st,xe,ye,float(sigma_km)*1000.,3.,anchor,False)
        fields.append(grid.apply_min_rain(r,.1,False))
    return np.stack(fields)


def metrics(rain,scaling):
    res=np.stack([flow_and_residual(rt,rn,scaling)[2] for rt,rn in zip(rain[:-1],rain[1:])])
    m=summarize(res,rain[1:]);m["label"]=operational_label(m)
    return m


def one(task):
    row,cfg=task;flow.cv2.setNumThreads(1);sid=row["storm_id"]
    r=read_zarr_v3_array(Path(cfg["products"])/"grids"/f"{sid}.zarr"/"rainfall")
    result={"storm_id":sid}
    for scaling in ["pair_max","fixed15_trunc"]:
        m=metrics(r,scaling);result.update({f"{scaling}_{k}":v for k,v in m.items()})
    original=np.nan_to_num(r).astype(np.float32)
    m=summarize(original[1:]-original[:-1],r[1:]);m["label"]=operational_label(m)
    result.update({f"no_advection_{k}":v for k,v in m.items()})
    if cfg["no_fill"]:
        cache=pd.read_parquet(Path(cfg["data_root"])/row["cache_path"])
        values=pd.to_numeric(cache.rainfall,errors="coerce")
        filled=cache.rainfall_flag.fillna("").eq("M")&np.isfinite(values)
        result["filled_finite"]=int(filled.sum());result["filled_zero"]=int((filled&values.eq(0)).sum())
        result["filled_above_0p1"]=int((filled&values.gt(.1)).sum())
        if filled.any():
            rebuilt=reconstruct(cache,cfg["stations"],cfg["boundary"])
            np.testing.assert_array_equal(np.isnan(rebuilt),np.isnan(r))
            np.testing.assert_array_equal(np.nan_to_num(rebuilt),np.nan_to_num(r))
            cache.loc[filled,"rainfall"]=np.nan
            nf=reconstruct(cache,cfg["stations"],cfg["boundary"])
            result["no_fill_support_cells_changed"]=int(np.sum(np.isnan(nf)!=np.isnan(r)))
            for scaling in ["pair_max","fixed15_trunc"]:
                m=metrics(nf,scaling);result.update({f"nofill_{scaling}_{k}":v for k,v in m.items()})
        else:
            for scaling in ["pair_max","fixed15_trunc"]:
                for k in [*METRICS,"label"]:result[f"nofill_{scaling}_{k}"]=result[f"{scaling}_{k}"]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products",type=Path,required=True,help="Archived product directory with all_storm_results.csv and grids/")
    p.add_argument("--data-root",type=Path,help="Root against which cache_path resolves")
    p.add_argument("--stations",type=Path,help="Retained reconstruction coordinates, not later updates")
    p.add_argument("--boundary",type=Path)
    p.add_argument("--no-fill",action="store_true",help="Remove every finite M value at fixed cache windows, including zeros")
    p.add_argument("--workers",type=int,default=1);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.no_fill and not all([a.data_root,a.stations,a.boundary]):p.error("--no-fill requires --data-root --stations --boundary")
    if a.workers<1 or a.limit<0:p.error("workers >=1; limit >=0")
    data=pd.read_csv(a.products/"all_storm_results.csv");data=data.loc[data.status.eq("done")]
    if a.limit:data=data.head(a.limit)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    tasks=[(r,cfg) for r in data.to_dict("records")]
    if a.workers==1:rows=[one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn")) as pool:rows=list(pool.map(one,tasks))
    a.output.mkdir(parents=True,exist_ok=True);results=pd.DataFrame(rows).sort_values("storm_id")
    results.to_csv(a.output/"primary_and_controls.csv",index=False)
    (a.output/"run_parameters.json").write_text(json.dumps(cfg,indent=2)+"\n")
    print(f"Completed {len(results)} fixed windows: {results.pair_max_label.value_counts().to_dict()}")


if __name__=="__main__":main()

"""Separate year-round evaluation with unchanged screens, geometry and rule.

Uses complete-source chunks, reconstructs requested lead-in, records support
before classification. Resume by per-event JSON checkpoints. No observations
are exported outside internal/. Original 157-window products are read-only.
"""
import argparse
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'primary'))
from run_primary import grid
from reconstruction import reconstruct,configure
from flow_scaling_check import flow_and_residual
from no_advection_ablation import summarize,operational_label
from run_reconstruction_controls import donor_support
OUT=None
PARTS=None
EXPECTED_STATIONS=131


def calculate(rain,scaling):
    results=[flow_and_residual(rt,rn,scaling) for rt,rn in zip(rain[:-1],rain[1:])]
    displacement=np.stack([r[0] for r in results]);advected=np.stack([r[1] for r in results]);residual=np.stack([r[2] for r in results])
    m=summarize(residual,rain[1:]);m['label']=operational_label(m)
    return m,displacement,advected,residual


def initialize(output,parts,stations,boundary,expected_stations):
    global OUT,PARTS,EXPECTED_STATIONS
    OUT=Path(output);PARTS=Path(parts);EXPECTED_STATIONS=expected_stations
    configure(stations,boundary)


@lru_cache(maxsize=1)
def source_dataset():
    return ds.dataset(PARTS,format='parquet')


def support(cache):
    rainy=cache.loc[cache.rainfall.ge(.1)]
    return {'rows':len(cache),'stations_total':int(cache.stationid.nunique()),
            'times_total':int(cache.time.nunique()),'rainy_records_total':len(rainy),
            'rainy_stations_total':int(rainy.stationid.nunique()),
            'rainy_times_total':int(rainy.time.nunique()),
            'max_rainy_stations_per_frame':int(rainy.groupby('time').stationid.nunique().max()) if len(rainy) else 0,
            'max_frame_max':float(cache.rainfall.max())}


def one(row):
    cv2.setNumThreads(1)
    sid=row['storm_id'];target=OUT/'checkpoints'/f'{sid}.json'
    if target.exists():return json.loads(target.read_text())
    start=time.monotonic()
    dataset=source_dataset()
    begin=pd.Timestamp(row['storm_start'])-pd.Timedelta(minutes=120)
    end=pd.Timestamp(row['storm_end'])
    cache=dataset.to_table(filter=(ds.field('time')>=begin)&(ds.field('time')<=end)).to_pandas().sort_values(['time','stationid']).reset_index(drop=True)
    expected=pd.date_range(begin,end,freq='15min')
    assert np.array_equal(cache.time.unique(),expected.to_numpy())
    assert not cache.duplicated(['time','stationid']).any()
    assert cache.groupby('time').stationid.nunique().eq(EXPECTED_STATIONS).all()
    record={**row,**support(cache),'complete_120min_lead':True}
    record['reconstruction_engine']='cached station weights; original summation order'
    passed=(record['rainy_records_total']>=20 and record['rainy_stations_total']>=3 and record['rainy_times_total']>=4 and record['max_rainy_stations_per_frame']>=2)
    record['status']='done' if passed else 'skipped_low_support'
    if passed:
        cache.to_parquet(OUT/'restricted/caches'/f'{sid}.parquet',index=False)
        records=[]
        for sigma in [1.,1.5,2.5,3.5,5.]:
            rain,times,x,y=reconstruct(cache,sigma=sigma)
            m,displacement,_,residual=calculate(rain,'pair_max')
            records.append({'sigma_km':sigma,**m})
            if sigma==2.5:
                base=rain
                record.update({f'pair_max_{k}':v for k,v in m.items()})
                mask=np.stack([donor_support(rt,d) for rt,d in zip(rain[:-1],displacement)])
                supported=summarize(np.where(mask,residual,np.nan),np.where(mask,rain[1:],np.nan))
                supported['label']=operational_label(supported)
                record.update({f'donor_{k}':v for k,v in supported.items()})
        record['scale_controls']=records
        unanchored,_,_,_=reconstruct(cache,anchor=False)
        m,_,_,_=calculate(unanchored,'pair_max')
        record.update({f'unanchored_{k}':v for k,v in m.items()})
        for scaling in ['fixed15_trunc','fixed15_round','log1p_fixed']:
            m,_,_,_=calculate(base,scaling)
            record.update({f'{scaling}_{k}':v for k,v in m.items()})
        zero=np.nan_to_num(base).astype(np.float32)
        m=summarize(zero[1:]-zero[:-1],base[1:]);m['label']=operational_label(m)
        record.update({f'no_advection_{k}':v for k,v in m.items()})
    record['elapsed_seconds']=time.monotonic()-start
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(target)
    return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--catalogue',type=Path,default=None)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source-parts',type=Path,required=True)
    ap.add_argument('--stations',type=Path,required=True);ap.add_argument('--boundary',type=Path,required=True)
    ap.add_argument('--expected-stations',type=int,default=131)
    args=ap.parse_args()
    initialize(args.output,args.source_parts,args.stations,args.boundary,args.expected_stations)
    for folder in ['checkpoints','restricted/caches']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.catalogue if args.catalogue else OUT/'eligible_baseline.csv')
    if args.limit:data=data.head(args.limit)
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.output,args.source_parts,args.stations,args.boundary,args.expected_stations)) as pool:
        for i,result in enumerate(pool.map(one,data.to_dict('records')),1):
            rows.append(result)
            print(f"full {i}/{len(data)} {result['storm_id']} {result['status']} {result['elapsed_seconds']:.1f}s; wall {time.monotonic()-start:.0f}s",flush=True)
    flat=[{k:v for k,v in row.items() if k!='scale_controls'} for row in rows]
    scales=[{'storm_id':row['storm_id'],**item} for row in rows for item in row.get('scale_controls',[])]
    suffix='_pilot' if args.limit else ''
    pd.DataFrame(flat).to_csv(OUT/f'results{suffix}.csv',index=False)
    pd.DataFrame(scales).to_csv(OUT/f'scale_results{suffix}.csv',index=False)
    report={'evaluated':len(rows),'elapsed_seconds':time.monotonic()-start,'workers':args.workers,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_primary_157_and_rule_unchanged':True,
            'completed':sum(r['status']=='done' for r in rows)}
    (OUT/f'run{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

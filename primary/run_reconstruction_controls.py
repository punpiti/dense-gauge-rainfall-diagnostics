"""Fixed-window pair-max donor-support and reconstruction-scale controls.

Run beside run_primary.py. Inputs remain restricted; supply the exact retained
coordinate version explicitly. No event relinking, threshold fitting or
coordinate correction is performed by this entrypoint.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from run_primary import reconstruct, flow_and_residual, flow, read_zarr_v3_array, summarize, operational_label


def donor_support(source, displacement):
    h,w=source.shape
    xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    support=flow.cv2.remap(np.isfinite(source).astype(np.float32),
                           xx-displacement[...,0].astype(np.float32),
                           yy-displacement[...,1].astype(np.float32),
                           interpolation=flow.cv2.INTER_LINEAR,
                           borderMode=flow.cv2.BORDER_CONSTANT,borderValue=0.)
    return support==1.


def self_test():
    rain=np.ones((4,5),np.float32);rain[1,1]=np.nan
    d=np.zeros((4,5,2),np.float32)
    np.testing.assert_array_equal(donor_support(rain,d),np.isfinite(rain))
    d[1,2,0]=.5
    assert not donor_support(rain,d)[1,2]
    d[1,2,0]=0
    assert donor_support(rain,d)[1,2]
    d[0,0,0]=.5
    assert not donor_support(rain,d)[0,0]


def calculate(rain):
    vectors=[];residuals=[]
    for rt,rn in zip(rain[:-1],rain[1:]):
        d,_,rr,_=flow_and_residual(rt,rn,'pair_max')
        vectors.append(d);residuals.append(rr)
    residual=np.stack(residuals)
    m=summarize(residual,rain[1:]);m['label']=operational_label(m)
    return m,np.stack(vectors),residual


def one(task):
    row,cfg=task;sid=row['storm_id'];flow.cv2.setNumThreads(1)
    rain=read_zarr_v3_array(Path(cfg['products'])/'grids'/f'{sid}.zarr'/'rainfall')
    cache=pd.read_parquet(Path(cfg['data_root'])/row['cache_path'])
    base,d,residual=calculate(rain)
    results=[]
    for sigma in [1.,1.5,2.5,3.5,5.]:
        rebuilt=reconstruct(cache,cfg['stations'],cfg['boundary'],sigma_km=sigma)
        if sigma==2.5:
            np.testing.assert_array_equal(rebuilt,rain)
            m=base
        else:
            m,_,_=calculate(rebuilt)
        results.append({'storm_id':sid,'control':'scale','sigma_km':sigma,**m})
    keep=np.stack([donor_support(rt,dd) for rt,dd in zip(rain[:-1],d)])
    m=summarize(np.where(keep,residual,np.nan),np.where(keep,rain[1:],np.nan))
    m['label']=operational_label(m)
    results.append({'storm_id':sid,'control':'supported_donors','sigma_km':2.5,**m})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--products',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--stations',type=Path,help='Reconstruction-matched coordinates, not later updates')
    p.add_argument('--boundary',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args();self_test()
    if a.self_test:
        print('Synthetic donor-support checks: PASS');return
    if not all([a.products,a.data_root,a.stations,a.boundary,a.output]):
        p.error('--products --data-root --stations --boundary --output are required')
    if a.workers<1 or a.limit<0:p.error('workers >=1; limit >=0')
    data=pd.read_csv(a.products/'all_storm_results.csv');data=data.loc[data.status.eq('done')]
    if data.storm_id.duplicated().any():raise ValueError('Duplicate storm IDs')
    if a.limit:data=data.head(a.limit)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    tasks=[(r,cfg) for r in data.to_dict('records')]
    rows=[]
    if a.workers==1:
        for t in tasks:rows.extend(one(t))
    else:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,result in enumerate(pool.map(one,tasks),1):
                rows.extend(result);print(f'Completed {i}/{len(tasks)}',flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows).sort_values(['storm_id','control','sigma_km'])
    result.to_csv(a.output/'reconstruction_controls.csv',index=False)
    cfg['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg['opencv']=flow.cv2.__version__
    cfg['storms']=len(data)
    (a.output/'run_parameters.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(f'Completed {len(data)} fixed windows; thresholds unchanged.')


if __name__=='__main__':main()

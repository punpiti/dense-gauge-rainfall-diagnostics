"""Fixed 7.5-km reconstruction cutoff on all 3,304 rebuilt windows.

Keep sigma, weights, anchoring, observations, windows and the Boolean rule
explicit. Same-cutoff finite support must match at every cell and time. No
threshold refitting or source/label changes. Original 3-sigma support is also
measured from the same caches for comparison with the retained controls.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from reconstruction import reconstruct,configure
from run_catalogue import calculate
from no_advection_ablation import METRICS
SOURCE=None
OUT=None


def initialize(source,output,stations,boundary):
    global SOURCE,OUT
    SOURCE=Path(source);OUT=Path(output);configure(stations,boundary)


SIGMAS=[1.,1.5,2.5,3.5,5.]


def support_summary(rain,metrics):
    # Support belongs to each later rainfall frame used by the indicators.
    counts=np.isfinite(rain[1:]).sum(axis=(1,2));peak=int(np.argmax(counts))
    return {'later_frame_count':len(counts),'finite_cell_times':int(counts.sum()),
            'supported_cells_min':int(counts.min()),'supported_cells_max':int(counts.max()),
            'supported_cells_mean':float(counts.mean()),
            'supported_cell_fraction_mean':float(counts.mean()/(rain.shape[1]*rain.shape[2])),
            'area_gate_possible_in_any_frame':bool(counts.max()>=500),
            'peak_area_over_max_supported':float(metrics['peak_pos_area_cells']/counts.max()) if counts.max() else np.nan}


def one(row):
    cv2.setNumThreads(1);sid=row['storm_id'];path=OUT/'checkpoints'/f'{sid}.json'
    if path.exists():return json.loads(path.read_text())
    started=time.monotonic();cache=pd.read_parquet(SOURCE/'restricted/caches'/f'{sid}.parquet')
    records=[];common=None
    for sigma in SIGMAS:
        original,_,_,_=reconstruct(cache,sigma=sigma)
        reference={k:row[f'scale_{sigma}_{k}'] for k in METRICS}
        records.append({'sigma_km':sigma,'cutoff_policy':'3sigma','cutoff_km':3*sigma,
                        'label':row[f'scale_{sigma}_label'],**reference,**support_summary(original,reference)})
        fixed,_,_,_=reconstruct(cache,sigma=sigma,cutoff_km=7.5)
        mask=np.isfinite(fixed)
        if common is None:common=mask
        else:np.testing.assert_array_equal(mask,common)
        metrics,_,_,_=calculate(fixed,'pair_max')
        if sigma==2.5:
            np.testing.assert_array_equal(fixed,original)
            for k in METRICS:np.testing.assert_allclose(metrics[k],row['pair_max_'+k],rtol=0,atol=1e-5)
            assert metrics['label']==row['pair_max_label']
        records.append({'sigma_km':sigma,'cutoff_policy':'fixed_7p5km','cutoff_km':7.5,
                        **metrics,**support_summary(fixed,metrics)})
    result={'storm_id':sid,'identical_fixed_cutoff_support_all_sigmas':True,
            'sigma_2p5_primary_reproduces':True,'seconds':time.monotonic()-started,'rows':records}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stations',type=Path,required=True);ap.add_argument('--boundary',type=Path,required=True)
    args=ap.parse_args();initialize(args.input,args.output,args.stations,args.boundary);(OUT/'checkpoints').mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE/'results.csv');data=data.loc[data.status.eq('done')].copy()
    scales=pd.read_csv(SOURCE/'scale_results.csv')
    for sigma in SIGMAS:
        sub=scales.loc[scales.sigma_km.eq(sigma)].set_index('storm_id')
        for k in [*METRICS,'label']:data[f'scale_{sigma}_{k}']=data.storm_id.map(sub[k])
    if args.limit:data=data.head(args.limit)
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.input,args.output,args.stations,args.boundary)) as pool:
        for i,record in enumerate(pool.map(one,data.to_dict('records')),1):
            rows.extend({'storm_id':record['storm_id'],**r} for r in record['rows'])
            if i%100==0 or i==len(data):print(f'fixed footprint {i}/{len(data)}; {time.monotonic()-start:.0f}s',flush=True)
    result=pd.DataFrame(rows);suffix='_pilot' if args.limit else ''
    result.to_csv(OUT/f'storm_results{suffix}.csv',index=False)
    summary=[];reference=data.set_index('storm_id').pair_max_label
    for (policy,sigma),g in result.groupby(['cutoff_policy','sigma_km']):
        summary.append({'cutoff_policy':policy,'sigma_km':sigma,'n':len(g),
                        'high_residual':int(g.label.eq('high_residual').sum()),
                        'changed_from_primary':int(g.label.ne(g.storm_id.map(reference)).sum()),
                        'supported_cells_mean_median':float(g.supported_cells_mean.median()),
                        'supported_cell_fraction_mean_median':float(g.supported_cell_fraction_mean.median()),
                        'unable_to_reach_area_gate':int((~g.area_gate_possible_in_any_frame).sum()),
                        'peak_area_median':float(g.peak_pos_area_cells.median()),
                        'peak_area_over_max_supported_median':float(g.peak_area_over_max_supported.median())})
    pd.DataFrame(summary).to_csv(OUT/f'summary{suffix}.csv',index=False)
    qa={'n':len(data),'cutoff_km':7.5,'sigma_2p5_baseline_reproduces_all':True,
        'finite_support_bit_identical_across_all_sigmas':True,
        'primary_and_rule_unchanged':True,'seconds':time.monotonic()-start,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/f'validation{suffix}.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':main()

"""Fixed-window station-cell anchoring ablation with exact baseline check."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from run_primary import reconstruct,metrics,flow,read_zarr_v3_array


def one(task):
    row,cfg=task;flow.cv2.setNumThreads(1)
    sid=row['storm_id'];cache=pd.read_parquet(Path(cfg['data_root'])/row['cache_path'])
    baseline=reconstruct(cache,cfg['stations'],cfg['boundary'])
    archived=read_zarr_v3_array(Path(cfg['products'])/'grids'/f'{sid}.zarr'/'rainfall')
    np.testing.assert_array_equal(baseline,archived)
    unanchored=reconstruct(cache,cfg['stations'],cfg['boundary'],anchor_stations=False)
    np.testing.assert_array_equal(np.isnan(baseline),np.isnan(unanchored))
    a=metrics(baseline,'pair_max');b=metrics(unanchored,'pair_max')
    return {'storm_id':sid,**{f'primary_{k}':v for k,v in a.items()},**{f'unanchored_{k}':v for k,v in b.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--products',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--stations',type=Path,required=True);p.add_argument('--boundary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args();data=pd.read_csv(args.products/'all_storm_results.csv');data=data.loc[data.status.eq('done')]
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        result=pd.DataFrame(pool.map(one,[(row,cfg) for row in data.to_dict('records')]))
    args.output.mkdir(parents=True,exist_ok=True);result.to_csv(args.output/'anchoring.csv',index=False)
    report={'n':len(result),'grids_reproduce_exactly':True,'finite_support_unchanged':True,
            'groups':result.unanchored_label.value_counts().to_dict(),
            'changed':int(result.primary_label.ne(result.unanchored_label).sum())}
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

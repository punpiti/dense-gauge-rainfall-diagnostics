"""Fixed +/-20% threshold perturbations; no fitting or preferred-rule selection."""
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
METRICS=['peak_pos_area_cells','peak_pos_sum','event_pos_sum','n_strong_frames']
BASE=np.array([500.,1000.,4000.,10.])

def labels(frame, thresholds):
    a,p,e,n=[frame[k].to_numpy() for k in METRICS]
    aa,pp,ee,nn=thresholds
    return (a>=aa)&((p>=pp)|(e>=ee)|(n>=nn))


def evaluate(frame, name):
    baseline=labels(frame,BASE);rows=[]
    for factors in product([.8,1.,1.2],repeat=4):
        threshold=BASE*np.array(factors)
        current=labels(frame,threshold)
        row={'dataset':name,'n':len(frame),'high_residual':int(current.sum()),
             'changed':int((current!=baseline).sum()),
             'high_to_other':int((baseline&~current).sum()),'other_to_high':int((~baseline&current).sum()),
             'perturbed_count':sum(x!=1. for x in factors)}
        for key,f,t in zip(METRICS,factors,threshold):
            row[key+'_factor']=f;row[key+'_threshold']=t
        rows.append(row)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--metric-prefix',default='pair_max_');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=pd.read_csv(args.input)
    if 'status' in data:data=data.loc[data.status.eq('done')].copy()
    data=data.rename(columns={args.metric_prefix+k:k for k in METRICS})
    result=pd.DataFrame(evaluate(data,'supplied_windows'));args.output.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output/'factorial.csv',index=False)
    result.loc[result.perturbed_count.le(1)].to_csv(args.output/'one_at_a_time.csv',index=False)
    print(f'{len(data)} windows; high-residual range {result.high_residual.min()}--{result.high_residual.max()}')

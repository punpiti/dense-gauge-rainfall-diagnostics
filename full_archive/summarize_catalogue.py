"""Source-backed cohort summary for the control figure and release checks."""
import argparse
import json
from pathlib import Path
import pandas as pd

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();all_results=pd.read_csv(args.input/'results.csv')
    data=all_results.loc[all_results.status.eq('done')].copy();data['storm_start']=pd.to_datetime(data.storm_start)
    assert data.storm_id.is_unique
    scales=pd.read_csv(args.input/'scale_results.csv');assert len(scales)==5*len(data)
    reference=data.set_index('storm_id').pair_max_label
    report={'eligible_before_support_screen':len(all_results),'passed_support':len(data),
            'excluded_low_support':int(all_results.status.eq('skipped_low_support').sum()),
            'by_start_month':data.groupby(data.storm_start.dt.month).size().to_dict(),
            'scale_controls':[],'motion_and_reference_controls':{}}
    for sigma,g in scales.groupby('sigma_km'):
        report['scale_controls'].append({'sigma_km':float(sigma),'high_residual':int(g.label.eq('high_residual').sum()),
                                        'changed_from_pair_max':int(g.label.ne(g.storm_id.map(reference)).sum())})
    for name in ['pair_max','fixed15_trunc','fixed15_round','log1p_fixed','no_advection','donor','unanchored']:
        report['motion_and_reference_controls'][name]={'high_residual':int(data[name+'_label'].eq('high_residual').sum()),
                'changed_from_pair_max':int(data[name+'_label'].ne(data.pair_max_label).sum())}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Summarized {len(data)} support-qualified windows.')

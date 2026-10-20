"""Synthetic integration check: one fragment-cut event and one intact event.

No real observations. The test runs the complete extraction/linkage audit.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd


def main():
    here=Path(__file__).resolve().parent
    script=here/'audit_archive_windows.py'
    if not script.exists():raise RuntimeError('Run alongside audit_archive_windows.py')
    with tempfile.TemporaryDirectory(prefix='archive-window-test-') as t:
        root=Path(t)
        times=pd.date_range('2000-11-14','2000-11-18',freq='15min')
        full=pd.DataFrame({'time':np.repeat(times,131),'stationid':np.tile(np.arange(1,132),len(times))})
        full['rainfall']=0.;full['rainfall_flag']='N'
        specifications=[('CUT','2000-11-15 12:00','2000-11-15 13:00','2000-11-15 11:45'),
                        ('INTACT','2000-11-16 14:00','2000-11-16 15:00','2000-11-16 14:00')]
        for sid,start,end,actual_start in specifications:
            wet=full.time.between(pd.Timestamp(actual_start),pd.Timestamp(end))&full.stationid.le(2)
            full.loc[wet,'rainfall']=2.
        full.to_csv(root/'archive.csv.bz2',index=False,compression='bz2')
        partition_start=pd.Timestamp('2000-11-15 12:00')
        pd.DataFrame({'stationid':[f'S{i:03d}' for i in range(1,132)],'year':2000,
                      'first_time':partition_start,'last_time':pd.Timestamp('2000-12-31 23:45')}).to_csv(root/'partitions.csv',index=False)
        windows=[];members=[]
        for sid,start,end,actual_start in specifications:
            start,end=pd.Timestamp(start),pd.Timestamp(end)
            begin=max(partition_start,start-pd.Timedelta(minutes=120))
            cache=full.loc[full.time.between(begin,end)].copy()
            cache.to_parquet(root/f'{sid}.parquet',index=False)
            windows.append({'storm_id':sid,'storm_start':start,'storm_end':end,
                            'cache_path':f'{sid}.parquet','convective_rule_label':'non_convective'})
            for station in ['S001','S002']:
                members.append({'storm_id':sid,'stationid':station,
                                 'event_start':start,'event_end':end})
        pd.DataFrame(windows).to_csv(root/'windows.csv',index=False)
        pd.DataFrame(members).to_parquet(root/'members.parquet',index=False)
        command=[sys.executable,str(script),'--archive',str(root/'archive.csv.bz2'),
                 '--windows',str(root/'windows.csv'),'--data-root',str(root),
                 '--partitions',str(root/'partitions.csv'),'--members',str(root/'members.parquet'),
                 '--cluster-script',str(here/'processing/cluster_station_events_to_storms.py'),
                 '--output',str(root/'results'),'--buffer-hours','3','--chunksize','1000']
        r=subprocess.run(command,text=True,capture_output=True)
        if r.returncode:raise RuntimeError(r.stdout+r.stderr)
        qa=json.loads((root/'results/validation.json').read_text())
        assert qa['start_changed']==['CUT'] and qa['end_changed']==[]
        assert qa['fragment_affected']==['CUT']
        assert qa['all_retained_cache_values_and_flags_match_complete_archive']
        assert qa['all_complete_components_closed']
        # Reusing the retained extraction gives identical scientific findings.
        r=subprocess.run(command+['--analyse-only'],text=True,capture_output=True)
        if r.returncode:raise RuntimeError(r.stdout+r.stderr)
        again=json.loads((root/'results/validation.json').read_text())
        assert qa==again
        print('PASS: cut-event detection, intact-event preservation, availability and extraction reuse.')


if __name__=='__main__':main()

"""Rebuild a separate full-source catalogue; never mutate historical products.

Chunk filenames are unique. Restricted rainfall stays in internal/. Events
are linked before screening, so long connected systems cannot be shortened
by an eligibility filter. No new calibration or threshold fitting.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=None
SOURCE=None
EXPECTED_ROWS=None


def load_cluster():
    path = ROOT/'primary/processing/cluster_station_events_to_storms.py'
    spec = importlib.util.spec_from_file_location('cluster', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def extract():
    folder = OUT / 'restricted/source_parts'
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob('*.parquet')):
        raise RuntimeError('Refusing to overwrite an existing extraction')
    states = {'baseline': {}, 'no_fill': {}}
    finished = {'baseline': [], 'no_fill': []}
    total = 0
    last = None
    start = time.monotonic()
    months = {}
    for number, chunk in enumerate(pd.read_csv(
        SOURCE, usecols=['time','stationid','rainfall','rainfall_flag'],
        dtype={k:'string' for k in ['time','stationid','rainfall','rainfall_flag']},
        chunksize=2_000_000), 1):
        chunk['time'] = pd.to_datetime(chunk.time, format='%Y-%m-%d %H:%M:%S')
        chunk['stationid'] = pd.to_numeric(chunk.stationid).astype('int16')
        chunk['rainfall'] = pd.to_numeric(chunk.rainfall, errors='coerce').astype(float)
        assert chunk.time.is_monotonic_increasing
        assert not chunk.duplicated(['time','stationid']).any()
        if last is not None:
            assert chunk.time.iloc[0] >= last
        last = chunk.time.iloc[-1]
        chunk.to_parquet(folder / f'chunk-{number:04d}.parquet', index=False)
        for month, n in chunk.groupby(chunk.time.dt.month).size().items():
            months[str(month)] = months.get(str(month), 0) + int(n)
        rainy = chunk.loc[chunk.rainfall.gt(.1)].sort_values(['stationid','time'])
        for mode in states:
            selected = rainy if mode == 'baseline' else rainy.loc[rainy.rainfall_flag.fillna('').ne('M')]
            for station, group in selected.groupby('stationid', sort=False):
                ids = group.time.diff().gt(pd.Timedelta(minutes=120)).cumsum()
                aggregates = group.groupby(ids).agg(
                    event_start=('time','min'), event_end=('time','max'),
                    n_rain_slots=('rainfall','size'), total_rainfall=('rainfall','sum'),
                    max_rainfall=('rainfall','max'), min_rainfall=('rainfall','min'))
                for item in aggregates.to_dict('records'):
                    previous = states[mode].get(station)
                    if previous is not None and item['event_start'] - previous['event_end'] <= pd.Timedelta(minutes=120):
                        previous['event_end'] = item['event_end']
                        for key in ['n_rain_slots','total_rainfall']:
                            previous[key] += item[key]
                        previous['max_rainfall'] = max(previous['max_rainfall'],item['max_rainfall'])
                        previous['min_rainfall'] = min(previous['min_rainfall'],item['min_rainfall'])
                    else:
                        if previous is not None:
                            finished[mode].append(previous)
                        states[mode][station] = {'stationid': f'S{station:03d}', **item}
        total += len(chunk)
        print(f'archive {number}: {total:,} rows; {time.monotonic()-start:.0f}s',flush=True)
    if EXPECTED_ROWS is not None:assert total == EXPECTED_ROWS
    for mode in states:
        finished[mode].extend(states[mode].values())
        events = pd.DataFrame(finished[mode])
        events['event_id'] = 'EV_S' + events.stationid + '_' + events.event_start.dt.strftime('%Y%m%d%H%M')
        events['average_rainfall'] = events.total_rainfall / events.n_rain_slots
        events['event_length_minutes'] = (events.event_end-events.event_start).dt.total_seconds()/60
        events = events.sort_values(['event_start','event_end','stationid','event_id']).reset_index(drop=True)
        assert events.event_id.is_unique
        events.to_parquet(OUT / f'restricted/events_{mode}.parquet',index=False)
    report = {'source':str(SOURCE),'source_bytes':SOURCE.stat().st_size,
              'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'rows':total,'rows_by_month':months,'chunks':number,
              'elapsed_seconds':time.monotonic()-start,'unique_chunk_names':True}
    (OUT/'extraction.json').write_text(json.dumps(report,indent=2)+'\n')


def catalogue():
    cluster, path = load_cluster()
    report = {'cluster_code':str(path),'cluster_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'rain_threshold':.1,'station_gap_minutes':120,'merge_gap_minutes':30,
              'screen':{'min_events':5,'min_stations':5,'min_duration':30,'max_duration':360}}
    for mode in ['baseline','no_fill']:
        events = pd.read_parquet(OUT / f'restricted/events_{mode}.parquet')
        storms,members = cluster.build_storm_clusters(events,30)
        storms['storm_id'] = storms.storm_id.str.replace('STORM_', 'FULL_',regex=False)
        members['storm_id'] = members.storm_id.str.replace('STORM_', 'FULL_',regex=False)
        storms.to_parquet(OUT/f'restricted/storms_{mode}.parquet',index=False)
        members.to_parquet(OUT/f'restricted/members_{mode}.parquet',index=False)
        eligible = storms.loc[storms.n_member_events.ge(5)&storms.n_member_stations.ge(5)&storms.storm_duration_minutes.between(30,360)].copy()
        eligible.to_csv(OUT/f'eligible_{mode}.csv',index=False)
        report[mode] = {'station_events':len(events),'all_components':len(storms),'eligible':len(eligible),
                        'by_start_month':eligible.groupby(eligible.storm_start.dt.month).size().to_dict(),
                        'by_start_year':eligible.groupby(eligible.storm_start.dt.year).size().to_dict()}
        print(json.dumps({mode:report[mode]},indent=2),flush=True)
    (OUT/'catalogue.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--catalogue-only',action='store_true')
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-rows',type=int)
    args=parser.parse_args();OUT=args.output;SOURCE=args.archive;EXPECTED_ROWS=args.expected_rows;OUT.mkdir(parents=True,exist_ok=True)
    if not args.catalogue_only: extract()
    catalogue()

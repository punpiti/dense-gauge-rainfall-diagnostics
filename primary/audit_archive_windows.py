"""Audit fixed storm windows against the complete cleaned archive, read-only.

Only the union of local windows plus a configurable context is retained from
the sequential compressed CSV scan. No annual catalogue is regenerated.
Raw extracted rows stay in a restricted Parquet file, never in release assets.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd


def merge_intervals(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start <= result[-1][1] + pd.Timedelta(minutes=15):
            result[-1][1]=max(result[-1][1],end)
        else:result.append([start,end])
    return result


def load_cluster(path):
    spec=importlib.util.spec_from_file_location('retained_cluster',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def events(frame):
    rows=[]
    for station,g in frame.groupby('stationid',sort=True):
        rainy=g.loc[g.rainfall.gt(.1)].sort_values('time')
        if rainy.empty:continue
        group=rainy.time.diff().gt(pd.Timedelta(minutes=120)).cumsum()
        for _,r in rainy.groupby(group,sort=False):
            start,end=r.time.iloc[0],r.time.iloc[-1]
            rows.append({'event_id':f'EV_SS{int(station):03d}_{start:%Y%m%d%H%M}',
                         'stationid':f'S{int(station):03d}','event_start':start,
                         'event_end':end,'n_rain_slots':len(r),
                         'total_rainfall':float(r.rainfall.sum())})
    return pd.DataFrame(rows).sort_values(['event_start','event_end','stationid','event_id']).reset_index(drop=True)


def signature(frame):
    return set(zip(frame.stationid.astype(str),pd.to_datetime(frame.event_start),pd.to_datetime(frame.event_end)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--partitions',type=Path,required=True)
    p.add_argument('--members',type=Path,required=True)
    p.add_argument('--cluster-script',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--buffer-hours',type=float,default=24)
    p.add_argument('--chunksize',type=int,default=2000000)
    p.add_argument('--analyse-only',action='store_true')
    a=p.parse_args()
    if a.buffer_hours<3:p.error('buffer must exceed station gap plus linkage gap')
    a.output.mkdir(parents=True,exist_ok=True)
    restricted=a.output/'restricted';restricted.mkdir(exist_ok=True)
    primary=pd.read_csv(a.windows)
    assert primary.storm_id.is_unique
    caches={};ranges=[]
    for row in primary.itertuples(index=False):
        c=pd.read_parquet(a.data_root/row.cache_path)
        c['time']=pd.to_datetime(c.time)
        c['stationid']=pd.to_numeric(c.stationid.astype(str).str.removeprefix('S')).astype(int)
        caches[row.storm_id]=c
        ranges.append([c.time.min()-pd.Timedelta(hours=a.buffer_hours),c.time.max()+pd.Timedelta(hours=a.buffer_hours)])
    intervals=merge_intervals(ranges)
    interval_frame=pd.DataFrame(intervals,columns=['start','end'])
    interval_frame.to_csv(a.output/'extraction_intervals.csv',index=False)
    extracted=restricted/'complete_context.parquet'
    source_stat=a.archive.stat()
    if not a.analyse_only:
        starts=np.array([x.value for x,y in intervals],dtype=np.int64)
        ends=np.array([y.value for x,y in intervals],dtype=np.int64)
        parts=[];total=0;kept=0;last=None;t0=time.monotonic()
        reader=pd.read_csv(a.archive,usecols=['time','stationid','rainfall','rainfall_flag'],
                           dtype={'time':'string','stationid':'string','rainfall':'string','rainfall_flag':'string'},
                           chunksize=a.chunksize,compression='bz2')
        for i,chunk in enumerate(reader,1):
            dates=pd.to_datetime(chunk.time,format='%Y-%m-%d %H:%M:%S',errors='raise')
            assert dates.is_monotonic_increasing
            if last is not None:assert dates.iloc[0]>=last
            last=dates.iloc[-1]
            ns=dates.to_numpy(dtype='datetime64[ns]').view('i8')
            loc=np.searchsorted(starts,ns,side='right')-1
            keep=(loc>=0)&(ns<=ends[np.maximum(loc,0)])
            selected=chunk.loc[keep].copy()
            if not selected.empty:
                selected['time']=dates.loc[keep]
                selected['stationid']=pd.to_numeric(selected.stationid).astype(int)
                selected['rainfall']=pd.to_numeric(selected.rainfall,errors='coerce').astype(float)
                parts.append(selected);kept+=len(selected)
            total+=len(chunk)
            if i%5==0:print(f'scanned {total:,}; selected {kept:,}; elapsed {time.monotonic()-t0:.0f}s',flush=True)
        complete=pd.concat(parts,ignore_index=True)
        complete.to_parquet(extracted,index=False)
        extraction={'source':str(a.archive),'source_bytes':source_stat.st_size,
                    'source_mtime_ns':source_stat.st_mtime_ns,'rows_scanned':total,
                    'rows_selected':len(complete),'intervals':len(intervals),
                    'buffer_hours':a.buffer_hours,'monotonic_timestamps':True,
                    'extracted_parquet_sha256':hashlib.sha256(extracted.read_bytes()).hexdigest()}
        (a.output/'extraction.json').write_text(json.dumps(extraction,indent=2)+'\n')
        print('Archive extraction complete.',flush=True)
    else:
        extraction=json.loads((a.output/'extraction.json').read_text())
        assert extraction['source_bytes']==source_stat.st_size and extraction['source_mtime_ns']==source_stat.st_mtime_ns
        assert extraction['buffer_hours']==a.buffer_hours
        complete=pd.read_parquet(extracted)
    assert len(complete)==extraction['rows_selected']
    complete['rainfall']=pd.to_numeric(complete.rainfall,errors='coerce').astype(float)
    assert not complete.duplicated(['time','stationid']).any()
    parts=pd.read_csv(a.partitions)
    parts['stationid']=pd.to_numeric(parts.stationid.str.removeprefix('S')).astype(int)
    parts['first_time']=pd.to_datetime(parts.first_time);parts['last_time']=pd.to_datetime(parts.last_time)
    complete['year']=complete.time.dt.year
    complete=complete.merge(parts[['stationid','year','first_time','last_time']],on=['stationid','year'],validate='many_to_one')
    complete['retained']=complete.time.between(complete.first_time,complete.last_time)
    retained_members=pd.read_parquet(a.members)
    retained_members['stationid']=retained_members.stationid.astype(str).str.replace(r'^S*','S',regex=True)
    cluster=load_cluster(a.cluster_script)
    results=[];member_changes=[];availability=[];component_receipts=[]
    # Reconstruct station events/clusters once per merged local context.
    for region,(begin,end) in enumerate(intervals):
        local=complete.loc[complete.time.between(begin,end)].copy()
        timestamps=pd.date_range(begin,end,freq='15min')
        assert local.groupby('time').stationid.nunique().reindex(timestamps).eq(131).all()
        full_events=events(local)
        full_storms,full_members=cluster.build_storm_clusters(full_events,30)
        base_events=events(local.loc[local.retained])
        base_storms,base_members=cluster.build_storm_clusters(base_events,30)
        selected=primary.loc[primary.storm_id.map(lambda sid:caches[sid].time.min()>=begin and caches[sid].time.max()<=end)]
        for row in selected.itertuples(index=False):
            sid=row.storm_id;cache=caches[sid]
            start,finish=pd.Timestamp(row.storm_start),pd.Timestamp(row.storm_end)
            baseline=base_storms.loc[base_storms.storm_start.eq(start)&base_storms.storm_end.eq(finish)]
            assert len(baseline)==1,f'{sid}: retained local cluster does not reproduce window'
            bm=base_members.loc[base_members.storm_id.eq(baseline.iloc[0].storm_id)]
            expected=retained_members.loc[retained_members.storm_id.eq(sid)]
            assert signature(bm)==signature(expected),f'{sid}: retained members differ'
            fc=full_storms.loc[full_storms.storm_start.le(finish)&full_storms.storm_end.ge(start)]
            assert len(fc)==1,f'{sid}: full-context storm component ambiguous'
            full=fc.iloc[0]
            assert full.storm_start<=start and full.storm_end>=finish
            fm=full_members.loc[full_members.storm_id.eq(full.storm_id)]
            # A >=135-min clear margin closes station chains (120 min) and
            # consequently the 30-min cross-station component inside context.
            closed=full.storm_start-begin>pd.Timedelta(minutes=120) and end-full.storm_end>pd.Timedelta(minutes=120)
            assert closed,f'{sid}: expand extraction buffer to close connected component'
            fsig,bsig=signature(fm),signature(bm)
            for key in sorted(fsig-bsig):member_changes.append({'storm_id':sid,'change':'added_or_extended_full_event','stationid':key[0],'event_start':key[1],'event_end':key[2]})
            for key in sorted(bsig-fsig):member_changes.append({'storm_id':sid,'change':'retained_event_replaced','stationid':key[0],'event_start':key[1],'event_end':key[2]})
            original=cache[['time','stationid','rainfall','rainfall_flag']].set_index(['time','stationid']).sort_index()
            archive=local[['time','stationid','rainfall','rainfall_flag','retained']].set_index(['time','stationid']).reindex(original.index)
            np.testing.assert_array_equal(np.isfinite(original.rainfall),np.isfinite(archive.rainfall))
            error=np.abs(original.rainfall.to_numpy()-archive.rainfall.to_numpy())
            maxerr=float(np.nanmax(error)) if np.isfinite(error).any() else 0.
            assert maxerr<=1e-10,f'{sid}: cached rainfall differs from complete source'
            assert original.rainfall_flag.fillna('<NA>').astype(str).equals(archive.rainfall_flag.fillna('<NA>').astype(str))
            whole_cache=local.loc[local.time.between(cache.time.min(),cache.time.max())].set_index(['time','stationid'])
            absent=whole_cache.index.difference(original.index)
            absent_finite=int(np.isfinite(whole_cache.loc[absent,'rainfall']).sum())
            # Requested lead-in may precede the retained cache; inspect it too.
            target=local.loc[local.time.between(start-pd.Timedelta(minutes=120),finish)].copy()
            for stamp,g in target.groupby('time'):
                finite=np.isfinite(g.rainfall)
                availability.append({'storm_id':sid,'time':stamp,'archive_finite_stations':int(finite.sum()),
                                     'retained_finite_stations':int((finite&g.retained).sum()),
                                     'archive_rows_outside_fragment':int((~g.retained).sum()),
                                     'finite_values_outside_fragment':int((finite&~g.retained).sum())})
            outside=target.loc[~target.retained]
            changed_start=full.storm_start!=start;changed_end=full.storm_end!=finish
            record={'storm_id':sid,'region':region,'retained_start':start,'retained_end':finish,
                    'full_component_start':full.storm_start,'full_component_end':full.storm_end,
                    'start_changed':bool(changed_start),'end_changed':bool(changed_end),
                    'duration_delta_minutes':float(((full.storm_end-full.storm_start)-(finish-start)).total_seconds()/60),
                    'initiation_hour_changed':full.storm_start.hour!=start.hour,
                    'member_events_retained':len(bm),'member_events_full':len(fm),
                    'member_signature_changed':fsig!=bsig,'closed_complete_context':bool(closed),
                    'cache_values_max_abs_error':maxerr,'cache_flags_match':True,
                    'cache_absent_archive_rows':len(absent),
                    'cache_absent_finite_values':absent_finite,
                    'requested_lead_outside_fragment_rows':len(outside),
                    'requested_lead_outside_fragment_finite':int(np.isfinite(outside.rainfall).sum()),
                    'primary_high_residual':row.convective_rule_label=='convective',
                    'min_margin_after_fragment_start_minutes':float((start-target.first_time.max()).total_seconds()/60),
                    'min_margin_before_fragment_end_minutes':float((target.last_time.min()-finish).total_seconds()/60)}
            record['fragment_affected']=bool(changed_start or changed_end or fsig!=bsig or record['requested_lead_outside_fragment_finite'])
            results.append(record)
            component_receipts.append({'storm_id':sid,'component_start':full.storm_start,'component_end':full.storm_end,
                                       'context_start':begin,'context_end':end,'closed':bool(closed)})
    result=pd.DataFrame(results).sort_values('storm_id')
    assert len(result)==len(primary) and result.storm_id.is_unique
    result.to_csv(a.output/'window_integrity.csv',index=False)
    pd.DataFrame(member_changes).to_csv(a.output/'member_boundary_changes.csv',index=False)
    pd.DataFrame(availability).to_csv(a.output/'station_availability.csv',index=False)
    pd.DataFrame(component_receipts).to_csv(a.output/'component_closure.csv',index=False)
    report={'status':'PASS','fixed_windows':len(result),'context_buffer_hours':a.buffer_hours,
            'all_retained_boundaries_and_members_reproduced':True,
            'all_complete_components_closed':bool(result.closed_complete_context.all()),
            'all_retained_cache_values_and_flags_match_complete_archive':True,
            'cache_missing_finite_archive_values':result.loc[result.cache_absent_finite_values.gt(0),'storm_id'].tolist(),
            'max_cache_value_error':float(result.cache_values_max_abs_error.max()),
            'start_changed':result.loc[result.start_changed,'storm_id'].tolist(),
            'end_changed':result.loc[result.end_changed,'storm_id'].tolist(),
            'member_signatures_changed':result.loc[result.member_signature_changed,'storm_id'].tolist(),
            'initiation_hour_changed':result.loc[result.initiation_hour_changed,'storm_id'].tolist(),
            'fragment_affected':result.loc[result.fragment_affected,'storm_id'].tolist(),
            'requested_lead_values_outside_fragment':result.loc[result.requested_lead_outside_fragment_finite.gt(0),'storm_id'].tolist(),
            'full_year_catalogue_built':False,'primary_windows_or_rule_changed':False,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

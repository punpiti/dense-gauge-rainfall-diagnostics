"""Full-cohort context, distributions and descriptive monthly standardization.

All 3,304 fixed primary labels remain unchanged. Gauge intensity is measured
inside linked storm bounds; residual metrics retain their original 120-min lead.
Monthly standardization is a descriptive composition calculation, not causality.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'primary'))
from run_primary import grid
from reconstruction import reconstruct,configure
from no_advection_ablation import METRICS
SOURCE=None
OUT=None
FIG=None
HISTORICAL=None


def initialize(source,output,stations,boundary,historical):
    global SOURCE,OUT,FIG,HISTORICAL
    SOURCE=Path(source);OUT=Path(output);FIG=OUT/'figures';HISTORICAL=Path(historical)
    FIG.mkdir(parents=True,exist_ok=True);configure(stations,boundary)


THRESHOLDS=dict(zip(METRICS,[500,1000,4000,10]))


def details(row):
    cache=pd.read_parquet(SOURCE/'restricted/caches'/f"{row['storm_id']}.parquet")
    cache['time']=pd.to_datetime(cache.time)
    linked=cache.loc[cache.time.between(pd.Timestamp(row['storm_start']),pd.Timestamp(row['storm_end']))]
    assert len(linked)>0
    np.testing.assert_allclose(cache.rainfall.max(),row['max_frame_max'],rtol=0,atol=1e-12)
    rain,_,_,_=reconstruct(cache)
    rainy=np.isfinite(rain[1:])&(rain[1:]>=.1)
    valid=int((rainy.sum(axis=(1,2))>=3).sum())
    assert row['pair_max_n_strong_frames']<=valid
    return {'storm_id':row['storm_id'],'linked_max_gauge_accumulation':float(linked.rainfall.max()),
            'window_max_gauge_accumulation':float(cache.rainfall.max()),'valid_rainy_frames':valid,
            'strong_frame_fraction':float(row['pair_max_n_strong_frames']/valid) if valid else np.nan}


def hourly(data):
    palette={'Other class':'#236B9A','High-residual class':'#B77916'}
    frame=data.copy();frame['hour']=frame.storm_start.dt.hour
    frame['class']=frame.pair_max_label.map({'high_residual':'High-residual class','other':'Other class'})
    counts=frame.groupby(['hour','class']).size().reindex(pd.MultiIndex.from_product([range(24),palette],names=['hour','class']),fill_value=0).rename('storms').reset_index()
    totals=frame['class'].value_counts();counts['class_total']=counts['class'].map(totals)
    counts['within_class_percent']=100*counts.storms/counts.class_total
    assert counts.storms.sum()==len(data)
    np.testing.assert_allclose(counts.groupby('class').within_class_percent.sum(),100)
    counts.to_csv(OUT/'hourly_counts_and_percentages.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,6.5),sharex=True,layout='constrained')
    for i,(ax,y,label) in enumerate(zip(axes,['storms','within_class_percent'],['Number of storm windows','Within-class percentage (%)'])):
        sns.barplot(data=counts,x='hour',y=y,hue='class',hue_order=list(palette),palette=palette,errorbar=None,ax=ax)
        for patch in ax.containers[1]:patch.set_hatch('//');patch.set_edgecolor('#444444');patch.set_linewidth(.5)
        ax.set(ylabel=label,xlabel='Linked storm start hour (local time)' if i else '',ylim=(0,None))
        ax.text(.008,.96,f'({chr(97+i)})',transform=ax.transAxes,va='top',fontweight='bold')
        if i:ax.get_legend().remove()
        else:
            handles,_=ax.get_legend_handles_labels();ax.legend(handles,[f'{key} (n={totals[key]:,})' for key in palette],title=None,frameon=False,loc='upper left',bbox_to_anchor=(.03,1.16),ncol=2)
    fig.savefig(FIG/'hourly_distribution.png',dpi=220,bbox_inches='tight');plt.close(fig)
    return {label:g.sort_values(['storms','hour'],ascending=[False,True]).head(3)[['hour','storms','within_class_percent']].to_dict('records') for label,g in counts.groupby('class')}


def distributions(data):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');rows=[]
    for ax,(metric,title) in zip(axes.flat,METRICS.items()):
        values=np.sort(data['pair_max_'+metric].to_numpy(float));rank=np.arange(len(values))
        normalized=(values-values.min())/(values.max()-values.min());distance=np.abs(normalized-rank/(len(values)-1))
        k=int(np.argmax(distance))
        # Independent chord distance in the same normalized coordinate system.
        independent=np.abs((rank/(len(values)-1))-normalized)/np.sqrt(2)
        assert int(np.argmax(independent))==k
        rows.append({'metric':metric,'n':len(values),'p25':float(np.quantile(values,.25)),
                     'median':float(np.median(values)),'p75':float(np.quantile(values,.75)),
                     'retrospective_knee':float(values[k]),'knee_rank':k+1,'fixed_threshold':THRESHOLDS[metric]})
        sns.lineplot(x=rank+1,y=values,estimator=None,errorbar=None,sort=False,color='#236B9A',ax=ax)
        ax.axhline(THRESHOLDS[metric],color='#B77916',label=f'Fixed rule: {THRESHOLDS[metric]:,}')
        ax.axhline(values[k],color='#444444',ls='--',label=f'Retrospective knee: {values[k]:,.1f}')
        ax.set(xlabel=f'Storm rank (N={len(values):,})',ylabel=title);ax.legend(frameon=False,fontsize=9)
    fig.savefig(FIG/'figure_threshold_selection.png',dpi=220,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'threshold_distributions.csv',index=False)
    return rows


def composition(data):
    original=pd.read_csv(HISTORICAL,parse_dates=['storm_start'])
    original['month']=original.storm_start.dt.month;data=data.copy();data['month']=data.storm_start.dt.month
    data['high']=data.pair_max_label.eq('high_residual')
    monthly=data.groupby('month').agg(windows=('storm_id','size'),high_residual=('high','sum')).reindex(range(1,13))
    monthly['high_residual_fraction']=monthly.high_residual/monthly.windows
    original_counts=original.groupby('month').size().reindex(range(1,13),fill_value=0)
    monthly['historical_windows']=original_counts
    monthly['historical_month_weight']=original_counts/len(original)
    monthly['rebuilt_month_weight']=monthly.windows/len(data)
    pfull=float(data.high.mean());pold=float(original.convective_rule_label.eq('convective').mean())
    standardized=float((monthly.historical_month_weight*monthly.high_residual_fraction).sum())
    composition=pfull-standardized;within=standardized-pold
    np.testing.assert_allclose(composition+within,pfull-pold,rtol=0,atol=1e-14)
    monthly.to_csv(OUT/'monthly_class_composition.csv')
    seasons=[]
    for name,months in [('May--October',list(range(5,11))),('November--April',[11,12,1,2,3,4])]:
        g=data.loc[data.month.isin(months)];seasons.append({'calendar_block':name,'n':len(g),'high_residual':int(g.high.sum()),'fraction':float(g.high.mean())})
    return {'historical_fraction':pold,'rebuilt_fraction':pfull,'rebuilt_rates_historical_month_weights':standardized,
            'gap_percentage_points':100*(pfull-pold),'monthly_composition_contribution_pp':100*composition,
            'remaining_within_month_difference_pp':100*within,'calendar_blocks':seasons,
            'descriptive_standardization_not_causal_or_independent_validation':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stations',type=Path,required=True);parser.add_argument('--boundary',type=Path,required=True)
    parser.add_argument('--historical-window-table',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();initialize(args.input,args.output,args.stations,args.boundary,args.historical_window_table)
    sns.set_theme(style='ticks',context='paper',font_scale=1.1)
    data=pd.read_csv(SOURCE/'results.csv',parse_dates=['storm_start','storm_end']);data=data.loc[data.status.eq('done')].copy()
    assert len(data)>0 and data.storm_id.is_unique
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.input,args.output,args.stations,args.boundary,args.historical_window_table)) as pool:
        detail=pd.DataFrame(pool.map(details,data.to_dict('records')))
    data=data.merge(detail,on='storm_id',validate='one_to_one');data.to_csv(OUT/'context_per_window.csv',index=False)
    classes={}
    for label,g in data.groupby('pair_max_label'):
        fields=['storm_duration_minutes','linked_max_gauge_accumulation','window_max_gauge_accumulation','strong_frame_fraction',*['pair_max_'+k for k in METRICS]]
        classes[label]={'n':len(g),**{field:{'p25':float(g[field].quantile(.25)),'median':float(g[field].median()),'p75':float(g[field].quantile(.75)),'p90':float(g[field].quantile(.9))} for field in fields}}
    result={'n':len(data),'classes':classes,'hourly_peaks':hourly(data),'threshold_distributions':distributions(data),'composition':composition(data)}
    maysep=data.loc[data.storm_start.dt.month.between(5,9)].copy();maysep['hour']=maysep.storm_start.dt.hour
    counts=maysep.groupby(['pair_max_label','hour']).size().rename('windows').reset_index();counts.to_csv(OUT/'may_september_hourly.csv',index=False)
    result['may_september_n']=len(maysep)
    result['may_september_peaks']={k:g.sort_values(['windows','hour'],ascending=[False,True]).head(3).to_dict('records') for k,g in counts.groupby('pair_max_label')}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

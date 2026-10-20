"""Full-cohort motion/reference comparison, preserving the original benchmark."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'primary'))
from run_primary import grid
from no_advection_ablation import METRICS
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
data=pd.read_csv(args.input);data=data.loc[data.status.eq('done')].copy();assert len(data)>0
rows=[]
for scaling in ['pair_max','fixed15_trunc','fixed15_round','log1p_fixed']:
    correlations=[float(spearmanr(data[f'{scaling}_{m}'],data[f'no_advection_{m}']).statistic) for m in METRICS]
    rows.append({'scaling':scaling,'n':len(data),'high_residual':int(data[f'{scaling}_label'].eq('high_residual').sum()),
                 'changed_from_primary':int(data[f'{scaling}_label'].ne(data.pair_max_label).sum()),
                 'no_advection_agreement':int(data[f'{scaling}_label'].eq(data.no_advection_label).sum()),
                 'spearman_min':min(correlations),'spearman_max':max(correlations)})
pd.DataFrame(rows).to_csv(OUT/'motion_grouping.csv',index=False)
a=data.pair_max_label.eq('high_residual');b=data.no_advection_label.eq('high_residual')
observed=float((a==b).mean());expected=float(a.mean()*b.mean()+(1-a.mean())*(1-b.mean()))
result={'n':len(data),'agreement':int((a==b).sum()),'kappa':(observed-expected)/(1-expected),
        'other_to_high':int((~a&b).sum()),'high_to_other':int((a&~b).sum()),
        'metrics':{m:{'pair_max_median':float(data['pair_max_'+m].median()),
                      'no_advection_median':float(data['no_advection_'+m].median()),
                      'spearman':float(spearmanr(data['pair_max_'+m],data['no_advection_'+m]).statistic)} for m in METRICS},
        'scalings':rows}
(OUT/'no_advection_summary.json').write_text(json.dumps(result,indent=2)+'\n')
sns.set_theme(style='ticks',context='paper',font_scale=1.1)
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
data['class']=data.pair_max_label.map({'high_residual':'High-residual class','other':'Other class'})
palette={'Other class':'#236B9A','High-residual class':'#B77916'}
for ax,(metric,title) in zip(axes.flat,METRICS.items()):
    sns.scatterplot(data=data,x='pair_max_'+metric,y='no_advection_'+metric,hue='class',style='class',hue_order=list(palette),style_order=list(palette),palette=palette,markers={'Other class':'o','High-residual class':'^'},s=10,alpha=.45,linewidth=0,legend=ax is axes[0,0],ax=ax)
    lower=min(float(data['pair_max_'+metric].min()),float(data['no_advection_'+metric].min()));upper=max(float(data['pair_max_'+metric].max()),float(data['no_advection_'+metric].max()))
    ax.plot([lower,upper],[lower,upper],ls='--',color='#777',lw=1)
    ax.set(title=title.replace('\n',' '),xlabel='Pair-max transport correction',ylabel='No advection')
fig.savefig(OUT/'fig_no_advection_ablation.png',dpi=220);plt.close(fig)
print(json.dumps(result,indent=2))

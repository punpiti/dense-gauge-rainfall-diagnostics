"""Figure from completed year-round results, with explicit denominators."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--rebuild-summary',type=Path,required=True)
parser.add_argument('--fixed-summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
summary=json.loads(args.rebuild_summary.read_text());n=summary['passed_support']
sns.set_theme(style='ticks',context='paper',font_scale=1.1)
fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
months=pd.DataFrame({'month':list(range(1,13)), 'windows':[summary['by_start_month'].get(str(m),0) for m in range(1,13)]})
sns.barplot(data=months,x='month',y='windows',color='#497b96',errorbar=None,ax=axes[0])
axes[0].set(xlabel='Linked storm start month',ylabel='Support-qualified windows',title=f'(a) Complete-source evaluation (n = {n:,})')
records=[]
for row in summary['scale_controls']:
    records.append({'sigma_km':row['sigma_km'],'high_residual_percent':100*row['high_residual']/n,'sample':r'$3\sigma$ cutoff'})
fixed=pd.read_csv(args.fixed_summary)
for row in fixed.loc[fixed.cutoff_policy.eq('fixed_7p5km')].itertuples(index=False):
    records.append({'sigma_km':row.sigma_km,'high_residual_percent':100*row.high_residual/n,'sample':'Fixed 7.5-km cutoff'})
frame=pd.DataFrame(records);frame.to_csv(OUT/'scale_plot_rows.csv',index=False)
sns.lineplot(data=frame,x='sigma_km',y='high_residual_percent',hue='sample',style='sample',markers=True,dashes=False,errorbar=None,palette=['#ba6038','#245b80'],ax=axes[1])
axes[1].set(xlabel=r'Gaussian bandwidth $\sigma$ (km)',ylabel='High-residual windows (%)',title=f'(b) Support control (n = {n:,})',xticks=[1,1.5,2.5,3.5,5])
axes[1].legend(title=None,fontsize=8,loc='upper left')
sns.despine(fig=fig)
for ext in ['pdf','png']:fig.savefig(OUT/f'fig_complete_archive.{ext}',dpi=240)
print(f'Figure complete: {n} support-qualified windows.')

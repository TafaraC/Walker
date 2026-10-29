"""Regenerate presentation figures from saved results; no simulator required."""
from pathlib import Path
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/walker-presentation-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
ARCHIVE = ROOT / 'experiments/archive_20260905_pre_variants/gail_dagger'
def read(path):
    with path.open() as f:
        return {r['policy_label']: r for r in csv.DictReader(f)}
old = read(ARCHIVE / 'evaluation/comparison.csv')
old['gail_dagger'] = old.get('gail_dagger', old.get('gail'))
new = read(ROOT / 'evaluation/comparison.csv')
names = {'bc':'BC', 'dagger':'DAgger', 'dagger_mixed':'Mixed DAgger', 'gail_dagger':'GAIL-DAgger', 'gail_bc':'GAIL-BC', 'gail_random':'GAIL-random', 'oracle':'FC Portugal'}
colors = {'bc':'#2676b8','dagger':'#e18a24','dagger_mixed':'#2d9872','gail_dagger':'#9454a5','gail_bc':'#39a9bb','gail_random':'#c34e55','oracle':'#667085'}
plt.rcParams.update({'font.size':13,'axes.titlesize':16,'axes.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
def save(fig, name):
    fig.tight_layout(pad=1.6)
    for ext in ['png','pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)
def bars(ax, data, keys, field, title, ylabel, log=False):
    vals = [float(data[k][field]) for k in keys]
    ax.bar(range(len(keys)),vals,color=[colors[k] for k in keys],width=.62)
    ax.set_xticks(range(len(keys)),[names[k] for k in keys])
    ax.set_title(title); ax.set_ylabel(ylabel)
    if log: ax.set_yscale('log'); ax.set_ylim(min(vals)*.45,max(vals)*4)
    else: ax.set_ylim(min(0,min(vals)*1.3),max(vals)*1.3)
    for i,v in enumerate(vals): ax.annotate(f'{v:.4g}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=11)
    ax.grid(axis='y',alpha=.17); ax.set_axisbelow(True)
def returns(data, keys, name, title):
    fig, ax = plt.subplots(figsize=(11,4.5))
    for i,k in enumerate(keys):
        r=data[k]; m=float(r['normalized_return_mean']); lo=float(r['normalized_return_ci95_low']); hi=float(r['normalized_return_ci95_high'])
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[k],capsize=5,markersize=9)
    ax.axvline(1,color='#667085',ls='--',label='Oracle parity'); ax.set_yticks(range(len(keys)),[names[k] for k in keys]); ax.invert_yaxis()
    ax.set_xlabel('Normalized target-progress return (mean and approximate 95% CI)'); ax.set_title(title); ax.legend(loc='best'); ax.grid(axis='x',alpha=.2)
    save(fig,name)
returns(old,['bc','dagger','gail_dagger'],'01_original_return','Original evaluation: all three policies near oracle parity')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,f,t in zip(axs,['expert_test_action_mse','on_policy_oracle_action_mse_mean'],['Held-out expert states','States visited by each policy']): bars(ax,old,['bc','dagger','gail_dagger'],f,t,'16-D latent-action MSE (log scale)',True)
save(fig,'02_original_fidelity')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,f,t in zip(axs,['expert_test_action_mse','on_policy_oracle_action_mse_mean'],['Offline error: 78.5% reduction','On-policy error: 95.0% reduction']): bars(ax,new,['bc','dagger','dagger_mixed'],f,t,'16-D latent-action MSE (log scale)',True)
save(fig,'03_mixed_dagger')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
keys=['gail_dagger','gail_bc','gail_random']
bars(axs[0],new,keys,'expert_test_action_mse','GAIL inherits the initialization gap','Held-out action MSE (log scale)',True)
bars(axs[1],new,keys,'mean_forward_velocity_mps_mean','Random initialization fails to walk forward','Torso-forward velocity (m/s)')
save(fig,'04_gail_initialization')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for key,path in [('gail_dagger',ARCHIVE/'artifacts'),('gail_bc',ROOT/'artifacts/gail_bc'),('gail_random',ROOT/'artifacts/gail_random')]:
    files=sorted(path.glob('*rollout_metrics_iter_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
    rows=[json.loads(p.read_text()) for p in files]
    for ax,field,title in zip(axs,['targets_reached','falls'],['Targets reached per 4,000-step rollout','Falls per 4,000-step rollout']):
        ax.plot([r['iteration'] for r in rows],[r[field] for r in rows],marker='o',label=names[key],color=colors[key]); ax.set_title(title); ax.set_xlabel('Rollout index (before update)'); ax.set_xticks([1,3,5,7,10]); ax.grid(alpha=.2)
axs[0].legend(); save(fig,'05_gail_training')
returns(new,['bc','dagger','dagger_mixed','gail_dagger','gail_bc','gail_random'],'06_expanded_return','Expanded evaluation: five viable policies, one failed initialization')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
keys=['bc','gail_bc','gail_random']
bars(axs[0],new,keys,'return_target_progress_m_mean','Progress exposes the stationary failure','Target progress in 60 active seconds (m)')
bars(axs[1],new,keys,'jerk_cost_rate_rad2_per_s6_mean','Low jerk alone gives the wrong ranking','Jerk cost rate (rad²/s⁶)')
save(fig,'07_stationary_failure')
print('Generated', len(list(OUT.glob('*.png'))), 'figures, each in PNG and PDF.')

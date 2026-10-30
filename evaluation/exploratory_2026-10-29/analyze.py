"""Read-only EDA of completed Walker research. Run with the fcportugal Python.

No simulation, checkpoint selection, fitting of policies/probes, or input mutation.
Requires numpy, pandas, scipy, matplotlib. Outputs adjacent to this script.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
os.environ.setdefault('MPLCONFIGDIR', '/tmp/walker-eda-mpl')
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)
SOURCES = {
 'parent':'experiments/consolidated/v1/evaluation/all_all_correct/comparison.json',
 'scratch':'experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json',
 'controls':'experiments/submission_controls/v1/evaluation/summary/comparison.json',
 'probes':'experiments/consolidated/v1/scratch_v1/probing/report.json',
 'curves':'experiments/submission_controls/v1/curves/curves.json',
}
D = {k:json.loads((ROOT/p).read_text()) for k,p in SOURCES.items()}
METRICS = ['return_target_progress_m','mean_forward_velocity_mps','mean_planar_speed_mps',
 'mean_target_direction_error','targets_reached','falls','on_policy_latent_action_mse',
 'normalized_return','joint_occupancy_sliced_wasserstein','jerk_cost_rate_rad2_per_s6',
 'torso_roll_rms_deg','torso_pitch_rms_deg']
ALG=['bc','dagger_mixed','gail_bc','gail_dagger_mixed']
ANAME={'bc':'BC','dagger_mixed':'DAgger','gail_bc':'GAIL–BC','gail_dagger_mixed':'GAIL–DAgger','gail_random':'Random-start GAIL'}
SHORT={'oracle':'Oracle','random':'Uniform random','zero_native':'Zero native',
 'bc_original_matched':'BC matched updates',
 **{f'original_hidden_42_{a}':f'Frozen {ANAME[a]}' for a in ALG},
 'scratch_original_hidden_42_gail_random':'Original random-start GAIL'}
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#777777']
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'})
manifest={'sources':{k:{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
                     for k,p in SOURCES.items()},'trace_sources':[],
 'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
 'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
 'bootstrap_seed':20261029,'bootstrap_resamples':5000}
lookup={}; duplicates=0
for source in ['parent','scratch','controls']:
 agg={(x['suite'],x['policy']):x for x in D[source]['aggregates']}
 for r in D[source]['runs']:
  key=(r['row']['id'],r['controller']['label'])
  if key in lookup:
   duplicates+=1
   old=lookup[key]
   assert old['row']==r['row'],key
   assert old['controller'].get('sha256')==r['controller'].get('sha256'),key
   for m in METRICS:
    if old.get(m) is not None and r.get(m) is not None:
     assert np.isclose(old[m],r[m],rtol=1e-9,atol=1e-9),(key,m)
  else:
   lookup[key]=dict(r,first_source=source)
  a=agg.get((r['row']['suite'],key[1]),{})
  offline=r.get('held_out_expert_action_mse',a.get('held_out_expert_action_mse'))
  if offline is not None:
   old_off=lookup[key].get('held_out_expert_action_mse')
   if old_off is not None: assert np.isclose(old_off,offline,rtol=1e-7),key
   lookup[key]['held_out_expert_action_mse']=offline
rows=[]
for r in lookup.values():
 c=r['controller']; row=r['row']; label=c['label']; algorithm=c.get('algorithm',label)
 cohort=('original' if c['domain']=='original' else 'baseline' if c['domain']=='baseline'
         else 'scratch' if label.startswith('scratch_') else 'continued')
 if label.startswith('bc_varied_'): cohort='scratch200';algorithm='bc'
 if label=='bc_original_matched':algorithm='bc_matched'
 arch=c['architecture']
 family=SHORT.get(label,f'{cohort.title()} {arch} {ANAME.get(algorithm,algorithm)}')
 rows.append({**{m:r.get(m) for m in METRICS},'held_out_expert_action_mse':r.get('held_out_expert_action_mse'),
  'round':row['id'],'suite':row['suite'],'slip':row['slip'],'parent_mean':row.get('parent_mean'),
  'label':label,'seed':c.get('seed'),'cohort':cohort,'arch':arch,'algorithm':algorithm,
  'family':family,'active_seconds':r['active_seconds'],'first_source':r['first_source'],
  'checkpoint_sha256':c.get('sha256')})
df=pd.DataFrame(rows)
assert len(df)==2223,len(df)
assert (df.active_seconds==60).all()
assert not df.duplicated(['round','label']).any()
oracle=df[df.label=='oracle'].set_index('round').return_target_progress_m
random=df[df.label=='random'].set_index('round').return_target_progress_m
df['progress_minus_oracle_m']=df.return_target_progress_m-df['round'].map(oracle)
assert np.allclose(df.normalized_return,
 (df.return_target_progress_m-df['round'].map(random))/(df['round'].map(oracle)-df['round'].map(random)))
df.to_csv(OUT/'deployments.csv',index=False)
s=df[df.suite.isin(['grouped','confirmation'])].copy()
assert (s.groupby('label').size()==30).all()
coverage=df.groupby(['family','suite']).agg(deployments=('label','size'),
 checkpoints=('label','nunique'),active_minutes=('active_seconds',lambda x:x.sum()/60)).reset_index()
coverage.to_csv(OUT/'coverage.csv',index=False)

# Tail summaries on individual checkpoints. SD across deployment rounds is not training-seed uncertainty.
summ=[]
for label,g in s.groupby('label'):
 v=g.return_target_progress_m;delta=g.progress_minus_oracle_m
 summ.append(dict(label=label,family=g.family.iloc[0],seed=g.seed.iloc[0],cohort=g.cohort.iloc[0],
  arch=g.arch.iloc[0],algorithm=g.algorithm.iloc[0],n=len(g),mean_progress=v.mean(),
  sd_progress=v.std(ddof=1),median_progress=v.median(),q25_progress=v.quantile(.25),
  q75_progress=v.quantile(.75),min_progress=v.min(),mean_worst_3_progress=v.nsmallest(3).mean(),
  mean_oracle_delta=delta.mean(),sd_oracle_delta=delta.std(ddof=1),
  mean_worst_3_oracle_delta=delta.nsmallest(3).mean(),mean_best_3_oracle_delta=delta.nlargest(3).mean(),
  rounds_below_oracle_2m=(delta < -2).sum(),total_falls=g.falls.sum(),fall_rounds=(g.falls>0).sum(),
  no_target_rounds=(g.targets_reached==0).sum(),mean_targets=g.targets_reached.mean(),
  mse=g.held_out_expert_action_mse.mean(),onpolicy=g.on_policy_latent_action_mse.mean(),
  swd=g.joint_occupancy_sliced_wasserstein.mean(),jerk=g.jerk_cost_rate_rad2_per_s6.mean(),
  velocity=g.mean_forward_velocity_mps.mean(),direction=g.mean_target_direction_error.mean()))
sm=pd.DataFrame(summ);sm.to_csv(OUT/'checkpoint_summary.csv',index=False)
fs=sm.groupby('family').agg(checkpoints=('label','size'),mean_progress=('mean_progress','mean'),
 training_seed_sd_of_mean=('mean_progress','std'),mean_within_checkpoint_round_sd=('sd_progress','mean'),
 mean_oracle_delta=('mean_oracle_delta','mean'),mean_worst_3_oracle_delta=('mean_worst_3_oracle_delta','mean'),
 total_falls=('total_falls','sum'),fall_rounds=('fall_rounds','sum'),n=('n','sum'),
 mse=('mse','mean'),onpolicy=('onpolicy','mean'),swd=('swd','mean'),jerk=('jerk','mean')).reset_index()
fs.to_csv(OUT/'family_summary.csv',index=False)

# Paired effects, average training replicas within each round before resampling rounds.
# Resample within suite x parent group, preserving the designed mixture; not training-population CIs.
rng=np.random.default_rng(20261029)
effects=[]
def compare(name, left, right, metric='return_target_progress_m'):
 l=s[s.label.isin(left)].groupby('round')[metric].mean()
 r=s[s.label.isin(right)].groupby('round')[metric].mean()
 z=(l-r).dropna(); assert len(z)==30,(name,len(z))
 meta=s[['round','suite','parent_mean']].drop_duplicates().set_index('round').loc[z.index]
 draws=np.zeros(5000)
 for _,idx in meta.groupby(['suite','parent_mean']).groups.items():
  v=z.loc[idx].to_numpy();draws+=rng.choice(v,size=(5000,len(v)),replace=True).sum(axis=1)/len(z)
 lo,hi=np.quantile(draws,[.025,.975])
 item=dict(comparison=name,metric=metric,left=';'.join(left),right=';'.join(right),n_rounds=len(z),
  mean=z.mean(),median=z.median(),sd=z.std(ddof=1),q25=z.quantile(.25),q75=z.quantile(.75),
  wins=(z>0).sum(),ci_low=lo,ci_high=hi,
  grouped=z.loc[meta.suite=='grouped'].mean(),confirmation=z.loc[meta.suite=='confirmation'].mean())
 effects.append(item)
 return z

def labels(cohort,arch,algorithm):
 return sm[(sm.cohort==cohort)&(sm.arch==arch)&(sm.algorithm==algorithm)].label.tolist()
for a in ALG:
 for c in ['continued','scratch']:
  for m in ['return_target_progress_m','falls','on_policy_latent_action_mse']:
   compare(f'Context | {c} | {ANAME[a]}',labels(c,'input',a),labels(c,'hidden',a),m)
 for arch in ['hidden','input']:
  compare(f'History | {arch} | {ANAME[a]}',labels('continued',arch,a),labels('scratch',arch,a))
for c in ['continued','scratch']:
 for arch in ['hidden','input']:
  for a,b in [('gail_bc','bc'),('gail_dagger_mixed','dagger_mixed')]:
   for m in ['return_target_progress_m','falls','on_policy_latent_action_mse']:
    compare(f'GAIL refinement | {c} {arch} | {ANAME[a]}',labels(c,arch,a),labels(c,arch,b),m)
for m in ['return_target_progress_m','falls','on_policy_latent_action_mse','joint_occupancy_sliced_wasserstein']:
 compare('Matched BC minus frozen DAgger',['bc_original_matched'],['original_hidden_42_dagger_mixed'],m)
 compare('Matched BC minus old BC',['bc_original_matched'],['original_hidden_42_bc'],m)
for arch in ['hidden','input']:
 compare(f'BC 200 vs 100 | {arch}',labels('scratch200',arch,'bc'),labels('scratch',arch,'bc'))
pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv',index=False)

# Balanced two-way descriptive sum-of-squares decomposition, 30 rounds x trained controllers.
# Interaction contains policy-by-round differences AND simulator variation (no exact repeats).
walk=s[(s.algorithm.isin(ALG+['bc_matched'])) & (s.cohort!='baseline')]
variance=[]
for metric in ['return_target_progress_m','mean_forward_velocity_mps','mean_target_direction_error',
               'on_policy_latent_action_mse','joint_occupancy_sliced_wasserstein','jerk_cost_rate_rad2_per_s6']:
 mat=walk.pivot(index='round',columns='label',values=metric).to_numpy()
 assert np.isfinite(mat).all()
 mean=mat.mean();rm=mat.mean(axis=1,keepdims=True);pm=mat.mean(axis=0,keepdims=True)
 ss=np.array([mat.shape[1]*((rm-mean)**2).sum(),mat.shape[0]*((pm-mean)**2).sum(),((mat-rm-pm+mean)**2).sum()])
 assert np.isclose(ss.sum(),((mat-mean)**2).sum())
 variance.append(dict(metric=metric,n_policies=mat.shape[1],round_fraction=ss[0]/ss.sum(),
  policy_fraction=ss[1]/ss.sum(),interaction_fraction=ss[2]/ss.sum(),
  total_variance_population=mat.var()))
pd.DataFrame(variance).to_csv(OUT/'variance_decomposition.csv',index=False)
# Family-balanced sensitivity: GAIL variants and replicas must not silently dominate
# the shared-round conclusion just because more checkpoints were evaluated.
sens=[]
for a in ALG:
 mat=walk[walk.algorithm==a].pivot(index='round',columns='label',values='return_target_progress_m').to_numpy()
 sens.append(dict(algorithm=a,n_policies=mat.shape[1],round_fraction=
                  mat.shape[1]*((mat.mean(axis=1)-mat.mean())**2).sum()/((mat-mat.mean())**2).sum()))
pd.DataFrame(sens).to_csv(OUT/'variance_sensitivity.csv',index=False)
walk.groupby('parent_mean').agg(deployments=('label','size'),falls=('falls','sum'),
 fall_runs=('falls',lambda x:(x>0).sum()),progress=('return_target_progress_m','mean')).to_csv(OUT/'surface_groups.csv')

# Correlation as descriptive structure, no p-values across dependent replicas/deployments.
fields=['return_target_progress_m','mean_forward_velocity_mps','mean_target_direction_error',
 'on_policy_latent_action_mse','joint_occupancy_sliced_wasserstein','falls','jerk_cost_rate_rad2_per_s6']
rawcorr=walk[fields].corr(method='spearman')
residual=walk[fields]-walk.groupby('round')[fields].transform('mean')
rescorr=residual.corr(method='spearman')
rawcorr.to_csv(OUT/'correlation_raw.csv');rescorr.to_csv(OUT/'correlation_round_centered.csv')

# Trace-level checks and error concentration. Eight fixed models, all 30 sampled rounds.
selected=['oracle','original_hidden_42_bc','original_hidden_42_dagger_mixed','bc_original_matched',
 'varied_hidden_42_bc','scratch_varied_hidden_42_bc','zero_native','scratch_original_hidden_42_gail_random']
trace_rows=[]
trace_roots=[ROOT/'experiments/consolidated/v1/evaluation/all_all_correct/runs',
 ROOT/'experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/runs',
 ROOT/'experiments/submission_controls/v1/evaluation/runs']
for label in selected:
 for _,r in df[df.label==label].iterrows():
  paths=[p/r['round']/label/'trace.npz' for p in trace_roots]
  p=next((p for p in paths if p.exists()),None);assert p is not None,(label,r['round'])
  manifest['trace_sources'].append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  with np.load(p,allow_pickle=False) as t:
   assert len(t['actions'])==3000
   error=np.square(t['actions'].astype(float)-t['oracle_actions'].astype(float)).mean(axis=1)
   assert np.isclose(error.mean(),r.on_policy_latent_action_mse,rtol=2e-6,atol=1e-9)
   assert np.isclose(t['target_progress'].sum(),r.return_target_progress_m,rtol=1e-6,atol=1e-6)
   path=float(t['planar_speed'].sum()*.02)
   # Localization is vision-updated at 40 ms, despite 20 ms action/trace steps.
   # Diagnostic windows exclude resets and changes in the current target.
   ix=np.arange(0,len(t['positions'])-2,2)
   valid=(t['episode_id'][ix]==t['episode_id'][ix+2]) & np.all(t['targets'][ix]==t['targets'][ix+2],axis=1)
   ix=ix[valid]
   movement=t['positions'][ix+2]-t['positions'][ix]
   toward=t['targets'][ix]-t['positions'][ix]
   mn=np.linalg.norm(movement,axis=1);dn=np.linalg.norm(toward,axis=1)
   defined=(mn>1e-8)&(dn>1e-8)
   direction40=np.linalg.norm(movement[defined]/mn[defined,None]-toward[defined]/dn[defined,None],axis=1)
   trace_rows.append(dict(label=label,round=r['round'],suite=r.suite,slip=r.slip,falls=r.falls,
    zero_recorded_speed_fraction=float(np.mean(t['planar_speed']==0)),
    legacy_direction_error=float(t['target_direction_error'].mean()),
    direction_error_40ms_defined=float(direction40.mean()),
    valid_40ms_windows=int(len(ix)),defined_direction_40ms_windows=int(defined.sum()),
    low_speed_40ms_fraction=float(np.mean(mn/.04<.1)),
    low_speed_40ms_fraction_005=float(np.mean(mn/.04<.05)),
    low_speed_40ms_fraction_02=float(np.mean(mn/.04<.2)),
    low_speed_fraction=float(np.mean(t['planar_speed']<.1)),
    low_speed_fraction_005=float(np.mean(t['planar_speed']<.05)),
    low_speed_fraction_02=float(np.mean(t['planar_speed']<.2)),
    negative_progress_fraction=float(np.mean(t['target_progress']<0)),
    path_length_m=path,progress_per_path_length=r.return_target_progress_m/path if path else np.nan,
    action_mse=error.mean(),median_frame_mse=np.median(error),q95_frame_mse=np.quantile(error,.95),
    top1pct_error_share=np.sort(error)[-30:].sum()/error.sum() if error.sum() else np.nan,
    top10pct_error_share=np.sort(error)[-300:].sum()/error.sum() if error.sum() else np.nan,
    progress=r.return_target_progress_m))
td=pd.DataFrame(trace_rows);td.to_csv(OUT/'trace_summary.csv',index=False)
td.groupby(['suite','label']).mean(numeric_only=True).to_csv(OUT/'trace_means_by_suite.csv')
td=td[td.suite.isin(['grouped','confirmation'])]
td.groupby('label').mean(numeric_only=True).to_csv(OUT/'trace_means.csv')

# Full probe-target inventory, compare both hidden layers with architecture-matched random features.
pr=pd.DataFrame(D['probes']['results']);pr=pr[pr.labels=='true'].copy()
probe_rows=[]
for _,r in pr.iterrows():
 rep=r.representation
 if not rep.endswith(('_h1','_h2')) or rep.startswith(('random_','expert_')): continue
 layer=rep[-2:];base=rep[:-3];base=base.removesuffix('_correct')
 # Probing labels omit the correct-context suffix on input-conditioned policies.
 arch='input' if '_input_' in base else 'hidden'
 random_reps=[f'random_{arch}_{seed}_{layer}' for seed in [42,43,44]]
 rand=pr[(pr.representation.isin(random_reps))&(pr.target==r.target)].r2
 raw=pr[(pr.representation=='raw_observation')&(pr.target==r.target)].r2
 assert len(rand)==3 and len(raw)==1,(rep,r.target)
 probe_rows.append(dict(representation=rep,layer=layer,target=r.target,r2=r.r2,
  random_mean_r2=rand.mean(),random_max_r2=rand.max(),raw_r2=raw.iloc[0],
  gain_over_random_mean=r.r2-rand.mean(),gain_over_raw=r.r2-raw.iloc[0]))
pp=pd.DataFrame(probe_rows);pp.to_csv(OUT/'probe_target_comparisons.csv',index=False)

# Figures: also collect all panels into a portable atlas PDF.
atlas=PdfPages(OUT/'EDA_FIGURES.pdf')
figures=[]
def save(fig,name,caption):
 fig.text(.5,.018,caption,ha='center',va='bottom',fontsize=8,color='#444',wrap=True)
 fig.tight_layout(rect=(0,.075,1,1))
 fig.savefig(FIG/f'{name}.png',dpi=180);fig.savefig(FIG/f'{name}.pdf')
 atlas.savefig(fig);plt.close(fig);figures.append(dict(name=name,caption=caption))

# 1: empirical distributions: same five selected policies, identical rounds.
keys=['oracle','original_hidden_42_bc','original_hidden_42_dagger_mixed','bc_original_matched']
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for i,key in enumerate(keys):
 vals=np.sort(s[s.label==key].return_target_progress_m)
 axs[0].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=SHORT[key],color=COLORS[i])
axs[0].set(xlabel='Target progress (m / 60 active seconds)',ylabel='Fraction of rounds at or below value',title='Progress distributions overlap substantially')
axs[0].legend(fontsize=8)
vals=[s[s.label==key].progress_minus_oracle_m for key in keys[1:]]
axs[1].boxplot(vals,tick_labels=[SHORT[k].replace(' ','\n',1) for k in keys[1:]],showmeans=True)
for i,v in enumerate(vals):axs[1].scatter(np.full(len(v),i+1)+np.linspace(-.1,.1,len(v)),v,s=12,alpha=.55,color=COLORS[i+1])
axs[1].axhline(0,color='gray',ls='--');axs[1].set(ylabel='Paired progress minus oracle (m)',title='Similar averages can hide different bad rounds')
save(fig,'01_progress_distributions','30 shared surface/task rounds per model; whiskers = 1.5 IQR; dots = rounds; triangle = mean. No training-seed CI.')

# 2: shared round structure and slip response, actual slip logarithmic.
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for key in keys:
 g=s[s.label==key].sort_values('slip');axs[0].scatter(g.slip,g.return_target_progress_m,s=23,alpha=.65,label=SHORT[key])
axs[0].set(xscale='log',xlabel='Actual fixed slip',ylabel='Target progress (m)',title='Much variation is shared across controllers');axs[0].legend(fontsize=8)
v=pd.DataFrame(variance);vnames=['Progress','Forward velocity','Direction error','On-policy MSE','Joint SWD','Jerk rate']
left=np.zeros(len(v))
for col,name,color in [('round_fraction','Shared round',COLORS[0]),('policy_fraction','Controller mean',COLORS[1]),('interaction_fraction','Residual / interaction',COLORS[3])]:
 axs[1].barh(vnames,v[col]*100,left=left,color=color,label=name);left+=v[col]*100
axs[1].set(xlabel='Share of descriptive sum of squares (%)',title='Different metrics vary for different reasons');axs[1].invert_yaxis();axs[1].legend(fontsize=8,loc='lower right')
save(fig,'02_shared_variation','Balanced panel of walking controllers; round includes surface AND task. Residual includes simulator variation; this is not causal ANOVA.')

# 3 mean-tail plot, every family including unproductive controls (separate panel).
fig,axs=plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={'width_ratios':[1.5,1]})
for c,col in zip(['original','continued','scratch','scratch200'],COLORS):
 g=sm[(sm.cohort==c)&sm.algorithm.isin(ALG+['bc_matched'])]
 axs[0].scatter(g.mean_oracle_delta,g.mean_worst_3_oracle_delta,s=42,alpha=.7,label=c,color=col)
for key in keys[1:]:
 r=sm[sm.label==key].iloc[0];axs[0].annotate(SHORT[key],(r.mean_oracle_delta,r.mean_worst_3_oracle_delta),fontsize=8,xytext=(5,3),textcoords='offset points')
axs[0].axvline(0,color='gray',ls=':');axs[0].set(xlabel='Mean progress minus oracle (m)',ylabel='Mean of 3 worst paired deficits (m)',title='Average progress and lower-tail reliability');axs[0].legend(fontsize=8)
g=fs[fs.family.isin([SHORT[k] for k in keys]+['Zero native','Uniform random','Original random-start GAIL'])]
axs[1].scatter(g.mean_progress,g.total_falls/g.n,s=55,color=COLORS[0])
annotation_y={'Oracle':.08,'Frozen BC':.48,'Frozen DAgger':.23,'BC matched updates':.015,
              'Zero native':.55,'Original random-start GAIL':.28,'Uniform random':23.5}
for _,r in g.iterrows():
 axs[1].annotate(r.family,(r.mean_progress,r.total_falls/r.n),fontsize=8,
                 xytext=(35 if r.mean_progress>20 else 2,annotation_y[r.family]),
                 arrowprops={'arrowstyle':'-','color':'#999','lw':.6})
axs[1].set_yscale('symlog',linthresh=.1);axs[1].set_xlim(-2,53)
axs[1].set(xlabel='Mean target progress (m)',ylabel='Falls / active minute',title='Few falls alone do not establish locomotion')
save(fig,'03_mean_tail_stability','Left: 30 rounds/checkpoint; tails are descriptive order statistics. Right: same rounds, symmetric-log fall axis (linear below 0.1).')

# 4 separate within-model and between-training-seed consistency.
seedf=fs[fs.checkpoints==3].copy().sort_values('mean_within_checkpoint_round_sd')
fig,axs=plt.subplots(1,2,figsize=(12,7))
y=np.arange(len(seedf))
for ax,field,title in zip(axs,['mean_within_checkpoint_round_sd','training_seed_sd_of_mean'],['Variation across 30 rounds','Variation of means across 3 training replicas']):
 ax.barh(y,seedf[field],color=COLORS[0]);ax.set_yticks(y,seedf.family if ax==axs[0] else []);ax.invert_yaxis();ax.set(xlabel='Standard deviation of progress (m)',title=title)
save(fig,'04_two_sources_of_variation','These standard deviations describe different quantities. Continued replicas share an original parent; n=3 gives a weak estimate of training variability.')

# 5 context: paired effect intervals, all algorithm/history combinations.
e=pd.DataFrame(effects);g=e[(e.comparison.str.startswith('Context'))&(e.metric=='return_target_progress_m')]
fig,axs=plt.subplots(1,2,figsize=(12,5))
y=np.arange(len(g));axs[0].errorbar(g['mean'],y,xerr=np.array([g['mean']-g.ci_low,g.ci_high-g['mean']]),fmt='o',color=COLORS[0],capsize=3)
axs[0].scatter(g.grouped,y,marker='s',s=25,label='First 15');axs[0].scatter(g.confirmation,y,marker='^',s=25,label='Confirmation 15')
axs[0].set_yticks(y,g.comparison.str.replace('Context | ','',regex=False));axs[0].invert_yaxis();axs[0].axvline(0,color='gray',ls='--');axs[0].set(xlabel='Input-conditioned minus hidden progress (m)',title='Context effects depend on history and algorithm');axs[0].legend(fontsize=8)
g2=e[(e.comparison.str.startswith('GAIL refinement'))&(e.metric=='return_target_progress_m')]
y=np.arange(len(g2));axs[1].errorbar(g2['mean'],y,xerr=np.array([g2['mean']-g2.ci_low,g2.ci_high-g2['mean']]),fmt='o',color=COLORS[3],capsize=3)
axs[1].set_yticks(y,g2.comparison.str.replace('GAIL refinement | ','',regex=False));axs[1].invert_yaxis();axs[1].axvline(0,color='gray',ls='--');axs[1].set(xlabel='GAIL minus its initializer progress (m)',title='Small GAIL gains are not uniform')
save(fig,'05_paired_effects','95% stratified round-bootstrap intervals conditional on observed checkpoints; exploratory, no multiplicity correction or population-of-training-seeds claim.')

# 6 full and round-centered correlation structure for successful trained families.
fig,axs=plt.subplots(1,2,figsize=(12,5.8));names=['Progress','Forward speed','Legacy direction','Action MSE','Joint SWD','Falls','Jerk']
for ax,corr,title in [(axs[0],rawcorr,'Pooled deployments'),(axs[1],rescorr,'After subtracting each round mean')]:
 im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(7),names,rotation=55,ha='right');ax.set_yticks(range(7),names);ax.set_title(title)
 for i in range(7):
  for j in range(7):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(corr.iloc[i,j])>.65 else 'black')
# Manual spacing avoids tight_layout/colorbar conflicts.
fig.subplots_adjust(bottom=.28,wspace=.5,right=.88)
fig.colorbar(im,cax=fig.add_axes([.92,.28,.016,.55]),label='Spearman correlation')
fig.text(.5,.03,'Walking policies only; dependent deployments, descriptive correlations. Round centering removes shared round levels, not all confounding.',ha='center',fontsize=8)
for ext in ['png','pdf']:fig.savefig(FIG/f'06_metric_relationships.{ext}',dpi=180)
atlas.savefig(fig);plt.close(fig);figures.append({'name':'06_metric_relationships','caption':'Spearman correlations before and after round centering; walking controllers only.'})

# 7 trace error histogram and concentration, no pseudoreplication.
fig,axs=plt.subplots(1,2,figsize=(12,5))
tkeys=keys[1:]
for i,key in enumerate(tkeys):
 g=td[td.label==key];axs[0].hist(np.log10(g.action_mse),bins=np.linspace(-3.5,1,19),histtype='step',lw=2,label=SHORT[key],color=COLORS[i])
axs[0].set(xlabel='log10 mean action error per 60-second run',ylabel='Number of rounds',title='Action error has a skewed distribution');axs[0].legend(fontsize=8)
axs[1].boxplot([td[td.label==k].top1pct_error_share*100 for k in tkeys],tick_labels=[SHORT[k].replace(' ','\n',1) for k in tkeys],showmeans=True)
axs[1].set(ylabel='Share of total squared error in top 1% of frames (%)',title='Does mean MSE hide brief error bursts?')
save(fig,'07_error_distribution','Each histogram entry and box-plot observation is a deployment, not an independent time step. Top 1% = 30 of 3,000 frames.')

# 8 action fidelity vs task/physical outputs (log x), checkpoint aggregates.
fig,axs=plt.subplots(1,3,figsize=(13,4.6))
for c,col in zip(['original','continued','scratch','scratch200'],COLORS):
 g=sm[(sm.cohort==c)&sm.algorithm.isin(ALG+['bc_matched'])]
 for ax,m,ylabel in zip(axs,['mean_progress','swd','jerk'],['Target progress (m)','Joint occupancy SWD','Jerk rate (rad²/s⁶)']):
  ax.scatter(g.mse,g[m],color=col,label=c,s=33,alpha=.7);ax.set(xscale='log',xlabel='Held-out latent-action MSE',ylabel=ylabel)
axs[0].set_title('Task progress');axs[1].set_title('Joint-space fidelity');axs[2].set_title('Kinematic smoothness');axs[0].legend(fontsize=8)
save(fig,'08_fidelity_tradeoffs','One point per checkpoint, 30-round means. Shared held-out test pool; architectures and training histories are mixed, so trends are not causal.')

# 9 all targets for original BC/DAgger h1/h2 against random and raw observation.
pg=pp[pp.representation.isin([f'original_hidden_42_{a}_{h}' for a in ['bc','dagger_mixed'] for h in ['h1','h2']])]
targets=sorted(pg.target.unique());reps=sorted(pg.representation.unique())
fig,axs=plt.subplots(1,2,figsize=(13,10))
for ax,metric,title in zip(axs,['gain_over_random_mean','gain_over_raw'],['Gain over matched random-network readouts','Gain over raw-observation readouts']):
 mat=pg.pivot(index='target',columns='representation',values=metric).loc[targets,reps]
 im=ax.imshow(mat,vmin=-.5,vmax=.5,cmap='RdBu',aspect='auto')
 ax.set_yticks(range(len(targets)),[t.replace('_',' ') for t in targets] if ax==axs[0] else [])
 ax.set_xticks(range(len(reps)),[r.replace('original_hidden_42_','').replace('dagger_mixed','DAgger') for r in reps],rotation=35,ha='right');ax.set_title(title)
 for i in range(len(targets)):
  for j in range(len(reps)):
   v=mat.iloc[i,j];ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>.35 else 'black')
fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.02,pad=.03,label='Difference in held-out R²')
fig.subplots_adjust(left=.29,right=.9,wspace=.15,bottom=.17,top=.94)
fig.text(.5,.025,'All 30 pre-existing probe targets; same oracle-state corpus. Colors saturate at ±0.5; printed values are exact to 2 dp. No new probes fitted.',ha='center',fontsize=8)
for ext in ['png','pdf']:fig.savefig(FIG/f'09_probe_controls.{ext}',dpi=180)
atlas.savefig(fig);plt.close(fig);figures.append({'name':'09_probe_controls','caption':'Full target inventory for original BC/DAgger; raw and random features are necessary controls.'})

# 10 paired progression of BC controls: SD and median rather than a single bar.
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for arch,col in [('hidden',COLORS[0]),('input',COLORS[1])]:
 for seed in [42,43,44]:
  gs=[sm[(sm.cohort==c)&(sm.arch==arch)&(sm.seed==seed)&(sm.algorithm=='bc')].iloc[0] for c in ['scratch','scratch200','continued']]
  axs[0].plot(range(3),[g.mse for g in gs],marker='o',alpha=.6,color=col,label=arch if seed==42 else None)
  axs[1].plot(range(3),[g.mean_progress for g in gs],marker='o',alpha=.6,color=col,label=arch if seed==42 else None)
for ax in axs:ax.set_xticks(range(3),['Varied-only\n100 epochs','Varied-only\n200 epochs','Original + varied\n100 + 100 epochs']);ax.grid(axis='y',alpha=.2)
axs[0].set(yscale='log',ylabel='Held-out action MSE',title='Optimization and data history both remain visible');axs[0].legend()
axs[1].set(ylabel='Mean sampled target progress (m; zoomed axis)',title='Better fidelity need not mean more progress')
save(fig,'10_bc_budget_history','Lines pair seed labels, not continuous training for the continued arm; three replicas per architecture. 200-epoch fits have not established convergence.')

# 11: Measurement audit; explicitly distinguish stale position frames from immobility.
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
example=ROOT/'experiments/consolidated/v1/evaluation/all_all_correct/runs/grouped_group_0_draw_0/oracle/trace.npz'
with np.load(example,allow_pickle=False) as t:
 x=np.arange(24)*.02
 axs[0].plot(x,t['planar_speed'][:24],'-o',markersize=3,label='20 ms trace differences')
 v=np.linalg.norm(t['positions'][2:26:2]-t['positions'][:24:2],axis=1)/.04
 axs[0].plot(x[::2],v,'-s',markersize=4,label='40 ms position windows')
axs[0].set(xlabel='Time since displayed segment start (s)',ylabel='Planar speed estimate (m/s)',title='Position freshness differs from the action rate');axs[0].legend(fontsize=8,loc='upper right')
for i,key in enumerate(keys):
 g=td[td.label==key]
 axs[1].plot([0,1],[g.legacy_direction_error.mean(),g.direction_error_40ms_defined.mean()],'-o',label=SHORT[key])
axs[1].set_xticks([0,1],['Existing 20 ms metric','40 ms diagnostic\n(defined directions only)'])
axs[1].set(ylabel='Mean unit-vector direction error',title='Stale frames add an artificial tracking penalty');axs[1].legend(fontsize=8)
save(fig,'11_localization_audit','Left: one sampled oracle segment. Right: 30 sampled rounds/model. Diagnostic excludes reset/target-change windows; historical metrics unchanged.')

# 12: Original optimization history and the different signs of deployment changes.
curves=pd.DataFrame(D['curves']['rows'])
curves.to_csv(OUT/'resource_curves.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
gc=curves[curves.algorithm=='dagger_mixed']
axs[0].plot(gc.cumulative_supervised_updates/1000,gc.test_mse,'-o',label='DAgger rounds 0–7',color=COLORS[1])
bc0=curves[curves.algorithm=='bc'].iloc[0]
bcfinal=df[(df.label=='bc_original_matched')&(df.suite=='original')].held_out_expert_action_mse.iloc[0]
axs[0].scatter([bc0.cumulative_supervised_updates/1000,211.05],[bc0.test_mse,bcfinal],marker='s',s=60,color=COLORS[0],label='BC: initial / matched')
axs[0].set(yscale='log',xlabel='Supervised optimizer updates (thousands)',ylabel='Original expert-test action MSE',title='Original fidelity rankings depend on allocation');axs[0].legend(fontsize=8)
dv=df[df.label.isin(['bc_original_matched','original_hidden_42_bc'])].pivot(index='round',columns='label',values='return_target_progress_m')
delta=dv.bc_original_matched-dv.original_hidden_42_bc
categories=[('original','Original scene (5)'),('nominal','Scalar 0.1 (3)'),('sampled','Sampled slips (30)')]
for i,(suite,name) in enumerate(categories):
 ids=df[df.suite.isin(['grouped','confirmation']) if suite=='sampled' else df.suite==suite]['round'].unique()
 v=delta.loc[ids].dropna()
 axs[1].scatter(np.full(len(v),i)+np.linspace(-.1,.1,len(v)),v,s=22,alpha=.65,color=COLORS[i])
 axs[1].plot([i-.2,i+.2],[v.mean(),v.mean()],color='black',lw=2)
axs[1].set_xticks(range(3),[x[1] for x in categories]);axs[1].axhline(0,color='gray',ls=':')
axs[1].set(ylabel='Matched BC minus initial BC progress (m)',title='Progress change is not uniformly positive')
save(fig,'12_optimization_and_scene','Left: nearby initial BC/DAgger points are 66,500 / 64,500 updates, not equal. Right: paired rounds; black lines are means; unequal suite sizes.')
atlas.close()
manifest.update(unique_deployments=len(df),duplicate_records_verified=duplicates,
 unique_controllers=int(df.label.nunique()),sampled_rounds=int(s['round'].nunique()),
 active_hours=float(df.active_seconds.sum()/3600),figures=figures,
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
manifest['implementation_sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
 ['consolidated/eval_worker.py','world/World.py','world/Robot.py','representation_probing/run.py']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['unique_deployments','duplicate_records_verified','unique_controllers','sampled_rounds','active_hours']},indent=2))
print('DONE:',OUT)

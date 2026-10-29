"""Read-only analysis: publication figures from completed research artifacts.

No simulation, training, new significance tests or checkpoint selection.
Run from any directory with NumPy and Matplotlib installed.
"""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/walker-final-figures')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'figures_final'
SOURCES = {
    'controls': ROOT/'experiments/submission_controls/v1/evaluation/summary/comparison.json',
    'curves': ROOT/'experiments/submission_controls/v1/curves/curves.json',
    'parent': ROOT/'experiments/consolidated/v1/evaluation/all_all_correct/comparison.json',
    'scratch': ROOT/'experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json',
    'probes': ROOT/'experiments/representation_probing/consolidated_v1/report.json',
    'scratch_probes': ROOT/'experiments/consolidated/v1/scratch_v1/probing/report.json',
}
D = {k: json.loads(p.read_text()) for k,p in SOURCES.items()}
C = ['#2878B5', '#DE8F05', '#008D80', '#9C4F96']
ALG = ['bc','dagger_mixed','gail_bc','gail_dagger_mixed']
NAMES = ['BC','Mixed DAgger','GAIL–BC','GAIL–DM']
plt.rcParams.update({'font.size':12, 'axes.titlesize':14, 'axes.labelsize':12,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'figure.facecolor':'white'})
OUT.mkdir(exist_ok=True)
VALUES = {}

def save(fig, name, note):
    fig.text(.5,.025,note,ha='center',fontsize=10,color='#485A65')
    fig.tight_layout(rect=(0,.075,1,1),pad=1.4)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=190)
    plt.close(fig)

def aggregate(source, suite, name):
    if suite=='sampled' and source!='controls':
        rows=[r for r in D[source]['runs'] if r['controller']['label']==name
              and r['row']['suite'] in ('grouped','confirmation')]
        assert len(rows)==30,(source,name,len(rows))
        fields=['held_out_expert_action_mse','return_target_progress_m',
                'jerk_cost_rate_rad2_per_s6','joint_occupancy_sliced_wasserstein']
        return {**{k:float(np.mean([r[k] for r in rows])) for k in fields},
                'total_falls':sum(r['falls'] for r in rows)}
    return next(r for r in D[source]['aggregates'] if r['suite']==suite and r['policy']==name)

def cohort(source, prefix, arch, alg, field, suite='sampled'):
    return [aggregate(source,suite,f'{prefix}_{arch}_{s}_{alg}'+('_correct' if arch=='input' else ''))[field]
            for s in (42,43,44)]

def bars(ax, values, labels, colors, ylabel, title, log=False, digits=4):
    ax.bar(range(len(values)), values, color=colors, width=.6)
    ax.set_xticks(range(len(values)),labels,fontsize=10)
    ax.set_ylabel(ylabel);ax.set_title(title)
    if log: ax.set_yscale('log');ax.set_ylim(min(values)*.55,max(values)*2.1)
    else: ax.set_ylim(min(0,min(values)*1.4),max(values)*1.22 if max(values)>0 else 1)
    for i,v in enumerate(values):
        ax.annotate(f'{v:.{digits}f}',(i,v),xytext=(0,5 if v>=0 else -12),
                    textcoords='offset points',ha='center',fontsize=10)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)

def main():
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    initial=[aggregate('scratch','original',f'original_hidden_42_{a}') for a in ALG]
    for ax,field,title,unit,log,digits in zip(axs,
        ['held_out_expert_action_mse','return_target_progress_m'],
        ['Initial allocation: large fidelity differences','Similar progress to the oracle'],
        ['Latent-action MSE (log scale)','Target progress (m)'],[True,False],[5,2]):
        bars(ax,[r[field] for r in initial],NAMES,C,unit,title,log,digits)
    axs[1].axhline(44.44587020874023,color='#555',ls='--',lw=1,label='Oracle');axs[1].legend(fontsize=9)
    save(fig,'11_initial','Initial allocations · 5 deployments/controller · BC receives fewer optimizer updates than DAgger')
    labels=['BC\n100 epochs','Mixed\nDAgger','BC\nmatched updates']
    keys=['original_hidden_42_bc','original_hidden_42_dagger_mixed','bc_original_matched']
    rows=[aggregate('controls','original',k) for k in keys]
    fig,axs=plt.subplots(1,3,figsize=(12,4.4))
    for ax,field,title,unit,log,digits in zip(axs,
        ['held_out_expert_action_mse','return_target_progress_m','joint_occupancy_sliced_wasserstein'],
        ['Matching updates reverses MSE ranking','Progress stays similar','Occupancy does not follow MSE'],
        ['Latent-action MSE (log scale)','Target progress (m)','Joint occupancy SWD'],[True,False,False],[5,2,4]):
        bars(ax,[r[field] for r in rows],labels,C[:3],unit,title,log,digits)
    axs[1].axhline(44.44587020874023,color='#555',ls='--',lw=1,label='Oracle');axs[1].legend(fontsize=9)
    VALUES['01_budget'] = rows
    save(fig,'01_budget','Original scene · 5 deployments/controller · one training seed · means, not superiority tests')

    # Actual sampled slip, not Gaussian parent means. Same task draws across policies.
    fig,ax=plt.subplots(figsize=(12,4.4))
    for name,label,col in [('oracle','Oracle','#555'),(keys[0],'Frozen BC',C[0]),
                           (keys[1],'Frozen DAgger',C[1])]:
        rs=[r for r in D['controls']['runs'] if r['controller']['label']==name and r['row']['suite'] in ('grouped','confirmation')]
        ax.scatter([r['row']['slip'] for r in rs],[r['return_target_progress_m'] for r in rs],s=32,alpha=.75,label=label,color=col)
    ax.set_xscale('log');ax.set_xlabel('Actual fixed slip in each run (log scale)');ax.set_ylabel('Target progress (m)')
    ax.set_title('Oracle parity can coexist with slower walking on changed surfaces')
    ax.legend(ncol=3);ax.grid(alpha=.18)
    save(fig,'02_slip','30 paired surface/task rounds · slip held fixed per run · task seeds also vary across rounds')

    fig,ax=plt.subplots(figsize=(12,4.4))
    for i,a in enumerate(ALG):
        frozen=aggregate('scratch','sampled',f'original_hidden_42_{a}')['held_out_expert_action_mse']
        cont=cohort('scratch','varied','hidden',a,'held_out_expert_action_mse')
        ax.plot([i-.18,i+.18],[frozen,np.mean(cont)],color=C[i],lw=2)
        ax.scatter(i-.18,frozen,marker='s',s=70,color=C[i]);ax.scatter([i+.18]*3,cont,s=38,color=C[i],marker='o')
        ax.text(i-.18,frozen*1.15,f'{frozen:.5f}',ha='center',fontsize=10)
        ax.text(i+.18,np.mean(cont)*.75,f'{np.mean(cont):.5f}',ha='center',fontsize=10)
    ax.set_xticks(range(4),NAMES);ax.set_yscale('log');ax.set_ylim(.0008,.4)
    ax.set_ylabel('Sampled-test latent MSE (log scale)');ax.set_title('Continuation improves fidelity at the initial allocations')
    ax.grid(axis='y',alpha=.18)
    save(fig,'03_continuation','Square: frozen original model · dots: 3 continued hidden models · more training and changed data are confounded')

    fig,ax=plt.subplots(figsize=(12,4.4))
    for j,(prefix,col,label) in enumerate([('varied',C[0],'Continued'),('scratch_varied',C[1],'Varied-only')]):
        for i,a in enumerate(ALG):
            vals=np.array(cohort('scratch',prefix,'input',a,'return_target_progress_m'))-np.array(cohort('scratch',prefix,'hidden',a,'return_target_progress_m'))
            y=i+(j-.5)*.25
            ax.scatter(vals,[y]*3,color=col,s=30,alpha=.65,label=label if i==0 else None)
            ax.scatter(np.mean(vals),y,color=col,marker='D',s=65)
    ax.axvline(0,color='#555',lw=1);ax.set_yticks(range(4),NAMES);ax.invert_yaxis()
    ax.set_xlabel('Input-conditioned minus hidden mean progress (m)');ax.set_title('Explicit slip has small, inconsistent progress benefits')
    ax.legend();ax.grid(axis='x',alpha=.18)
    save(fig,'04_context','Dots: per-training-seed differences on shared 30 rounds · diamonds: means · initial allocations, not new 200-epoch fits')

    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    valsout={}
    for ax,arch in zip(axs,['hidden','input']):
        vals=[cohort('controls','scratch_varied',arch,'bc','held_out_expert_action_mse'),
              [aggregate('controls','sampled',f'bc_varied_{arch}_{s}_200')['held_out_expert_action_mse'] for s in (42,43,44)],
              cohort('controls','varied',arch,'bc','held_out_expert_action_mse')]
        bars(ax,[np.mean(v) for v in vals],['Varied-only\n100 epochs','Varied-only\n200 epochs','Original 100\n+ varied 100'],C[:3],
             'Sampled-test latent MSE (log scale)',arch.title()+' context',True,5)
        for i,v in enumerate(vals):ax.scatter(i+np.linspace(-.12,.12,3),v,color='#172B3A',s=22,zorder=4)
        valsout[arch]=vals
    VALUES['05_pretraining']=valsout
    save(fig,'05_pretraining','Bars: means; dots: 3 training replicas · last two conditions allocate 34 million exposures; data/history still differ')

    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for j,(prefix,label,col) in enumerate([('varied','Continued',C[0]),('scratch_varied','Varied-only',C[1])]):
        falls=[sum(cohort('scratch',prefix,'hidden',a,'total_falls')) for a in ALG]
        jerk=[np.mean(cohort('scratch',prefix,'hidden',a,'jerk_cost_rate_rad2_per_s6'))/1e6 for a in ALG]
        for ax,vals in zip(axs,[falls,jerk]):
            pos=np.arange(4)+(j-.5)*.34
            ax.bar(pos,vals,width=.32,color=col,label=label)
            for xx,v in zip(pos,vals):ax.text(xx,v+.1 if ax==axs[0] else v+.5,f'{v:g}' if ax==axs[0] else f'{v:.2f}',ha='center',fontsize=9)
    for ax in axs:ax.set_xticks(range(4),NAMES,fontsize=10);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axs[0].set_ylabel('Falls per 90 active minutes');axs[0].set_ylim(0,8);axs[0].set_title('Few events; no robust population ranking');axs[0].legend(fontsize=10)
    axs[1].set_ylabel('Jerk rate (million rad²/s⁶)');axs[1].set_ylim(0,50);axs[1].set_title('Kinematic smoothness, not energy')
    save(fig,'06_mechanics','Hidden models at initial allocations · 3 training replicas × 30 rounds · recovery excluded from active time')

    targets=['phase_sin_times_roll_deg','future_200ms_forward_velocity_mps']
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    pv={}
    for ax,t,title in zip(axs,targets,['Phase-conditioned roll','Future forward velocity (200 ms)']):
        for i,a in enumerate(ALG):
            names=[f'varied_hidden_{s}_{a}_h1' for s in (42,43,44)]
            v=[r['r2'] for r in D['probes']['results'] if r['representation'] in names and r['labels']=='true' and r['target']==t]
            assert len(v)==3,(names,t,v)
            ax.scatter(i+np.linspace(-.08,.08,3),v,color=C[i],s=25,alpha=.7)
            ax.scatter(i,np.mean(v),marker='D',color=C[i],s=65)
            ax.text(i,np.mean(v)+.012,f'{np.mean(v):.4f}',ha='center',fontsize=10)
            pv[t+'_'+a]=v
        ax.set_xticks(range(4),NAMES,fontsize=10);ax.set_ylim(.5,.78);ax.set_ylabel('Held-out ridge R² (zoomed axis)');ax.set_title(title);ax.grid(axis='y',alpha=.18)
    VALUES['07_probes']=pv
    save(fig,'07_probes','First-layer features on shared oracle states · dots: 3 replicas, diamonds: means · new matched-BC models were not probed')

    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for ax,suite in zip(axs,['original','sampled']):
        rs=[aggregate('controls',suite,k) for k in ['zero_native','oracle']]
        bars(ax,[r['return_target_progress_m'] for r in rs],['Zero native\nactions','Oracle'],['#929DA5',C[2]],'Target progress (m)',suite.title(),False,3)
        ax.set_ylim(-4,55)
    save(fig,'08_zero','Same gait/IK engine and resets · original: 5 rounds; sampled: 30 · zero actions are not expert-mean actions')

    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    rows=D['curves']['rows'];d=[r for r in rows if r['algorithm']=='dagger_mixed']
    axs[0].plot([r['cumulative_supervised_updates']/1000 for r in d],[r['test_mse'] for r in d],'o-',color=C[1],label='Mixed DAgger rounds')
    axs[0].scatter([66.5,211.05],[rows[0]['test_mse'],aggregate('controls','original','bc_original_matched')['held_out_expert_action_mse']],color=[C[0],C[2]],s=75,marker='s',label='BC original / matched')
    axs[0].set_xlabel('Allocated supervised updates (thousands)');axs[0].set_title('Data and fitting accumulate together')
    for a,col,label in [('gail_bc',C[0],'GAIL–BC'),('gail_dagger_mixed',C[1],'GAIL–DM')]:
        rr=[r for r in rows if r['algorithm']==a]
        axs[1].plot([r['total_lineage_learner_steps']/1000 for r in rr],[r['test_mse'] for r in rr],'o-',color=col,label=label)
    axs[1].set_xlabel('Learner steps including initializer (thousands)');axs[1].set_title('Conservative GAIL changes little')
    for ax in axs:ax.set_yscale('log');ax.set_ylabel('Original-test latent MSE (log scale)');ax.legend(fontsize=10);ax.grid(alpha=.18)
    save(fig,'09_resources','Retrospective fidelity curves, not return convergence · GAIL–DM inherits 70k learner steps; GAIL–BC inherits BC fitting')

    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    rs=[aggregate('controls','sampled',k) for k in keys]
    bars(axs[0],[r['held_out_expert_action_mse'] for r in rs],labels,C[:3],'Sampled-test MSE (log scale)','Original-only fitting improves changed-surface fidelity',True,5)
    bars(axs[1],[r['total_falls'] for r in rs],labels,C[:3],'Falls per 30 active minutes','Fewer recorded falls after longer original-only BC',False,0)
    save(fig,'10_frozen_control','All models trained only in original scene · shared 30 sampled rounds · one training seed/model; not proof of zero fall risk')

    manifest={'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()},
              'values':VALUES,'conventions':'Means and replica points only; no invented error bars. See captions for units and populations.',
              'figures':sorted(p.name for p in OUT.glob('*.pdf'))}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated',len(manifest['figures']),'figures in PNG/PDF with source hashes.')

if __name__=='__main__':main()

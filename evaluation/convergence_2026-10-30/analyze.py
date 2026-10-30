"""Reproducible offline convergence EDA. Does not change policies or source metrics."""
import os
from pathlib import Path
import sys
import json
import hashlib
import platform
os.environ.setdefault('MPLCONFIGDIR', '/tmp/walker-convergence-mpl')
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from variable_training.common import read, digest, save, verified

OUT = Path(__file__).resolve().parent
BATCH = ROOT/'experiments/convergence_controls/all_v1'
PROBE = ROOT/'experiments/representation_probing/convergence_all_v1'
SOURCES = [ROOT/p for p in [
    'experiments/consolidated/v1/evaluation/all_all_correct/comparison.json',
    'experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json',
    'experiments/submission_controls/v1/evaluation/summary/comparison.json',
    'experiments/convergence_controls/all_v1/evaluation/summary_all/comparison.json']]
METRICS = ['return_target_progress_m', 'held_out_expert_action_mse', 'on_policy_latent_action_mse',
           'joint_occupancy_sliced_wasserstein', 'mean_forward_velocity_mps',
           'mean_target_direction_error', 'falls', 'jerk_cost_rate_rad2_per_s6',
           'torso_roll_rms_deg', 'torso_pitch_rms_deg']
MSE, ON, PROGRESS, SWD = METRICS[1], METRICS[2], METRICS[0], METRICS[3]
ORIG = ['original_hidden_42_bc', 'bc_original_matched', 'bc_original_convergence',
        'original_hidden_42_dagger_mixed', 'dagger_original_fixed_aggregate_convergence']
NAMES = ['BC initial', 'BC matched', 'BC extended', 'DAgger initial', 'DAgger extended']
COLORS = ['#56B4E9', '#0072B2', '#003B5C', '#E69F00', '#A04D00']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'figure.facecolor': 'white'})


def load():
    lookup = {}
    for path in SOURCES:
        report = read(path)
        aggregates = {(a['suite'], a['policy']): a for a in report['aggregates']}
        for raw in report['runs']:
            r = dict(raw)
            key = (r['row']['id'], r['controller']['label'])
            a = aggregates.get((r['row']['suite'], key[1]), {})
            if MSE not in r:
                r[MSE] = a.get(MSE)
            if key in lookup:
                old = lookup[key]
                assert old['row'] == r['row'] and old['controller'].get('sha256') == r['controller'].get('sha256')
                for metric in METRICS:
                    if old.get(metric) is not None and r.get(metric) is not None:
                        assert np.isclose(old[metric], r[metric], rtol=1e-7, atol=1e-9), (key, metric)
            lookup[key] = r
    rows = []
    for (round_id, label), r in lookup.items():
        c = r['controller']; row = r['row']
        rows.append(dict(label=label, round=round_id, suite=row['suite'], slip=row['slip'],
                         parent_mean=row.get('parent_mean'), seed=c.get('seed'),
                         arch=c['architecture'], domain=c['domain'], algorithm=c.get('algorithm', label),
                         active_seconds=r['active_seconds'], checkpoint_sha256=c.get('sha256'),
                         **{m: r.get(m) for m in METRICS}))
    df = pd.DataFrame(rows)
    assert len(df) == 3205 and not df.duplicated(['round', 'label']).any()
    assert (df.active_seconds == 60).all()
    oracle = df[df.label == 'oracle'].set_index('round')[PROGRESS]
    df['oracle_delta'] = df[PROGRESS] - df['round'].map(oracle)
    return df


def bootstrap(z, meta):
    """Conditional round bootstrap, preserving suite x Gaussian-parent mixture."""
    rng = np.random.default_rng(20261030)
    draws = np.zeros(5000)
    for _, idx in meta.loc[z.index].groupby(['suite', 'parent_mean']).groups.items():
        v = z.loc[idx].to_numpy()
        draws += rng.choice(v, (5000, len(v)), replace=True).sum(axis=1)/len(z)
    return np.quantile(draws, [.025, .975])


def main():
    out_fig = OUT/'figures'; out_fig.mkdir(exist_ok=True)
    summary = BATCH/'evaluation/summary_all'
    assert verified(summary, read(summary/'completed.json')['identity'])
    results = read(summary/'comparison.json')
    assert len(results['training']) == 26 and sum(not r['reused'] for r in results['runs']) == 982
    df = load(); df.to_csv(OUT/'deployments.csv', index=False)
    sampled = df[df.suite.isin(['grouped', 'confirmation'])]
    meta = sampled[['round', 'suite', 'parent_mean']].drop_duplicates().set_index('round')
    jobs = {p.parent.name: read(p) for p in sorted((BATCH/'training').glob('*/job.json'))
            if read(p)['task'] == 'control_train'}
    labels = list(jobs)
    engine = [f'engine_random_network_{s}' for s in (42,43,44)]
    relevant = set(labels + engine + ORIG + ['oracle','random','zero_native'] +
                   [j['reference_label'] for j in jobs.values()])
    table = []
    for suite, part in [('original',df[df.suite=='original']), ('nominal',df[df.suite=='nominal']), ('sampled',sampled)]:
        for label, g in part[part.label.isin(relevant)].groupby('label'):
            table.append(dict(label=label, suite=suite, runs=len(g), total_falls=int(g.falls.sum()),
                active_minutes=g.active_seconds.sum()/60, progress_sd=g[PROGRESS].std(ddof=1),
                oracle_delta_mean=g.oracle_delta.mean(), oracle_delta_sd=g.oracle_delta.std(ddof=1),
                worst_three_oracle_delta=g.oracle_delta.nsmallest(min(3,len(g))).mean(),
                **{m: g[m].mean() for m in METRICS}))
    means = pd.DataFrame(table); means.to_csv(OUT/'checkpoint_summary.csv',index=False)
    training = []
    for t in results['training']:
        j = jobs[t['label']]
        training.append({**{k:j[k] for k in ('label','domain','cohort','architecture','algorithm','seed','reference_label')},
            **{k:t[k] for k in ('initial_validation_mse','validation_mse','best_epoch','epochs_run',
              'gradient_updates','selected_gradient_updates','total_lineage_gradient_updates','stop_reason')},
            'inherited_updates':t['inherited_costs']['gradient_updates'],
            'validation_reduction_pct':100*(1-t['validation_mse']/t['initial_validation_mse'])})
    tr = pd.DataFrame(training); tr.to_csv(OUT/'training_summary.csv',index=False)
    # Selected epoch zero supplies an exact-weight repeatability comparison.
    # Compare tensors (serialization bytes can differ), then pair the same task rows.
    import torch
    repeats=[]
    for label in tr[tr.best_epoch==0].label:
        j=jobs[label]
        initial=torch.load(j['model'],map_location='cpu',weights_only=True)
        final=torch.load(BATCH/'training'/label/'policy.pth',map_location='cpu',weights_only=True)
        assert initial.keys()==final.keys() and all(torch.equal(initial[k],final[k]) for k in initial)
        a=sampled[sampled.label==label].set_index('round')
        b=sampled[sampled.label==j['reference_label']].set_index('round')
        for m in METRICS:
            z=(a[m]-b[m]).dropna()
            repeats.append(dict(label=label,reference=j['reference_label'],metric=m,
                identical_weight_tensors=True,n_rounds=len(z),mean_delta=z.mean(),
                sd_delta=z.std(ddof=1),max_absolute_delta=z.abs().max(),
                old_falls=int(b.falls.sum()),new_falls=int(a.falls.sum())))
    pd.DataFrame(repeats).to_csv(OUT/'identical_weight_repeats.csv',index=False)
    tr.groupby(['domain','cohort','architecture','algorithm']).agg(
        replicas=('label','size'), additional_epochs_mean=('epochs_run','mean'),
        minimum_epochs=('epochs_run','min'), maximum_epochs=('epochs_run','max'),
        lineage_updates_mean=('total_lineage_gradient_updates','mean'),
        validation_reduction_pct=('validation_reduction_pct','mean')).to_csv(OUT/'training_families.csv')
    effects = []
    def compare(name,left,right,metric=PROGRESS):
        l = sampled[sampled.label.isin(left)].groupby('round')[metric].mean()
        r = sampled[sampled.label.isin(right)].groupby('round')[metric].mean()
        z = (l-r).dropna(); assert len(z)==30,(name,metric)
        lo,hi = bootstrap(z,meta)
        effects.append(dict(comparison=name,metric=metric,left=';'.join(left),right=';'.join(right),
                            mean=z.mean(),sd=z.std(ddof=1),ci_low=lo,ci_high=hi,
                            grouped=z[meta.loc[z.index].suite=='grouped'].mean(),
                            confirmation=z[meta.loc[z.index].suite=='confirmation'].mean()))
        return z
    for label,j in jobs.items():
        for m in (PROGRESS,ON,SWD,'falls'):
            compare(label+' minus initializer',[label],[j['reference_label']],m)
    def group(cohort,arch,algo):
        return tr[(tr.cohort==cohort)&(tr.architecture==arch)&(tr.algorithm==algo)].label.tolist()
    family = []
    for cohort in ('continued','scratch'):
        for arch in ('hidden','input'):
            for algo in ('bc','dagger_mixed'):
                ls=group(cohort,arch,algo); refs=[jobs[x]['reference_label'] for x in ls]
                for stage, cs in [('initializer',refs),('extended',ls)]:
                    g=sampled[sampled.label.isin(cs)]
                    family.append(dict(cohort=cohort,arch=arch,algorithm=algo,stage=stage,
                        runs=len(g),falls_total=int(g.falls.sum()), **{m:g[m].mean() for m in METRICS},
                        mean_replica_progress_sd=g.groupby('label')[PROGRESS].mean().std(ddof=1)))
                for m in (PROGRESS,ON,SWD,'falls'):
                    compare(f'Extended minus initializer | {cohort} {arch} {algo}',ls,refs,m)
            for m in (PROGRESS,ON,SWD,'falls'):
                compare(f'DAgger minus BC | {cohort} {arch}',group(cohort,arch,'dagger_mixed'),group(cohort,arch,'bc'),m)
        for algo in ('bc','dagger_mixed'):
            compare(f'Context | {cohort} {algo}',group(cohort,'input',algo),group(cohort,'hidden',algo))
    for arch in ('hidden','input'):
        for algo in ('bc','dagger_mixed'):
            compare(f'Continuation minus scratch | {arch} {algo}',group('continued',arch,algo),group('scratch',arch,algo))
    for m in (PROGRESS,ON,SWD,'falls'):
        compare('Original extended DAgger minus BC',[ORIG[4]],[ORIG[2]],m)
    for ref in ('oracle','zero_native','random'):
        compare('Random networks minus '+ref,engine,[ref])
    ef=pd.DataFrame(effects);ef.to_csv(OUT/'paired_effects.csv',index=False)
    fam=pd.DataFrame(family);fam.to_csv(OUT/'family_summary.csv',index=False)
    # Balanced 26-model panel. Residual includes interactions and simulator variation.
    panel=sampled[sampled.label.isin(labels)]
    decompositions=[]
    for m in (PROGRESS,ON,SWD,'jerk_cost_rate_rad2_per_s6'):
        a=panel.pivot(index='round',columns='label',values=m).to_numpy()
        assert a.shape==(30,26) and np.isfinite(a).all()
        rm=a.mean(axis=1,keepdims=True);cm=a.mean(axis=0,keepdims=True);grand=a.mean()
        ss=np.array([26*((rm-grand)**2).sum(),30*((cm-grand)**2).sum(),((a-rm-cm+grand)**2).sum()])
        assert np.isclose(ss.sum(),((a-grand)**2).sum())
        decompositions.append(dict(metric=m,round_fraction=ss[0]/ss.sum(),policy_fraction=ss[1]/ss.sum(),residual_fraction=ss[2]/ss.sum()))
    pd.DataFrame(decompositions).to_csv(OUT/'variance_decomposition.csv',index=False)
    fields=[PROGRESS,ON,SWD,'falls','jerk_cost_rate_rad2_per_s6']
    panel[fields].corr(method='spearman').to_csv(OUT/'correlations_raw.csv')
    (panel[fields]-panel.groupby('round')[fields].transform('mean')).corr(method='spearman').to_csv(OUT/'correlations_round_centered.csv')
    pdf=PdfPages(OUT/'EDA_FIGURES.pdf')
    def figure(fig,name):
        fig.tight_layout();fig.savefig(out_fig/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(out_fig/(name+'.pdf'),bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.7))
    orig=means[means.suite=='original'].set_index('label').loc[ORIG]
    for ax,m,title in zip(axes,[MSE,ON,SWD],['Held-out latent-action MSE','On-policy latent-action MSE','Joint-occupancy SWD']):
        bars=ax.bar(NAMES,orig[m],color=COLORS)
        ax.bar_label(bars,fmt='%.4f',fontsize=9,padding=3);ax.tick_params(axis='x',rotation=30)
        ax.set_title(title)
        if m!=SWD:ax.set_yscale('log');ax.set_ylim(orig[m].min()*.6,orig[m].max()*2)
    fig.suptitle('Original scene: the fidelity ranking depends on optimization allocation',y=1.02)
    figure(fig,'01_original_fidelity')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric,title in zip(axes,[PROGRESS,'oracle_delta'],['Raw progress across 30 sampled rounds','Progress minus oracle on each matching round']):
        arrays=[sampled[sampled.label==l][metric].to_numpy() for l in ORIG]
        ax.boxplot(arrays,tick_labels=NAMES,showfliers=True)
        ax.tick_params(axis='x',rotation=25);ax.set_title(title);ax.set_ylabel('m per 60 active seconds')
        if metric=='oracle_delta':ax.axhline(0,color='gray',lw=1)
    figure(fig,'02_round_paired_distributions')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(cohort,arch) in zip(axes.flat,[(c,a) for c in ('continued','scratch') for a in ('hidden','input')]):
        g=fam[(fam.cohort==cohort)&(fam.arch==arch)]
        for algo,color in [('bc','#0072B2'),('dagger_mixed','#E69F00')]:
            v=g[g.algorithm==algo].set_index('stage').loc[['initializer','extended'],MSE]
            ax.plot([0,1],v,'o-',color=color,label=algo)
            for x,y in enumerate(v):ax.annotate(f'{y:.5f}',(x,y),xytext=(8,6),textcoords='offset points')
        ax.set_xticks([0,1],['Initializer','Validation-stopped']);ax.set_yscale('log')
        ax.set_xlim(-.15,1.4);ax.set_title(f'{cohort.title()}, {arch}');ax.set_ylabel('Held-out latent-action MSE');ax.legend()
    figure(fig,'03_varied_fidelity_gaps')
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,m,title in zip(axes,[PROGRESS,ON],['Progress change (m / minute)','On-policy MSE change']):
        e=ef[(ef.comparison.str.startswith('Extended minus initializer'))&(ef.metric==m)]
        y=np.arange(len(e));ax.errorbar(e['mean'],y,xerr=np.vstack([e['mean']-e.ci_low,e.ci_high-e['mean']]),fmt='o')
        ax.set_yticks(y,e.comparison.str.replace('Extended minus initializer | ','',regex=False))
        ax.axvline(0,color='gray',lw=1);ax.set_title(title)
    fig.suptitle('Extended fitting: paired changes, 95% conditional round-bootstrap intervals')
    figure(fig,'04_paired_changes')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for label,color,name in [(ORIG[2],COLORS[2],'BC'),(ORIG[4],COLORS[4],'DAgger')]:
        h=pd.DataFrame(read(BATCH/'training'/label/'history.json'));t=tr[tr.label==label].iloc[0]
        axes[0].plot((h.gradient_updates+t.inherited_updates)/1000,h.validation_mse,label=name,color=color)
        axes[0].scatter((t.inherited_updates+t.selected_gradient_updates)/1000,t.validation_mse,color=color,marker='*',s=120)
    axes[0].set_yscale('log');axes[0].set_xlabel('Historical performed + additional updates (thousands)')
    axes[0].set_ylabel('Validation MSE');axes[0].set_title('Original: unequal costs to reach stopping');axes[0].legend()
    for cohort,marker in [('continued','o'),('scratch','s')]:
        for algo,color in [('bc','#0072B2'),('dagger_mixed','#E69F00')]:
            t=tr[(tr.cohort==cohort)&(tr.algorithm==algo)]
            axes[1].scatter(t.total_lineage_gradient_updates/1000,t.validation_mse,c=color,marker=marker,label=f'{cohort} {algo}')
    axes[1].set_yscale('log');axes[1].set_xlabel('Total performed lineage updates (thousands)')
    axes[1].set_ylabel('Selected validation MSE');axes[1].set_title('Varied: all seeds and both input architectures');axes[1].legend(fontsize=8)
    figure(fig,'05_resources_and_stopping')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    els=['oracle','zero_native']+engine+['random'];e=means[means.suite=='sampled'].set_index('label').loc[els]
    for ax,metric,title in zip(axes,[PROGRESS,'total_falls',SWD],['Progress (m / minute)','Falls across 30 minutes','Joint-occupancy SWD']):
        bars=ax.bar(['Oracle','Zero','Net 42','Net 43','Net 44','IID random'],e[metric],color=['#009E73']+['#888888']*4+['#D55E00'])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.set_title(title);ax.tick_params(axis='x',rotation=30)
    figure(fig,'06_random_network_engine')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,m,title in zip(axes,[SWD,'jerk_cost_rate_rad2_per_s6'],['Joint-occupancy distance','Kinematic jerk rate (rad²/s⁶)']):
        for cohort,marker in [('continued','o'),('scratch','s')]:
            for algo,color in [('bc','#0072B2'),('dagger_mixed','#E69F00')]:
                ls=tr[(tr.cohort==cohort)&(tr.algorithm==algo)].label
                a=means[(means.suite=='sampled')&means.label.isin(ls)]
                ax.scatter(a[MSE],a[m],c=color,marker=marker,s=35+12*a.total_falls,label=f'{cohort} {algo}')
        ax.set_xscale('log');ax.set_xlabel('Held-out latent-action MSE');ax.set_ylabel(title)
    axes[0].legend(fontsize=8);fig.suptitle('Each point is one checkpoint; larger markers indicate more falls')
    figure(fig,'07_fidelity_mechanics')
    probe_analysis(figure, jobs, tr)
    pdf.close()
    files=SOURCES+[PROBE/'plan.json',PROBE/'report.json',PROBE/'initializer_geometry.json',Path(__file__),OUT/'probe.py']
    files+=list((BATCH/'training').glob('*/history.json'))+list((BATCH/'training').glob('*/metrics.json'))
    save(OUT/'manifest.json',dict(sources={str(p.relative_to(ROOT)):digest(p) for p in files},
         deployments=3205,new_deployments=982,trained_checkpoints=26,engine_checkpoints=3,
         bootstrap_seed=20261030,bootstrap_resamples=5000,
         interval_scope='Conditional on observed checkpoints; suite x parent Gaussian stratified paired rounds; no multiplicity correction',
         python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    print(f'[EDA] {OUT}',flush=True)


def probe_analysis(figure,jobs,tr):
    report=read(PROBE/'report.json');p=pd.DataFrame(report['results'])
    true=p[p.labels=='true'].copy();true.to_csv(OUT/'probe_scores_true.csv',index=False)
    p[p.labels!='true'].to_csv(OUT/'probe_scores_shuffled.csv',index=False)
    lookup=true.set_index(['representation','target'])
    deltas=[]
    for label,j in jobs.items():
        for layer in (1,2):
            for target in true.target.unique():
                a=lookup.loc[(f'{label}_h{layer}',target)]
                b=lookup.loc[(f'{j["reference_label"]}_h{layer}',target)]
                raw='raw_observation_with_context' if j['architecture']=='input' else 'raw_observation'
                random=[lookup.loc[(f'random_{j["architecture"]}_{s}_h{layer}',target)].r2 for s in (42,43,44)]
                null=p[(p.representation==f'{label}_h{layer}')&(p.target==target)&(p.labels!='true')].r2.mean()
                deltas.append(dict(label=label,reference=j['reference_label'],layer=layer,target=target,
                    cohort=j['cohort'],arch=j['architecture'],algorithm=j['algorithm'],seed=j['seed'],
                    r2=a.r2,reference_r2=b.r2,r2_delta=a.r2-b.r2,rmse=a.rmse,
                    raw_r2=lookup.loc[(raw,target)].r2,random_r2=np.mean(random),shuffled_r2=null,
                    expert_r2=lookup.loc[('expert_h1',target)].r2))
    d=pd.DataFrame(deltas);d.to_csv(OUT/'probe_changes.csv',index=False)
    pd.DataFrame(report['expert_comparison']).to_csv(OUT/'expert_comparison.csv',index=False)
    pd.DataFrame(report['persistence']).to_csv(OUT/'probe_persistence.csv',index=False)
    geometry=pd.DataFrame(read(PROBE/'initializer_geometry.json'))
    geometry.drop(columns=['per_run_cka']).to_csv(OUT/'initializer_cka.csv',index=False)
    pd.DataFrame([{k:v for k,v in g.items() if k in ['representation','expert_cka']} for g in report['geometry']]).to_csv(OUT/'expert_cka.csv',index=False)
    # All predefined targets, not only the most favorable properties.
    for layer in (1,2):
        fig,ax=plt.subplots(figsize=(14,8))
        x=d[(d.layer==layer)&(d.cohort!='original')].copy()
        x['family']=x.cohort+' / '+x.arch+' / '+x.algorithm
        mat=x.pivot_table(index='target',columns='family',values='r2_delta')
        v=max(.01,float(np.abs(mat.to_numpy()).max()))
        im=ax.imshow(mat.to_numpy(),cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto')
        ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=30,ha='right',fontsize=8)
        ax.set_yticks(range(len(mat)),mat.index,fontsize=8)
        ax.set_title(f'Layer {layer}: R² change after extended fitting, averaged over three observed seeds')
        fig.colorbar(im,ax=ax,label='R² after minus before (not percentage retained)')
        figure(fig,f'0{7+layer}_all_probe_changes_h{layer}')
    targets=['phase_sin_times_roll_deg','phase_cos_times_pitch_deg','future_200ms_forward_velocity_mps',
             'future_200ms_roll_deg','left_force_norm_N','command_forward_displacement_m']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,target in zip(axes.flat,targets):
        for layer,style in [(1,'-'),(2,'--')]:
            y=[lookup.loc[(f'{l}_h{layer}',target)].r2 for l in ORIG]
            line=ax.plot(range(3),y[:3],'o'+style,label=f'Layer {layer}')[0]
            ax.plot([3,4],y[3:],'o'+style,color=line.get_color())
        ax.axhline(lookup.loc[('raw_observation',target)].r2,color='gray',label='Raw observations')
        ax.axhline(np.mean([lookup.loc[(f'random_hidden_{s}_h1',target)].r2 for s in (42,43,44)]),color='purple',ls=':',label='Random layer 1')
        ax.set_xticks(range(5),NAMES,rotation=30,ha='right',fontsize=8)
        ax.set_title(target.replace('_',' '),fontsize=9);ax.set_ylabel('Held-out R²')
    axes[0,0].legend(fontsize=8)
    figure(fig,'10_probes_original_controls')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for layer,ax in zip((1,2),axes):
        g=geometry[geometry.layer==layer].merge(tr[['label','validation_reduction_pct']],left_on='policy',right_on='label')
        ax.scatter(g.mean_per_run_cka,g.validation_reduction_pct,c='#0072B2')
        ax.set_xlabel('Mean within-run CKA to initializer');ax.set_ylabel('Validation MSE reduction (%)')
        ax.set_title(f'Layer {layer}: geometry versus output improvement')
    figure(fig,'11_similarity_and_fidelity')


if __name__=='__main__':
    main()

"""Fit only new scratch representations; reuse verified parent probes and controls."""
import argparse
import copy
import csv
from pathlib import Path
import numpy as np
import torch

from consolidated_probing.run import prepare, inputs, SEEDS, OUTPUT as PARENT_PROBES
from consolidated.models import load_policy
from probing.run import features, fit_probe, shuffle_blocks
from representation_probing.run import scores
from representation_probing.expert import expert_features, cka, spectrum
from variable_training.common import ROOT, read, save, digest, verified, commit
from consolidated.run import DEFAULT, immutable
from .run import Parent, ScratchExperiment


def parent_probes(directory,spec):
    if read(directory/'plan.json')!=spec:
        raise ValueError('Parent probe plan changed; cannot reuse old fits')
    rows=[];receipts={}
    files=sorted((directory/'probes').glob('*/*/completed.json'))
    if len(files)!=300:raise ValueError('Require all300 completed parent probe fits')
    for receipt in files:
        d=receipt.parent;label=d.parent.name;mode=d.name
        identity=dict(plan_sha256=digest(directory/'plan.json'),representation=label,labels=mode)
        if not verified(d,identity):raise ValueError(f'Incomplete parent probe: {d}')
        rows.extend(read(d/'metrics.json'));receipts[str(receipt)]=digest(receipt)
    report=read(directory/'report.json')
    # Report order is not a contract, but exact score content is.
    key=lambda r:(r['representation'],r['labels'],r['target'])
    if sorted(rows,key=key)!=sorted(report['results'],key=key):raise ValueError('Parent probe report mismatch')
    return rows,report,receipts


def new_features(controllers,spec,data):
    for c in controllers:
        model=load_policy(c['model'],c['kind'],c['architecture'])
        net=model.actor_mean if c['kind']=='gail' else model.net
        xs=inputs(data,spec,c['architecture'])
        layers={s:features(net,x) for s,x in xs.items()}
        for i in (0,1):yield c['label']+f'_h{i+1}',{s:h[i] for s,h in layers.items()}


def geometry(xs,teacher,groups,label):
    x=xs['test'];per_run={};null={}
    for g in np.unique(groups):
        mask=groups==g;xx=x[mask];tt=teacher[mask]
        per_run[str(int(g))]=cka(tt,xx)
        null[str(int(g))]=cka(tt,xx[np.random.default_rng(2026+int(g)).permutation(len(xx))])
    return dict(representation=label,expert_cka=cka(teacher,x),per_run_cka=per_run,
                permuted_per_run_cka=null,spectrum=spectrum(x))


def execute(output,plan,parent_spec,data,weights,old_rows,old_report,controllers,report_only=False):
    rows=list(old_rows);geo=copy.deepcopy(old_report['geometry'])
    truth={s:d[1] for s,d in data.items()};groups={s:d[2] for s,d in data.items()}
    teacher=expert_features(data['test'][0],weights)[0]
    for label,xs in new_features(controllers,parent_spec,data):
        for seed in [None]+SEEDS:
            mode='true' if seed is None else f'shuffled_{seed}'
            directory=output/'probes'/label/mode
            ident=dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)
            if not verified(directory,ident):
                if report_only:raise ValueError(f'Missing new probe: {directory}')
                if directory.exists() and any(directory.iterdir()):raise ValueError(f'Archive incomplete probe first: {directory}')
                ys=truth if seed is None else {s:shuffle_blocks(y,groups[s],seed+1000*i) for i,(s,y) in enumerate(truth.items())}
                pred,fit=fit_probe(xs,ys,groups,parent_spec['alphas'])
                metrics=scores(ys['test'],pred,ys['train'],groups['test'],parent_spec['target_names'])
                for k,r in enumerate(metrics):r.update(representation=label,labels=mode,alpha=float(fit['alpha'][k]))
                directory.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(directory/'probe.npz',**fit)
                np.savez_compressed(directory/'predictions.npz',predictions=pred,targets=ys['test'],
                                    trajectory_index=groups['test'],step_index=data['test'][3])
                save(directory/'metrics.json',metrics);commit(directory,ident)
                print(f'[New probe] {label}/{mode}',flush=True)
            else:print(f'[Resume probe] {label}/{mode}',flush=True)
            rows.extend(read(directory/'metrics.json'))
        geo.append(geometry(xs,teacher,groups['test'],label))
    lookup={(r['representation'],r['target']):r for r in rows if r['labels']=='true'}
    comparisons=[];paired=[]
    labels={c['label']:c for c in controllers}
    for (label,target),r in lookup.items():
        base=label.rsplit('_h',1)[0]
        if not base.startswith(('original_','varied_','scratch_')):continue
        e=lookup[('expert_h1',target)]
        comparisons.append(dict(representation=label,target=target,student_r2=r['r2'],expert_r2=e['r2'],
                                r2_gap=r['r2']-e['r2'] if r['r2'] is not None and e['r2'] is not None else None))
        if base in labels and labels[base]['algorithm']!='gail_random':
            ref=label.removeprefix('scratch_');p=lookup[(ref,target)]
            paired.append(dict(representation=label,reference=ref,target=target,
                               r2_delta=r['r2']-p['r2'] if r['r2'] is not None and p['r2'] is not None else None,
                               rmse_delta=r['rmse']-p['rmse']))
    save(output/'report.json',dict(protocol=plan['protocol'],results=rows,geometry=geo,
         expert_comparison=comparisons,initialization_comparison=paired,persistence=old_report['persistence'],
         reused_parent=plan['parent_probes'],parent_report_sha256=plan['parent_report_sha256']))
    for name,records in [('probe_scores.csv',rows),('expert_comparison.csv',comparisons),('initialization_comparison.csv',paired)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Scratch and continuation representation comparison','',
           '300 verified parent fits reused; only new scratch hidden layers are fitted.',
           'Identical shared oracle states, splits, targets, penalties and controls. No slip prediction target.',
           'Positive R² delta indicates more accessible information, not causal control usefulness.',
           '', '| Scratch representation | Property | R² | RMSE |','|---|---|---:|---:|']
    lines += [f'| {r["representation"]} | {r["target"]} | {r["r2"]} | {r["rmse"]:.5g} |'
              for r in rows if r['labels']=='true' and r['representation'].startswith('scratch_')]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output/"report.md"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','run','report'])
    p.add_argument('--batch',type=Path,default=DEFAULT)
    p.add_argument('--scratch',type=Path)
    p.add_argument('--parent-probes',type=Path,default=PARENT_PROBES)
    p.add_argument('--output',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    parent=Parent(args.batch)
    root=(args.scratch or parent.root/'scratch_v1').resolve()
    exp=ScratchExperiment.readonly(root,parent)
    parent_probe_root=args.parent_probes.resolve()
    stored=read(parent_probe_root/'plan.json')
    spec,data,weights=prepare(parent.root/'evaluation/all_all_correct',stored['stride'])
    old_rows,old_report,receipts=parent_probes(parent_probe_root,spec)
    controllers=exp.scratch_controllers()
    plan=dict(protocol='consolidated_scratch_probes_v1',parent_probes=str(parent_probe_root),
              parent_plan_sha256=digest(parent_probe_root/'plan.json'),parent_report_sha256=digest(parent_probe_root/'report.json'),
              parent_receipts=receipts,controllers=controllers,scratch_identity=exp.identity,
              sources={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'scratch_comparison').glob('*.py')})
    output=(args.output or root/'probing').resolve()
    if output==parent_probe_root or parent_probe_root.is_relative_to(output):raise ValueError('Cannot overwrite parent probes')
    print(f'[Reuse] 300 parent fits; {len(controllers)*8} new fits on matching oracle states.')
    if args.stage=='audit':return
    immutable(output/'plan.json',plan)
    execute(output,plan,spec,data,weights,old_rows,old_report,controllers,args.stage=='report')


if __name__=='__main__':main()

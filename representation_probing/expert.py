"""Append expert probes and teacher/student comparisons without changing v1."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .run import OUTPUT as BASELINE, spec_for, scores, representations
from probing.run import fit_probe, shuffle_blocks
from variable_training.common import ROOT, read, save, digest, verified, commit
from math_ops.Neural_Network import run_mlp

EXPERT=ROOT/'behaviors/custom/Walk/walk_R0.pkl'
OUTPUT=ROOT/'experiments/representation_probing/nao_expert_comparison_v1'


def load_expert(path=EXPERT):
    # Only the repository's trusted local oracle pickle; no arbitrary CLI pickle.
    with Path(path).open('rb') as f:weights=pickle.load(f)
    if len(weights)!=2:raise ValueError('Expected one expert hidden layer')
    for layer,(nout,nin) in zip(weights,[(64,63),(16,64)]):
        if np.shape(layer[0])!=(nout,) or np.shape(layer[1])!=(nout,nin):
            raise ValueError('Unexpected expert architecture')
        if not all(np.isfinite(v).all() for v in layer[:2]):raise ValueError('Invalid expert weights')
    return weights


def expert_features(x,weights):
    x=np.asarray(x,dtype=np.float32)
    h=np.tanh(x@weights[0][1].T+weights[0][0])
    output=h@weights[1][1].T+weights[1][0]
    return h.astype(np.float64),output


def random_expert(x,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);layer=nn.Linear(63,64)
    with torch.inference_mode():
        return torch.tanh(layer(torch.as_tensor(x,dtype=torch.float32))).numpy().astype(np.float64)


def parent_rows(baseline,plan):
    rows=[];receipts={}
    labels=['raw_observation','phase_command_only']+[f'{n}_h{i}' for n in list(plan['models'])+
            [f'random_seed_{s}' for s in plan['control_seeds']] for i in (1,2)]
    for label in labels:
        for mode in ['true']+[f'shuffled_{s}' for s in plan['control_seeds']]:
            d=baseline/'probes'/label/mode
            ident=dict(plan_sha256=digest(baseline/'plan.json'),representation=label,labels=mode)
            if not verified(d,ident):raise ValueError(f'Missing parent: {d}')
            rows+=read(d/'metrics.json');receipts[str(d/'completed.json')]=digest(d/'completed.json')
    d=baseline/'interventions'
    if not verified(d,dict(plan_sha256=digest(baseline/'plan.json'),task='layer_reset_action_sensitivity')):
        raise ValueError('Incomplete parent interventions')
    receipts[str(d/'completed.json')]=digest(d/'completed.json')
    return rows,receipts


def prepare(baseline):
    previous=read(baseline/'plan.json')
    current,data=spec_for(Path(previous['data']),previous['stride'])
    if current!=previous:raise ValueError('Parent inputs changed; cannot append')
    rows,receipts=parent_rows(baseline,previous)
    weights=load_expert();maxerr=0.
    for entry in previous['entries']:
        with np.load(entry['path'],allow_pickle=False) as trace:
            # Check all source observations, not just selected probe rows.
            _,pred=expert_features(trace['obs'],weights)
            if not np.allclose(pred,trace['oracle_actions'],rtol=1e-4,atol=1e-5):
                raise ValueError('Oracle checkpoint does not reproduce recorded actions')
            maxerr=max(maxerr,float(np.abs(pred-trace['oracle_actions']).max()))
    sources=dict(previous['sources'])
    for p in (Path(__file__),ROOT/'math_ops/Neural_Network.py'):
        sources[str(p.relative_to(ROOT))]=digest(p)
    spec=dict(protocol='nao_expert_append_v1',baseline=str(baseline),
        baseline_plan_sha256=digest(baseline/'plan.json'),parent_receipts=receipts,
        expert_path=str(EXPERT),expert_sha256=digest(EXPERT),expert_architecture='63-64tanh-16linear',
        expert_action_max_abs_error=maxerr,sources=sources,
        new_representations=['expert_h1']+[f'expert_random_seed_{s}_h1' for s in previous['control_seeds']],
        target_names=previous['target_names'],alphas=previous['alphas'],control_seeds=previous['control_seeds'],
        geometry='unscaled centered linear CKA, pooled and per-run; within-run row permutation seed2026',
        numpy_version=np.__version__,torch_version=str(torch.__version__))
    return spec,previous,data,rows,weights


def cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    denominator=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.clip(np.sum((x.T@y)**2)/denominator,0,1)) if denominator>1e-20 else None


def spectrum(x):
    centered=x-x.mean(0)
    eig=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(x)),0)
    total=eig.sum()
    return dict(width=x.shape[1],constant_fraction=float(np.mean(np.std(x,axis=0)<1e-8)),
                participation_ratio=float(total**2/(eig@eig)) if total>1e-20 else None,
                eigenvalues_descending=eig[::-1].tolist())


def geometry(previous,data,weights):
    teacher=expert_features(data['test'][0],weights)[0];groups=data['test'][2]
    output=[]
    featureset=list(representations(previous,data))
    featureset += [(f'expert_random_seed_{seed}_h1',{'test':random_expert(data['test'][0],seed)})
                   for seed in previous['control_seeds']]
    featureset += [('expert_h1',{'test':teacher})]
    for label,arrays in featureset:
        x=arrays['test'];perrun={};null={}
        for g in np.unique(groups):
            mask=groups==g;tx=teacher[mask];xx=x[mask]
            perrun[str(int(g))]=cka(tx,xx)
            perm=np.random.default_rng(2026+int(g)).permutation(len(xx))
            null[str(int(g))]=cka(tx,xx[perm])
        valid=[v for v in perrun.values() if v is not None]
        output.append(dict(representation=label,expert_cka=cka(teacher,x),
            per_run_cka=perrun,mean_run_cka=float(np.mean(valid)) if valid else None,
            permuted_per_run_cka=null,spectrum=spectrum(x)))
    return output


def comparisons(rows):
    truth={(r['representation'],r['target']):r for r in rows if r['labels']=='true'}
    result=[]
    for (label,target),r in truth.items():
        if not any(label.startswith(n+'_h') for n in ('bc','dagger_mixed','gail_bc','gail_dagger')):continue
        e=truth[('expert_h1',target)]
        result.append(dict(representation=label,target=target,student_r2=r['r2'],expert_r2=e['r2'],
            r2_gap=r['r2']-e['r2'] if r['r2'] is not None and e['r2'] is not None else None,
            student_rmse=r['rmse'],expert_rmse=e['rmse'],
            rmse_ratio_to_expert=r['rmse']/e['rmse'] if e['rmse']>1e-12 else None,
            macro_run_rmse_gap=r['macro_run_rmse']-e['macro_run_rmse']))
    return result


def run(output,spec,previous,data,rows,weights):
    groups={s:d[2] for s,d in data.items()};truth={s:d[1] for s,d in data.items()}
    for label in spec['new_representations']:
        xs={s:expert_features(d[0],weights)[0] if label=='expert_h1' else
            random_expert(d[0],int(label.split('_')[3])) for s,d in data.items()}
        for seed in [None]+spec['control_seeds']:
            mode='true' if seed is None else f'shuffled_{seed}'
            d=output/'probes'/label/mode
            ident=dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)
            if verified(d,ident):continue
            if d.exists() and any(d.iterdir()):raise ValueError(f'Archive incomplete subprobe: {d}')
            ys=truth if seed is None else {s:shuffle_blocks(y,groups[s],seed+1000*i) for i,(s,y) in enumerate(truth.items())}
            pred,fit=fit_probe(xs,ys,groups,spec['alphas'])
            metrics=scores(ys['test'],pred,ys['train'],groups['test'],spec['target_names'])
            for k,r in enumerate(metrics):r.update(representation=label,labels=mode,alpha=float(fit['alpha'][k]))
            d.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(d/'probe.npz',**fit)
            np.savez_compressed(d/'predictions.npz',predictions=pred,targets=ys['test'],
                                trajectory_index=groups['test'],step_index=data['test'][3])
            save(d/'metrics.json',metrics);commit(d,ident)
            print(f'[Expert probe] {label}/{mode}',flush=True)
    d=output/'geometry';ident=dict(plan_sha256=digest(output/'plan.json'),task='geometry')
    if not verified(d,ident):
        if d.exists() and any(d.iterdir()):raise ValueError('Archive incomplete geometry output')
        save(d/'metrics.json',geometry(previous,data,weights));commit(d,ident)
    report(output,spec,previous,data,rows,weights)


def report(output,spec,previous,data,rows,weights):
    rows=list(rows)
    for label in spec['new_representations']:
        for mode in ['true']+[f'shuffled_{s}' for s in spec['control_seeds']]:
            d=output/'probes'/label/mode
            if not verified(d,dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)):
                raise ValueError(f'Missing probe: {d}')
            rows+=read(d/'metrics.json')
    d=output/'geometry'
    if not verified(d,dict(plan_sha256=digest(output/'plan.json'),task='geometry')):raise ValueError('Missing geometry')
    differences=comparisons(rows)
    save(output/'report.json',dict(results=rows,expert_comparison=differences,geometry=read(d/'metrics.json'),
        note='Retained decodability and representation similarity, not percent knowledge transferred or R1 benefit.'))
    import csv
    for name,records in [('expert_comparison.csv',differences),('probe_scores.csv',rows)]:
        with (output/name).open('w',newline='') as f:
            if records:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Expert and learned source representations','',
        'Expert: one64-D tanh layer; students: two256-D ReLU layers. Both student layers compared to expert h1.',
        'Completed student probes are reused. All30 general properties retained; no slip target.',
        'Positive R2 gap means better linear decoding, not more transferred knowledge. CKA is descriptive, not causal.',
        '', '| Student layer | Property | Expert R2 | Student R2 | R2 gap | RMSE ratio |',
        '|---|---|---:|---:|---:|---:|']
    for r in differences:
        fmt=lambda v:'undefined' if v is None else f'{v:.4f}'
        lines.append('| '+' | '.join([r['representation'],r['target']]+[fmt(r[k]) for k in
                     ['expert_r2','student_r2','r2_gap','rmse_ratio_to_expert']])+' |')
    lines+=['','## Whole-layer structure','','| Representation | Pooled expert CKA | Mean within-run CKA |','|---|---:|---:|']
    for r in read(d/'metrics.json'):
        lines.append(f'| {r["representation"]} | {r["expert_cka"]} | {r["mean_run_cka"]} |')
    lines+=['','Controls, spectra and per-run/permuted CKA: report.json. References and limitations: representation_probing/EXPERT.md.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');print(output/'report.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','run','report'])
    p.add_argument('--baseline',type=Path,default=BASELINE);p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args();torch.set_num_threads(1)
    prepared=prepare(args.baseline.resolve());spec=prepared[0]
    if args.stage=='audit':
        print('Expert:',spec['expert_architecture'],'max action error:',spec['expert_action_max_abs_error'])
        print('Parent receipts verified;16 new fits; existing64 reused.');return
    output=args.output.resolve()
    if (output/'plan.json').exists():
        if read(output/'plan.json')!=spec:raise ValueError('Inputs changed; use new --output')
    elif args.stage=='report':raise ValueError('Missing plan')
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('Nonempty output')
        save(output/'plan.json',spec)
    (run if args.stage=='run' else report)(output,*prepared)


if __name__=='__main__':main()

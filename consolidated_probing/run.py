"""Shared-state locomotion regression probes for all consolidated actors and expert."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from consolidated.models import load_policy
from probing.run import features, fit_probe, shuffle_blocks
from representation_probing.run import dataset, scores
from representation_probing.expert import EXPERT, load_expert, expert_features, random_expert, cka, spectrum
from variable_training.common import ROOT, read, save, digest, verified, commit

DEFAULT = ROOT/'experiments/consolidated/v1/evaluation/all_all_correct'
OUTPUT = ROOT/'experiments/representation_probing/consolidated_v1'
SEEDS = [42, 43, 44]


def prepare(source, stride):
    if stride < 1:
        raise ValueError('Stride must be positive')
    schedule = read(source/'schedule.json')
    if schedule['identity']['protocol'] != 'consolidated_imitation_v1':
        raise ValueError('Expected consolidated evaluation')
    if not (source/'comparison.json').is_file():
        raise ValueError('Finish evaluation/report first')
    oracle = next(c for c in schedule['controllers'] if c['kind'] == 'oracle')
    entries, seen, seeds = [], set(), set()
    weights = load_expert()
    for row in schedule['rows']:
        if row['suite'] not in ('grouped', 'confirmation'):
            continue
        directory = source/'runs'/row['id']/oracle['label']
        receipt = read(directory/'completed.json')
        identity = receipt['identity']
        job = identity['spec']
        if (job['row'] != row or job['controller'] != oracle or
                job['steps'] != schedule['steps'] or
                any(identity.get(k) != v for k,v in schedule['identity'].items())):
            raise ValueError(f'Oracle identity mismatch: {directory}')
        if not verified(directory, identity):
            raise ValueError(f'Incomplete oracle: {directory}')
        with np.load(directory/'trace.npz', allow_pickle=False) as trace:
            if len(trace['obs']) != schedule['steps']:
                raise ValueError('Incomplete trajectory')
            if not np.allclose(expert_features(trace['obs'], weights)[1], trace['oracle_actions'], rtol=1e-4, atol=1e-5):
                raise ValueError('Expert does not reproduce recorded actions')
        sha = digest(directory/'trace.npz')
        if sha in seen or row['seed'] in seeds:
            raise ValueError('Duplicate trajectory or seed')
        seen.add(sha); seeds.add(row['seed'])
        split = 'test' if row['suite']=='confirmation' else ('validation' if row['id'].endswith('_draw_2') else 'train')
        entries.append(dict(path=str(directory/'trace.npz'), sha256=sha, row=row, split=split,
                            receipt_sha256=digest(directory/'completed.json')))
    if [sum(e['split']==s for e in entries) for s in ('train','validation','test')] != [10,5,15]:
        raise ValueError('Require full 10/5/15 oracle-run split; smoke is insufficient')
    models = {}
    for controller in schedule['controllers']:
        if controller['kind'] in ('oracle','random'):
            continue
        if controller['mode'] not in ('none','correct'):
            continue
        if digest(controller['model']) != controller['sha256']:
            raise ValueError('Changed evaluated checkpoint')
        load_policy(controller['model'],controller['kind'],controller['architecture'])
        models[controller['label']] = controller
    if len(models) != 28:
        raise ValueError('Expected all28 consolidated policies')
    spec = dict(protocol='consolidated_general_probes_v1', data=str(source),
                schedule_sha256=digest(source/'schedule.json'), entries=entries, models=models,
                stride=stride, horizon=10, control_seeds=SEEDS,
                alphas=[1e-6,1e-5,1e-4,.001,.01,.1,1.,10.,100.],
                expert_sha256=digest(EXPERT), numpy_version=np.__version__, torch_version=str(torch.__version__),
                sources={str(p.relative_to(ROOT)):digest(p) for p in
                         [Path(__file__), ROOT/'probing/run.py', ROOT/'representation_probing/run.py',
                          ROOT/'representation_probing/expert.py', ROOT/'consolidated/models.py',
                          ROOT/'context_training/models.py', ROOT/'behaviors/custom/Walk/Env.py']})
    arrays, names = dataset(spec)
    spec['target_names'] = names
    spec['split_counts'] = {s:dict(rows=len(d[0]),runs=len(np.unique(d[2]))) for s,d in arrays.items()}
    return spec, arrays, weights


def inputs(data, spec, architecture):
    result = {}
    for split, d in data.items():
        x = d[0]
        if architecture == 'input':
            slips = np.array([float(f'{spec["entries"][int(g)]["row"]["slip"]:.12g}') for g in d[2]])
            x = np.column_stack([x,np.log1p(slips)/np.log(11)])
        result[split] = x
    return result


def random_actor(width, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(width,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,16))
    return net.eval().requires_grad_(False)


def representations(spec, data, weights):
    raw = inputs(data,spec,'hidden')
    contextual = inputs(data,spec,'input')
    yield 'raw_observation', raw
    yield 'raw_observation_with_context', contextual
    yield 'phase_command_only', {s:x[:,55:61] for s,x in raw.items()}
    yield 'expert_h1', {s:expert_features(x,weights)[0] for s,x in raw.items()}
    for seed in SEEDS:
        yield f'expert_random_{seed}_h1', {s:random_expert(x,seed) for s,x in raw.items()}
    for label, meta in spec['models'].items():
        policy = load_policy(meta['model'],meta['kind'],meta['architecture'])
        net = policy.actor_mean if meta['kind']=='gail' else policy.net
        xs = contextual if meta['architecture']=='input' else raw
        h = {s:features(net,x) for s,x in xs.items()}
        for i in (0,1):
            yield f'{label}_h{i+1}', {s:v[i] for s,v in h.items()}
    for architecture, xs, width in [('hidden',raw,63),('input',contextual,64)]:
        for seed in SEEDS:
            net = random_actor(width,seed)
            h = {s:features(net,x) for s,x in xs.items()}
            for i in (0,1):
                yield f'random_{architecture}_{seed}_h{i+1}', {s:v[i] for s,v in h.items()}


def execute(output,spec,data,weights,report_only=False):
    groups={s:d[2] for s,d in data.items()}
    truth={s:d[1] for s,d in data.items()}
    teacher=expert_features(data['test'][0],weights)[0]
    rows, geometry = [], []
    for label,xs in representations(spec,data,weights):
        for seed in [None]+SEEDS:
            mode='true' if seed is None else f'shuffled_{seed}'
            directory=output/'probes'/label/mode
            identity=dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)
            if not verified(directory,identity):
                if report_only: raise ValueError(f'Incomplete probe: {directory}')
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError(f'Archive incomplete probe before retry: {directory}')
                ys=truth if seed is None else {s:shuffle_blocks(y,groups[s],seed+1000*i) for i,(s,y) in enumerate(truth.items())}
                pred,fit=fit_probe(xs,ys,groups,spec['alphas'])
                metrics=scores(ys['test'],pred,ys['train'],groups['test'],spec['target_names'])
                for k,r in enumerate(metrics):
                    r.update(representation=label,labels=mode,alpha=float(fit['alpha'][k]))
                directory.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(directory/'probe.npz',**fit)
                np.savez_compressed(directory/'predictions.npz',predictions=pred,targets=ys['test'],
                                    trajectory_index=groups['test'],step_index=data['test'][3])
                save(directory/'metrics.json',metrics);commit(directory,identity)
                print(f'[Probe] {label}/{mode}',flush=True)
            rows.extend(read(directory/'metrics.json'))
        x=xs['test'];per_run={};null={}
        for g in np.unique(groups['test']):
            mask=groups['test']==g;xx=x[mask];tt=teacher[mask]
            per_run[str(int(g))]=cka(tt,xx)
            null[str(int(g))]=cka(tt,xx[np.random.default_rng(2026+int(g)).permutation(len(xx))])
        geometry.append(dict(representation=label,expert_cka=cka(teacher,x),per_run_cka=per_run,
                             permuted_per_run_cka=null,spectrum=spectrum(x)))
    expert={r['target']:r for r in rows if r['representation']=='expert_h1' and r['labels']=='true'}
    comparisons=[]
    for r in rows:
        if r['labels']!='true' or r['representation'].rsplit('_h',1)[0] not in spec['models']: continue
        e=expert[r['target']]
        comparisons.append(dict(representation=r['representation'],target=r['target'],student_r2=r['r2'],expert_r2=e['r2'],
                                r2_gap=r['r2']-e['r2'] if r['r2'] is not None and e['r2'] is not None else None))
    persistence=scores(data['test'][1][:,-6:],data['test'][4],data['train'][1][:,-6:],groups['test'],spec['target_names'][-6:])
    save(output/'report.json',dict(results=rows,expert_comparison=comparisons,geometry=geometry,persistence=persistence))
    for name,records in [('probe_scores.csv',rows),('expert_comparison.csv',comparisons)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Consolidated locomotion representation probes','',
           'Shared oracle states;30 locomotion properties; no slip target. Replicas retained separately.',
           'Decodability/CKA are not percentages of knowledge transferred or causal evidence.',
           '', '| Representation | Property | R2 | RMSE |','|---|---|---:|---:|']
    lines += [f'| {r["representation"]} | {r["target"]} | {r["r2"]} | {r["rmse"]:.5g} |' for r in rows if r['labels']=='true']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output / "report.md"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','run','report'])
    p.add_argument('--data',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--stride',type=int,default=7)
    args=p.parse_args();torch.set_num_threads(1)
    spec,data,weights=prepare(args.data.resolve(),args.stride)
    if args.stage=='audit':
        print(f'Models: {len(spec["models"])} + expert;75 representations x4 controls =300 multitarget fits')
        print(spec['split_counts']);return
    output=args.output.resolve()
    if (output/'plan.json').exists():
        if read(output/'plan.json')!=spec: raise ValueError('Inputs changed; choose a new output')
    elif args.stage=='report': raise ValueError('Missing plan')
    else:
        if output.exists() and any(output.iterdir()): raise ValueError('Nonempty unversioned output')
        save(output/'plan.json',spec)
    execute(output,spec,data,weights,args.stage=='report')


if __name__=='__main__':
    main()

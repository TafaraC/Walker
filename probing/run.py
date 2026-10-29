"""Linear regression probes on shared, receipt-verified oracle trajectories."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from context_training.evaluate import job_identity, validate_result
from variable_training.common import ROOT, digest, read, save, verified, commit

POLICIES = {'bc': 'bc_walk_policy.pth', 'dagger_mixed': 'dagger_mixed_policy.pth',
            'gail_dagger': 'gail_dagger_policy.pth', 'gail_bc': 'gail_bc_policy.pth'}
TARGETS = ['log1p_slip', 'head_height_m', 'torso_roll_deg', 'torso_pitch_deg',
           'next_200ms_mean_forward_velocity_mps']
DEFAULT_DATA = ROOT/'experiments/context_conditioning/bc_input_v1/evaluation_all_v1'
DEFAULT_OUTPUT = ROOT/'experiments/probing/original_policies_v1'


def actor(path=None, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(63, 256), nn.ReLU(), nn.Linear(256, 256),
                            nn.ReLU(), nn.Linear(256, 16))
    if path is not None:
        state = torch.load(path, map_location='cpu', weights_only=True)
        prefix = 'actor_mean.' if 'actor_mean.0.weight' in state else 'net.'
        net.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Non-finite actor')
    net.eval().requires_grad_(False)
    return net


def features(net, obs):
    layers = [[], []]
    with torch.inference_mode():
        for chunk in np.array_split(obs, max(1, (len(obs)+1023)//1024)):
            h1 = net[:2](torch.as_tensor(chunk, dtype=torch.float32))
            h2 = net[2:4](h1)
            layers[0].append(h1.numpy()); layers[1].append(h2.numpy())
    return [np.concatenate(h).astype(np.float64) for h in layers]


def make_spec(data, stride):
    if stride < 1:
        raise ValueError('stride must be positive')
    plan = read(data/'plan.json')
    if plan['protocol'] != 'context_bc_evaluation_v1':
        raise ValueError('Expected context evaluation oracle traces')
    oracle = next(c for c in plan['controllers'] if c['architecture'] == 'oracle')
    entries, seen, seeds = [], set(), set()
    for row in plan['rounds']:
        if row['suite'] not in ('grouped', 'confirmation'):
            continue
        split = ('test' if row['suite'] == 'confirmation' else
                 'validation' if row['id'].endswith('_draw_2') else 'train')
        directory = data/'runs'/row['id']/oracle['label']
        identity = job_identity(plan, row, oracle)
        if not verified(directory, identity):
            raise ValueError(f'Incomplete oracle run: {directory}')
        validate_result(directory, identity)
        sha = digest(directory/'trace.npz')
        if sha in seen or row['seed'] in seeds:
            raise ValueError('Duplicate trajectory or task seed across probe splits')
        seen.add(sha); seeds.add(row['seed'])
        entries.append(dict(path=str(directory/'trace.npz'), sha256=sha, row=row,
                            split=split, receipt_sha256=digest(directory/'completed.json')))
    if {s: sum(e['split'] == s for e in entries) for s in ('train','validation','test')} != {
            'train': 10, 'validation': 5, 'test': 15}:
        raise ValueError('Expected 10 train/5 validation/15 test whole oracle runs')
    models = {}
    for label, filename in POLICIES.items():
        path = ROOT/filename
        actor(path)
        models[label] = dict(path=str(path), sha256=digest(path))
    return dict(protocol='walker_linear_probes_v1', data_plan_sha256=digest(data/'plan.json'),
                data=str(data), entries=entries, models=models, stride=stride, horizon=10,
                targets=TARGETS, alphas=[.01, .1, 1., 10., 100.], control_seeds=[42,43,44],
                sources={str(p.relative_to(ROOT)): digest(p) for p in
                         [Path(__file__), ROOT/'probing/__init__.py',
                          ROOT/'context_training/evaluate.py', ROOT/'variable_training/common.py',
                          ROOT/'configure_field_slip.py']},
                torch_version=str(torch.__version__), numpy_version=np.__version__,
                lineage='gail_dagger uses original DAgger initialization, NOT mixed DAgger')


def sample_trace(trace, slip, stride, horizon=10):
    obs = trace['obs']
    episode = trace['episode_id']
    if obs.ndim != 2 or obs.shape[1] != 63 or not 0 < slip <= 10:
        raise ValueError('Invalid observation/slip contract')
    # Velocity[t] is the transition t->t+1. Require the endpoint state as well,
    # excluding windows crossing a recovery (including recovery after last action).
    indices = np.arange(0, len(obs)-horizon, stride)
    indices = np.asarray([i for i in indices if np.all(episode[i:i+horizon+1] == episode[i])], dtype=int)
    if len(indices) < 2:
        raise ValueError('Too few uninterrupted samples')
    targets = np.column_stack([np.full(len(indices), np.log1p(slip)),
        trace['head_height_m'][indices], trace['torso_roll_deg'][indices],
        trace['torso_pitch_deg'][indices],
        [trace['forward_velocity_mps'][i:i+horizon].mean() for i in indices]])
    if not np.isfinite(obs).all() or not np.isfinite(targets).all():
        raise ValueError('Non-finite trace')
    return obs[indices].astype(np.float64), targets, indices


def dataset(spec):
    parts = {s: [] for s in ('train','validation','test')}
    for number, entry in enumerate(spec['entries']):
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Trace changed after audit')
        with np.load(entry['path'], allow_pickle=False) as trace:
            x,y,indices = sample_trace(trace, entry['row']['slip'], spec['stride'], spec['horizon'])
        parts[entry['split']].append((x,y,np.full(len(x),number),indices))
    return {s: tuple(np.concatenate([p[k] for p in rows]) for k in range(4))
            for s,rows in parts.items()}


def shuffle_blocks(y, groups, seed):
    """Reassign complete target trajectories independently within each split.

    Interpolate donor indices when recovery filtering gives unequal lengths.
    Preserves coarse temporal structure; not the NLP word-type control task.
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(groups))
    if len(ids) < 2:
        raise ValueError('Need multiple trajectory blocks')
    out = np.empty_like(y)
    for dest, source in zip(ids, np.roll(ids, 1)):
        di, si = np.flatnonzero(groups == dest), np.flatnonzero(groups == source)
        out[di] = y[si[np.rint(np.linspace(0, len(si)-1, len(di))).astype(int)]]
    return out


def macro_mse(y, prediction, groups):
    return np.mean([np.square(y[groups == g]-prediction[groups == g]).mean(axis=0)
                    for g in np.unique(groups)], axis=0)


def fit_probe(xs, ys, groups, alphas):
    xm, scale = xs['train'].mean(0), xs['train'].std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    ym, yscl = ys['train'].mean(0), ys['train'].std(0)
    yscl = np.where(yscl < 1e-8, 1., yscl)
    x = {s: (v-xm)/scale for s,v in xs.items()}
    y = (ys['train']-ym)/yscl
    # Objective mean squared error + alpha*||w||^2; intercept unpenalized.
    eigen, vectors = np.linalg.eigh(x['train'].T@x['train']/len(y))
    rhs = vectors.T@(x['train'].T@y/len(y))
    weights, scores = [], []
    for alpha in alphas:
        w = vectors@(rhs/(np.maximum(eigen,0)[:,None]+alpha))
        weights.append(w)
        scores.append(macro_mse(ys['validation'], (x['validation']@w)*yscl+ym, groups['validation']))
    best = np.argmin(scores, axis=0)
    w = np.column_stack([weights[j][:,k] for k,j in enumerate(best)])
    pred = x['test']@w*yscl+ym
    return pred, dict(weights=w, feature_mean=xm, feature_scale=scale,
                      target_mean=ym, target_scale=yscl, alpha=np.asarray(alphas)[best])


def metrics(y, pred, train_y, groups):
    mse = np.square(y-pred).mean(0)
    variance = np.square(y-y.mean(0)).mean(0)
    baseline = np.broadcast_to(train_y.mean(0), y.shape)
    rows = []
    for k,name in enumerate(TARGETS):
        rows.append(dict(target=name, rmse=float(np.sqrt(mse[k])),
                         mae=float(np.abs(y[:,k]-pred[:,k]).mean()),
                         r2=float(1-mse[k]/variance[k]) if variance[k]>1e-12 else None,
                         macro_run_rmse=float(np.mean([np.sqrt(np.square(y[groups==g,k]-pred[groups==g,k]).mean()) for g in np.unique(groups)])),
                         mean_baseline_rmse=float(np.sqrt(np.square(y[:,k]-baseline[:,k]).mean()))))
    return rows


def representations(spec, data):
    yield 'raw_observation', {s: d[0] for s,d in data.items()}
    models = [(name, meta['path'], None) for name,meta in spec['models'].items()]
    models += [(f'random_seed_{seed}', None, seed) for seed in spec['control_seeds']]
    for label,path,seed in models:
        if path is not None and digest(path) != spec['models'][label]['sha256']:
            raise ValueError('Checkpoint changed after audit')
        net = actor(path, seed or 42)
        activations = {s: features(net,d[0]) for s,d in data.items()}
        for layer in (0,1):
            yield f'{label}_h{layer+1}', {s: v[layer] for s,v in activations.items()}


def run(output, spec):
    data = dataset(spec)
    groups = {s: d[2] for s,d in data.items()}
    true_y = {s: d[1] for s,d in data.items()}
    for label,xs in representations(spec,data):
        for seed in [None]+spec['control_seeds']:
            mode = 'true' if seed is None else f'shuffled_{seed}'
            directory = output/'probes'/label/mode
            identity = dict(plan_sha256=digest(output/'plan.json'), representation=label, labels=mode)
            if verified(directory,identity):
                print(f'[Resume] {label}/{mode}', flush=True); continue
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f'Incomplete output preserved; choose new --output or archive: {directory}')
            ys = true_y if seed is None else {s: shuffle_blocks(y,groups[s],seed+i*1000)
                                            for i,(s,y) in enumerate(true_y.items())}
            prediction, fitted = fit_probe(xs,ys,groups,spec['alphas'])
            scores = metrics(ys['test'],prediction,ys['train'],groups['test'])
            directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'probe.npz', **fitted)
            np.savez_compressed(directory/'test_predictions.npz', predictions=prediction,
                targets=ys['test'], trajectory_index=groups['test'], step_index=data['test'][3])
            for k,r in enumerate(scores):
                r.update(representation=label, labels=mode, alpha=float(fitted['alpha'][k]))
            save(directory/'metrics.json',scores)
            commit(directory,identity)
            print(f'[Probe] {label}/{mode}',flush=True)
    report(output,spec)


def report(output,spec):
    rows=[]
    labels=['raw_observation']+[f'{name}_h{i}' for name in list(spec['models'])+
                              [f'random_seed_{s}' for s in spec['control_seeds']] for i in (1,2)]
    for label in labels:
        for mode in ['true']+[f'shuffled_{s}' for s in spec['control_seeds']]:
            directory=output/'probes'/label/mode
            if not verified(directory,dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)):
                raise ValueError(f'Missing probe: {directory}')
            rows.extend(read(directory/'metrics.json'))
    save(output/'report.json',dict(protocol=spec['protocol'],results=rows))
    with (output/'report.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Frozen-policy linear probes','',
           'Shared oracle states; held-out whole runs. Decodability is not causal use or R1 transfer evidence.',
           'R2 is pooled over rows (not independent trials); inspect per-run prediction artifacts and shuffled controls.',
           '', '| Representation | Target | Test R2 | RMSE | Mean baseline RMSE |',
           '|---|---|---:|---:|---:|']
    for r in rows:
        if r['labels']=='true':
            r2='undefined' if r['r2'] is None else f'{r["r2"]:.4f}'
            lines.append(f'| {r["representation"]} | {r["target"]} | {r2} | {r["rmse"]:.5g} | {r["mean_baseline_rmse"]:.5g} |')
    lines += ['', 'All shuffled-label scores, selected regularization and macro-run RMSE are in report.csv/json.',
              'Method and interpretation limits: probing/README.md. No significance claim from row count.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output/"report.md"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','run','report'))
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--stride',type=int,default=10)
    args=parser.parse_args()
    torch.set_num_threads(1)
    spec=make_spec(args.data.resolve(),args.stride)
    if args.stage=='audit':
        data=dataset(spec)
        print({s:dict(rows=len(d[0]),trajectories=len(np.unique(d[2]))) for s,d in data.items()})
        print(spec['lineage']);return
    output=args.output.resolve()
    if (output/'plan.json').exists():
        if read(output/'plan.json')!=spec:
            raise ValueError('Inputs changed; use a new --output')
    elif args.stage=='report':
        raise ValueError('No completed probing plan')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Refusing nonempty unversioned output')
        save(output/'plan.json',spec)
    (run if args.stage=='run' else report)(output,spec)


if __name__=='__main__':
    main()

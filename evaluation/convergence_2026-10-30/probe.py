"""Probe completed convergence checkpoints and their exact initializers, read-only.

Run from any directory with the fcportugal interpreter. Raw fits stay in experiments.
The existing regression, control-task, split and CKA implementations are reused.
"""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from consolidated_probing.run import prepare, execute, inputs
from consolidated.models import load_policy
from probing.run import features
from representation_probing.expert import cka
from variable_training.common import read, save, digest, verified
from consolidated.run import immutable

BATCH = ROOT/'experiments/convergence_controls/all_v1'
OUTPUT = ROOT/'experiments/representation_probing/convergence_all_v1'


def audit(batch):
    schedule = read(batch/'evaluation/schedule_all.json')
    if len(schedule['controllers']) != 29 or schedule['steps'] != 3000:
        raise ValueError('Expected the full 26-fit plus three-engine batch')
    paths = [batch/'evaluation/summary_all']
    paths += [batch/'training'/c['label'] for c in schedule['controllers']]
    paths += [batch/'evaluation/runs'/r['id']/c['label']
              for r in schedule['rows'] for c in schedule['controllers']
              if not (r['slip'] is None and c['domain'] == 'varied')]
    if len(paths) != 1012:
        raise ValueError('Unexpected completion coverage')
    receipts = {}
    for directory in paths:
        receipt = directory/'completed.json'
        if not verified(directory, read(receipt)['identity']):
            raise ValueError(f'Incomplete artifact: {directory}')
        receipts[str(receipt)] = digest(receipt)
    print('[Audit] 26 fits, three random networks, 982 deployments and report verified', flush=True)
    return receipts


def plan(batch):
    receipts = audit(batch)
    spec, data, weights = prepare(ROOT/'experiments/consolidated/v1/evaluation/all_all_correct', 7)
    report = read(batch/'evaluation/summary_all/comparison.json')
    lookup = {r['controller']['label']: r['controller'] for r in report['runs']}
    current = read(batch/'evaluation/schedule_all.json')['controllers']
    models = {c['label']: c for c in current}
    pairs = []
    for c in current:
        if 'reference_label' not in c:
            continue
        ref = lookup[c['reference_label']]
        models[ref['label']] = ref
        job = read(batch/'training'/c['label']/'job.json')
        if digest(job['model']) != ref['sha256']:
            raise ValueError('Initializer does not match its evaluated reference')
        pairs.append(dict(policy=c['label'], reference=ref['label']))
    # Preserve the original 100-epoch BC point as well as its matched-update successor.
    models['original_hidden_42_bc'] = lookup['original_hidden_42_bc']
    for c in models.values():
        if digest(c['model']) != c['sha256']:
            raise ValueError(f'Checkpoint changed: {c["label"]}')
    spec.update(protocol='convergence_shared_oracle_probes_v1', models=models,
                pairs=pairs, batch=str(batch), audited_receipts=receipts,
                comparison_sha256=digest(batch/'evaluation/summary_all/comparison.json'))
    spec['sources'][str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    return spec, data, weights


def paired_geometry(spec, data):
    rows = []
    groups = data['test'][2]
    for pair in spec['pairs']:
        layers = []
        for label in (pair['policy'], pair['reference']):
            c = spec['models'][label]
            model = load_policy(c['model'], c['kind'], c['architecture'])
            layers.append(features(model.net, inputs(data, spec, c['architecture'])['test']))
        for i in (0, 1):
            a, b = layers[0][i], layers[1][i]
            per_run = {str(int(g)): cka(a[groups == g], b[groups == g]) for g in np.unique(groups)}
            rows.append(dict(**pair, layer=i+1, pooled_cka=cka(a, b), per_run_cka=per_run,
                             mean_per_run_cka=float(np.mean(list(per_run.values())))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'run', 'report'])
    parser.add_argument('--batch', type=Path, default=BATCH)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec, data, weights = plan(args.batch.resolve())
    n = 2*len(spec['models']) + 19
    print(f'[Plan] {len(spec["models"])} checkpoints; {n} representations; {n*4} multitarget fits', flush=True)
    if args.stage == 'audit':
        return
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'experiments/representation_probing'):
        raise ValueError('Keep raw fits inside experiments/representation_probing')
    if args.stage == 'report' and not (output/'plan.json').exists():
        raise ValueError('Missing probe plan')
    immutable(output/'plan.json', spec)
    execute(output, spec, data, weights, report_only=args.stage == 'report')
    save(output/'initializer_geometry.json', paired_geometry(spec, data))
    print(f'[Complete] {output}', flush=True)


if __name__ == '__main__':
    main()

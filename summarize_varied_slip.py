"""Offline slip reports and reproducible, human-chosen sampled-slip plans."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from configure_field_slip import digest
from run_varied_slip import OUT, POLICIES, SEEDS, PROTOCOL, FIXED, save


def load_condition(root, condition, slips):
    from summarize_evaluation import jerk_cost
    rows = []
    for seed in SEEDS:
        for _, label, _ in POLICIES:
            directory = root / 'conditions' / condition / 'runs' / f'{label}_seed_{seed}'
            path = directory / 'raw' / f'{label}_seed_{seed}.json'
            receipt = json.loads((directory / 'completed.json').read_text())
            if receipt['identity']['protocol'] != PROTOCOL:
                raise ValueError(f'Incompatible completion receipt: {directory}')
            for name, expected in receipt['files'].items():
                if digest(directory / name) != expected:
                    raise ValueError(f'Completed artifact changed: {directory / name}')
            item = json.loads(path.read_text())
            if (item['active_steps'] != 3000 or item['seed'] != seed
                    or item['policy_label'] != label or item['environment_id'] != condition
                    or item['field_contact_slip'] != slips[seed]
                    or item.get('protocol') != PROTOCOL):
                raise ValueError(f'Incompatible run: {path}')
            manifest = json.loads((directory / 'environment_manifest.json').read_text())
            if (manifest['field_contact_slip'] != slips[seed]
                    or manifest.get('scene_setter_method') != 'single_argument_v1'
                    or item['scene_sha256'] != digest(directory / 'soccer.rsg')
                    or item['scene_sha256'] != manifest['sha256']):
                raise ValueError(f'Configuration mismatch: {path}')
            with np.load(item['trace'], allow_pickle=False) as trace:
                if len(trace['obs']) != 3000:
                    raise ValueError(f'Incomplete trace: {path}')
                for field in trace.files:
                    if not np.isfinite(trace[field]).all():
                        raise ValueError(f'Non-finite trace: {path}, {field}')
                item['torso_roll_rms_deg'] = float(np.sqrt(np.mean(trace['torso_roll_deg'] ** 2)))
                item['torso_pitch_rms_deg'] = float(np.sqrt(np.mean(trace['torso_pitch_deg'] ** 2)))
                item['minimum_head_height_m'] = float(trace['head_height_m'].min())
                item['jerk_cost_rate_rad2_per_s6'] = jerk_cost(trace['joint_positions_deg'], trace['episode_id'])[1]
            item['source_summary'] = str(path)
            item['source_sha256'] = digest(path)
            rows.append(item)
    by_key = {(r['policy_label'], r['seed']): r for r in rows}
    for row in rows:
        seed = row['seed']
        random = by_key['random', seed]['return_target_progress_m']
        oracle = by_key['oracle', seed]['return_target_progress_m']
        row['normalized_return'] = ((row['return_target_progress_m'] - random) / (oracle - random)
                                    if oracle - random > 1e-8 else None)
    return rows


def validate_draws(root):
    plan = json.loads((root / 'sampled_draws.json').read_text())
    if plan['fixed_report_sha256'] != digest(root / 'fixed_report.json'):
        raise ValueError('Fixed report changed; review and regenerate plan')
    report = json.loads((root / 'fixed_report.json').read_text())
    for row in report['per_run']:
        if row['source_sha256'] != digest(Path(row['source_summary'])):
            raise ValueError('Fixed source results changed')
    low, high = plan['low'], plan['high']
    if not (.025 <= low < high <= 10.) or not plan['rationale'].strip():
        raise ValueError('Sample bounds must be within configured [.025, 10]')
    values = np.random.default_rng(plan['draw_seed']).uniform(low, high, len(SEEDS))
    if [r['seed'] for r in plan['draws']] != SEEDS:
        raise ValueError('Expected three paired evaluation seeds')
    if not np.array_equal(values, [r['slip'] for r in plan['draws']]):
        raise ValueError('Draws do not match the declared uniform distribution')
    return plan


def report(root, sampled=False):
    conditions = {name: {seed: slip for seed in SEEDS} for name, slip in FIXED.items()}
    if sampled:
        conditions['sampled'] = {r['seed']: r['slip'] for r in validate_draws(root)['draws']}
    per_condition = {name: load_condition(root, name, slips) for name, slips in conditions.items()}
    nominal = {(r['policy_label'], r['seed']): r['return_target_progress_m'] for r in per_condition['nominal']}
    rows, aggregates = [], []
    metrics = ['return_target_progress_m', 'normalized_return', 'retention', 'targets_reached', 'falls',
               'torso_roll_rms_deg', 'torso_pitch_rms_deg', 'minimum_head_height_m', 'jerk_cost_rate_rad2_per_s6']
    for condition, items in per_condition.items():
        for item in items:
            denominator = nominal[item['policy_label'], item['seed']]
            item['retention'] = item['return_target_progress_m'] / denominator if abs(denominator) > 1e-8 else None
        rows.extend(items)
        for _, label, _ in POLICIES:
            group = [r for r in items if r['policy_label'] == label]
            aggregate = {'condition': condition, 'policy': label, 'n': len(group)}
            for metric in metrics:
                values = [r[metric] for r in group if r[metric] is not None]
                aggregate[metric] = float(np.mean(values)) if values else None
            mean_nominal = np.mean([nominal[label, seed] for seed in SEEDS])
            aggregate['retention_ratio_of_means'] = (aggregate['return_target_progress_m'] / mean_nominal
                                                    if abs(mean_nominal) > 1e-8 else None)
            aggregates.append(aggregate)
    stem = 'sampled_report' if sampled else 'fixed_report'
    save(root / f'{stem}.json', {'per_run': rows, 'aggregates': aggregates})
    with (root / f'{stem}.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    lines = ['# Frozen-policy slip evaluation', '',
             'Three paired seeds; 3,000 active steps each. J = target progress in metres.',
             'Normalized return uses same-condition, same-seed oracle and random baselines.',
             'Retention below is ratio of mean J to nominal mean J. Per-seed ratios are also saved.',
             'Near-zero or negative nominal returns make retention misleading (especially random/GAIL-random).',
             'Original startup/fall accounting is preserved; recovery is excluded from active steps.', '',
             '| Condition | Policy | Mean J (m) | Normalized | Retention | Mean falls | Mean targets |',
             '|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'undefined' if x is None else f'{x:.3f}'
    for r in aggregates:
        lines.append('| ' + ' | '.join([r['condition'], r['policy'], *[fmt(r[k]) for k in
                     ['return_target_progress_m', 'normalized_return', 'retention_ratio_of_means', 'falls', 'targets_reached']]]) + ' |')
    lines += ['', 'Stability metrics (roll/pitch RMS, minimum head height, smoothed joint jerk) are in JSON/CSV.',
              'Low jerk and zero falls alone do not establish successful locomotion.',
              'Reports are descriptive: three seeds do not establish statistical significance.',
              'Fixed report review: choose bounds only where the oracle and viable policies retain useful progress;',
              'inspect falls and torso stability, then record the reasoning when generating the sampled plan.']
    (root / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    print(root / f'{stem}.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fixed', 'sampled', 'plan'])
    parser.add_argument('--low', type=float)
    parser.add_argument('--high', type=float)
    parser.add_argument('--draw-seed', type=int, default=260925)
    parser.add_argument('--rationale')
    args = parser.parse_args()
    if args.action != 'plan':
        report(OUT, args.action == 'sampled')
        return
    if (args.low is None or args.high is None or not (.025 <= args.low < args.high <= 10.)
            or not args.rationale or not args.rationale.strip()):
        parser.error('plan requires .025 <= --low < --high <= 10 and --rationale')
    if (OUT / 'sampled_draws.json').exists():
        parser.error('Plan already exists; archive the plan and sampled condition before changing it')
    report(OUT)  # Require all 120 completed fixed evaluations before choosing bounds.
    values = np.random.default_rng(args.draw_seed).uniform(args.low, args.high, len(SEEDS))
    save(OUT / 'sampled_draws.json', {'distribution': 'uniform', 'low': args.low, 'high': args.high,
         'mean': (args.low + args.high) / 2, 'variance': (args.high - args.low) ** 2 / 12,
         'draw_seed': args.draw_seed, 'rationale': args.rationale,
         'fixed_report_sha256': digest(OUT / 'fixed_report.json'),
         'draws': [{'seed': seed, 'slip': float(value)} for seed, value in zip(SEEDS, values)]})
    validate_draws(OUT)
    print('Plan saved. Review sampled_draws.json before running the sampled stage.')


if __name__ == '__main__':
    main()

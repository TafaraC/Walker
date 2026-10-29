"""Identical-policy repeats: measure variability without changing startup behaviour."""
import argparse
import itertools
from pathlib import Path

import numpy as np

from variable_training.common import ROOT, digest, read, save, simulator_lock, verified
from variable_training.server import require_free_ports
from .evaluate import DEFAULT_ROOT, make_spec, evaluate, job_identity, validate_result


def make_repeat_spec(root, repeats=5, training_seed=42, steps=3000):
    if repeats < 2 or steps <= 0:
        raise ValueError('At least two repeats and positive steps required')
    spec = make_spec(root, 'all', ['correct'], steps)
    spec['controllers'] = [c for c in spec['controllers']
                           if c['training_seed'] in (None, training_seed)]
    if len(spec['controllers']) != 3:
        raise ValueError('Training seed must identify a completed hidden/input pair')
    # Predeclared diagnostic cases; use exact existing slip/task pairs, not rounded approximations.
    wanted = {'nominal_0': 'nominal', 'confirmation_group_0_draw_0': 'low',
              'grouped_group_4_draw_2': 'high'}
    cases = [dict(row, case=wanted[row['id']]) for row in spec['rounds'] if row['id'] in wanted]
    spec.update(study='bc_repeatability_v1', suite='repeatability', repeats=repeats,
                selected_training_seed=training_seed,
                startup_protocol='unchanged context evaluator; diagnostic only')
    # Interleave repeated cases; keep task seed, model and physical/supplied slip identical.
    spec['rounds'] = [dict(row, id=f'{row["case"]}_repeat_{i:02d}',
                           suite='repeatability', group=row['case'], repeat=i)
                      for i in range(repeats) for row in cases]
    spec['sources']['context_training/repeatability.py'] = digest(Path(__file__))
    return spec


def summarize(output, spec):
    runs, groups = [], {}
    for row in spec['rounds']:
        for controller in spec['controllers']:
            directory = output/'runs'/row['id']/controller['label']
            identity = job_identity(spec, row, controller)
            if not verified(directory, identity):
                raise ValueError(f'Missing repeat: {directory}')
            result = validate_result(directory, identity)
            with np.load(directory/'trace.npz', allow_pickle=False) as trace:
                initial = {k: np.asarray(trace[k][0]).tolist() for k in
                    ('obs','actions','oracle_actions','joint_positions_deg','positions','targets',
                     'head_height_m','torso_roll_deg','torso_pitch_deg')}
            item = dict(round=row['id'], case=row['case'], physical_slip=row['slip'],
                        task_seed=row['seed'], controller=controller['label'],
                        repeat=row['repeat'], supplied_slip=result['supplied_slip'],
                        progress_m=result['return_target_progress_m'], falls=result['falls'],
                        targets=result['targets_reached'], initial=initial)
            runs.append(item)
            groups.setdefault((row['case'],controller['label']),[]).append(item)
    summary=[]
    for (case,label), items in groups.items():
        progress=np.array([r['progress_m'] for r in items])
        pairs=[]
        for a,b in itertools.combinations(items,2):
            pairs.append(dict(repeats=[a['repeat'],b['repeat']],
                abs_progress_delta_m=abs(a['progress_m']-b['progress_m']),
                initial_obs_maxabs=float(np.max(np.abs(np.array(a['initial']['obs'])-b['initial']['obs']))),
                initial_position_distance_m=float(np.linalg.norm(np.array(a['initial']['positions'])-b['initial']['positions'])),
                initial_joint_maxabs_deg=float(np.max(np.abs(np.array(a['initial']['joint_positions_deg'])-b['initial']['joint_positions_deg']))),
                initial_target_distance_m=float(np.linalg.norm(np.array(a['initial']['targets'])-b['initial']['targets']))))
        summary.append(dict(case=case,controller=label,n=len(items),mean_progress_m=float(progress.mean()),
            std_progress_m=float(progress.std(ddof=1)),min_progress_m=float(progress.min()),
            max_progress_m=float(progress.max()),total_falls=sum(r['falls'] for r in items),
            mean_pairwise_abs_progress_delta_m=float(np.mean([p['abs_progress_delta_m'] for p in pairs])),
            max_initial_obs_difference=max(p['initial_obs_maxabs'] for p in pairs),
            max_initial_position_distance_m=max(p['initial_position_distance_m'] for p in pairs),
            pairs=pairs))
    save(output/'repeatability.json',dict(study=spec['study'],runs=runs,summary=summary,
        interpretation='Pairwise differences are dependent diagnostics, not independent samples or confidence intervals.'))
    lines=['# BC repeatability diagnostics','',
        'Each case/controller repeats the same model, task seed and physical/supplied slip.',
        'Startup behaviour is unchanged. This quantifies variability; it does not identify or fix its cause.','',
        '| Case | Controller | N | Mean progress m | SD m | Mean pairwise absolute difference m | Falls | Max initial observation difference |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f'| {s["case"]} | {s["controller"]} | {s["n"]} | {s["mean_progress_m"]:.4f} | '
            f'{s["std_progress_m"]:.4f} | {s["mean_pairwise_abs_progress_delta_m"]:.4f} | '
            f'{s["total_falls"]} | {s["max_initial_obs_difference"]:.6f} |')
    lines += ['', 'Use initial observation, position, joint and target differences to distinguish unequal starts',
        'from later divergence. Observation coordinates mix units; max absolute difference is diagnostic only.',
        'Do not delete outliers or treat pairwise differences as independent replicates.',
        'Compare the spread with the previous ~0.01–0.37 m architecture differences before interpreting them.',
        'Any startup fix requires a separate protocol and new matched comparisons, not mixed old/new results.']
    (output/'repeatability.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output/"repeatability.md"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','evaluate','report'))
    parser.add_argument('--training-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--training-seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--retries',type=int,default=1)
    args=parser.parse_args()
    if args.repeats < 2 or args.steps <= 0 or args.retries < 0:
        parser.error('Require repeats>=2, steps>0 and retries>=0')
    root=args.training_root.resolve()
    output=(args.output or root/'repeatability_v1').resolve()
    spec=make_repeat_spec(root,args.repeats,args.training_seed,args.steps)
    print(f'[Plan] {len(spec["rounds"])*len(spec["controllers"])} deployments; no new training')
    with simulator_lock():
        path=output/'plan.json'
        if path.exists() and read(path)!=spec:
            raise ValueError('Repeatability inputs changed; choose a new --output')
        if args.stage=='report' and not path.exists():
            raise ValueError('No saved plan')
        save(path,spec)
        if args.stage=='evaluate':
            from check_simulator_config import main as preflight
            if preflight():
                raise RuntimeError('Simulator preflight failed')
            require_free_ports()
            evaluate(output,spec,args.retries)
            summarize(output,spec)
        elif args.stage=='report':
            summarize(output,spec)
        else:
            print(f'[Plan] {path}; no simulator started')


if __name__=='__main__':
    main()

"""Paired owned-server evaluation for completed context BC checkpoints."""
import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys

import numpy as np

from variable_training.common import ROOT, begin, commit, digest, read, save, verified, simulator_lock, source_hashes
from variable_training.plan import truncated_draws
from variable_training.server import SYSTEM, SCENE, simulator, launch, require_free_ports
from .models import load_checkpoint

DEFAULT_ROOT = ROOT/'experiments/context_conditioning/bc_input_v1'
PROTOCOL = 'context_bc_evaluation_v1'


def rounds(seed, suite):
    if suite not in ('smoke', 'grouped', 'confirmation', 'nominal', 'all'):
        raise ValueError('Unknown suite')
    rows = []
    for kind in ('grouped', 'confirmation'):
        if suite not in (kind, 'all', 'smoke') or (suite == 'smoke' and kind != 'grouped'):
            continue
        for group, mean in enumerate([.025, .1, .2, 1., 10.]):
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, group, 20261009 if kind=='grouped' else 20261010]))
            for repeat in range(3):
                slip = 0.
                while slip <= 0:
                    slip = truncated_draws(rng, 1, 0., 10., mean, .5*mean)[0]
                rows.append(dict(id=f'{kind}_group_{group}_draw_{repeat}', suite=kind,
                    group=f'group_{group}', parent_mean=mean, parent_std=.5*mean, slip=slip,
                    seed=seed*100000+(92000 if kind=='grouped' else 94000)+group*3+repeat))
        if suite=='smoke':
            return rows[:1]
    if suite in ('nominal', 'all'):
        rows += [dict(id=f'nominal_{i}', suite='nominal', group='nominal',
                      slip=.1, seed=seed*100000+96000+i) for i in range(3)]
    return rows


def controllers(training_root, modes):
    plan_path = training_root/'plan.json'
    plan = read(plan_path)
    if set(plan['architectures']) != {'hidden', 'input'}:
        raise ValueError('Evaluation requires both matched hidden and input training arms')
    result = [dict(label='oracle', architecture='oracle', training_seed=None, mode='none', model=None)]
    for seed in plan['seeds']:
        for architecture in ('hidden', 'input'):
            directory = training_root/architecture/f'seed_{seed}'
            if not verified(directory, dict(plan_sha256=digest(plan_path), architecture=architecture, seed=seed)):
                raise ValueError(f'Training incomplete: {directory}')
            _, loaded = load_checkpoint(directory/'policy.pt')
            if loaded != architecture:
                raise ValueError('Checkpoint architecture differs from training plan')
            for mode in (['none'] if architecture=='hidden' else modes):
                result.append(dict(label=f'{architecture}_seed_{seed}_{mode}', architecture=architecture,
                    training_seed=seed, mode=mode, model=str(directory/'policy.pt'),
                    model_sha256=digest(directory/'policy.pt')))
    return result


def supplied_context(controller, row):
    if controller['mode']=='correct':
        return float(f'{row["slip"]:.12g}')  # Actual scalar scene encoding.
    if controller['mode']=='constant':
        return .1
    if controller['mode']=='wrong':
        return 10. if row['slip'] < 1. else .025
    return None


def make_spec(training_root, suite, modes, steps):
    plan = read(training_root/'plan.json')
    source_plan = read(plan['dataset']['plan'])
    if digest(plan['dataset']['plan']) != plan['dataset']['plan_sha256']:
        raise ValueError('Original data plan changed')
    sources = source_hashes()
    for name in ('__init__.py', 'models.py', 'evaluate.py', 'eval_worker.py'):
        path = Path(__file__).parent/name
        sources[str(path.relative_to(ROOT))] = digest(path)
    return dict(protocol=PROTOCOL, training_plan_sha256=digest(training_root/'plan.json'),
        suite=suite, modes=modes, steps=steps, rounds=rounds(source_plan['seed'], suite),
        controllers=controllers(training_root, modes), sources=sources,
        environment={str(p): digest(p) for p in
            [SCENE, SYSTEM/'rcssserver3d.rb', SYSTEM/'naosoccersim.rb', Path.home()/'.simspark/spark.rb']},
        metrics=dict(smoothing_window=5, wasserstein_projections=256, wasserstein_seed=12345),
        ablations=dict(constant=.1, wrong='10 if physical slip < 1 else 0.025'))


def job_identity(spec, row, controller):
    return dict(protocol=PROTOCOL, sources=spec['sources'], environment=spec['environment'],
        training_plan_sha256=spec['training_plan_sha256'], row=row, controller=controller,
        steps=spec['steps'], supplied_slip=supplied_context(controller, row))


def validate_result(directory, identity):
    from configure_field_slip import status
    manifest = read(directory/'environment_manifest.json')
    scene = status(directory/'soccer.rsg', identity['row']['slip'])
    if (not manifest.get('server_scene_import_verified') or not scene['matches_expected'] or
            scene['scene_setter_method'] != 'single_argument_v1' or
            manifest['sha256'] != scene['sha256']):
        raise ValueError('Unverified physical slip')
    result = read(directory/'result.json')
    for key in ('row', 'controller', 'supplied_slip'):
        if result.get(key) != identity[key]:
            raise ValueError(f'Evaluation result metadata mismatch: {key}')
    if result['active_steps'] != identity['steps'] or result['active_seconds'] != identity['steps']*.02:
        raise ValueError('Incomplete active horizon')
    with np.load(directory/'trace.npz', allow_pickle=False) as trace:
        for key, width in [('obs',63), ('actions',16), ('oracle_actions',16), ('joint_positions_deg',22)]:
            if trace[key].shape != (identity['steps'], width):
                raise ValueError(f'Invalid trace shape: {key}')
        for key in ('episode_id','target_progress','forward_velocity_mps','target_direction_error'):
            if trace[key].shape != (identity['steps'],):
                raise ValueError(f'Invalid trace shape: {key}')
        if any(len(trace[key]) != identity['steps'] or not np.isfinite(trace[key]).all() for key in trace.files):
            raise ValueError('Non-finite or incomplete trace')
        if not np.isclose(trace['target_progress'].sum(), result['return_target_progress_m']):
            raise ValueError('Progress summary differs from trace')
    return result


def evaluate(output, spec, retries):
    for row in spec['rounds']:
        for controller in spec['controllers']:
            directory = output/'runs'/row['id']/controller['label']
            identity = job_identity(spec, row, controller)
            if verified(directory, identity):
                validate_result(directory, identity)
                print(f'[Resume] {row["id"]} {controller["label"]}', flush=True)
                continue
            for attempt in range(retries+1):
                begin(directory)
                save(directory/'job.json', identity)
                print(f'[BC evaluation] {row["id"]} slip={row["slip"]:.8g} '
                      f'{controller["label"]} supplied={identity["supplied_slip"]}', flush=True)
                try:
                    with simulator(directory, row['slip']) as manifest:
                        metadata = {k:v for k,v in manifest.items() if k != 'server_process'}
                        metadata.update(evaluation_protocol=PROTOCOL,
                            evaluator_sha256=digest(Path(__file__).parent/'eval_worker.py'),
                            supplied_slip=identity['supplied_slip'])
                        save(directory/'environment_manifest.json',metadata)
                        launch([sys.executable, '-u', '-m', 'context_training.eval_worker',
                                str(directory/'job.json')], directory/'evaluation.log',
                               server=manifest['server_process'])
                    validate_result(directory, identity)
                    if controller['model'] and digest(controller['model']) != controller['model_sha256']:
                        raise ValueError('Model changed during evaluation')
                    if any(digest(ROOT/name) != value for name,value in spec['sources'].items()):
                        raise ValueError('Source changed during evaluation')
                    if any(digest(name) != value for name,value in spec['environment'].items()):
                        raise ValueError('Environment configuration changed during evaluation')
                    commit(directory, identity)
                    break
                except Exception as error:
                    save(directory/'failure.json', dict(error=repr(error), attempt=attempt+1))
                    if attempt == retries or not isinstance(error, (RuntimeError, subprocess.CalledProcessError, OSError)):
                        raise
                    print('[Retry] Failed run preserved; restarting its owned server.', flush=True)


def report(output, spec):
    from summarize_evaluation import jerk_cost, sliced_wasserstein
    results = []
    for row in spec['rounds']:
        oracle_dir = output/'runs'/row['id']/'oracle'
        # Verify all receipts before reading the corresponding oracle reference.
        for controller in spec['controllers']:
            directory = output/'runs'/row['id']/controller['label']
            identity = job_identity(spec, row, controller)
            if not verified(directory, identity):
                raise ValueError(f'Missing evaluation: {directory}')
            validate_result(directory, identity)
        oracle = read(oracle_dir/'result.json')['return_target_progress_m']
        with np.load(oracle_dir/'trace.npz', allow_pickle=False) as data:
            expert_joints = data['joint_positions_deg'].copy()
        for controller in spec['controllers']:
            directory = output/'runs'/row['id']/controller['label']
            result = read(directory/'result.json')
            with np.load(directory/'trace.npz', allow_pickle=False) as data:
                joints = data['joint_positions_deg']
                jerk, rate = jerk_cost(joints, data['episode_id'], spec['metrics']['smoothing_window'])
                distance = (0. if controller['architecture']=='oracle' else sliced_wasserstein(
                    joints, expert_joints, spec['metrics']['wasserstein_projections'],
                    spec['metrics']['wasserstein_seed']))
            result.update(oracle_return_ratio=result['return_target_progress_m']/oracle if oracle > 1e-8 else None,
                jerk_integrated_rad2_per_s5=jerk, jerk_cost_rate_rad2_per_s6=rate,
                joint_occupancy_sliced_wasserstein=distance)
            results.append(result)
    aggregates = []
    for suite in dict.fromkeys(r['row']['suite'] for r in results):
        for controller in spec['controllers']:
            selected = [r for r in results if r['row']['suite']==suite and r['controller']==controller]
            ratios = [r['oracle_return_ratio'] for r in selected if r['oracle_return_ratio'] is not None]
            aggregates.append(dict(suite=suite, policy=controller['label'], rounds=len(selected),
                mean_progress_m=float(np.mean([r['return_target_progress_m'] for r in selected])),
                mean_oracle_ratio=float(np.mean(ratios)) if ratios else None,
                falls=sum(r['falls'] for r in selected),
                mean_targets=float(np.mean([r['targets_reached'] for r in selected]))))
    # Paired deltas remain separated by fine-tuning seed and evaluation round.
    paired = []
    lookup = {(r['row']['id'],r['controller']['label']):r for r in results}
    for r in results:
        c = r['controller']
        if c['architecture'] != 'input':
            continue
        control_label = (f'hidden_seed_{c["training_seed"]}_none' if c['mode']=='correct'
                         else f'input_seed_{c["training_seed"]}_correct')
        control = lookup[(r['row']['id'],control_label)]
        paired.append(dict(round=r['row']['id'],training_seed=c['training_seed'],
            policy=c['label'],reference=control_label,
            progress_delta_m=r['return_target_progress_m']-control['return_target_progress_m'],
            falls_delta=r['falls']-control['falls']))
    save(output/'report.json', dict(protocol=PROTOCOL, metrics=spec['metrics'], runs=results,
                                   aggregates=aggregates, paired_differences=paired))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(aggregates[0]))
    writer.writeheader(); writer.writerows(aggregates)
    (output/'report.csv').write_text(buffer.getvalue())
    lines = ['# Context BC evaluation', '',
        'Oracle ratios are not random-normalized returns. Fine-tuning seeds share a pretrained initializer.',
        'Grouped rounds were previously inspected; confirmation rounds are a separate predeclared schedule.',
        'Each run holds physical slip fixed. Wrong/constant modes change only supplied context.', '',
        '| Suite | Policy | Rounds | Mean progress m | Mean oracle ratio | Falls | Mean targets |',
        '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        ratio = 'N/A' if a['mean_oracle_ratio'] is None else f'{a["mean_oracle_ratio"]:.4f}'
        lines.append(f'| {a["suite"]} | {a["policy"]} | {a["rounds"]} | {a["mean_progress_m"]:.4f} | '
                     f'{ratio} | {a["falls"]} | {a["mean_targets"]:.2f} |')
    lines += ['', 'Raw per-run metrics and paired differences are in report.json.',
        'Joint occupancy uses standardized sliced Wasserstein-1 (not exact 22-D Wasserstein).',
        'Jerk is smoothed angular-position jerk, not torque or energy; reset boundaries are excluded.',
        'On-policy latent MSE is not held-out expert-state MSE. See training metrics for the latter.',
        'Do not pool rounds as independent training replicas or interpret no falls as proof of stability.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan','evaluate','report'))
    parser.add_argument('--training-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--suite', choices=('smoke','grouped','confirmation','nominal','all'), default='grouped')
    parser.add_argument('--modes', nargs='+', choices=('correct','constant','wrong'), default=['correct'])
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--retries', type=int, default=1)
    args = parser.parse_args()
    if args.steps <= 0 or args.retries < 0 or len(set(args.modes)) != len(args.modes) or 'correct' not in args.modes:
        parser.error('Positive steps, nonnegative retries, unique modes including correct required')
    root = args.training_root.resolve()
    output = (args.output or root/f'evaluation_{args.suite}_v1').resolve()
    spec = make_spec(root,args.suite,args.modes,args.steps)
    print(f'[Plan] {len(spec["rounds"])} rounds x {len(spec["controllers"])} controllers = '
          f'{len(spec["rounds"])*len(spec["controllers"])} runs; {args.steps} active steps each')
    # Shared lock serializes live runs AND report/plan writes across owned-server pipelines.
    with simulator_lock():
        path = output/'plan.json'
        if path.exists() and read(path) != spec:
            raise ValueError('Evaluation inputs changed; choose a new --output')
        if args.stage=='report' and not path.exists():
            raise ValueError('No saved evaluation plan')
        save(path,spec)
        if args.stage=='evaluate':
            from check_simulator_config import main as preflight
            if preflight():
                raise RuntimeError('Simulator preflight failed')
            require_free_ports()
            evaluate(output,spec,args.retries)
            report(output,spec)
        elif args.stage=='report':
            report(output,spec)
        else:
            print(f'[Plan] Saved {path}; no simulator started')


if __name__=='__main__':
    main()

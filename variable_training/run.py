"""Plan, collect, train and evaluate four continuation policies with owned servers."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

from .common import ROOT, MODELS, PROTOCOL, begin, commit, digest, read, save, simulator_lock, source_hashes, verified
from .common import evaluation_identity, verified_evaluation
from .plan import make_plan, validate_plan, truncated_draws
from .server import SCENE, SYSTEM, simulator, launch, require_free_ports


def validate_shard(path, count, gail=False):
    with np.load(path, allow_pickle=False) as data:
        shapes = {'obs': (count, 63), 'actions': (count, 16),
                  'executed_actions': (count, 16), 'episode_id': (count,),
                  'dones': (count,), 'logprobs': (count,), 'values': (count,), 'bootstrap_value': ()}
        for key, shape in shapes.items():
            if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f'Invalid {key}: {path}')
        if gail and not np.array_equal(data['actions'], data['executed_actions']):
            raise ValueError('GAIL action must be the executed action')


class Experiment:
    def __init__(self, plan_path):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read(self.plan_path)
        validate_plan(self.plan)
        # Evaluation extension must not change the training plan/checkpoint hash.
        self.evaluation_rounds = 1 if self.plan['quick'] else len(self.plan['training_environment']['slips'])
        self.root = self.plan_path.parent
        self.identity = dict(protocol=PROTOCOL, plan=digest(self.plan_path), sources=source_hashes(),
                             initializers={k: digest(ROOT/v) for k,v in MODELS.items()},
                             environment={str(p): digest(p) for p in
                                          [SCENE, SYSTEM/'rcssserver3d.rb', SYSTEM/'naosoccersim.rb',
                                           Path.home()/'.simspark/spark.rb']})
        self.arms = []

    def job(self, directory, spec, inputs=(), live=False):
        directory = Path(directory)
        identity = dict(self.identity, spec=spec, inputs={str(p): digest(p) for p in inputs})
        if verified(directory, identity):
            if spec['task'] == 'collect':
                validate_shard(directory/'data.npz', spec['row']['steps'], spec['kind']=='gail')
            print(f'[Resume] {directory.relative_to(self.root)}', flush=True)
            return directory
        begin(directory)
        save(directory/'job.json', spec)
        try:
            command = [sys.executable, '-u', '-m', 'variable_training.worker', str(directory/'job.json')]
            if live:
                with simulator(directory, spec['row']['slip']) as manifest:
                    launch(command, directory/'worker.log', server=manifest['server_process'])
                validate_shard(directory/'data.npz', spec['row']['steps'], spec['kind']=='gail')
            else:
                launch(command, directory/'worker.log')
                # A successful update must yield finite policy tensors and metrics.
                import torch
                state = torch.load(directory/'policy.pth', map_location='cpu', weights_only=True)
                if not state or not all(torch.isfinite(tensor).all() for tensor in state.values()):
                    raise ValueError('Invalid model checkpoint')
                read(directory/'metrics.json')
            if identity['inputs'] != {str(p): digest(p) for p in inputs}:
                raise ValueError('Training inputs changed while job was running')
            commit(directory, identity)
        except BaseException as error:
            save(directory/'failure.json', {'error': repr(error)})
            raise
        return directory

    def rows(self, group, arm):
        return [dict(row, slip=.1 if arm=='nominal' else row['slip'])
                for row in self.plan['groups'][group]]

    def demonstrations(self, arm, group):
        paths = []
        for row in self.rows(group, arm):
            directory = self.root/'data'/arm/group/row['id']
            self.job(directory, dict(task='collect', kind='oracle', row=row), live=True)
            paths.append(str(directory/'data.npz'))
        return paths

    def train_supervised(self, label, arm):
        settings = self.plan['settings']
        training = self.demonstrations(arm, 'train')
        validation = self.demonstrations(arm, 'validation')
        self.demonstrations(arm, 'test')
        model, optimizer = str(ROOT/MODELS[label]), None
        iterations = 0 if label=='bc' else sum(k.startswith('dagger_') for k in self.plan['groups'])
        for iteration in range(iterations+1):
            base = self.root/'policies'/label/arm/f'iteration_{iteration:03d}'
            if iteration:
                for row in self.rows(f'dagger_{iteration}', arm):
                    rollout = base/'segments'/row['id']
                    spec = dict(task='collect', kind='dagger', row=row, model=model,
                                beta=max(0., 1.-.2*iteration))
                    self.job(rollout, spec, [model], live=True)
                    training.append(str(rollout/'data.npz'))
            spec = dict(task='supervised', train=training.copy(), validation=validation,
                        model=model, optimizer=optimizer, seed=self.plan['seed']+iteration,
                        epochs=settings['bc_epochs' if label=='bc' else 'dagger_epochs'],
                        learning_rate=settings['supervised_lr'], batch_size=settings['batch_size'])
            update = self.job(base/'update', spec,
                              training+validation+[model]+([optimizer] if optimizer else []))
            model, optimizer = str(update/'policy.pth'), str(update/'optimizer.pth')
        save(self.root/'policies'/label/arm/'final.json',
             dict(model=model, sha256=digest(model), protocol=PROTOCOL,
                  initializer=MODELS[label], plan=digest(self.plan_path)))

    def train_gail(self, label, arm):
        self.demonstrations(arm, 'validation')
        self.demonstrations(arm, 'test')
        groups = [k for k in self.plan['groups'] if k.startswith('gail_')]
        # Frozen demonstration pool on the exact preplanned learner surfaces.
        expert = {group: self.demonstrations(arm, group) for group in groups}
        normalizer = [path for group in groups for path in expert[group]]
        model, optimizer, discriminator = str(ROOT/MODELS[label]), None, None
        for iteration, group in enumerate(groups, 1):
            base = self.root/'policies'/label/arm/f'iteration_{iteration:03d}'
            rollouts = []
            for row in self.rows(group, arm):
                directory = base/'segments'/row['id']
                self.job(directory, dict(task='collect', kind='gail', row=row, model=model), [model], live=True)
                rollouts.append(str(directory/'data.npz'))
            spec = dict(task='gail', model=model, optimizer=optimizer, discriminator=discriminator,
                        expert=expert[group], normalizer=normalizer, rollouts=rollouts,
                        seed=self.plan['seed']+iteration, iteration=iteration, label=label)
            update = self.job(base/'update', spec, normalizer+rollouts+[model]+
                              ([optimizer, discriminator] if optimizer else []))
            model, optimizer, discriminator = (str(update/name) for name in
                                                ('policy.pth', 'optimizer.pth', 'discriminator.pth'))
        save(self.root/'policies'/label/arm/'final.json',
             dict(model=model, sha256=digest(model), protocol=PROTOCOL,
                  initializer=MODELS[label], plan=digest(self.plan_path)))

    def evaluation_rows(self, suite='gaussian'):
        if suite == 'grouped':
            rows=[]
            for group, mean in enumerate(self.plan['training_environment']['slips']):
                rng=np.random.default_rng(np.random.SeedSequence([self.plan['seed'],group,20261009]))
                for repeat in range(1 if self.plan['quick'] else 3):
                    value=0.
                    while value <= 0:
                        value=truncated_draws(rng,1,0.,10.,mean,.5*mean)[0]
                    rows.append(dict(id=f'group_{group}_draw_{repeat}',group=f'group_{group}',
                                     parent_mean=mean,parent_std=.5*mean,
                                     low=0.,low_exclusive=True,high=10.,slip=value,
                                     seed=self.plan['seed']*100000+92000+group*3+repeat))
            return rows
        if suite not in ('gaussian', 'fixed', 'all'):
            raise ValueError(f'Unknown evaluation suite: {suite}')
        rows = list(self.plan['evaluation'])
        planned = [r for r in rows if r['id'].startswith('sampled_')]
        count = getattr(self, 'evaluation_rounds', len(planned))
        if count != len(planned):
            d = self.plan['distribution']
            rng = np.random.default_rng(self.plan['seed'])
            extended = [dict(id=f'sampled_{i}',
                             slip=truncated_draws(rng,1,d['low'],d['high'],d['mean'],d['std'])[0],
                             seed=self.plan['seed']*100000+90000+i) for i in range(count)]
            if extended[:len(planned)] != planned:
                raise ValueError('Evaluation extension would alter original paired draws')
            rows = extended + [r for r in rows if not r['id'].startswith('sampled_')]
        return [row for row in rows if suite == 'all' or
                row['id'].startswith('sampled_') == (suite == 'gaussian')]

    def evaluation_directory(self, suite):
        return self.root/('evaluation_grouped_v1' if suite=='grouped' else 'evaluation')

    def evaluate(self, arms, frozen_only=False, suite='gaussian'):
        distribution = (dict(kind='grouped_truncated_gaussian',means=self.plan['training_environment']['slips'],
                             std_fraction=.5,low=0.,low_exclusive=True,high=10.,
                             draws_per_group=1 if self.plan['quick'] else 3)
                        if suite=='grouped' else self.plan['distribution'])
        schedule = dict(training_plan_sha256=digest(self.plan_path),distribution=distribution,
                        rounds=self.evaluation_rows('grouped' if suite=='grouped' else 'gaussian'))
        schedule_path=(self.evaluation_directory(suite)/'schedule.json' if suite=='grouped' else
                       self.root/f'evaluation_schedule_{len(schedule["rounds"])}_rounds.json')
        if schedule_path.exists() and read(schedule_path)!=schedule:
            raise ValueError('Saved evaluation schedule differs from requested paired draws')
        save(schedule_path,schedule)
        policies = [('oracle', 'oracle', None)]
        for label, checkpoint in MODELS.items():
            kind = 'gail' if label.startswith('gail') else 'bc'
            if frozen_only:
                policies.append((kind, label+'_frozen', str(ROOT/checkpoint)))
            if not frozen_only:
                for arm in arms:
                    info = read(self.root/'policies'/label/arm/'final.json')
                    if info['sha256'] != digest(info['model']) or info['plan'] != digest(self.plan_path):
                        raise ValueError('Final model provenance mismatch')
                    policies.append((kind, label+'_'+arm, info['model']))
        for row in self.evaluation_rows(suite):
            print(f'[Evaluation round] {row["id"]} | fixed slip={row["slip"]:.8g} | '
                  f'seed={row["seed"]} | {len(policies)} controllers', flush=True)
            for kind, label, model in policies:
                directory = self.evaluation_directory(suite)/row['id']/label
                identity = dict(evaluation_identity(self.identity), row=row, label=label,
                                model=digest(model) if model else None,
                                expert_data=digest(ROOT/'expert_train_data.npz'))
                if verified_evaluation(directory, identity):
                    continue
                begin(directory)
                command = [sys.executable, '-u', str(ROOT/'evaluate_policy.py'), '--policy', kind,
                           '--label', label, '--expert-data', str(ROOT/'expert_train_data.npz'),
                           '--steps', str(self.plan['settings']['eval_steps']), '--seed', str(row['seed']),
                           '--environment-id', row['id'], '--field-slip', str(row['slip']),
                           '--output-dir', str(directory/'raw')]
                if model:
                    command += ['--model', model]
                with simulator(directory, row['slip']) as manifest:
                    launch(command, directory/'evaluation.log', server=manifest['server_process'])
                result = read(directory/'raw'/f'{label}_seed_{row["seed"]}.json')
                if (result['active_steps'] != self.plan['settings']['eval_steps'] or
                        result['field_contact_slip'] != row['slip'] or result['seed'] != row['seed']):
                    raise ValueError('Incomplete evaluation summary')
                with np.load(result['trace'], allow_pickle=False) as trace:
                    if any(len(trace[k]) != result['active_steps'] or not np.isfinite(trace[k]).all()
                           for k in trace.files):
                        raise ValueError('Incomplete evaluation trace')
                commit(directory, identity)

    def report(self, arms, frozen_only=False, suite='gaussian'):
        rows = []
        labels = ['oracle'] + ([k+'_frozen' for k in MODELS] if frozen_only else [])
        if not frozen_only:
            labels += [k+'_'+arm for k in MODELS for arm in arms]
        for row in self.evaluation_rows(suite):
            base = self.evaluation_directory(suite)/row['id']
            oracle_path = base/'oracle'/'raw'/f'oracle_seed_{row["seed"]}.json'
            oracle = read(oracle_path)['return_target_progress_m']
            for label in labels:
                receipt = base/label/'completed.json'
                metadata = read(receipt)
                if suite=='grouped' and (metadata['identity'].get('row')!=row or
                                         metadata['identity'].get('label')!=label):
                    raise ValueError(f'Grouped evaluation round mismatch: {receipt}')
                historical = evaluation_identity(metadata['identity'], legacy=True)
                if any(historical.get(k) != v for k,v in evaluation_identity(self.identity).items()):
                    raise ValueError(f'Evaluation belongs to different inputs: {receipt}')
                verified(receipt.parent, metadata['identity'])
                summary = next((receipt.parent/'raw').glob('*.json'))
                item = read(summary)
                item['oracle_return_ratio'] = item['return_target_progress_m']/oracle if oracle > 1e-8 else None
                item['suite'] = row['id']
                item['evaluation_slip'] = row['slip']
                item['evaluation_seed'] = row['seed']
                if suite=='grouped':
                    item.update(group=row['group'],parent_mean=row['parent_mean'],parent_std=row['parent_std'])
                rows.append(item)
        nominal = {}
        for label in labels:
            values = [r['return_target_progress_m'] for r in rows
                      if r['policy_label']==label and r['suite'].startswith('nominal_')]
            nominal[label] = float(np.mean(values)) if values else None
        for item in rows:
            reference = nominal[item['policy_label']]
            item['nominal_retention_ratio_of_run_to_nominal_mean'] = (
                item['return_target_progress_m']/reference if reference is not None and reference > 1e-8 else None)
        costs = []
        for receipt in sorted(self.root.glob('data/*/*/*/completed.json')):
            verified(receipt.parent, read(receipt)['identity'])
            costs.append(dict(directory=str(receipt.parent), **read(receipt.parent/'metrics.json')))
        for receipt in sorted(self.root.glob('policies/*/*/iteration_*/segments/*/completed.json')):
            verified(receipt.parent, read(receipt)['identity'])
            costs.append(dict(directory=str(receipt.parent), **read(receipt.parent/'metrics.json')))
        fidelity = []
        if not frozen_only:
            import torch
            from train import BCPolicy
            from gail_models import GAILActorCritic
            from .worker import load_shards
            for arm in arms:
                shard_paths = []
                for row in self.rows('test', arm):
                    directory = self.root/'data'/arm/'test'/row['id']
                    stored = read(directory/'completed.json')
                    verified(directory, stored['identity'])
                    shard_paths.append(str(directory/'data.npz'))
                x, y = load_shards(shard_paths)
                for label, initial in MODELS.items():
                    models = [(a, read(
                        self.root/'policies'/label/a/'final.json')['model']) for a in arms]
                    for variant, checkpoint in models:
                        policy = GAILActorCritic(63,16) if label.startswith('gail') else BCPolicy(63,16)
                        policy.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                        policy.eval()
                        squared = 0.
                        with torch.no_grad():
                            for start in range(0, len(x), 512):
                                bx = torch.from_numpy(x[start:start+512])
                                prediction = policy.actor_mean(bx) if label.startswith('gail') else policy(bx)
                                squared += float((prediction-torch.from_numpy(y[start:start+512])).square().sum())
                        fidelity.append(dict(policy=label+'_'+variant, test_arm=arm, samples=len(x),
                                             latent_element_mse=squared/y.size,
                                             latent_squared_l2_per_state=squared/len(x)))
        aggregates=[]
        for label in labels:
            selected=[r for r in rows if r['policy_label']==label]
            ratios=[r['oracle_return_ratio'] for r in selected if r['oracle_return_ratio'] is not None]
            aggregates.append(dict(policy=label,rounds=len(selected),
                mean_progress_m=float(np.mean([r['return_target_progress_m'] for r in selected])),
                mean_oracle_ratio=float(np.mean(ratios)) if ratios else None,
                total_falls=sum(r['falls'] for r in selected),
                mean_targets=float(np.mean([r['targets_reached'] for r in selected]))))
        grouped=[]
        if suite=='grouped':
            for group in dict.fromkeys(r['group'] for r in rows):
                for label in labels:
                    selected=[r for r in rows if r['group']==group and r['policy_label']==label]
                    ratios=[r['oracle_return_ratio'] for r in selected if r['oracle_return_ratio'] is not None]
                    grouped.append(dict(group=group,policy=label,parent_mean=selected[0]['parent_mean'],
                        parent_std=selected[0]['parent_std'],rounds=len(selected),
                        mean_progress_m=float(np.mean([r['return_target_progress_m'] for r in selected])),
                        mean_oracle_ratio=float(np.mean(ratios)) if ratios else None,
                        total_falls=sum(r['falls'] for r in selected),
                        mean_targets=float(np.mean([r['targets_reached'] for r in selected]))))
        output=self.evaluation_directory(suite) if suite=='grouped' else self.root
        distribution=(dict(kind='grouped_truncated_gaussian',means=self.plan['training_environment']['slips'],
                           std_fraction=.5,low=0.,low_exclusive=True,high=10.)
                      if suite=='grouped' else self.plan['distribution'])
        save(output/'report.json', dict(protocol=PROTOCOL, evaluation_suite=suite, runs=rows, collection_costs=costs,
             training_environment=self.plan['training_environment'],
             evaluation_distribution=distribution, aggregates=aggregates, group_aggregates=grouped,
             held_out_action_fidelity=fidelity,
             note='Oracle ratios are not random-normalized returns; random policy is outside this study.'))
        lines = ['# Variable-slip comparison', '',
                 f'Evaluation suite: {suite}. Oracle ratios are not random-normalized returns. '
                 'Nominal retention is unavailable without nominal runs in the selected suite. '
                 'Quick plans are engineering tests only.', '',
                 'Each round holds one sampled slip constant for every controller, using a shared task seed and fresh servers. '
                 'Summary differences are descriptive; rounds are not independent training replicas.', '',
                 '| Round | Fixed slip | Seed | Policy | Progress (m) | Oracle ratio | Falls | Targets |',
                 '|---|---:|---:|---|---:|---:|---:|---:|']
        for r in rows:
            ratio = 'n/a' if r['oracle_return_ratio'] is None else f'{r["oracle_return_ratio"]:.3f}'
            lines.append(f'| {r["suite"]} | {r["evaluation_slip"]:.6f} | {r["evaluation_seed"]} | {r["policy_label"]} | {r["return_target_progress_m"]:.3f} | {ratio} | {r["falls"]} | {r["targets_reached"]} |')
        lines += ['', '## Across-round summary', '',
                  '| Policy | Rounds | Mean progress (m) | Mean oracle ratio | Total falls | Mean targets |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in aggregates:
            ratio='n/a' if r['mean_oracle_ratio'] is None else f'{r["mean_oracle_ratio"]:.3f}'
            lines.append(f'| {r["policy"]} | {r["rounds"]} | {r["mean_progress_m"]:.3f} | {ratio} | {r["total_falls"]} | {r["mean_targets"]:.2f} |')
        if grouped:
            lines += ['', '## Per-Gaussian-group comparison', '',
                      'Parent mean/std are before truncation to (0,10]. The group centred at10 is cut off on its upper side; sampled means need not equal parent means.', '',
                      '| Group | Parent mean | Parent std | Policy | Draws | Mean progress m | Mean oracle ratio | Total falls | Mean targets |',
                      '|---|---:|---:|---|---:|---:|---:|---:|---:|']
            for r in grouped:
                ratio='n/a' if r['mean_oracle_ratio'] is None else f'{r["mean_oracle_ratio"]:.3f}'
                lines.append(f'| {r["group"]} | {r["parent_mean"]} | {r["parent_std"]} | {r["policy"]} | {r["rounds"]} | {r["mean_progress_m"]:.3f} | {ratio} | {r["total_falls"]} | {r["mean_targets"]:.2f} |')
        if fidelity:
            lines += ['', '## Held-out latent action error', '',
                      '| Policy | Test surface arm | Element MSE | Squared L2/state |', '|---|---|---:|---:|']
            for item in fidelity:
                lines.append(f'| {item["policy"]} | {item["test_arm"]} | {item["latent_element_mse"]:.6g} | {item["latent_squared_l2_per_state"]:.6g} |')
        (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'collect', 'bc', 'dagger_mixed', 'gail_bc',
                                         'gail_dagger', 'all', 'evaluate', 'baseline', 'report'])
    parser.add_argument('--plan', default='experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json')
    parser.add_argument('--arm', choices=['both', 'varied', 'nominal'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--low', type=float, default=.025)
    parser.add_argument('--high', type=float, default=10.)
    parser.add_argument('--mean', type=float, default=5.0125)
    parser.add_argument('--std', type=float, default=2.5)
    parser.add_argument('--quick', action='store_true', help='Tiny engineering plan, never research results')
    parser.add_argument('--frozen-only', action='store_true', help='Report just oracle and frozen baselines')
    parser.add_argument('--suite', choices=['grouped','gaussian', 'fixed', 'all'], default='grouped',
                        help='Evaluation/report suite (default: grouped Gaussians; gaussian keeps previous five-round suite)')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.stage == 'plan':
        plan = make_plan(args.seed, args.low, args.high, args.mean, args.std, args.quick)
        path = Path(args.plan).resolve()
        if path.exists() and read(path) != plan:
            raise ValueError('Plan exists with different settings; choose a new --plan path')
        save(path, plan)
        print(path)
        return
    arms = ['varied', 'nominal'] if args.arm=='both' else [args.arm]
    # Report is offline; all other stages obey the same lock as Phase 2.
    if args.stage == 'report':
        Experiment(args.plan).report(arms, frozen_only=args.frozen_only, suite=args.suite)
        return
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable, str(ROOT/'check_simulator_config.py')], check=True)
        subprocess.run([sys.executable, '-c',
                        'from scripts.commons.Script import Script; Script.build_cpp_modules(); '
                        'from cpp.ball_predictor import ball_predictor; '
                        'from cpp.localization import localization; from cpp.a_star import a_star'], check=True)
        exp = Experiment(args.plan)
        if args.stage in ('evaluate', 'baseline'):
            exp.evaluate(arms, frozen_only=args.stage=='baseline', suite=args.suite)
            return
        for arm in arms:
            if args.stage in ('collect', 'all'):
                for group in exp.plan['groups']:
                    if not group.startswith('dagger_'):
                        exp.demonstrations(arm, group)
            if args.stage in ('bc', 'all'):
                exp.train_supervised('bc', arm)
            if args.stage in ('dagger_mixed', 'all'):
                exp.train_supervised('dagger_mixed', arm)
            for label in ('gail_bc', 'gail_dagger'):
                if args.stage in (label, 'all'):
                    exp.train_gail(label, arm)


if __name__ == '__main__':
    main()

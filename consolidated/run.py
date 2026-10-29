"""Two public entry points backed by immutable jobs and owned simulator processes."""
import argparse
import csv
import io
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

from variable_training.common import ROOT, begin, commit, digest, read, save, verified, simulator_lock, source_hashes
from .plan import PROTOCOL, ALGORITHMS, make_plan, segments, evaluation_rows
from .server import SCENE, SYSTEM, simulator, launch, require_free_ports

DEFAULT = ROOT/'experiments/consolidated/v1'


def training_summary(root, plan):
    records=[read(p) for p in root.glob('policies/*/*/seed_*/*/final.json')]
    paths=list(root.glob('data/*/*/*/metrics.json'))+list(root.glob('policies/*/*/seed_*/*/iteration_*/segments/*/metrics.json'))
    totals={key:0 for key in ('active_steps','oracle_queries','rollout_and_recovery_steps','startup_received_steps')}
    for path in paths:
        if not (path.parent/'completed.json').exists(): continue
        metrics=read(path)
        for key in totals: totals[key]+=metrics.get(key,0)
    save(root/'training_summary.json',dict(plan=plan,completed_policies=records,
         unique_collection_totals=totals,
         accounting='Shared demonstration collections counted once; validation/test included here. Per-policy active training budgets and inherited lineage are in final records.'))
    return len(records)


def sources():
    result = source_hashes()
    for folder in ('consolidated', 'context_training'):
        for path in (ROOT/folder).glob('*.py'):
            result[str(path.relative_to(ROOT))] = digest(path)
    return result


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Inputs changed; use a new output directory: {path}')
    if not path.exists():
        save(path, value)


def validate_artifacts(directory, spec):
    """Validate complete horizons and actual checkpoint schema before committing."""
    task = spec['task']
    if task == 'collect':
        count = spec['row']['steps']
        dim = 64 if spec.get('architecture') == 'input' else 63
        with np.load(directory/'data.npz', allow_pickle=False) as data:
            shapes = dict(obs=(count, dim), actions=(count, 16), executed_actions=(count, 16),
                          episode_id=(count,), logprobs=(count,), values=(count,),
                          dones=(count,), bootstrap_value=(), slip=())
            for name, shape in shapes.items():
                if data[name].shape != shape or not np.isfinite(data[name]).all():
                    raise ValueError(f'Invalid collected array {name}')
            if spec['kind']=='gail' and not np.array_equal(data['actions'], data['executed_actions']):
                raise ValueError('GAIL stored action differs from executed action')
        if read(directory/'metrics.json')['active_steps'] != count:
            raise ValueError('Incomplete collection')
    elif task == 'evaluate':
        result = read(directory/'result.json')
        for key in ('row', 'controller', 'supplied_slip'):
            if result[key] != spec[key]:
                raise ValueError(f'Evaluation metadata mismatch: {key}')
        if result['active_steps'] != spec['steps']:
            raise ValueError('Incomplete evaluation')
        with np.load(directory/'trace.npz', allow_pickle=False) as data:
            for name, width in [('obs', 63), ('actions', 16), ('oracle_actions', 16), ('joint_positions_deg', 22)]:
                if data[name].shape != (spec['steps'], width):
                    raise ValueError(f'Invalid trace: {name}')
            if any(len(data[k]) != spec['steps'] or not np.isfinite(data[k]).all() for k in data.files):
                raise ValueError('Incomplete/nonfinite evaluation trace')
            if not np.isclose(data['target_progress'].sum(), result['return_target_progress_m']):
                raise ValueError('Trace/summary progress mismatch')
    else:
        from .models import load_policy
        load_policy(directory/'policy.pth', 'gail' if task=='gail' or spec.get('kind')=='gail' else 'bc',
                    spec['architecture'])
        if task != 'initialize':
            read(directory/'metrics.json')


class Experiment:
    def __init__(self, root, plan, retries=1):
        import torch
        self.root, self.plan, self.retries = root, plan, retries
        self._audited = set()
        self.identity = dict(protocol=PROTOCOL, plan=digest(root/'plan.json'), sources=sources(),
            libraries=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                           device='cuda' if torch.cuda.is_available() else 'cpu'),
            environment={str(p):digest(p) for p in [SCENE, SYSTEM/'rcssserver3d.rb',
                SYSTEM/'naosoccersim.rb', Path.home()/'.simspark/spark.rb']})
        immutable(root/'runtime_identity.json', self.identity)

    def job(self, directory, spec, inputs=(), live=False):
        directory = Path(directory)
        identity = dict(self.identity, spec=spec, inputs={str(p):digest(p) for p in inputs})
        if verified(directory, identity):
            validate_artifacts(directory, spec)
            print(f'[Resume] {directory.relative_to(self.root)}', flush=True)
            return directory
        for attempt in range(self.retries+1 if live else 1):
            begin(directory)
            save(directory/'job.json', spec)
            module = 'consolidated.eval_worker' if spec['task']=='evaluate' else 'consolidated.worker'
            command = [sys.executable, '-u', '-m', module, str(directory/'job.json')]
            try:
                if live:
                    with simulator(directory, spec['row']['slip']) as manifest:
                        launch(command, directory/'worker.log', server=manifest['server_process'])
                else:
                    launch(command, directory/'worker.log')
                validate_artifacts(directory, spec)
                if identity['inputs'] != {str(p):digest(p) for p in inputs}:
                    raise ValueError('Job inputs changed during execution')
                commit(directory, identity)
                return directory
            except Exception as error:
                save(directory/'failure.json', {'error':repr(error), 'attempt':attempt})
                if not live or attempt == self.retries:
                    raise

    def data(self, domain, group):
        b = self.plan['budgets']
        totals = dict(seed=b['dagger_seed'], extra=b['expert_train']-b['dagger_seed'],
                      validation=b['validation'], test=b['test'])
        offset = ['seed', 'extra', 'validation', 'test'].index(group)*100
        paths = []
        for row in segments(self.plan, domain, group, totals[group], 4200000+offset):
            base = self.root/'data'/domain/group/row['id']
            self.job(base, dict(task='collect', kind='oracle', architecture='hidden', row=row), live=True)
            paths.append(str(base/'data.npz'))
        return paths

    def final_path(self, domain, architecture, seed, algorithm):
        return self.root/'policies'/domain/architecture/f'seed_{seed}'/algorithm/'final.json'

    def audit_ancestry(self, directory):
        directory=Path(directory)
        if directory in self._audited:
            return
        stored=read(directory/'completed.json')['identity']
        if any(stored[k]!=self.identity[k] for k in self.identity) or not verified(directory,stored):
            raise ValueError(f'Invalid completed dependency: {directory}')
        for filename,expected in stored['inputs'].items():
            if digest(filename)!=expected:
                raise ValueError(f'Completed input changed: {filename}')
            parent=Path(filename).parent
            if (parent/'completed.json').exists():
                self.audit_ancestry(parent)
        self._audited.add(directory)

    def final(self, domain, architecture, seed, algorithm):
        path = self.final_path(domain, architecture, seed, algorithm)
        record = read(path)
        if record['plan'] != digest(self.root/'plan.json') or record['sha256'] != digest(record['model']):
            raise ValueError(f'Final policy identity changed: {path}')
        update = Path(record['model']).parent
        self.audit_ancestry(update)
        for test in (self.root/'data'/domain/'test').glob('*/data.npz'):
            self.audit_ancestry(test.parent)
        return record

    def train(self, domain, architecture, seed, algorithm):
        final_path = self.final_path(domain, architecture, seed, algorithm)
        if final_path.exists():
            return self.final(domain, architecture, seed, algorithm)
        b, settings = self.plan['budgets'], self.plan['settings']
        base = final_path.parent
        kind = 'gail' if algorithm.startswith('gail') else 'bc'
        initializer = None
        if kind == 'gail':
            parent = 'bc' if algorithm == 'gail_bc' else 'dagger_mixed'
            initializer = self.train(domain, architecture, seed, parent)
        elif domain == 'varied':
            initializer = self.train('original', 'hidden', 42, algorithm)
        init = dict(task='initialize', architecture=architecture, kind=kind, seed=seed,
                    model=initializer['model'] if initializer else None)
        model = str(self.job(base/'initialization', init,
                    [initializer['model']] if initializer else [])/'policy.pth')
        optimizer, discriminator = None, None
        seed_data = self.data(domain, 'seed')
        validation = self.data(domain, 'validation')
        tests = self.data(domain, 'test')
        oracle_data = seed_data + self.data(domain, 'extra') if algorithm != 'dagger_mixed' else seed_data
        training = list(oracle_data)
        iterations = b['iterations'] if algorithm != 'bc' else 0
        for iteration in range(1 if kind=='gail' else 0, iterations+1):
            stage = base/f'iteration_{iteration:03d}'
            rollouts = []
            if iteration:
                count = b['learner_steps']//b['iterations']
                for row in segments(self.plan, domain, f'rollout_{iteration}', count, seed*100000+1000+iteration*20):
                    spec = dict(task='collect', kind='gail' if kind=='gail' else 'dagger',
                                architecture=architecture, model=model, row=row,
                                beta=max(0., 1.-.2*iteration))
                    collected = self.job(stage/'segments'/row['id'], spec, [model], live=True)
                    rollouts.append(str(collected/'data.npz'))
                if kind != 'gail':
                    training.extend(rollouts)
            if kind == 'gail':
                spec = dict(task='gail', architecture=architecture, model=model, optimizer=optimizer,
                    discriminator=discriminator, expert=oracle_data, normalizer=oracle_data,
                    rollouts=rollouts, seed=seed+iteration, iteration=iteration, label=algorithm,
                    ppo_epochs=settings['ppo_epochs'], discriminator_epochs=settings['discriminator_epochs'])
                inputs = oracle_data+rollouts+[model]+([optimizer,discriminator] if optimizer else [])
            else:
                spec = dict(task='supervised', architecture=architecture, model=model, optimizer=optimizer,
                    train=training.copy(), validation=validation, seed=seed+iteration,
                    epochs=settings['bc_epochs' if algorithm=='bc' else 'dagger_epochs'],
                    learning_rate=settings['learning_rate'], batch_size=settings['batch_size'])
                inputs = training+validation+[model]+([optimizer] if optimizer else [])
            update = self.job(stage/'update', spec, inputs)
            model, optimizer = str(update/'policy.pth'), str(update/'optimizer.pth')
            discriminator = str(update/'discriminator.pth') if kind=='gail' else None
        record = dict(model=model, sha256=digest(model), plan=digest(self.root/'plan.json'),
            domain=domain, architecture=architecture, seed=seed, algorithm=algorithm, kind=kind,
            initializer=initializer, expert_training_samples=b['expert_train'],
            active_oracle_labels=0 if kind=='gail' else b['expert_train'],
            learner_steps=0 if algorithm=='bc' else b['learner_steps'],
            validation_samples=b['validation'], test_samples=b['test'],
            updates=[dict(path=str(p),metrics=read(p)) for p in sorted(base.glob('iteration_*/update/metrics.json'))],
            offline_test=self.offline_score(model, kind, architecture, tests))
        save(final_path, record)
        return record

    @staticmethod
    def offline_score(model_path, kind, architecture, paths):
        from .models import load_policy
        from .worker import load_shards
        import torch
        x,y = load_shards(paths, architecture)
        model = load_policy(model_path, kind, architecture)
        error = 0.
        with torch.no_grad():
            for start in range(0,len(x),1024):
                tensor = torch.from_numpy(x[start:start+1024])
                predicted = model.actor_mean(tensor) if kind=='gail' else model(tensor)
                error += float(np.square(predicted.numpy()-y[start:start+1024]).sum())
        return dict(element_mse=error/y.size, squared_l2_per_state=error/len(y), samples=len(y))

    def controllers(self, cohort):
        result = [dict(label=kind, kind=kind, architecture='hidden', model=None, domain='baseline',
                       seed=None, algorithm=kind, mode='none') for kind in ('oracle','random')]
        for domain in ('original','varied'):
            if cohort != 'all' and domain != cohort:
                continue
            for architecture in (['hidden'] if domain=='original' else self.plan['architectures']):
                for seed in ([42] if domain=='original' else self.plan['varied_seeds']):
                    for algorithm in ALGORITHMS:
                        info = self.final(domain,architecture,seed,algorithm)
                        result.append(dict(label=f'{domain}_{architecture}_{seed}_{algorithm}',
                            kind=info['kind'], architecture=architecture, model=info['model'],
                            sha256=info['sha256'], domain=domain, seed=seed, algorithm=algorithm,
                            mode='correct' if architecture=='input' else 'none'))
        return result

    def evaluate(self, suite, cohort, modes, report_only=False):
        rows = evaluation_rows(suite, self.plan['quick'])
        if cohort=='varied' and suite=='original':
            raise ValueError('Varied/context models use the varied suite; choose --suite varied')
        controllers = self.controllers(cohort)
        expanded=[]
        for c in controllers:
            for mode in (modes if c['architecture']=='input' else ['none']):
                expanded.append(dict(c, mode=mode, label=c['label']+(f'_{mode}' if c['architecture']=='input' else '')))
        output = self.root/'evaluation'/f'{suite}_{cohort}_{"-".join(modes)}'
        schedule = dict(identity=self.identity, rows=rows, controllers=expanded,
                        steps=self.plan['settings']['evaluation_steps'])
        if not report_only:
            immutable(output/'schedule.json', schedule)
        elif read(output/'schedule.json') != schedule:
            raise ValueError('Report schedule changed')
        for row in rows:
            domain = 'original' if row['slip'] is None else 'varied'
            # Training expert pool fixes random action bounds; never use held-out labels for bounds.
            expert_paths = sorted((self.root/'data'/domain/'seed').glob('*/data.npz')) + sorted((self.root/'data'/domain/'extra').glob('*/data.npz'))
            if not expert_paths:
                raise ValueError(f'Missing {domain} training expert pool')
            expert_file = self.root/'data'/domain/'random_bounds_pool.npz'
            from .worker import load_shards
            if not report_only:
                x,y = load_shards([str(p) for p in expert_paths])
                # Bounds only; preserving min/max gives exactly the original uniform action rule.
                with_bounds = np.stack([y.min(axis=0),y.max(axis=0)])
                if not expert_file.exists():
                    np.savez_compressed(expert_file, obs=np.zeros((2,63),np.float32), actions=with_bounds)
                else:
                    with np.load(expert_file) as bounds:
                        if not np.array_equal(bounds['actions'],with_bounds):
                            raise ValueError('Random bounds changed')
            for c in expanded:
                if row['slip'] is None and c['domain']=='varied':
                    continue
                supplied = float(f'{row["slip"]:.12g}') if row['slip'] is not None else None
                if c['mode']=='constant': supplied=.1
                if c['mode']=='wrong': supplied=10. if row['slip']<1 else .025
                spec=dict(task='evaluate',row=row,controller=c,supplied_slip=supplied,
                          steps=schedule['steps'],expert_file=str(expert_file))
                directory=output/'runs'/row['id']/c['label']
                inputs=[str(expert_file)]+([c['model']] if c['model'] else [])
                if report_only:
                    identity=dict(self.identity,spec=spec,inputs={str(p):digest(p) for p in inputs})
                    if not verified(directory,identity): raise ValueError(f'Missing evaluation: {directory}')
                    validate_artifacts(directory,spec)
                else:
                    self.job(directory,spec,inputs,live=True)
        from .report import report
        report(self, output, schedule)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline',choices=['training','evaluation'])
    parser.add_argument('stage',choices=['plan','run','report'],nargs='?',default='plan')
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--domain',choices=['all','original','varied'],default='all')
    parser.add_argument('--algorithm',choices=['all',*ALGORITHMS],default='all')
    parser.add_argument('--suite',choices=['all','original','varied'],default='all')
    parser.add_argument('--cohort',choices=['all','original','varied'],default='all')
    parser.add_argument('--context-modes',nargs='+',choices=['correct','constant','wrong'],default=['correct'])
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--retries',type=int,default=1)
    args=parser.parse_args()
    if args.retries<0: parser.error('--retries must be nonnegative')
    if len(set(args.context_modes))!=len(args.context_modes) or 'correct' not in args.context_modes:
        parser.error('--context-modes must contain correct and no duplicates')
    if args.quick and args.output==DEFAULT:
        parser.error('--quick requires a separate --output')
    root=args.output.resolve()
    plan=make_plan(args.quick)
    immutable(root/'plan.json',plan)
    if args.stage=='plan':
        print(f'Plan: {root/"plan.json"}\nBudgets: {plan["budgets"]}')
        if args.pipeline=='evaluation':
            n_original=4 if args.cohort in ('all','original') else 0
            n_varied=4*len(plan['varied_seeds'])*(1+len(args.context_modes)) if args.cohort in ('all','varied') else 0
            rows=evaluation_rows(args.suite,plan['quick'])
            total=sum(2+n_original+(n_varied if row['slip'] is not None else 0) for row in rows)
            print(f'Evaluation: {total} deployments over {len(rows)} rounds; 3000 active steps each (32 for quick).')
        return
    if args.pipeline=='training' and args.stage=='report':
        count=training_summary(root,plan)
        print(f'{count} policy records: {root/"training_summary.json"}')
        return
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if args.stage=='report':
        Experiment(root,plan,args.retries).evaluate(args.suite,args.cohort,args.context_modes,True)
        return
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
        subprocess.run([sys.executable,'-c','from scripts.commons.Script import Script; Script.build_cpp_modules()'],check=True)
        exp=Experiment(root,plan,args.retries)
        if args.pipeline=='evaluation':
            exp.evaluate(args.suite,args.cohort,args.context_modes)
            return
        for domain in (['original','varied'] if args.domain=='all' else [args.domain]):
            for architecture in (['hidden'] if domain=='original' else plan['architectures']):
                for seed in ([42] if domain=='original' else plan['varied_seeds']):
                    for algorithm in (ALGORITHMS if args.algorithm=='all' else [args.algorithm]):
                        exp.train(domain,architecture,seed,algorithm)
        training_summary(root,plan)


if __name__=='__main__':
    main()

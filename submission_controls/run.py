"""Separate controls; historical training/evaluation jobs are read-only dependencies."""
import argparse
import csv
import io
import math
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import torch

from consolidated.run import DEFAULT as PARENT, Experiment, immutable, sources, validate_artifacts
from consolidated.plan import evaluation_rows
from consolidated.server import simulator, launch, require_free_ports
from scratch_comparison.run import Parent
from variable_training.common import ROOT, read, save, digest, begin, commit, verified, simulator_lock

PROTOCOL = 'submission_controls_v1'
DEFAULT = ROOT/'experiments/submission_controls/v1'


def make_plan(parent, quick=False, architectures=('hidden', 'input'), diagnostics=False):
    return dict(protocol=PROTOCOL, parent=str(parent), quick=quick,
                architectures=['hidden'] if quick else list(architectures),
                seeds=[42] if quick else [42, 43, 44], diagnostics=diagnostics,
                matched_updates=3 if quick else 211050,
                varied_epochs=2 if quick else 200,
                batch_size=32 if quick else 256, learning_rate=1e-4,
                evaluation_steps=32 if quick else 3000,
                patience=2 if quick else 25, min_delta=1e-5,
                diagnostic_max_epochs=4 if quick else 500,
                diagnostic_min_epochs=1 if quick else 100,
                selection='validation-only; allocated and selected update costs both recorded',
                scope='Original matched-update BC; varied-only 200-epoch BC; zero native gait; original round fidelity curves')


def controller(label, model=None, domain='original', architecture='hidden', seed=42,
               kind='bc', **extra):
    return dict(label=label, kind=kind, model=str(model) if model else None,
                sha256=digest(model) if model else None, domain=domain,
                architecture=architecture, seed=seed, **extra)


class Study:
    def __init__(self, output, plan, retries=1):
        self.root, self.plan, self.retries = output.resolve(), plan, retries
        self.parent = Parent(Path(plan['parent']))
        if self.parent.plan['quick']:
            raise ValueError('Use the completed full parent; controls have their own --quick mode')
        if self.root.is_relative_to(self.parent.root) or self.parent.root.is_relative_to(self.root):
            raise ValueError('Controls output must be separate from the parent tree')
        immutable(self.root/'plan.json', plan)
        self.parent.check_runtime()
        src = sources()
        for folder in ('submission_controls', 'scratch_comparison'):
            src.update({str(p.relative_to(ROOT)): digest(p) for p in (ROOT/folder).glob('*.py')})
        self.identity = dict(protocol=PROTOCOL, plan=digest(self.root/'plan.json'), sources=src,
                             parent_identity=digest(self.parent.root/'runtime_identity.json'),
                             environment=self.parent.identity['environment'],
                             python=sys.version, numpy=np.__version__, torch=torch.__version__,
                             cuda_available=torch.cuda.is_available())
        immutable(self.root/'runtime_identity.json', self.identity)
        self.audited = set()
        self.checked_runtimes = set()

    def audit_source(self, directory):
        """Verify original OR scratch receipts without adding new files to their identities."""
        directory = Path(directory).resolve()
        if directory in self.audited:
            return
        scratch = self.parent.root/'scratch_v1'
        if not directory.is_relative_to(self.parent.root):
            raise ValueError(f'Historical dependency outside parent: {directory}')
        owner = scratch if directory.is_relative_to(scratch) else self.parent.root
        runtime = read(owner/'runtime_identity.json')
        receipt = read(directory/'completed.json')
        identity = receipt['identity']
        if any(identity.get(k) != v for k, v in runtime.items()) or not verified(directory, identity):
            raise ValueError(f'Invalid historical receipt: {directory}')
        if owner not in self.checked_runtimes:
            if runtime['plan'] != digest(owner/'plan.json'):
                raise ValueError('Historical plan changed')
            for name, expected in runtime['sources'].items():
                if digest(ROOT/name) != expected:
                    raise ValueError(f'Historical source changed: {name}')
            for name, expected in runtime['environment'].items():
                if digest(name) != expected:
                    raise ValueError(f'Historical environment changed: {name}')
            self.checked_runtimes.add(owner)
        for name, expected in identity['inputs'].items():
            if digest(name) != expected:
                raise ValueError(f'Historical input changed: {name}')
            if (Path(name).parent/'completed.json').exists():
                self.audit_source(Path(name).parent)
        validate_artifacts(directory, identity['spec'])
        self.audited.add(directory)

    def training_jobs(self):
        p = self.parent.root
        original = p/'policies/original/hidden/seed_42'
        dagger_cost = sum(read(original/f'dagger_mixed/iteration_{i:03d}/update/metrics.json')['gradient_updates']
                          for i in range(8))
        if dagger_cost != 211050:
            raise ValueError('Parent DAgger update allocation differs from this controls protocol')
        defs = [('bc_original_matched', original/'bc/iteration_000/update',
                 'original', 'hidden', 42, dict(updates=self.plan['matched_updates']))]
        for architecture in self.plan['architectures']:
            for seed in self.plan['seeds']:
                folder = p/f'scratch_v1/policies/varied/{architecture}/seed_{seed}/bc/iteration_000/update'
                defs.append((f'bc_varied_{architecture}_{seed}_200', folder, 'varied', architecture,
                             seed, dict(epochs=self.plan['varied_epochs'])))
        if self.plan['diagnostics']:
            defs += [('bc_original_early', original/'bc/iteration_000/update', 'original', 'hidden', 42,
                      dict(epochs=self.plan['diagnostic_max_epochs'], early_stop=True)),
                     ('dagger_original_fixed_aggregate_early', original/'dagger_mixed/iteration_007/update',
                      'original', 'hidden', 49,
                      dict(epochs=self.plan['diagnostic_max_epochs'], early_stop=True, minimum_epochs=1))]
        result = []
        for label, directory, domain, architecture, seed, budget in defs:
            self.audit_source(directory)
            source = read(directory/'job.json')
            # BC controls start at the exact saved random initialization, not epoch100.
            model, optimizer = source['model'], None
            inherited = {}
            if label.startswith('dagger_'):
                model, optimizer = str(directory/'policy.pth'), str(directory/'optimizer.pth')
                inherited = dict(gradient_updates=211050, sample_exposures=54000000,
                                 oracle_labels=170000, learner_steps=70000,
                                 note='Continuation on fixed final aggregate; no new oracle queries/rollouts')
            self.audit_source(Path(model).parent)
            for path in source['train'] + source['validation']:
                self.audit_source(Path(path).parent)
            job = dict(task='control_train', label=label, domain=domain, architecture=architecture,
                       model=model, optimizer=optimizer, train=source['train'], validation=source['validation'],
                       seed=seed, batch_size=self.plan['batch_size'], learning_rate=self.plan['learning_rate'],
                       quick=self.plan['quick'], patience=self.plan['patience'], min_delta=self.plan['min_delta'],
                       minimum_epochs=self.plan['diagnostic_min_epochs'], inherited_costs=inherited)
            job.update(budget)
            result.append(job)
        return result

    def validate(self, directory, job):
        if job['task'] == 'evaluate':
            validate_artifacts(directory, job)
        else:
            from consolidated.models import load_policy
            model = load_policy(directory/'policy.pth', 'bc', job.get('architecture', 'hidden'))
            if job['task'] == 'zero':
                with torch.no_grad():
                    if not torch.equal(model(torch.randn(5, 63)), torch.zeros(5, 16)):
                        raise ValueError('Zero baseline is not identically zero')
            else:
                m = read(directory/'metrics.json')
                if not job.get('early_stop') and m['gradient_updates'] != m['target_gradient_updates']:
                    raise ValueError('Training budget incomplete')
                if not math.isfinite(m['validation_mse']):
                    raise ValueError('Invalid validation MSE')

    def job(self, directory, job, inputs=(), live=False, check_only=False):
        identity = dict(runtime=self.identity, spec=job, inputs={str(p): digest(p) for p in inputs})
        if verified(directory, identity):
            self.validate(directory, job)
            print(f'[Resume] {directory.relative_to(self.root)}', flush=True)
            return directory
        if check_only:
            raise ValueError(f'Incomplete control: {directory}; run its training/evaluation stage')
        for attempt in range(self.retries+1 if live else 1):
            begin(directory)
            save(directory/'job.json', job)
            try:
                if job['task'] == 'zero':
                    from train import BCPolicy
                    model = BCPolicy(63, 16)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.zero_()
                    torch.save(model.state_dict(), directory/'policy.pth')
                else:
                    module = 'consolidated.eval_worker' if live else 'submission_controls.train'
                    command = [sys.executable, '-u', '-m', module, str(directory/'job.json')]
                    if live:
                        with simulator(directory, job['row']['slip']) as manifest:
                            launch(command, directory/'worker.log', server=manifest['server_process'])
                    else:
                        launch(command, directory/'worker.log')
                self.validate(directory, job)
                if identity['inputs'] != {str(p): digest(p) for p in inputs}:
                    raise ValueError('Inputs changed during job')
                commit(directory, identity)
                return directory
            except Exception as error:
                save(directory/'failure.json', dict(error=repr(error), attempt=attempt))
                if not live or attempt == self.retries:
                    raise

    def trained(self, check_only=False):
        result = []
        for job in self.training_jobs():
            inputs = job['train']+job['validation']+[job['model']]+([job['optimizer']] if job['optimizer'] else [])
            directory = self.job(self.root/'training'/job['label'], job, inputs, check_only=check_only)
            result.append(controller(job['label'], directory/'policy.pth', job['domain'],
                                     job['architecture'], job['seed'], algorithm=job['label']))
        return result

    def zero(self, check_only=False):
        directory = self.job(self.root/'training/zero_native', dict(task='zero', architecture='hidden'),
                             check_only=check_only)
        return controller('zero_native', directory/'policy.pth', domain='baseline', seed=None,
                          semantics='All16 native gait actions zero; standard gait/IK/filter/reset unchanged')

    def rows(self):
        # The saved schedule is authoritative: no fresh Gaussian draws.
        path = self.parent.root/'evaluation/all_all_correct/schedule.json'
        schedule = read(path)
        if schedule['identity'] != self.parent.identity:
            raise ValueError('Parent evaluation schedule identity differs')
        rows = schedule['rows']
        if rows != evaluation_rows('all', False):
            raise ValueError('Unexpected parent evaluation rows')
        if self.plan['quick']:
            rows = [next(r for r in rows if r['suite']=='original'),
                    next(r for r in rows if r['suite']=='grouped')]
        return rows

    def evaluation(self, check_only=False, zero_only=False):
        controllers = ([] if zero_only else self.trained(check_only=True)) + [self.zero(check_only)]
        if self.plan['quick']:
            controllers += [controller(k, domain='baseline', kind=k, seed=None) for k in ('oracle', 'random')]
        schedule = dict(rows=self.rows(), controllers=controllers, steps=self.plan['evaluation_steps'],
                        parent_schedule_sha256=digest(self.parent.root/'evaluation/all_all_correct/schedule.json'))
        output = self.root/('evaluation_zero' if zero_only else 'evaluation')
        immutable(output/'schedule.json', schedule)
        for row in schedule['rows']:
            domain = 'original' if row['slip'] is None else 'varied'
            expert = self.parent.root/f'data/{domain}/random_bounds_pool.npz'
            for c in controllers:
                if row['slip'] is None and c['domain']=='varied':
                    continue
                job = dict(task='evaluate', row=row, controller=c,
                           supplied_slip=None if row['slip'] is None else float(f'{row["slip"]:.12g}'),
                           steps=schedule['steps'], expert_file=str(expert))
                inputs = [str(expert)]+([c['model']] if c['model'] else [])
                self.job(output/'runs'/row['id']/c['label'], job, inputs, live=True, check_only=check_only)
        return output, schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'audit', 'train', 'evaluate', 'zero', 'curves', 'report', 'run'])
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--architectures', nargs='+', choices=['hidden', 'input'], default=['hidden', 'input'])
    parser.add_argument('--diagnostics', action='store_true', help='Also early-stop BC and fixed-aggregate DAgger')
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--zero-only', action='store_true', help='Report the independent zero evaluation stage')
    args = parser.parse_args()
    if args.retries < 0 or len(set(args.architectures)) != len(args.architectures):
        parser.error('Invalid retries or duplicated architectures')
    parent = args.parent.resolve()
    output = (args.output or (DEFAULT.parent/'smoke_v1' if args.quick else
                              DEFAULT.parent/'diagnostics_v1' if args.diagnostics else DEFAULT)).resolve()
    if args.quick and output == DEFAULT.resolve():
        parser.error('Smoke must use a separate output directory')
    plan = make_plan(parent, args.quick, args.architectures, args.diagnostics)
    if args.stage == 'plan':
        if output.is_relative_to(parent) or parent.is_relative_to(output):
            parser.error('Output must be separate from historical experiments')
        immutable(output/'plan.json', plan)
        print(f'Plan: {output}/plan.json\n{plan}')
        print('Default full:7 BC controls,274 new deployments; old controls reused. Optional diagnostics add2 actors.')
        return
    study = Study(output, plan, args.retries)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.stage == 'audit':
        jobs = study.training_jobs()
        study.rows()
        for domain in ('original', 'varied'):
            study.parent.data(domain, 'test')
        print(f'[Audit] {len(jobs)} training specifications and shared datasets verified; no training/simulation')
        return
    if args.stage in ('train', 'run'):
        study.trained()
        study.zero()
    if args.stage in ('evaluate', 'zero', 'run'):
        # Build/check before taking the simulator lock; train never holds the live lock.
        with simulator_lock():
            require_free_ports()
            subprocess.run([sys.executable, str(ROOT/'check_simulator_config.py')], check=True)
            subprocess.run([sys.executable, '-c',
                            'from scripts.commons.Script import Script; Script.build_cpp_modules()'], check=True)
            # Parent sources/environment must still match after a possible native rebuild.
            study.parent.check_runtime()
            study.evaluation(zero_only=args.stage=='zero')
    if args.stage in ('curves', 'run'):
        from .report import curves
        curves(study)
    if args.stage in ('report', 'evaluate', 'zero', 'run'):
        from .report import report
        output, schedule = study.evaluation(check_only=True, zero_only=args.stage=='zero' or args.zero_only)
        report(study, output, schedule)


if __name__ == '__main__':
    main()

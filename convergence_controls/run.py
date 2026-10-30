"""Isolated validation-stopping diagnostics and untrained-network engine controls.

Copied/adapted from submission_controls; historical artifacts are read-only.
"""
import argparse
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import torch

from consolidated.run import DEFAULT as PARENT, immutable, sources, validate_artifacts
from consolidated.server import simulator, launch, require_free_ports
from scratch_comparison.run import Parent
from submission_controls.run import Study as OriginalStudy, controller
from variable_training.common import ROOT, read, save, digest, begin, commit, verified, simulator_lock

PROTOCOL = 'convergence_and_random_network_controls_v1'
DEFAULT = ROOT/'experiments/convergence_controls/v1'
CONTROLS = ROOT/'experiments/submission_controls/v1'


def make_plan(parent, controls=CONTROLS, quick=False, max_epochs=1000, patience=25,
              min_delta=1e-5, minimum_epochs=25, domain='original'):
    if max_epochs < 1 or patience < 1 or not 1 <= minimum_epochs <= max_epochs or not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError('Invalid stopping rule: require positive caps/patience and finite nonnegative delta')
    if domain not in ('original','varied','all'):
        raise ValueError('Invalid domain selection')
    return dict(protocol=PROTOCOL, parent=str(parent), controls=str(controls), quick=quick,domain=domain,
                architectures=['hidden','input'],cohorts=['continued','scratch'],
                seeds=[42] if quick else [42,43,44], batch_size=32 if quick else 256,
                learning_rate=1e-4, evaluation_steps=32 if quick else 3000,
                maximum_additional_epochs=4 if quick else max_epochs,
                patience=2 if quick else patience, min_delta=min_delta,
                minimum_additional_epochs=1 if quick else minimum_epochs,
                selection='Minimum validation element MSE including the inherited endpoint; test never used',
                scope='Continue selected original/varied BC and mixed DAgger on fixed datasets; '
                      'evaluate untrained random networks with normal observations and unchanged gait engine')


def random_network(seed, directory):
    """Same constructor/default initialization as original BC; no optimizer or learning."""
    from train import BCPolicy
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BCPolicy(63, 16)
    torch.save(model.state_dict(), Path(directory)/'policy.pth')
    save(Path(directory)/'metrics.json', dict(training_updates=0, seed=seed,
         architecture='63 -> 256 ReLU -> 256 ReLU -> 16 native actions',
         semantics='Fixed randomly initialized network; real observations; deterministic state-dependent outputs'))


class Study(OriginalStudy):
    def __init__(self, output, plan, retries=1):
        self.root, self.plan, self.retries = output.resolve(), plan, retries
        self.parent = Parent(Path(plan['parent']))
        self.controls = Path(plan['controls']).resolve()
        if self.parent.plan['quick']:
            raise ValueError('A completed full parent is required; this copy has a separate --quick mode')
        for historical in (self.parent.root, self.controls):
            if self.root.is_relative_to(historical) or historical.is_relative_to(self.root):
                raise ValueError('New output must be separate from both historical trees')
        immutable(self.root/'plan.json',plan)
        self.parent.check_runtime()
        src=sources()
        for folder in ('submission_controls','scratch_comparison','convergence_controls'):
            src.update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/folder).glob('*.py')})
        src['convergence_controls/COPY_PROVENANCE.json']=digest(ROOT/'convergence_controls/COPY_PROVENANCE.json')
        self.identity=dict(protocol=PROTOCOL,plan=digest(self.root/'plan.json'),sources=src,
            parent_identity=digest(self.parent.root/'runtime_identity.json'),
            controls_identity=digest(self.controls/'runtime_identity.json'),
            environment=self.parent.identity['environment'],python=sys.version,numpy=np.__version__,
            torch=torch.__version__,cuda_available=torch.cuda.is_available())
        immutable(self.root/'runtime_identity.json',self.identity)
        self.audited=set();self.checked_runtimes=set()

    def audit_source(self,directory):
        directory=Path(directory).resolve()
        if not directory.is_relative_to(self.controls):
            return super().audit_source(directory)
        if directory in self.audited:return
        runtime=read(self.controls/'runtime_identity.json')
        identity=read(directory/'completed.json')['identity']
        if identity.get('runtime')!=runtime or not verified(directory,identity):
            raise ValueError(f'Invalid submission-control receipt: {directory}')
        if self.controls not in self.checked_runtimes:
            if runtime['plan']!=digest(self.controls/'plan.json'):
                raise ValueError('Historical controls plan changed')
            for name,expected in runtime['sources'].items():
                if digest(ROOT/name)!=expected:raise ValueError(f'Historical source changed: {name}')
            for name,expected in runtime['environment'].items():
                if digest(name)!=expected:raise ValueError(f'Historical environment changed: {name}')
            if runtime['parent_identity']!=digest(self.parent.root/'runtime_identity.json'):
                raise ValueError('Controls and parent lineages differ')
            self.checked_runtimes.add(self.controls)
        for name,expected in identity['inputs'].items():
            if digest(name)!=expected:raise ValueError(f'Historical input changed: {name}')
            if (Path(name).parent/'completed.json').exists():self.audit_source(Path(name).parent)
        OriginalStudy.validate(self,directory,identity['spec'])
        self.audited.add(directory)

    def training_jobs(self):
        original=self.parent.root/'policies/original/hidden/seed_42'
        definitions=[]
        if self.plan['domain'] in ('original','all'):
            definitions += [dict(label='bc_original_convergence',folder=self.controls/'training/bc_original_matched',
                domain='original',architecture='hidden',seed=42,cohort='original',algorithm='bc',
                reference_label='bc_original_matched',reference_root=self.controls/'evaluation',controls=True),
                dict(label='dagger_original_fixed_aggregate_convergence',folder=original/'dagger_mixed/iteration_007/update',
                domain='original',architecture='hidden',seed=42,cohort='original',algorithm='dagger_mixed',
                reference_label='original_hidden_42_dagger_mixed',reference_root=self.parent.root/'evaluation/all_all_correct',controls=False)]
        if self.plan['domain'] in ('varied','all'):
            for cohort in self.plan['cohorts']:
                for architecture in self.plan['architectures']:
                    for seed in self.plan['seeds']:
                        for algorithm in ('bc','dagger_mixed'):
                            owner=self.parent.root/('scratch_v1' if cohort=='scratch' else '')
                            base=owner/f'policies/varied/{architecture}/seed_{seed}/{algorithm}'
                            control=cohort=='scratch' and algorithm=='bc'
                            if control:
                                reference=f'bc_varied_{architecture}_{seed}_200'
                                folder=self.controls/'training'/reference
                                reference_root=self.controls/'evaluation'
                            else:
                                folder=base/('iteration_000/update' if algorithm=='bc' else 'iteration_007/update')
                                reference=f'{"scratch_" if cohort=="scratch" else ""}varied_{architecture}_{seed}_{algorithm}'+('_correct' if architecture=='input' else '')
                                reference_root=owner/'evaluation/all_all_correct'
                            definitions.append(dict(label=f'{algorithm}_varied_{cohort}_{architecture}_{seed}_convergence',
                                folder=folder,domain='varied',architecture=architecture,seed=seed,cohort=cohort,
                                algorithm=algorithm,reference_label=reference,reference_root=reference_root,controls=control))
        result=[]
        for d in definitions:
            folder=d['folder'];self.audit_source(folder)
            source=read(folder/'job.json');metrics=read(folder/'metrics.json')
            if metrics['train_samples']!=170000:raise ValueError('Expected a fixed 170,000-example dataset')
            # Selected policy and its synchronized optimizer: do not discard a better
            # historical checkpoint by starting from a later, worse endpoint.
            model=folder/'policy.pth';optimizer=folder/'optimizer.pth'
            if d['controls']:
                parts=[metrics];domains=1
            else:
                base=folder.parent.parent
                bases=[base]
                if d['domain']=='varied' and d['cohort']=='continued':bases.insert(0,original/d['algorithm'])
                parts=[];domains=len(bases)
                for base in bases:
                    for i in ([0] if d['algorithm']=='bc' else range(8)):
                        update=base/f'iteration_{i:03d}/update'
                        self.audit_source(update);parts.append(read(update/'metrics.json'))
            costs=dict(gradient_updates=sum(m['gradient_updates'] for m in parts),
                sample_exposures=sum(m['sample_exposures'] for m in parts),oracle_labels=170000*domains,
                learner_steps=70000*domains if d['algorithm']=='dagger_mixed' else 0,
                final_fit_best_epoch=metrics['best_epoch'],final_fit_performed_updates=metrics['gradient_updates'],
                final_fit_selected_updates=metrics.get('selected_gradient_updates',
                    metrics['best_epoch']*((metrics['train_samples']+source['batch_size']-1)//source['batch_size'])),
                accounting='Performed historical fitting budgets including discarded epochs; optimizer starts at selected checkpoint')
            if d['domain']=='original' and costs['gradient_updates']!=211050:
                raise ValueError('Original diagnostics must start from the 211,050-update study')
            for path in source['train']+source['validation']:self.audit_source(Path(path).parent)
            if set(source['train'])&set(source['validation']):raise ValueError('Training/validation overlap')
            result.append(dict(task='control_train',label=d['label'],domain=d['domain'],architecture=d['architecture'],
                cohort=d['cohort'],algorithm=d['algorithm'],reference_label=d['reference_label'],
                reference_evaluation_root=str(d['reference_root']),
                model=str(model),optimizer=str(optimizer),train=source['train'],validation=source['validation'],
                seed=d['seed'],batch_size=self.plan['batch_size'],learning_rate=self.plan['learning_rate'],
                quick=self.plan['quick'],early_stop=True,epochs=self.plan['maximum_additional_epochs'],
                patience=self.plan['patience'],min_delta=self.plan['min_delta'],
                minimum_epochs=self.plan['minimum_additional_epochs'],inherited_costs=costs,
                source_receipt=str(folder/'completed.json')))
        for domain in {j['domain'] for j in result}:
            pools=[j['validation'] for j in result if j['domain']==domain]
            if any(p!=pools[0] for p in pools):raise ValueError('Validation pools must match within domain')
        return result

    def validate(self,directory,job):
        if job['task']=='random_network':
            from consolidated.models import load_policy
            from train import BCPolicy
            model=load_policy(directory/'policy.pth','bc','hidden')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(job['seed']);expected=BCPolicy(63,16)
            if any(not torch.equal(v,expected.state_dict()[k]) for k,v in model.state_dict().items()):
                raise ValueError('Random network differs from its recorded initialization')
            if read(directory/'metrics.json')['training_updates']!=0:
                raise ValueError('Engine control must be untrained')
        else:
            super().validate(directory,job)
            if job['task']=='control_train':
                m=read(directory/'metrics.json');h=read(directory/'history.json')
                if m['stop_reason'] not in ('validation_patience','budget_exhausted'):
                    raise ValueError('Invalid convergence stopping reason')
                if m['validation_mse']!=min(r['validation_mse'] for r in h):
                    raise ValueError('Checkpoint was not selected by minimum validation loss')
                if h[0]['epoch']!=0 or m['validation_mse']>h[0]['validation_mse']:
                    raise ValueError('Inherited checkpoint was not included in selection')

    def job(self,directory,job,inputs=(),live=False,check_only=False):
        identity=dict(runtime=self.identity,spec=job,inputs={str(p):digest(p) for p in inputs})
        if verified(directory,identity):
            self.validate(directory,job);print(f'[Resume] {directory.relative_to(self.root)}',flush=True);return directory
        if check_only:raise ValueError(f'Incomplete convergence control: {directory}')
        for attempt in range(self.retries+1 if live else 1):
            begin(directory);save(directory/'job.json',job)
            try:
                if job['task']=='random_network':random_network(job['seed'],directory)
                else:
                    module='consolidated.eval_worker' if live else 'convergence_controls.train'
                    command=[sys.executable,'-u','-m',module,str(directory/'job.json')]
                    if live:
                        with simulator(directory,job['row']['slip']) as manifest:
                            launch(command,directory/'worker.log',server=manifest['server_process'])
                    else:launch(command,directory/'worker.log')
                self.validate(directory,job)
                if identity['inputs']!={str(p):digest(p) for p in inputs}:raise ValueError('Inputs changed during job')
                commit(directory,identity);return directory
            except Exception as error:
                save(directory/'failure.json',dict(error=repr(error),attempt=attempt))
                if not live or attempt==self.retries:raise

    def trained(self,check_only=False):
        result=[]
        for job in self.training_jobs():
            inputs=job['train']+job['validation']+[job['model'],job['optimizer'],job['source_receipt']]
            directory=self.job(self.root/'training'/job['label'],job,inputs,check_only=check_only)
            result.append(controller(job['label'],directory/'policy.pth',job['domain'],job['architecture'],job['seed'],
                algorithm=job['algorithm'],cohort=job['cohort'],reference_label=job['reference_label'],
                reference_evaluation_root=job['reference_evaluation_root']))
        return result

    def engine_models(self,check_only=False):
        result=[]
        for seed in self.plan['seeds']:
            label=f'engine_random_network_{seed}'
            directory=self.job(self.root/'training'/label,dict(task='random_network',architecture='hidden',seed=seed),check_only=check_only)
            result.append(controller(label,directory/'policy.pth',domain='baseline',seed=seed,
                algorithm='untrained_random_network',semantics='Fixed random weights; normal observations into native gait actions'))
        return result

    def evaluation(self,check_only=False,engine_only=False):
        controllers=([] if engine_only else self.trained(check_only=True))+self.engine_models(check_only)
        if self.plan['quick']:
            controllers += [controller(k,domain='baseline',kind=k,seed=None) for k in ('oracle','random')]
            zero=self.controls/'training/zero_native';self.audit_source(zero)
            controllers.append(controller('zero_native',zero/'policy.pth',domain='baseline',seed=None))
        scope='engine' if engine_only else 'all'
        schedule=dict(rows=self.rows(),controllers=controllers,steps=self.plan['evaluation_steps'],scope=scope,
                      parent_schedule_sha256=digest(self.parent.root/'evaluation/all_all_correct/schedule.json'))
        output=self.root/'evaluation'
        immutable(output/f'schedule_{scope}.json',schedule)
        for row in schedule['rows']:
            domain='original' if row['slip'] is None else 'varied'
            expert=self.parent.root/f'data/{domain}/random_bounds_pool.npz'
            for c in controllers:
                if row['slip'] is None and c['domain']=='varied':continue
                job=dict(task='evaluate',row=row,controller=c,
                         supplied_slip=None if row['slip'] is None else float(f'{row["slip"]:.12g}'),
                         steps=schedule['steps'],expert_file=str(expert))
                self.job(output/'runs'/row['id']/c['label'],job,
                         [str(expert)]+([c['model']] if c['model'] else []),live=True,check_only=check_only)
        return output,schedule


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','audit','train','evaluate','engine','report','run'])
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--controls',type=Path,default=CONTROLS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--domain',choices=['original','varied','all'],default='original')
    parser.add_argument('--max-epochs',type=int,default=1000,help='Hard cap of additional epochs per diagnostic')
    parser.add_argument('--patience',type=int,default=25)
    parser.add_argument('--min-delta',type=float,default=1e-5)
    parser.add_argument('--minimum-epochs',type=int,default=25)
    parser.add_argument('--retries',type=int,default=1)
    parser.add_argument('--engine-only',action='store_true',help='Report only random-network engine evaluations')
    args=parser.parse_args()
    if args.retries<0:parser.error('Retries cannot be negative')
    if args.engine_only and args.stage!='report':parser.error('--engine-only is for report; use stage engine to execute')
    parent,controls=args.parent.resolve(),args.controls.resolve()
    base=DEFAULT if args.domain=='original' else DEFAULT.parent/f'{args.domain}_v1'
    output=(args.output or (DEFAULT.parent/f'smoke_{args.domain}_v1' if args.quick else base)).resolve()
    if args.quick and output==DEFAULT.resolve():parser.error('Smoke needs a separate output')
    try:plan=make_plan(parent,controls,args.quick,args.max_epochs,args.patience,args.min_delta,args.minimum_epochs,args.domain)
    except ValueError as e:parser.error(str(e))
    for old in (parent,controls):
        if output.is_relative_to(old) or old.is_relative_to(output):parser.error('Output must be outside historical trees')
    if args.stage=='plan':
        immutable(output/'plan.json',plan)
        print(f'Plan: {output}/plan.json\n{plan}')
        print('Full sizes: original=2 fits/190 new deployments; varied=24 fits/906 deployments; all=26 fits/982 deployments. '
              'Includes 3 untrained networks (114 deployments). --quick uses reduced replicas/data/horizons.')
        return
    study=Study(output,plan,args.retries)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if args.stage=='audit':
        jobs=study.training_jobs();study.rows()
        for domain in ('original','varied'):study.parent.data(domain,'test')
        study.audit_source(controls/'training/zero_native')
        print(f'[Audit] {len(jobs)} fixed-dataset diagnostics verified; no training or simulation');return
    if args.stage in ('train','run'):
        study.trained();study.engine_models()
    if args.stage in ('evaluate','engine','run'):
        if args.stage!='engine':study.trained(check_only=True)
        study.engine_models()
        with simulator_lock():
            require_free_ports()
            subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
            subprocess.run([sys.executable,'-c','from scripts.commons.Script import Script; Script.build_cpp_modules()'],check=True)
            study.parent.check_runtime()
            study.evaluation(engine_only=args.stage=='engine')
    if args.stage in ('report','evaluate','engine','run'):
        from .report import report
        output,schedule=study.evaluation(check_only=True,engine_only=args.stage=='engine' or args.engine_only)
        report(study,output,schedule)


if __name__=='__main__':main()

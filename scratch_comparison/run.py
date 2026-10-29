"""Append scratch training/evaluation to an immutable completed consolidated batch."""
import argparse
import copy
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import torch

from consolidated.run import Experiment, DEFAULT, immutable, sources, validate_artifacts, training_summary
from consolidated.plan import ALGORITHMS, segments
from consolidated.models import initialize
from consolidated.server import require_free_ports
from variable_training.common import ROOT, read, save, digest, verified, begin, commit, simulator_lock

PROTOCOL='consolidated_scratch_comparison_v1'


def configurations(plan):
    yield 'original','hidden',42,'gail_random'
    for arch in plan['architectures']:
        for seed in plan['varied_seeds']:
            for algorithm in [*ALGORITHMS]+(['gail_random'] if arch=='hidden' else []):
                yield 'varied',arch,seed,algorithm


def scratch_initialize(job,directory):
    """Paired random hidden/input actors; no learned source weights for BC/DAgger."""
    if job.get('model'):
        return initialize(job,directory)
    from train import BCPolicy
    from gail_models import GAILActorCritic
    torch.manual_seed(job['seed'])
    base=BCPolicy(63,16) if job['kind']=='bc' else GAILActorCritic(63,16)
    state=base.state_dict()
    if job['architecture']=='input':
        if job['kind']!='bc': raise ValueError('Random GAIL is hidden-context only')
        # Hidden and input start with the same random function of existing observations.
        state['net.0.weight']=torch.cat([state['net.0.weight'],torch.zeros(256,1)],dim=1)
        base=BCPolicy(64,16)
        base.load_state_dict(state)
    torch.save(base.state_dict(),directory/'policy.pth')


class Parent:
    def __init__(self,root):
        self.root=root.resolve()
        self.plan=read(self.root/'plan.json')
        self.identity=read(self.root/'runtime_identity.json')
        if self.identity['protocol']!='consolidated_imitation_v1' or self.identity['plan']!=digest(self.root/'plan.json'):
            raise ValueError('Invalid parent consolidated batch')
        self.audited=set()

    def check_runtime(self):
        # Exact saved dependencies only. Extension code is not added to historical identities.
        for name,expected in self.identity['sources'].items():
            if digest(ROOT/name)!=expected: raise ValueError(f'Parent source changed: {name}')
        for name,expected in self.identity['environment'].items():
            if digest(name)!=expected: raise ValueError(f'Parent environment changed: {name}')

    def audit(self,directory):
        directory=Path(directory).resolve()
        if directory in self.audited: return
        if not directory.is_relative_to(self.root): raise ValueError('Dependency outside parent batch')
        receipt=read(directory/'completed.json');identity=receipt['identity']
        if any(identity.get(k)!=v for k,v in self.identity.items()) or not verified(directory,identity):
            raise ValueError(f'Invalid parent receipt: {directory}')
        for filename,expected in identity['inputs'].items():
            if digest(filename)!=expected: raise ValueError(f'Parent input changed: {filename}')
            dependency=Path(filename).parent
            if (dependency/'completed.json').exists(): self.audit(dependency)
        validate_artifacts(directory,identity['spec'])
        self.audited.add(directory)

    def data(self,domain,group):
        b=self.plan['budgets'];total={'seed':b['dagger_seed'],'extra':b['expert_train']-b['dagger_seed'],
                                    'validation':b['validation'],'test':b['test']}[group]
        offset=['seed','extra','validation','test'].index(group)*100
        result=[]
        for row in segments(self.plan,domain,group,total,4200000+offset):
            directory=self.root/'data'/domain/group/row['id']
            self.audit(directory)
            job=read(directory/'job.json')
            if job!=dict(task='collect',kind='oracle',architecture='hidden',row=row):
                raise ValueError(f'Unexpected shared data: {directory}')
            result.append(str(directory/'data.npz'))
        return result

    def evaluation(self):
        output=self.root/'evaluation/all_all_correct'
        schedule=read(output/'schedule.json')
        if schedule['identity']!=self.identity: raise ValueError('Parent evaluation identity changed')
        expected={(r['id'],c['label']):(r,c) for r in schedule['rows'] for c in schedule['controllers']
                  if r['slip'] is not None or c['domain']!='varied'}
        comparison=read(output/'comparison.json')
        actual={(r['row']['id'],r['controller']['label']) for r in comparison['runs']}
        if len(comparison['runs'])!=len(expected) or actual!=set(expected):
            raise ValueError('Parent evaluation is incomplete or duplicated')
        for index,((row_id,label),(row,c)) in enumerate(expected.items(),1):
            directory=output/'runs'/row_id/label
            self.audit(directory)
            result=read(directory/'result.json')
            if result['row']!=row or result['controller']!=c: raise ValueError('Parent evaluation metadata changed')
            # Cached reporting fields may be reused only for exactly matching raw results.
            cached=next(r for r in comparison['runs'] if r['row']['id']==row_id and r['controller']['label']==label)
            if any(cached.get(k)!=v for k,v in result.items()): raise ValueError('Parent report/raw result mismatch')
            if index%100==0:print(f'[Audit cached evaluation] {index}/{len(expected)}',flush=True)
        return output,schedule,comparison


def make_plan(parent):
    plan=copy.deepcopy(parent.plan)
    plan.update(protocol=PROTOCOL,parent=str(parent.root),parent_plan_sha256=digest(parent.root/'plan.json'),
                parent_identity_sha256=digest(parent.root/'runtime_identity.json'),
                scratch_algorithms=list(ALGORITHMS)+['gail_random'],
                lineage={'supervised':'random initialization; no original-environment pretraining',
                         'gail_bc':'same-domain scratch BC', 'gail_dagger_mixed':'same-domain scratch mixed DAgger',
                         'gail_random':'direct random actor; original hidden seed42 and varied hidden replicas only'},
                accounting='Same incremental domain budgets as continuation; inherited total budgets differ; shared parent datasets reused')
    return plan


class ScratchExperiment(Experiment):
    @classmethod
    def readonly(cls,root,parent):
        self=cls.__new__(cls)
        self.root=Path(root).resolve();self.parent=parent;self.retries=0;self._audited=set()
        self.plan=read(self.root/'plan.json');self.identity=read(self.root/'runtime_identity.json')
        if self.plan!=make_plan(parent) or self.identity['plan']!=digest(self.root/'plan.json'):
            raise ValueError('Scratch plan changed')
        if self.identity['parent_identity_sha256']!=digest(parent.root/'runtime_identity.json'):
            raise ValueError('Parent identity changed')
        for name,expected in self.identity['sources'].items():
            if digest(ROOT/name)!=expected:raise ValueError(f'Scratch source changed: {name}')
        return self

    def __init__(self,root,parent,retries=1):
        self.root=root.resolve();self.parent=parent;self.plan=make_plan(parent);self.retries=retries;self._audited=set()
        immutable(self.root/'plan.json',self.plan)
        src=sources()
        src.update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'scratch_comparison').glob('*.py')})
        self.identity=dict(parent.identity,protocol=PROTOCOL,plan=digest(self.root/'plan.json'),sources=src,
                           libraries=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
                                          device='cuda' if torch.cuda.is_available() else 'cpu'),
                           parent_identity_sha256=digest(parent.root/'runtime_identity.json'))
        immutable(self.root/'runtime_identity.json',self.identity)

    def data(self,domain,group):return self.parent.data(domain,group)

    def audit_ancestry(self,directory):
        if not Path(directory).resolve().is_relative_to(self.root):
            return self.parent.audit(directory)
        return super().audit_ancestry(directory)

    def final(self,*key):
        record=super().final(*key)
        if (record['domain'],record['architecture'],record['seed'],record['algorithm'])!=key:
            raise ValueError('Final record coordinates changed')
        self.data(key[0],'test')
        return record

    def job(self,directory,spec,inputs=(),live=False):
        if spec['task']!='initialize': return super().job(directory,spec,inputs,live)
        identity=dict(self.identity,spec=spec,inputs={str(p):digest(p) for p in inputs})
        directory=Path(directory)
        if verified(directory,identity):
            validate_artifacts(directory,spec);print(f'[Resume] {directory}',flush=True);return directory
        begin(directory);save(directory/'job.json',spec)
        scratch_initialize(spec,directory)
        validate_artifacts(directory,spec)
        if identity['inputs']!={str(p):digest(p) for p in inputs}:raise ValueError('Initializer changed')
        commit(directory,identity)
        return directory

    def train(self,domain,architecture,seed,algorithm):
        key=(domain,architecture,seed,algorithm)
        if key not in set(configurations(self.plan)): raise ValueError(f'Unsupported scratch configuration: {key}')
        path=self.final_path(*key)
        if path.exists():
            print(f'[Resume trained] {key}',flush=True);return self.final(*key)
        b=self.plan['budgets'];settings=self.plan['settings'];base=path.parent
        kind='gail' if algorithm.startswith('gail') else 'bc'
        initializer=None
        if algorithm in ('gail_bc','gail_dagger_mixed'):
            initializer=self.train(domain,architecture,seed,'bc' if algorithm=='gail_bc' else 'dagger_mixed')
        init=dict(task='initialize',architecture=architecture,kind=kind,seed=seed,
                  model=initializer['model'] if initializer else None)
        model=str(self.job(base/'initialization',init,[initializer['model']] if initializer else [])/'policy.pth')
        optimizer=discriminator=None
        seed_data=self.data(domain,'seed');validation=self.data(domain,'validation');tests=self.data(domain,'test')
        expert=seed_data+self.data(domain,'extra') if algorithm!='dagger_mixed' else seed_data
        training=list(expert)
        for iteration in range(1 if kind=='gail' else 0,(0 if algorithm=='bc' else b['iterations'])+1):
            stage=base/f'iteration_{iteration:03d}';rollouts=[]
            if iteration:
                for row in segments(self.plan,domain,f'rollout_{iteration}',b['learner_steps']//b['iterations'],seed*100000+1000+iteration*20):
                    spec=dict(task='collect',kind='gail' if kind=='gail' else 'dagger',architecture=architecture,
                              model=model,row=row,beta=max(0.,1.-.2*iteration))
                    collected=self.job(stage/'segments'/row['id'],spec,[model],live=True)
                    rollouts.append(str(collected/'data.npz'))
                if kind!='gail':training.extend(rollouts)
            if kind=='gail':
                spec=dict(task='gail',architecture=architecture,model=model,optimizer=optimizer,
                          discriminator=discriminator,expert=expert,normalizer=expert,rollouts=rollouts,
                          seed=seed+iteration,iteration=iteration,label=algorithm,
                          ppo_epochs=settings['ppo_epochs'],discriminator_epochs=settings['discriminator_epochs'])
                files=expert+rollouts+[model]+([optimizer,discriminator] if optimizer else [])
            else:
                spec=dict(task='supervised',architecture=architecture,model=model,optimizer=optimizer,
                          train=training.copy(),validation=validation,seed=seed+iteration,
                          epochs=settings['bc_epochs' if algorithm=='bc' else 'dagger_epochs'],
                          learning_rate=settings['learning_rate'],batch_size=settings['batch_size'])
                files=training+validation+[model]+([optimizer] if optimizer else [])
            update=self.job(stage/'update',spec,files)
            model=str(update/'policy.pth');optimizer=str(update/'optimizer.pth')
            discriminator=str(update/'discriminator.pth') if kind=='gail' else None
        record=dict(model=model,sha256=digest(model),plan=digest(self.root/'plan.json'),domain=domain,
                    architecture=architecture,seed=seed,algorithm=algorithm,kind=kind,initializer=initializer,
                    initialization_cohort='scratch',expert_training_samples=b['expert_train'],
                    active_oracle_labels=0 if kind=='gail' else b['expert_train'],
                    learner_steps=0 if algorithm=='bc' else b['learner_steps'],validation_samples=b['validation'],
                    test_samples=b['test'],updates=[dict(path=str(p),metrics=read(p)) for p in sorted(base.glob('iteration_*/update/metrics.json'))],
                    offline_test=self.offline_score(model,kind,architecture,tests))
        save(path,record);return record

    def scratch_controllers(self):
        result=[]
        for domain,arch,seed,algorithm in configurations(self.plan):
            info=self.final(domain,arch,seed,algorithm)
            result.append(dict(label=f'scratch_{domain}_{arch}_{seed}_{algorithm}'+('_correct' if arch=='input' else ''),
                               kind=info['kind'],architecture=arch,model=info['model'],sha256=info['sha256'],
                               domain=domain,seed=seed,algorithm=algorithm,mode='correct' if arch=='input' else 'none',
                               initialization_cohort='scratch'))
        return result

    def evaluate_append(self,report_only=False):
        from .report import report
        parent_output,schedule,cached=self.parent.evaluation()
        controllers=self.scratch_controllers()
        output=self.root/'evaluation/all_all_correct'
        combined=dict(identity=self.identity,rows=schedule['rows'],steps=schedule['steps'],
                      controllers=schedule['controllers']+controllers,
                      reused_parent=str(parent_output),parent_schedule_sha256=digest(parent_output/'schedule.json'),
                      parent_comparison_sha256=digest(parent_output/'comparison.json'))
        immutable(output/'schedule.json',combined)
        print(f'[Reuse] {len(cached["runs"])} completed parent deployments; no simulator replay',flush=True)
        for row in schedule['rows']:
            domain='original' if row['slip'] is None else 'varied'
            expert_file=self.parent.root/'data'/domain/'random_bounds_pool.npz'
            for c in controllers:
                if row['slip'] is None and c['domain']=='varied':continue
                spec=dict(task='evaluate',row=row,controller=c,
                          supplied_slip=float(f'{row["slip"]:.12g}') if row['slip'] is not None else None,
                          steps=schedule['steps'],expert_file=str(expert_file))
                files=[str(expert_file),c['model']];directory=output/'runs'/row['id']/c['label']
                if report_only:
                    identity=dict(self.identity,spec=spec,inputs={p:digest(p) for p in files})
                    if not verified(directory,identity):raise ValueError(f'Incomplete new evaluation: {directory}')
                    validate_artifacts(directory,spec)
                else:self.job(directory,spec,files,live=True)
        report(self,output,combined,cached,parent_output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pipeline',choices=['training','evaluation'])
    p.add_argument('stage',choices=['plan','audit','run','report'],nargs='?',default='plan')
    p.add_argument('--output',type=Path,default=DEFAULT,help='Existing completed consolidated batch')
    p.add_argument('--extension-output',type=Path,help='Separate scratch output; default <output>/scratch_v1')
    p.add_argument('--algorithm',choices=['all',*ALGORITHMS,'gail_random'],default='all')
    p.add_argument('--quick',action='store_true',help='Require a quick parent; inherit its small budgets')
    p.add_argument('--retries',type=int,default=1)
    args=p.parse_args()
    if args.retries<0:p.error('Retries must be nonnegative')
    parent=Parent(args.output);parent.check_runtime()
    if bool(parent.plan['quick'])!=args.quick:p.error('--quick must match the parent batch')
    root=(args.extension_output or parent.root/'scratch_v1').resolve()
    if root==parent.root or parent.root.is_relative_to(root):p.error('Extension must not overwrite/contain parent root')
    plan=make_plan(parent)
    n=len(list(configurations(plan)))
    if args.stage in ('plan','audit'):
        if args.stage=='audit':
            for domain in ('original','varied'):
                for group in ('seed','extra','validation','test'):parent.data(domain,group)
            _,schedule,old=parent.evaluation()
        else:
            schedule=read(parent.root/'evaluation/all_all_correct/schedule.json');old={'runs':[]}
        new=sum(1 for row in schedule['rows'] for d,*_ in configurations(plan) if row['slip'] is not None or d=='original')
        print(f'{n} NEW models; {new} NEW deployments. Parent outputs preserved/reused.\nOutput: {root}\nBudgets: {plan["budgets"]}')
        if args.stage=='audit':print(f'Parent data/receipts verified; {len(old["runs"])} cached deployments reusable.')
        return
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if args.stage=='report':
        exp=ScratchExperiment(root,parent,args.retries)
        if args.pipeline=='training':training_summary(root,plan)
        else:exp.evaluate_append(True)
        return
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
        subprocess.run([sys.executable,'-c','from scripts.commons.Script import Script; Script.build_cpp_modules()'],check=True)
        parent.check_runtime()
        exp=ScratchExperiment(root,parent,args.retries)
        if args.pipeline=='evaluation':exp.evaluate_append()
        else:
            for key in configurations(plan):
                if args.algorithm=='all' or key[-1]==args.algorithm:exp.train(*key)
            training_summary(root,plan)


if __name__=='__main__':main()

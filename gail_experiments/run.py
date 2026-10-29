"""Three GAIL initializers, larger budgets, fresh servers and paired checkpoint tests."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
import torch

from variable_training.common import ROOT, read, save, digest, begin, commit, verified, simulator_lock
from consolidated.run import immutable, sources, validate_artifacts, Experiment
from consolidated.server import simulator, launch, require_free_ports
from consolidated.plan import segments
from scratch_comparison.run import Parent

DEFAULT=ROOT/'experiments/gail_improvement/original_v1'


def make_plan(args):
    if args.rollout_steps<=0 or args.steps<=0 or args.steps%args.rollout_steps or args.rollout_steps%10:
        raise ValueError('Budget must be positive and divisible by rollout steps; rollout steps divisible by ten')
    if args.rollout_steps<20 or args.eval_every<1:raise ValueError('Invalid rollout or evaluation cadence')
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or any(s<0 or s>=2**31 for s in args.seeds):
        raise ValueError('Training seeds must be distinct nonnegative integers below 2**31')
    if len(set(args.recipes))!=len(args.recipes):raise ValueError('Duplicate recipes')
    if args.quick:
        steps,rollout,eval_every,eval_steps,seeds=200,100,1,100,[42]
    else:
        steps,rollout,eval_every,eval_steps,seeds=args.steps,args.rollout_steps,args.eval_every,3000,args.seeds
    return dict(protocol='gail_improvement_v1',parent=str(args.parent.resolve()),domain=args.domain,
        recipes=args.recipes,seeds=seeds,initializers=['random','bc','dagger_mixed'],
        active_steps=steps,rollout_steps=rollout,iterations=steps//rollout,eval_every=eval_every,
        evaluation_steps=eval_steps,quick=args.quick,slips=[.025,.1,.2,1.,10.],
        validation_seeds=[901] if args.quick else [901,902,903],
        confirmation_seeds=[1901] if args.quick else [1901,1902,1903,1904,1905],
        exploration_std=.2,
        settings=dict(actor_lr=3e-5,min_actor_lr=1e-6,max_actor_lr=1e-4,critic_lr=3e-4,
            disc_lr=1e-4,batch_size=1024,disc_epochs=2,ppo_epochs=4,critic_epochs=4,
            r1_coefficient=1.,entropy_coefficient=.001,target_kl=.01,max_kl=.02),
        selection='highest validation progress among checkpoints with total falls <= initializer falls + 1; ties prefer lower iteration',
        scope='exploratory; no task reward; hidden-context same-domain expert data')


class Study:
    def __init__(self,root,plan):
        self.root=root;self.plan=plan;self.parent=Parent(Path(plan['parent']))
        self.parent.check_runtime()
        src=sources()
        for folder in ['scratch_comparison','gail_experiments']:
            src.update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/folder).glob('*.py') if not p.name.startswith('test_')})
        self.identity=dict(protocol=plan['protocol'],plan=digest(root/'plan.json'),sources=src,
                           parent_identity=digest(self.parent.root/'runtime_identity.json'),
                           environment=self.parent.identity['environment'],
                           libraries=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                                          device='cuda' if torch.cuda.is_available() else 'cpu'))
        immutable(root/'runtime_identity.json',self.identity)
        self.expert=self.parent.data(plan['domain'],'seed')+self.parent.data(plan['domain'],'extra')
        self.validation=self.parent.data(plan['domain'],'validation')
        self.tests=self.parent.data(plan['domain'],'test')

    def job(self,directory,spec,inputs=(),live=False,read_only=False):
        identity=dict(self.identity,spec=spec,inputs={str(p):digest(p) for p in inputs})
        if verified(directory,identity):
            validate_artifacts(directory,spec)
            return directory
        if read_only:raise ValueError(f'Incomplete job: {directory}')
        begin(directory);save(directory/'job.json',spec)
        module=('consolidated.eval_worker' if spec['task']=='evaluate' else
                'consolidated.worker' if spec['task']=='collect' else 'gail_experiments.worker')
        try:
            command=[sys.executable,'-u','-m',module,str(directory/'job.json')]
            if live:
                with simulator(directory,spec['row']['slip']) as manifest:
                    launch(command,directory/'worker.log',server=manifest['server_process'])
            else:launch(command,directory/'worker.log')
            validate_artifacts(directory,spec)
            if identity['inputs']!={str(p):digest(p) for p in inputs}:raise ValueError('Input changed during job')
            commit(directory,identity)
        except BaseException as exc:
            save(directory/'failure.json',dict(error=repr(exc)));raise
        return directory

    def initializer(self,name,seed):
        if name=='random':return None
        parent_seed=42 if self.plan['domain']=='original' else seed
        path=self.parent.root/'policies'/self.plan['domain']/'hidden'/f'seed_{parent_seed}'/name/'final.json'
        record=read(path)
        if digest(record['model'])!=record['sha256']:raise ValueError(f'Initializer changed: {path}')
        self.parent.audit(Path(record['model']).parent)
        return record['model']

    def checkpoint(self,recipe,name,seed,iteration):
        return self.root/'policies'/recipe/name/f'seed_{seed}'/f'iteration_{iteration:03d}'/'update'/'policy.pth'

    def controller(self,recipe,name,seed,iteration):
        model=self.checkpoint(recipe,name,seed,iteration)
        # Prove completed checkpoint dependencies still have their saved hashes.
        receipt=read(model.parent/'completed.json')
        if not verified(model.parent,receipt['identity']) or any(receipt['identity'].get(k)!=v for k,v in self.identity.items()):
            raise ValueError('Checkpoint receipt changed')
        for filename,h in receipt['identity']['inputs'].items():
            if digest(filename)!=h:raise ValueError(f'Checkpoint input changed: {filename}')
        return dict(label=f'{recipe}_{name}_{seed}_iter_{iteration:03d}',kind='gail',architecture='hidden',
                    model=str(model),sha256=digest(model),seed=seed,recipe=recipe,initializer=name,iteration=iteration)

    def rows(self,suite):
        seeds=self.plan['validation_seeds' if suite=='validation' else 'confirmation_seeds']
        slips=[None] if self.plan['domain']=='original' else self.plan['slips']
        return [dict(id=f'{suite}_{i}_{seed}',suite=suite,slip=slip,seed=seed)
                for i,slip in enumerate(slips) for seed in seeds]

    def evaluate_controller(self,controller,suite,read_only=False):
        results=[]
        for row in self.rows(suite):
            directory=self.root/'evaluation'/suite/row['id']/controller['label']
            spec=dict(task='evaluate',row=row,controller=controller,supplied_slip=row['slip'],
                      steps=self.plan['evaluation_steps'],expert_file=self.expert[0])
            inputs=([controller['model']] if controller.get('model') else [])+[self.expert[0]]
            self.job(directory,spec,inputs,live=True,read_only=read_only)
            results.append(read(directory/'result.json'))
        return results

    def assess(self,recipe,name,seed,iteration,read_only=False):
        c=self.controller(recipe,name,seed,iteration)
        results=self.evaluate_controller(c,'validation',read_only)
        return dict(iteration=iteration,progress=float(np.mean([r['return_target_progress_m'] for r in results])),
                    falls=sum(r['falls'] for r in results),
                    validation_action_mse=Experiment.offline_score(c['model'],'gail','hidden',self.validation)['element_mse'])

    def train(self):
        for recipe in self.plan['recipes']:
            for name in self.plan['initializers']:
                for seed in self.plan['seeds']:
                    source=self.initializer(name,seed)
                    model=self.checkpoint(recipe,name,seed,0)
                    self.job(model.parent,dict(task='initialize',kind='gail',architecture='hidden',seed=seed,
                        model=source,exploration_std=self.plan['exploration_std']),[source] if source else [])
                    # Baseline is evaluated before any PPO update, not inferred from training rollouts.
                    self.assess(recipe,name,seed,0)
                    for iteration in range(1,self.plan['iterations']+1):
                        update=self.checkpoint(recipe,name,seed,iteration).parent
                        rollouts=[]
                        for row in segments(self.plan,self.plan['domain'],f'rollout_{iteration}',self.plan['rollout_steps'],
                                            seed*100000+1000+iteration*20):
                            d=update.parent/'segments'/row['id']
                            self.job(d,dict(task='collect',kind='gail',architecture='hidden',model=str(model),row=row),[model],live=True)
                            rollouts.append(str(d/'data.npz'))
                        optimizer=str(model.parent/'optimizer.pth') if iteration>1 else None
                        discriminator=str(model.parent/'discriminator.pth') if iteration>1 else None
                        spec=dict(task='gail',kind='gail',architecture='hidden',recipe=recipe,model=str(model),
                            optimizer=optimizer,discriminator=discriminator,iteration=iteration,seed=seed+iteration,
                            expert=self.expert,normalizer=self.expert,rollouts=rollouts,label=f'{recipe}_{name}_{seed}',
                            settings=self.plan['settings'],ppo_epochs=3,discriminator_epochs=1)
                        self.job(update,spec,self.expert+rollouts+[str(model)]+([optimizer,discriminator] if optimizer else []))
                        model=update/'policy.pth'
                        if iteration%self.plan['eval_every']==0 or iteration==self.plan['iterations']:
                            self.assess(recipe,name,seed,iteration)

    def evaluation(self,read_only=False):
        selected=[]
        for recipe in self.plan['recipes']:
            for name in self.plan['initializers']:
                for seed in self.plan['seeds']:
                    curve=[self.assess(recipe,name,seed,i,read_only) for i in checkpoints(self.plan)]
                    best=select_checkpoint(curve)
                    selected.append(dict(recipe=recipe,initializer=name,seed=seed,selected=best['iteration'],curve=curve))
                    for i in sorted(set([0,best['iteration'],self.plan['iterations']])):
                        self.evaluate_controller(self.controller(recipe,name,seed,i),'confirmation',read_only)
        # These references are independently collected under the new protocol.
        for kind in ['oracle','random']:
            c=dict(label=kind,kind=kind,architecture='hidden',model=None,seed=None)
            self.evaluate_controller(c,'confirmation',read_only)
        save(self.root/'selection.json',dict(rule=self.plan['selection'],selected=selected))
        from .report import report
        report(self)


def checkpoints(plan):
    return sorted(set([0,plan['iterations']]+list(range(plan['eval_every'],plan['iterations']+1,plan['eval_every']))))


def select_checkpoint(curve):
    if not curve or curve[0]['iteration']!=0:raise ValueError('Initializer baseline required')
    allowed=[r for r in curve if r['falls']<=curve[0]['falls']+1]
    return max(allowed,key=lambda r:(r['progress'],-r['iteration']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','audit','train','evaluate','run','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--parent',type=Path,default=ROOT/'experiments/consolidated/v1')
    parser.add_argument('--domain',choices=['original','varied'],default='original')
    parser.add_argument('--recipes',choices=['adaptive','legacy'],nargs='+',default=['adaptive'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--steps',type=int,default=350000)
    parser.add_argument('--rollout-steps',type=int,default=10000)
    parser.add_argument('--eval-every',type=int,default=7)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=args.output.resolve();plan=make_plan(args)
    # Prevent an accidental quick run from claiming the full default output.
    if args.quick and root==DEFAULT:root=ROOT/'experiments/gail_improvement/smoke_v1'
    if root==args.parent.resolve() or args.parent.resolve().is_relative_to(root):
        raise ValueError('Output cannot replace or contain the parent experiment')
    immutable(root/'plan.json',plan)
    if args.stage=='plan':
        print(f'{root / "plan.json"}\n{len(plan["seeds"])*3*len(plan["recipes"])} actors; '
              f'{plan["active_steps"]} rollout steps each; checkpoints {checkpoints(plan)}');return
    study=Study(root,plan)
    if args.stage=='audit':
        for name in plan['initializers']:
            for seed in plan['seeds']:study.initializer(name,seed)
        print('[Audit] Expert/validation/test receipts and source initializers verified. No simulator launched.');return
    if args.stage=='report':study.evaluation(read_only=True);return
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
        if args.stage in ['train','run']:study.train()
        if args.stage in ['evaluate','run']:study.evaluation()


if __name__=='__main__':main()

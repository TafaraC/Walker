"""Offline contract tests; no live simulator or research data writes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .plan import make_plan, segments, evaluation_rows
from .models import initialize, load_policy, predict
from .worker import supervised, gail, load_shards
from .run import immutable, validate_artifacts
from variable_training.common import save, commit, verified


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def shard(self, name, count=20, slip=.2):
        path=self.root/name
        rng=np.random.default_rng(4 if name=='train.npz' else 5)
        np.savez_compressed(path,obs=rng.normal(size=(count,63)).astype('f'),
                            actions=rng.normal(scale=.01,size=(count,16)).astype('f'),slip=slip)
        return str(path)

    def test_exact_balanced_budgets_and_schedule(self):
        p=make_plan()
        b=p['budgets']
        self.assertEqual(b['expert_train'],b['dagger_seed']+b['learner_steps'])
        self.assertEqual(b['learner_steps'],70000)
        for domain in ('original','varied'):
            rows=segments(p,domain,'expert',170000,42)
            self.assertEqual(sum(r['steps'] for r in rows),170000)
            self.assertEqual(sum(r['steps'] for r in rows if r['mode']=='short'),85000)
            for i in range(7):
                self.assertEqual(sum(r['steps'] for r in segments(p,domain,str(i),10000,42)),10000)
        from context_training.evaluate import rounds
        self.assertEqual(evaluation_rows('varied'),rounds(42,'all'))
        self.assertEqual(len(evaluation_rows('all')),38)
        self.assertTrue(all(r['slip'] is None for r in evaluation_rows('original')))

    def test_context_initialization_and_gail_lineage(self):
        original=self.root/'original';original.mkdir()
        initialize(dict(seed=42,architecture='hidden',kind='bc',model=None),original)
        conditional=self.root/'input';conditional.mkdir()
        initialize(dict(seed=42,architecture='input',kind='bc',model=str(original/'policy.pth')),conditional)
        x=np.zeros((5,63),dtype='f')
        hidden=load_policy(original/'policy.pth','bc','hidden')
        model=load_policy(conditional/'policy.pth','bc','input')
        np.testing.assert_allclose(predict(hidden,'hidden',x,None),predict(model,'input',x,10),atol=1e-7)
        actor=self.root/'actor';actor.mkdir()
        initialize(dict(seed=42,architecture='input',kind='gail',model=str(conditional/'policy.pth')),actor)
        g=load_policy(actor/'policy.pth','gail','input')
        np.testing.assert_array_equal(predict(g,'input',x,.025),predict(model,'input',x,.025))
        self.assertEqual(g.critic[0].in_features,64)

    def test_supervised_context_train_then_segmented_gail(self):
        train,val=self.shard('train.npz'),self.shard('validation.npz')
        initial=self.root/'initial';initial.mkdir()
        initialize(dict(seed=42,architecture='hidden',kind='bc',model=None),initial)
        conditioned=self.root/'conditioned';conditioned.mkdir()
        initialize(dict(seed=42,architecture='input',kind='bc',model=str(initial/'policy.pth')),conditioned)
        update=self.root/'update';update.mkdir()
        spec=dict(task='supervised',architecture='input',model=str(conditioned/'policy.pth'),
                  train=[train],validation=[val],seed=42,epochs=1,batch_size=8,learning_rate=1e-4)
        with patch('torch.cuda.is_available',return_value=False):
            supervised(spec,update)
        validate_artifacts(update,spec)
        x,y=load_shards([train],'input')
        self.assertEqual(x.shape,(20,64))
        actor=self.root/'gail_initial';actor.mkdir()
        initialize(dict(seed=42,architecture='input',kind='gail',model=str(update/'policy.pth')),actor)
        policy=load_policy(actor/'policy.pth','gail','input')
        paths=[]
        for i in range(2):
            obs=x[i*10:(i+1)*10]
            with torch.no_grad():
                dist,value=policy(torch.from_numpy(obs));actions=dist.sample()
                logp=dist.log_prob(actions).sum(-1).numpy()
            path=self.root/f'rollout{i}.npz'
            np.savez_compressed(path,obs=obs,actions=actions.numpy(),logprobs=logp,
                values=value.numpy().reshape(-1),dones=np.zeros(10,bool),bootstrap_value=.1)
            paths.append(str(path))
        target=self.root/'gail_update';target.mkdir()
        job=dict(task='gail',architecture='input',model=str(actor/'policy.pth'),expert=[train],
                 normalizer=[train],rollouts=paths,seed=42,iteration=1,label='test',
                 ppo_epochs=1,discriminator_epochs=1)
        with patch('torch.cuda.is_available',return_value=False):
            gail(job,target)
        validate_artifacts(target,job)
        with np.load(target/'rollouts.npz') as data:
            np.testing.assert_array_equal(data['segment_offsets'],[0,10,20])
            self.assertEqual(len(data['segment_bootstraps']),2)

    def test_immutable_receipts_and_context_shapes(self):
        directory=self.root/'job';directory.mkdir()
        save(directory/'result.json',{'x':1})
        identity={'source':'a'}
        commit(directory,identity)
        self.assertTrue(verified(directory,identity))
        with self.assertRaises(ValueError):verified(directory,{'source':'b'})
        save(directory/'result.json',{'x':2})
        with self.assertRaises(ValueError):verified(directory,identity)
        path=self.root/'plan.json';immutable(path,{'a':1})
        with self.assertRaises(ValueError):immutable(path,{'a':2})
        p=self.shard('train.npz')
        self.assertEqual(load_shards([p],'hidden')[0].shape[1],63)
        self.assertEqual(load_shards([p],'input')[0].shape[1],64)

    def test_final_dependency_audit_reaches_earlier_artifacts(self):
        from .run import Experiment
        from variable_training.common import digest
        exp=Experiment.__new__(Experiment);exp.identity={'protocol':'test'};exp._audited=set()
        parent=self.root/'parent';parent.mkdir()
        save(parent/'data.json',{'sample':1})
        commit(parent,dict(exp.identity,inputs={}))
        child=self.root/'child';child.mkdir()
        save(child/'model.json',{'weight':2})
        commit(child,dict(exp.identity,inputs={str(parent/'data.json'):digest(parent/'data.json')}))
        exp.audit_ancestry(child)
        self.assertIn(parent,exp._audited)
        exp._audited.clear()
        save(parent/'data.json',{'sample':3})
        with self.assertRaises(ValueError):exp.audit_ancestry(child)

    def test_original_scene_copy_and_scalar_rewrite(self):
        from . import server
        system=self.root/'system';system.mkdir()
        scene=system/'soccer.rsg';scene.write_text('(setContactSlip 0.1 0.1)\n')
        (system/'naosoccersim.rb').write_text("scene.importScene('rsg/agent/nao/soccer.rsg')\nrandomServer.seed(0)\n")
        (system/'rcssserver3d.rb').write_text('run "naosoccersim.rb"\n')
        with patch.object(server,'SYSTEM',system),patch.object(server,'SCENE',scene):
            a=server.prepare(self.root/'legacy',None,None)
            b=server.prepare(self.root/'scalar',5.0133423123456,None)
        self.assertEqual((self.root/'legacy/soccer.rsg').read_bytes(),scene.read_bytes())
        self.assertEqual(a['scene_setter_method'],'legacy_two_argument')
        self.assertEqual(b['scene_setter_method'],'single_argument_v1')
        self.assertTrue(b['matches_expected'])

    def test_training_dependency_graph_and_dagger_budget(self):
        from .run import Experiment
        from variable_training.common import read
        exp=Experiment.__new__(Experiment)
        exp.root=self.root;exp.plan=make_plan(True)
        save(self.root/'plan.json',exp.plan)
        calls={}
        def job(directory,spec,inputs=(),live=False):
            directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
            calls[str(directory)]=spec
            for name in ('policy.pth','optimizer.pth','discriminator.pth','data.npz'):
                (directory/name).write_bytes(b'synthetic graph placeholder')
            return directory
        exp.job=job
        exp.final=lambda domain,arch,seed,alg:read(exp.final_path(domain,arch,seed,alg))
        exp.offline_score=lambda *args:dict(element_mse=0,squared_l2_per_state=0,samples=20)
        record=exp.train('varied','input',42,'gail_dagger_mixed')
        self.assertEqual(record['initializer']['algorithm'],'dagger_mixed')
        self.assertEqual(record['initializer']['architecture'],'input')
        self.assertEqual(record['initializer']['initializer']['domain'],'original')
        supervised_jobs=[s for s in calls.values() if s['task']=='supervised' and s['architecture']=='input']
        self.assertEqual(len(supervised_jobs),2)
        # Ten seed shards plus ten DAgger rollout shards: no BC-only extra oracle rows.
        self.assertEqual(len(supervised_jobs[0]['train']),10)
        self.assertEqual(len(supervised_jobs[1]['train']),20)
        gail_jobs=[s for s in calls.values() if s['task']=='gail']
        self.assertEqual(len(gail_jobs),1)
        self.assertEqual(len(gail_jobs[0]['expert']),20)
        rollout=[s for s in calls.values() if s['task']=='collect' and s['kind']=='gail']
        self.assertEqual(sum(s['row']['steps'] for s in rollout),exp.plan['budgets']['learner_steps'])

    def test_report_normalization_and_paired_context(self):
        from types import SimpleNamespace
        from .report import report
        from variable_training.common import read
        rows=[dict(id='nominal',suite='nominal',slip=.1,seed=101),
              dict(id='grouped',suite='grouped',slip=1.,seed=102)]
        controllers=[dict(label='oracle',kind='oracle',architecture='hidden',model=None,seed=None,domain='baseline'),
                     dict(label='random',kind='random',architecture='hidden',model=None,seed=None,domain='baseline'),
                     dict(label='varied_hidden_42_bc',kind='bc',architecture='hidden',model='fake',seed=42,domain='varied',mode='none',algorithm='bc'),
                     dict(label='varied_input_42_bc_correct',kind='bc',architecture='input',model='fake',seed=42,domain='varied',mode='correct',algorithm='bc')]
        for row in rows:
            for i,c in enumerate(controllers):
                directory=self.root/'runs'/row['id']/c['label'];directory.mkdir(parents=True)
                progress=10. if row['id']=='grouped' else [10.,0.,10.,11.][i]
                result=dict(row=row,controller=c,return_target_progress_m=progress,falls=0,
                    falls_per_active_minute=0,mean_forward_velocity_mps=1,
                    mean_target_direction_error=.2,on_policy_latent_action_mse=.01)
                save(directory/'result.json',result)
                np.savez_compressed(directory/'trace.npz',joint_positions_deg=np.zeros((10,22)),
                    episode_id=np.zeros(10),torso_roll_deg=np.zeros(10),torso_pitch_deg=np.zeros(10))
        exp=SimpleNamespace(root=self.root,plan=make_plan(),offline_score=lambda *args:dict(element_mse=.02))
        report(exp,self.root,dict(rows=rows,controllers=controllers))
        output=read(self.root/'comparison.json')
        self.assertAlmostEqual(output['runs'][3]['normalized_return'],1.1)
        self.assertTrue(all(r['normalized_return'] is None for r in output['runs'][4:]))
        self.assertAlmostEqual(output['paired_differences'][0]['progress_delta_m'],1.)
        self.assertIsInstance(output['aggregates'][0]['normalized_return_ci95'],dict)

    def test_conditioned_rollout_labels_logprobs_and_fall_boundaries(self):
        from types import SimpleNamespace
        from .worker import collect
        class Agent:
            def __init__(self,**kwargs):
                self.world=SimpleNamespace(step=0,robot=SimpleNamespace(
                    loc_head_position=np.zeros(3),get_command=lambda:b''))
                self.scom=SimpleNamespace(commit_and_send=lambda _:None,receive=self.receive)
            def receive(self): self.world.step+=1
            def terminate(self): pass
        class Walk:
            def __init__(self,agent): self.agent=agent
            def observe(self,*args): return np.zeros(63,dtype='f')
            def get_expert_action(self,obs):
                assert obs.shape==(63,)
                return np.zeros(16,dtype='f')
            def execute_action(self,action): assert action.shape==(16,)
        class Recovery:
            def __init__(self): self.calls=0;self.fall_count=0
            def check_and_recover(self,agent):
                self.calls+=1
                if self.calls==3:
                    self.fall_count+=1
                    return True
                return False
        modules={'agent.Base_Agent':SimpleNamespace(Base_Agent=Agent),
                 'behaviors.custom.Walk.Walk':SimpleNamespace(Walk=Walk),
                 'bc_sim':SimpleNamespace(FallRecovery=Recovery),
                 'collect_seed':SimpleNamespace(reset_server_state=lambda:True)}
        base=self.root/'base';base.mkdir()
        initialize(dict(seed=42,architecture='hidden',kind='bc',model=None),base)
        condition=self.root/'condition';condition.mkdir()
        initialize(dict(seed=42,architecture='input',kind='bc',model=str(base/'policy.pth')),condition)
        actor=self.root/'actor';actor.mkdir()
        initialize(dict(seed=42,architecture='input',kind='gail',model=str(condition/'policy.pth')),actor)
        for kind,model_dir in [('dagger',condition),('gail',actor)]:
            directory=self.root/kind;directory.mkdir()
            spec=dict(task='collect',kind=kind,architecture='input',model=str(model_dir/'policy.pth'),beta=0.,
                row=dict(mode='long',steps=4,seed=42,slip=5.0133423123456))
            with patch.dict('sys.modules',modules): collect(spec,directory)
            validate_artifacts(directory,spec)
            with np.load(directory/'data.npz') as data:
                self.assertEqual(data['obs'].shape,(4,64))
                self.assertTrue(data['dones'][1])
                if kind=='dagger':
                    np.testing.assert_array_equal(data['actions'],np.zeros((4,16)))
                else:
                    policy=load_policy(model_dir/'policy.pth','gail','input')
                    with torch.no_grad():
                        dist,_=policy(torch.from_numpy(data['obs']))
                        logp=dist.log_prob(torch.from_numpy(data['actions'])).sum(-1).numpy()
                    np.testing.assert_allclose(logp,data['logprobs'],rtol=1e-5,atol=1e-5)


if __name__=='__main__': unittest.main()

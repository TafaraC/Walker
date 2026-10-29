import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from consolidated.plan import make_plan as base_plan, evaluation_rows
from consolidated.models import load_policy, predict
from variable_training.common import save, read, digest, commit
from .run import configurations, scratch_initialize, ScratchExperiment, Parent
from .report import aggregate, report as evaluation_report
from .probe import execute
from representation_probing.expert import load_expert


class ScratchTests(unittest.TestCase):
    def test_counts_and_random_conditions(self):
        plan=base_plan();keys=list(configurations(plan))
        self.assertEqual(len(keys),28)
        random=[k for k in keys if k[-1]=='gail_random']
        self.assertEqual(len(random),4)
        self.assertEqual({k[1] for k in random},{'hidden'})
        self.assertIn(('original','hidden',42,'gail_random'),random)
        count=sum(1 for r in evaluation_rows('all') for d,*_ in keys if r['slip'] is not None or d=='original')
        self.assertEqual(count,929)
        self.assertEqual(len(keys)*8,224)
        quick=base_plan(True)
        self.assertEqual(len(list(configurations(quick))),10)

    def test_initialization_no_learned_parent_and_zero_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for arch in ('hidden','input'):
                d=root/arch;d.mkdir()
                scratch_initialize(dict(seed=42,kind='bc',architecture=arch,model=None),d)
            hidden=load_policy(root/'hidden/policy.pth','bc','hidden')
            contextual=load_policy(root/'input/policy.pth','bc','input')
            obs=np.random.default_rng(5).normal(size=(12,63)).astype(np.float32)
            np.testing.assert_allclose(predict(hidden,'hidden',obs,.1),predict(contextual,'input',obs,10.),atol=1e-6)
            self.assertTrue(torch.equal(contextual.net[0].weight[:,-1],torch.zeros(256)))
            random=root/'gail';random.mkdir()
            scratch_initialize(dict(seed=42,kind='gail',architecture='hidden',model=None),random)
            actor=load_policy(random/'policy.pth','gail','hidden')
            self.assertTrue(np.isfinite(predict(actor,'hidden',obs,.1)).all())
            with self.assertRaises(ValueError):
                scratch_initialize(dict(seed=42,kind='gail',architecture='input',model=None),random)

    def test_initialization_resume_does_not_initialize_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=ScratchExperiment.__new__(ScratchExperiment);exp.identity={'protocol':'test'}
            directory=Path(tmp)/'init'
            spec=dict(task='initialize',architecture='hidden',kind='bc',seed=42,model=None)
            exp.job(directory,spec)
            before=digest(directory/'policy.pth')
            with patch('scratch_comparison.run.scratch_initialize',side_effect=AssertionError('retrained')):
                exp.job(directory,spec)
            self.assertEqual(before,digest(directory/'policy.pth'))
            (directory/'policy.pth').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):exp.job(directory,spec)

    def test_train_lineage_and_budgets_without_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=ScratchExperiment.__new__(ScratchExperiment)
            exp.root=Path(tmp);exp.plan=base_plan(True)
            save(exp.root/'plan.json',exp.plan)
            calls=[]
            exp.data=lambda domain,group:[f'shared/{domain}/{group}.npz']
            exp.offline_score=lambda *args:dict(element_mse=0,samples=20)
            exp.final=lambda *key:read(exp.final_path(*key))
            def job(directory,spec,inputs=(),live=False):
                directory.mkdir(parents=True,exist_ok=True)
                calls.append(copy.deepcopy(spec));save(directory/'job.json',spec)
                (directory/'policy.pth').write_bytes(b'test-policy')
                return directory
            exp.job=job
            random=exp.train('original','hidden',42,'gail_random')
            self.assertIsNone(random['initializer'])
            self.assertIsNone(calls[0]['model'])
            gail=exp.train('varied','input',42,'gail_bc')
            self.assertEqual(gail['initializer']['algorithm'],'bc')
            self.assertIsNone(gail['initializer']['initializer'])
            self.assertFalse(any(c.get('row',{}).get('slip') is None for c in calls if c.get('kind')=='oracle'))
            da=exp.train('varied','hidden',42,'dagger_mixed')
            self.assertIsNone(da['initializer'])
            supervised=[c for c in calls if c['task']=='supervised' and c['architecture']=='hidden']
            self.assertEqual(len(supervised[-1]['train']),11) # one shared seed shard +10 learner shards
            self.assertEqual(supervised[-1]['epochs'],1)
            previous=len(calls);exp.train('varied','hidden',42,'dagger_mixed')
            self.assertEqual(previous,len(calls))

    def test_parent_audit_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'plan.json',base_plan(True))
            identity=dict(protocol='consolidated_imitation_v1',plan=digest(root/'plan.json'),sources={},environment={})
            save(root/'runtime_identity.json',identity)
            d=root/'dependency';d.mkdir();(d/'policy.pth').write_bytes(b'placeholder')
            spec=dict(task='initialize',kind='bc',architecture='hidden')
            commit(d,dict(identity,spec=spec,inputs={}))
            parent=Parent(root)
            with patch('scratch_comparison.run.validate_artifacts'):
                parent.audit(d)
            (d/'policy.pth').write_bytes(b'changed')
            with self.assertRaises(ValueError):Parent(root).audit(d)

    def test_report_exposure_not_pooled_seeds(self):
        base=dict(row={'suite':'grouped'},controller=dict(label='x',seed=42),falls=1,
                  return_target_progress_m=10.,normalized_return=1.)
        records=aggregate([base,dict(base,falls=0)])
        self.assertEqual(records[0]['rounds'],2)
        self.assertEqual(records[0]['total_falls'],1)
        self.assertEqual(records[0]['normalized_return'],1.)
        self.assertIsNone(records[0]['held_out_expert_action_mse'])

    def test_evaluation_appends_without_duplicate_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';out=root/'new'
            row=dict(id='grouped_test',suite='grouped',slip=.1,seed=101)
            def controller(label,domain='varied',arch='hidden',scratch=False):
                c=dict(label=label,domain=domain,architecture=arch,seed=42,algorithm='bc',kind='bc',model='model')
                if scratch:c['initialization_cohort']='scratch'
                return c
            parents=[controller('oracle','baseline'),controller('random','baseline'),
                     controller('varied_hidden_42_bc'),controller('varied_input_42_bc_correct',arch='input')]
            new=[controller('scratch_varied_hidden_42_bc',scratch=True),
                 controller('scratch_varied_input_42_bc_correct',arch='input',scratch=True)]
            def result(c,progress):return dict(row=row,controller=c,return_target_progress_m=progress,falls=0,
                                             normalized_return=progress/10.,on_policy_latent_action_mse=.01)
            cached=dict(runs=[result(c,0 if c['label']=='random' else 10.) for c in parents])
            original=copy.deepcopy(cached)
            joints=np.random.default_rng(4).normal(size=(10,22)).astype(np.float32)
            p=old/'runs'/row['id']/'oracle';p.mkdir(parents=True)
            np.savez(p/'trace.npz',joint_positions_deg=joints)
            for c in new:
                d=out/'runs'/row['id']/c['label'];save(d/'result.json',result(c,11.))
                np.savez(d/'trace.npz',joint_positions_deg=joints,episode_id=np.zeros(10),
                         torso_roll_deg=np.zeros(10),torso_pitch_deg=np.zeros(10))
            exp=SimpleNamespace(plan={'protocol':'test'},offline_score=lambda *a:{'element_mse':.01},data=lambda *a:[])
            schedule=dict(rows=[row],controllers=parents+new,parent_comparison_sha256='parenthash')
            evaluation_report(exp,out,schedule,cached,old)
            combined=read(out/'comparison.json')
            self.assertEqual(len(combined['runs']),6)
            self.assertEqual(cached,original)
            self.assertEqual(combined['runs'][:4],cached['runs'])
            self.assertEqual(combined['runs'][4]['normalized_return'],1.1)
            self.assertEqual(len(combined['paired_differences']),3)

    def test_probe_append_and_resume(self):
        rng=np.random.default_rng(1);data={}
        for split in ('train','validation','test'):
            x=rng.normal(size=(24,63));data[split]=(x,x[:,:6],np.repeat([0,1,2],8),np.arange(24),x[:,:6])
        spec=dict(alphas=[.01,1.],target_names=[f't{i}' for i in range(6)])
        old=[]
        for label in ('expert_h1','varied_hidden_42_bc_h1'):
            for t in spec['target_names']:
                old.append(dict(representation=label,target=t,labels='true',r2=.5,rmse=1.,alpha=.01,
                                mean_baseline_rmse=1.,macro_run_rmse=1.,contact_accuracy=None,
                                contact_balanced_accuracy=None,contact_positive_fraction=None))
        old_report=dict(geometry=[],persistence=[])
        controller=dict(label='scratch_varied_hidden_42_bc',algorithm='bc')
        reps=[(controller['label']+'_h1',{s:d[0][:,:6] for s,d in data.items()})]
        plan=dict(protocol='test',parent_probes='unchanged',parent_report_sha256='test')
        with tempfile.TemporaryDirectory() as tmp,patch('scratch_comparison.probe.new_features',return_value=reps):
            out=Path(tmp);save(out/'plan.json',plan)
            execute(out,plan,spec,data,load_expert(),old,old_report,[controller])
            report=read(out/'report.json')
            self.assertEqual(report['results'][:12],old)
            self.assertEqual(len(report['results']),36)
            self.assertEqual(len(report['initialization_comparison']),6)
            with patch('scratch_comparison.probe.fit_probe',side_effect=AssertionError('refit')):
                execute(out,plan,spec,data,load_expert(),old,old_report,[controller])


if __name__=='__main__':unittest.main()

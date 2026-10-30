import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from train import BCPolicy
from consolidated.models import load_policy
from variable_training.common import read,save
from .run import Study,make_plan,controller
from .train import fit_arrays,Stopper,train


class ConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_plan_and_stopping_rule(self):
        p=make_plan(Path('/tmp/parent'))
        self.assertEqual(p['maximum_additional_epochs'],1000)
        self.assertEqual(p['seeds'],[42,43,44])
        q=make_plan(Path('/tmp/parent'),quick=True)
        self.assertEqual(q['evaluation_steps'],32)
        for kw in [dict(patience=0),dict(min_delta=float('nan')),dict(max_epochs=10,minimum_epochs=25)]:
            with self.assertRaises(ValueError):make_plan(Path('/tmp/parent'),**kw)
        s=Stopper(2,.01)
        self.assertFalse(s.observe(1));self.assertFalse(s.observe(.995));self.assertTrue(s.observe(.994))
        self.assertFalse(s.observe(.98))

    def fit(self,root,lr=0.,epochs=10,patience=2,optimizer=None):
        initial=root/'initial.pth';torch.manual_seed(7)
        if not initial.exists():torch.save(BCPolicy(63,16).state_dict(),initial)
        rng=np.random.default_rng(1)
        x=rng.normal(size=(10,63)).astype(np.float32);y=rng.normal(size=(10,16)).astype(np.float32)
        job=dict(model=str(initial),optimizer=optimizer,architecture='hidden',seed=42,learning_rate=lr,
                 batch_size=10,epochs=epochs,device='cpu',early_stop=True,minimum_epochs=2,
                 patience=patience,min_delta=1e-5,inherited_costs={'gradient_updates':211050,'sample_exposures':54000000})
        return fit_arrays(job,root/'fit',x,y,x[:3],y[:3])

    def test_plateau_keeps_inherited_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fit(root)
            self.assertEqual(m['stop_reason'],'validation_patience')
            self.assertEqual(m['epochs_run'],2);self.assertEqual(m['best_epoch'],0)
            self.assertEqual(m['selected_lineage_gradient_updates'],211050)
            self.assertEqual(m['total_lineage_gradient_updates'],211052)
            self.assertEqual(read(root/'fit/history.json')[0]['epoch'],0)
            a=torch.load(root/'initial.pth',weights_only=True);b=torch.load(root/'fit/policy.pth',weights_only=True)
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_optimizer_restored_and_cap_not_called_convergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=BCPolicy(63,16);optim=torch.optim.Adam(model.parameters(),lr=1e-4)
            optim.zero_grad();model(torch.ones(2,63)).square().mean().backward();optim.step()
            torch.save(model.state_dict(),root/'initial.pth');torch.save(optim.state_dict(),root/'optimizer.pth')
            m=self.fit(root,lr=1e-4,epochs=1,optimizer=str(root/'optimizer.pth'))
            self.assertEqual(m['stop_reason'],'budget_exhausted');self.assertFalse(m['validation_stopping_reached'])
            state=torch.load(root/'fit/last_optimizer.pth',weights_only=True)['state']
            self.assertTrue(all(float(x['step'])==2 for x in state.values()))
            self.assertLessEqual(m['validation_mse'],m['initial_validation_mse'])

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):train({'train':['same'],'validation':['same']},Path('/unused'))

    def study(self,root):
        s=Study.__new__(Study);s.root=root;s.identity={'test':True};s.retries=0;s.plan={'seeds':[42,43],'quick':True}
        return s

    def test_random_network_is_fixed_state_dependent_and_receipted(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.study(Path(tmp));cs=s.engine_models();models=[load_policy(c['model'],'bc','hidden') for c in cs]
            x=torch.randn(8,63)
            with torch.no_grad():
                a,b=models[0](x),models[1](x)
                self.assertFalse(torch.equal(a,b));self.assertFalse(torch.equal(a,torch.zeros_like(a)))
                self.assertTrue(torch.equal(a,models[0](x)));self.assertFalse(torch.equal(a[0],a[1]))
            self.assertEqual(s.engine_models(True),cs)
            with patch('convergence_controls.run.random_network',side_effect=AssertionError('must not recreate')):
                s.engine_models()
            Path(cs[0]['model']).write_bytes(b'tampered')
            with self.assertRaises(ValueError):s.engine_models(True)

    def test_missing_checks_do_not_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=self.study(Path(tmp))
            with self.assertRaises(ValueError):s.engine_models(True)
            self.assertFalse((s.root/'training').exists())

    def test_evaluation_shares_engine_jobs_and_skips_original_for_varied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=self.study(root);s.plan={'quick':False,'evaluation_steps':3000}
            s.parent=type('Parent',(),{'root':root/'parent'})()
            save(s.parent.root/'evaluation/all_all_correct/schedule.json',{'test':True})
            rows=[dict(id='original_101',suite='original',slip=None,seed=101),
                  dict(id='grouped_0',suite='grouped',slip=.2,seed=102)]
            s.rows=lambda:rows
            original=controller('original_fit');varied=controller('varied_fit',domain='varied',architecture='input')
            engine=controller('engine_random_network_42',domain='baseline')
            s.trained=lambda check_only:[original,varied]
            s.engine_models=lambda check_only:[engine]
            jobs=[]
            s.job=lambda directory,job,*args,**kwargs:jobs.append((directory,job))
            s.evaluation(engine_only=True);engine_jobs=list(jobs);jobs.clear()
            s.evaluation()
            self.assertEqual(len(jobs),5)
            self.assertTrue(all(item in jobs for item in engine_jobs))
            self.assertFalse(any(j['row']['slip'] is None and j['controller']['domain']=='varied' for _,j in jobs))

    def test_engine_report_is_offline_and_independent_of_later_training(self):
        from .report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=self.study(root)
            s.parent=type('Parent',(),{'root':root/'parent'})()
            data=root/'test.npz';np.savez(data,obs=np.zeros((2,63),np.float32),actions=np.zeros((2,16),np.float32))
            s.parent.data=lambda domain,group:[str(data)]
            row=dict(id='original_101',suite='original',slip=None,seed=101)
            cs=[controller(k,kind=k,seed=None) for k in ['oracle','random']]
            schedule=dict(rows=[row],controllers=cs,steps=32,scope='engine');out=root/'evaluation'
            save(out/'schedule_engine.json',schedule)
            for i,c in enumerate(cs):
                d=out/'runs'/row['id']/c['label']
                save(d/'result.json',dict(row=row,controller=c,active_seconds=.64,active_steps=32,
                     return_target_progress_m=1.-i,falls=0,falls_per_active_minute=0,on_policy_latent_action_mse=0,
                     mean_forward_velocity_mps=1.-i,mean_target_direction_error=0))
                save(d/'completed.json',{'test':True})
                np.savez(d/'trace.npz',joint_positions_deg=np.zeros((32,22)),episode_id=np.zeros(32),
                         torso_roll_deg=np.zeros(32),torso_pitch_deg=np.zeros(32))
            report(s,out,schedule)
            result=read(out/'summary_engine/comparison.json')
            self.assertEqual([r['normalized_return'] for r in result['runs']],[1.,0.])
            save(root/'training/later/metrics.json',{'new_training':'must not invalidate engine report'})
            report(s,out,schedule)
            with self.assertRaises(ValueError):report(s,out,dict(schedule,steps=3000))


if __name__=='__main__':unittest.main()

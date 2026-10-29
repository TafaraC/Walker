import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from gail_models import GAILActorCritic
from variable_training.common import read, save
from .run import make_plan, checkpoints, select_checkpoint
from .update import exact_kl, actor_snapshot, restore_actor, train
from .report import report


torch.set_num_threads(1)


def args(**changes):
    values=dict(parent=Path('/tmp/parent'),domain='original',recipes=['adaptive'],seeds=[42,43,44],
                steps=350000,rollout_steps=10000,eval_every=7,quick=False)
    values.update(changes)
    return argparse.Namespace(**values)


class DesignTests(unittest.TestCase):
    def test_budgets(self):
        p=make_plan(args())
        self.assertEqual(p['iterations'],35)
        self.assertEqual(checkpoints(p),[0,7,14,21,28,35])
        self.assertFalse(set(p['validation_seeds']) & set(p['confirmation_seeds']))
        self.assertEqual(p['active_steps']*len(p['seeds'])*len(p['initializers']),3150000)

    def test_invalid(self):
        for changed in [dict(rollout_steps=0),dict(steps=-1),dict(steps=123),dict(seeds=[42,42]),dict(eval_every=0)]:
            with self.assertRaises(ValueError):make_plan(args(**changed))

    def test_quick_is_distinct(self):
        p=make_plan(args(quick=True));self.assertEqual(p['active_steps'],200)
        self.assertEqual(p['seeds'],[42]);self.assertEqual(checkpoints(p),[0,1,2])

    def test_selection_can_keep_initializer(self):
        curve=[dict(iteration=0,progress=40,falls=0),dict(iteration=7,progress=45,falls=3),
               dict(iteration=14,progress=39,falls=0)]
        self.assertEqual(select_checkpoint(curve)['iteration'],0)
        curve.append(dict(iteration=21,progress=41,falls=1))
        self.assertEqual(select_checkpoint(curve)['iteration'],21)

    def test_snapshot_restores_optimizer_and_actor(self):
        p=GAILActorCritic(63,16);opt=torch.optim.Adam(list(p.actor_mean.parameters())+[p.actor_log_std],lr=.001)
        x=torch.randn(4,63);old=p(x)[0];snap=actor_snapshot(p,opt)
        oldmean=old.mean.detach();oldstd=old.stddev.detach()
        with torch.no_grad():p.actor_mean[-1].bias.add_(.1)
        self.assertGreater(exact_kl(p,x,oldmean,oldstd),0)
        restore_actor(p,opt,snap)
        self.assertAlmostEqual(exact_kl(p,x,oldmean,oldstd),0,places=6)


class UpdateTests(unittest.TestCase):
    def fixture(self,root,iteration=1,previous=None):
        torch.manual_seed(7)
        policy=GAILActorCritic(63,16)
        if previous:policy.load_state_dict(torch.load(previous/'policy.pth',weights_only=True))
        model=root/f'initial_{iteration}.pth';torch.save(policy.state_dict(),model)
        rng=np.random.default_rng(8)
        x=rng.normal(size=(40,63)).astype('float32')
        y=rng.normal(size=(40,16)).astype('float32')
        data=root/'expert.npz';np.savez_compressed(data,obs=x,actions=y)
        with torch.no_grad():
            d,v=policy(torch.from_numpy(x));actions=d.sample();lp=d.log_prob(actions).sum(-1)
        rollout=root/f'rollout_{iteration}.npz'
        np.savez_compressed(rollout,obs=x,actions=actions.numpy(),executed_actions=actions.numpy(),
                            values=v.squeeze(-1).numpy(),logprobs=lp.numpy(),dones=np.zeros(40,dtype=bool),bootstrap_value=0.)
        cfg=make_plan(args())['settings'];cfg.update(batch_size=16,disc_epochs=1,ppo_epochs=1,critic_epochs=1)
        return dict(seed=9,iteration=iteration,model=str(model),expert=[str(data)],normalizer=[str(data)],
                    rollouts=[str(rollout)],settings=cfg,
                    optimizer=str(previous/'optimizer.pth') if previous else None,
                    discriminator=str(previous/'discriminator.pth') if previous else None)

    def test_real_update_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);first=root/'first';second=root/'second';first.mkdir();second.mkdir()
            job=self.fixture(root)
            with patch('torch.cuda.is_available',return_value=False):train(job,first)
            m=read(first/'metrics.json')
            self.assertTrue(np.isfinite(m['reward_mean']));self.assertEqual(m['anchor_coefficient'],0)
            self.assertLessEqual(m['accepted_full_kl'],job['settings']['max_kl'])
            self.assertGreater(m['actor_mean_shift_mse'],0)
            job=self.fixture(root,2,first)
            with patch('torch.cuda.is_available',return_value=False):train(job,second)
            self.assertTrue((second/'optimizer.pth').exists())

    def test_mismatched_rollout_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=self.fixture(root)
            state=torch.load(job['model'],weights_only=True);state['actor_mean.4.bias']+=1
            torch.save(state,job['model'])
            with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(ValueError,'not from'):
                train(job,root)

    def test_kl_rejection_restores_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=self.fixture(root)
            with patch('torch.cuda.is_available',return_value=False),patch('gail_experiments.update.exact_kl',side_effect=[1.,1.,1.,1.,0.]):
                train(job,root)
            before=torch.load(job['model'],weights_only=True);after=torch.load(root/'policy.pth',weights_only=True)
            for k in before:
                if k.startswith('actor'):self.assertTrue(torch.equal(before[k],after[k]))
            self.assertEqual(read(root/'metrics.json')['accepted_epochs'],0)


class ReportingTests(unittest.TestCase):
    def test_confirmation_roles_and_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=dict(recipes=['adaptive'],initializers=['bc'],iterations=1,rollout_steps=100,protocol='test')
            save(root/'plan.json',plan)
            save(root/'selection.json',dict(selected=[dict(recipe='adaptive',initializer='bc',seed=42,selected=0,
                 curve=[dict(iteration=0,progress=40.,falls=0,validation_action_mse=.1),
                        dict(iteration=1,progress=39.,falls=0,validation_action_mse=.2)])]))
            row=dict(id='confirmation',seed=1901,slip=None)
            def controller(recipe,name,seed,i):return dict(label=f'actor_{i}',model=str(root/f'actor_{i}.pth'))
            study=argparse.Namespace(root=root,plan=plan,tests=[],controller=controller,rows=lambda _: [row])
            for label,progress in [('oracle',40.),('random',0.),('actor_0',40.),('actor_1',41.)]:
                d=root/'evaluation/confirmation/confirmation'/label;d.mkdir(parents=True)
                save(d/'result.json',dict(return_target_progress_m=progress,falls=0,targets_reached=5,
                     mean_forward_velocity_mps=.7,mean_target_direction_error=.1,on_policy_latent_action_mse=.1))
                np.savez_compressed(d/'trace.npz',joint_positions_deg=np.ones((20,22)),episode_id=np.zeros(20))
            with patch('gail_experiments.report.Experiment.offline_score',return_value={'element_mse':.1}):
                report(study)
            data=read(root/'comparison.json')
            self.assertEqual(len(data['comparisons']),3)
            self.assertEqual(data['comparisons'][1]['iteration'],0)
            self.assertEqual(data['comparisons'][2]['paired_progress_delta_m'],1.)
            self.assertIn('Across-seed',(root/'comparison.md').read_text())


if __name__=='__main__':unittest.main()

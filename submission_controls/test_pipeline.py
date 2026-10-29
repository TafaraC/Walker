import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train import BCPolicy
from consolidated.models import load_policy
from variable_training.common import read, save
from .run import make_plan, Study, controller
from .train import update_target, Stopper, fit_arrays


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_budget(self):
        self.assertEqual(update_target(170000, 256, epochs=317), 210805)
        self.assertEqual(211050-210805, 245)
        self.assertEqual(update_target(170000, 256, updates=211050), 211050)
        with self.assertRaises(ValueError):
            update_target(170000, 256, epochs=10, updates=100)

    def test_plan_smoke_is_not_research(self):
        plan = make_plan(Path('/tmp/parent'), True)
        self.assertEqual(plan['matched_updates'], 3)
        self.assertEqual(plan['evaluation_steps'], 32)
        self.assertEqual(plan['architectures'], ['hidden'])
        full = make_plan(Path('/tmp/parent'))
        self.assertEqual(full['varied_epochs'], 200)
        self.assertEqual(full['seeds'], [42,43,44])

    def test_early_stop_min_delta(self):
        stopper = Stopper(2, .01)
        self.assertFalse(stopper.observe(1.))
        self.assertFalse(stopper.observe(.995))
        self.assertTrue(stopper.observe(.994))
        self.assertFalse(stopper.observe(.9))

    def fit(self, root, **settings):
        init = root/'initial.pth'
        torch.manual_seed(7)
        torch.save(BCPolicy(63,16).state_dict(), init)
        job = dict(model=str(init), architecture='hidden', seed=42, learning_rate=1e-4,
                   batch_size=4, updates=4, device='cpu')
        job.update(settings)
        rng = np.random.default_rng(3)
        x = rng.normal(size=(10,63)).astype(np.float32)
        y = rng.normal(size=(10,16)).astype(np.float32)
        vx, vy = x[:3].copy(), y[:3].copy()
        return job, fit_arrays(job, root/'fit', x, y, vx, vy)

    def test_partial_epoch_and_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, metrics = self.fit(root)
            self.assertEqual(metrics['gradient_updates'], 4)
            self.assertEqual(metrics['sample_exposures'], 14)
            history = read(root/'fit/history.json')
            self.assertTrue(history[-1]['partial_epoch'])
            self.assertEqual(metrics['validation_mse'], min(r['validation_mse'] for r in history))
            for name in ('policy.pth','last_policy.pth'):
                load_policy(root/'fit'/name, 'bc', 'hidden')
            self.assertTrue((root/'fit/optimizer.pth').exists())

    def test_early_stop_has_cap_and_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, metrics = self.fit(Path(tmp), updates=None, epochs=10, early_stop=True,
                                  minimum_epochs=2, patience=1, min_delta=1e9)
            self.assertEqual(metrics['stop_reason'], 'validation_patience')
            self.assertEqual(metrics['epochs_run'], 2)
            self.assertLess(metrics['gradient_updates'], metrics['target_gradient_updates'])

    def test_zero_policy_receipt_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Study.__new__(Study)
            study.root, study.identity, study.retries = Path(tmp), {'test':True}, 0
            c = study.zero()
            model = load_policy(c['model'], 'bc', 'hidden')
            with torch.no_grad():
                self.assertTrue(torch.equal(model(torch.randn(32,63)), torch.zeros(32,16)))
            self.assertEqual(study.zero(True), c)
            Path(c['model']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                study.zero(True)

    def test_missing_job_check_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Study.__new__(Study)
            study.root, study.identity, study.retries = Path(tmp), {'test':True}, 0
            with self.assertRaises(ValueError):
                study.zero(True)
            self.assertFalse((Path(tmp)/'training/zero_native').exists())

    def test_report_smoke_uses_matched_horizon(self):
        from .report import report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = Study.__new__(Study)
            study.root, study.parent = root, type('Parent', (), {'root': root/'parent'})()
            study.plan, study.identity = {'quick':True}, {'test':True}
            dataset = root/'test.npz'
            np.savez(dataset, obs=np.zeros((2,63),np.float32), actions=np.zeros((2,16),np.float32))
            study.parent.data = lambda domain, group: [str(dataset)]
            row = dict(id='original_101', suite='original', slip=None, seed=101)
            controllers = [controller(k, kind=k, seed=None) for k in ('oracle','random')]
            schedule = dict(rows=[row], controllers=controllers, steps=32)
            out = root/'evaluation'
            save(out/'schedule.json', schedule)
            for i,c in enumerate(controllers):
                directory=out/'runs'/row['id']/c['label']
                save(directory/'result.json', dict(row=row, controller=c, active_seconds=.64, active_steps=32,
                     return_target_progress_m=1.-i, falls=0, falls_per_active_minute=0,
                     on_policy_latent_action_mse=0, mean_forward_velocity_mps=1.-i,
                     mean_target_direction_error=0))
                save(directory/'completed.json', {'test':True})
                np.savez(directory/'trace.npz',joint_positions_deg=np.zeros((32,22)),
                         episode_id=np.zeros(32),torso_roll_deg=np.zeros(32),torso_pitch_deg=np.zeros(32))
            report(study,out,schedule)

            result=read(out/'summary/comparison.json')
            self.assertEqual([r['normalized_return'] for r in result['runs']], [1.,0.])
            report(study,out,schedule)
            broken = dict(schedule, steps=3000)
            with self.assertRaises(ValueError):
                report(study,out,broken)

    def test_input_model_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init = root/'input.pth'
            torch.save(BCPolicy(64,16).state_dict(),init)
            job = dict(model=str(init), architecture='input', seed=42, learning_rate=1e-4,
                       batch_size=4, epochs=2, device='cpu')
            x, y = np.zeros((7,64),np.float32), np.zeros((7,16),np.float32)
            metrics = fit_arrays(job,root/'fit',x,y,x[:3],y[:3])
            self.assertEqual(metrics['gradient_updates'],4)
            self.assertEqual(metrics['sample_exposures'],14)
            self.assertEqual(load_policy(root/'fit/policy.pth','bc','input').net[0].in_features,64)

    def test_overlap_rejected_before_fit(self):
        from .train import train
        with self.assertRaises(ValueError):
            train({'train':['same.npz'],'validation':['same.npz']},Path('/nonexistent'))


if __name__ == '__main__':
    unittest.main()

"""Offline regression tests; temporary synthetic data only, no simulator."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from configure_field_slip import status
from train import BCPolicy
from variable_training.common import commit, digest, read, save
from variable_training.plan import make_plan
from .data import audit, load_split
from .models import checkpoint, encode_slip, initialize, load_checkpoint, policy_inputs, predict
from .run import main, train_one


def fixture(root):
    plan = make_plan(quick=True)
    path = root / 'plan.json'
    save(path, plan)
    for split in ('train', 'validation', 'test'):
        for row in plan['groups'][split]:
            directory = root / 'data' / 'varied' / split / row['id']
            directory.mkdir(parents=True)
            (directory/'soccer.rsg').write_text(f'(setContactSlip {row["slip"]:.12g})\n')
            manifest = status(directory/'soccer.rsg', row['slip'])
            manifest['server_scene_import_verified'] = True
            save(directory/'environment_manifest.json', manifest)
            rng = np.random.default_rng(row['seed'])
            obs = rng.normal(size=(row['steps'], 63)).astype(np.float32)
            actions = rng.normal(size=(row['steps'], 16)).astype(np.float32)
            np.savez(directory/'data.npz', obs=obs, actions=actions, executed_actions=actions)
            save(directory/'metrics.json', dict(slip=row['slip'], active_steps=row['steps'],
                 oracle_queries=row['steps'], oracle_control_steps=row['steps'], model_sha256=None))
            commit(directory, dict(plan=digest(path), spec=dict(task='collect', kind='oracle', row=row)))
    return path


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_encoding_and_invalid_inputs(self):
        self.assertAlmostEqual(float(encode_slip(10.)), 1.)
        self.assertTrue(np.all(np.diff(encode_slip([.025, .1, .2, 1., 10.])) > 0))
        for bad in [0, -1, 10.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                encode_slip(bad)
        obs = np.zeros((3, 63), np.float32)
        self.assertEqual(policy_inputs(obs, .1, 'input').shape, (3, 64))
        self.assertEqual(policy_inputs(obs[0], .1, 'input').shape, (64,))
        self.assertEqual(policy_inputs(obs, [.1, 1., 10.], 'hidden').shape, (3, 63))
        with self.assertRaises(ValueError):
            policy_inputs(obs, [.1, .2], 'input')
        with self.assertRaises(ValueError):
            policy_inputs(np.zeros((3, 64)), .1, 'input')

    def test_initialization_preserves_predictions_and_can_learn_context(self):
        torch.manual_seed(7)
        old = BCPolicy(63, 16).eval()
        model = initialize(old.state_dict(), 'input')
        obs = np.random.default_rng(7).normal(size=(9, 63)).astype(np.float32)
        expected = old(torch.from_numpy(obs)).detach().numpy()
        for slip in [.025, .1, 10.]:
            np.testing.assert_allclose(predict(model, 'input', obs, slip), expected, atol=1e-6)
        self.assertEqual(int(torch.count_nonzero(model.net[0].weight[:, -1])), 0)
        model(torch.from_numpy(policy_inputs(obs, 10., 'input'))).square().mean().backward()
        self.assertGreater(float(model.net[0].weight.grad[:, -1].abs().sum()), 0)
        hidden = initialize(old.state_dict(), 'hidden')
        np.testing.assert_array_equal(predict(hidden, 'hidden', obs, .1),
                                      predict(hidden, 'hidden', obs, 10.))

    def test_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'policy.pt'
            model = initialize(BCPolicy(63, 16).state_dict(), 'input')
            torch.save(checkpoint(model, 'input'), path)
            restored, architecture = load_checkpoint(path)
            self.assertEqual(architecture, 'input')
            self.assertEqual(restored.net[0].in_features, 64)
            payload = checkpoint(model, 'input')
            payload['context']['encoding'] = 'unknown'
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_audit_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = fixture(Path(tmp))
            dataset = audit(plan)
            entry = dataset['splits']['train'][0]
            x, y, slip = load_split([entry])
            self.assertEqual(x.shape, (entry['steps'], 63))
            self.assertEqual(y.shape, (entry['steps'], 16))
            self.assertTrue(np.all(slip == entry['slip']))
            Path(entry['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                audit(plan)

    def test_reject_wrong_plan_and_unverified_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = fixture(Path(tmp))
            dataset = audit(plan)
            directory = Path(dataset['splits']['train'][0]['path']).parent
            receipt = read(directory/'completed.json')
            manifest = read(directory/'environment_manifest.json')
            manifest['server_scene_import_verified'] = False
            save(directory/'environment_manifest.json', manifest)
            commit(directory, receipt['identity'])
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                audit(plan)
            receipt['identity']['plan'] = 'changed'
            commit(directory, receipt['identity'])
            with self.assertRaisesRegex(ValueError, 'provenance'):
                audit(plan)

    def test_offline_training_both_architectures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root)
            dataset = audit(plan)
            splits = {key: load_split(value) for key, value in dataset['splits'].items()}
            initial = root/'initial.pth'
            torch.save(BCPolicy(63, 16).state_dict(), initial)
            metrics = []
            for architecture in ('hidden', 'input'):
                directory = root/architecture
                directory.mkdir()
                metrics.append(train_one(directory, architecture, 7,
                    dict(epochs=2, batch_size=16, learning_rate=1e-4), initial, splits, torch.device('cpu')))
                model, loaded = load_checkpoint(directory/'policy.pt')
                self.assertEqual(loaded, architecture)
                self.assertTrue(np.isfinite(metrics[-1]['test']['element_mse']))
            self.assertEqual(metrics[0]['gradient_updates'], metrics[1]['gradient_updates'])
            self.assertEqual(metrics[1]['parameter_count']-metrics[0]['parameter_count'], 256)

    def test_cli_resume_and_changed_settings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = fixture(root/'source')
            initial = root/'initial.pth'
            torch.save(BCPolicy(63, 16).state_dict(), initial)
            args = ['context_training.run', 'train', '--data-plan', str(plan),
                    '--output', str(root/'output'), '--initializer', str(initial),
                    '--epochs', '1', '--seeds', '7']
            with patch('sys.argv', args):
                main()
            receipt = root/'output/input/seed_7/completed.json'
            before = receipt.read_bytes()
            with patch('sys.argv', args), patch('context_training.run.train_one', side_effect=AssertionError):
                main()
            self.assertEqual(before, receipt.read_bytes())
            with patch('sys.argv', args+['--learning-rate', '.01']):
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    main()


if __name__ == '__main__':
    unittest.main()

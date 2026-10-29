"""Offline tests: no SimSpark processes or sockets are opened."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from variable_training.common import begin, commit, read, save, verified
from variable_training.plan import make_plan, truncated_draws, validate_plan
from variable_training.run import Experiment, validate_shard
from variable_training.worker import supervised, gail
from variable_training.gail_update import segmented_gae
from variable_training import server
from variable_training.common import simulator_lock
from variable_training.common import evaluation_identity, verified_evaluation, _LEGACY_EVALUATION_METADATA
from train import BCPolicy
from gail_models import GAILActorCritic


class PipelineTests(unittest.TestCase):
    def test_decimal_slip_validation_and_exact_resume_compatibility(self):
        from configure_field_slip import apply, status
        old='721c61da95cdee36ba6c30b0bef249109183c5077e5d229cbf5614828ca6dc17'
        new='2c2337a3368cbffcbdb992a22915657c13bf3e39b9f113afd29674c893625542'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scene=root/'soccer.rsg'
            scene.write_text('(setContactSlip 0.1)\n')
            slip=5.01334232429452
            result=apply(scene,slip,root/'backup')
            self.assertTrue(result['matches_expected'])
            self.assertEqual(scene.read_text(),'(setContactSlip 5.01334232429)\n')
            self.assertFalse(status(scene,slip+1e-8)['matches_expected'])
            identity=dict(sources={'configure_field_slip.py':old},row={'slip':slip})
            commit(root,identity)
            receipt=(root/'completed.json').read_bytes()
            self.assertTrue(verified_evaluation(root,dict(identity,sources={'configure_field_slip.py':new})))
            self.assertEqual(receipt,(root/'completed.json').read_bytes())
            with self.assertRaises(ValueError):
                verified_evaluation(root,dict(identity,sources={'configure_field_slip.py':'unknown'}))
            with self.assertRaises(ValueError):
                verified_evaluation(root,dict(identity,row={'slip':1.}))
            scene.write_text('(setContactSlip 9)\n')
            with self.assertRaises(ValueError):
                verified_evaluation(root,dict(identity,sources={'configure_field_slip.py':new}))
    def test_audited_legacy_evaluation_resume_remains_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            sources = {name: sorted(approved)[0] for name, approved in _LEGACY_EVALUATION_METADATA.items()}
            sources['evaluate_policy.py'] = 'unchanged-client'
            old = dict(sources=sources, plan='same-plan', model='same-model', row={'slip': .1})
            (directory/'trace').write_text('original data')
            commit(directory, old)
            original_receipt = (directory/'completed.json').read_bytes()
            new = evaluation_identity(old)
            self.assertTrue(verified_evaluation(directory, new))
            self.assertEqual((directory/'completed.json').read_bytes(), original_receipt)
            for field, value in [('plan', 'changed'), ('model', 'changed'), ('row', {'slip': .2}),
                                 ('sources', {'evaluate_policy.py': 'changed-client'})]:
                with self.assertRaises(ValueError):
                    verified_evaluation(directory, dict(new, **{field:value}))
            unreviewed = dict(old, sources=dict(sources, **{'variable_training/run.py': 'unknown'}))
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                evaluation_identity(unreviewed, legacy=True)
            (directory/'trace').write_text('corruption')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                verified_evaluation(directory, new)

    def test_gaussian_only_report_needs_no_fixed_runs(self):
        from variable_training.common import MODELS
        with tempfile.TemporaryDirectory() as tmp:
            exp = Experiment.__new__(Experiment)
            exp.root, exp.identity, exp.plan = Path(tmp), {}, make_plan()
            self.assertEqual(len(exp.evaluation_rows()), 3)
            self.assertEqual(len(exp.evaluation_rows('fixed')), 9)
            self.assertEqual(len(exp.evaluation_rows('all')), 12)
            for row in exp.evaluation_rows():
                for label in ['oracle'] + [k+'_frozen' for k in MODELS]:
                    directory = exp.root/'evaluation'/row['id']/label
                    save(directory/'raw'/f'{label}_seed_{row["seed"]}.json',
                         dict(policy_label=label, return_target_progress_m=10., falls=0, targets_reached=2))
                    commit(directory, {})
            exp.report(['varied'], frozen_only=True)
            report = read(exp.root/'report.json')
            self.assertEqual(len(report['runs']), 15)
            self.assertEqual(len(report['aggregates']),5)
            self.assertTrue(all(r['rounds']==3 for r in report['aggregates']))
            self.assertEqual(report['evaluation_distribution']['purpose'],'evaluation_only')
            self.assertIn('Fixed slip',(exp.root/'report.md').read_text())
            self.assertIn('Across-round summary',(exp.root/'report.md').read_text())
            self.assertTrue(all(r['nominal_retention_ratio_of_run_to_nominal_mean'] is None
                                for r in report['runs']))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gaussian_plan_and_splits(self):
        plan = make_plan()
        validate_plan(plan)
        self.assertEqual(sum(r['steps'] for r in plan['groups']['train']), 100000)
        self.assertEqual(sum(r['steps'] for k,v in plan['groups'].items()
                             if k.startswith('dagger_') for r in v), 70000)
        rows = [r for v in plan['groups'].values() for r in v]
        self.assertEqual(len({r['id'] for r in rows}), len(rows))
        self.assertEqual(len({r['seed'] for r in rows}), len(rows))
        self.assertFalse({r['seed'] for r in rows} & {r['seed'] for r in plan['evaluation']})
        d = plan['distribution']
        values = truncated_draws(np.random.default_rng(7), 10000, d['low'], d['high'], d['mean'], d['std'])
        self.assertTrue(all(d['low'] < v < d['high'] for v in values))
        self.assertAlmostEqual(np.mean(values), plan['distribution']['effective_mean'],
                               delta=4*np.sqrt(d['effective_variance']/len(values)))
        self.assertEqual(plan, make_plan())
        plan['groups']['train'][0]['slip'] = .3
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_five_round_extension_preserves_training_plan_and_prefix(self):
        exp=Experiment.__new__(Experiment)
        exp.plan=make_plan(); exp.evaluation_rounds=5
        import copy
        before=copy.deepcopy(exp.plan)
        original=[r for r in before['evaluation'] if r['id'].startswith('sampled_')]
        rows=exp.evaluation_rows()
        self.assertEqual(len(rows),len(before['training_environment']['slips']))
        self.assertEqual(rows[:3],original)
        self.assertEqual(len({r['seed'] for r in rows}),5)
        self.assertEqual(exp.plan,before)
        self.assertEqual(len(exp.evaluation_rows('fixed')),9)
        self.assertEqual(len(exp.evaluation_rows('all')),14)

    def test_grouped_gaussians_are_positive_paired_and_separate(self):
        exp=Experiment.__new__(Experiment)
        exp.plan=make_plan(); exp.root=Path('/unused')
        rows=exp.evaluation_rows('grouped')
        self.assertEqual(rows,exp.evaluation_rows('grouped'))
        self.assertEqual(len(rows),15)
        self.assertEqual(len({r['seed'] for r in rows}),15)
        self.assertTrue(all(0<r['slip']<=10 for r in rows))
        self.assertNotEqual(exp.evaluation_directory('grouped'),exp.evaluation_directory('gaussian'))
        for i,mean in enumerate(exp.plan['training_environment']['slips']):
            group=[r for r in rows if r['group']==f'group_{i}']
            self.assertEqual(len(group),3)
            self.assertTrue(all(r['parent_mean']==mean and r['parent_std']==mean*.5 for r in group))

    def test_grouped_report_keeps_group_summaries_and_old_report(self):
        from variable_training.common import MODELS
        with tempfile.TemporaryDirectory() as tmp:
            exp=Experiment.__new__(Experiment)
            exp.root=Path(tmp); exp.plan=make_plan(); exp.identity={}
            original=exp.root/'report.md'; original.write_text('old report')
            for row in exp.evaluation_rows('grouped'):
                for label in ['oracle']+[k+'_frozen' for k in MODELS]:
                    directory=exp.evaluation_directory('grouped')/row['id']/label
                    save(directory/'raw'/f'{label}_seed_{row["seed"]}.json',dict(
                        policy_label=label,return_target_progress_m=10.,falls=0,targets_reached=2))
                    commit(directory,dict(row=row,label=label))
            exp.report(['varied'],frozen_only=True,suite='grouped')
            result=read(exp.evaluation_directory('grouped')/'report.json')
            self.assertEqual(len(result['runs']),75)
            self.assertEqual(len(result['group_aggregates']),25)
            self.assertTrue(all(r['rounds']==3 for r in result['group_aggregates']))
            self.assertEqual(original.read_text(),'old report')

    def test_scope_and_nominal_pairing(self):
        from variable_training.common import MODELS
        self.assertEqual(set(MODELS), {'bc', 'dagger_mixed', 'gail_bc', 'gail_dagger'})
        exp = Experiment.__new__(Experiment)
        exp.plan = make_plan(quick=True)
        varied, nominal = exp.rows('train', 'varied'), exp.rows('train', 'nominal')
        for v,n in zip(varied, nominal):
            self.assertEqual({k:x for k,x in v.items() if k!='slip'},
                             {k:x for k,x in n.items() if k!='slip'})
            self.assertEqual(n['slip'], .1)

    def test_startup_timeout_cleans_owned_server(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(server, 'require_free_ports'), \
                patch.object(server, 'prepare', return_value={'startup_marker': 'missing'}), \
                patch.object(server, 'status', return_value={'matches_expected': True, 'sha256': 'same'}), \
                patch.object(server.subprocess, 'Popen') as process, \
                patch.object(server.time, 'sleep'):
            process.return_value.poll.return_value = None
            with self.assertRaisesRegex(RuntimeError, 'startup marker'):
                with server.simulator(Path(tmp), .1):
                    self.fail('Should not start a client')
            process.return_value.terminate.assert_called_once()
            process.return_value.wait.assert_called_once_with(timeout=10)

    def test_lock_refuses_concurrent_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp, patch('variable_training.common.ROOT', Path(tmp)):
            with simulator_lock():
                with self.assertRaisesRegex(RuntimeError, 'Another simulator'):
                    with simulator_lock():
                        self.fail('Must not acquire existing lock')

    def test_receipt_and_failed_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'job'
            begin(path)
            (path/'old').write_text('failure evidence')
            begin(path)
            self.assertEqual(len(list(Path(tmp).glob('job_failed_*'))), 1)
            (path/'output').write_text('done')
            commit(path, {'plan': 1})
            self.assertTrue(verified(path, {'plan': 1}))
            with self.assertRaises(ValueError):
                verified(path, {'plan': 2})
            (path/'output').write_text('modified')
            with self.assertRaises(ValueError):
                verified(path, {'plan': 1})

    def test_segment_gae_and_fall_bootstrap(self):
        adv, ret = segmented_gae(np.array([1., 2.]), np.zeros(2), np.array([False, True]),
                                 np.array([0, 1, 2]), np.array([10., 999.]))
        np.testing.assert_allclose(adv, [10.9, 2.], rtol=1e-6)
        np.testing.assert_array_equal(ret, adv)
        with self.assertRaises(ValueError):
            segmented_gae(np.ones(2), np.zeros(2), np.zeros(2), [0, 2, 1], [0., 0.])

    def test_supervised_warm_start_and_whole_run_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = BCPolicy(63, 16)
            torch.save(model.state_dict(), root/'initial.pth')
            for name, value in [('train', 0.), ('validation', 1.)]:
                np.savez(root/f'{name}.npz', obs=np.full((4,63), value, dtype=np.float32),
                         actions=np.zeros((4,16), dtype=np.float32))
            out = root/'out'; out.mkdir()
            job = dict(seed=1, train=[str(root/'train.npz')], validation=[str(root/'validation.npz')],
                       model=str(root/'initial.pth'), epochs=1, learning_rate=0., batch_size=2)
            with patch('torch.cuda.is_available', return_value=False):
                supervised(job, out)
            state = torch.load(out/'policy.pth', weights_only=True)
            for k,v in model.state_dict().items():
                torch.testing.assert_close(state[k], v)
            self.assertEqual(read(out/'metrics.json')['gradient_updates'], 2)
            job['validation'] = job['train']
            with self.assertRaises(ValueError):
                supervised(job, out)

    def test_gail_two_segments_offline_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = GAILActorCritic(63,16)
            torch.save(model.state_dict(), root/'initial.pth')
            torch.manual_seed(11)
            expert, rollouts = [], []
            for i in range(2):
                obs = torch.randn(8,63)
                with torch.no_grad():
                    dist, value = model(obs)
                    actions = dist.sample()
                    logs = dist.log_prob(actions).sum(-1)
                e, r = root/f'e{i}.npz', root/f'r{i}.npz'
                np.savez(e, obs=obs.numpy(), actions=actions.numpy())
                np.savez(r, obs=obs.numpy(), actions=actions.numpy(), logprobs=logs.numpy(),
                         values=value.squeeze(-1).numpy(), dones=np.zeros(8,dtype=bool), bootstrap_value=float(i))
                expert.append(str(e)); rollouts.append(str(r))
            out = root/'update'; out.mkdir()
            with patch('torch.cuda.is_available', return_value=False):
                gail(dict(expert=expert, normalizer=expert, rollouts=rollouts,
                          model=str(root/'initial.pth'), iteration=1, seed=4, label='gail_bc'), out)
            self.assertEqual(read(out/'metrics.json')['segment_count'], 2)
            for name in ('policy.pth','optimizer.pth','discriminator.pth'):
                self.assertTrue((out/name).is_file())

    def test_job_resume_does_not_update_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Experiment.__new__(Experiment)
            exp.identity, exp.root = {'test': True}, Path(tmp)
            spec = dict(task='supervised')
            def fake_launch(command, log):
                out = Path(command[-1]).parent
                torch.save(BCPolicy(63,16).state_dict(), out/'policy.pth')
                save(out/'metrics.json', {'done': True})
            with patch('variable_training.run.launch', side_effect=fake_launch) as launch:
                exp.job(exp.root/'update', spec)
                exp.job(exp.root/'update', spec)
                self.assertEqual(launch.call_count, 1)


if __name__ == '__main__':
    unittest.main()

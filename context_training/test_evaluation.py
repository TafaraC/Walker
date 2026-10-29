"""Evaluation orchestration tests without native modules, sockets or simulators."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from configure_field_slip import status
from variable_training.common import commit, read, save
from variable_training.plan import make_plan
from variable_training.run import Experiment
from .evaluate import rounds, supplied_context, job_identity, validate_result, evaluate, report


def fake_result(directory, identity):
    n = identity['steps']
    (directory/'soccer.rsg').write_text(f'(setContactSlip {identity["row"]["slip"]:.12g})\n')
    manifest = status(directory/'soccer.rsg', identity['row']['slip'])
    manifest['server_scene_import_verified'] = True
    save(directory/'environment_manifest.json',manifest)
    trace = dict(obs=np.zeros((n,63)), actions=np.zeros((n,16)), oracle_actions=np.zeros((n,16)),
                 joint_positions_deg=np.zeros((n,22)), episode_id=np.zeros(n),
                 target_progress=np.ones(n),forward_velocity_mps=np.ones(n),target_direction_error=np.zeros(n))
    np.savez(directory/'trace.npz', **trace)
    save(directory/'result.json',dict(row=identity['row'],controller=identity['controller'],
        supplied_slip=identity['supplied_slip'],active_steps=n,active_seconds=n*.02,
        return_target_progress_m=float(n),falls=0,targets_reached=1))


def fake_spec():
    return dict(sources={}, environment={}, training_plan_sha256='test',steps=8,
        rounds=rounds(42,'smoke'),metrics=dict(smoothing_window=5,wasserstein_projections=4,wasserstein_seed=1),
        controllers=[dict(label='oracle',architecture='oracle',training_seed=None,mode='none',model=None),
                     dict(label='hidden_seed_42_none',architecture='hidden',training_seed=42,mode='none',model=None),
                     dict(label='input_seed_42_correct',architecture='input',training_seed=42,mode='correct',model=None),
                     dict(label='input_seed_42_wrong',architecture='input',training_seed=42,mode='wrong',model=None)])


class EvaluationTests(unittest.TestCase):
    def test_worker_hidden_has_no_supplied_context(self):
        import torch
        from train import BCPolicy
        from .eval_worker import evaluate as worker
        robot=SimpleNamespace(loc_head_position=np.array([0.,0.,.5]),imu_torso_orientation=0.,
            imu_torso_roll=0.,imu_torso_pitch=0.,joints_position=np.zeros(22),get_command=lambda:b'')
        def receive():
            robot.loc_head_position[0]+=.01
        agent=SimpleNamespace(world=SimpleNamespace(robot=robot),
            scom=SimpleNamespace(commit_and_send=lambda value:None,receive=receive),terminate=lambda:None)
        walk=SimpleNamespace(observe=lambda *args:np.zeros(63,dtype=np.float32),
            get_expert_action=lambda obs:np.zeros(16,dtype=np.float32),execute_action=lambda action:None)
        recovery=SimpleNamespace(fall_count=0,check_and_recover=lambda agent:False)
        modules={'agent.Base_Agent':SimpleNamespace(Base_Agent=lambda **kwargs:agent),
                 'behaviors.custom.Walk.Walk':SimpleNamespace(Walk=lambda agent:walk),
                 'bc_sim':SimpleNamespace(FallRecovery=lambda:recovery),
                 'collect_seed':SimpleNamespace(reset_server_state=lambda:None)}
        spec=fake_spec();job=job_identity(spec,spec['rounds'][0],spec['controllers'][1])
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules',modules), \
             patch('bc_data.sample_own_half_target',return_value=np.array([10.,0.])), \
             patch('context_training.eval_worker.load_checkpoint',return_value=(BCPolicy(63,16).eval(),'hidden')):
            worker(job,Path(tmp))
            result=read(Path(tmp)/'result.json')
            self.assertIsNone(result['supplied_slip'])
            self.assertEqual(result['active_steps'],8)
            self.assertGreater(result['return_target_progress_m'],0)

    def test_schedule_pairing_and_confirmation(self):
        exp = Experiment.__new__(Experiment)
        exp.plan = make_plan()
        previous = exp.evaluation_rows('grouped')
        current = rounds(42,'grouped')
        self.assertEqual([(r['slip'],r['seed']) for r in previous],[(r['slip'],r['seed']) for r in current])
        self.assertEqual(len(rounds(42,'all')),33)
        self.assertEqual(len(rounds(42,'smoke')),1)
        other=rounds(42,'confirmation')
        self.assertFalse({r['seed'] for r in current} & {r['seed'] for r in other})
        self.assertTrue(all(0 < r['slip'] <= 10 for r in other))

    def test_supplied_context_does_not_change_physics(self):
        row = dict(slip=5.01334232429452)
        original = dict(row)
        self.assertEqual(supplied_context(dict(mode='correct'),row),5.01334232429)
        self.assertEqual(supplied_context(dict(mode='constant'),row),.1)
        self.assertEqual(supplied_context(dict(mode='wrong'),row),.025)
        self.assertEqual(supplied_context(dict(mode='wrong'),dict(slip=.1)),10.)
        self.assertIsNone(supplied_context(dict(mode='none'),row))
        self.assertEqual(row,original)

    def test_validation_rejects_context_horizon_and_scene_changes(self):
        spec=fake_spec(); identity=job_identity(spec,spec['rounds'][0],spec['controllers'][2])
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);fake_result(d,identity)
            validate_result(d,identity)
            result=read(d/'result.json');result['supplied_slip']=10.
            save(d/'result.json',result)
            with self.assertRaisesRegex(ValueError,'metadata'):
                validate_result(d,identity)
            fake_result(d,identity)
            (d/'soccer.rsg').write_text('(setContactSlip 10)\n')
            with self.assertRaisesRegex(ValueError,'physical slip'):
                validate_result(d,identity)

    def test_owned_retry_resume_and_strict_report(self):
        spec=fake_spec(); attempts=[]
        @contextmanager
        def fake_server(directory, slip):
            yield {'server_process':object()}
        def fake_launch(command, log, server):
            directory=log.parent
            attempts.append(directory)
            if len(attempts)==1:
                raise RuntimeError('Synthetic server failure')
            fake_result(directory,read(directory/'job.json'))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with patch('context_training.evaluate.simulator',fake_server),patch('context_training.evaluate.launch',fake_launch):
                evaluate(output,spec,1)
            self.assertEqual(len(attempts),5)
            self.assertEqual(len(list((output/'runs'/spec['rounds'][0]['id']).glob('oracle_failed_*'))),1)
            with patch('context_training.evaluate.simulator',side_effect=AssertionError('must resume')):
                evaluate(output,spec,1)
            report(output,spec)
            result=read(output/'report.json')
            self.assertEqual(len(result['runs']),4)
            self.assertEqual(len(result['paired_differences']),2)
            self.assertTrue(all(r['progress_delta_m']==0 for r in result['paired_differences']))
            d=output/'runs'/spec['rounds'][0]['id']/'oracle'
            (d/'trace.npz').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                report(output,spec)

    def test_report_rejects_missing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                report(Path(tmp),fake_spec())


if __name__=='__main__':
    unittest.main()

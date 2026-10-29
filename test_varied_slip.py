"""Offline contract tests; never start SimSpark or connect sockets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import subprocess
import sys

import numpy as np
import run_varied_slip as runner
from configure_field_slip import digest, read_slip
from summarize_varied_slip import validate_draws


class SlipContracts(unittest.TestCase):
    def test_dead_server_aborts_waiting_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = Mock()
            server.poll.return_value = -6
            with self.assertRaises(subprocess.CalledProcessError):
                runner.launch([sys.executable, '-c', 'import time; time.sleep(10)'],
                              Path(tmp) / 'evaluation.log', server=server)

    def test_isolated_run_resume_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'run'
            def fake_launch(command, log, **kwargs):
                scene = directory / 'soccer.rsg'
                scene.write_text('(setContactSlip 0.1 0.1)')
                for name in ('environment_manifest.json', 'rcssserver3d.rb', 'naosoccersim.rb'):
                    (directory / name).write_text('{}')
                raw = directory / 'raw'
                raw.mkdir()
                trace = raw / 'oracle_seed_101.npz'
                np.savez(trace, obs=np.zeros((3000, 63)), actions=np.zeros((3000, 16)),
                         oracle_actions=np.zeros((3000, 16)), joint_positions_deg=np.zeros((3000, 22)),
                         episode_id=np.zeros(3000), target_progress=np.zeros(3000))
                runner.save(raw / 'oracle_seed_101.json', dict(active_steps=3000, seed=101,
                            policy_label='oracle', environment_id='nominal', field_contact_slip=.1,
                            trace=str(trace)))
            with patch.object(runner, 'simulator') as sim, \
                    patch.object(runner, 'fingerprints', return_value={'test': 'hash'}), \
                    patch.object(runner, 'launch', side_effect=fake_launch) as launch:
                sim.return_value.__enter__.return_value = {'sha256': 'test', 'server_process': None}
                args = ('nominal', .1, 'oracle', 'oracle', None, 101, directory)
                runner.run_isolated(*args)
                runner.run_isolated(*args)
                self.assertEqual(sim.call_count, 1)
                self.assertEqual(launch.call_count, 1)
                (directory / 'raw' / 'oracle_seed_101.npz').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    runner.run_isolated(*args)

    def test_fixed_block_isolates_every_run_without_prompts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'OUT', Path(tmp)), \
                patch.object(runner, 'run_isolated') as run, \
                patch('builtins.input', side_effect=AssertionError('Unexpected prompt')):
            runner.block('nominal', .1, 'fixed')
            self.assertEqual(run.call_count, 41)
            self.assertEqual(len({call.args[-1] for call in run.call_args_list}), 41)

    def test_local_configs_and_nominal_unchanged(self):
        original = digest(runner.SCENE)
        with tempfile.TemporaryDirectory() as tmp:
            for slip in (.025, .1, .2):
                directory = Path(tmp) / str(slip)
                manifest = runner.prepare(directory, slip, 101)
                self.assertEqual(read_slip(directory / 'soccer.rsg')[2], slip)
                nao = (directory / 'naosoccersim.rb').read_text()
                self.assertIn(str(directory / 'soccer.rsg'), nao)
                self.assertIn('raise "Walker scene import failed" unless walker_imported', nao)
                self.assertIn(digest(directory / 'soccer.rsg'), nao)
                self.assertNotIn("scene.importScene('rsg/agent/nao/soccer.rsg')", nao)
                self.assertIn('randomServer.seed(0)', nao)
                server = (directory / 'rcssserver3d.rb').read_text()
                self.assertIn(str(directory / 'naosoccersim.rb'), server)
                self.assertNotIn('WALKER_SCENE_VERIFIED', server)
                self.assertGreater(nao.index('WALKER_SCENE_VERIFIED'),
                                   nao.index("scene.importScene("))
                self.assertEqual(manifest['sha256'], digest(directory / 'soccer.rsg'))
                self.assertEqual(len(manifest['frozen_models']), 6)
            for slip in (float('nan'), float('inf'), -.1):
                with self.assertRaises(ValueError):
                    runner.prepare(Path(tmp) / 'invalid', slip, 101)
        self.assertEqual(original, digest(runner.SCENE))

    def test_server_uses_explicit_config_and_cleans_up_on_timeout(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, 'prepare', return_value={'startup_marker': 'EXPECTED'}), \
                patch.object(runner.subprocess, 'Popen') as popen, \
                patch.object(runner.time, 'sleep'):
            popen.return_value.poll.return_value = None
            directory = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'server.log'):
                with runner.simulator(directory, .1):
                    self.fail('Must not evaluate without a scene marker')
            self.assertEqual(popen.call_args.args[0],
                             ['rcssserver3d', '--script-path', str(directory / 'rcssserver3d.rb')])
            popen.return_value.terminate.assert_called_once()
            popen.return_value.wait.assert_called_once_with(timeout=10)

    def test_original_scripts_preserved(self):
        base = runner.OUT / 'baseline_testing'
        for name, expected in json.loads((base / 'sha256.json').read_text()).items():
            self.assertEqual(digest(runner.ROOT / name), expected)
            self.assertEqual(digest(base / name), expected)
        self.assertNotIn('fresh_start', (runner.ROOT / 'evaluate_policy.py').read_text())

    def test_draw_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.save(root / 'fixed_report.json', {'per_run': []})
            values = np.random.default_rng(42).uniform(.05, .15, 5)
            plan = {'low': .05, 'high': .15, 'rationale': 'Test only', 'draw_seed': 42,
                    'fixed_report_sha256': digest(root / 'fixed_report.json'),
                    'draws': [{'seed': s, 'slip': float(v)} for s, v in zip(runner.SEEDS, values)]}
            runner.save(root / 'sampled_draws.json', plan)
            self.assertEqual(len(validate_draws(root)['draws']), 5)
            plan['draws'][0]['slip'] = .2
            runner.save(root / 'sampled_draws.json', plan)
            with self.assertRaises(ValueError):
                validate_draws(root)

    def test_occupied_ports_do_not_start_server(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'OUT', Path(tmp)), \
                patch('sys.argv', ['run_varied_slip.py', 'smoke']), \
                patch.object(runner.socket, 'socket') as socket, \
                patch.object(runner.subprocess, 'Popen') as popen:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                runner.main()
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contact_slip.run import plan, audit_trace, server


class ContactTests(unittest.TestCase):
    def test_plan_scope_and_distribution(self):
        p=plan()
        self.assertEqual(set(p['policies']),{'oracle','bc','dagger_mixed','gail_bc','gail_dagger'})
        self.assertEqual(len(p['modes'])*len(p['seeds'])*len(p['policies']),75)
        for params in [dict(std=0),dict(mean=-1),dict(low=-.1),dict(high=float('nan'))]:
            with self.assertRaises(ValueError): plan(**params)

    def test_audit_rejects_empty_constant_or_out_of_bounds_noise(self):
        config=plan()['distribution']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'contacts.csv'
            def write(values):
                with path.open('w') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['contact','x','y','z','original_slip1','original_slip2','applied_slip'])
                    for i,value in enumerate(values,1): writer.writerow([i,0,0,0,.005,.005,value])
            for values in ([],[.1,.1],[.1,.5]):
                write(values)
                with self.assertRaises(ValueError): audit_trace(path,'gaussian',config)
            write([.08,.12])
            self.assertAlmostEqual(audit_trace(path,'gaussian',config)['mean'],.1)
            with self.assertRaises(ValueError): audit_trace(path,'constant',config)
            write([.1,.1])
            self.assertEqual(audit_trace(path,'constant',config)['variance'],0.)

    def test_startup_requires_hook_and_scene_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('contact_slip.run.require_free_ports'), \
             patch('contact_slip.run.prepare',return_value={'startup_marker':'scene marker'}), \
             patch('contact_slip.run.digest',return_value='same'), \
             patch('contact_slip.run.hook_env',return_value={}), \
             patch('contact_slip.run.subprocess.Popen') as process, \
             patch('contact_slip.run.time.sleep'):
            process.return_value.poll.return_value=None
            with self.assertRaisesRegex(RuntimeError,'startup verification'):
                with server(Path(tmp),'gaussian',plan()['distribution'],42,Path(tmp)/'hook.so'):
                    self.fail('Must not run an unverified client')
            process.return_value.terminate.assert_called_once()
            process.return_value.wait.assert_called_once_with(timeout=10)


if __name__=='__main__': unittest.main()

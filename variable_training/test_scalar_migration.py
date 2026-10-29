import tempfile
import unittest
from pathlib import Path

from configure_field_slip import apply, read_slip, status
from variable_training.plan import make_plan, validate_plan
from variable_training.common import PROTOCOL, save, commit, verified_evaluation
from run_varied_slip import OUT, PROTOCOL as FIXED_PROTOCOL, FIXED, SEEDS


class MigrationTests(unittest.TestCase):
    def test_wide_range_and_three_evaluation_seeds(self):
        p=make_plan()
        self.assertEqual(p['distribution']['high'],10.)
        self.assertEqual(len([r for r in p['evaluation'] if r['id'].startswith('sampled_')]),3)
        self.assertEqual(SEEDS,[101,202,303])
        self.assertEqual(list(FIXED.values()),[.025,.1,.2,1.,10.])
        self.assertEqual(len(make_plan(quick=True)['evaluation']),4)

    def test_scalar_write_and_legacy_read_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scene=root/'scene'; backup=root/'backup'
            original='(setContactSlipMode true)\n(setContactSlip 0.1 0.1)\n'
            scene.write_text(original)
            for value in (.025,.1,.2,1.,10.):
                apply(scene,value,backup)
                self.assertEqual(scene.read_text(),f'(setContactSlipMode true)\n(setContactSlip {value:.12g})\n')
                self.assertEqual(status(scene)['scene_setter_method'],'single_argument_v1')
                self.assertEqual(read_slip(scene)[2],value)
            self.assertEqual(backup.read_text(),original)
            apply(scene,.1,backup,setter_method='legacy')
            self.assertEqual(scene.read_text(),original)
            for value in (float('nan'),float('inf'),-.1):
                with self.assertRaises(ValueError): apply(scene,value,backup)

    def test_legacy_plan_and_receipt_cannot_resume_new_protocol(self):
        p=make_plan(); self.assertEqual(p['protocol'],PROTOCOL)
        self.assertEqual(p['scene_setter_method'],'single_argument_v1')
        old=dict(p,protocol='variable_slip_hidden_v1')
        with self.assertRaises(ValueError): validate_plan(old)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); save(root/'result.json',{})
            commit(root,dict(protocol='variable_slip_hidden_v1'))
            with self.assertRaises(ValueError): verified_evaluation(root,dict(protocol=PROTOCOL))
        self.assertEqual(OUT.name,'scalar_wide_v4')
        self.assertEqual(FIXED_PROTOCOL,'isolated_server_scalar_wide_slip_v4')

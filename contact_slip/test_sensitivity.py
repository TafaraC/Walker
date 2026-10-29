import csv
import tempfile
import unittest
from pathlib import Path

from contact_slip.sensitivity import audit, plan, report
from variable_training.common import commit, save, read


class SensitivityTests(unittest.TestCase):
    def test_crossed_plan(self):
        p=plan()
        self.assertEqual(len(p['slips'])*len(p['seeds'])*len(p['policies']),30)
        self.assertEqual(p['steps'],3000)
        self.assertEqual(p['slips'],[.025,.1,.2])

    def test_stress_plan_is_oracle_only(self):
        p=plan('stress')
        self.assertEqual(p['policies'],{'oracle':None})
        self.assertEqual(p['slips'],[.1,.5,1.,2.,5.,10.])
        self.assertEqual(len(p['slips'])*len(p['seeds']),30)
        self.assertEqual(p['steps'],3000)
        with self.assertRaises(ValueError): plan('unknown')

    def test_smoke_report_warns_on_unchanged_classes_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); settings=plan('stress')
            for slip in settings['slips']:
                directory=root/'smoke'/str(slip)/'oracle_101'
                save(directory/'raw/oracle_seed_101.json',dict(return_target_progress_m=44.,falls=0))
                (directory/'contacts.csv').write_text(
                    'contact,original_slip1,original_slip2,applied_slip,original_mode,applied_mode,slip1_enabled,slip2_enabled\n'
                    '1,0.005,0.005,0.005,768,768,1,1\n')
                commit(directory,dict(plan=settings))
            report(root,settings,smoke=True)
            self.assertTrue(read(root/'smoke_report.json')['identical_recorded_classes'])
            self.assertIn('WARNING',(root/'smoke_report.md').read_text())
            (directory/'contacts.csv').write_text('changed')
            with self.assertRaises(ValueError): report(root,settings,smoke=True)

    def test_passive_flags_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.csv'
            def write(mode=768, applied_mode=768, applied=.005, bit1=1, bit2=1):
                with path.open('w') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['contact','original_slip1','original_slip2','applied_slip',
                                     'original_mode','applied_mode','slip1_enabled','slip2_enabled'])
                    writer.writerow([1,.005,.007,applied,mode,applied_mode,bit1,bit2])
            write()
            result=audit(path)
            self.assertEqual(result['axes'][1]['mean'],.007)
            self.assertEqual(result['contact_classes'][0]['count'],1)
            write(mode=0,applied_mode=0,bit1=0,bit2=0)
            self.assertEqual(audit(path)['axes'][0]['mean'],0)
            for kwargs in [dict(applied=.1),dict(applied_mode=0),dict(bit1=0)]:
                write(**kwargs)
                with self.assertRaises(ValueError): audit(path)
            path.write_text('contact,original_mode,applied_mode\n')
            with self.assertRaises(ValueError): audit(path)


if __name__=='__main__': unittest.main()

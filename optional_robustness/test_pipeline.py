import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from optional_robustness.run import plan, summarize
from optional_robustness.run import main, write_report
from variable_training.common import save, read


class PlanTests(unittest.TestCase):
    def test_default_design(self):
        p=plan([.1,1,5,10,20,50,100],[101,202,303],3000)
        self.assertEqual(len(p['slips'])*len(p['seeds']),21)
        self.assertEqual(p['scene_method'],'scalar')
        self.assertEqual(p['policies'],{'oracle':None})

    def test_invalid_settings(self):
        for slips,seeds,steps in [([1],[1],3000),([.1,.1],[1],3000),
                                   ([.1,float('nan')],[1],3000),([.1,-1],[1],3000),
                                   ([.1],[1,1],3000),([.1],[-1],3000),([.1],[1],0)]:
            with self.assertRaises(ValueError):plan(slips,seeds,steps)

    def test_absolute_failure_and_paired_retention(self):
        def row(slip,seed,progress,falls):
            return dict(slip=slip,seed=seed,metrics=dict(return_target_progress_m=progress,
                falls=falls,active_seconds=60,mean_forward_velocity_mps=progress/60,
                mean_target_direction_error=.5,targets_reached=0))
        rows=[row(.1,1,40,0),row(.1,2,20,0),row(100,1,0,0),row(100,2,-2,2)]
        g=summarize(rows,[.1,100])[1]
        self.assertEqual(g['progress_mean_m'],-1)
        self.assertEqual(g['paired_retention_mean'],-.05)
        self.assertEqual(g['falls_per_active_minute'],1)

    def test_zero_reference_not_divided(self):
        metrics=dict(return_target_progress_m=0,falls=0,active_seconds=60,
                     mean_forward_velocity_mps=0,mean_target_direction_error=1,targets_reached=0)
        g=summarize([dict(slip=.1,seed=1,metrics=metrics)],[.1])[0]
        self.assertIsNone(g['paired_retention_mean'])

    def test_plan_only_and_changed_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=['runner','plan','--output',tmp]
            with patch('sys.argv',args), patch('optional_robustness.run.evaluate') as evaluate:
                main()
                evaluate.assert_not_called()
            self.assertEqual(read(Path(tmp)/'plan.json')['steps'],3000)
            with patch('sys.argv',args+['--steps','100']), self.assertRaises(ValueError):
                main()

    def test_interpretation_report_from_verified_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            metrics=dict(return_target_progress_m=40,falls=0,active_seconds=60,
                         mean_forward_velocity_mps=.7,mean_target_direction_error=.1,targets_reached=5)
            save(root/'smoke_report.json',dict(runs=[dict(slip=.1,seed=101,metrics=metrics)],
                                              identical_recorded_classes=True))
            with patch('optional_robustness.run.report') as verify_report:
                write_report(root,plan([.1],[101],3000),True)
                verify_report.assert_called_once()
            self.assertIn('WARNING',(root/'smoke_interpretation.md').read_text())
            self.assertEqual(read(root/'smoke_interpretation.json')['groups'][0]['paired_retention_mean'],1)


if __name__=='__main__':unittest.main()

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from variable_training.common import MODELS, save, digest
from variable_training.plan import make_plan, TRAINING_SLIPS
from variable_training.run import Experiment


class FixedGridTests(unittest.TestCase):
    def test_training_grid_is_independent_of_gaussian_parameters(self):
        a=make_plan(); b=make_plan(mean=2.,std=1.)
        self.assertEqual(a['groups'],b['groups'])
        self.assertNotEqual(a['evaluation'],b['evaluation'])
        for name in ('train','validation','test'):
            self.assertEqual({r['slip'] for r in a['groups'][name]},set(TRAINING_SLIPS))
        self.assertEqual(sum(r['steps'] for r in a['groups']['validation']),10000)
        self.assertEqual(sum(r['steps'] for r in a['groups']['test']),10000)
        for rows in a['groups'].values():
            for long,short in zip(rows[::2],rows[1::2]):
                self.assertEqual(long['slip'],short['slip'])
                self.assertIn(long['slip'],TRAINING_SLIPS)

    def test_evaluation_pairs_models_with_same_draw_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Experiment.__new__(Experiment)
            exp.root=Path(tmp); exp.identity={}; exp.plan=make_plan()
            exp.plan_path=exp.root/'plan.json'; save(exp.plan_path,exp.plan)
            checkpoint=exp.root/'model'; checkpoint.write_bytes(b'offline stub')
            for label in MODELS:
                save(exp.root/'policies'/label/'varied/final.json',
                     dict(model=str(checkpoint),sha256=digest(checkpoint),plan=digest(exp.plan_path)))
            calls=[]; current=[]
            @contextmanager
            def fake_server(directory,slip):
                current[:] = [slip]
                yield {'server_process':None}
            def fake_launch(cmd,*args,**kwargs):
                option=lambda key: cmd[cmd.index(key)+1]
                label=option('--label'); seed=int(option('--seed')); steps=int(option('--steps'))
                slip=float(option('--field-slip')); self.assertEqual(slip,current[0])
                calls.append((label,seed,slip))
                out=Path(option('--output-dir')); out.mkdir(parents=True)
                trace=out/'trace.npz'; np.savez(trace,obs=np.zeros((steps,63)))
                save(out/f'{label}_seed_{seed}.json',dict(active_steps=steps,field_contact_slip=slip,
                                                       seed=seed,trace=str(trace)))
            with patch('variable_training.run.simulator',fake_server), patch('variable_training.run.launch',fake_launch):
                exp.evaluate(['varied'])
            self.assertEqual(len(calls),15)
            for start in range(0,15,5):
                block=calls[start:start+5]
                self.assertEqual(len({(seed,slip) for _,seed,slip in block}),1)
                self.assertEqual({label for label,_,_ in block},{'oracle',*[k+'_varied' for k in MODELS]})
            self.assertFalse(any('frozen' in label for label,_,_ in calls))

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from probing.run import actor, features, sample_trace, shuffle_blocks, fit_probe, metrics, run
from variable_training.common import save, read


class ProbeTests(unittest.TestCase):
    def test_actor_loading_and_frozen_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            net=actor(seed=11)
            x=np.random.default_rng(1).normal(size=(20,63))
            for prefix in ('net.','actor_mean.'):
                path=Path(tmp)/'policy.pt'
                torch.save({prefix+k:v for k,v in net.state_dict().items()},path)
                loaded=actor(path)
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                np.testing.assert_array_equal(features(net,x)[1],features(loaded,x)[1])
                with torch.no_grad():
                    np.testing.assert_allclose(loaded[4](torch.tensor(features(loaded,x)[1],dtype=torch.float32)).numpy(),
                                               loaded(torch.tensor(x,dtype=torch.float32)).numpy())

    def test_windows_do_not_cross_reset(self):
        n=50
        trace=dict(obs=np.zeros((n,63)),episode_id=np.repeat([0,1],25),
                   head_height_m=np.ones(n),torso_roll_deg=np.zeros(n),
                   torso_pitch_deg=np.zeros(n),forward_velocity_mps=np.arange(n,dtype=float))
        x,y,idx=sample_trace(trace,.1,1)
        self.assertTrue(all(i<15 or i>=25 for i in idx))
        self.assertEqual(y[0,-1],4.5)
        self.assertTrue((idx+10<n).all())
        trace['obs'][0,0]=np.nan
        with self.assertRaises(ValueError):sample_trace(trace,.1,1)

    def test_block_control(self):
        groups=np.repeat(np.arange(5),4)
        y=np.repeat(np.arange(5),4)[:,None].astype(float)
        shuffled=shuffle_blocks(y,groups,42)
        self.assertTrue(np.all(shuffled!=y))
        self.assertEqual(set(shuffled[:,0]),set(y[:,0]))
        np.testing.assert_array_equal(shuffled,shuffle_blocks(y,groups,42))
        for g in range(5):self.assertEqual(len(np.unique(shuffled[groups==g])),1)

    def test_linear_signal_and_no_test_selection(self):
        rng=np.random.default_rng(7)
        xs={s:rng.normal(size=(80,4)) for s in ('train','validation','test')}
        w=rng.normal(size=(4,5))
        ys={s:x@w+2 for s,x in xs.items()}
        groups={s:np.repeat(np.arange(4),20) for s in xs}
        pred,fit=fit_probe(xs,ys,groups,[.000001,1.])
        self.assertLess(np.square(pred-ys['test']).mean(),1e-8)
        altered={**ys,'test':ys['test']+10000}
        _,other=fit_probe(xs,altered,groups,[.000001,1.])
        np.testing.assert_array_equal(fit['weights'],other['weights'])
        np.testing.assert_allclose(fit['feature_mean'],xs['train'].mean(0))
        scored=metrics(ys['test'],pred,ys['train'],groups['test'])
        self.assertGreater(scored[0]['r2'],.999)

    def test_synthetic_end_to_end_resume_and_integrity(self):
        # Synthetic engineering test, not a research run; use a tiny feature net.
        from unittest.mock import patch
        rng=np.random.default_rng(3)
        data={s:(rng.normal(size=(60,4)),rng.normal(size=(60,5)),
                 np.repeat(np.arange(3),20),np.tile(np.arange(20),3))
              for s in ('train','validation','test')}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            spec=dict(protocol='test',models={},control_seeds=[],alphas=[.1,1.])
            save(output/'plan.json',spec)
            with patch('probing.run.dataset',return_value=data), patch('probing.run.representations',
                    side_effect=lambda *_:iter([('raw_observation',{s:d[0] for s,d in data.items()})])):
                run(output,spec)
                self.assertEqual(len(read(output/'report.json')['results']),5)
                run(output,spec)
                (output/'probes/raw_observation/true/metrics.json').write_text('[]')
                with self.assertRaises(ValueError):run(output,spec)

    def test_constant_columns_and_undefined_r2(self):
        xs={s:np.ones((20,3)) for s in ('train','validation','test')}
        ys={s:np.ones((20,5)) for s in xs}
        groups={s:np.repeat([0,1],10) for s in xs}
        pred,fit=fit_probe(xs,ys,groups,[.1,1.])
        np.testing.assert_array_equal(pred,ys['test'])
        self.assertTrue(np.isfinite(fit['weights']).all())
        self.assertIsNone(metrics(ys['test'],pred,ys['train'],groups['test'])[0]['r2'])


if __name__=='__main__':
    unittest.main()

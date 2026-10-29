import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from math_ops.Neural_Network import run_mlp
from representation_probing.expert import load_expert,expert_features,random_expert,cka,spectrum,comparisons,run
from variable_training.common import save,read


class ExpertTests(unittest.TestCase):
    def test_native_equivalence(self):
        w=load_expert();x=np.random.default_rng(3).normal(size=(40,63)).astype(np.float32)
        h,y=expert_features(x,w)
        self.assertEqual(h.shape,(40,64))
        np.testing.assert_allclose(y,np.stack([run_mlp(v,w) for v in x]),rtol=1e-4,atol=1e-5)

    def test_cka_and_spectrum(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(200,12))
        q=np.linalg.qr(rng.normal(size=(12,12)))[0]
        self.assertAlmostEqual(cka(x,3*x@q+7),1.)
        self.assertLess(cka(x,x[rng.permutation(len(x))]),.3)
        self.assertIsNone(cka(x,np.ones((200,2))))
        self.assertIsNone(spectrum(np.ones((20,3)))['participation_ratio'])

    def test_random_control_and_ratios(self):
        x=np.ones((10,63));np.testing.assert_array_equal(random_expert(x,42),random_expert(x,42))
        self.assertFalse(np.array_equal(random_expert(x,42),random_expert(x,43)))
        base=dict(target='test',labels='true',r2=None,rmse=0.,macro_run_rmse=0.)
        result=comparisons([dict(base,representation='expert_h1'),dict(base,representation='bc_h1')])[0]
        self.assertIsNone(result['r2_gap']);self.assertIsNone(result['rmse_ratio_to_expert'])

    def test_synthetic_append_resume_corruption(self):
        rng=np.random.default_rng(6)
        data={s:(rng.normal(size=(40,63)),rng.normal(size=(40,2)),np.repeat([0,1],20),np.arange(40),np.zeros((40,2)))
              for s in ['train','validation','test']}
        spec=dict(new_representations=['expert_h1'],control_seeds=[],alphas=[.001,.01],target_names=['a','b'])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);save(output/'plan.json',spec)
            with patch('representation_probing.expert.geometry',return_value=[]):
                run(output,spec,{},data,[],load_expert())
                run(output,spec,{},data,[],load_expert())
            self.assertEqual(len(read(output/'report.json')['results']),2)
            (output/'probes/expert_h1/true/metrics.json').write_text('[]')
            with self.assertRaises(ValueError):run(output,spec,{},data,[],load_expert())


if __name__=='__main__':unittest.main()

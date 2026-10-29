import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .run import inputs, random_actor, execute
from representation_probing.expert import load_expert
from variable_training.common import save, read


class PipelineTests(unittest.TestCase):
    def test_context_inputs(self):
        x=np.zeros((4,63))
        d={'test':(x,None,np.array([0,0,1,1]),None,None)}
        spec={'entries':[{'row':{'slip':.1}},{'row':{'slip':10.}}]}
        result=inputs(d,spec,'input')['test']
        self.assertEqual(result.shape,(4,64))
        np.testing.assert_array_equal(result[:,:63],x)
        np.testing.assert_allclose(result[:,63],[np.log1p(.1)/np.log(11)]*2+[1.]*2)
        self.assertEqual(inputs(d,spec,'hidden')['test'].shape,(4,63))

    def test_random_controls(self):
        for width in (63,64):
            a=random_actor(width,42);b=random_actor(width,42)
            self.assertEqual(a[0].in_features,width)
            for x,y in zip(a.parameters(),b.parameters()):
                self.assertFalse(x.requires_grad)
                np.testing.assert_array_equal(x.numpy(),y.numpy())

    def test_regression_report_resume_and_tamper(self):
        rng=np.random.default_rng(8)
        data={}
        for s in ('train','validation','test'):
            x=rng.normal(size=(24,63));y=x[:,:6]
            data[s]=(x,y,np.repeat([0,1,2],8),np.arange(24),y.copy())
        spec={'models':{'sample':{}},'alphas':[.01,1.],
              'target_names':[f'target_{i}' for i in range(6)]}
        reps=[('expert_h1',{s:d[0][:,:4] for s,d in data.items()}),
              ('sample_h1',{s:d[0][:,:6] for s,d in data.items()})]
        with tempfile.TemporaryDirectory() as tmp, patch('consolidated_probing.run.representations',return_value=reps):
            out=Path(tmp);save(out/'plan.json',spec)
            execute(out,spec,data,load_expert())
            report=read(out/'report.json')
            self.assertEqual(len(report['results']),48)
            self.assertEqual(len(report['expert_comparison']),6)
            execute(out,spec,data,load_expert(),True)
            metrics=out/'probes/sample_h1/true/metrics.json'
            metrics.write_text('[]')
            with self.assertRaises(ValueError):execute(out,spec,data,load_expert(),True)


if __name__=='__main__':unittest.main()

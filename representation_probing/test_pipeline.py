import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from probing.run import actor
from representation_probing.run import targets,reset_layer,scores,run
from variable_training.common import save,read


def trace():
    n=80;o=np.zeros((n,63));o[:,0]=.2;o[0,0]=0
    o[:,55]=np.tile(np.arange(8)/7,10);o[:,56]=1
    o[:,1]=.9;o[:,3]=2;o[:,5]=.4;o[:,23]=.8;o[:,14]=.2
    return dict(obs=o,episode_id=np.repeat([0,1],40),forward_velocity_mps=np.arange(n,dtype=float),
                target_direction_error=np.arange(n,dtype=float)/100)


class Tests(unittest.TestCase):
    def test_targets_and_boundaries(self):
        x,y,idx,names,p=targets(trace(),stride=1)
        self.assertEqual(len(names),30)
        self.assertNotIn(0,idx);self.assertNotIn(40,idx)
        self.assertFalse(any(30<=i<40 for i in idx))
        self.assertEqual(y[0,names.index('torso_roll_deg')],30)
        self.assertEqual(y[0,names.index('left_force_contact_gt10N')],1)
        self.assertEqual(y[0,names.index('left_ankle_hip_x_m')],.1)
        self.assertEqual(y[0,names.index('future_200ms_forward_velocity_mps')],5.5)
        self.assertEqual(p[0,0],0)

    def test_reset_preserves_original_and_other_layers(self):
        net=actor(seed=17);before={k:v.clone() for k,v in net.state_dict().items()}
        for kind,index in [('reset_input',0),('reset_middle',2),('reset_output',4)]:
            changed=reset_layer(net,kind,42)
            for k,v in net.state_dict().items():
                self.assertTrue(torch.equal(v,before[k]))
                self.assertEqual(torch.equal(changed.state_dict()[k],v),not k.startswith(f'{index}.'))
            self.assertFalse(any(p.requires_grad for p in changed.parameters()))

    def test_binary_constant_class(self):
        y=np.ones((20,1));g=np.repeat([0,1],10)
        r=scores(y,y,y,g,['left_force_contact_gt10N'])[0]
        self.assertEqual(r['contact_accuracy'],1)
        self.assertIsNone(r['contact_balanced_accuracy']);self.assertIsNone(r['r2'])

    def test_end_to_end_synthetic(self):
        from representation_probing.run import representations
        rng=np.random.default_rng(1)
        data={s:(rng.normal(size=(30,63)),rng.normal(size=(30,6)),
                 np.repeat([0,1,2],10),np.arange(30),rng.normal(size=(30,6))) for s in ('train','validation','test')}
        spec=dict(models={},control_seeds=[],alphas=[.001,.01],target_names=[f'target{i}' for i in range(6)])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);save(output/'plan.json',spec)
            with patch('representation_probing.run.interventions',return_value=[]):
                run(output,spec,data);run(output,spec,data)
            self.assertEqual(len(read(output/'report.json')['results']),12)
            (output/'probes/raw_observation/true/metrics.json').write_text('[]')
            with self.assertRaises(ValueError):run(output,spec,data)


if __name__=='__main__':unittest.main()

"""Synthetic offline tests for repeatability and FiLM; never starts a server."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train import BCPolicy
from context_training.test_pipeline import fixture
from context_training.data import audit,load_split
from context_training.test_evaluation import fake_spec,fake_result
from context_training.evaluate import rounds,job_identity as old_identity
from context_training.repeatability import make_repeat_spec,summarize
from variable_training.common import save,read,commit,digest
from .models import initialize,predict,policy_inputs,checkpoint,load_checkpoint
from .run import train_one
from .evaluate import make_spec,job_identity,report,evaluate


class FilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_and_context_gradients(self):
        torch.manual_seed(42)
        old=BCPolicy(63,16).eval()
        model=initialize(old.state_dict(),'film')
        x=np.random.default_rng(1).normal(size=(8,63)).astype(np.float32)
        expected=old(torch.from_numpy(x)).detach().numpy()
        for slip in [.025,.1,10.]:
            np.testing.assert_allclose(predict(model,'film',x,slip),expected,atol=1e-7)
        model(torch.from_numpy(policy_inputs(x,10.,'film'))).square().mean().backward()
        self.assertGreater(float(model.modulation.weight.grad.abs().sum()),0)
        self.assertEqual(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in old.parameters()),1024)

    def test_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.pt'
            m=initialize(BCPolicy(63,16).state_dict(),'film')
            torch.save(checkpoint(m,'film'),path)
            restored,a=load_checkpoint(path)
            self.assertEqual(a,'film')
            obs=np.zeros(63,np.float32)
            np.testing.assert_array_equal(predict(m,a,obs,.1),predict(restored,a,obs,.1))
            payload=checkpoint(m,'film');payload['film']['location']='wrong'
            torch.save(payload,path)
            with self.assertRaises(ValueError):load_checkpoint(path)

    def test_synthetic_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=audit(fixture(root/'data'))
            splits={k:load_split(v) for k,v in data['splits'].items()}
            initial=root/'initial.pth';torch.save(BCPolicy(63,16).state_dict(),initial)
            output=root/'update';output.mkdir()
            m=train_one(output,'film',42,dict(epochs=2,batch_size=16,learning_rate=1e-4),initial,splits,torch.device('cpu'))
            self.assertTrue(np.isfinite(m['test']['element_mse']))
            self.assertEqual(load_checkpoint(output/'policy.pt')[1],'film')

    def test_repeat_schedule_and_diagnostics(self):
        base=fake_spec();base['rounds']=rounds(42,'all');base['controllers']=base['controllers'][:3]
        with patch('context_training.repeatability.make_spec',return_value=base):
            spec=make_repeat_spec(Path('/unused'),repeats=2)
        self.assertEqual(len(spec['rounds'])*len(spec['controllers']),18)
        for case in ['low','nominal','high']:
            rs=[r for r in spec['rounds'] if r['case']==case]
            self.assertEqual(rs[0]['seed'],rs[1]['seed'])
            self.assertEqual(rs[0]['slip'],rs[1]['slip'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for row in spec['rounds']:
                for c in spec['controllers']:
                    d=root/'runs'/row['id']/c['label'];d.mkdir(parents=True)
                    identity=old_identity(spec,row,c);fake_result(d,identity)
                    with np.load(d/'trace.npz') as data:arrays={k:data[k] for k in data.files}
                    arrays.update(positions=np.zeros((spec['steps'],2)),targets=np.ones((spec['steps'],2)),
                        head_height_m=np.ones(spec['steps']),torso_roll_deg=np.zeros(spec['steps']),torso_pitch_deg=np.zeros(spec['steps']))
                    np.savez(d/'trace.npz',**arrays);commit(d,identity)
            summarize(root,spec)
            results=read(root/'repeatability.json')
            self.assertEqual(len(results['summary']),9)
            self.assertTrue(all(s['std_progress_m']==0 for s in results['summary']))

    def test_append_only_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);controls=root/'controls';film=root/'film';baseline=controls/'evaluation'
            common=dict(dataset={},initializer_sha256='same',settings={},seeds=[42],device='cpu',torch_version='test',numpy_version='test')
            save(controls/'plan.json',dict(common,architectures=['hidden','input']))
            save(film/'plan.json',dict(common,protocol='film_bc_pilot_v1',architectures=['film']))
            previous=fake_spec();previous.update(protocol='context_bc_evaluation_v1',training_plan_sha256=digest(controls/'plan.json'))
            save(baseline/'plan.json',previous)
            for row in previous['rounds']:
                for c in previous['controllers']:
                    d=baseline/'runs'/row['id']/c['label'];d.mkdir(parents=True)
                    identity=old_identity(previous,row,c);fake_result(d,identity);commit(d,identity)
            d=film/'film'/'seed_42';d.mkdir(parents=True)
            torch.save(checkpoint(initialize(BCPolicy(63,16).state_dict(),'film'),'film'),d/'policy.pt')
            commit(d,dict(plan_sha256=digest(film/'plan.json'),architecture='film',seed=42))
            spec=make_spec(film,baseline,'all',['correct','wrong'])
            self.assertEqual({c['architecture'] for c in spec['controllers']},{'film'})
            self.assertEqual(spec['rounds'],previous['rounds'])
            output=root/'output'
            for row in spec['rounds']:
                for c in spec['controllers']:
                    d=output/'runs'/row['id']/c['label'];d.mkdir(parents=True)
                    identity=job_identity(spec,row,c);fake_result(d,identity);commit(d,identity)
            with patch('film_training.evaluate.simulator',side_effect=AssertionError('must resume')):
                evaluate(output,spec,1)
            report(output,spec)
            result=read(output/'comparison.json')
            self.assertEqual(len(result['runs']),5) # 3 reused controls + 2 FiLM modes
            self.assertEqual(len(result['paired_differences']),3)
            plan=read(film/'plan.json');plan['device']='different';save(film/'plan.json',plan)
            with self.assertRaisesRegex(ValueError,'training mismatch'):
                make_spec(film,baseline,'all',['correct'])


if __name__=='__main__':unittest.main()

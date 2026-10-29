"""Reuse existing networks and context encoding; checkpoint identity lives in manifests."""
import numpy as np
import torch
from train import BCPolicy
from gail_models import GAILActorCritic
from context_training.models import policy_inputs


def load_policy(path, kind, architecture):
    dim = 64 if architecture == 'input' else 63
    model = GAILActorCritic(dim, 16) if kind == 'gail' else BCPolicy(dim, 16)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite policy')
    return model.eval()


def predict(model, architecture, obs, slip):
    x = policy_inputs(obs, .1 if slip is None else slip, architecture)
    with torch.no_grad():
        tensor = torch.from_numpy(x)
        output = model.actor_mean(tensor) if isinstance(model, GAILActorCritic) else model(tensor)
    return output.cpu().numpy()


def initialize(job, directory):
    torch.manual_seed(job['seed'])
    dim = 64 if job['architecture'] == 'input' else 63
    kind = job['kind']
    model = GAILActorCritic(dim, 16) if kind == 'gail' else BCPolicy(dim, 16)
    if job.get('model'):
        state = torch.load(job['model'], map_location='cpu', weights_only=True)
        if kind == 'gail':
            state = {k.replace('net.', 'actor_mean.', 1): v for k, v in state.items()}
            target = model.state_dict()
            target.update(state)
            state = target
        key = 'actor_mean.0.weight' if kind == 'gail' else 'net.0.weight'
        if state[key].shape[1] == 63 and dim == 64:
            state[key] = torch.cat((state[key], torch.zeros(256, 1)), dim=1)
        model.load_state_dict(state, strict=True)
    elif dim == 64:
        raise ValueError('Input conditioning requires its specified pretrained initializer')
    torch.save(model.state_dict(), directory/'policy.pth')

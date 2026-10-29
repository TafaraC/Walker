"""BC context contract; no simulator imports and no change to legacy checkpoints."""
import math

import numpy as np
import torch

from train import BCPolicy


SCHEMA = 'context_bc_v1'
CONTEXT = {'name': 'field_contact_slip', 'encoding': 'log1p(slip)/log(11)',
           'minimum_exclusive': 0.0, 'maximum': 10.0}
ARCHITECTURES = ('hidden', 'input')


def encode_slip(slip):
    values = np.asarray(slip, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values > 10):
        raise ValueError('Context slip must be finite and in (0, 10]; no clipping')
    return (np.log1p(values) / math.log(11)).astype(np.float32)


def policy_inputs(observations, slip, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {architecture}')
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim not in (1, 2) or obs.shape[-1] != 63 or not np.isfinite(obs).all():
        raise ValueError('Expected finite 63-D observations or a batch thereof')
    context = np.broadcast_to(encode_slip(slip), obs.shape[:-1])
    if architecture == 'hidden':
        return obs.copy()
    return np.concatenate((obs, context[..., None]), axis=-1)


def initialize(state, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {architecture}')
    original = BCPolicy(63, 16)
    original.load_state_dict(state, strict=True)
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError('Non-finite initialization')
    model = BCPolicy(64 if architecture == 'input' else 63, 16)
    weights = {key: value.detach().clone() for key, value in state.items()}
    if architecture == 'input':
        weights['net.0.weight'] = torch.cat(
            (weights['net.0.weight'], weights['net.0.weight'].new_zeros((256, 1))), dim=1)
    model.load_state_dict(weights, strict=True)
    return model


def checkpoint(model, architecture):
    return {'schema': SCHEMA, 'architecture': architecture, 'context': dict(CONTEXT),
            'observation_dim': 63, 'action_dim': 16,
            'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != SCHEMA or payload.get('context') != CONTEXT or
            payload.get('architecture') not in ARCHITECTURES or
            payload.get('observation_dim') != 63 or payload.get('action_dim') != 16):
        raise ValueError('Incompatible context checkpoint metadata')
    architecture = payload['architecture']
    model = BCPolicy(64 if architecture == 'input' else 63, 16)
    model.load_state_dict(payload['state_dict'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Non-finite checkpoint')
    return model.eval(), architecture


def predict(model, architecture, observations, supplied_slip):
    """Explicit context argument keeps actual environment slip separate from ablations."""
    values = policy_inputs(observations, supplied_slip, architecture)
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.from_numpy(values).to(device)).cpu().numpy()

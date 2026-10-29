"""One affine FiLM modulation at the final 256-dimensional hidden representation."""
import torch
from torch import nn

from train import BCPolicy
from context_training import models as legacy

CONTEXT = legacy.CONTEXT
ARCHITECTURES = ('hidden', 'input', 'film')
FILM = {'location': 'after_second_relu', 'features': 256,
        'conditioner': 'linear_1_to_512', 'scale': '1+delta_gamma', 'initialization': 'identity'}


class FiLMPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = BCPolicy(63,16).net
        self.modulation = nn.Linear(1,512)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, packed):
        # Packed transport is 64-D; only 63 state features enter the backbone.
        features = self.net[:4](packed[..., :63])
        delta_gamma,beta = self.modulation(packed[...,63:64]).chunk(2,dim=-1)
        return self.net[4]((1+delta_gamma)*features+beta)


def policy_inputs(observations, slip, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown architecture')
    return legacy.policy_inputs(observations,slip,'input' if architecture=='film' else architecture)


def initialize(state, architecture):
    if architecture != 'film':
        return legacy.initialize(state,architecture)
    original=legacy.initialize(state,'hidden')  # Strict shape/finite validation.
    model=FiLMPolicy()
    model.net.load_state_dict(original.net.state_dict(),strict=True)
    return model


def checkpoint(model, architecture):
    if architecture != 'film':
        return legacy.checkpoint(model,architecture)
    return dict(schema='film_bc_v1',architecture='film',context=dict(CONTEXT),film=dict(FILM),
                observation_dim=63,action_dim=16,
                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})


def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')=='context_bc_v1':
        return legacy.load_checkpoint(path)
    if (payload.get('schema')!='film_bc_v1' or payload.get('architecture')!='film' or
        payload.get('context')!=CONTEXT or payload.get('film')!=FILM or
        payload.get('observation_dim')!=63 or payload.get('action_dim')!=16):
        raise ValueError('Incompatible FiLM metadata')
    model=FiLMPolicy()
    model.load_state_dict(payload['state_dict'],strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Non-finite FiLM checkpoint')
    return model.eval(),'film'


def predict(model, architecture, observations, supplied_slip):
    values=policy_inputs(observations,supplied_slip,architecture)
    with torch.no_grad():
        return model(torch.from_numpy(values).to(next(model.parameters()).device)).cpu().numpy()

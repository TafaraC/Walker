"""Immutable paired plans; no simulator or native-module imports."""
import math
import numpy as np
from .common import PROTOCOL


def truncated_draws(rng, count, low, high, mean, std):
    if not all(math.isfinite(x) for x in (low, high, mean, std)) or not (0 <= low < high and std > 0):
        raise ValueError('Finite nonnegative bounds and positive Gaussian std required')
    # Keep expected rejection cost reasonable for an interactive plan command.
    a, b = (low - mean) / std, (high - mean) / std
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))
    z = cdf(b) - cdf(a)
    if z < 1e-4:
        raise ValueError('Gaussian places too little mass within bounds')
    result = []
    while len(result) < count:
        draws = rng.normal(mean, std, max(64, 2 * (count - len(result))))
        result.extend(float(x) for x in draws if low <= x <= high)
    return result[:count]


TRAINING_SLIPS = [.025, .1, .2, 1., 10.]


def make_plan(seed=42, low=.025, high=10., mean=5.0125, std=2.5, quick=False):
    rng = np.random.default_rng(seed)
    draw = lambda: truncated_draws(rng, 1, low, high, mean, std)[0]
    groups = {}
    counter = 0
    pair_counter = 0
    def pair_group(name, pairs, steps):
        nonlocal counter, pair_counter
        rows = []
        for pair in range(pairs):
            slip = TRAINING_SLIPS[pair_counter % len(TRAINING_SLIPS)]
            pair_counter += 1
            for mode in ('long', 'short'):
                counter += 1
                rows.append(dict(id=f'{name}_{pair:03d}_{mode}', slip=slip, mode=mode,
                                 steps=steps, seed=seed * 100000 + counter))
        groups[name] = rows
    pair_group('train', 1 if quick else 10, 32 if quick else 5000)
    pair_group('validation', 1 if quick else 5, 32 if quick else 1000)
    pair_group('test', 1 if quick else 5, 32 if quick else 1000)
    for iteration in range(1, (1 if quick else 7) + 1):
        pair_group(f'dagger_{iteration}', 1, 32 if quick else 5000)
    for iteration in range(1, (1 if quick else 10) + 1):
        # Each long/short pair shares slip, producing matched expert/learner mixtures.
        pair_group(f'gail_{iteration}', 1 if quick else 2, 32 if quick else 1000)
    evaluations = [dict(id=f'sampled_{i}', slip=draw(), seed=seed * 100000 + 90000 + i)
                   for i in range(1 if quick else 3)]
    for name, value in [('nominal', .1), ('high_traction', low), ('low_traction', high)]:
        evaluations += [dict(id=f'{name}_{i}', slip=value, seed=seed * 100000 + 91000 + i)
                        for i in range(1 if quick else 3)]
    a, b = (low - mean) / std, (high - mean) / std
    phi = lambda x: math.exp(-x*x/2) / math.sqrt(2*math.pi)
    z = .5 * (math.erf(b/math.sqrt(2)) - math.erf(a/math.sqrt(2)))
    shift = (phi(a) - phi(b)) / z
    return dict(protocol=PROTOCOL, scene_setter_method='single_argument_v1', seed=seed, quick=quick,
                training_environment=dict(kind='fixed_grid', slips=TRAINING_SLIPS,
                    assignment='round-robin long/short pairs; nominal control always .1'),
                distribution=dict(kind='truncated_gaussian', low=low, high=high, mean=mean, std=std,
                                  effective_mean=mean+std*shift,
                                  effective_variance=std**2*(1+(a*phi(a)-b*phi(b))/z-shift**2),
                                  purpose='evaluation_only',
                                  sampler='numpy.PCG64 rejection; one fixed slip per paired evaluation round'),
                groups=groups, evaluation=evaluations,
                settings=dict(bc_epochs=1 if quick else 100, dagger_epochs=1 if quick else 50,
                              batch_size=256, supervised_lr=1e-4,
                              eval_steps=32 if quick else 3000))


def validate_plan(plan):
    d = plan['distribution']
    expected = make_plan(plan['seed'], d['low'], d['high'], d['mean'], d['std'], plan['quick'])
    if plan != expected:
        raise ValueError('Plan differs from declared deterministic Gaussian plan; create a new plan')

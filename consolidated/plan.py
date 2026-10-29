"""Offline protocol specification. Counts are active samples, not SGD steps."""
from context_training.evaluate import rounds

PROTOCOL = 'consolidated_imitation_v1'
SLIPS = [.025, .1, .2, 1., 10.]
ALGORITHMS = ('bc', 'dagger_mixed', 'gail_bc', 'gail_dagger_mixed')


def make_plan(quick=False):
    return dict(protocol=PROTOCOL, quick=quick, seed=42,
        varied_seeds=[42] if quick else [42, 43, 44],
        architectures=['hidden', 'input'], algorithms=list(ALGORITHMS),
        slips=SLIPS, original_environment='installed scene copied byte-for-byte',
        budgets=dict(expert_train=40 if quick else 170000,
                     dagger_seed=20 if quick else 100000,
                     learner_steps=20 if quick else 70000,
                     validation=20 if quick else 10000,
                     test=20 if quick else 10000,
                     iterations=1 if quick else 7),
        settings=dict(bc_epochs=1 if quick else 100, dagger_epochs=1 if quick else 50,
                      batch_size=256, learning_rate=1e-4,
                      ppo_epochs=1 if quick else 3, discriminator_epochs=1,
                      evaluation_steps=32 if quick else 3000),
        context='log1p(actual scalar slip)/log(11)',
        lineage=dict(original='BC and mixed DAgger start from the same seeded random weights',
                     varied='BC/DAgger warm-start corresponding NEW original policy',
                     gail='actor initialized from same-domain same-context same-seed BC/mixed DAgger'),
        accounting='Training samples exclude separate validation/test; GAIL uses 170k expert rows plus 70k PPO steps and initializer cost')


def segments(plan, domain, group, total, seed):
    """Equal surface/mode weights; original has same segment structure."""
    if total % 10:
        raise ValueError('Sample budget must be divisible by ten for balanced surface/mode coverage')
    return [dict(id=f'{group}_{i}_{mode}', slip=None if domain=='original' else slip,
                 mode=mode, steps=total//10, seed=seed+2*i+j)
            for i, slip in enumerate(plan['slips'])
            for j, mode in enumerate(('long', 'short'))]


def evaluation_rows(suite, quick=False):
    original = [dict(id=f'original_{s}', suite='original', slip=None, seed=s)
                for s in (101, 202, 303, 404, 505)]
    varied = rounds(42, 'all')
    if quick:
        original, varied = original[:1], varied[:1]
    if suite == 'original':
        return original
    if suite == 'varied':
        return varied
    if suite == 'all':
        return original+varied
    raise ValueError(suite)

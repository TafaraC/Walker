"""Offline BC pilot. Does not start or connect to a simulator."""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from variable_training.common import ROOT, begin, commit, digest, read, save, verified
from .data import audit, load_split
from .models import ARCHITECTURES, CONTEXT, checkpoint, initialize, load_checkpoint, policy_inputs


DEFAULT_DATA = ROOT / 'experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json'
DEFAULT_OUTPUT = ROOT / 'experiments/context_conditioning/bc_input_v1'


def sources():
    paths = [Path(__file__).parent / name for name in ('__init__.py', 'run.py', 'data.py', 'models.py')]
    paths += [ROOT / name for name in ('train.py', 'bc_data.py', 'configure_field_slip.py',
                                      'variable_training/common.py', 'variable_training/plan.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.training.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another context trainer is using this output directory')
        yield


def errors(model, x, y, slips, architecture, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    totals = np.empty(len(y), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            stop = start + batch_size
            inputs = policy_inputs(x[start:stop], slips[start:stop], architecture)
            difference = model(torch.from_numpy(inputs).to(device)).cpu().numpy() - y[start:stop]
            totals[start:stop] = np.square(difference.astype(np.float64)).sum(axis=1)
    if not np.isfinite(totals).all():
        raise ValueError('Non-finite held-out prediction')
    return dict(element_mse=float(totals.mean() / 16), squared_l2_per_state=float(totals.mean()),
                samples=len(y), by_slip={str(value): dict(samples=int((slips == value).sum()),
                    element_mse=float(totals[slips == value].mean() / 16))
                    for value in np.unique(slips)})


def train_one(directory, architecture, seed, settings, initializer, splits, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    state = torch.load(initializer, map_location='cpu', weights_only=True)
    model = initialize(state, architecture).to(device)
    x, y, slips = splits['train']
    vx, vy, vs = splits['validation']
    loader = DataLoader(TensorDataset(torch.from_numpy(policy_inputs(x, slips, architecture)),
                                     torch.from_numpy(y)),
                        batch_size=settings['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    best, best_epoch, updates = float('inf'), 0, 0
    history = []
    start_time = time.monotonic()
    for epoch in range(settings['epochs']):
        model.train()
        total = 0.
        for bx, by in loader:
            optimizer.zero_grad()
            loss = (model(bx.to(device)) - by.to(device)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            updates += 1
            total += loss.item() * len(bx)
        validation = errors(model, vx, vy, vs, architecture)
        value = validation['element_mse']
        history.append(dict(epoch=epoch + 1, train_element_mse=total / len(x),
                            validation_element_mse=value))
        if value < best:
            best, best_epoch = value, epoch + 1
            torch.save(checkpoint(model, architecture), directory / 'policy.pt')
            torch.save(optimizer.state_dict(), directory / 'optimizer.pt')
        print(f'[BC {architecture} seed={seed}] {epoch+1}/{settings["epochs"]} '
              f'validation MSE={value:.8f}', flush=True)
    model, loaded_architecture = load_checkpoint(directory / 'policy.pt')
    model.to(device)
    if loaded_architecture != architecture:
        raise ValueError('Saved architecture changed')
    # Test is evaluated only after validation-based checkpoint selection.
    tx, ty, ts = splits['test']
    metrics = dict(architecture=architecture, seed=seed, best_epoch=best_epoch,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   gradient_updates=updates, sample_exposures=len(x)*settings['epochs'],
                   training_samples=len(x), new_oracle_queries=0, new_environment_steps=0,
                   inherited_demonstration_samples={key: len(value[0]) for key, value in splits.items()},
                   wall_seconds=time.monotonic()-start_time, device=str(device),
                   validation=errors(model, vx, vy, vs, architecture),
                   test=errors(model, tx, ty, ts, architecture), history=history)
    save(directory / 'metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'train'))
    parser.add_argument('--data-plan', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--initializer', type=Path, default=ROOT/'bc_walk_policy.pth')
    parser.add_argument('--architectures', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if (args.epochs <= 0 or args.batch_size <= 0 or not np.isfinite(args.learning_rate) or
            args.learning_rate <= 0 or any(s < 0 or s >= 2**32 for s in args.seeds) or
            len(set(args.seeds)) != len(args.seeds) or
            len(set(args.architectures)) != len(args.architectures)):
        parser.error('Positive finite settings, unique architectures and unique uint32 seeds required')
    dataset = audit(args.data_plan)
    for split, entries in dataset['splits'].items():
        print(f'[Audit] {split}: {len(entries)} trajectories, '
              f'{sum(r["steps"] for r in entries)} samples; slips={sorted({r["slip"] for r in entries})}')
    initializer = args.initializer.resolve()
    # Validate initializer shape and finiteness, even for audit-only runs.
    initial_hash = digest(initializer)
    initialize(torch.load(initializer, map_location='cpu', weights_only=True), 'input')
    if args.stage == 'audit':
        print('[Audit] Complete; no training, simulator, or experiment writes.')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    identity = dict(protocol='context_bc_pilot_v1', dataset=dataset, context=CONTEXT,
                    architectures=args.architectures, seeds=args.seeds,
                    initializer=str(initializer), initializer_sha256=initial_hash,
                    settings=dict(epochs=args.epochs, batch_size=args.batch_size,
                                  learning_rate=args.learning_rate), sources=sources(),
                    device=args.device, torch_version=str(torch.__version__), numpy_version=np.__version__)
    output = args.output.resolve()
    with output_lock(output):
        plan_path = output / 'plan.json'
        if plan_path.exists() and read(plan_path) != identity:
            raise ValueError('Experiment inputs changed; choose a new --output directory')
        save(plan_path, identity)
        splits = None
        summary = []
        for seed in args.seeds:
            for architecture in args.architectures:
                directory = output / architecture / f'seed_{seed}'
                job_identity = dict(plan_sha256=digest(plan_path), architecture=architecture, seed=seed)
                if verified(directory, job_identity):
                    print(f'[Resume] {architecture} seed={seed}', flush=True)
                    summary.append(read(directory / 'metrics.json'))
                    continue
                if splits is None:
                    splits = {key: load_split(entries) for key, entries in dataset['splits'].items()}
                begin(directory)
                try:
                    metrics = train_one(directory, architecture, seed, identity['settings'],
                                        initializer, splits, torch.device(args.device))
                    if (audit(args.data_plan) != dataset or digest(initializer) != initial_hash or
                            sources() != identity['sources'] or read(plan_path) != identity):
                        raise ValueError('Inputs changed during training; no completion receipt written')
                    commit(directory, job_identity)
                    summary.append(metrics)
                except BaseException as error:
                    save(directory / 'failure.json', {'error': repr(error)})
                    raise
        save(output / 'summary.json', {'protocol': identity['protocol'], 'runs': summary,
                                      'scope': 'held-out latent action fidelity; no live evaluation'})
        print(f'[Complete] Offline BC pilot: {output / "summary.json"}')


if __name__ == '__main__':
    main()

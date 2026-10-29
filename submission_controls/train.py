"""Exact optimizer budgets and validation-only checkpoint selection; no simulator."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from consolidated.models import load_policy
from consolidated.worker import load_shards
from variable_training.common import read, save


def update_target(samples, batch_size, epochs=None, updates=None):
    if samples < 1 or batch_size < 1 or (epochs is None) == (updates is None):
        raise ValueError('Specify exactly one positive epoch/update budget')
    result = math.ceil(samples / batch_size) * epochs if epochs is not None else updates
    if result < 1:
        raise ValueError('Budget must be positive')
    return int(result)


class Stopper:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.reference, self.stale = float('inf'), 0

    def observe(self, value):
        if value < self.reference - self.min_delta:
            self.reference, self.stale = value, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def fit_arrays(job, directory, x, y, vx, vy):
    """Public small-array entry point also used by offline regression tests."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not len(x) or not len(vx) or x.shape[1] != vx.shape[1] or y.shape[1] != 16:
        raise ValueError('Invalid training/validation arrays')
    if not all(np.isfinite(a).all() for a in (x, y, vx, vy)):
        raise ValueError('Nonfinite data')
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    device = torch.device(job.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    policy = load_policy(job['model'], 'bc', job['architecture']).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=job['learning_rate'])
    if job.get('optimizer'):
        optimizer.load_state_dict(torch.load(job['optimizer'], map_location=device, weights_only=True))
    train = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                       batch_size=job['batch_size'], shuffle=True,
                       generator=torch.Generator().manual_seed(job['seed']))
    val = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)), batch_size=1024)
    target = update_target(len(x), job['batch_size'], job.get('epochs'), job.get('updates'))
    diagnostic = job.get('early_stop', False)
    stopper = Stopper(job.get('patience', 25), job.get('min_delta', 1e-5))
    best, updates, exposures, epoch = float('inf'), 0, 0, 0
    best_update, best_epoch, best_exposures = None, None, None
    history = []
    stop_reason = 'budget_exhausted'
    print(f'[Control training] device={device}, target_updates={target}', flush=True)
    while updates < target:
        epoch += 1
        policy.train()
        for bx, by in train:
            optimizer.zero_grad()
            loss = (policy(bx.to(device)) - by.to(device)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            updates += 1
            exposures += len(bx)
            if updates == target:
                break
        policy.eval()
        with torch.no_grad():
            error = sum((policy(bx.to(device))-by.to(device)).square().sum().item() for bx, by in val)
        mse = error / vy.size
        if not math.isfinite(mse):
            raise ValueError('Nonfinite validation loss')
        if mse < best:
            best, best_update, best_epoch, best_exposures = mse, updates, epoch, exposures
            torch.save(policy.state_dict(), directory/'policy.pth')
            torch.save(optimizer.state_dict(), directory/'optimizer.pth')
        row = dict(epoch=epoch, gradient_updates=updates, sample_exposures=exposures,
                   validation_mse=mse, partial_epoch=(updates % len(train) != 0))
        history.append(row)
        save(directory/'history.json', history)
        print(f'[Control training] {row}', flush=True)
        should_stop = stopper.observe(mse)
        if diagnostic and epoch >= job.get('minimum_epochs', 100) and should_stop:
            stop_reason = 'validation_patience'
            break
    # Preserve budget endpoint separately from the validation-selected checkpoint.
    torch.save(policy.state_dict(), directory/'last_policy.pth')
    torch.save(optimizer.state_dict(), directory/'last_optimizer.pth')
    metrics = dict(validation_mse=best, best_epoch=best_epoch,
                   selected_gradient_updates=best_update, selected_sample_exposures=best_exposures,
                   gradient_updates=updates, sample_exposures=exposures, epochs_run=epoch,
                   target_gradient_updates=target, stop_reason=stop_reason,
                   train_samples=len(x), validation_samples=len(vx), device=str(device),
                   selection='minimum validation element MSE; test data never used for selection',
                   inherited_costs=job.get('inherited_costs', {}))
    if not diagnostic and updates != target:
        raise ValueError('Incomplete fixed-budget training')
    save(directory/'metrics.json', metrics)
    return metrics


def train(job, directory):
    if set(job['train']) & set(job['validation']):
        raise ValueError('Training/validation shard overlap')
    x, y = load_shards(job['train'], job['architecture'])
    vx, vy = load_shards(job['validation'], job['architecture'])
    if job.get('quick'):
        x, y, vx, vy = x[:64], y[:64], vx[:32], vy[:32]
    fit_arrays(job, directory, x, y, vx, vy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    train(read(args.job), args.job.parent)

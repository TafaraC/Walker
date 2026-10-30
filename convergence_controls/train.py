"""Validation stopping on fixed BC/DAgger datasets, including the inherited endpoint."""
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
    if (x.ndim != 2 or vx.ndim != 2 or y.shape != (len(x),16) or vy.shape != (len(vx),16)
            or not len(x) or not len(vx) or x.shape[1] != vx.shape[1]):
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
        if any(group['lr'] != job['learning_rate'] for group in optimizer.param_groups):
            raise ValueError('Inherited optimizer learning rate differs from declared rule')
    train = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                       batch_size=job['batch_size'], shuffle=True,
                       generator=torch.Generator().manual_seed(job['seed']))
    val = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)), batch_size=1024)
    target = update_target(len(x), job['batch_size'], job.get('epochs'), job.get('updates'))
    diagnostic = job.get('early_stop', False)
    stopper = Stopper(job.get('patience', 25), job.get('min_delta', 1e-5))
    def validation_mse():
        policy.eval()
        with torch.no_grad():
            error = sum((policy(bx.to(device))-by.to(device)).square().sum().item() for bx, by in val)
        mse = error / vy.size
        if not math.isfinite(mse):
            raise ValueError('Nonfinite validation loss')
        return mse

    # Continuing a trained checkpoint must never select a worse first new epoch
    # simply because the inherited endpoint was omitted from the candidates.
    best = validation_mse()
    updates = exposures = epoch = best_update = best_epoch = best_exposures = 0
    torch.save(policy.state_dict(), directory/'policy.pth')
    torch.save(optimizer.state_dict(), directory/'optimizer.pth')
    history = [dict(epoch=0,gradient_updates=0,sample_exposures=0,validation_mse=best,
                    training_mse=None,partial_epoch=False)]
    save(directory/'history.json',history)
    stopper.observe(best)
    stop_reason = 'budget_exhausted'
    print(f'[Control training] device={device}, target_updates={target}', flush=True)
    while updates < target:
        epoch += 1
        policy.train()
        train_error = train_elements = 0
        for bx, by in train:
            optimizer.zero_grad()
            loss = (policy(bx.to(device)) - by.to(device)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            train_error += float(loss.detach()) * by.numel()
            train_elements += by.numel()
            updates += 1
            exposures += len(bx)
            if updates == target:
                break
        mse = validation_mse()
        if mse < best:
            best, best_update, best_epoch, best_exposures = mse, updates, epoch, exposures
            torch.save(policy.state_dict(), directory/'policy.pth')
            torch.save(optimizer.state_dict(), directory/'optimizer.pth')
        row = dict(epoch=epoch, gradient_updates=updates, sample_exposures=exposures,
                   validation_mse=mse,training_mse=train_error/train_elements,
                   partial_epoch=(updates % len(train) != 0),
                   improvement_from_initial=history[0]['validation_mse']-mse)
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
    inherited = job.get('inherited_costs', {})
    metrics = dict(validation_mse=best, initial_validation_mse=history[0]['validation_mse'],best_epoch=best_epoch,
                   selected_gradient_updates=best_update, selected_sample_exposures=best_exposures,
                   gradient_updates=updates, sample_exposures=exposures, epochs_run=epoch,
                   target_gradient_updates=target, stop_reason=stop_reason,
                   train_samples=len(x), validation_samples=len(vx), device=str(device),
                   selection='minimum validation element MSE INCLUDING inherited endpoint; test never used',
                   inherited_costs=inherited,optimizer_restored=bool(job.get('optimizer')),
                   total_lineage_gradient_updates=inherited.get('gradient_updates',0)+updates,
                   selected_lineage_gradient_updates=inherited.get('gradient_updates',0)+best_update,
                   total_lineage_sample_exposures=inherited.get('sample_exposures',0)+exposures,
                   selected_lineage_sample_exposures=inherited.get('sample_exposures',0)+best_exposures,
                   patience=stopper.patience,min_delta=stopper.min_delta,
                   minimum_additional_epochs=job.get('minimum_epochs',100),
                   plateau_reference=stopper.reference,epochs_without_material_improvement=stopper.stale,
                   validation_stopping_reached=stop_reason=='validation_patience',
                   caveat='Validation stopping on fixed data; not mathematical or online DAgger convergence')
    if not diagnostic and updates != target:
        raise ValueError('Incomplete fixed-budget training')
    save(directory/'metrics.json', metrics)
    return metrics


def train(job, directory):
    if set(job['train']) & set(job['validation']):
        raise ValueError('Training/validation shard overlap')
    x, y = load_shards(job['train'], job['architecture'])
    vx, vy = load_shards(job['validation'], job['architecture'])
    if not job.get('quick') and (len(x)!=170000 or len(vx)!=10000):
        raise ValueError('Expected fixed 170,000 training / 10,000 validation examples')
    if job.get('quick'):
        x, y, vx, vy = x[:64], y[:64], vx[:32], vy[:32]
    fit_arrays(job, directory, x, y, vx, vy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    train(read(args.job), args.job.parent)

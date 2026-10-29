"""Read-only reuse of completed, scalar-slip oracle trajectories."""
from pathlib import Path

import numpy as np

from configure_field_slip import status
from variable_training.common import digest, read, verified
from variable_training.plan import validate_plan
from .models import encode_slip


def audit(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    validate_plan(plan)
    result = {'plan': str(plan_path), 'plan_sha256': digest(plan_path), 'splits': {}}
    seen_paths, seen_data = set(), set()
    for split in ('train', 'validation', 'test'):
        entries = []
        for row in plan['groups'][split]:
            directory = plan_path.parent / 'data' / 'varied' / split / row['id']
            receipt_path = directory / 'completed.json'
            receipt = read(receipt_path)
            identity = receipt['identity']
            if (identity.get('plan') != result['plan_sha256'] or
                    identity.get('spec') != dict(task='collect', kind='oracle', row=row)):
                raise ValueError(f'Oracle shard provenance mismatch: {directory}')
            required = {'data.npz', 'metrics.json', 'environment_manifest.json', 'soccer.rsg'}
            if not required.issubset(receipt['files']):
                raise ValueError(f'Missing provenance artifacts: {directory}')
            verified(directory, identity)  # Historical receipt, not current collector hashes.
            manifest = read(directory / 'environment_manifest.json')
            scene = status(directory / 'soccer.rsg', row['slip'])
            if (not manifest.get('server_scene_import_verified') or
                    manifest.get('scene_setter_method') != 'single_argument_v1' or
                    scene['scene_setter_method'] != 'single_argument_v1' or
                    not scene['matches_expected'] or scene['sha256'] != manifest['sha256'] or
                    scene['field_contact_slip'] != manifest['field_contact_slip']):
                raise ValueError(f'Unverified scalar slip: {directory}')
            slip = scene['field_contact_slip']
            encode_slip(slip)
            metrics = read(directory / 'metrics.json')
            if (metrics['slip'] != row['slip'] or metrics['active_steps'] != row['steps'] or
                    metrics['oracle_queries'] != row['steps'] or
                    metrics['oracle_control_steps'] != row['steps'] or
                    metrics['model_sha256'] is not None):
                raise ValueError(f'Not a complete oracle trajectory: {directory}')
            path = (directory / 'data.npz').resolve()
            sha = digest(path)
            if path in seen_paths or sha in seen_data:
                raise ValueError('Duplicate trajectory across dataset splits')
            seen_paths.add(path)
            seen_data.add(sha)
            with np.load(path, allow_pickle=False) as data:
                for name, dims in [('obs', 63), ('actions', 16)]:
                    if (data[name].shape != (row['steps'], dims) or
                            not np.isfinite(data[name]).all()):
                        raise ValueError(f'Invalid {name} array: {path}')
                if not np.array_equal(data['actions'], data['executed_actions']):
                    raise ValueError(f'Oracle labels differ from executed actions: {path}')
            entries.append(dict(path=str(path), sha256=sha, slip=slip, steps=row['steps'],
                                mode=row['mode'], trajectory_seed=row['seed'],
                                receipt_sha256=digest(receipt_path)))
        if not entries:
            raise ValueError(f'Empty split: {split}')
        result['splits'][split] = entries
    return result


def load_split(entries):
    observations, actions, slips = [], [], []
    for entry in entries:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Dataset changed after audit')
        with np.load(entry['path'], allow_pickle=False) as data:
            observations.append(data['obs'].astype(np.float32))
            actions.append(data['actions'].astype(np.float32))
        slips.append(np.full(entry['steps'], entry['slip'], dtype=np.float64))
    return np.concatenate(observations), np.concatenate(actions), np.concatenate(slips)

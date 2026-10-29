import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'fixed_grid_training_gaussian_evaluation_v4'
MODELS = {'bc': 'bc_walk_policy.pth', 'dagger_mixed': 'dagger_mixed_policy.pth',
          'gail_bc': 'gail_bc_policy.pth', 'gail_dagger': 'gail_dagger_policy.pth'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    files = list(ROOT.glob('*.py'))
    for folder in ('variable_training', 'agent', 'behaviors', 'communication', 'world', 'math_ops', 'cpp'):
        for extension in ('*.py', '*.so', '*.pkl'):
            files += list((ROOT / folder).rglob(extension))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}


@contextmanager
def simulator_lock():
    # Same lock as ongoing Phase 2 evaluations: never compete for its server.
    path = ROOT / 'experiments/phase2_domain_robustness/.runner.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another simulator pipeline is running; finish it before training.')
        yield


def verified(directory, identity):
    receipt = Path(directory) / 'completed.json'
    if not receipt.exists():
        return False
    stored = read(receipt)
    if stored['identity'] != identity:
        raise ValueError(f'Inputs changed; choose a new experiment: {directory}')
    if not stored['files']:
        raise ValueError(f'Empty completion receipt: {directory}')
    for name, expected in stored['files'].items():
        if digest(Path(directory) / name) != expected:
            raise ValueError(f'Completed artifact changed: {directory}/{name}')
    return True


# Audited pre-Gaussian-selector receipts. These files only orchestrate/report
# evaluations or manage metadata; actual client/server sources remain checked.
_LEGACY_EVALUATION_METADATA = {
    'variable_training/run.py': {
        '4649f70853e120c33f756f8abfa03a628f2d8e804b72e590399e5fceab79c831',
        'e303eada39abbc51bd6492dcece8f7d5f54f6ead539f75fd1a6cdd259c60cd34'},
    'variable_training/test_pipeline.py': {
        '12f65457dbda2071a550495158e87b9b56364cbb429a7323c50f7de5a57c858c',
        '37c3a7f799e7c6b37084e6f9d869d2e40bfac380b9cfcc0c381663943f7bfbf8'},
    'variable_training/common.py': {
        '2fcf8bed646170a0c9fd33458ac5e32f2378057c93ded3b8bd4be3da05a0036d'},
}


def evaluation_identity(identity, legacy=False):
    result = dict(identity)
    if 'sources' not in result:
        return result
    sources = dict(result['sources'])
    # Audited validation-only correction: writer/scene bytes are identical.
    # Canonicalize only these exact versions, never ignore arbitrary changes.
    slip_validator_versions = {
        '721c61da95cdee36ba6c30b0bef249109183c5077e5d229cbf5614828ca6dc17',
        '2c2337a3368cbffcbdb992a22915657c13bf3e39b9f113afd29674c893625542',
    }
    if sources.get('configure_field_slip.py') in slip_validator_versions:
        sources['configure_field_slip.py'] = '721c61da95cdee36ba6c30b0bef249109183c5077e5d229cbf5614828ca6dc17'
    for name, approved in _LEGACY_EVALUATION_METADATA.items():
        if legacy and name in sources and sources[name] not in approved:
            raise ValueError(f'Unreviewed legacy evaluation metadata source: {name}')
        sources.pop(name, None)
    result['sources'] = sources
    return result


def verified_evaluation(directory, identity):
    receipt = Path(directory) / 'completed.json'
    if not receipt.exists():
        return False
    stored = read(receipt)['identity']
    if evaluation_identity(stored, legacy=True) != evaluation_identity(identity):
        raise ValueError(f'Evaluation inputs changed; choose a new experiment: {directory}')
    # Validate using the original identity; do not rewrite historical provenance.
    verified(directory, stored)
    if stored != evaluation_identity(identity):
        print(f'[Resume] Compatible legacy evaluation verified: {directory}', flush=True)
    return True


def begin(directory):
    directory = Path(directory)
    if directory.exists():
        archived = directory.with_name(directory.name + '_failed_' + uuid.uuid4().hex[:10])
        directory.rename(archived)
        print(f'[Archive] Incomplete attempt preserved at {archived}', flush=True)
    directory.mkdir(parents=True)


def commit(directory, identity):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name not in ('completed.json', 'completed.json.tmp')}
    save(directory / 'completed.json', {'identity': identity, 'files': files})

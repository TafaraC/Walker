"""Frozen-policy slip experiments using explicit local scenes and fresh servers."""
import argparse
import json
import math
import signal
import fcntl
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid
import threading
from contextlib import contextmanager

from configure_field_slip import apply, digest, status

ROOT = Path(__file__).resolve().parent
SYSTEM = Path('/usr/share/rcssserver3d')
SCENE = SYSTEM / 'rsg/agent/nao/soccer.rsg'
OUT = ROOT / 'experiments/phase2_domain_robustness/scalar_wide_v4'
POLICIES = [('random', 'random', None), ('oracle', 'oracle', None),
            ('bc', 'bc', 'bc_walk_policy.pth'), ('dagger', 'dagger', 'dagger_policy.pth'),
            ('dagger', 'dagger_mixed', 'dagger_mixed_policy.pth'),
            *[('gail', name, name + '_policy.pth') for name in ('gail_dagger', 'gail_bc', 'gail_random')]]
SEEDS = [101, 202, 303]
FIXED = {'high_traction': .025, 'nominal': .1, 'low_traction': .2,
         'slip_1': 1., 'slip_10': 10.}
PROTOCOL = 'isolated_server_scalar_wide_slip_v4'


def fingerprints():
    paths = ['evaluate_policy.py', 'imitation_utils.py', 'expert_train_data.npz',
             'expert_test_data.npz']
    paths += [model for _, _, model in POLICIES if model]
    paths += [str(p.relative_to(ROOT)) for folder in
              ('agent', 'behaviors', 'communication', 'world', 'math_ops')
              for p in (ROOT / folder).rglob('*.py')]
    paths += [p.name for p in ROOT.glob('*.py')]
    return {p: digest(ROOT / p) for p in paths if (ROOT / p).is_file()}


def validate_result(path, condition, slip, label, seed):
    import numpy as np
    item = json.loads(path.read_text())
    for key, expected in dict(active_steps=3000, seed=seed, policy_label=label,
                              environment_id=condition, field_contact_slip=slip).items():
        if item.get(key) != expected:
            raise ValueError(f'Invalid {key}: {path}')
    with np.load(item['trace'], allow_pickle=False) as trace:
        required = {'obs': (3000, 63), 'actions': (3000, 16),
                    'oracle_actions': (3000, 16), 'joint_positions_deg': (3000, 22),
                    'episode_id': (3000,), 'target_progress': (3000,)}
        for key, shape in required.items():
            if key not in trace or trace[key].shape != shape:
                raise ValueError(f'Invalid trace {key}: {path}')
        for key in trace.files:
            if len(trace[key]) != 3000 or not np.isfinite(trace[key]).all():
                raise ValueError(f'Incomplete/non-finite trace: {path}')
    return item


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(directory, slip, seed):
    if not math.isfinite(slip) or slip < 0:
        raise ValueError('Slip must be finite and non-negative')
    directory.mkdir(parents=True, exist_ok=True)
    scene = directory / 'soccer.rsg'
    shutil.copy2(SCENE, scene)
    apply(scene, slip, directory / 'soccer.nominal.rsg')
    nao = (SYSTEM / 'naosoccersim.rb').read_text()
    source = "scene.importScene('rsg/agent/nao/soccer.rsg')"
    assert nao.count(source) == 1
    token = uuid.uuid4().hex
    marker = f'WALKER_SCENE_VERIFIED token={token} slip={slip:.12g} sha256={digest(scene)}'
    # Check the bytes inside the server process immediately before import.
    checked_import = (
        'require "digest"\n'
        f'  walker_scene = {json.dumps(str(scene))}\n'
        f'  raise "Walker scene checksum mismatch" unless Digest::SHA256.file(walker_scene).hexdigest == "{digest(scene)}"\n'
        '  walker_imported = scene.importScene(walker_scene)\n'
        '  raise "Walker scene import failed" unless walker_imported\n'
        f'  puts "{marker}"\n  STDOUT.flush'
    )
    nao = nao.replace(source, checked_import)
    assert nao.count('randomServer.seed(0)') == 1
    (directory / 'naosoccersim.rb').write_text(nao)
    server = (SYSTEM / 'rcssserver3d.rb').read_text()
    source = 'run "naosoccersim.rb"'
    assert server.count(source) == 1
    server = server.replace(
        source,
        f'run "{directory / "naosoccersim.rb"}"',
    )
    (directory / 'rcssserver3d.rb').write_text(server)
    manifest = status(scene, slip)
    if not manifest['matches_expected']:
        raise RuntimeError('Generated scene slip does not match requested value')
    manifest.update(startup_marker=marker, verification_token=token,
                    verification_scope='server-side scene SHA-256 and successful import; no live ODE readback')
    manifest.update(protocol=PROTOCOL, simulator_seed="unchanged: randomServer.seed(0)", system_scene=status(SCENE, .1),
                    config_sha256={p.name: digest(p) for p in directory.glob('*.rb')},
                    evaluator_sha256=digest(ROOT / 'evaluate_policy.py'),
                    spark_config_sha256=digest(Path.home() / '.simspark/spark.rb'),
                    frozen_models={model: digest(ROOT / model) for _, _, model in POLICIES if model})
    save(directory / 'environment_manifest.json', manifest)
    return manifest


@contextmanager
def simulator(directory, slip):
    manifest = prepare(directory, slip, None)
    with (directory / 'server.log').open('w') as log:
        server = subprocess.Popen(['rcssserver3d', '--script-path', str(directory / 'rcssserver3d.rb')],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited; inspect {directory / "server.log"}')
                if manifest['startup_marker'] in (directory / 'server.log').read_text().splitlines():
                    verified = status(directory / 'soccer.rsg', slip)
                    if (not verified['matches_expected'] or verified['sha256'] != manifest['sha256']
                            or verified['scene_setter_method'] != 'single_argument_v1'):
                        raise RuntimeError('Local scene changed during server startup')
                    for name, expected in manifest['config_sha256'].items():
                        if digest(directory / name) != expected:
                            raise RuntimeError(f'Startup configuration changed: {name}')
                    manifest['server_scene_import_verified'] = True
                    save(directory / 'environment_manifest.json', manifest)
                    print(f'[Slip verification] Server imported verified scene: slip={slip:.12g}, sha256={verified["sha256"]}', flush=True)
                    break
                time.sleep(.1)
            else:
                raise RuntimeError(f'Server startup marker missing; inspect {directory / "server.log"}')
            yield dict(manifest, server_process=server)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def launch(command, log_path, env=None, server=None):
    # Tee progress while retaining the original command's evaluation behavior.
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        stopped = threading.Event()
        def watch_server():
            while not stopped.wait(.2):
                if server.poll() is not None and process.poll() is None:
                    # Abort a client waiting forever for a dead simulator.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        if not stopped.wait(5) and process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
        watcher = threading.Thread(target=watch_server, daemon=True) if server else None
        if watcher:
            watcher.start()
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            stopped.set()
            if watcher:
                watcher.join(timeout=1)
            process.stdout.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def run_isolated(condition, slip, kind, label, model, seed, directory):
    receipt = directory / 'completed.json'
    expected = dict(protocol=PROTOCOL, condition=condition, slip=slip, seed=seed,
                    label=label, inputs=fingerprints(),
                    spark_sha256=digest(Path.home() / '.simspark/spark.rb'),
                    system_scene_sha256=digest(SCENE),
                    system_scripts={name: digest(SYSTEM / name) for name in
                                    ('rcssserver3d.rb', 'naosoccersim.rb')})
    path = directory / 'raw' / f'{label}_seed_{seed}.json'
    if receipt.exists():
        stored = json.loads(receipt.read_text())
        if stored['identity'] != expected:
            raise ValueError(f'Inputs changed; archive incompatible run: {directory}')
        for name, sha in stored['files'].items():
            if digest(directory / name) != sha:
                raise ValueError(f'Completed artifact changed: {directory / name}')
        validate_result(path, condition, slip, label, seed)
        print(f'[Evaluation] Skipping verified {condition} {label} seed {seed}', flush=True)
        return
    if directory.exists():
        directory.rename(directory.with_name(directory.name + '_failed_' + uuid.uuid4().hex[:8]))
    directory.mkdir(parents=True)
    command = [sys.executable, '-u', 'evaluate_policy.py', '--policy', kind,
               '--label', label, '--expert-data', 'expert_train_data.npz',
               '--steps', '3000', '--seed', str(seed), '--output-dir', str(directory / 'raw'),
               '--environment-id', condition, '--field-slip', str(slip)]
    if model:
        command += ['--model', model]
    try:
        with simulator(directory, slip) as manifest:
            launch(command, directory / 'evaluation.log', server=manifest['server_process'])
            result = validate_result(path, condition, slip, label, seed)
            result.update(environment_manifest=str(directory / 'environment_manifest.json'),
                          scene_sha256=manifest['sha256'], protocol=PROTOCOL)
            save(path, result)
        names = ['soccer.rsg', 'environment_manifest.json', 'rcssserver3d.rb',
                 'naosoccersim.rb', str(path.relative_to(directory)),
                 str(Path(result['trace']).resolve().relative_to(directory.resolve()))]
        save(receipt, {'identity': expected,
                       'files': {name: digest(directory / name) for name in names}})
    except BaseException as error:
        save(directory / 'failure.json', {'error': repr(error)})
        raise


def block(condition, slip, stage, seeds=SEEDS, suffix=''):
    base = OUT / 'launches' / (condition + suffix)
    base.mkdir(parents=True, exist_ok=True)
    try:
        if stage in ('smoke', 'fixed'):
            run_isolated(condition, slip, 'oracle', 'oracle', None, 901,
                         OUT / 'smoke' / condition)
        if stage != 'smoke':
            for kind, label, model in POLICIES:
                for seed in seeds:
                    run_isolated(condition, slip, kind, label, model, seed,
                                 OUT / 'conditions' / condition / 'runs' / f'{label}_seed_{seed}')
            save(base / 'completed.json', {'protocol': PROTOCOL, 'condition': condition,
                                          'slip': slip, 'seeds': seeds})
    except BaseException as error:
        save(base / 'failure.json', {'error': repr(error), 'condition': condition, 'slip': slip})
        raise
    (base / 'failure.json').unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'fixed', 'sampled'])
    parser.add_argument('--condition', choices=['all', *FIXED], default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.stage == 'sampled' and args.condition != 'all':
        parser.error('--condition applies only to smoke/fixed')
    OUT.mkdir(parents=True, exist_ok=True)
    # Same lock as variable training and contact diagnostics, despite new output root.
    with (ROOT / 'experiments/phase2_domain_robustness/.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (3100, 3200):
            with socket.socket() as sock:
                sock.settimeout(1)
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} occupied; stop the existing simulator')
        subprocess.run([sys.executable, 'check_simulator_config.py'], check=True)
        for required in ['expert_train_data.npz', 'expert_test_data.npz'] + [
                model for _, _, model in POLICIES if model]:
            if not (ROOT / required).is_file():
                raise FileNotFoundError(f'Required evaluation artifact: {required}')
        subprocess.run([sys.executable, '-c',
                        'from scripts.commons.Script import Script; Script.build_cpp_modules(); '
                        'from cpp.ball_predictor import ball_predictor; '
                        'from cpp.localization import localization; from cpp.a_star import a_star'], check=True)
        original = status(SCENE, .1)
        if not original['matches_expected']:
            raise RuntimeError('Installed scene must be nominal')
        try:
            if args.stage == 'sampled':
                from summarize_varied_slip import validate_draws
                plan = validate_draws(OUT)
                print(json.dumps(plan, indent=2))
                for draw in plan['draws']:
                    if not block('sampled', draw['slip'], 'sampled', [draw['seed']], f'_seed_{draw["seed"]}'):
                        break
            else:
                for name, slip in FIXED.items():
                    if args.condition in ('all', name):
                        if not block(name, slip, args.stage):
                            break
        finally:
            verified = status(SCENE, .1)
            verified['matches_original_sha256'] = verified['sha256'] == original['sha256']
            verified['method'] = 'Local override only; owned server stopped; installed scene unchanged'
            save(OUT / 'nominal_restoration.json', verified)
            if not verified['matches_expected'] or not verified['matches_original_sha256']:
                raise RuntimeError('Nominal verification failed')


if __name__ == '__main__':
    main()

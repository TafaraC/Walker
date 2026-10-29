"""Server ownership copied from the isolated evaluation protocol; no evaluation edits."""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from configure_field_slip import apply, digest, status
from .common import ROOT, MODELS, PROTOCOL, save

SYSTEM = Path('/usr/share/rcssserver3d')
SCENE = SYSTEM / 'rsg/agent/nao/soccer.rsg'
POLICIES = [(None, label, model) for label, model in MODELS.items()]


def require_free_ports():
    for port in (3100, 3200):
        with socket.socket() as sock:
            sock.settimeout(.5)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} occupied; no existing server will be killed')


def prepare(directory, slip, seed, setter_method='scalar'):
    if not math.isfinite(slip) or slip < 0:
        raise ValueError('Slip must be finite and non-negative')
    directory.mkdir(parents=True, exist_ok=True)
    scene = directory / 'soccer.rsg'
    shutil.copy2(SCENE, scene)
    apply(scene, slip, directory / 'soccer.nominal.rsg', setter_method=setter_method)
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
    require_free_ports()
    original = status(SCENE, .1)
    if not original['matches_expected']:
        raise RuntimeError('Installed scene must remain nominal (slip 0.1)')
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
            restored = status(SCENE, .1)
            restored['matches_original_sha256'] = restored['sha256'] == original['sha256']
            save(directory / 'nominal_restoration.json', restored)
            if not restored['matches_original_sha256']:
                raise RuntimeError('Installed nominal scene changed during run')


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



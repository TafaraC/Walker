"""Select a walking policy and fixed slip for a managed-server RoboViz demo."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def choices(root):
    from demo_agents import POLICIES
    result = []
    for label, kind, model in POLICIES:
        if kind == 'random':
            continue
        if model is None or (ROOT / model).is_file():
            result.append((f'historical: {label}', kind,
                           str(ROOT / model) if model else None, None))
    for path in sorted((root / 'policies').glob('*/*/seed_*/*/final.json')):
        record = json.loads(path.read_text())
        model = Path(record['model'])
        if model.is_file():
            result.append((str(path.parent.relative_to(root / 'policies')),
                           record['kind'], str(model), record['architecture']))
    return result


def parse_slip(value):
    if value.strip().lower() == 'original':
        return None
    slip = float(value)
    if not math.isfinite(slip) or slip < 0:
        raise ValueError('Slip must be finite and non-negative, or original')
    return float(f'{slip:.12g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'experiments/consolidated/v1')
    parser.add_argument('--model', help='Exact menu label; omit for interactive selection')
    parser.add_argument('--slip', help='Non-negative scalar slip, or original; omit to select')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--list', action='store_true', help='List available models without starting a server')
    args = parser.parse_args()
    os.chdir(ROOT)
    options = choices(args.root.resolve())
    for i, option in enumerate(options, 1):
        print(f'{i:3}. {option[0]}')
    if args.list:
        return
    if args.model:
        matches = [p for p in options if p[0] == args.model]
        if not matches:
            raise ValueError('Model label not found; use --list')
        selected = matches[0]
    else:
        number = int(input('Model number (0 exits): '))
        if number == 0:
            return
        if not 1 <= number <= len(options):
            raise ValueError('Invalid model number')
        selected = options[number - 1]
    slip = parse_slip(args.slip if args.slip is not None else
                      input('Slip [0.025 / 0.1 / 0.2 / 1 / 10 / original]: '))
    if selected[3] == 'input' and slip is None:
        raise ValueError('Input-conditioned models require an explicit scalar slip; original is a legacy scene')
    from variable_training.common import simulator_lock
    from consolidated.server import simulator, require_free_ports
    from .demo import run_demo
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable, str(ROOT / 'check_simulator_config.py')], check=True)
        subprocess.run([sys.executable, '-c',
                        'from scripts.commons.Script import Script; Script.build_cpp_modules()'], check=True)
        logs = ROOT / 'logs/viewer'
        logs.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='demo_', dir=logs))
        print(f'[Viewer] Logs: {directory}')
        with simulator(directory, slip):
            input('Open/reconnect RoboViz to localhost:3200, then press Enter to walk (30 seconds)... ')
            run_demo(selected[:3], args.seed, architecture=selected[3], slip=slip)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\n[Viewer] Stopped.')
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'[Viewer error] {error}', file=sys.stderr)
        sys.exit(1)

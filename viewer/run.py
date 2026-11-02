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
import time

ROOT = Path(__file__).resolve().parents[1]
CONVERGENCE_ROOT = ROOT / 'experiments/convergence_controls/all_v1'
SUBMISSION_ROOT = ROOT / 'experiments/submission_controls/v1'
SCRATCH_ROOT = ROOT / 'experiments/consolidated/v1/scratch_v1'


def receipt_choices(root, prefix='convergence'):
    """Discover selected checkpoints, retaining recorded architecture and receipt checks.

    Used for the convergence (validation-stopped) batch and the submission controls
    (matched-update BC, 200-epoch varied-only BC and the zero-action controller)."""
    from variable_training.common import digest
    result = []
    for receipt_path in sorted((root / 'training').glob('*/completed.json')):
        directory = receipt_path.parent
        model = directory / 'policy.pth'
        job_path = directory / 'job.json'
        if not model.is_file() or not job_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        job = json.loads(job_path.read_text())
        if job.get('task') not in ('control_train', 'random_network', 'zero'):
            continue
        if (receipt.get('identity', {}).get('spec') != job or
                any(receipt.get('files', {}).get(p.name) != digest(p)
                    for p in (job_path, model))):
            raise ValueError(f'{prefix} checkpoint/metadata differs from its receipt: {directory}')
        architecture = job.get('architecture')
        if architecture not in ('hidden', 'input'):
            raise ValueError(f'Unknown {prefix} architecture: {directory}')
        # Supervised BC/DAgger, untrained engine controls and the zero controller all use BCPolicy.
        result.append((f'{prefix}/{directory.name}', 'bc', str(model), architecture))
    return result


def consolidated_choices(root, prefix=''):
    result = []
    for path in sorted((root / 'policies').glob('*/*/seed_*/*/final.json')):
        record = json.loads(path.read_text())
        model = Path(record['model'])
        if model.is_file():
            result.append((prefix + str(path.parent.relative_to(root / 'policies')),
                           record['kind'], str(model), record['architecture']))
    return result


def choices(root, convergence_root=CONVERGENCE_ROOT, submission_root=SUBMISSION_ROOT,
            scratch_root=SCRATCH_ROOT):
    from demo_agents import POLICIES
    result = []
    for label, kind, model in POLICIES:
        if kind == 'random' and not (ROOT / 'expert_train_data.npz').is_file():
            continue  # the uniform-random baseline samples within the expert action range
        if model is None or (ROOT / model).is_file():
            result.append((f'historical: {label}', kind,
                           str(ROOT / model) if model else None, None))
    result.extend(consolidated_choices(root))
    if scratch_root.resolve() != root.resolve():
        result.extend(consolidated_choices(scratch_root, 'scratch/'))
    result.extend(receipt_choices(convergence_root, 'convergence'))
    result.extend(receipt_choices(submission_root, 'controls'))
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
    parser.add_argument('--convergence-root', type=Path, default=CONVERGENCE_ROOT,
                        help='Completed convergence batch to include alongside consolidated models')
    parser.add_argument('--submission-root', type=Path, default=SUBMISSION_ROOT,
                        help='Submission controls: matched-update BC, 200-epoch varied-only BC, zero action')
    parser.add_argument('--scratch-root', type=Path, default=SCRATCH_ROOT,
                        help='Varied-only (scratch) family, including GAIL from random weights')
    parser.add_argument('--model', help='Exact menu label; omit for interactive selection')
    parser.add_argument('--slip', help='Non-negative scalar slip, or original; omit to select')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--list', action='store_true', help='List available models without starting a server')
    parser.add_argument('--start-delay', type=float, metavar='SECONDS',
                        help='Start walking automatically this many seconds after the simulator is up, '
                             'instead of waiting for Enter (for unattended recording)')
    args = parser.parse_args()
    os.chdir(ROOT)
    options = choices(args.root.resolve(), args.convergence_root.resolve(),
                      args.submission_root.resolve(), args.scratch_root.resolve())
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
            if args.start_delay is None:
                input('Open/reconnect RoboViz to localhost:3200, then press Enter to walk (30 seconds)... ')
            else:
                print(f'[Viewer] Simulator ready; RoboViz can reconnect to localhost:3200. '
                      f'Walking starts in {args.start_delay:g} s.', flush=True)
                time.sleep(max(0.0, args.start_delay))
            run_demo(selected[:3], args.seed, architecture=selected[3], slip=slip)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\n[Viewer] Stopped.')
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'[Viewer error] {error}', file=sys.stderr)
        sys.exit(1)

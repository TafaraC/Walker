"""Managed-server, passively audited oracle stress sweep; never trains policies."""
import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys

from contact_slip.run import build
from contact_slip.sensitivity import evaluate, report
from variable_training.common import ROOT, digest, read, save, simulator_lock
from variable_training.server import require_free_ports

DEFAULT = ROOT/'experiments/optional_robustness/oracle_wide_v1'


def plan(slips, seeds, steps):
    if not slips or any(not math.isfinite(x) or x <= 0 for x in slips):
        raise ValueError('Slip values must be finite and positive')
    if len(set(slips)) != len(slips) or .1 not in slips:
        raise ValueError('Include reference slip 0.1 exactly once; no duplicate slips')
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Seeds must be unique uint32 values')
    if steps < 1:
        raise ValueError('Steps must be positive')
    return dict(protocol='optional_oracle_wide_v1', suite='oracle_wide_scalar',
                scene_method='scalar', slips=sorted(slips), seeds=seeds, steps=steps,
                policies={'oracle': None},
                optional_sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')
                                  if not p.name.startswith('test_')})


def summarize(rows, slips):
    """Absolute outcomes and paired retention; no oracle/random normalization."""
    reference={r['seed']:r['metrics']['return_target_progress_m'] for r in rows if r['slip']==.1}
    groups=[]
    for slip in slips:
        rs=[r for r in rows if r['slip']==slip]
        if not rs:
            raise ValueError(f'Missing slip {slip}')
        values=[r['metrics']['return_target_progress_m'] for r in rs]
        retention=[r['metrics']['return_target_progress_m']/reference[r['seed']]
                   for r in rs if reference[r['seed']]>1e-8]
        seconds=sum(r['metrics']['active_seconds'] for r in rs)
        if seconds <= 0:
            raise ValueError('Invalid active duration')
        mean=lambda k:sum(r['metrics'][k] for r in rs)/len(rs)
        groups.append(dict(slip=slip, runs=len(rs), progress_mean_m=sum(values)/len(values),
                           progress_min_m=min(values), progress_max_m=max(values),
                           paired_retention_mean=sum(retention)/len(retention) if retention else None,
                           retention_valid_n=len(retention), forward_velocity_mps=mean('mean_forward_velocity_mps'),
                           direction_error=mean('mean_target_direction_error'),
                           targets_total=sum(r['metrics']['targets_reached'] for r in rs),
                           falls_total=sum(r['metrics']['falls'] for r in rs),
                           falls_per_active_minute=sum(r['metrics']['falls'] for r in rs)*60/seconds))
    return groups


def write_report(root, settings, smoke):
    # Existing report verifies completed receipts and audits every contact CSV.
    report(root,settings,smoke=smoke)
    source=root/('smoke_report.json' if smoke else 'report.json')
    data=read(source)
    groups=summarize(data['runs'],settings['slips'])
    name='smoke_interpretation' if smoke else 'interpretation'
    save(root/f'{name}.json',dict(protocol=settings['protocol'],groups=groups,
                                identical_recorded_classes=data['identical_recorded_classes'],
                                source_sha256=digest(source)))
    lines=['# Optional oracle wide-slip stress sweep','',
           'Exploratory evidence only; excluded from the completed paper dataset.',
           'Each run holds scalar scene slip fixed. Fresh owned server per run; passive ODE audit.',
           'Retention is paired to scalar 0.1, not the legacy original scene and not random-normalized return.', '',
           '| Slip | Runs | Progress mean m | Range m | Retention | Forward m/s | Direction error | Targets | Falls/min |',
           '|---|---|---|---|---|---|---|---|---|']
    for g in groups:
        retention='undefined' if g['paired_retention_mean'] is None else f"{g['paired_retention_mean']:.3f}"
        lines.append(f"| {g['slip']:g} | {g['runs']} | {g['progress_mean_m']:.3f} | "
                     f"{g['progress_min_m']:.3f}–{g['progress_max_m']:.3f} | {retention} | "
                     f"{g['forward_velocity_mps']:.3f} | {g['direction_error']:.3f} | "
                     f"{g['targets_total']} | {g['falls_per_active_minute']:.3f} |")
    lines += ['', '## Review before selecting training bounds','',
              '- Read contact_audit.json and the source report: enabled solver slip must respond to the intervention.',
              '- Low progress/velocity or zero targets can indicate functional failure without falling.',
              '- Seed ranges are descriptive, not confidence intervals; report absolute metrics alongside retention.',
              '- Expert failure does not establish that the task is feasible for RL or another policy.',
              '- No automatic failure threshold, training-range selection, RL training or policy promotion occurs.',
              '- Contact audits include startup, recovery and ball contacts; they are not foot-labelled sliding measurements.']
    if data['identical_recorded_classes']:
        lines += ['', '**WARNING: recorded slip/mode classes are identical across all runs. Do not infer changed dynamics from scene settings alone.**']
    if smoke:
        lines += ['', '**Smoke uses one task seed per slip; it is a diagnostic, not the full crossed-seed study.**']
    (root/f'{name}.md').write_text('\n'.join(lines)+'\n')
    print(root/f'{name}.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','smoke','evaluate','report','report-smoke'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--slips',type=float,nargs='+',default=[.1,1,5,10,20,50,100])
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    parser.add_argument('--steps',type=int,default=3000)
    args=parser.parse_args()
    settings=plan(args.slips,args.seeds,args.steps)
    root=args.output.resolve()
    if (root/'plan.json').exists() and read(root/'plan.json') != settings:
        raise ValueError('Plan or optional source changed; use a new output directory and preserve old results')
    if args.stage.startswith('report'):
        if not (root/'plan.json').exists(): raise ValueError('No saved experiment plan')
        write_report(root,settings,args.stage=='report-smoke')
        return
    if args.stage=='plan':
        save(root/'plan.json',settings)
        print(f'{root / "plan.json"}: {len(settings["slips"])} smoke runs; '
              f'{len(settings["slips"])*len(settings["seeds"])} full runs')
        return
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
        save(root/'plan.json',settings)
        smoke=args.stage=='smoke'
        evaluate(root,settings,build(),smoke=smoke)
        write_report(root,settings,smoke)


if __name__=='__main__':
    main()

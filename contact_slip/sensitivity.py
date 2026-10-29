"""Passive solver-bound slip audit and crossed-seed locomotion sensitivity pilot."""
import argparse
from collections import Counter
import csv
import math
import signal
from pathlib import Path
import subprocess
import sys

import numpy as np
from contact_slip.run import build, hook_env, native_runtime, server, selftest, PACKAGE
from variable_training.common import ROOT, begin, commit, digest, read, save, simulator_lock, source_hashes, verified
from variable_training.server import SCENE, SYSTEM, launch, require_free_ports

PROTOCOL = 'passive_slip_sensitivity_v1'
DEFAULT = ROOT/'experiments/slip_sensitivity/v1'


def plan(suite='standard'):
    if suite == 'scalar':
        return dict(protocol=PROTOCOL, suite='oracle_scalar_scene_slip_v1', scene_method='scalar',
                    slips=[.025,.1,.2,1.,10.], seeds=[101,202,303,404,505],
                    steps=3000, policies={'oracle':None})
    if suite == 'stress':
        return dict(protocol=PROTOCOL, suite='oracle_large_scene_slip_v1',
                    slips=[.1, .5, 1., 2., 5., 10.], seeds=[101,202,303,404,505],
                    steps=3000, policies={'oracle': None})
    if suite != 'standard':
        raise ValueError('Unknown sensitivity suite')
    return dict(protocol=PROTOCOL, slips=[.025, .1, .2], seeds=[101,202,303,404,505],
                steps=3000, policies={'oracle': None, 'bc': 'bc_walk_policy.pth'})


def audit(path):
    values = [[], []]
    enabled = [0, 0]
    count = 0
    classes = Counter()
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            count += 1
            if int(row['contact']) != count or row['original_mode'] != row['applied_mode']:
                raise ValueError('Contact sequence or unchanged-mode verification failed')
            if float(row['original_slip1']) != float(row['applied_slip']):
                raise ValueError('Audit modified slip')
            classes[(float(row['original_slip1']),float(row['original_slip2']),int(row['original_mode']))] += 1
            for axis in range(2):
                value = float(row[f'original_slip{axis+1}'])
                bit = int(row[f'slip{axis+1}_enabled'])
                if not math.isfinite(value) or value < 0 or bit not in (0, 1):
                    raise ValueError('Invalid contact slip evidence')
                if bit != bool(int(row['original_mode']) & (0x100 << axis)):
                    raise ValueError('Inconsistent slip enable bit')
                enabled[axis] += bit
                values[axis].append(value if bit else 0.)
    if not count:
        raise ValueError('No field contacts intercepted; physics audit failed')
    return dict(count=count, includes_startup_recovery_and_ball=True,
                contact_classes=[dict(slip1=a,slip2=b,mode=mode,count=n)
                                 for (a,b,mode),n in sorted(classes.items())],
                axes=[dict(enabled_contacts=n, mean=float(np.mean(v)), minimum=min(v), maximum=max(v))
                      for n, v in zip(enabled, values)])


def evaluate(root, settings, library, smoke=False):
    identity = dict(plan=settings, sources=source_hashes(),
                    audit_sources={p.name:digest(p) for p in PACKAGE.glob('*.py')},
                    native_runtime=native_runtime(), library=digest(library),
                    models={k:digest(ROOT/v) for k,v in settings['policies'].items() if v},
                    expert_data=digest(ROOT/'expert_train_data.npz'),
                    environment={str(p):digest(p) for p in [SCENE,SYSTEM/'rcssserver3d.rb',
                                 SYSTEM/'naosoccersim.rb',Path.home()/'.simspark/spark.rb']})
    for slip in settings['slips']:
        for seed in settings['seeds'][:1] if smoke else settings['seeds']:
            for label, model in ({'oracle':None} if smoke else settings['policies']).items():
                directory=root/('smoke' if smoke else 'runs')/str(slip)/f'{label}_{seed}'
                key=dict(identity,slip=slip,seed=seed,label=label)
                if verified(directory,key):
                    print(f'[Resume] slip={slip} {label} seed={seed}',flush=True)
                    continue
                begin(directory)
                command=[sys.executable,'-u',str(ROOT/'evaluate_policy.py'),'--policy','oracle' if model is None else 'bc',
                         '--label',label,'--seed',str(seed),'--steps',str(settings['steps']),
                         '--output-dir',str(directory/'raw'),'--expert-data',str(ROOT/'expert_train_data.npz'),
                         '--environment-id',f'sensitivity_{slip}','--field-slip',str(slip)]
                if model: command += ['--model',str(ROOT/model)]
                try:
                    with server(directory,'audit',dict(low=.025,high=.2,mean=.1,std=.035),seed,library,
                                scene_slip=slip,scene_method=settings.get('scene_method','legacy')) as process:
                        launch(command,directory/'evaluation.log',server=process)
                    result=read(directory/'raw'/f'{label}_seed_{seed}.json')
                    if result['active_steps'] != settings['steps'] or result['seed'] != seed:
                        raise ValueError('Incomplete evaluation')
                    with np.load(result['trace'],allow_pickle=False) as trace:
                        if not trace.files or any(len(trace[k])!=settings['steps'] or not np.isfinite(trace[k]).all() for k in trace.files):
                            raise ValueError('Incomplete/nonfinite trace')
                    save(directory/'contact_audit.json',audit(directory/'contacts.csv'))
                    commit(directory,key)
                except BaseException as error:
                    save(directory/'failure.json',{'error':repr(error)})
                    raise


def report(root, settings, smoke=False):
    rows=[]
    for slip in settings['slips']:
        for seed in settings['seeds'][:1] if smoke else settings['seeds']:
            for label in ['oracle'] if smoke else settings['policies']:
                directory=root/('smoke' if smoke else 'runs')/str(slip)/f'{label}_{seed}'
                receipt=read(directory/'completed.json')
                if receipt['identity']['plan']!=settings: raise ValueError('Different plan')
                verified(directory,receipt['identity'])
                rows.append(dict(slip=slip,seed=seed,label=label,
                                 metrics=read(directory/'raw'/f'{label}_seed_{seed}.json'),
                                 contacts=audit(directory/'contacts.csv')))
    lines=['# Passive slip sensitivity','',
           f'Scene setter method: {settings.get("scene_method","legacy")}.',
           f'{1 if smoke else len(settings["seeds"])} crossed task seed(s), {settings["steps"]} active steps per run. Contact summaries include startup/recovery and ball contacts.',
           'Slip statistics use zero when the corresponding ODE slip flag is disabled. These are solver inputs, not measured foot sliding.', '',
           '| Policy | Field slip | Progress m | Paired change from .1 m | Falls total | ODE slip1 range | ODE slip2 range |',
           '|---|---:|---:|---:|---:|---|---|']
    for label in ['oracle'] if smoke else settings['policies']:
        nominal={r['seed']:r['metrics']['return_target_progress_m'] for r in rows if r['label']==label and r['slip']==.1}
        for slip in settings['slips']:
            group=[r for r in rows if r['label']==label and r['slip']==slip]
            means=[r['metrics']['return_target_progress_m'] for r in group]
            delta=[r['metrics']['return_target_progress_m']-nominal[r['seed']] for r in group]
            ranges=[f"{min(r['contacts']['axes'][a]['minimum'] for r in group):.6g}–{max(r['contacts']['axes'][a]['maximum'] for r in group):.6g}" for a in range(2)]
            lines.append(f"| {label} | {slip} | {np.mean(means):.3f} | {np.mean(delta):+.3f} | {sum(r['metrics']['falls'] for r in group)} | {' | '.join(ranges)} |")
    lines += ['', 'Inspect enable counts and per-run distributions in report.json before attributing weak response to robustness.',
              'Similar aggregate ranges can mask different contact classes. No automatic pass/fail claim of adaptation is made.']
    signatures={tuple((c['slip1'],c['slip2'],c['mode']) for c in r['contacts']['contact_classes']) for r in rows}
    if len(signatures)==1:
        lines += ['', '**WARNING: every run has the same recorded slip/mode classes.**',
                  'Different contact counts or progress do not demonstrate an effective slip change. Diagnose the configuration path before expanding this experiment.']
    if smoke:
        lines += ['', 'Single-seed diagnostic only: not evidence of statistical robustness or a safe training range.']
    name='smoke_report' if smoke else 'report'
    save(root/f'{name}.json',dict(protocol=PROTOCOL,suite=settings.get('suite','standard'),runs=rows,
                                 identical_recorded_classes=len(signatures)==1))
    (root/f'{name}.md').write_text('\n'.join(lines)+'\n')
    print(root/f'{name}.md')


def main():
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['selftest','smoke','evaluate','report','report-smoke'])
    parser.add_argument('--suite',choices=['standard','stress','scalar'],default='standard')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    defaults={'standard':DEFAULT, 'stress':ROOT/'experiments/slip_sensitivity/oracle_stress_v1',
              'scalar':ROOT/'experiments/slip_sensitivity/oracle_scalar_v1'}
    root=(args.output or defaults[args.suite]).resolve()
    settings=plan(args.suite)
    if args.stage=='selftest':
        import tempfile
        library=build(); selftest(library)
        with tempfile.TemporaryDirectory(prefix='passive-slip-') as tmp:
            path=Path(tmp)/'contacts.csv'
            subprocess.run([str(library.parent/'ode_probe')],check=True,
                           env=hook_env(library,path,'audit',dict(low=.025,high=.2,mean=.1,std=.035),42))
            result=audit(path)
            assert result['count']==1000
            assert all(a['minimum']==a['maximum']==.005 and a['enabled_contacts']==1000 for a in result['axes'])
        print('[Selftest] Passive audit preserved .005 slip and mode flags; no simulator launched.')
        return
    if (root/'plan.json').exists() and read(root/'plan.json')!=settings:
        raise ValueError('Plan changed; choose a new output directory')
    if args.stage in ('report','report-smoke'):
        report(root,settings,smoke=args.stage=='report-smoke'); return
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
        save(root/'plan.json',settings)
        evaluate(root,settings,build(),smoke=args.stage=='smoke')
        report(root,settings,smoke=args.stage=='smoke')


if __name__=='__main__': main()

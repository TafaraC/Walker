"""Append FiLM-only deployments to a verified prior BC comparison; never rerun controls."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from variable_training.common import ROOT, begin, commit, digest, read, save, verified, simulator_lock
from variable_training.server import simulator, launch, require_free_ports
from context_training.evaluate import job_identity as old_identity, validate_result
from .models import load_checkpoint

PROTOCOL='film_bc_append_evaluation_v1'
DEFAULT_ROOT=ROOT/'experiments/context_conditioning/bc_film_v1'
DEFAULT_BASELINE=ROOT/'experiments/context_conditioning/bc_input_v1/evaluation_all_v1'


def verify_baseline(baseline):
    spec=read(baseline/'plan.json')
    if spec.get('protocol')!='context_bc_evaluation_v1':
        raise ValueError('Expected prior context BC evaluation')
    # Report aggregates are not trusted as a substitute for raw completion receipts.
    for row in spec['rounds']:
        for controller in spec['controllers']:
            directory=baseline/'runs'/row['id']/controller['label']
            identity=old_identity(spec,row,controller)
            if not verified(directory,identity):
                raise ValueError(f'Baseline incomplete: {directory}')
            validate_result(directory,identity)
    return spec


def make_spec(training_root, baseline, suite, modes):
    previous=verify_baseline(baseline)
    controls_plan=read(baseline.parent/'plan.json')
    if digest(baseline.parent/'plan.json')!=previous['training_plan_sha256']:
        raise ValueError('Baseline training plan changed')
    plan=read(training_root/'plan.json')
    if plan.get('protocol')!='film_bc_pilot_v1' or 'film' not in plan['architectures']:
        raise ValueError('Expected completed FiLM training plan')
    for key in ('dataset','initializer_sha256','settings','seeds','device','torch_version','numpy_version'):
        if plan[key]!=controls_plan[key]:
            raise ValueError(f'FiLM/control training mismatch: {key}')
    # Reuse only when existing physics/controller sources still match the baseline.
    for name,value in previous['sources'].items():
        if digest(ROOT/name)!=value:
            raise ValueError(f'Baseline source changed: {name}; new matched comparison required')
    for name,value in previous['environment'].items():
        if digest(name)!=value:
            raise ValueError(f'Baseline environment changed: {name}')
    selected=[r for r in previous['rounds'] if suite=='all' or r['suite']==suite]
    if suite=='smoke':
        selected=previous['rounds'][:1]
    if not selected:
        raise ValueError('Requested suite is not present in the prior comparison')
    controllers=[]
    for seed in plan['seeds']:
        directory=training_root/'film'/f'seed_{seed}'
        if not verified(directory,dict(plan_sha256=digest(training_root/'plan.json'),architecture='film',seed=seed)):
            raise ValueError(f'FiLM training incomplete: {directory}')
        _,architecture=load_checkpoint(directory/'policy.pt')
        if architecture!='film':
            raise ValueError('Wrong checkpoint architecture')
        for mode in modes:
            controllers.append(dict(label=f'film_seed_{seed}_{mode}',architecture='film',training_seed=seed,
                mode=mode,model=str(directory/'policy.pt'),model_sha256=digest(directory/'policy.pt')))
    sources=dict(previous['sources'])
    for name in ('film_training/__init__.py','film_training/models.py','film_training/evaluate.py',
                 'film_training/eval_worker.py'):
        sources[name]=digest(ROOT/name)
    return dict(protocol=PROTOCOL,training_plan_sha256=digest(training_root/'plan.json'),
        baseline=str(baseline),baseline_plan_sha256=digest(baseline/'plan.json'),
        suite=suite,modes=modes,steps=previous['steps'],rounds=selected,controllers=controllers,
        sources=sources,environment=previous['environment'],metrics=previous['metrics'],
        evaluation_role='append only; prior confirmation draws are now reused comparison cases')


def job_identity(spec,row,controller):
    result=old_identity(spec,row,controller)
    result.update(protocol=PROTOCOL,baseline_plan_sha256=spec['baseline_plan_sha256'])
    return result


def evaluate(output, spec, retries):
    for row in spec['rounds']:
        for controller in spec['controllers']:
            directory = output/'runs'/row['id']/controller['label']
            identity = job_identity(spec, row, controller)
            if verified(directory, identity):
                validate_result(directory, identity)
                print(f'[Resume] {row["id"]} {controller["label"]}', flush=True)
                continue
            for attempt in range(retries+1):
                begin(directory)
                save(directory/'job.json', identity)
                print(f'[BC evaluation] {row["id"]} slip={row["slip"]:.8g} '
                      f'{controller["label"]} supplied={identity["supplied_slip"]}', flush=True)
                try:
                    with simulator(directory, row['slip']) as manifest:
                        metadata = {k:v for k,v in manifest.items() if k != 'server_process'}
                        metadata.update(evaluation_protocol=PROTOCOL,
                            evaluator_sha256=digest(Path(__file__).parent/'eval_worker.py'),
                            supplied_slip=identity['supplied_slip'])
                        save(directory/'environment_manifest.json',metadata)
                        launch([sys.executable, '-u', '-m', 'film_training.eval_worker',
                                str(directory/'job.json')], directory/'evaluation.log',
                               server=manifest['server_process'])
                    validate_result(directory, identity)
                    if controller['model'] and digest(controller['model']) != controller['model_sha256']:
                        raise ValueError('Model changed during evaluation')
                    if any(digest(ROOT/name) != value for name,value in spec['sources'].items()):
                        raise ValueError('Source changed during evaluation')
                    if any(digest(name) != value for name,value in spec['environment'].items()):
                        raise ValueError('Environment configuration changed during evaluation')
                    commit(directory, identity)
                    break
                except Exception as error:
                    save(directory/'failure.json', dict(error=repr(error), attempt=attempt+1))
                    if attempt == retries or not isinstance(error, (RuntimeError, subprocess.CalledProcessError, OSError)):
                        raise
                    print('[Retry] Failed run preserved; restarting its owned server.', flush=True)


def report(output,spec):
    from summarize_evaluation import jerk_cost,sliced_wasserstein
    baseline=Path(spec['baseline'])
    if digest(baseline/'plan.json')!=spec['baseline_plan_sha256']:
        raise ValueError('Baseline plan changed')
    previous=verify_baseline(baseline)
    # Include the saved oracle/hidden/input-correct controls, not duplicate live deployments.
    controls=[c for c in previous['controllers'] if c['architecture'] in ('oracle','hidden') or c['mode']=='correct']
    results=[]
    for row in spec['rounds']:
        with np.load(baseline/'runs'/row['id']/'oracle'/'trace.npz',allow_pickle=False) as data:
            expert=data['joint_positions_deg'].copy()
        oracle=read(baseline/'runs'/row['id']/'oracle'/'result.json')['return_target_progress_m']
        for c in controls+spec['controllers']:
            reused=c['architecture']!='film'
            directory=(baseline if reused else output)/'runs'/row['id']/c['label']
            identity=old_identity(previous,row,c) if reused else job_identity(spec,row,c)
            if not verified(directory,identity):
                raise ValueError(f'Missing FiLM evaluation: {directory}')
            result=validate_result(directory,identity)
            with np.load(directory/'trace.npz',allow_pickle=False) as data:
                jerk,rate=jerk_cost(data['joint_positions_deg'],data['episode_id'],spec['metrics']['smoothing_window'])
                sw=0. if c['architecture']=='oracle' else sliced_wasserstein(data['joint_positions_deg'],expert,
                    spec['metrics']['wasserstein_projections'],spec['metrics']['wasserstein_seed'])
            result.update(reused_baseline=reused,oracle_return_ratio=result['return_target_progress_m']/oracle if oracle>1e-8 else None,
                jerk_integrated_rad2_per_s5=jerk,jerk_cost_rate_rad2_per_s6=rate,joint_occupancy_sliced_wasserstein=sw)
            results.append(result)
    lookup={(r['row']['id'],r['controller']['label']):r for r in results}
    paired=[]
    for r in results:
        c=r['controller']
        if c['architecture']!='film':
            continue
        references=([f'hidden_seed_{c["training_seed"]}_none',f'input_seed_{c["training_seed"]}_correct']
            if c['mode']=='correct' else [f'film_seed_{c["training_seed"]}_correct'])
        for label in references:
            control=lookup[(r['row']['id'],label)]
            paired.append(dict(round=r['row']['id'],policy=c['label'],reference=label,
                training_seed=c['training_seed'],progress_delta_m=r['return_target_progress_m']-control['return_target_progress_m'],
                falls_delta=r['falls']-control['falls']))
    summaries=[]
    for suite in dict.fromkeys(r['row']['suite'] for r in results):
        for c in controls+spec['controllers']:
            rs=[r for r in results if r['row']['suite']==suite and r['controller']==c]
            summaries.append(dict(suite=suite,policy=c['label'],n=len(rs),
                mean_progress_m=float(np.mean([r['return_target_progress_m'] for r in rs])),
                total_falls=sum(r['falls'] for r in rs),reused_baseline=c['architecture']!='film'))
    save(output/'comparison.json',dict(protocol=PROTOCOL,runs=results,paired_differences=paired,summary=summaries))
    lines=['# Appended FiLM BC comparison','',
        'Oracle, hidden BC and input-correct BC are verified saved runs, not newly simulated controls.',
        'Slips/task seeds match exactly; startup states may differ. Prior confirmation rounds are reused, not untouched FiLM tests.',
        'Consult the repeatability study before interpreting small differences. No training replicas are pooled into significance claims.','',
        '| Suite | Policy | Runs | Mean progress m | Falls | Reused control |','|---|---|---:|---:|---:|---|']
    for s in summaries:
        lines.append(f'| {s["suite"]} | {s["policy"]} | {s["n"]} | {s["mean_progress_m"]:.4f} | {s["total_falls"]} | {s["reused_baseline"]} |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output/"comparison.md"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','evaluate','report'))
    parser.add_argument('--training-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--suite',choices=('smoke','grouped','confirmation','nominal','all'),default='all')
    parser.add_argument('--modes',nargs='+',choices=('correct','constant','wrong'),default=['correct'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--retries',type=int,default=1)
    args=parser.parse_args()
    if args.retries<0 or len(set(args.modes))!=len(args.modes) or 'correct' not in args.modes:
        parser.error('Require nonnegative retries and unique modes including correct')
    root=args.training_root.resolve()
    spec=make_spec(root,args.baseline.resolve(),args.suite,args.modes)
    output=(args.output or root/f'evaluation_append_{args.suite}_v1').resolve()
    print(f'[Plan] {len(spec["rounds"])*len(spec["controllers"])} NEW FiLM runs; no control runs')
    with simulator_lock():
        path=output/'plan.json'
        if path.exists() and read(path)!=spec:
            raise ValueError('Inputs changed; choose a new --output')
        if args.stage=='report' and not path.exists():
            raise ValueError('No saved plan')
        save(path,spec)
        if args.stage=='evaluate':
            from check_simulator_config import main as preflight
            if preflight():
                raise RuntimeError('Simulator preflight failed')
            require_free_ports()
            evaluate(output,spec,args.retries)
            report(output,spec)
        elif args.stage=='report':
            report(output,spec)


if __name__=='__main__':
    main()

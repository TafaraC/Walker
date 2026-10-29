"""Owned-server evaluation of independent truncated-Gaussian ODE field contacts."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import numpy as np
from variable_training.common import ROOT, MODELS, begin, commit, digest, read, save, simulator_lock, verified
from variable_training.server import prepare, require_free_ports, launch, SCENE, SYSTEM

PACKAGE = Path(__file__).resolve().parent
PROTOCOL = 'independent_effective_contact_slip_v1'
DEFAULT = ROOT/'experiments/contact_slip/gaussian_v1'


def native_runtime():
    plugin = Path('/usr/lib/simspark/odeimps.so').resolve()
    dependencies = subprocess.check_output(['ldd', str(plugin)], text=True)
    files = [plugin]
    for line in dependencies.splitlines():
        if 'libode.so' in line:
            files += [Path(token).resolve() for token in line.split() if token.startswith('/')]
    if len(files) < 2:
        raise RuntimeError('Cannot identify the installed ODE runtime')
    return {str(path):digest(path) for path in files}


def build():
    sources = {name: digest(PACKAGE/name) for name in ('contact_slip.cpp','ode_probe.cpp')}
    compiler = subprocess.check_output(['g++','--version'], text=True).splitlines()[0]
    identity = dict(sources=sources, compiler=compiler, native_runtime=native_runtime(),
                    headers={str(p):digest(p) for p in sorted(Path('/usr/include/ode').glob('*.h'))})
    if not identity['headers']:
        raise RuntimeError('ODE development headers required (libode-dev)')
    output = PACKAGE/'build'/hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    if verified(output, identity):
        return output/'libcontact_slip.so'
    begin(output)
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared',str(PACKAGE/'contact_slip.cpp'),
                    '-o',str(output/'libcontact_slip.so'),'-lode','-ldl','-pthread'], check=True)
    subprocess.run(['g++','-std=c++17','-O2',str(PACKAGE/'ode_probe.cpp'),
                    '-o',str(output/'ode_probe'),'-lode'], check=True)
    commit(output, identity)
    return output/'libcontact_slip.so'


def hook_env(library, trace, mode, config, seed):
    env = dict(os.environ)
    if env.get('LD_PRELOAD'):
        raise RuntimeError('Unset LD_PRELOAD before this experiment; do not combine interposers')
    env.update(LD_PRELOAD=str(library), CONTACT_TRACE=str(trace), CONTACT_MODE=mode,
               CONTACT_LOW=str(config['low']), CONTACT_HIGH=str(config['high']),
               CONTACT_MEAN=str(config['mean']), CONTACT_STD=str(config['std']), CONTACT_SEED=str(seed))
    return env


def audit_trace(path, mode, config):
    count, mean, m2, low, high = 0, 0., 0., float('inf'), -float('inf')
    with path.open() as stream:
        for row in csv.DictReader(stream):
            count += 1
            value = float(row['applied_slip'])
            if int(row['contact']) != count or not math.isfinite(value):
                raise ValueError('Invalid contact trace sequence')
            if not config['low'] <= value <= config['high']:
                raise ValueError('Contact draw outside truncation bounds')
            if mode=='constant' and value != config['mean']:
                raise ValueError('Constant control did not apply the specified slip')
            for key in ('x','y','z','original_slip1','original_slip2'):
                if not math.isfinite(float(row[key])):
                    raise ValueError('Non-finite contact evidence')
            delta = value-mean; mean += delta/count; m2 += delta*(value-mean)
            low, high = min(low,value), max(high,value)
    if not count or (mode=='gaussian' and count>1 and low==high):
        raise ValueError('No verified varying field contacts; refuse this experiment')
    return dict(count=count, mean=mean, variance=m2/count, minimum=low, maximum=high,
                includes_startup_and_recovery=True)


def selftest(library):
    config = dict(low=.025,high=.2,mean=.1,std=.035)
    with tempfile.TemporaryDirectory(prefix='contact-slip-') as tmp:
        paths = []
        for index, mode in enumerate(('gaussian','gaussian','constant')):
            path = Path(tmp)/f'{index}.csv'
            subprocess.run([str(library.parent/'ode_probe')], env=hook_env(library,path,mode,config,42), check=True)
            stats = audit_trace(path,mode,config)
            if stats['count'] != 1000:
                raise ValueError('Ground selection intercepted the wrong contacts')
            if mode=='gaussian' and (abs(stats['mean']-.1)>.01 or not .0005<stats['variance']<.002):
                raise ValueError('Sample moments inconsistent with intended Gaussian')
            paths.append(path)
        if paths[0].read_bytes() != paths[1].read_bytes():
            raise ValueError('Seeded contact stream is not reproducible')
    print('[Selftest] 1,000 ground contacts sampled; wall/out-of-field contacts excluded; replay verified.')


@contextmanager
def server(directory, mode, config, seed, library, scene_slip=.1, scene_method='legacy'):
    if scene_method not in ('legacy','scalar'):
        raise ValueError('Unknown scene setter method')
    require_free_ports()
    original = digest(SCENE)
    # Preserve historical diagnostic arms; scalar arm converts explicitly below.
    manifest = prepare(directory, scene_slip, None, setter_method='legacy')
    if scene_method=='scalar':
        from contact_slip.scalar_scene import convert
        manifest = convert(directory,manifest,scene_slip)
    manifest.update(protocol='passive_slip_sensitivity_v1' if mode=='audit' else PROTOCOL,
                    contact_mode=mode, contact_seed=seed,
                    distribution=None if mode=='audit' else config,
                    library_sha256=digest(library) if mode!='scene' else None,
                    scope='ODE ground-plane contact points within x=+-15,y=+-10,z_plane=0')
    save(directory/'environment_manifest.json',manifest)
    env = dict(os.environ) if mode=='scene' else hook_env(library,directory/'contacts.csv',mode,config,seed)
    with (directory/'server.log').open('w') as log:
        process = subprocess.Popen(['rcssserver3d','--script-path',str(directory/'rcssserver3d.rb')],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited: {directory}/server.log')
                content = (directory/'server.log').read_text()
                if manifest['startup_marker'] in content.splitlines() and (mode=='scene' or 'CONTACT_SLIP_READY ' in content):
                    for name, sha in manifest['config_sha256'].items():
                        if digest(directory/name)!=sha:
                            raise ValueError('Startup script changed')
                    if digest(directory/'soccer.rsg') != manifest['sha256']:
                        raise ValueError('Scene changed')
                    break
                time.sleep(.1)
            else:
                raise RuntimeError(f'Scene/hook startup verification missing: {directory}/server.log')
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            if digest(SCENE)!=original:
                raise RuntimeError('Installed scene changed')


def plan(low=.025, high=.2, mean=.1, std=.035):
    if not all(math.isfinite(v) for v in (low,high,mean,std)) or not (0<=low<high and low<=mean<=high and std>0):
        raise ValueError('Invalid truncated Gaussian')
    a,b=(low-mean)/std,(high-mean)/std
    if .5*(math.erf(b/math.sqrt(2))-math.erf(a/math.sqrt(2)))<1e-4:
        raise ValueError('Truncation probability too small')
    return dict(protocol=PROTOCOL, distribution=dict(low=low,high=high,mean=mean,std=std),
                seeds=[101,202,303,404,505], contact_seeds=[60101,60202,60303,60404,60505], steps=3000,
                modes=['scene','constant','gaussian'], policies={'oracle':None,**MODELS})


def evaluate(root, settings, library, smoke=False):
    # Pin physical/runtime sources without touching either existing pipeline.
    from variable_training.common import source_hashes
    identity = dict(plan=settings, sources=source_hashes(),
                    contact_sources={p.name:digest(p) for p in PACKAGE.glob('*.py')},
                    native_library=digest(library), native_runtime=native_runtime(),
                    models={k:digest(ROOT/v) for k,v in MODELS.items()},
                    expert_data=digest(ROOT/'expert_train_data.npz'),
                    environment={str(p):digest(p) for p in [SCENE,SYSTEM/'rcssserver3d.rb',
                                 SYSTEM/'naosoccersim.rb',Path.home()/'.simspark/spark.rb']})
    modes = ['gaussian'] if smoke else settings['modes']
    for mode in modes:
        for seed,contact_seed in zip(settings['seeds'][:1] if smoke else settings['seeds'],settings['contact_seeds']):
            for label,model in ({'oracle':None} if smoke else settings['policies']).items():
                directory=root/('smoke' if smoke else 'runs')/mode/f'{label}_{seed}'
                run_identity=dict(identity,mode=mode,seed=seed,contact_seed=contact_seed,label=label)
                if verified(directory,run_identity):
                    print(f'[Resume] {mode} {label} {seed}',flush=True); continue
                begin(directory)
                kind='oracle' if model is None else 'gail' if label.startswith('gail') else 'bc'
                command=[sys.executable,'-u',str(ROOT/'evaluate_policy.py'),'--policy',kind,'--label',label,
                         '--expert-data',str(ROOT/'expert_train_data.npz'),'--steps',str(settings['steps']),
                         '--seed',str(seed),'--output-dir',str(directory/'raw'),
                         '--environment-id',f'contact_{mode}','--field-slip','0.1']
                if model: command+=['--model',str(ROOT/model)]
                try:
                    with server(directory,mode,settings['distribution'],contact_seed,library) as process:
                        launch(command,directory/'evaluation.log',server=process)
                    summary_path=directory/'raw'/f'{label}_seed_{seed}.json'
                    result=read(summary_path)
                    if result['active_steps']!=settings['steps'] or result['seed']!=seed:
                        raise ValueError('Incomplete evaluation')
                    with np.load(result['trace'],allow_pickle=False) as trace:
                        if any(len(trace[k])!=settings['steps'] or not np.isfinite(trace[k]).all() for k in trace.files):
                            raise ValueError('Incomplete/non-finite trace')
                    contacts=None if mode=='scene' else audit_trace(directory/'contacts.csv',mode,settings['distribution'])
                    result.update(protocol=PROTOCOL,contact_mode=mode,contact_seed=contact_seed,
                                  contact_statistics=contacts,
                                  field_contact_slip_semantics='static scene value; effective solver slip specified by contact_mode')
                    save(summary_path,result)
                    commit(directory,run_identity)
                except BaseException as error:
                    save(directory/'failure.json',{'error':repr(error)}); raise


def report(root,settings):
    rows=[]
    for mode in settings['modes']:
        for seed in settings['seeds']:
            for label in settings['policies']:
                directory=root/'runs'/mode/f'{label}_{seed}'
                receipt=read(directory/'completed.json')
                if receipt['identity']['plan']!=settings: raise ValueError('Different plan')
                verified(directory,receipt['identity'])
                rows.append(read(directory/'raw'/f'{label}_seed_{seed}.json'))
    lines=['# Independent contact-slip evaluation','',
           'Effective ODE slip, not a spatial surface map. Oracle ratios are not Scorenorm.','',
           '| Mode | Policy | Mean progress (m) | Mean falls |','|---|---|---:|---:|']
    for mode in settings['modes']:
        for label in settings['policies']:
            values=[r for r in rows if r['contact_mode']==mode and r['policy_label']==label]
            lines.append(f'| {mode} | {label} | {np.mean([r["return_target_progress_m"] for r in values]):.3f} | {np.mean([r["falls"] for r in values]):.3f} |')
    save(root/'report.json',dict(protocol=PROTOCOL,runs=rows))
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build','selftest','plan','smoke','evaluate','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    for name,value in [('low',.025),('high',.2),('mean',.1),('std',.035)]:
        parser.add_argument('--'+name,type=float,default=value)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    root=args.output.resolve()
    if args.stage=='plan':
        settings=plan(args.low,args.high,args.mean,args.std)
        if (root/'plan.json').exists() and read(root/'plan.json')!=settings:
            raise ValueError('Use a new output directory for a changed plan')
        save(root/'plan.json',settings); print(root/'plan.json'); return
    if args.stage in ('build','selftest'):
        library=build()
        if args.stage=='selftest': selftest(library)
        print(library); return
    settings=read(root/'plan.json')
    if settings!=plan(**settings['distribution']): raise ValueError('Invalid or edited plan')
    if args.stage=='report': report(root,settings); return
    with simulator_lock():
        require_free_ports()
        subprocess.run([sys.executable,str(ROOT/'check_simulator_config.py')],check=True)
        subprocess.run([sys.executable,'-c',
                        'from scripts.commons.Script import Script; Script.build_cpp_modules(); '
                        'from cpp.ball_predictor import ball_predictor; '
                        'from cpp.localization import localization; from cpp.a_star import a_star'],check=True)
        library=build()
        selftest(library)
        evaluate(root,settings,library,smoke=args.stage=='smoke')


if __name__=='__main__': main()

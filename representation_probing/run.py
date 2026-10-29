"""General source-model probes and in-memory layer-reset diagnostics."""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from probing.run import actor, features, fit_probe, shuffle_blocks, make_spec as audit_source, DEFAULT_DATA
from variable_training.common import ROOT, read, save, digest, verified, commit

OUTPUT = ROOT/'experiments/representation_probing/nao_sources_v1'


def targets(trace, stride=7, horizon=10):
    o = np.asarray(trace['obs'], dtype=np.float64)
    ep = trace['episode_id']
    idx = np.asarray([i for i in range(0,len(o)-horizon,stride)
                      if np.all(ep[i:i+horizon+1] == ep[i])], dtype=int)
    if len(idx)<2 or o.shape[1]!=63 or not np.isfinite(o).all():
        raise ValueError('Invalid or insufficient trajectory')
    if not np.allclose(o[:,56]+o[:,57],1) or np.any((o[:,55]<0)|(o[:,55]>1)):
        raise ValueError('Unexpected gait-generator observation contract')
    # Generator phase is not measured stance. Reset phase values are synthetic.
    phase=np.pi*(o[:,55]+o[:,57])
    left_force=np.linalg.norm(o[:,14:17]*100,axis=1)
    right_force=np.linalg.norm(o[:,20:23]*100,axis=1)
    vals={
        'generator_phase_sin':np.sin(phase), 'generator_phase_cos':np.cos(phase),
        'left_force_contact_gt10N':(left_force>10).astype(float),
        'right_force_contact_gt10N':(right_force>10).astype(float),
        'left_force_norm_N':left_force,'right_force_norm_N':right_force,
        'head_height_m':o[:,1]/3,'head_vertical_velocity_mps':o[:,2]*2,
        'torso_roll_deg':o[:,3]*15,'torso_pitch_deg':o[:,4]*15,
    }
    for k,axis in enumerate('xyz'):
        vals[f'gyro_{axis}_degps']=o[:,5+k]*100
        vals[f'left_ankle_hip_{axis}_m']=o[:,23+k]/[8,8,5][k]
        vals[f'right_ankle_hip_{axis}_m']=o[:,26+k]/[8,8,5][k]
    vals.update(command_forward_displacement_m=o[:,58]*.5,
                command_lateral_displacement_m=o[:,59]*.5,
                command_orientation_deg=o[:,60]*45,
                phase_sin_times_roll_deg=np.sin(phase)*o[:,3]*15,
                phase_cos_times_pitch_deg=np.cos(phase)*o[:,4]*15)
    names=list(vals)
    y=np.column_stack([v[idx] for v in vals.values()])
    future={
        'future_200ms_forward_velocity_mps':np.asarray([trace['forward_velocity_mps'][i:i+horizon].mean() for i in idx]),
        'future_200ms_direction_error':np.asarray([trace['target_direction_error'][i:i+horizon].mean() for i in idx]),
        'future_200ms_roll_deg':o[idx+horizon,3]*15,
        'future_200ms_pitch_deg':o[idx+horizon,4]*15,
        'future_200ms_left_force_contact_gt10N':(left_force[idx+horizon]>10).astype(float),
        'future_200ms_right_force_contact_gt10N':(right_force[idx+horizon]>10).astype(float),
    }
    persistence=np.column_stack([trace['forward_velocity_mps'][np.maximum(idx-1,0)],trace['target_direction_error'][np.maximum(idx-1,0)],
        o[idx,3]*15,o[idx,4]*15,(left_force[idx]>10).astype(float),(right_force[idx]>10).astype(float)])
    names+=list(future)
    y=np.column_stack([y]+list(future.values()))
    if not np.isfinite(y).all():raise ValueError('Invalid target')
    # Discard synthetic generator/reset observations, including later recoveries.
    valid=(o[idx,0]>0)&(idx>0)&(ep[np.maximum(idx-1,0)]==ep[idx])
    return o[idx][valid],y[valid],idx[valid],names,persistence[valid]


def dataset(spec):
    parts={s:[] for s in ('train','validation','test')}
    names=None
    for number,e in enumerate(spec['entries']):
        if digest(e['path'])!=e['sha256']:raise ValueError('Changed source trace')
        with np.load(e['path'],allow_pickle=False) as t:
            x,y,idx,names,persist=targets(t,spec['stride'],spec['horizon'])
        parts[e['split']].append((x,y,np.full(len(x),number),idx,persist))
    return {s:tuple(np.concatenate([r[k] for r in rows]) for k in range(5)) for s,rows in parts.items()},names


def spec_for(data,stride):
    spec=audit_source(data,stride)
    spec.update(protocol='nao_general_representations_v1',stride=stride,
                alphas=[1e-6,1e-5,1e-4,.001,.01,.1,1.,10.,100.],
                targets='see target_names; no slip target',contact_threshold_N=10.,
                interventions=['reset_input','reset_middle','reset_output'])
    for p in (Path(__file__),ROOT/'representation_probing/__init__.py',ROOT/'probing/run.py',
              ROOT/'behaviors/custom/Walk/Env.py'):
        spec['sources'][str(p.relative_to(ROOT))]=digest(p)
    data_arrays,names=dataset(spec)
    spec['target_names']=names
    spec['split_counts']={s:dict(rows=len(d[0]),runs=len(np.unique(d[2]))) for s,d in data_arrays.items()}
    return spec,data_arrays


def scores(y,pred,train_y,groups,names):
    out=[]
    for k,name in enumerate(names):
        error=(y[:,k]-pred[:,k])**2
        var=np.var(y[:,k])
        item=dict(target=name,rmse=float(np.sqrt(error.mean())),
                  r2=float(1-error.mean()/var) if var>1e-12 else None,
                  mean_baseline_rmse=float(np.sqrt(((y[:,k]-train_y[:,k].mean())**2).mean())),
                  macro_run_rmse=float(np.mean([np.sqrt(error[groups==g].mean()) for g in np.unique(groups)])),
                  contact_accuracy=None,contact_balanced_accuracy=None,contact_positive_fraction=None)
        if 'contact_gt10N' in name:
            truth=y[:,k]>.5;guess=pred[:,k]>.5
            item['contact_positive_fraction']=float(truth.mean())
            item['contact_accuracy']=float((truth==guess).mean())
            item['contact_balanced_accuracy']=float(np.mean([(guess[truth==v]==v).mean() for v in (False,True)])) if len(np.unique(truth))==2 else None
        out.append(item)
    return out


def reset_layer(net,layer,seed):
    changed=copy.deepcopy(net)
    fresh=actor(seed=seed)
    index={'reset_input':0,'reset_middle':2,'reset_output':4}[layer]
    changed[index].load_state_dict(fresh[index].state_dict())
    return changed


def predict(net,x):
    with torch.inference_mode():
        return net(torch.as_tensor(x,dtype=torch.float32)).numpy()


def interventions(spec,data):
    x,y,groups,indices,_=data['test']
    rows=[]
    for label,meta in spec['models'].items():
        if digest(meta['path'])!=meta['sha256']:raise ValueError('Changed checkpoint')
        net=actor(meta['path']);original=predict(net,x)
        for kind in spec['interventions']:
            for seed in spec['control_seeds']:
                changed=reset_layer(net,kind,seed)
                delta=predict(changed,x)-original
                rows.append(dict(policy=label,intervention=kind,seed=seed,
                    action_rms_change=float(np.sqrt(np.square(delta).mean())),
                    per_run_action_rms={str(int(g)):float(np.sqrt(np.square(delta[groups==g]).mean())) for g in np.unique(groups)}))
    return rows


def representations(spec,data):
    yield 'raw_observation',{s:d[0] for s,d in data.items()}
    # Restricted input control tests whether phase + command alone explains a label.
    yield 'phase_command_only',{s:d[0][:,55:61] for s,d in data.items()}
    for label,path,seed in ([(n,m['path'],42) for n,m in spec['models'].items()]+
                            [(f'random_seed_{s}',None,s) for s in spec['control_seeds']]):
        if path is not None and digest(path)!=spec['models'][label]['sha256']:raise ValueError('Changed checkpoint')
        net=actor(path,seed)
        h={s:features(net,d[0]) for s,d in data.items()}
        for layer in (0,1):yield f'{label}_h{layer+1}',{s:v[layer] for s,v in h.items()}


def run(output,spec,data):
    groups={s:d[2] for s,d in data.items()};true={s:d[1] for s,d in data.items()}
    for label,xs in representations(spec,data):
        for seed in [None]+spec['control_seeds']:
            mode='true' if seed is None else f'shuffled_{seed}'
            directory=output/'probes'/label/mode
            identity=dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)
            if verified(directory,identity):continue
            if directory.exists() and any(directory.iterdir()):raise ValueError(f'Archive incomplete subprobe before retry: {directory}')
            ys=true if seed is None else {s:shuffle_blocks(y,groups[s],seed+1000*i) for i,(s,y) in enumerate(true.items())}
            pred,fit=fit_probe(xs,ys,groups,spec['alphas'])
            result=scores(ys['test'],pred,ys['train'],groups['test'],spec['target_names'])
            directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'probe.npz',**fit)
            np.savez_compressed(directory/'predictions.npz',predictions=pred,targets=ys['test'],
                trajectory_index=groups['test'],step_index=data['test'][3])
            for k,r in enumerate(result):r.update(representation=label,labels=mode,alpha=float(fit['alpha'][k]))
            save(directory/'metrics.json',result);commit(directory,identity)
            print(f'[Representation probe] {label}/{mode}',flush=True)
    directory=output/'interventions'
    identity=dict(plan_sha256=digest(output/'plan.json'),task='layer_reset_action_sensitivity')
    if not verified(directory,identity):
        if directory.exists() and any(directory.iterdir()):raise ValueError('Archive incomplete interventions first')
        save(directory/'metrics.json',interventions(spec,data));commit(directory,identity)
    report(output,spec,data)


def report(output,spec,data):
    rows=[]
    labels=['raw_observation','phase_command_only']+[f'{n}_h{i}' for n in list(spec['models'])+
            [f'random_seed_{s}' for s in spec['control_seeds']] for i in (1,2)]
    for label in labels:
        for mode in ['true']+[f'shuffled_{s}' for s in spec['control_seeds']]:
            d=output/'probes'/label/mode
            if not verified(d,dict(plan_sha256=digest(output/'plan.json'),representation=label,labels=mode)):
                raise ValueError(f'Incomplete probe: {d}')
            rows+=read(d/'metrics.json')
    d=output/'interventions'
    if not verified(d,dict(plan_sha256=digest(output/'plan.json'),task='layer_reset_action_sensitivity')):
        raise ValueError('Incomplete interventions')
    persistence=scores(data['test'][1][:,-6:],data['test'][4],data['train'][1][:,-6:],data['test'][2],spec['target_names'][-6:])
    save(output/'report.json',dict(results=rows,persistence=persistence,interventions=read(d/'metrics.json')))
    with (output/'report.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# General NAO representation probes','',
        'Source-domain diagnostics only. Readability and action sensitivity do not establish R1 learning usefulness.',
        'All reset operations are on in-memory copies, not source checkpoints. No slip prediction target.',
        '', '| Representation | Property | R2 | RMSE | Alpha |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['labels']=='true':lines.append(f'| {r["representation"]} | {r["target"]} | {r["r2"]} | {r["rmse"]:.5g} | {r["alpha"]} |')
    lines+=['','See report.json for persistence baselines and per-run reset sensitivity; CSV/JSON include shuffled controls.',
            'Contact probes are thresholded ridge readouts, not calibrated probabilities. Inspect class balance.',
            'Method and transfer handoff: representation_probing/README.md.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output/"report.md"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','run','report'])
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--stride',type=int,default=7)
    args=p.parse_args();torch.set_num_threads(1)
    spec,data=spec_for(args.data.resolve(),args.stride)
    if args.stage=='audit':
        print(spec['split_counts']);print('Targets:',spec['target_names']);return
    output=args.output.resolve()
    if (output/'plan.json').exists():
        if read(output/'plan.json')!=spec:raise ValueError('Changed inputs; choose new --output')
    elif args.stage=='report':raise ValueError('No plan')
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('Nonempty unversioned output')
        save(output/'plan.json',spec)
    (run if args.stage=='run' else report)(output,spec,data)


if __name__=='__main__':main()

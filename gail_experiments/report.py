"""Confirmation comparisons against each actor's exact iteration-zero baseline."""
import csv
import numpy as np
from consolidated.run import Experiment
from summarize_evaluation import jerk_cost, sliced_wasserstein
from variable_training.common import read, save, digest


def report(study):
    rows=[];comparisons=[]
    selected=read(study.root/'selection.json')['selected']
    evaluation=study.root/'evaluation/confirmation'
    def metrics(controller):
        result=[]
        test=Experiment.offline_score(controller['model'],'gail','hidden',study.tests)['element_mse']
        for row in study.rows('confirmation'):
            directory=evaluation/row['id']/controller['label']
            r=read(directory/'result.json')
            oracle=read(evaluation/row['id']/'oracle/result.json')['return_target_progress_m']
            random=read(evaluation/row['id']/'random/result.json')['return_target_progress_m']
            with np.load(directory/'trace.npz',allow_pickle=False) as trace, np.load(evaluation/row['id']/'oracle/trace.npz',allow_pickle=False) as teacher:
                jerk,rate=jerk_cost(trace['joint_positions_deg'],trace['episode_id'],5)
                swd=sliced_wasserstein(trace['joint_positions_deg'],teacher['joint_positions_deg'],256,12345)
            r.update(held_out_action_mse=test,joint_swd=swd,jerk_rate=rate,
                     jerk_integrated=jerk,normalized_return=(r['return_target_progress_m']-random)/(oracle-random)
                     if abs(oracle-random)>1e-8 else None)
            result.append(r)
        return result
    def mean(rs,k):
        values=[r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(values)) if values else None
    for item in selected:
        recipe,name,seed=item['recipe'],item['initializer'],item['seed']
        cache={i:metrics(study.controller(recipe,name,seed,i)) for i in set([0,item['selected'],study.plan['iterations']])}
        baseline=cache[0]
        for role,iteration in [('initializer',0),('selected',item['selected']),('final',study.plan['iterations'])]:
            rs=cache[iteration]
            a=dict(recipe=recipe,initializer=name,seed=seed,role=role,iteration=iteration,
                   training_steps=iteration*study.plan['rollout_steps'],runs=len(rs),
                   progress_m=mean(rs,'return_target_progress_m'),normalized_return=mean(rs,'normalized_return'),
                   falls=sum(r['falls'] for r in rs),targets=sum(r['targets_reached'] for r in rs),
                   forward_velocity_mps=mean(rs,'mean_forward_velocity_mps'),direction_error=mean(rs,'mean_target_direction_error'),
                   test_action_mse=mean(rs,'held_out_action_mse'),on_policy_action_mse=mean(rs,'on_policy_latent_action_mse'),
                   joint_swd=mean(rs,'joint_swd'),jerk_rate=mean(rs,'jerk_rate'),
                   paired_progress_delta_m=float(np.mean([r['return_target_progress_m']-b['return_target_progress_m'] for r,b in zip(rs,baseline)])),
                   fall_delta=sum(r['falls']-b['falls'] for r,b in zip(rs,baseline)),
                   test_mse_delta=mean(rs,'held_out_action_mse')-mean(baseline,'held_out_action_mse'))
            comparisons.append(a)
        for rs in cache.values():rows.extend(rs)
    save(study.root/'comparison.json',dict(protocol=study.plan['protocol'],runs=rows,comparisons=comparisons,
         selection=selected,plan_sha256=digest(study.root/'plan.json'),
         caveats=['Selection uses validation only; confirmation seeds never choose a checkpoint.',
                  'Small gains are descriptive, not statistical proof. All pretrained original seeds share source seed42.',
                  'No task reward or auxiliary BC loss in adaptive recipe; discriminator expert actions are noise-convolved.',
                  'The network still uses the existing gait/IK interface.',
                  'Changes combine budget and algorithm choices; run legacy recipe at the same budget to isolate budget effects.']))
    with (study.root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    lines=['# Exploratory GAIL: does it improve its initializer?','',
           'Confirmation runs are separate from checkpoint-selection seeds. No automatic replacement of any original policy.',
           'All deltas compare the exact iteration-zero actor on the same confirmation rounds. No significance claim.', '',
           '| Recipe / init / seed | Role | Steps | Progress m | Δ m | Falls | Δ falls | Test MSE | On-policy MSE | Joint SWD | Jerk rate |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for a in comparisons:
        lines.append(f"| {a['recipe']} / {a['initializer']} / {a['seed']} | {a['role']} | {a['training_steps']} | "
                     f"{a['progress_m']:.4f} | {a['paired_progress_delta_m']:+.4f} | {a['falls']} | {a['fall_delta']:+d} | "
                     f"{a['test_action_mse']:.6f} | {a['on_policy_action_mse']:.6f} | {a['joint_swd']:.5f} | {a['jerk_rate']:.4g} |")
    lines+=['','## Across-seed descriptive comparison','',
            'Shared pretrained sources mean these are adaptation replicas, not independent pretraining experiments.',
            'Selected checkpoint results below are on confirmation seeds, not the validation scores used to select them.','',
            '| Recipe / init | Role | Training replicas | Mean progress Δm | Seed Δm range | Total fall Δ | Mean test-MSE Δ |',
            '|---|---|---|---|---|---|---|']
    for recipe in study.plan['recipes']:
        for name in study.plan['initializers']:
            for role in ['selected','final']:
                group=[r for r in comparisons if r['recipe']==recipe and r['initializer']==name and r['role']==role]
                deltas=[r['paired_progress_delta_m'] for r in group]
                lines.append(f"| {recipe} / {name} | {role} | {len(group)} | {np.mean(deltas):+.4f} | "
                             f"{min(deltas):+.4f} to {max(deltas):+.4f} | {sum(r['fall_delta'] for r in group):+d} | "
                             f"{np.mean([r['test_mse_delta'] for r in group]):+.6f} |")
    lines+=['','## Validation learning curves','',
            '| Recipe / init / seed | Steps | Progress m | Falls | Validation action MSE |',
            '|---|---|---|---|---|']
    for item in selected:
        for point in item['curve']:
            lines.append(f"| {item['recipe']} / {item['initializer']} / {item['seed']} | "
                         f"{point['iteration']*study.plan['rollout_steps']} | {point['progress']:.4f} | "
                         f"{point['falls']} | {point['validation_action_mse']:.6f} |")
    lines+=['','## Interpretation boundaries','',
            '- A progress improvement can accompany worse imitation. Inspect action MSE and occupancy as well as falls.',
            '- GAIL matches expert occupancy; it does not directly optimize task progress beyond the expert.',
            '- Failed random-start training is informative, not a reason to silently replace its initialization.',
            '- Selected and final results are both reported, including initializer retention when no checkpoint qualifies.',
            '- Per-update discriminator accuracy, reward variation, exact KL, rejection counts, learning rate and training MSE are saved under policies/.',
            '- Baseline reference progress is from newly collected oracle/random confirmation runs, not old cached task seeds.']
    (study.root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(study.root/'comparison.md',flush=True)

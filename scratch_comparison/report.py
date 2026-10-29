"""Combine cached parent scores with newly evaluated scratch policies."""
import copy
import csv
import numpy as np
from summarize_evaluation import confidence_interval, jerk_cost, sliced_wasserstein
from variable_training.common import read, save


def aggregate(runs):
    result=[]
    metrics=('return_target_progress_m','normalized_return','mean_forward_velocity_mps',
             'mean_target_direction_error','falls_per_active_minute','on_policy_latent_action_mse',
             'held_out_expert_action_mse','joint_occupancy_sliced_wasserstein',
             'jerk_cost_rate_rad2_per_s6','torso_roll_rms_deg','torso_pitch_rms_deg')
    keys=list(dict.fromkeys((r['row']['suite'],r['controller']['label']) for r in runs))
    for suite,label in keys:
        rows=[r for r in runs if r['row']['suite']==suite and r['controller']['label']==label]
        c=rows[0]['controller']
        a=dict(suite=suite,policy=label,initialization_cohort=c.get('initialization_cohort','parent'),
               training_seed=c['seed'],rounds=len(rows),total_falls=sum(r['falls'] for r in rows))
        for key in metrics:
            values=[r[key] for r in rows if r.get(key) is not None]
            a[key]=float(np.mean(values)) if values else None
            a[key+'_valid_n']=len(values)
        values=[r['normalized_return'] for r in rows if r['normalized_return'] is not None]
        a['normalized_return_ci95']=confidence_interval(values) if values else None
        result.append(a)
    return result


def report(exp,output,schedule,parent,parent_output):
    runs=copy.deepcopy(parent['runs']);offline={}
    parent_lookup={(r['row']['id'],r['controller']['label']):r for r in parent['runs']}
    for row in schedule['rows']:
        with np.load(parent_output/'runs'/row['id']/'oracle/trace.npz',allow_pickle=False) as trace:
            oracle_joints=trace['joint_positions_deg'].copy()
        oracle=parent_lookup[(row['id'],'oracle')]['return_target_progress_m']
        random=parent_lookup[(row['id'],'random')]['return_target_progress_m']
        for c in schedule['controllers']:
            if c.get('initialization_cohort')!='scratch':continue
            if row['slip'] is None and c['domain']=='varied':continue
            directory=output/'runs'/row['id']/c['label']
            result=read(directory/'result.json')
            with np.load(directory/'trace.npz',allow_pickle=False) as trace:
                jerk,rate=jerk_cost(trace['joint_positions_deg'],trace['episode_id'],5)
                swd=sliced_wasserstein(trace['joint_positions_deg'],oracle_joints,256,12345)
                roll=float(np.sqrt(np.mean(trace['torso_roll_deg']**2)))
                pitch=float(np.sqrt(np.mean(trace['torso_pitch_deg']**2)))
            denom=oracle-random
            domain='original' if row['slip'] is None else 'varied'
            key=(c['label'],domain)
            if key not in offline:
                offline[key]=exp.offline_score(c['model'],c['kind'],c['architecture'],exp.data(domain,'test'))
            result.update(normalized_return=(result['return_target_progress_m']-random)/denom if abs(denom)>1e-8 else None,
                          normalization_valid=abs(denom)>1e-8,
                          oracle_return_ratio=result['return_target_progress_m']/oracle if abs(oracle)>1e-8 else None,
                          joint_occupancy_sliced_wasserstein=swd,jerk_integrated_rad2_per_s5=jerk,
                          jerk_cost_rate_rad2_per_s6=rate,torso_roll_rms_deg=roll,torso_pitch_rms_deg=pitch,
                          held_out_expert_action_mse=offline[key]['element_mse'],held_out_test_domain=domain,
                          held_out_context='correct' if c['architecture']=='input' else 'hidden')
            runs.append(result)
    lookup={(r['row']['id'],r['controller']['label']):r for r in runs}
    paired=[]
    for r in runs:
        c=r['controller']
        if c.get('initialization_cohort')!='scratch' or c['algorithm']=='gail_random':continue
        parent_label=f'varied_{c["architecture"]}_{c["seed"]}_{c["algorithm"]}'+('_correct' if c['architecture']=='input' else '')
        reference=lookup[(r['row']['id'],parent_label)]
        paired.append(dict(round=r['row']['id'],policy=c['label'],reference=parent_label,
                           comparison='scratch_minus_continuation',
                           progress_delta_m=r['return_target_progress_m']-reference['return_target_progress_m'],
                           fall_delta=r['falls']-reference['falls']))
        if c['architecture']=='input':
            hidden_label=f'scratch_varied_hidden_{c["seed"]}_{c["algorithm"]}'
            hidden=lookup[(r['row']['id'],hidden_label)]
            paired.append(dict(round=r['row']['id'],policy=c['label'],reference=hidden_label,
                               comparison='scratch_input_minus_hidden',
                               progress_delta_m=r['return_target_progress_m']-hidden['return_target_progress_m'],
                               fall_delta=r['falls']-hidden['falls']))
    aggregates=aggregate(runs)
    save(output/'comparison.json',dict(protocol=exp.plan['protocol'],runs=runs,aggregates=aggregates,
         paired_differences=paired,reused_parent=str(parent_output),
         parent_comparison_sha256=schedule['parent_comparison_sha256'],
         caveats=parent.get('caveats',[])+[
             'Scratch and continuation match incremental varied-domain budgets, not inherited total training cost.',
             'Direct random GAIL retains the same supervised anchor and stabilization settings.',
             'Cached parent records are reused, not newly independent baseline trials.']))
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    lines=['# Scratch versus continuation: appended evaluation','',
           f'{len(parent["runs"])} parent deployments reused; {len(runs)-len(parent["runs"])} new deployments.',
           'Parent models/results unchanged. Per-replica scores; matched incremental, not lifetime training budgets.',
           '', '| Suite | Policy | Runs | Progress m | Normalized score | Falls |','|---|---|---:|---:|---:|---:|']
    for a in aggregates:
        score='undefined' if a['normalized_return'] is None else f'{a["normalized_return"]:.4f}'
        lines.append(f'| {a["suite"]} | {a["policy"]} | {a["rounds"]} | {a["return_target_progress_m"]:.3f} | {score} | {a["total_falls"]} |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f'[Report] {output / "comparison.md"}',flush=True)

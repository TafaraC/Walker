"""Legacy mechanical metrics with paired same-environment baselines and context deltas."""
import csv
import io
import numpy as np
from variable_training.common import read, save
from summarize_evaluation import confidence_interval, jerk_cost, sliced_wasserstein


def report(experiment, output, schedule):
    runs=[]
    offline={}
    for row in schedule['rows']:
        base=output/'runs'/row['id']
        oracle=read(base/'oracle'/'result.json')['return_target_progress_m']
        random=read(base/'random'/'result.json')['return_target_progress_m']
        with np.load(base/'oracle'/'trace.npz',allow_pickle=False) as trace:
            oracle_joints=trace['joint_positions_deg'].copy()
        for controller in schedule['controllers']:
            if row['slip'] is None and controller['domain']=='varied': continue
            directory=base/controller['label']
            result=read(directory/'result.json')
            with np.load(directory/'trace.npz',allow_pickle=False) as trace:
                jerk,rate=jerk_cost(trace['joint_positions_deg'],trace['episode_id'],5)
                distance=sliced_wasserstein(trace['joint_positions_deg'],oracle_joints,256,12345)
                torso_roll=float(np.sqrt(np.mean(trace['torso_roll_deg']**2)))
                torso_pitch=float(np.sqrt(np.mean(trace['torso_pitch_deg']**2)))
            denominator=oracle-random
            result.update(normalized_return=(result['return_target_progress_m']-random)/denominator
                          if abs(denominator)>1e-8 else None,
                          normalization_valid=abs(denominator)>1e-8,
                          oracle_return_ratio=result['return_target_progress_m']/oracle if abs(oracle)>1e-8 else None,
                          joint_occupancy_sliced_wasserstein=distance,
                          jerk_integrated_rad2_per_s5=jerk,jerk_cost_rate_rad2_per_s6=rate,
                          torso_roll_rms_deg=torso_roll,torso_pitch_rms_deg=torso_pitch)
            if controller['model']:
                test_domain='original' if row['slip'] is None else 'varied'
                key=(controller['label'],test_domain)
                if key not in offline:
                    paths=[str(p) for p in sorted((experiment.root/'data'/test_domain/'test').glob('*/data.npz'))]
                    offline[key]=experiment.offline_score(controller['model'],controller['kind'],
                                                         controller['architecture'],paths)
                result['held_out_expert_action_mse']=offline[key]['element_mse']
                result['held_out_test_domain']=test_domain
                result['held_out_context']='correct' if controller['architecture']=='input' else 'hidden'
            runs.append(result)
    metrics=('return_target_progress_m','normalized_return','mean_forward_velocity_mps',
             'mean_target_direction_error','falls_per_active_minute','on_policy_latent_action_mse',
             'joint_occupancy_sliced_wasserstein','jerk_cost_rate_rad2_per_s6')
    aggregates=[]
    for suite in dict.fromkeys(r['row']['suite'] for r in runs):
        for controller in schedule['controllers']:
            rows=[r for r in runs if r['row']['suite']==suite and r['controller']==controller]
            if not rows: continue
            item=dict(suite=suite,policy=controller['label'],rounds=len(rows),
                      total_falls=sum(r['falls'] for r in rows),training_seed=controller['seed'])
            for key in metrics:
                values=[r[key] for r in rows if r[key] is not None]
                item[key]=float(np.mean(values)) if values else None
                item[key+'_valid_n']=len(values)
            # Keep CIs per replica; do not pool fine-tuning seeds as independent deployment samples.
            vals=[r['normalized_return'] for r in rows if r['normalized_return'] is not None]
            item['normalized_return_ci95']=confidence_interval(vals) if vals else None
            nominal=[r['return_target_progress_m'] for r in runs
                     if r['row']['suite']=='nominal' and r['controller']==controller]
            item['retention_vs_corrected_nominal']=(item['return_target_progress_m']/np.mean(nominal)
                if suite!='original' and nominal and abs(np.mean(nominal))>1e-8 else None)
            aggregates.append(item)
    paired=[]
    lookup={(r['row']['id'],r['controller']['label']):r for r in runs}
    for result in runs:
        c=result['controller']
        if c['architecture']!='input': continue
        reference=(f'varied_hidden_{c["seed"]}_{c["algorithm"]}' if c['mode']=='correct' else
                   f'varied_input_{c["seed"]}_{c["algorithm"]}_correct')
        control=lookup[(result['row']['id'],reference)]
        paired.append(dict(round=result['row']['id'],policy=c['label'],reference=reference,
                           progress_delta_m=result['return_target_progress_m']-control['return_target_progress_m'],
                           fall_delta=result['falls']-control['falls']))
    save(output/'comparison.json',dict(protocol=experiment.plan['protocol'],runs=runs,
        aggregates=aggregates,paired_differences=paired,
        caveats=['Confidence intervals are descriptive deployment variation, not independent policy-training evidence.',
                 'Original scene and corrected nominal .1 are distinct conditions.',
                 'Held-out MSE uses the shared test pool of the evaluated domain; on-policy MSE uses visited states.',
                 'Undefined normalization denominators are null; scores are not clipped.',
                 'Jerk is angular kinematic jerk; occupancy is sliced Wasserstein, not exact 22-D transport.']))
    buffer=io.StringIO()
    writer=csv.DictWriter(buffer,fieldnames=list(aggregates[0]))
    writer.writeheader();writer.writerows(aggregates)
    (output/'comparison.csv').write_text(buffer.getvalue())
    lines=['# Consolidated evaluation','',
           'Paired random/oracle normalization within each round. Each row retains its training replica.',
           'Original installed scene is separate from corrected scalar nominal. All runs use owned servers.', '',
           '| Suite | Policy | Runs | Progress m | Normalized return | Falls |',
           '|---|---|---:|---:|---:|---:|']
    for a in aggregates:
        score='N/A' if a['normalized_return'] is None else f'{a["normalized_return"]:.4f}'
        lines.append(f'| {a["suite"]} | {a["policy"]} | {a["rounds"]} | '
                     f'{a["return_target_progress_m"]:.3f} | {score} | {a["total_falls"]} |')
    lines+=['','Full mechanical metrics, valid normalization counts, per-replica CIs, held-out MSE and paired context deltas are in comparison.json.',
            'Resource accounting and initializer ancestry are in training_summary.json and each policy final.json.',
            'These are newly collected results; historical evaluations and FiLM are not pooled.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')

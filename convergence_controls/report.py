"""Offline fidelity/resource curves and paired mechanical reports; no policy selection."""
import csv
from pathlib import Path

import numpy as np

from consolidated.run import Experiment
from summarize_evaluation import jerk_cost, sliced_wasserstein
from variable_training.common import read, save, digest, verified, begin, commit


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)



def report(study, output, schedule):
    """Reuse audited old deployments; never run the simulator from a reporting command."""
    parent_eval = study.parent.root/'evaluation/all_all_correct'
    dependencies = [output/f"schedule_{schedule['scope']}.json"]
    raw = []
    for row in schedule['rows']:
        for c in schedule['controllers']:
            if row['slip'] is None and c['domain']=='varied':
                continue
            directory = output/'runs'/row['id']/c['label']
            raw.append((directory, read(directory/'result.json'), False))
    # Audited references; independent engine stage shares new runs with full evaluation.
    if not study.plan['quick']:
        for row in schedule['rows']:
            for label in ('oracle','random','original_hidden_42_bc','original_hidden_42_dagger_mixed'):
                directory = parent_eval/'runs'/row['id']/label
                study.audit_source(directory)
                raw.append((directory,read(directory/'result.json'),True))
            for label in ('zero_native','bc_original_matched'):
                directory = study.controls/'evaluation/runs'/row['id']/label
                study.audit_source(directory)
                raw.append((directory,read(directory/'result.json'),True))
    if not study.plan['quick']:
        seen={(r['row']['id'],r['controller']['label']) for _,r,_ in raw}
        for row in schedule['rows']:
            for c in schedule['controllers']:
                if 'reference_label' not in c or (row['slip'] is None and c['domain']=='varied'):
                    continue
                key=(row['id'],c['reference_label'])
                if key in seen:continue
                directory=Path(c['reference_evaluation_root'])/'runs'/row['id']/c['reference_label']
                study.audit_source(directory)
                raw.append((directory,read(directory/'result.json'),True));seen.add(key)
    lookup = {(r['row']['id'],r['controller']['label']):(d,r) for d,r,_ in raw}
    if len(lookup) != len(raw):
        raise ValueError('Duplicate evaluation records')
    for directory, result, _ in raw:
        if result['active_steps'] != schedule['steps']:
            raise ValueError('Cannot compare deployments with different active horizons')
        dependencies += [directory/'completed.json', directory/'trace.npz', directory/'result.json']
        if result['controller']['model']:
            dependencies.append(Path(result['controller']['model']))
    test_paths = {}
    for domain in ('original', 'varied'):
        test_paths[domain] = study.parent.data(domain, 'test')
        dependencies += [Path(p) for p in test_paths[domain]]
    for p in (sorted((study.root/'training').glob('*/metrics.json')) if schedule['scope']=='all' else []):
        if read(p.parent/'job.json')['task']=='control_train':
            dependencies += [p,p.parent/'history.json',p.parent/'completed.json']
    identity = dict(runtime=study.identity, inputs={str(p):digest(p) for p in dependencies})
    summary = output/f"summary_{schedule['scope']}"
    if verified(summary, identity):
        print(f'[Resume] {summary}')
        return
    begin(summary)
    scores, runs, oracle_traces = {}, [], {}
    for directory, result, reused in raw:
        row, c = result['row'], result['controller']
        key = row['id']
        oracle_dir, oracle_result = lookup[(key,'oracle')]
        random_result = lookup[(key,'random')][1]
        if key not in oracle_traces:
            with np.load(oracle_dir/'trace.npz', allow_pickle=False) as trace:
                oracle_traces[key] = trace['joint_positions_deg'].copy()
        with np.load(directory/'trace.npz', allow_pickle=False) as trace:
            jerk, rate = jerk_cost(trace['joint_positions_deg'], trace['episode_id'], 5)
            swd = sliced_wasserstein(trace['joint_positions_deg'], oracle_traces[key], 256, 12345)
            roll = float(np.sqrt(np.mean(trace['torso_roll_deg']**2)))
            pitch = float(np.sqrt(np.mean(trace['torso_pitch_deg']**2)))
        denominator = oracle_result['return_target_progress_m']-random_result['return_target_progress_m']
        item = dict(result, reused=reused,
                    normalized_return=(result['return_target_progress_m']-random_result['return_target_progress_m'])/denominator
                    if abs(denominator)>1e-8 else None,
                    joint_occupancy_sliced_wasserstein=swd, jerk_integrated_rad2_per_s5=jerk,
                    jerk_cost_rate_rad2_per_s6=rate, torso_roll_rms_deg=roll, torso_pitch_rms_deg=pitch)
        if c['model']:
            domain = 'original' if row['slip'] is None else 'varied'
            score_key = (c['model'], domain, c['kind'], c['architecture'])
            if score_key not in scores:
                scores[score_key] = Experiment.offline_score(c['model'], c['kind'], c['architecture'], test_paths[domain])
            item['held_out_expert_action_mse'] = scores[score_key]['element_mse']
        runs.append(item)
    aggregates = []
    metrics = ['return_target_progress_m', 'normalized_return', 'held_out_expert_action_mse',
               'on_policy_latent_action_mse', 'joint_occupancy_sliced_wasserstein',
               'mean_forward_velocity_mps', 'mean_target_direction_error', 'falls_per_active_minute',
               'jerk_cost_rate_rad2_per_s6', 'torso_roll_rms_deg', 'torso_pitch_rms_deg']
    # Pool grouped/confirmation only, never mix scalar nominal into sampled means.
    for suite in ('original', 'sampled', 'nominal'):
        for label in sorted({r['controller']['label'] for r in runs}):
            selected = [r for r in runs if r['controller']['label']==label and
                        (r['row']['suite'] in ('grouped','confirmation') if suite=='sampled' else r['row']['suite']==suite)]
            if not selected:
                continue
            entry = dict(suite=suite, policy=label, runs=len(selected), reused=selected[0]['reused'],
                         total_falls=sum(r['falls'] for r in selected),
                         active_minutes=sum(r['active_seconds'] for r in selected)/60)
            for metric in metrics:
                values = [r[metric] for r in selected if r.get(metric) is not None]
                entry[metric] = float(np.mean(values)) if values else None
            aggregates.append(entry)
    training = []
    for p in (sorted((study.root/'training').glob('*/metrics.json')) if schedule['scope']=='all' else []):
        job = read(p.parent/'job.json')
        if job['task'] != 'control_train':
            continue
        study.job(p.parent, job, job['train']+job['validation']+[job['model']]+
                  ([job['optimizer']] if job['optimizer'] else [])+[job['source_receipt']], check_only=True)
        training.append(dict(label=p.parent.name, **read(p)))
    paired = []
    scored = {(r['row']['id'],r['controller']['label']):r for r in runs}
    for result in runs:
        if result['reused'] or result['controller'].get('algorithm') is None:
            continue
        c, round_id = result['controller'], result['row']['id']
        if c['algorithm']=='untrained_random_network':
            references = ['oracle','zero_native','random']
        else:
            references = [c.get('reference_label'),'oracle']
            references += [other['label'] for other in schedule['controllers']
                if other.get('cohort')==c.get('cohort') and other['domain']==c['domain']
                and other['architecture']==c['architecture'] and other['seed']==c['seed']
                and other.get('algorithm') in ('bc','dagger_mixed')]
        references = list(dict.fromkeys(x for x in references if x and x!=c['label']))
        for reference in references:
            old = scored.get((round_id,reference))
            if old is None:  # Smoke has fresh baselines, not historical long-horizon policies.
                continue
            item = dict(round=round_id, suite=result['row']['suite'], policy=c['label'], reference=reference)
            for metric in metrics:
                if result.get(metric) is not None and old.get(metric) is not None:
                    item[metric+'_delta'] = result[metric]-old[metric]
            paired.append(item)
    save(summary/'comparison.json', dict(runs=runs, aggregates=aggregates, training=training, paired_differences=paired,
         caveats=['Original and scalar nominal are distinct; sampled pools grouped+confirmation only.',
                  'Reused runs are not new measurements; each random network retains its initialization seed.',
                  'Original fits start at equal performed updates; varied histories retain different inherited costs. Additional optimization can differ.',
                  'Validation stopping concerns fixed datasets, not proven optimization or online DAgger convergence.',
                  'Random networks use real observations and native outputs; weights stay fixed with zero training.',
                  'Direction error retains the historical 20 ms / 40 ms localization caveat; SWD is occupancy and jerk is kinematic.']))
    write_csv(summary/'comparison.csv', aggregates)
    write_csv(summary/'paired_differences.csv', paired)
    lines = ['# Convergence and untrained-network engine controls', '',
             'Historical results are read-only references. Stopping decisions use validation only.', '',
             '| Suite | Policy | Runs | Progress m | Action MSE | Joint SWD | Falls / active min |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        mse = '—' if r['held_out_expert_action_mse'] is None else f'{r["held_out_expert_action_mse"]:.5f}'
        lines.append(f'| {r["suite"]} | {r["policy"]} | {r["runs"]} | {r["return_target_progress_m"]:.3f} | '
                     f'{mse} | {r["joint_occupancy_sliced_wasserstein"]:.5f} | {r["total_falls"]} / {r["active_minutes"]:.1f} |')
    lines += ['', '## Validation stopping and resource accounting', '',
              '| Diagnostic | Initial validation MSE | Best validation MSE | Selected additional epoch | Additional epochs run | Stop reason | Total lineage updates |',
              '|---|---:|---:|---:|---:|---|---:|']
    for t in training:
        lines.append(f'| {t["label"]} | {t["initial_validation_mse"]:.7f} | {t["validation_mse"]:.7f} | '
                     f'{t["best_epoch"]} | {t["epochs_run"]} | {t["stop_reason"]} | {t["total_lineage_gradient_updates"]} |')
    lines += ['', 'A budget-exhausted diagnostic has not met its validation stopping rule.',
              'Passing patience is an operational plateau criterion, not proof of global convergence.',
              'Useful progress, falls and occupancy must be interpreted together for engine controls.',
              'Legacy tracking values include stale-position frames; no historical metrics were silently corrected.']
    (summary/'comparison.md').write_text('\n'.join(lines)+'\n')
    if training:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups={}
        for t in training:
            job=read(study.root/'training'/t['label']/'job.json')
            key=(job['domain'],job.get('cohort','original'),job['architecture'])
            groups.setdefault(key,[]).append((t,job))
        cols=min(2,len(groups));rows=(len(groups)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(7*cols,4*rows),squeeze=False)
        for ax,(key,items) in zip(axes.flat,groups.items()):
            for t,job in items:
                h=read(study.root/'training'/t['label']/'history.json')
                ax.plot([r['gradient_updates'] for r in h],[r['validation_mse'] for r in h],
                        label=f"{job['algorithm']} seed {job['seed']}")
            ax.set(xlabel='Additional optimizer updates',ylabel='Validation latent-action MSE',yscale='log',
                   title=' / '.join(key))
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
        fig.suptitle('Fixed-dataset validation trajectories; inherited selected checkpoint at zero')
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(summary/'validation_curves.png',dpi=170);fig.savefig(summary/'validation_curves.pdf');plt.close(fig)
    commit(summary, identity)
    print(f'[Report] {summary/"comparison.md"}')

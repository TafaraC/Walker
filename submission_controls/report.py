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


def curves(study):
    """Original-scene endpoints only. No claim that these are return convergence curves."""
    parent = study.parent.root
    validation = study.parent.data('original', 'validation')
    tests = study.parent.data('original', 'test')
    rows, dependencies = [], validation+tests
    for algorithm in ('bc', 'dagger_mixed', 'gail_bc', 'gail_dagger_mixed'):
        base = parent/f'policies/original/hidden/seed_42/{algorithm}'
        previous_updates, previous_exposures = 0, 0
        iterations = [0] if algorithm=='bc' else range(8)
        for iteration in iterations:
            if algorithm.startswith('gail') and iteration==0:
                directory = base/'initialization'
                updates = exposures = selected = 0
            else:
                directory = base/f'iteration_{iteration:03d}/update'
                metrics = read(directory/'metrics.json')
                updates = metrics.get('gradient_updates', 0)
                exposures = metrics.get('sample_exposures', 0)
                selected = metrics.get('best_epoch')
            study.audit_source(directory)
            model = directory/'policy.pth'
            kind = 'gail' if algorithm.startswith('gail') else 'bc'
            current = 0 if algorithm=='bc' else iteration*10000
            inherited = (70000 if algorithm=='gail_dagger_mixed' else 0)
            inherited_updates = (211050 if algorithm=='gail_dagger_mixed' else
                                 66500 if algorithm=='gail_bc' else 0)
            row = dict(algorithm=algorithm, iteration=iteration, model=str(model), sha256=digest(model),
                       learner_environment_steps=current, inherited_learner_steps=inherited,
                       total_lineage_learner_steps=current+inherited,
                       active_dagger_queries=current if algorithm=='dagger_mixed' else 0,
                       oracle_labels_including_seed=(100000+current if algorithm=='dagger_mixed' else 170000),
                       inherited_dagger_queries=inherited,
                       unique_lineage_oracle_labels=240000 if algorithm=='gail_dagger_mixed' else
                           100000+current if algorithm=='dagger_mixed' else 170000,
                       cumulative_supervised_updates=previous_updates+updates+inherited_updates,
                       cumulative_supervised_exposures=previous_exposures+exposures+
                           (54000000 if algorithm=='gail_dagger_mixed' else 17000000 if algorithm=='gail_bc' else 0),
                       selected_epoch_in_round=selected,
                       resource_note='Allocated completed-round budgets; selected checkpoint may precede round end. '
                                     'GAIL supervised columns count initializer only, not adversarial optimizer work.')
            dependencies.extend([str(model), str(directory/'completed.json')])
            rows.append(row)
            previous_updates += updates
            previous_exposures += exposures
    identity = dict(runtime=study.identity, inputs={str(p):digest(p) for p in dependencies},
                    scope='original shared held-out data, original BC/DAgger/GAIL saved rounds')
    output = study.root/'curves'
    if verified(output, identity):
        print(f'[Resume] {output}')
        return
    begin(output)
    for row in rows:
        kind = 'gail' if row['algorithm'].startswith('gail') else 'bc'
        row['validation_mse'] = Experiment.offline_score(row['model'], kind, 'hidden', validation)['element_mse']
        row['test_mse'] = Experiment.offline_score(row['model'], kind, 'hidden', tests)['element_mse']
    save(output/'curves.json', dict(rows=rows, caveat='Retrospective fidelity curves, not normalized-return convergence. '
                                  'Test scores never choose a checkpoint; costs include explicit lineage columns.'))
    write_csv(output/'curves.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    dagger = [r for r in rows if r['algorithm']=='dagger_mixed']
    axes[0].plot([r['oracle_labels_including_seed'] for r in dagger],
                 [r['test_mse'] for r in dagger], 'o-', label='Mixed DAgger')
    bc = next(r for r in rows if r['algorithm']=='bc')
    axes[0].scatter([170000], [bc['test_mse']], label='Original BC (100 epochs)', marker='s')
    axes[0].set_xlabel('Oracle labels including expert seed')
    for algorithm in ('gail_bc', 'gail_dagger_mixed'):
        subset = [r for r in rows if r['algorithm']==algorithm]
        axes[1].plot([r['total_lineage_learner_steps'] for r in subset],
                     [r['test_mse'] for r in subset], 'o-', label=algorithm)
    axes[1].set_xlabel('Learner simulation steps including initializer')
    for axis in axes:
        axis.set_ylabel('Held-out latent-action MSE')
        axis.set_yscale('log')
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.suptitle('Original saved checkpoints: finite-budget fidelity, not return convergence')
    fig.tight_layout()
    fig.savefig(output/'fidelity_resources.png', dpi=160)
    plt.close(fig)
    commit(output, identity)
    print(f'[Curves] {output}')


def report(study, output, schedule):
    """Reuse audited old deployments; never run the simulator from a reporting command."""
    parent_eval = study.parent.root/'evaluation/all_all_correct'
    dependencies = [output/'schedule.json']
    raw = []
    for row in schedule['rows']:
        for c in schedule['controllers']:
            if row['slip'] is None and c['domain']=='varied':
                continue
            directory = output/'runs'/row['id']/c['label']
            raw.append((directory, read(directory/'result.json'), False))
    # Historical reference controllers are selected by exact model family and seed.
    if not study.plan['quick']:
        for row in schedule['rows']:
            labels = ['oracle', 'random', 'original_hidden_42_bc', 'original_hidden_42_dagger_mixed']
            for label in labels:
                directory = parent_eval/'runs'/row['id']/label
                study.audit_source(directory)
                raw.append((directory, read(directory/'result.json'), True))
            if row['slip'] is not None:
                for c in schedule['controllers']:
                    if c['domain'] != 'varied':
                        continue
                    suffix = '_correct' if c['architecture']=='input' else ''
                    for scratch in (False, True):
                        label = f'{"scratch_" if scratch else ""}varied_{c["architecture"]}_{c["seed"]}_bc{suffix}'
                        root = study.parent.root/('scratch_v1/evaluation/all_all_correct' if scratch else
                                                 'evaluation/all_all_correct')
                        directory = root/'runs'/row['id']/label
                        study.audit_source(directory)
                        raw.append((directory, read(directory/'result.json'), True))
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
    identity = dict(runtime=study.identity, inputs={str(p):digest(p) for p in dependencies})
    summary = output/'summary'
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
    for p in sorted((study.root/'training').glob('*/metrics.json')):
        job = read(p.parent/'job.json')
        study.job(p.parent, job, job['train']+job['validation']+[job['model']]+
                  ([job['optimizer']] if job['optimizer'] else []), check_only=True)
        training.append(dict(label=p.parent.name, **read(p)))
    paired = []
    scored = {(r['row']['id'],r['controller']['label']):r for r in runs}
    for result in runs:
        if result['reused']:
            continue
        c, round_id = result['controller'], result['row']['id']
        if c['label']=='zero_native':
            references = ['oracle']
        elif c['domain']=='varied':
            suffix = '_correct' if c['architecture']=='input' else ''
            reference = f'varied_{c["architecture"]}_{c["seed"]}_bc{suffix}'
            references = [reference, 'scratch_'+reference]
        else:
            references = ['original_hidden_42_bc', 'original_hidden_42_dagger_mixed']
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
                  'Reused runs are not new measurements; replicas retain individual identities.',
                  'Matched optimizer updates are not equal collection/query cost.',
                  'Early-stop diagnostics concern validation stopping, not proven algorithm convergence.',
                  'Zero native actions retain gait/IK and resets; SWD is occupancy, jerk is kinematic.']))
    write_csv(summary/'comparison.csv', aggregates)
    write_csv(summary/'paired_differences.csv', paired)
    lines = ['# Submission controls', '',
             'Finite-budget controls. Existing results are verified references, not reruns.', '',
             '| Suite | Policy | Runs | Progress m | Action MSE | Joint SWD | Falls / active min |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        mse = '—' if r['held_out_expert_action_mse'] is None else f'{r["held_out_expert_action_mse"]:.5f}'
        lines.append(f'| {r["suite"]} | {r["policy"]} | {r["runs"]} | {r["return_target_progress_m"]:.3f} | '
                     f'{mse} | {r["joint_occupancy_sliced_wasserstein"]:.5f} | {r["total_falls"]} / {r["active_minutes"]:.1f} |')
    lines += ['', 'Full mechanics, per-run normalization, selected/budget training costs are in comparison.json.',
              'No checkpoint is chosen on test scores. Do not interpret a small mean difference as established superiority.']
    (summary/'comparison.md').write_text('\n'.join(lines)+'\n')
    commit(summary, identity)
    print(f'[Report] {summary/"comparison.md"}')

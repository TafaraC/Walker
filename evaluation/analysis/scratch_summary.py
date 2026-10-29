"""Read-only extension of the consolidated narrative from saved scratch results."""
import hashlib
import json
import numpy as np


def build(root):
    base = root / 'experiments/consolidated/v1/scratch_v1'
    paths = [base/'evaluation/all_all_correct/comparison.json',
             base/'probing/report.json', base/'training_summary.json']
    e, p, t = [json.loads(path.read_text()) for path in paths]
    assert len(e['runs']) == 1949 and len(p['results']) == 15720
    assert len(t['completed_policies']) == 28
    assert all(r['active_steps'] == 3000 for r in e['runs'])
    out = []
    def say(s): out.append(s.strip()+'\n')
    def table(head, rows):
        say('| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+
            '\n'.join('| '+' | '.join(map(str, r))+' |' for r in rows))
    def mean(rs, k): return float(np.mean([r[k] for r in rs if r.get(k) is not None]))
    def f(v): return f'{v:.4f}'
    algorithms = ['bc', 'dagger_mixed', 'gail_bc', 'gail_dagger_mixed']
    names = dict(bc='BC', dagger_mixed='DAgger-M', gail_bc='GAIL-BC', gail_dagger_mixed='GAIL-DM', gail_random='GAIL-random')
    def runs(a, arch='hidden', cohort='scratch', suites=('grouped','confirmation'), domain='varied'):
        return [r for r in e['runs'] if r['row']['suite'] in suites
                and r['controller']['algorithm'] == a and r['controller']['architecture'] == arch
                and r['controller']['domain'] == domain
                and r['controller'].get('initialization_cohort','parent') == cohort]
    def label(a, arch): return names[a]+' / '+arch
    def reps(a, arch, cohort, layer=1):
        return [('scratch_' if cohort=='scratch' else '')+f'varied_{arch}_{s}_{a}'+
                ('_correct' if arch=='input' else '')+f'_h{layer}' for s in [42,43,44]]
    def probe(ns, target):
        return mean([r for r in p['results'] if r['representation'] in ns and r['labels']=='true' and r['target']==target], 'r2')
    say('''## Appendix F. Completed scratch-initialization comparison — 16 October 2026

This extension answers whether original-environment pretraining was necessary for
the varied-domain results, and whether GAIL could learn useful walking directly
from random actor weights under the same configured rollout budget. It adds a
fourth behavioural comparison after original, frozen-on-varied, and continued
varied training. Sections 1–10 remain the parent-batch analysis, not pooled claims
about every new model. The revised conclusions below incorporate the new evidence.

### F.1 Completion, experimental design and accounting

All **28 additional checkpoints, 929 new deployments and 224 new multitarget
probe fits** are present. Combined with reused parent evidence, this gives
**56 trained checkpoints, 1,949 deployments, 524 fits, 131 representations and
15,720 target-level probe scores**. The 1,020 parent deployments were reused
exactly, not run again. Saved artifact hashes were checked for all 929 new
evaluation receipts, 224 new and 300 parent probe receipts; all 28 new checkpoint
hashes matched. Combined probe scores matched the individual metrics files.
These are saved-artifact integrity checks, not a new live physics audit.

The new supervised families start randomly directly in the varied environment:
BC and mixed DAgger, each hidden-context and input-conditioned, seeds 42/43/44.
GAIL-BC and GAIL-DM then start from those newly trained supervised policies.
Thus “scratch GAIL-BC” removes original-environment pretraining from its lineage;
it does **not** mean its actor starts randomly at the PPO stage. Separately,
direct GAIL-random has one original-environment seed and three varied-hidden
seeds. There is no input-conditioned direct-random GAIL condition.

BC still uses 170,000 examples; mixed DAgger uses 100,000 seed examples plus
70,000 learner-state oracle labels; every GAIL actor gets 70,000 PPO rollout
steps with the same static expert-pool contract and stabilization settings.
Shared expert data are reused. Incremental varied-domain budgets match the
continuation conditions, but lifetime training does not: continuation inherits
original-environment training. This is an initialization-history comparison,
not an isolation of pretraining from additional lifetime optimization.

The new collection summary records 1,540,000 active steps and 420,000 oracle
queries, with 1,541,188 rollout-plus-recovery steps and 7,700 startup received
steps reported separately. Reused demonstrations and inherited initializer costs
must not be counted as newly collected data or described as absent supervision.

The same 30 sampled rounds (15 grouped and 15 confirmation) and three scalar-0.1
rounds are reused. Each slip is fixed throughout a 3,000-active-step deployment.
Each varied family below pools three seeds × 30 sampled rounds = 90 active
minutes. Original GAIL-random additionally has five original-scene runs, giving
38 deployments; the other 27 new models have 33 each: 38 + 27 × 33 = 929.
Probe states, trajectory splits, penalties, targets and controls are unchanged.
''')
    say('''### F.2 Scratch versus continuation: locomotion and imitation

Each row contrasts matching algorithm/architecture families over identical
sampled rounds. “Cont.” denotes original-pretrained continuation; “scratch”
denotes the new lineage without that original pretraining. Progress and scores
are deployment means. MSE is elementwise latent-action error; lower is better.
The normalized score is averaged per round, not computed from pooled returns.
''')
    rows=[]
    for a in algorithms:
        for arch in ['hidden','input']:
            c,s=runs(a,arch,'parent'),runs(a,arch)
            rows.append([label(a,arch),f(mean(c,'return_target_progress_m')),f(mean(s,'return_target_progress_m')),
                         f(mean(s,'return_target_progress_m')-mean(c,'return_target_progress_m')),
                         f(mean(c,'normalized_return')),f(mean(s,'normalized_return')),
                         f(mean(c,'held_out_expert_action_mse')),f(mean(s,'held_out_expert_action_mse')),
                         f(mean(c,'on_policy_latent_action_mse')),f(mean(s,'on_policy_latent_action_mse'))])
    table(['Family','Cont. m','Scratch m','Δ m','Cont. score','Scratch score','Cont. test MSE','Scratch test MSE','Cont. on-policy MSE','Scratch on-policy MSE'],rows)
    say('''The clearest new finding is a **progress–fidelity trade-off**, not a universal
benefit of pretraining. Scratch BC and GAIL-BC gain about 0.39–0.41 m per minute
against their continuation counterparts, but their held-out action errors are
roughly 5.6 times larger. Scratch hidden BC reaches 33.192 m versus 32.784 m,
while its test MSE rises from 0.01995 to 0.11177. More progress therefore does
not mean closer reproduction of the oracle's actions.

Mixed DAgger retains a substantial fidelity advantage even without original
pretraining: scratch hidden DAgger test MSE is 0.01035 versus scratch BC 0.11177,
approximately a 10.8-fold reduction. Yet continuation DAgger is still much more
precise (0.00187). Its roughly 5.5-fold advantage over scratch DAgger is evidence
for the value of the complete training history at these budgets, not proof
that original-environment data alone caused it. The optimizer-exposure caveat
between BC and DAgger from Section 7 continues to apply.

Task progress for the DAgger lineages remains much closer: scratch-minus-
continuation changes range from −0.056 m for input DAgger to +0.143 m for input
GAIL-DM. These small pooled differences do not establish a robust algorithm
ranking or statistical significance across independent full training histories.
''')
    say('''### F.3 Mechanical stability and occupancy

Each entry is “continuation / scratch”. Falls are totals over 90 active minutes
per family; divide by 90 for falls per active minute. Jerk rate is displayed in
millions of rad²/s⁶. SWD and direction error use the definitions in Section 2.
''')
    rows=[]
    for a in algorithms:
        for arch in ['hidden','input']:
            c,s=runs(a,arch,'parent'),runs(a,arch)
            rows.append([label(a,arch),f"{sum(r['falls'] for r in c)} / {sum(r['falls'] for r in s)}"]+
                        [f(mean(c,k)/scale)+' / '+f(mean(s,k)/scale) for k,scale in
                         [('mean_forward_velocity_mps',1),('mean_target_direction_error',1),
                          ('joint_occupancy_sliced_wasserstein',1),('jerk_cost_rate_rad2_per_s6',1e6),
                          ('torso_roll_rms_deg',1),('torso_pitch_rms_deg',1)]])
    table(['Family','Falls','Forward m/s','Direction error','Joint SWD','Jerk rate ×10⁶','Roll RMS°','Pitch RMS°'],rows)
    say('''No new supervised-initialized family shows a gross balance collapse. Scratch
GAIL-BC has two falls per architecture versus four/five for continuation, but
these are sparse counts, not a demonstrated population-level reduction. Input
DAgger has eight falls in both cohorts; scratch input GAIL-DM has five versus
four. There is no uniform scratch or context stability advantage.

BC's extra progress also accompanies a larger occupancy distance: hidden SWD
increases from 0.05019 to 0.05737 and input SWD from 0.04886 to 0.06166. DAgger's
occupancy differences are much smaller and not uniformly worse. Lower jerk
alone does not resolve this trade-off and must not be interpreted as energy
efficiency without an energy measurement.
''')
    say('''### F.4 Environment groups, replica variation and context

The following table separates the grouped and confirmation schedules and shows
the range of the three seed-level paired mean progress differences. Seeds share
data and evaluation rounds; this range is descriptive, not a confidence interval.
The scalar-0.1 column is the nine-run scratch mean, not original-scene retention.
''')
    rows=[]
    for a in algorithms:
        for arch in ['hidden','input']:
            deltas=[]
            for seed in [42,43,44]:
                c=[r for r in runs(a,arch,'parent') if r['controller']['seed']==seed]
                s=[r for r in runs(a,arch) if r['controller']['seed']==seed]
                deltas.append(mean(s,'return_target_progress_m')-mean(c,'return_target_progress_m'))
            rows.append([label(a,arch)]+[f(mean(runs(a,arch,suites=(suite,)),'return_target_progress_m')) for suite in ['grouped','confirmation','nominal']]+
                        [f(min(deltas)),f(max(deltas))])
    table(['Family','Grouped m','Confirmation m','Scalar 0.1 m','Min seed Δm','Max seed Δm'],rows)
    rows=[]
    for a in algorithms:
        s,h=runs(a,'input'),runs(a,'hidden')
        rows.append([names[a],f(mean(s,'return_target_progress_m')-mean(h,'return_target_progress_m')),
                     f(mean(s,'held_out_expert_action_mse')-mean(h,'held_out_expert_action_mse')),
                     sum(r['falls'] for r in s)-sum(r['falls'] for r in h)])
    table(['Scratch input minus hidden','Progress Δm','Test MSE Δ','Fall-count Δ'],rows)
    say('''Context remains a small, inconsistent intervention: input BC adds about
0.082 m and input GAIL-BC 0.063 m, whereas input DAgger loses about 0.120 m.
Input GAIL-DM gains about 0.053 m. There is no broad reversal of the earlier
conclusion that a strong hidden-context policy is competitive. Correct-context
evaluation still cannot distinguish useful conditioning from extra capacity;
wrong/constant-context interventions remain missing.

For completeness, the next table reports scratch means by Gaussian parent group,
pooling hidden/input and their three seeds (18 runs per method per parent mean).
The oracle has three rounds per group and is not duplicated as independent data.
Parent mean is a sampling label, not the actual fixed slip used in every run.
''')
    groups=list(dict.fromkeys(r['row']['group'] for r in e['runs'] if r['row']['suite']=='grouped'))
    rows=[]
    for group in groups:
        rr=[r for r in e['runs'] if r['row']['suite']=='grouped' and r['row']['group']==group]
        oracle=[r for r in rr if r['controller']['label']=='oracle']
        vals=[group,f(min(r['row']['slip'] for r in rr))+'–'+f(max(r['row']['slip'] for r in rr)),f(mean(oracle,'return_target_progress_m'))]
        for a in algorithms:
            ss=[r for r in rr if r['controller'].get('initialization_cohort')=='scratch' and r['controller']['algorithm']==a]
            vals.append(f(mean(ss,'return_target_progress_m')))
        rows.append(vals)
    table(['Parent group','Actual slip range','Oracle m','Scratch BC m','Scratch DAgger m','Scratch GAIL-BC m','Scratch GAIL-DM m'],rows)
    say('''### F.5 Direct random-initialized GAIL: stable standing is not successful walking

These actors start randomly at the GAIL stage, unlike the supervised-initialized
families. They still retain the configured supervised anchor and conservative
PPO/discriminator settings; this is not unrestricted or purely adversarial
learning from scratch. The result tests this configuration and 70,000-step
budget, not whether GAIL can ever learn locomotion without pretraining.
''')
    rows=[]
    for domain,suites in [('original',('original',)),('original',('grouped','confirmation')),('varied',('grouped','confirmation'))]:
        rs=runs('gail_random',domain=domain,suites=suites)
        rows.append([domain+' trained / '+('original scene' if suites==('original',) else 'sampled'),len(rs),f(mean(rs,'return_target_progress_m')),
                     f(mean(rs,'normalized_return')),sum(r['targets_reached'] for r in rs),sum(r['falls'] for r in rs),
                     f(mean(rs,'held_out_expert_action_mse')),f(mean(rs,'joint_occupancy_sliced_wasserstein')),f(mean(rs,'jerk_cost_rate_rad2_per_s6')/1e6)])
    table(['Condition','Runs','Progress m','Score','Targets total','Falls total','Test MSE','Joint SWD','Jerk rate ×10⁶'],rows)
    say('''Both direct-random conditions fail to acquire useful target locomotion in
this batch. Varied GAIL-random averages −0.367 m and a normalized score of
−0.0157, compared with scores near one for the supervised initializers.
Its much lower jerk rate is not evidence of better walking: the behavioural
objective is largely unmet. Original GAIL-random has zero falls in the original
scene despite near-zero progress, illustrating why falls cannot stand alone as
a success measure. Negative normalized scores are valid: zero is a random
reference, not an absolute mathematical floor.
''')
    say('''### F.6 What the new probes add

Values below are mean first-layer test R² over three trained seeds, with pooled
expert CKA averaged over those same representations. “Cont. / scratch” retains
the direction used above. These are readouts on the **same oracle states**, not
predictions of each learner's own future trajectories.
''')
    rows=[]
    for a in algorithms:
        for arch in ['hidden','input']:
            c,s=reps(a,arch,'parent'),reps(a,arch,'scratch')
            rows.append([label(a,arch)]+[f(probe(c,target))+' / '+f(probe(s,target)) for target in
                         ['phase_sin_times_roll_deg','future_200ms_forward_velocity_mps','future_200ms_pitch_deg']]+
                        [f(mean([g for g in p['geometry'] if g['representation'] in c],'expert_cka'))+' / '+
                         f(mean([g for g in p['geometry'] if g['representation'] in s],'expert_cka'))])
    table(['Family','Phase × roll R²','Future velocity R²','Future pitch R²','Expert CKA'],rows)
    say('''Continuation preserves more linearly accessible phase-conditioned roll
information across these families. Hidden DAgger is 0.7065 versus scratch
0.6935; input DAgger is 0.7073 versus 0.6854. This is consistent with different
representational organization after longer training, but it does not identify
the cause of the fidelity advantage.

Other properties go the opposite way: scratch DAgger improves future-pitch
readout to about 0.610–0.611 from 0.586–0.588. Scratch hidden DAgger also has
higher expert CKA (0.9428 versus 0.9212), despite worse action MSE. Therefore,
neither higher CKA nor higher R² on one future target is a reliable proxy for
better imitation. GAIL's selected readouts remain very close to its initializer
in both cohorts, reinforcing the conclusion that this conservative update
changes the measured hidden representation little.

The direct-random actors are an especially useful cautionary example: they can
expose information on oracle states while failing to walk themselves. The next
table includes their first layer and the existing random-feature/raw controls.
''')
    rows=[]
    for n,ns in [('Raw63',['raw_observation']),('Untrained random63 h1',[f'random_hidden_{s}_h1' for s in [42,43,44]]),
                 ('Varied GAIL-random h1',reps('gail_random','hidden','scratch'))]:
        rows.append([n]+[f(probe(ns,target)) for target in ['generator_phase_sin','phase_sin_times_roll_deg','future_200ms_forward_velocity_mps','future_200ms_pitch_deg']])
    table(['Representation','Phase sin R²','Phase × roll R²','Future velocity R²','Future pitch R²'],rows)
    say('''Varied GAIL-random future-velocity R² is about 0.6114—close to the untrained
random-feature control—despite failed locomotion. Decoding information supplied
by oracle observations is not evidence that the policy learned to exploit it.
The original width/nonlinearity confound with the expert, periodic shuffled-label
controls and absence of causal interventions still apply. No R1 claim follows.

### F.7 Revised overall conclusions

1. **Original-environment pretraining is not necessary for viable varied-domain walking at these budgets.** Scratch BC and mixed DAgger, and GAIL initialized from them, reach approximately oracle-level progress. Their complete training histories differ from continuation, so this is not a matched-lifetime compute claim.
2. **Pretrained continuation remains substantially better at matching actions.** Approximately fivefold lower errors coexist with similar or slightly lower progress. Fidelity and performance should be reported separately, not collapsed into one winner.
3. **DAgger's fidelity advantage survives scratch initialization.** The varied-domain result is not simply inherited from the original DAgger parent. Extra supervised optimization remains a competing explanation alongside learner-state coverage.
4. **GAIL strongly depends on a competent initializer under this configuration.** Direct random starts fail; supervised starts work. Additional GAIL yields small, mixed behavioural differences and near-initializer probe profiles, not a demonstrated general improvement.
5. **Explicit slip context remains marginal and inconsistent.** Scratch initialization does not reveal a broad context advantage absent in the continuation batch.
6. **Representational similarity, decodability and control success are distinct.** Higher scratch DAgger CKA coexists with worse fidelity, and failed GAIL-random still admits informative probes. These are stronger reasons to avoid interpreting CKA as a percentage of transferred knowledge.

These comparisons reuse controls and draws; they are not new independent
confirmation experiments. Original-scene retention of varied-trained models,
matched lifetime optimization, independent full training histories, context
interventions and causal feature ablations remain open. The new data strengthen
the descriptive findings without resolving those causal questions.

### F.8 New source artifacts and reproducibility

- [Combined evaluation JSON](../experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json) and [per-replica CSV](../experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.csv).
- [Scratch training summary](../experiments/consolidated/v1/scratch_v1/training_summary.json).
- [Combined probe report](../experiments/consolidated/v1/scratch_v1/probing/report.json) and [paired initialization probe comparisons](../experiments/consolidated/v1/scratch_v1/probing/initialization_comparison.csv).
- [Scratch protocol](../scratch_comparison/README.md); [read-only extension builder](analysis/scratch_summary.py).
- [Preserved report before this extension](report_archive/CONSOLIDATED_RESULTS_REPORT_2026-10-15_before_scratch_2026-10-16.md). Earlier meeting summaries and source experiments are unchanged.

The papers in Appendix E remain the methodological references. This extension
does not introduce a new learning algorithm or claim that literature establishes
the measured outcomes. Source hashes below fix the new evidence used here.
''')
    table(['Source','SHA-256'],[[str(path.relative_to(root)),hashlib.sha256(path.read_bytes()).hexdigest()] for path in paths])
    return '\n'.join(out)

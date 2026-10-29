"""Print the staged consolidated results report from immutable JSON artifacts."""
import collections
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EP=ROOT/'experiments/consolidated/v1/evaluation/all_all_correct/comparison.json'
PP=ROOT/'experiments/representation_probing/consolidated_v1/report.json'
TP=ROOT/'experiments/consolidated/v1/training_summary.json'
E=json.loads(EP.read_text());P=json.loads(PP.read_text());T=json.loads(TP.read_text())
RUNS=E['runs'];PR=P['results']
A=['bc','dagger_mixed','gail_bc','gail_dagger_mixed']
N={'bc':'BC','dagger_mixed':'DAgger-M','gail_bc':'GAIL-BC','gail_dagger_mixed':'GAIL-DM'}
F=[('original','hidden',a) for a in A]+[('varied',h,a) for h in ['hidden','input'] for a in A]
OUT=[]
def say(s=''):
    lines=[]
    for line in s.strip().splitlines():
        if not line.startswith('|') and '`' not in line and '](' not in line:
            line=re.sub(r'(?<=[A-Za-z²])(?=\d)|(?<=\d)(?=[A-Za-z])(?!e[+-]?\d)', ' ', line)
            line=re.sub(r'(?<=⁶)(?=[A-Za-z])', ' ', line)
            line=re.sub(r'(?<=[%°])(?=[A-Za-z])', ' ', line)
            line=re.sub(r',(?=[A-Za-z])|(?<=[A-Za-z]),(?=\d)', ', ', line)
            line=re.sub(r':(?=[A-Za-z0-9])', ': ', line)
            line=re.sub(r'(?<=\))(?=[A-Za-z])', ' ', line)
            line=line.replace('R 1','R1').replace('PPOsteps','PPO steps').replace('h 1','h1').replace('h 2','h2')
        lines.append(line)
    OUT.append('\n'.join(lines)+'\n')
def table(head,rows):
    say('| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows))
def fmt(v,n=4):return '—' if v is None else f'{v:.{n}f}'
def label(f):return ('Frozen' if f[0]=='original' else ('Varied hidden' if f[1]=='hidden' else 'Varied input'))+' '+N[f[2]]
def select(f,suites=('grouped','confirmation')):
    return [r for r in RUNS if r['row']['suite'] in suites and (r['controller']['domain'],r['controller']['architecture'],r['controller']['algorithm'])==f]
def avg(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]
    return float(np.mean(vs)) if vs else None
def falls(rs):return sum(r['falls'] for r in rs)
def probe_names(f,layer):
    return [c['label']+f'_h{layer}' for c in json.loads((EP.parent/'schedule.json').read_text())['controllers']
            if (c['domain'],c['architecture'],c['algorithm'])==f]
def probe(names,target,mode='true',key='r2'):
    vs=[r[key] for r in PR if r['representation'] in names and r['target']==target and
        (r['labels']=='true' if mode=='true' else r['labels']!='true') and r[key] is not None]
    return float(np.mean(vs)) if vs else None
def geom(names,k):return np.mean([r[k] for r in P['geometry'] if r['representation'] in names])

assert len(RUNS)==1020 and len(PR)==9000 and len(T['completed_policies'])==28
assert all(r['active_steps']==3000 and r['normalization_valid'] for r in RUNS)
assert len(list((PP.parent/'probes').glob('*/*/completed.json')))==300

say('''# Consolidated imitation-learning results: experimental design, environmental robustness and representations

Originally reported 15 October 2026; updated 16 October 2026. Sections 1–10 and Appendices A–D describe the parent consolidated generation: 28 policies, 1,020 deployments and 300 probe fits. **Appendix F adds the completed scratch comparison**, bringing the total to 56 policies, 1,949 deployments and 524 fits. Appendix E retains the paper reference guide. Historical four-policy, slip-sensitivity and FiLM studies remain separate.

**Latest findings:** Scratch supervised lineages walk successfully without original-environment pretraining, but continuation retains approximately fivefold better action fidelity. Mixed DAgger remains the strongest supervised fidelity baseline. Direct random-initialized GAIL fails to acquire useful locomotion at this budget; explicit context remains inconsistent. See Appendix F for numerical comparisons, design and interpretation.

**Reading guide:** Sections1–2 explain design and metrics; Section3 covers original-environment results; Section4 tests frozen policies on changed surfaces; Section5 covers varied training and context; Section6 focuses on DAgger; Section7 accounts for training resources; Section8 interprets probes; Sections9–10 synthesize findings and limitations. Appendices retain additional numerical comparisons and source links.

## Executive summary

The main result is not a large gain in distance travelled. It is the separation between **task performance**, **imitation fidelity**, and **mechanical reliability**. In the original environment, all four learned controllers achieve approximately oracle-level target progress without recorded falls. Across the sampled surfaces, the original frozen policies remain close to oracle-normalized performance, but incur more falls and larger action errors. Varied-domain continuation produces substantially closer imitation and generally fewer falls, while average progress remains near the same oracle ceiling.

Mixed DAgger is particularly important: it has markedly lower action error than BC in this generation, and varied-trained DAgger retains that advantage. However, its training also uses more supervised optimizer work, so this cannot be attributed to learner-state queries alone. Explicit slip conditioning adds small and inconsistent behavioural gains; it is not a clear replacement for a strong hidden-context model.

The probes identify accessible phase, posture, contact-proxy, command and short-horizon motion information. They also show that the GAIL representations remain almost indistinguishable, at the displayed precision, from their supervised initializers. High decodability is not proof that a feature is used for control, nor proof of transfer to another robot.

## 1. Reading the experimental design

### 1.1 Three stages, followed by representation analysis

1. **Original environment:** train BC, mixed DAgger, GAIL-BC and GAIL-DM, then evaluate these four in the original installed scene.
2. **Frozen-policy environmental tests:** take those exact four checkpoints, change the environment, and evaluate without further learning.
3. **Varied-domain training and testing:** continue learning across five fixed training surfaces, with either hidden slip or explicit slip input; evaluate these new policies on the same sampled-slip rounds as the frozen models.
4. **Offline probes:** compare all actors and the expert on the same recorded oracle states, including matched random-network and raw-input controls.

“Original” is deliberately not called zero slip. The original arm preserves the installed legacy scene byte-for-byte. Corrected scalar slip0.1 is a separate physical condition, not a synonymous nominal label. Likewise, “Gaussian evaluation” means **draw one positive slip per round, then keep it fixed for the whole deployment**. It is not a spatially varying field or a new draw at every foot contact.

### 1.2 Which models were trained?

DAgger-M denotes mixed long/short-run DAgger. GAIL-DM denotes GAIL initialized from this new **mixed DAgger**, not the original-DAgger GAIL of the historical study. In every case FC Portugal remains the expert; BC/DAgger initialize GAIL but do not replace the expert demonstration source.
''')
table(['Family','Training environment','Network observations','Replicas','Models'],[
['Original four','Installed original scene','63 inputs; slip hidden','Seed42','4'],
['Varied hidden four','Scalar slip0.025,0.1,0.2,1,10','63 inputs; slip hidden','Seeds42/43/44','12'],
['Varied input four','Same five surfaces','63 inputs + normalized slip','Seeds42/43/44','12']])
say('''Each actor has two256-unit ReLU hidden layers and16 outputs. Input-conditioned actors append `log1p(slip)/log(11)`, creating64inputs; the extra input column begins at zero when expanding the original63-input checkpoint. These are not64-to16 single-layer policies. The16 outputs are latent gait-generator actions/residuals executed through the existing walking machinery, not22 independent motor velocities.

Original BC and DAgger start from the same seeded random weights. Varied BC and DAgger start from their corresponding newly trained original controller. Each varied GAIL actor starts from the corresponding varied supervised controller with the same architecture and seed. Therefore the three varied replicas are **continuation replicas sharing an original pretrained ancestor and expert pools**, not three independent repetitions of the entire from-scratch experiment.

### 1.3 Training rounds and data composition

Data collection balances long runs and short target changes. Short targets are sampled1–2m from the current position in the own-half region, with a0.4m arrival threshold; long targets use the own-half sampler with a0.5m threshold. This maintains the long-run/short-turn combination requested for mixed DAgger. All controllers share the simulator contract, observation format and action interface.
''')
table(['Procedure','Starting supervised data','New learner transitions','Final expert-labelled training pool','Update design'],[
['BC','170,000 oracle examples','0','170,000','100 supervised epochs'],
['DAgger-M','100,000 oracle examples','7×10,000','100,000 +70,000 =170,000','50 epochs initially and after each aggregation'],
['GAIL-BC','170,000 static oracle examples +BC initializer','7×10,000','170,000 static examples','Each round: discriminator1epoch +PPO up to3epochs'],
['GAIL-DM','170,000 static oracle examples +DAgger-M initializer','7×10,000','170,000 static examples','Same GAIL update schedule']])
say('''The seed pool is100,000 examples; an extra70,000 oracle examples completes the shared BC/GAIL pool. Separate validation and test pools contain10,000 examples each per environment domain. These are not included in the170,000 training count. BC/GAIL share oracle data rather than collecting separate copies for every replica; DAgger adds its own learner-state labels.

For varied training, **each collection round has ten segments: five surfaces ×two target modes**. The100,000-example seed pool has10,000 per segment; the70,000-example extra pool has7,000; each validation/test pool has1,000. Each10,000-step DAgger/GAIL learner round contributes1,000 transitions per surface/mode segment. Thus every learner iteration sees all five training slips, rather than proceeding through a low-to-high-slip curriculum. Original-domain training retains the same ten-segment structure, but all segments use the unchanged original scene.
''')
table(['DAgger stage','New learner labels','Aggregated training examples','Oracle execution probability β','Supervised epochs'],
      [['Initial',0,'100,000','Oracle seed data',50]]+
      [[i,'10,000',f'{100000+i*10000:,}',fmt(max(0,1-.2*i),1),50] for i in range(1,8)])
say('''DAgger queries the oracle at every collected learner-state example, including when the learner executes the action. β chooses **which action is executed**, not whether an expert label is obtained. It decays0.8,0.6,0.4,0.2,0,0,0. BC remains offline supervised training. GAIL instead samples its stochastic actor during collection, stores executed actions/log-probabilities/values, and learns from discriminator rewards; evaluation uses deterministic actor means.

BC/DAgger use Adam at1e-4, batch256, selecting the lowest validation-MSE checkpoint within each update. GAIL uses actor/critic/discriminator learning rates3e-6/3e-4/1e-4, gradient-penalty regularization, a supervised anchor and KL safeguards. Rollout segment ends and fall resets are handled as separate boundaries for return estimation. These conservative GAIL settings matter when interpreting the small later changes in representations.

### 1.4 Evaluation rounds and what the seeds mean
''')
table(['Suite','Environmental rounds','Controllers each round','Deployments','Role'],[
['Original','5 task seeds:101/202/303/404/505','4 original +oracle +random','30','Original-scene performance'],
['Grouped Gaussian','5parent means ×3draws','28 learned +oracle +random','450','Common varied-environment comparison'],
['Confirmation Gaussian','5parent means ×3other draws','Same30','450','Separate reused confirmation schedule'],
['Corrected nominal','Scalar0.1 ×3task seeds','Same30','90','Nominal retention under corrected physics'],
['Total','38environmental rounds','—','1,020','3,060,000 active evaluation steps']])
say('''Every deployment has3,000 active steps at20ms per step:60seconds of scored simulated control. Startup and fall recovery add simulator time outside that active duration. Each controller receives a fresh owned server, the same physical slip and task seed for a given round, own-half target sampling, and the same fall/reset rules. Checkpoints are frozen throughout evaluation. The server scene hash/import is verified; this is verification of the loaded configuration, not direct live ODE readback.

Training seeds42/43/44 control continuation/training randomness. Evaluation seeds control the task sequence; the Gaussian schedule uses its own reproducible seed construction. They are different sources of variation. Pairing the seed makes starting/task randomness comparable, but achieved trajectories and target-switch timing can still diverge between policies.

The two sampled suites use parent means0.025,0.1,0.2,1,10 and standard deviations equal to half each mean, truncated to(0,10]. A parent mean is not the empirical mean after truncation, particularly for the parent10 distribution. The actual realized ranges are listed below. The existing schedules were previously inspected elsewhere; “confirmation” here does not mean a newly blind dataset.
''')
rows=[]
for mu in [.025,.1,.2,1.,10.]:
    rr={r['row']['id']:r['row'] for r in RUNS if r['row'].get('parent_mean')==mu}
    ss=[r['slip'] for r in rr.values()]
    rows.append([mu,mu/2,len(ss),fmt(min(ss),6),fmt(max(ss),6),fmt(np.mean(ss),6)])
table(['Parent μ','Parent σ','Unique draws','Realized minimum','Realized maximum','Realized mean'],rows)
say('''### 1.5 How numbers are aggregated

Stage1 averages five deployments per original policy. For the main environmental tables, grouped and confirmation are combined:30deployments per frozen policy and30per varied training replica, or90per varied family. Because each replica has the same30rounds, pooling these balanced runs gives the same point mean as averaging the three replica means. **It does not make90 independent training experiments.** Fall counts are always shown with their exposure; replica ranges and paired context differences are reported separately.

The three corrected-nominal rounds are not mixed into the30sampled-round robustness summary. The original five-seed scene is not mixed with corrected scalar0.1. Original checkpoints are only one training seed, and all varied replicas inherit shared parents. The report is primarily descriptive; it does not use individual frames as independent statistical observations or assert significance from small point differences.

## 2. Metric definitions and practical interpretation
''')
table(['Metric','What it measures','How to read it'],[
['Target progress J (m)','Sum of reduction in distance to the current target','Higher is better; not total path length or final straight-line displacement'],
['Normalized return','(Jpolicy−Jrandom)/(Joracle−Jrandom), paired within each round','0=random reference;1=oracle parity; may exceed1 or be negative'],
['Held-out action MSE','Mean squared error per16latent-action element on10,000held-out oracle examples','Lower=closer local imitation on shared expert states'],
['On-policy action MSE','Same comparison on states each controller visits','Lower=closer local imitation, but state distributions differ'],
['Joint occupancy SWD','256-projection sliced Wasserstein over22joint angles, expert-standardized','Lower=closer occupancy; unitless and not exact22-D transport'],
['Forward velocity (m/s)','Body-heading projected translation per active step','Higher normally indicates forward locomotion; target progress still matters'],
['Direction tracking error','Norm of difference between unit movement and target directions','Lower=better; dimensionless, not degrees; undefined movement assigned1'],
['Falls/active minute','Detected balance losses divided by scored active minutes','Lower=better; reset time excluded; not a count restricted to unrecoverable falls'],
['Torso roll/pitch RMS (degrees)','Square-root mean squared orientation about zero','Lower means less angular excursion, not necessarily better locomotion by itself'],
['Angular jerk rate','Integrated squared third joint-angle derivative /valid differentiated duration','Lower=less high-frequency kinematic change, not measured torque or energy'],
['Probe R² /RMSE','Held-out predictability from a representation','Higher R²/lower RMSE=more linearly accessible labelled information'],
['CKA /participation ratio','Representation geometry similarity /effective variance dimension','Descriptive structure, not percent knowledge or causal usefulness']])
say('''The originally proposed squared-vector action error is16times the element-mean MSE tabulated here; the action interface is latent16-D, not directly22joint-velocity commands. The occupancy metric discards temporal ordering, so “similar trajectories” must not be inferred from SWD alone. It is normalized using each matched oracle run’s joint variation.

Jerk uses joint positions converted to radians, five-frame smoothing, then a third finite difference at20ms. It is computed separately inside recovery-delimited episodes. The integrated quantity has units rad²/s⁵; its duration-normalized rate has units rad²/s⁶. Tables scale the rate by10⁶ for readability. The metric is sensitive to derivative estimation, smoothing and episode edges. It is a **kinematic smoothness proxy**, not evidence of lower actuator torque, energy consumption or hardware wear.

Normalized scores are not clipped. A score near1 means matching an oracle facing the same surface, even when both walk much more slowly than in the original environment. The random policy is a reference, not an absolute mathematical floor. All1,020normalization denominators in these results are valid.

## 3. Stage1 — Original-environment policies in the original environment

### 3.1 Task performance and mechanical stability
''')
orig=[('baseline','hidden','oracle')]+F[:4]+[('baseline','hidden','random')]
table(['Controller','Progress m','Normalized score','Forward m/s','Direction error','Falls /5min','Jerk rate ×10⁶'],[
[(f[2].title() if f[0]=='baseline' else N[f[2]]),fmt(avg(select(f,['original']),'return_target_progress_m'),3),
 fmt(avg(select(f,['original']),'normalized_return')),fmt(avg(select(f,['original']),'mean_forward_velocity_mps')),
 fmt(avg(select(f,['original']),'mean_target_direction_error')),falls(select(f,['original'])),fmt(avg(select(f,['original']),'jerk_cost_rate_rad2_per_s6')/1e6,3)] for f in orig])
say('''All learned policies are close to the oracle’s44.446m target progress and have zero recorded falls in these five active minutes each. The best point score is BC1.0044, but the original per-policy descriptive95%intervals all include1. This supports successful oracle-level task imitation, not a meaningful ordering of the four controllers by return. Zero observed falls over five minutes does not establish a zero true fall probability.

For numerical context, the stored normalized-return intervals are BC[0.99359,1.01517],DAgger-M[0.98766,1.00927],GAIL-BC[0.99586,1.00911],GAIL-DM[0.98661,1.00015]. These use mean±1.96×standard error across five deployments; they are approximate descriptive intervals, not training-seed uncertainty or a multiple-comparison test.

### 3.2 Imitation fidelity separates models that look equally successful
''')
table(['Controller','Held-out MSE','On-policy MSE','Joint SWD','Roll RMS°','Pitch RMS°'],[
[N[f[2]],fmt(avg(select(f,['original']),'held_out_expert_action_mse'),6),fmt(avg(select(f,['original']),'on_policy_latent_action_mse'),6),
 fmt(avg(select(f,['original']),'joint_occupancy_sliced_wasserstein'),5),fmt(avg(select(f,['original']),'torso_roll_rms_deg'),3),fmt(avg(select(f,['original']),'torso_pitch_rms_deg'),3)] for f in F[:4]])
say('''Mixed DAgger’s held-out MSE is0.015345 versus BC0.110467:approximately86.1%lower. Its on-policy MSE is0.007821 versus0.074445:approximately89.5%lower. This is a strong fidelity difference despite nearly identical progress. GAIL-DM largely retains the mixed-DAgger fidelity scale; GAIL-BC remains close to BC’s error scale.

BC/GAIL-BC have somewhat lower jerk rates, but DAgger’s joint occupancy is at least comparable and its action error is much smaller. No single scalar identifies the universally “best” gait. In particular, the original task is sufficiently forgiving that considerably different latent-action errors can produce similar progress.

This is a new generation with standardized unique-example counts, a changed mixed-DAgger GAIL lineage and newly collected data. Earlier reports in which BC led on some fidelity metrics should not be silently overwritten or pooled with these values. This study establishes the current batch’s ranking, not a universal algorithmic ordering across all previous experimental designs.

## 4. Stage2 — Frozen original policies on sampled-slip surfaces

No policy weights change in this stage. This isolates how the original controllers behave when the environment changes, before any varied-domain continuation.
''')
table(['Controller','Progress m','Normalized score','Held-out MSE','On-policy MSE','Joint SWD','Falls /30min','Falls/min'],[
[(f[2].title() if f[0]=='baseline' else N[f[2]]),fmt(avg(select(f),'return_target_progress_m'),3),fmt(avg(select(f),'normalized_return')),
 fmt(avg(select(f),'held_out_expert_action_mse'),6),fmt(avg(select(f),'on_policy_latent_action_mse'),6),
 fmt(avg(select(f),'joint_occupancy_sliced_wasserstein'),5),falls(select(f)),fmt(falls(select(f))/30,3)] for f in orig])
say('''Average oracle progress drops to32.885m across the sampled surfaces, about26.0%below its original-scene mean. This broad comparison mixes different scene physics and task seeds, so it is descriptive rather than a matched estimate of a slip coefficient’s causal effect. Within the corrected varied suite, the slip-group analysis below makes the environmental burden much clearer.

The frozen policies still achieve scores0.9814–0.9983 relative to the same-round oracle, yet the stability/fidelity picture changes. BC and GAIL-BC each record11falls in30active minutes, compared with3for the oracle. Mixed DAgger has6, and GAIL-DM3. Frozen BC’s varied-pool held-out MSE is0.178830 and on-policy MSE0.290978. Mixed DAgger remains much closer at0.016337 and0.026471 respectively.

This demonstrates why oracle normalization must be accompanied by absolute performance and stability: high normalized return does not mean the environment had little effect. A controller can recover, continue scoring progress and still have a materially worse fall profile.

### 4.1 Environmental burden by Gaussian parent group

Each row below combines the three grouped and three confirmation draws for that parent distribution. The labels identify Gaussian **parents**, not exact fixed physical values. In particular, parent10 produced actual slips2.905–8.960, not six tests at10.
''')
table(['Parent μ','Oracle progress m','Frozen BC m','Frozen DAgger-M m','Frozen GAIL-BC m','Frozen GAIL-DM m','Oracle falls /6min'],[
[mu]+[fmt(avg([r for r in select(f) if r['row']['parent_mean']==mu],'return_target_progress_m'),3) for f in [('baseline','hidden','oracle')]+F[:4]]+
[falls([r for r in select(('baseline','hidden','oracle')) if r['row']['parent_mean']==mu])] for mu in [.025,.1,.2,1.,10.]])
say('''Oracle mean progress decreases from42.148m in the lowest parent group to17.476m in the highest, a58.5%reduction. Frozen controllers follow a similar decline. These groups also use different task seeds, but the broad monotonic environmental trend is shared across controllers. A score near1 at high slip should therefore be interpreted as keeping pace with a degraded oracle, not preserving low-slip speed.

## 5. Stage3 — Training across varied slip

### 5.1 Main hidden-context and input-conditioned comparison

The following means cover the same30sampled rounds used for the frozen models. Each varied-family row contains three continuation replicas,90deployments in total. Hidden models must act using their ordinary sensors/history-related gait inputs; input models additionally receive the actual scalar slip.
''')
table(['Family','Progress m','Normalized score','Held-out MSE','On-policy MSE','Joint SWD','Falls /90min'],[
[label(f),fmt(avg(select(f),'return_target_progress_m'),3),fmt(avg(select(f),'normalized_return')),
 fmt(avg(select(f),'held_out_expert_action_mse'),6),fmt(avg(select(f),'on_policy_latent_action_mse'),6),fmt(avg(select(f),'joint_occupancy_sliced_wasserstein'),5),falls(select(f))] for f in F[4:]])
say('''Varied hidden BC reduces held-out error from the frozen BC’s0.178830 to0.019949, an88.8%reduction on the same varied expert test pool. Varied hidden DAgger reduces0.016337 to0.001866, an88.6%reduction. These are additional-learning comparisons: varied continuations inherited original training and then received more data/updates. This batch does **not** contain a matched extra-training-on-original-only continuation control, so the improvement cannot be attributed solely to environmental diversity.

Fall rates improve more clearly than average target progress. Hidden BC falls4times/90min(0.044/min), versus frozen BC11/30min(0.367/min). Hidden DAgger falls5/90min(0.056/min), versus frozen DAgger6/30min(0.200/min). Hidden GAIL-BC falls4/90min and hidden GAIL-DM6/90min. Counts are small and exposure/replica structure differ; these are descriptive reductions, not formal causal risk estimates.

### 5.2 Mechanical behaviour, beyond falls
''')
table(['Family','Forward m/s','Direction error','Roll RMS°','Pitch RMS°','Jerk rate ×10⁶','Integrated jerk ×10⁹'],[
[label(f),fmt(avg(select(f),'mean_forward_velocity_mps')),fmt(avg(select(f),'mean_target_direction_error')),
 fmt(avg(select(f),'torso_roll_rms_deg'),3),fmt(avg(select(f),'torso_pitch_rms_deg'),3),fmt(avg(select(f),'jerk_cost_rate_rad2_per_s6')/1e6,3),fmt(avg(select(f),'jerk_integrated_rad2_per_s5')/1e9,3)] for f in F])
say('''Varied models cluster near0.516–0.520m/s forward velocity and roughly2.6°roll RMS, close to the oracle’s0.521m/s and2.607°. BC-derived varied models approach the oracle’s mechanical pattern rather than preserving the lower-jerk but less stable frozen BC pattern. DAgger-derived models remain around41×10⁶jerk-rate units. A lower jerk value alone would have incorrectly favoured frozen BC despite its higher fall frequency and action error.

### 5.3 What does explicit context add?

Compare each input policy with its hidden counterpart **at the same training seed and evaluation round**. The table averages the90paired differences over the30sampled rounds and three replicas. Fall differences are total input-minus-hidden counts over equal90minute exposure.
''')
look={(r['row']['id'],r['controller']['label']):r for r in RUNS}
rows=[]
for a in A:
    rr=select(('varied','input',a));ds=[];fds=[];byseed=[]
    for r in rr:
        c=r['controller'];h=look[(r['row']['id'],f'varied_hidden_{c["seed"]}_{a}')]
        ds.append(r['return_target_progress_m']-h['return_target_progress_m']);fds.append(r['falls']-h['falls'])
    for seed in [42,43,44]:
        ss=[r for r in rr if r['controller']['seed']==seed]
        byseed.append(np.mean([r['return_target_progress_m']-look[(r['row']['id'],f'varied_hidden_{seed}_{a}')]['return_target_progress_m'] for r in ss]))
    rows.append([N[a],fmt(np.mean(ds),4),fmt(100*np.mean(ds)/avg(select(('varied','hidden',a)),'return_target_progress_m'),3),sum(fds),'/'.join(fmt(v,4) for v in byseed)])
table(['Method','Mean Δprogress m','Relative Δprogress %','Δfalls /90min','Δprogress by seed42/43/44 m'],rows)
say('''The mean changes are small:about+0.099m for BC,+0.009m for DAgger,+0.055m for GAIL-BC and−0.029m for GAIL-DM over a60second deployment. There is no consistent reduction in falls or action error from supplying slip. Input DAgger has8falls versus hidden DAgger5; input GAIL-DM has4versus6, illustrating that even the direction of the stability difference depends on the method.

Only correct context was evaluated in this batch. There are no newly run wrong-context/constant-context interventions here, so small input-policy gains cannot establish beneficial causal use of slip. The oracle itself is not explicitly conditioned on slip; it reacts through its state inputs. That limits the direct supervised signal to exploit the extra scalar, but it does not prove the scalar is intrinsically useless—particularly for future-dynamics prediction or another learning objective.

### 5.4 Corrected-nominal retention

These are scalar0.1 tests, three task seeds per replica. They test performance near a training condition, not return to the byte-identical original scene.
''')
table(['Family','Progress m','Normalized score','Falls /deployments','Sampled/nominal progress %'],[
[label(f),fmt(avg(select(f,['nominal']),'return_target_progress_m'),3),fmt(avg(select(f,['nominal']),'normalized_return')),
 f'{falls(select(f,["nominal"]))}/{len(select(f,["nominal"]))}',fmt(100*avg(select(f),'return_target_progress_m')/avg(select(f,['nominal']),'return_target_progress_m'),2)] for f in F])
say('''Corrected-nominal progress remains near the oracle’s39.194m. This is evidence against a large loss at scalar0.1, not proof that the varied models retain their original-scene performance: the new varied policies were not evaluated in that original scene. The low nominal sample count also limits conclusions about rare falls. The final column is the ratio of mean progress over the30sampled rounds to mean progress in the three scalar-nominal rounds (balanced over replicas). It quantifies descriptive environmental retention, but uses different task seeds and is not a paired causal estimate of slip degradation.

## 6. Dedicated analysis — DAgger in varied-slip environments

### 6.1 Why DAgger is a stronger fidelity baseline here

Unlike BC, DAgger repeatedly asks the oracle what to do in states visited by its evolving controller. It also sees both target modes on every training surface in every aggregation round. This is a plausible reason for its substantially lower error on both expert and learner states, but the current design does not isolate that mechanism from its much larger supervised optimization exposure.

On sampled surfaces, hidden varied DAgger’s held-out MSE0.001866 is90.6%lower than hidden varied BC0.019949; on-policy MSE0.002119 is90.0%lower than BC0.021086. Joint occupancy SWD also improves,0.043459 versus0.050188. Yet progress differs by only0.061m per minute, and DAgger has5falls against BC4over90minutes. **DAgger is clearly closer to the oracle’s actions, but not clearly the universally safest or fastest controller.**

### 6.2 Where environmental gains appear
''')
table(['Parent μ','Frozen DAgger progress m','Hidden varied m','Input varied m','Frozen falls /6min','Hidden falls /18min','Input falls /18min'],[
[mu]+[fmt(avg([r for r in select(f) if r['row']['parent_mean']==mu],'return_target_progress_m'),3) for f in [('original','hidden','dagger_mixed'),('varied','hidden','dagger_mixed'),('varied','input','dagger_mixed')]]+
[f'{falls([r for r in select(f) if r["row"]["parent_mean"]==mu])}' for f in [('original','hidden','dagger_mixed'),('varied','hidden','dagger_mixed'),('varied','input','dagger_mixed')]] for mu in [.025,.1,.2,1.,10.]])
say('''The improvement is not uniform. Around parent1, frozen DAgger records2falls in6minutes; hidden varied DAgger records none in18minutes, while mean progress rises27.690→28.061m. At the highest parent group, progress rises17.245→17.506m for hidden and17.700m for input. At parent0.1, hidden continuation is essentially unchanged in progress; the input version is lower. It would therefore be misleading to report only one favourable high-slip example or claim monotonic benefits from conditioning.

### 6.3 Repeatability across continuation seeds
''')
table(['DAgger variant','Training seed','Mean normalized score','Progress m','Held-out MSE','Falls /30min'],[
[h,seed,fmt(avg(rs,'normalized_return')),fmt(avg(rs,'return_target_progress_m'),3),fmt(avg(rs,'held_out_expert_action_mse'),6),falls(rs)]
 for h in ['hidden','input'] for seed in [42,43,44]
 for rs in [[r for r in select(('varied',h,'dagger_mixed')) if r['controller']['seed']==seed]]])
say('''All six varied DAgger replicas stay close to oracle-normalized progress. The hidden replicas have1–2falls per30minutes; input replicas2–3. This makes hidden varied DAgger a particularly strong **fidelity-first reference**, without evidence that the context input is needed for the observed robustness. These are three continuations of a common ancestor; a stronger repeatability claim would require independent original training seeds as well.

### 6.4 Does adding GAIL improve the DAgger starting point?

For hidden varied policies, GAIL-DM changes progress32.844→32.851m, held-out MSE0.001866→0.002167 and falls5→6over90minutes. The corresponding input GAIL-DM progress is slightly lower than input DAgger, while its fall count is lower. These trade-offs do not support a reliable improvement from the extra70,000PPOsteps. The sensible conclusion is that GAIL mostly preserves a strong initializer under the tested regularization and budget, not that adversarial training is generally ineffective.

## 7. Algorithmic efficiency and resource accounting

The standardized170,000figure refers to unique expert-labelled examples, not optimizer steps. GAIL’s70,000figure refers to active on-policy environment transitions, not PPO gradient updates. Initializer and shared-data costs must be kept visible.
''')
table(['Per training phase','BC','DAgger-M','Each GAIL variant'],[
['Expert training examples','170,000','100,000seed +70,000learner labels','170,000static'],
['New learner transitions','0','70,000','70,000'],
['Live oracle learner-state queries','0','70,000','0'],
['Supervised sample exposures','17,000,000','54,000,000','Initializer cost +BC-anchor use'],
['Supervised gradient updates','66,500','211,050','Initializer cost +PPO/discriminator updates'],
['Inherited model cost','None original; originalBC if varied','None original; originalDAgger if varied','Corresponding trained supervised initializer']])
say('''DAgger’s eight supervised fits expose100k+110k+…+170k examples for50epochs each, giving54million sample exposures, versus BC’s17million. DAgger therefore receives about3.18times the supervised optimization exposure, even though both end with170,000unique labels. The result supports the implemented DAgger pipeline, but it is not a clean test of equal-compute BC versus DAgger.

Across the batch, the recorded unique collection totals are1,850,000active steps and870,000oracle queries, including shared validation/test collection. The active-step total decomposes into380,000shared original+varied expert examples,490,000learner steps for seven DAgger replicas, and980,000for fourteen GAIL replicas. Of the870,000queries,380,000belong to the shared expert collections and490,000to DAgger learner-state labels. Startup/recovery counters are logged separately; these are not wall-clock or energy measurements.

All98GAIL update rounds completed without an actor rollback flag. This does not establish unconstrained learning: the actor learning rate, KL checks and supervised anchor still strongly limit drift. Final-policy evaluation alone does not supply normalized-return convergence curves over intermediate query counts or PPO steps. Claims about sample complexity or convergence speed should therefore remain narrower than claims about final quality.

## 8. Stage4 — What the probing experiment actually tests

### 8.1 Shared states, held-out trajectories and controls

All28actors and the expert are frozen. The probes use the30oracle trajectories from the new grouped/confirmation evaluation, not the learner traces. Grouped draws0/1 supply10training runs; grouped draw2 supplies5validation runs; confirmation supplies15test runs. After stride7sampling and exclusion of reset/cross-recovery windows, the counts are4,270training,2,132validation and6,403test states. The future horizon is10steps, or0.2seconds.

The30targets span generator phase, force/contact proxies, torso/gyro state, ankle geometry, commands, phase-posture interactions and six future outcomes. Current properties decoded directly from observations are retention checks; future outcomes are under **oracle actions**, not the candidate’s own actions. Slip is not a target. A thresholded force indicator is not a separately measured contact event, and a phase-posture product is not a validated physical balance margin.

There are75representations:56student hidden layers,12random student layers across both63/64input architectures and three seeds, one expert layer, three random-expert layers, raw63input, raw64input including context, and restricted phase/command input. Four label conditions—true plus three trajectory-shuffled controls—produce300multitarget fits and9,000target scores. The expert’s saved weights reproduce recorded oracle actions before the audit accepts the data.

All probes are standardized linear ridge regressions. Scaling uses training data; each target selects its penalty using validation macro-run MSE, not the test set. Binary contact proxies also report thresholded accuracy/balanced accuracy, without claiming calibrated probabilities. No original policy is retrained and no layer-reset intervention is performed in this new probing batch.

### 8.2 Representative properties in the first hidden layer

Values below are held-out R². For varied models/random controls, each cell averages the corresponding three network seeds; this is a descriptive average of separate fits. Raw-input controls have no hidden layer. A value1means perfect prediction on these targets;0matches the test-target mean-variance baseline;negative values are allowed.
''')
targets=['generator_phase_sin','left_force_contact_gt10N','torso_roll_deg','phase_sin_times_roll_deg','future_200ms_forward_velocity_mps','future_200ms_pitch_deg']
base=[('Raw63',['raw_observation']),('Raw64+slip',['raw_observation_with_context']),('Phase/command only',['phase_command_only']),('Expert h1',['expert_h1']),('Random expert h1',[f'expert_random_{s}_h1' for s in [42,43,44]]),('Random63 h1',[f'random_hidden_{s}_h1' for s in [42,43,44]]),('Random64 h1',[f'random_input_{s}_h1' for s in [42,43,44]])]
table(['Representation','Phase sin','Left contact','Roll','Phase×roll','Future velocity','Future pitch'],
      [[n]+[fmt(probe(names,t)) for t in targets] for n,names in base+[(label(f)+' h1',probe_names(f,1)) for f in F]])
say('''The clearest learned-feature result is phase-conditioned posture. Raw63input yields R²0.3624 for phase×roll; random63h1 reaches0.4559; original BC h1 reaches0.6847 and original DAgger h1 reaches0.7121. This is a nonlinear interaction that a linear readout cannot recover simply by selecting one raw coordinate. The learned networks make it substantially more linearly accessible.

Conversely, raw observations predict current roll essentially perfectly because roll is directly supplied. Learned h1roll R²around0.96–0.98is not “better information” than raw input; it indicates that much of the observable posture survives the transformation. Phase and contact scores also have high raw/random baselines, so their absolute R²alone is not evidence of a learned gait concept.

### 8.3 Does explicit slip improve accessible future information?

For future forward velocity, raw63R²is0.5951 and raw64with slip is0.6318. Thus slip is informative about oracle motion over the held-out surfaces even without a learned representation. Random64h1also performs well,0.6298, versus random63h1at0.6114.

Varied BC h1improves0.6330(hidden)→0.6409(input); varied DAgger h1improves0.6294→0.6354. These increments are small and should be assessed against the strong raw64/random64controls, not just against hidden-policy features. They show that context-related predictive information can be present without producing a large behavioural improvement. They do not prove that the action head exploits it beneficially.

### 8.4 Layer1 versus layer2
''')
table(['Family','h1phase×roll R²','h2phase×roll R²','h1future velocity R²','h2future velocity R²','h1future pitch R²','h2future pitch R²'],[
[label(f)]+[fmt(probe(probe_names(f,l),t)) for t in ['phase_sin_times_roll_deg','future_200ms_forward_velocity_mps','future_200ms_pitch_deg'] for l in [1,2]] for f in F])
say('''Layer2is not uniformly more decodable. Phase×roll and future velocity generally weaken in the second layer, while future pitch often remains similar or improves slightly. This is consistent with reorganization toward the action output, but the probe does not establish why the reorganization occurs. A lower readout score can reflect inaccessible nonlinear coding, not absence of the underlying information.

DAgger’s markedly better action fidelity is also not accompanied by uniformly higher probe scores on every target. For example, hidden varied DAgger h1future pitch R²0.5876 is below BC’s0.5988, despite much lower action MSE. The probe targets are useful behavioural variables, not a complete list of the computations required for accurate control.

### 8.5 Expert comparison and representation geometry

All1,680student-layer/property comparisons have positive R²gaps relative to the expert hidden layer. It would be incorrect to interpret this as the students knowing more than the expert or transferring more than100%of its knowledge. The teacher has one64-unit tanh hidden layer; students have two256-unit ReLU layers. Width, nonlinearity and task-specific compression strongly affect linear readability.
''')
table(['Family','h1expert CKA','h2expert CKA','h1effective variance dimension','h2effective variance dimension'],[
[label(f),fmt(geom(probe_names(f,1),'expert_cka')),fmt(geom(probe_names(f,2),'expert_cka'))]+
[fmt(np.mean([r['spectrum']['participation_ratio'] for r in P['geometry'] if r['representation'] in probe_names(f,l)]),2) for l in [1,2]] for f in F])
say('''Raw63input already has expert CKA0.8968, and random63h1averages0.8944. Original BC h2has0.9492, whereas the far more faithful original DAgger h2has0.9148. Varied hidden DAgger h2falls further to0.8849 despite its much better action imitation. Therefore a higher whole-layer CKA is **not a ranking of control quality or expert-action fidelity**.

The effective variance dimensions of student layers are roughly5.7–6.7on this oracle-state dataset, despite256available units; expert h1is8.38. Participation ratio is a covariance-spectrum summary of the sampled activations. It is not a count of semantic concepts, proof that the network can be reduced to six neurons, or a general dimension estimate for all possible states.

### 8.6 GAIL and its initializer: near-identical measured representations

The first-layer phase×roll scores of original BC and GAIL-BC both round to0.6847; DAgger and GAIL-DM both round to0.7121. This near-equality persists for many h1/h2properties and CKA values in the varied families. These results are consistent with conservative fine-tuning preserving the supervised representation, and with the small behavioural differences seen earlier. They do not prove identical weights or that no useful change occurred in a narrow state region.

### 8.7 Controls prevent overinterpretation
''')
control_names=['raw_observation','expert_h1','original_hidden_42_dagger_mixed_h1','varied_hidden_42_dagger_mixed_h1','varied_input_42_dagger_mixed_correct_h1']
table(['Representation','Phase true /shuffled R²','Phase×roll true /shuffled R²','Future pitch true /shuffled R²'],[
[n]+[fmt(probe([n],t))+' / '+fmt(probe([n],t,'shuffled')) for t in ['generator_phase_sin','phase_sin_times_roll_deg','future_200ms_pitch_deg']] for n in control_names])
say('''Trajectory shuffling retains substantial periodic phase information:shuffled phase R²around0.5–0.6is not a zero-information null. For phase×roll and future pitch, the shuffled scores are much smaller, which makes those true-label comparisons more informative. Scores should not be called formal selectivity p-values; the null procedure is a project-specific temporal control.

The six future-target persistence baselines all have negative R². For example, carrying the previous forward velocity forward gives−2.6732, and keeping current pitch gives−0.4115. Learned readouts beat these baselines, but raw inputs/random nonlinear features already predict much of this short-horizon periodic behaviour. It is not evidence of an explicit learned world model or long-horizon balance planning.

## 9. Integrating behaviour and representations

1. **Task saturation hides fidelity differences.** Original BC and DAgger both walk at oracle-level progress, while DAgger’s action error is much lower. The progression into varied physics exposes additional stability differences that nominal return alone misses.
2. **Varied continuation improves local imitation and reliability more than speed.** This is supported across fidelity and fall metrics, but diversity and extra training are confounded without a matched nominal-only continuation arm.
3. **DAgger’s main advantage is faithful control, not an across-the-board return win.** Its strong results persist across the varied continuation replicas. The strongest candidate mechanism is better coverage/optimization of learner-relevant states; the current probes are consistent with retained phase/posture information but do not isolate that mechanism.
4. **Explicit context is measurable but not clearly necessary.** Slip improves predictability of future oracle motion in raw features. The learned input policies show small decoder gains, yet behavioural benefits are inconsistent. This is not a contradiction:available information need not be used in a way that improves the task.
5. **GAIL adds cost without a dependable final-policy advantage in this batch.** Its measured features mostly track its initializer, and fidelity sometimes worsens slightly. This conclusion applies to this anchored, conservative GAIL configuration and70krollout budget, not all adversarial imitation.
6. **Neither CKA nor a single probe target explains the ranking.** DAgger improves fidelity while some CKA/readout scores decrease. Representational similarity is an interpretation aid, not a surrogate benchmark.

## 10. Limitations and next evidence needed

- Original policies have one training seed. Varied replicas share parents/data; they are not full independent experimental replications.
- Shared60-second tests and consistent resets improve comparability, but rare-fall estimates remain imprecise. Falls/min excludes recovery time, and target sequences can diverge when arrival timing differs.
- Training exposure differs substantially across BC/DAgger/GAIL despite standardized unique-data budgets. Matched compute and matched extra-nominal-training controls would strengthen causal conclusions.
- Evaluation samples fixed surfaces within(0,10]. It does not test online slip changes, heterogeneous terrain, motor damage, unseen slip beyond10, or a different body.
- New varied policies are not tested in the preserved original scene. Scalar0.1retention is not a substitute for that missing condition.
- Only correct-context evaluation was run. Constant/wrong-context interventions and direct action sensitivity to the context channel would better establish whether the input is used beneficially.
- Probes cover selected properties on oracle states, not every representation or learner-state failure. Future outcomes are oracle-conditioned. Shared test data, multiple targets and random-feature baselines limit broad mechanistic claims.
- This report does not establish R1transfer usefulness. The R1study remains separate/deferred, and there are no causal representation-ablation results for this new generation.

The most valuable follow-up for the DAgger question is a matched-budget continuation comparison:additional nominal-only training versus varied training, with aligned optimizer exposure and independent original seeds. The most valuable context follow-up is wrong/constant-context testing on the same rounds. Both would answer more specific questions than expanding the model list without additional controls.

## Appendix A. Grouped versus confirmation scores and replica variation

Each varied row contains three replicas. Score SD below is the sample standard deviation of the **three replica means over30rounds**, not a confidence interval and not a frame-level statistic. Original models have one seed, so no training-replica SD is supplied.
''')
rows=[]
for f in F:
    rs=select(f);ss=sorted(set(r['controller']['seed'] for r in rs));means=[avg([r for r in rs if r['controller']['seed']==s],'normalized_return') for s in ss]
    rows.append([label(f),fmt(avg(select(f,['grouped']),'normalized_return')),fmt(avg(select(f,['confirmation']),'normalized_return')),
                 fmt(np.std(means,ddof=1),5) if len(ss)>1 else '—',fmt(min(means)),fmt(max(means))])
table(['Family','Grouped score','Confirmation score','Replica-mean SD','Minimum replica mean','Maximum replica mean'],rows)
say('''Grouped scores tend to be slightly above1and confirmation scores slightly below1. This schedule dependence is another reason not to declare a winner from a pooled difference of a few thousandths. The detailed per-replica source report remains the authoritative deployment-level record.

## Appendix B. Complete30-property first-layer probe profile

To keep this table readable, it lists the expert, raw input, matched random controls and the two supervised varied families with and without context. GAIL and original-model scores, both layers and all shuffled controls are preserved in the linked9,000-row source CSV. All numbers here are held-out R²; varied/random values average three replicas.
''')
names=json.loads((PP.parent/'plan.json').read_text())['target_names']
table(['Property','Expert','Raw63','Random63h1','Random64h1','Hidden BC','Input BC','Hidden DAgger','Input DAgger'],[
[t]+[fmt(probe(ns,t)) for ns in [['expert_h1'],['raw_observation'],[f'random_hidden_{s}_h1' for s in [42,43,44]],
 [f'random_input_{s}_h1' for s in [42,43,44]]]]+[fmt(probe(probe_names(('varied',h,a),1),t)) for a in ['bc','dagger_mixed'] for h in ['hidden','input']] for t in names])
say('''## Appendix C. Probe errors in physical units and contact classification

R² is useful for comparing targets of different scales, but RMSE makes prediction
error tangible. The table below uses the first hidden layer and the same held-out
test trajectories. Varied-policy values average the three separately fitted
replicas. Contact balanced accuracy averages sensitivity to positive and negative
force-threshold classes; it is not a calibrated contact probability.
''')
table(['Representation','Current roll RMSE°','Future velocity RMSE m/s','Future pitch RMSE°','Left contact balanced accuracy','Right contact balanced accuracy'],[
 [n]+[fmt(probe(ns,t,key=k)) for t,k in [('torso_roll_deg','rmse'),('future_200ms_forward_velocity_mps','rmse'),
 ('future_200ms_pitch_deg','rmse'),('left_force_contact_gt10N','contact_balanced_accuracy'),('right_force_contact_gt10N','contact_balanced_accuracy')]]
 for n,ns in [('Raw63',['raw_observation']),('Raw64+slip',['raw_observation_with_context']),('Expert h1',['expert_h1'])]+
 [(label(f)+' h1',probe_names(f,1)) for f in F]])
say('''## Appendix D. Source reports, provenance and reproducibility

- [Full evaluation JSON](../experiments/consolidated/v1/evaluation/all_all_correct/comparison.json): 1,020 run records, per-replica aggregates and paired context differences.
- [Evaluation CSV](../experiments/consolidated/v1/evaluation/all_all_correct/comparison.csv) and [readable source table](../experiments/consolidated/v1/evaluation/all_all_correct/comparison.md).
- [Training summary](../experiments/consolidated/v1/training_summary.json) and [training plan](../experiments/consolidated/v1/plan.json): budgets, nested initializer lineage and per-update diagnostics.
- [Probe JSON](../experiments/representation_probing/consolidated_v1/report.json), [all probe scores](../experiments/representation_probing/consolidated_v1/probe_scores.csv), [expert comparisons](../experiments/representation_probing/consolidated_v1/expert_comparison.csv) and [probe plan](../experiments/representation_probing/consolidated_v1/plan.json).
- [Training/evaluation protocol](../consolidated/README.md), [new probe protocol](../consolidated_probing/README.md) and [original target definitions](../representation_probing/README.md).
- [Historical meeting summary](MEETING_SUMMARY_2026-10-10.md) and [historical Phase 1 report](PHASE1_RESULTS_REPORT.md) remain unchanged. They concern earlier batches and should be cited as such.

Numerical tables in this report are generated from the saved JSON using [the read-only report builder](analysis/consolidated_summary.py). The builder prints Markdown; it does not retrain policies, recompute simulator traces or edit source artifacts. Source hashes below identify the exact input reports used. Probe methodology references remain documented in the linked protocol: Alain & Bengio (linear probes), Hewitt & Liang (control tasks), Kornblith et al. (CKA). The present implementation adapts linear decoding to continuous locomotion regression; it is not a reproduction of their complete experiments.
''')
table(['Source','SHA-256'],[[str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()] for p in [EP,PP,TP]])
OUT.append((ROOT/'evaluation/analysis/consolidated_references.md').read_text().strip())
from scratch_summary import build
OUT.append('\n'+build(ROOT).strip())
print('\n'.join(OUT))

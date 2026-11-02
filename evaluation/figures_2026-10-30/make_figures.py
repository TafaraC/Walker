"""Presentation-ready figures for the extended-fitting (convergence) batch.

Read-only. Uses only files tracked in the repository:
  evaluation/convergence_2026-10-30/{deployments,training_summary}.csv
  experiments/submission_controls/v1/curves/curves.json
  evaluation/figures_2026-10-30/damping_pairs.csv      (made by --damping-traces, see README)

Requires NumPy, pandas and Matplotlib.
Usage (from the repository root):
  python3 evaluation/figures_2026-10-30/make_figures.py
  python3 evaluation/figures_2026-10-30/make_figures.py --damping-traces <trace_runs.csv>   # rebuild damping table first
"""
import argparse, json, os, re
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/walker-fig-1030')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONV = ROOT / 'evaluation/convergence_2026-10-30'
OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)

BLUE, ORANGE, PURPLE, GREY, RUST = '#2878B5', '#C2700A', '#6B4C9A', '#7A8790', '#B4463C'
INK, MUTED = '#1E2B33', '#5B6B75'
plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK, 'axes.edgecolor': '#B8C2C8',
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
BOX = dict(vert=False, patch_artist=True, widths=.58, showmeans=True,
           meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor=INK, markersize=4.5),
           medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED, lw=1.1), capprops=dict(color=MUTED, lw=1.1),
           flierprops=dict(marker='o', markersize=3.2, markerfacecolor='none', markeredgecolor=MUTED, alpha=.8))


def boxes(ax, data, colors, pos):
    bp = ax.boxplot(data, positions=pos, manage_ticks=False, **BOX)
    for b, c in zip(bp['boxes'], colors):
        b.set_facecolor(c); b.set_alpha(.5); b.set_edgecolor(c); b.set_linewidth(1.2)


def grid(ax, axis='x'):
    ax.grid(axis=axis, alpha=.25); ax.set_axisbelow(True)


def save(fig, name, title, note):
    fig.suptitle(title, x=.01, ha='left', fontweight='bold', fontsize=12.5)
    fig.text(.01, .01, note, ha='left', va='bottom', fontsize=8.8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200)
    plt.close(fig)


D = pd.read_csv(CONV / 'deployments.csv')
S = D[D.suite.isin(['grouped', 'confirmation'])]
TR = pd.read_csv(CONV / 'training_summary.csv')
NUM = {}


def heldout(label):
    v = D[D.label == label].held_out_expert_action_mse.dropna()
    return float(v.iloc[0])


def per_round(label):
    return S[S.label == label].set_index('round').sort_index()

# ---------------------------------------------------------------- Figure 1: original-scene allocations
curves = json.loads((ROOT / 'experiments/submission_controls/v1/curves/curves.json').read_text())['rows']
dag = [r for r in curves if r['algorithm'] == 'dagger_mixed']
ORIG = [('BC · 100 epochs', 'original_hidden_42_bc', BLUE, 66500),
        ('BC · matched updates', 'bc_original_matched', BLUE, 211050),
        ('BC · validation-stopped', 'bc_original_convergence', BLUE, int(TR.set_index('label').loc['bc_original_convergence', 'total_lineage_gradient_updates'])),
        ('DAgger · 7 rounds', 'original_hidden_42_dagger_mixed', ORANGE, 211050),
        ('DAgger · validation-stopped', 'dagger_original_fixed_aggregate_convergence', ORANGE,
         int(TR.set_index('label').loc['dagger_original_fixed_aggregate_convergence', 'total_lineage_gradient_updates']))]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax = axs[0]
x = [r['cumulative_supervised_updates'] / 1e3 for r in dag]; y = [r['test_mse'] for r in dag]
ax.plot(x, y, '-o', color=ORANGE, lw=2, ms=4.5, markeredgecolor='white', label='DAgger (aggregation rounds)')
xe, ye = ORIG[4][3] / 1e3, heldout(ORIG[4][1])
ax.plot([x[-1], xe], [y[-1], ye], '--', color=ORANGE, lw=1.6)
ax.plot(xe, ye, 'o', color=ORANGE, ms=8, markeredgecolor='white', mew=1.2)
bx = [o[3] / 1e3 for o in ORIG[:3]]; by = [heldout(o[1]) for o in ORIG[:3]]
ax.plot(bx, by, '--', color=BLUE, lw=1.6); ax.plot(bx, by, 'o', color=BLUE, ms=8, markeredgecolor='white', mew=1.2, label='BC (same 170k examples)')
for (lab, l, c, u), yy in zip(ORIG, [*by, y[-1], ye]):
    txt = {'original_hidden_42_bc': '100 epochs', 'bc_original_matched': 'matched\nupdates', 'bc_original_convergence': 'validation-\nstopped',
           'original_hidden_42_dagger_mixed': 'round 7', 'dagger_original_fixed_aggregate_convergence': 'validation-\nstopped'}[l]
    off = {'original_hidden_42_bc': (8, 4), 'bc_original_matched': (-58, -14), 'bc_original_convergence': (-30, -30),
           'original_hidden_42_dagger_mixed': (8, 6), 'dagger_original_fixed_aggregate_convergence': (-20, 10)}[l]
    ax.annotate(f'{txt}\n{yy:.4f}', (u / 1e3, yy), xytext=off, textcoords='offset points', fontsize=8.3, color=c)
ax.axvline(211.05, color=MUTED, ls=':', lw=1); ax.text(213, .2, '211,050\nupdates', fontsize=8, color=MUTED, va='top')
ax.set_yscale('log'); ax.set_xlabel('Optimizer updates performed (thousands)'); ax.set_ylabel('Held-out expert-action MSE (log)')
ax.set_title('The ranking flips twice as fitting continues'); ax.legend(frameon=False, fontsize=9, loc='upper right'); grid(ax, 'both')
ax = axs[1]
oracle = per_round('oracle').return_target_progress_m
data, cols, labs = [], [], []
for lab, l, c, u in ORIG:
    r = per_round(l)
    data.append((r.return_target_progress_m - oracle.loc[r.index]).values); cols.append(c)
    labs.append(f'{lab}   ({int(r.falls.sum())} falls)')
pos = np.arange(len(ORIG))[::-1]
boxes(ax, data, cols, pos); ax.set_yticks(pos, labs); ax.axvline(0, color=INK, ls='--', lw=1)
ax.axhline(1.5, color='#D5DCE0', lw=.8)
ax.set_xlabel('Progress minus oracle on the same round (m per 60 s)'); ax.set_title('…while sampled-surface progress barely moves'); grid(ax)
NUM['fig1'] = {lab: dict(heldout=heldout(l), updates=u, gap_mean=float(np.mean(dd)), gap_sd=float(np.std(dd, ddof=1)))
               for (lab, l, c, u), dd in zip(ORIG, data)}
save(fig, '01_original_allocation',
     'Original scene, one training lineage: imitation error depends on how much fitting each method receives',
     'Left: held-out MSE on the 16 native gait actions. DAgger points are the saved aggregation rounds; dashed segments are further fitting on the '
     'fixed final data. "Validation-stopped": improvement below 1e-5 for 25 epochs on the shared validation pool, not proven convergence. '
     'Right: 30 sampled surface/task rounds, box = IQR, ◇ = mean, whiskers = 1.5 × IQR; falls over 30 active minutes.')

# ---------------------------------------------------------------- Figure 2: varied families
def fam(label):
    m = re.match(r'(bc|dagger_mixed)_varied_(continued|scratch)_(hidden|input)_(\d+)_convergence', label)
    if not m:
        return None
    alg, hist, arch, seed = m.groups()
    suf = '_correct' if arch == 'input' else ''
    if hist == 'continued':
        init = f'varied_{arch}_{seed}_{alg}{suf}'
    elif alg == 'bc':
        init = f'bc_varied_{arch}_{seed}_200'
    else:
        init = f'scratch_varied_{arch}_{seed}_dagger_mixed{suf}'
    return dict(ext=label, init=init, alg='BC' if alg == 'bc' else 'DAgger', history='Continued' if hist == 'continued' else 'Varied only',
                arch=arch, seed=int(seed))


F = pd.DataFrame([f for f in map(fam, D.label.unique()) if f])
F['mse_init'] = F.init.map(heldout); F['mse_ext'] = F.ext.map(heldout)
FAM = [(h, a, g) for h in ('Continued', 'Varied only') for a in ('hidden', 'input') for g in ('BC', 'DAgger')]
fig, axs = plt.subplots(1, 2, figsize=(14.5, 6), gridspec_kw=dict(width_ratios=[1, 1.1]), sharey=True)
pos = np.arange(len(FAM))[::-1]
names, gaps, cols, rows = [], [], [], []
for p, (h, a, g) in zip(pos, FAM):
    f = F[(F['history'] == h) & (F.arch == a) & (F.alg == g)]
    c = BLUE if g == 'BC' else ORANGE
    for col, mk, face in (('mse_init', 'o', 'white'), ('mse_ext', 'o', c)):
        axs[0].plot([f[col].min(), f[col].max()], [p, p], color=c, lw=5, alpha=.25, solid_capstyle='round')
    axs[0].annotate('', xy=(f.mse_ext.mean(), p), xytext=(f.mse_init.mean(), p),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.6, shrinkA=6, shrinkB=6))
    axs[0].plot(f.mse_init.mean(), p, 'o', ms=8, markerfacecolor='white', markeredgecolor=c, mew=1.8)
    axs[0].plot(f.mse_ext.mean(), p, 'o', ms=8, color=c, markeredgecolor='white')
    axs[0].text(f.mse_init.mean() * 1.12, p + .28, f'{f.mse_init.mean():.4f}', fontsize=8, color=MUTED)
    axs[0].text(f.mse_ext.mean() / 1.12, p + .28, f'{f.mse_ext.mean():.4f}', fontsize=8, color=INK, ha='right')
    # paired progress change: per round, averaged over the 3 seeds
    d = []
    fi = fe = 0
    for _, r in f.iterrows():
        e, i = per_round(r.ext), per_round(r.init)
        d.append((e.return_target_progress_m - i.return_target_progress_m.loc[e.index]).rename(r.seed))
        fi += int(i.falls.sum()); fe += int(e.falls.sum())
    dd = pd.concat(d, axis=1).mean(1)
    gaps.append(dd.values); cols.append(c)
    names.append(f'{h} · {a} · {g}')
    rows.append(dict(family=names[-1], mse_init=f.mse_init.mean(), mse_ext=f.mse_ext.mean(), progress_change_mean=dd.mean(),
                     progress_change_sd=dd.std(), falls_init=fi, falls_ext=fe))
    axs[1].text(1.02, p, f'falls {fi} → {fe}', va='center', fontsize=8.5, color=INK)
axs[0].set_xscale('log'); axs[0].set_yticks(pos, names)
axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axs[0].set_xticks([.002, .005, .01, .02, .05], ['0.002', '0.005', '0.01', '0.02', '0.05'])
axs[0].set_xlabel('Held-out expert-action MSE (log): ○ initializer → ● validation-stopped')
axs[0].set_title('Fidelity gaps collapse with further fitting'); grid(axs[0])
for y in (3.5,):
    for a in axs:
        a.axhline(y, color='#D5DCE0', lw=.8)
boxes(axs[1], gaps, cols, pos); axs[1].axvline(0, color=INK, ls='--', lw=1)
axs[1].set_xlim(-1.6, 1.45); axs[1].set_xlabel('Progress change, validation-stopped minus initializer (m per 60 s)')
axs[1].set_title('Task progress hardly changes'); grid(axs[1])
axs[0].legend(handles=[Line2D([], [], color=BLUE, lw=5, alpha=.35, label='range over 3 training seeds')], frameon=False, fontsize=8.5, loc='lower left')
pd.DataFrame(rows).to_csv(HERE / 'varied_family_changes.csv', index=False)
NUM['fig2'] = rows
save(fig, '02_varied_families',
     'Varied surfaces: further fitting removes most of the BC–DAgger fidelity gap without improving progress',
     'Each row: 3 training seeds. Initializers: continued = previous varied models; varied-only BC = 200-epoch control; varied-only DAgger = round-7 '
     'scratch model. Right: one value per sampled round (30), the seed-averaged paired change; box = IQR, ◇ = mean. Falls summed over the '
     '3 seeds × 30 runs (90 active minutes). Re-running identical weights gave per-round SDs of 0.78–0.88 m, so family-mean changes smaller than ~0.2 m are within run-to-run noise.')

# ---------------------------------------------------------------- Figure 3: engine control
ENG = [('Oracle (FC Portugal)', 'oracle', GREY), ('BC · validation-stopped', 'bc_original_convergence', BLUE),
       ('Zero native action', 'zero_native', PURPLE), ('Untrained network · seed 42', 'engine_random_network_42', PURPLE),
       ('Untrained network · seed 43', 'engine_random_network_43', PURPLE), ('Untrained network · seed 44', 'engine_random_network_44', PURPLE),
       ('Uniform random actions', 'random', RUST)]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]), sharey=True)
pos = np.arange(len(ENG))[::-1]
boxes(axs[0], [per_round(l).return_target_progress_m.values for _, l, _ in ENG], [c for *_, c in ENG], pos)
axs[0].set_yticks(pos, [n for n, *_ in ENG]); axs[0].axvline(0, color=INK, lw=.8)
axs[0].set_xlabel('Target progress per 60 s run (m), 30 sampled rounds'); axs[0].set_title('Only trained policies walk'); grid(axs[0])
fl = [int(per_round(l).falls.sum()) for _, l, _ in ENG]
axs[1].barh(pos, fl, color=[c for *_, c in ENG], alpha=.55, height=.55)
for p, v in zip(pos, fl):
    axs[1].text(v * 1.15 + .3, p, str(v), va='center', fontsize=9)
axs[1].set_xscale('symlog', linthresh=10); axs[1].set_xlim(0, 2500)
axs[1].set_xlabel('Falls over 30 active minutes (symlog)'); axs[1].set_title('Few falls ≠ good walking'); grid(axs[1])
NUM['fig3'] = {n: dict(progress_mean=float(per_round(l).return_target_progress_m.mean()), falls=f) for (n, l, _), f in zip(ENG, fl)}
save(fig, '03_engine_control',
     'The gait engine alone does not walk: untrained networks and zero actions make no useful progress',
     'Untrained networks: the standard 63→256→256→16 architecture with seeded random weights, real observations and the unchanged '
     'gait/IK/reset machinery. Uniform random: independent actions within the expert range. Low fall counts for the inactive controls reflect '
     'little movement, not stable walking.')

# ---------------------------------------------------------------- Figure 4: damping (supplementary)
ap = argparse.ArgumentParser(); ap.add_argument('--damping-traces'); args = ap.parse_args()
if args.damping_traces:
    T = pd.read_csv(args.damping_traces)
    T = T[~T['round'].str.startswith('original')]
    T['sd_ratio'] = np.mean([T[f'std_a{k}'] / T[f'std_o{k}'] for k in range(16)], 0)
    rows = []
    for _, f in F.iterrows():
        e, i = per_round(f.ext), per_round(f.init)
        rows.append(dict(ext=f.ext, init=f.init, alg=f.alg, history=f['history'], arch=f.arch, seed=f.seed,
                         sd_ratio_init=T[T.label == f.init].sd_ratio.median(), sd_ratio_ext=T[T.label == f.ext].sd_ratio.median(),
                         progress_change=float((e.return_target_progress_m - i.return_target_progress_m.loc[e.index]).mean())))
    pd.DataFrame(rows).to_csv(HERE / 'damping_pairs.csv', index=False)
P = pd.read_csv(HERE / 'damping_pairs.csv')
P['d_sd'] = P.sd_ratio_ext - P.sd_ratio_init
rho = float(P.d_sd.rank().corr(P.progress_change.rank()))
_rng = np.random.default_rng(0); _r = P.progress_change.rank().values
pv = float((np.abs([np.corrcoef(P.d_sd.rank().values, _rng.permutation(_r))[0, 1] for _ in range(20000)]) >= abs(rho)).mean())  # permutation p
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
for g, c in (('BC', BLUE), ('DAgger', ORANGE)):
    q = P[P.alg == g]
    axs[0].scatter(q.sd_ratio_init, q.sd_ratio_ext, s=40, color=c, alpha=.8, edgecolor='white', label=g)
    axs[1].scatter(q.d_sd * 100, q.progress_change, s=40, color=c, alpha=.8, edgecolor='white', label=g)
lim = [min(P.sd_ratio_init.min(), P.sd_ratio_ext.min()) - .002, 1.004]
axs[0].plot(lim, lim, color=MUTED, lw=.8, ls=':'); axs[0].axhline(1, color=MUTED, lw=.6); axs[0].axvline(1, color=MUTED, lw=.6)
axs[0].set_xlabel('Initializer: action SD ÷ oracle SD'); axs[0].set_ylabel('Validation-stopped: action SD ÷ oracle SD')
axs[0].set_title('Further fitting removes the small damping'); axs[0].legend(frameon=False)
axs[1].axhline(0, color=INK, lw=.8, ls='--'); axs[1].axvline(0, color=MUTED, lw=.6)
axs[1].set_xlabel('Change in SD ratio (percentage points)'); axs[1].set_ylabel('Paired progress change (m per 60 s)')
axs[1].set_title(f'…but that does not predict the progress change (ρ = {rho:.2f})')
for a in axs:
    grid(a, 'both')
NUM['fig4'] = dict(spearman=float(rho), p=float(pv), n=len(P))
save(fig, '04_damping_check',
     'Supplementary: does losing action damping explain the small progress losses? Not in these models',
     'One point per varied extended/initializer pair (24), sampled suite, per-step traces. Spearman p = ' + f"{pv:.2f}" + '. SD ratio = mean over the 16 actions of SD(learner) ÷ '
     'SD(oracle label at the same states), median over 30 runs. These initializers were already close to 1, so the test has limited range.')

(HERE / 'figure_numbers.json').write_text(json.dumps(NUM, indent=1, default=float))
print('figures:', sorted(p.name for p in OUT.glob('*.png')))

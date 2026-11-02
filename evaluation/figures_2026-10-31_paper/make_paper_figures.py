"""Paper versions of Figures 1-6 (IEEE two-column; Figures 1, 2 and 6 full text width, 3-5 one column).

Derived from evaluation/figures_2026-10-30/make_figures.py on 31 October 2026. Same data and computations;
changes are presentation only: drawn at print size (7.16 in wide), no suptitles or embedded footnotes (the LaTeX
captions carry them), panel letters, neutral panel titles, the paper's terms (expert, original field, altered
surfaces, zero action), and a dot plot instead of symlog bars for fall counts.

Figures 4-6 are extracted from the archived analysis scripts (kept unchanged in OneDrive Research/Walker/archive):
  Fig 4 <- walker_eda.py 'Fig 7: stability' (07_stability); Fig 6 <- walker_eda.py 'Fig 3: slip dependence'
  (03_slip_dependence); Fig 5 <- eda_sweep2/walker_eda_sweep2.py '1. GAIL history' (01_gail_before_after_fixes).
  Their data loading and calculations are copied as they were; only the drawing changed: print size, neutral titles,
  no embedded footnotes, counts instead of Wilson intervals, and Fig 5 reduced to three panels.

Usage (from the repository root):
  python3 evaluation/figures_2026-10-31_paper/make_paper_figures.py --out <folder> [--legacy <artifacts root>]
  (--legacy defaults to <repo>/artifacts, the Git-ignored legacy run logs used by Fig 5)
"""
import argparse, json, os, re
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/walker-fig-1031')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONV = ROOT / 'evaluation/convergence_2026-10-30'
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(HERE / 'figures')); ap.add_argument('--legacy', default=None)
ARGS = ap.parse_args()
OUT = Path(ARGS.out); OUT.mkdir(parents=True, exist_ok=True)
LEG = Path(ARGS.legacy) if ARGS.legacy else ROOT / 'artifacts'

W = 7.16  # IEEE \textwidth in inches
WC = 3.5  # IEEE \columnwidth in inches (Figures 3-5, 1 November 2026 page-limit revision)
BLUE, ORANGE, PURPLE, GREY, RUST = '#2878B5', '#C2700A', '#6B4C9A', '#7A8790', '#B4463C'
INK, MUTED = '#1E2B33', '#5B6B75'
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK, 'axes.edgecolor': '#B8C2C8',
                     'axes.linewidth': .7, 'xtick.major.width': .6, 'ytick.major.width': .6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
BOX = dict(vert=False, patch_artist=True, widths=.58, showmeans=True,
           meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor=INK, markersize=3, markeredgewidth=.8),
           medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(color=MUTED, lw=.8), capprops=dict(color=MUTED, lw=.8),
           flierprops=dict(marker='o', markersize=2.2, markerfacecolor='none', markeredgecolor=MUTED, markeredgewidth=.6, alpha=.8))


def boxes(ax, data, colors, pos):
    bp = ax.boxplot(data, positions=pos, manage_ticks=False, **BOX)
    for b, c in zip(bp['boxes'], colors):
        b.set_facecolor(c); b.set_alpha(.5); b.set_edgecolor(c); b.set_linewidth(.9)


def grid(ax, axis='x'):
    ax.grid(axis=axis, alpha=.25, lw=.5); ax.set_axisbelow(True)


def save(fig, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(pad=.3, w_pad=1.2, rect=rect)
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=220)
    plt.close(fig)


D = pd.read_csv(CONV / 'deployments.csv')
S = D[D.suite.isin(['grouped', 'confirmation'])]
TR = pd.read_csv(CONV / 'training_summary.csv')
NUM = {}


def heldout(label):
    return float(D[D.label == label].held_out_expert_action_mse.dropna().iloc[0])


def per_round(label):
    return S[S.label == label].set_index('round').sort_index()

# ---------------------------------------------------------------- Figure 1
curves = json.loads((ROOT / 'experiments/submission_controls/v1/curves/curves.json').read_text())['rows']
dag = [r for r in curves if r['algorithm'] == 'dagger_mixed']
upd = TR.set_index('label').total_lineage_gradient_updates
ORIG = [('BC, initial (100 epochs)', 'original_hidden_42_bc', BLUE, 66500),
        ('BC, matched updates', 'bc_original_matched', BLUE, 211050),
        ('BC, validation-stopped', 'bc_original_convergence', BLUE, int(upd['bc_original_convergence'])),
        ('DAgger, 7 rounds', 'original_hidden_42_dagger_mixed', ORANGE, 211050),
        ('DAgger, validation-stopped', 'dagger_original_fixed_aggregate_convergence', ORANGE,
         int(upd['dagger_original_fixed_aggregate_convergence']))]
fig, axs = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw=dict(width_ratios=[1, 1.05]))
ax = axs[0]
x = [r['cumulative_supervised_updates'] / 1e3 for r in dag]; y = [r['test_mse'] for r in dag]
ax.plot(x, y, '-o', color=ORANGE, lw=1.3, ms=2.8, markeredgecolor='white', mew=.5, label='DAgger (aggregation rounds)')
xe, ye = ORIG[4][3] / 1e3, heldout(ORIG[4][1])
ax.plot([x[-1], xe], [y[-1], ye], '--', color=ORANGE, lw=1)
ax.plot(xe, ye, 'o', color=ORANGE, ms=5, markeredgecolor='white', mew=.8)
bx = [o[3] / 1e3 for o in ORIG[:3]]; by = [heldout(o[1]) for o in ORIG[:3]]
ax.plot(bx, by, '--', color=BLUE, lw=1)
ax.plot(bx, by, 'o', color=BLUE, ms=5, markeredgecolor='white', mew=.8, label='BC (same 170k examples)')
LBL = {'original_hidden_42_bc': ('initial', (5, 3)), 'bc_original_matched': ('matched\nupdates', (-38, -10)),
       'bc_original_convergence': ('validation-\nstopped', (5, -19)), 'original_hidden_42_dagger_mixed': ('round 7', (5, 4)),
       'dagger_original_fixed_aggregate_convergence': ('validation-\nstopped', (-14, 7))}
for (lab, l, c, u), yy in zip(ORIG, [*by, y[-1], ye]):
    txt, off = LBL[l]
    ax.annotate(f'{txt}\n{yy:.4f}', (u / 1e3, yy), xytext=off, textcoords='offset points', fontsize=6.2, color=c, linespacing=1.05)
ax.axvline(211.05, color=MUTED, ls=':', lw=.7); ax.text(216, 2.9e-3, '211,050 updates', fontsize=6, color=MUTED, va='bottom')
ax.set_yscale('log'); ax.set_ylim(2.6e-3, .3); ax.set_xlim(0, 540)
ax.set_xlabel('Optimizer updates performed (thousands)'); ax.set_ylabel('Held-out action MSE (log)')
ax.set_title('(a) Held-out action error'); ax.legend(frameon=False, loc='upper right', handlelength=1.6); grid(ax, 'both')
ax = axs[1]
expert = per_round('oracle').return_target_progress_m
data, cols, labs = [], [], []
for lab, l, c, u in ORIG:
    r = per_round(l)
    data.append((r.return_target_progress_m - expert.loc[r.index]).values); cols.append(c)
    labs.append(f'{lab}\n{int(r.falls.sum())} falls')
pos = np.arange(len(ORIG))[::-1]
boxes(ax, data, cols, pos); ax.set_yticks(pos, labs); ax.axvline(0, color=INK, ls='--', lw=.7)
ax.axhline(1.5, color='#D5DCE0', lw=.6)
ax.set_xlabel('Progress minus expert, same round (m per 60 s)'); ax.set_title('(b) Progress relative to the expert'); grid(ax)
NUM['fig1'] = {lab: dict(heldout=heldout(l), updates=u, gap_mean=float(np.mean(dd))) for (lab, l, c, u), dd in zip(ORIG, data)}
save(fig, 'fig1_allocation')

# ---------------------------------------------------------------- Figure 2
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
    return dict(ext=label, init=init, alg='BC' if alg == 'bc' else 'DAgger', history='Continued' if hist == 'continued' else 'Varied-only',
                arch=arch, seed=int(seed))


F = pd.DataFrame([f for f in map(fam, D.label.unique()) if f])
F['mse_init'] = F.init.map(heldout); F['mse_ext'] = F.ext.map(heldout)
FAM = [(h, a, g) for h in ('Continued', 'Varied-only') for a in ('hidden', 'input') for g in ('BC', 'DAgger')]
fig, axs = plt.subplots(1, 2, figsize=(W, 3.15), gridspec_kw=dict(width_ratios=[1, 1.08]), sharey=True)
pos = np.arange(len(FAM))[::-1]
names, gaps, cols = [], [], []
for p, (h, a, g) in zip(pos, FAM):
    f = F[(F['history'] == h) & (F.arch == a) & (F.alg == g)]
    c = BLUE if g == 'BC' else ORANGE
    for col in ('mse_init', 'mse_ext'):
        axs[0].plot([f[col].min(), f[col].max()], [p, p], color=c, lw=3.5, alpha=.25, solid_capstyle='round')
    axs[0].annotate('', xy=(f.mse_ext.mean(), p), xytext=(f.mse_init.mean(), p),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1, shrinkA=4, shrinkB=4))
    axs[0].plot(f.mse_init.mean(), p, 'o', ms=4.5, markerfacecolor='white', markeredgecolor=c, mew=1.1)
    axs[0].plot(f.mse_ext.mean(), p, 'o', ms=4.5, color=c, markeredgecolor='white', mew=.6)
    far = f.mse_init.mean() / f.mse_ext.mean() > 1.6
    if far:
        axs[0].text(f.mse_init.mean() * 1.1, p + .3, f'{f.mse_init.mean():.4f}', fontsize=5.8, color=MUTED)
        axs[0].text(f.mse_ext.mean() / 1.1, p + .3, f'{f.mse_ext.mean():.4f}', fontsize=5.8, color=INK, ha='right')
    else:
        axs[0].text(max(f.mse_init.mean(), f.mse_ext.mean()) * 1.45, p - .05, f'{f.mse_init.mean():.4f} → {f.mse_ext.mean():.4f}',
                    fontsize=5.8, color=INK, va='center')
    d, fi, fe = [], 0, 0
    for _, r in f.iterrows():
        e, i = per_round(r.ext), per_round(r.init)
        d.append((e.return_target_progress_m - i.return_target_progress_m.loc[e.index]).rename(r.seed))
        fi += int(i.falls.sum()); fe += int(e.falls.sum())
    gaps.append(pd.concat(d, axis=1).mean(1).values); cols.append(c)
    names.append(f'{h}, {a}, {g}')
    axs[1].text(1.03, p, f'falls {fi}→{fe}', va='center', fontsize=6.2, color=INK, transform=axs[1].get_yaxis_transform())
axs[0].set_xscale('log'); axs[0].set_yticks(pos, names); axs[0].set_xlim(1.3e-3, .07)
axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axs[0].set_xticks([.002, .005, .01, .02, .05], ['0.002', '0.005', '0.01', '0.02', '0.05'])
axs[0].set_xlabel('Held-out action MSE (log): ○ before → ● after')
axs[0].set_title('(a) Held-out action error'); grid(axs[0])
for a in axs:
    a.axhline(3.5, color='#D5DCE0', lw=.6)
boxes(axs[1], gaps, cols, pos); axs[1].axvline(0, color=INK, ls='--', lw=.7)
axs[1].set_xlim(-1.7, 1.45); axs[1].set_xlabel('Progress change, after minus before (m per 60 s)')
axs[1].set_title('(b) Paired progress change'); grid(axs[1])
# seed-range bars are explained in the LaTeX caption (no legend: it collides with value labels at print size)
save(fig, 'fig2_varied')

# ---------------------------------------------------------------- Figure 3
ENG = [('Expert (FC Portugal)', 'oracle', GREY), ('BC, validation-stopped', 'bc_original_convergence', BLUE),
       ('Zero action', 'zero_native', PURPLE), ('Untrained network, seed 42', 'engine_random_network_42', PURPLE),
       ('Untrained network, seed 43', 'engine_random_network_43', PURPLE), ('Untrained network, seed 44', 'engine_random_network_44', PURPLE),
       ('Uniform random actions', 'random', RUST)]
fig, ax = plt.subplots(figsize=(WC, 2.35))
pos = np.arange(len(ENG))[::-1]
boxes(ax, [per_round(l).return_target_progress_m.values for _, l, _ in ENG], [c for *_, c in ENG], pos)
ax.set_yticks(pos, ['Expert (FC Portugal)', 'BC, val.-stopped', 'Zero action', 'Untrained net, seed 42',
                    'Untrained net, seed 43', 'Untrained net, seed 44', 'Uniform random'])
ax.axvline(0, color=INK, lw=.6); ax.set_xlim(-5, 52)
ax.set_xlabel('Progress per 60 s (m)'); grid(ax)
fl = [int(per_round(l).falls.sum()) for _, l, _ in ENG]
for p, v in zip(pos, fl):   # total falls in 30 active minutes, printed at the right edge
    ax.text(51.5, p, f'{v} falls' if v != 1 else '1 fall', va='center', ha='right', fontsize=6.2, color=INK)
save(fig, 'fig3_engine')
NUM['fig3'] = {n: dict(progress_mean=float(per_round(l).return_target_progress_m.mean()), falls=f) for (n, l, _), f in zip(ENG, fl)}

# ================================================================ Figures 4 and 6: data from archive/walker_eda.py
SRC = {'scratch': ROOT / 'experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json',
       'controls': ROOT / 'experiments/submission_controls/v1/evaluation/summary/comparison.json'}
ED = {k: json.loads(p.read_text()) for k, p in SRC.items()}
METRICS = ['held_out_expert_action_mse', 'return_target_progress_m', 'mean_forward_velocity_mps',
           'mean_target_direction_error', 'targets_reached', 'falls', 'on_policy_latent_action_mse',
           'normalized_return', 'joint_occupancy_sliced_wasserstein', 'jerk_cost_rate_rad2_per_s6',
           'torso_roll_rms_deg', 'torso_pitch_rms_deg']
ALGN = {'bc': 'BC', 'dagger_mixed': 'DAgger', 'gail_bc': 'GAIL-BC',
        'gail_dagger_mixed': 'GAIL-DM', 'gail_random': 'GAIL-random'}


def parse(label):
    m = re.match(r'^(scratch_)?(original|varied)_(hidden|input)_(\d+)_'
                 r'(bc|dagger_mixed|gail_bc|gail_dagger_mixed|gail_random)(_correct)?$', label)
    if m:
        scratch, dom, arch, seed, alg, _ = m.groups()
        hist = {'original': 'Original only', 'varied': 'Varied only' if scratch else 'Continued'}[dom]
        if scratch and dom == 'original':
            hist = 'Original only'
        return dict(group='trained', history=hist, arch=arch, seed=int(seed), alg=ALGN[alg], budget='initial')
    m = re.match(r'^bc_varied_(hidden|input)_(\d+)_200$', label)
    if m:
        return dict(group='trained', history='Varied only', arch=m[1], seed=int(m[2]), alg='BC', budget='200 ep')
    if label == 'bc_original_matched':
        return dict(group='trained', history='Original only', arch='hidden', seed=42, alg='BC', budget='matched')
    return dict(group='baseline', history='-', arch='-', seed=np.nan, alg=label, budget='-')


rows, seen = [], set()
for src in ('scratch', 'controls'):          # controls re-use some scratch runs: count each once
    for r in ED[src]['runs']:
        key = (r['row']['id'], r['controller']['label'])
        if key in seen:
            continue
        seen.add(key)
        d = dict(round=r['row']['id'], suite=r['row']['suite'], slip=r['row']['slip'], label=key[1])
        d.update(parse(key[1])); d.update({m: r.get(m) for m in METRICS})
        rows.append(d)
runs = pd.DataFrame(rows)
runs['pool'] = runs.suite.map({'grouped': 'sampled', 'confirmation': 'sampled', 'original': 'original', 'nominal': 'nominal'})
runs['slip_level'] = runs['round'].str.extract(r'group_(\d)')[0].astype(float).map({0: '0.025', 1: '0.1', 2: '0.2', 3: '1', 4: '10'})
runs['jerk_M'] = runs.jerk_cost_rate_rad2_per_s6 / 1e6
assert len(runs) == 2223, len(runs)
orc = runs[runs.label == 'oracle'].set_index('round').return_target_progress_m
runs['gap'] = runs.return_target_progress_m - runs['round'].map(orc)
ES = runs[runs.pool == 'sampled']
ET = ES[(ES.group == 'trained') & (ES.alg != 'GAIL-random')]
ALG_C = {'BC': BLUE, 'DAgger': ORANGE}
MAIN = [('Original only', 'BC', 'initial', 'Original, BC, 100 epochs'),
        ('Original only', 'BC', 'matched', 'Original, BC, matched updates'),
        ('Original only', 'DAgger', 'initial', 'Original, DAgger, 7 rounds'),
        ('Continued', 'BC', 'initial', 'Continued, BC'),
        ('Continued', 'DAgger', 'initial', 'Continued, DAgger'),
        ('Varied only', 'BC', 'initial', 'Varied-only, BC, 100 epochs'),
        ('Varied only', 'BC', '200 ep', 'Varied-only, BC, 200 epochs'),
        ('Varied only', 'DAgger', 'initial', 'Varied-only, DAgger')]


def cohort(df, hist, alg, budget, arch='hidden'):
    return df[(df.history == hist) & (df.alg == alg) & (df.budget == budget) & (df.arch == arch)]

# ---------------------------------------------------------------- Figure 4: stability (walker_eda.py Fig 7)
ent = [('Expert', ES[ES.label == 'oracle'], GREY, 'E')] + [(l, cohort(ET, h, a, b), ALG_C[a], h) for h, a, b, l in MAIN]
fig, ax = plt.subplots(figsize=(WC, 2.55))
pos = np.arange(len(ent))[::-1]
f4 = []
for p, (lab, df, c, h) in zip(pos, ent):
    k, n, tot = int((df.falls > 0).sum()), len(df), int(df.falls.sum())
    f4.append(dict(cohort=lab, runs=n, runs_with_fall=k, total_falls=tot))
    ax.plot([0, 100 * k / n], [p, p], color=c, lw=.9, alpha=.55)
    ax.plot(100 * k / n, p, 'o', color=c, ms=4, markeredgecolor='white', mew=.6)
    ax.text(100 * k / n + 1.5, p, f"{k}/{n} ({tot} fall{'' if tot == 1 else 's'})", va='center', fontsize=6, color=INK)
ax.set_yticks(pos, [e[0] for e in ent]); ax.set_xlim(0, 55)
ax.set_xlabel('Runs with at least one fall (%)'); grid(ax)
grp = [e[3] for e in ent]
for i in range(1, len(grp)):
    if grp[i] != grp[i - 1]:
        ax.axhline(pos[i] + .5, color='#D5DCE0', lw=.6)
NUM['fig4'] = f4
save(fig, 'fig4_stability')

# ---------------------------------------------------------------- Figure 6: slip groups (walker_eda.py Fig 3)
LEV = ['0.025', '0.1', '0.2', '1', '10']
BOXKW = dict(patch_artist=True, medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(color=MUTED, lw=.8),
             capprops=dict(color=MUTED, lw=.8), flierprops=dict(marker='o', markersize=2, markerfacecolor='none',
                                                               markeredgecolor=MUTED, markeredgewidth=.5, alpha=.7))


def tint(bp, c, a=.5):
    for bx in bp['boxes']:
        bx.set_facecolor(c); bx.set_alpha(a); bx.set_edgecolor(c)


fig, axs = plt.subplots(1, 2, figsize=(WC, 2.0))
tint(axs[0].boxplot([ET[ET.slip_level == L].gap for L in LEV], positions=range(5), widths=.55, **BOXKW), PURPLE, .35)
axs[0].axhline(0, color=INK, lw=.7, ls='--'); axs[0].set_ylim(-3.5, 2.5)
axs[0].set_title('(a) Learners minus expert'); axs[0].set_ylabel('Progress difference (m)')
f6 = []
for i, L in enumerate(LEV):
    d = ET[ET.slip_level == L]; k, n = int((d.falls > 0).sum()), len(d)
    assert n == 354, (L, n)
    f6.append(dict(slip_group=L, runs=n, runs_with_fall=k, total_falls=int(d.falls.sum())))
    axs[1].bar(i, 100 * k / n, width=.6, color=PURPLE, alpha=.45, edgecolor=PURPLE, lw=.8)
    axs[1].text(i, 100 * k / n + .4, str(k), ha='center', fontsize=6, color=INK)
axs[1].set_ylim(0, 14); axs[1].set_title('(b) Runs with a fall'); axs[1].set_ylabel('Share of 354 runs (%)')
for ax in axs:
    ax.set_xticks(range(5), LEV, fontsize=6); ax.set_xlabel('Slip group centre'); grid(ax, 'y')
NUM['fig6'] = f6
save(fig, 'fig6_surfaces')

# ---------------------------------------------------------------- Figure 5: GAIL history (walker_eda_sweep2.py section 1)
def it(f):
    return int(re.findall(r'iter(?:ation)?_(\d+)', str(f))[-1])


R = ROOT / 'experiments'
RUNS = [('Before changes (DAgger init.)', R / 'archive_20260905_pre_variants/legacy_gail_backups/gail_backup_20260905-022910', RUST),
        ('After, DAgger init.', R / 'archive_20260905_pre_variants/gail_dagger/artifacts', ORANGE),
        ('After, BC init.', LEG / 'gail_bc', BLUE),
        ('After, random init.', LEG / 'gail_random', PURPLE)]
H = []
for name, d, _ in RUNS:
    for f in d.glob('*training_metrics_iter_*.json'):
        m = json.loads(f.read_text()); m.update(run=name, iter=it(f)); H.append(m)
H = pd.DataFrame(H)
assert set(H.run) == {r[0] for r in RUNS}, set(H.run)
H['kl'] = H['accepted_full_kl'].fillna(H.get('approximate_kl'))
H['bal_acc'] = (H.discriminator_expert_accuracy + H.discriminator_policy_accuracy) / 2
fig, axs = plt.subplots(2, 1, figsize=(WC, 3.3), sharex=True)
panels = [('bal_acc', '(a) Discriminator accuracy', False, (.4, 1.02), 'Balanced accuracy'),
          ('expert_dataset_actor_mean_mse', '(b) Actor error on expert data', True, None, 'Ratio to iteration 1 (log)')]
for ax, (col, title, log, ylim, ylab) in zip(axs, panels):
    ax.set_ylabel(ylab)
    for name, _, c in RUNS:
        d = H[H.run == name].sort_values('iter'); y = d[col].values
        if col == 'expert_dataset_actor_mean_mse':
            y = y / y[0]
        ax.plot(d.iter, y, '-o', color=c, lw=1.1, ms=2.6, label=name, markeredgecolor='white', mew=.4)
    if log:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(*ylim)
    if col == 'bal_acc':
        ax.axhline(.5, color=MUTED, ls=':', lw=.7); ax.text(10, .43, 'chance', ha='right', fontsize=6, color=MUTED)
    if col == 'expert_dataset_actor_mean_mse':
        ax.axhline(1, color=MUTED, lw=.6)
    ax.set_title(title); ax.set_xticks(range(1, 11)); grid(ax, 'y')
axs[1].set_xlabel('GAIL iteration')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=1.8, columnspacing=1)
save(fig, 'fig5_gail', rect=(0, 0, 1, .89))

(OUT / 'figure_numbers.json').write_text(json.dumps(NUM, indent=1, default=float))
print('wrote', OUT); print(json.dumps({k: NUM[k] for k in ('fig4', 'fig6')}, default=float))

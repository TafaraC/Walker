"""Reduce per-step evaluation traces to compact per-run summaries (runs on the data host).
Usage: python3 reduce_traces.py <trace_root> <out_dir>
"""
import sys, json, re
from pathlib import Path
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
DT = 0.02
PRE = 150            # 3 s before a fall
BIN = 50             # 1 s bins for error-over-time
rows, prefall, bins, spectra = [], [], [], []

for p in sorted(root.rglob('trace.npz')):
    label, rnd = p.parent.name, p.parent.parent.name
    z = np.load(p)
    a, o = z['actions'].astype(np.float64), z['oracle_actions'].astype(np.float64)
    err = (a - o) ** 2                         # (T,16)
    step_err = err.mean(1)
    ep = z['episode_id']
    T = len(ep)
    falls_at = np.where(np.diff(ep) != 0)[0] + 1   # first step of the new episode
    roll, pitch = z['torso_roll_deg'], z['torso_pitch_deg']
    head, fwd = z['head_height_m'], z['forward_velocity_mps']
    J = z['joint_positions_deg'].astype(np.float64)
    # gait frequency: dominant frequency of each joint (detrended) in 0.5-5 Hz
    Jc = J - J.mean(0)
    F = np.abs(np.fft.rfft(Jc, axis=0)) ** 2
    freqs = np.fft.rfftfreq(T, DT)
    band = (freqs >= .5) & (freqs <= 5)
    pw = F[band].sum(1)                        # power summed over joints
    dom = freqs[band][np.argmax(pw)]
    spectra.append(np.interp(np.linspace(.5, 5, 91), freqs[band], pw / pw.sum()))
    row = dict(round=rnd, label=label, steps=T, falls=len(falls_at),
               mse=step_err.mean(), mse_median_step=np.median(step_err),
               mse_p95_step=np.quantile(step_err, .95), mse_max_step=step_err.max(),
               share_err_top5pct=np.sort(step_err)[-T // 20:].sum() / step_err.sum(),
               dom_freq_hz=dom, roll_rms=np.sqrt((roll ** 2).mean()), pitch_rms=np.sqrt((pitch ** 2).mean()),
               roll_abs_p99=np.quantile(np.abs(roll), .99), pitch_abs_p99=np.quantile(np.abs(pitch), .99),
               head_min=head.min(), head_p01=np.quantile(head, .01))
    for d in range(16):
        row[f'mse_d{d}'] = err[:, d].mean()
        row[f'std_a{d}'] = a[:, d].std()
        row[f'std_o{d}'] = o[:, d].std()
        row[f'corr_d{d}'] = np.corrcoef(a[:, d], o[:, d])[0, 1] if a[:, d].std() > 0 and o[:, d].std() > 0 else np.nan
    for j in range(22):
        row[f'jstd{j}'] = J[:, j].std()
    # error vs posture: mean step error in |roll| quintiles of this run
    q = np.quantile(np.abs(roll), [.2, .4, .6, .8])
    idx = np.digitize(np.abs(roll), q)
    for k in range(5):
        row[f'err_rollq{k}'] = step_err[idx == k].mean()
    rows.append(row)
    bins.append(dict(round=rnd, label=label, bins=step_err[:T // BIN * BIN].reshape(-1, BIN).mean(1).tolist()))
    for f in falls_at:
        if f >= PRE:
            w = slice(f - PRE, f)
            prefall.append(dict(round=rnd, label=label, step=int(f),
                                err=step_err[w].tolist(), roll=roll[w].tolist(), pitch=pitch[w].tolist(),
                                head=head[w].tolist(), fwd=fwd[w].tolist(),
                                err_dims=err[w].mean(0).tolist()))

import csv
keys = list(rows[0])
with open(out / 'trace_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
(out / 'trace_bins.json').write_text(json.dumps(bins))
(out / 'trace_prefall.json').write_text(json.dumps(prefall))
np.save(out / 'trace_spectra.npy', np.array(spectra))
(out / 'trace_spectra_index.json').write_text(json.dumps([[r['round'], r['label']] for r in rows]))
print(len(rows), 'runs;', len(prefall), 'pre-fall windows')

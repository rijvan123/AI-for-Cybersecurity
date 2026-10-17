"""estimate_full_cost.py - extrapolate the S10 (full-mode) CPU wall-clock from the S9 smoke timing logs.

Scaling rules (explicit, so the estimate can be audited):
  E2: fit_eval wall-clock x 5 seeds (near-duplicate measurement only on the first seed).
  E3: per-model fit+eval x 3 seeds; models absent from the smoke run are priced from the measured LightGBM/RF ratio
      (k-NN, extra trees, linear SVM).
  E4: smoke ran 1 fold of the random rung and 5 models; full = 5 folds x 8 models x 5 seeds (factor from fold count).
  E5/E6/E8/E10: factors from the smoke-to-full grid sizes stated in each experiment.
"""
import sys
import pandas as pd
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/claude/s9/runs/smoke')
rows = []


def wall(exp, stage=None, **flt):
    p = base / exp / 'timing.csv'
    if not p.exists():
        return None
    t = pd.read_csv(p)
    if stage:
        t = t[t.stage == stage]
    for k, v in flt.items():
        t = t[t[k] == v]
    return float(t.wall_s.sum())


def add(exp, smoke_s, factor, note):
    if smoke_s is not None:
        rows.append({'experiment': exp, 'smoke_s': round(smoke_s), 'factor': factor,
                     'full_s_est': round(smoke_s * factor), 'note': note})


add('E1', wall('e1'), 5.0, 'block ablation and NIDS port x 5 seeds')
add('E2', wall('e2', 'fit_eval'), 5.0, '5 seeds; near-dup measured on seed 0 only')
add('E2-gates', wall('e2', 'dedup_gate'), 1.0, 'seed 0 only')
e3 = wall('e3')
add('E3', e3, 3.0 * 1.5, '3 seeds; x1.5 for k-NN, extra trees and linear SVM absent from smoke')
add('E4', wall('e4'), 5.0 * 8 / 5 * 2.2, '5 seeds; 8/5 models; random rung 5 folds instead of 1 (~x2.2 overall)')
add('E5', wall('e5'), 5.0 * 4.0, '5 seeds; 16 shadows, 5 epsilons, 30 epochs, 50 rounds (x~4 per seed)')
add('E6', wall('e6'), 5.0 * 3.0, '5 seeds; x3 for all 15 LOFO folds and 2,000 instead of 500 explained rows')
add('E8', wall('e8'), 6.0, '2,000 attacked samples, budgets up to 50')
add('E10', wall('e10'), 5.0 * 2.5, '5 seeds x 5 folds; 6 models; 6 BODMAS fractions')


def chain_wall(script):
    """Wall-clock of a step without timing.csv, from the last chain run in runs/chain.log."""
    import re
    from datetime import datetime
    p = base.parent / 'chain.log'
    if not p.exists():
        return None
    s = e = None
    for line in p.read_text().splitlines():
        m = re.match(rf'== {script} (start|exit)\b.*?(\d\d:\d\d:\d\d)$', line)
        if m:
            tt = datetime.strptime(m.group(2), '%H:%M:%S')
            s, e = (tt, None) if m.group(1) == 'start' else (s, tt)
    return float((e - s).seconds) if s and e else None


n_fits = 0                                                                    # E2 fits with stored scores
if (base / 'e2' / 'scores').exists():
    for f in (base / 'e2' / 'scores').glob('*.npz'):
        n_fits += sum(1 for k in __import__('numpy').load(f).files if k.endswith('|s'))
add('E7', chain_wall('e7_decision_layer'), 5.0 * 4.0,
    f'5x the E2 fits ({n_fits} per seed); base-rate check on 200 fits x 200 subsamples (smoke: 40 x 50); upper bound')
add('Statistics', chain_wall('stats_analysis'), 5.0 * 5.0, 'B = 1,000 instead of 200; 5 seeds')
df = pd.DataFrame(rows)
df.loc[len(df)] = {'experiment': 'TOTAL', 'smoke_s': df.smoke_s.sum(), 'factor': None,
                   'full_s_est': df.full_s_est.sum(), 'note': f"~{df.full_s_est.sum() / 3600:.1f} h on 2 vCPU"}
print(df.to_string(index=False))
df.to_csv(base / 'full_cost_estimate.csv', index=False)

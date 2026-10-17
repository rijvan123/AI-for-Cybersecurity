"""hypothesis_check.py - direction check of every pre-registered hypothesis (S9 report section 3.9) on run outputs.

Smoke mode (seed 0, reduced grids) gives DIRECTION ONLY: no confidence intervals, no tests, no claims. In full mode
the same statistics are the inputs to the formal decision rules (S10 adds group-bootstrap CIs and Holm-corrected
tests from stats_analysis.py). Every value is measured-on-Claude from runs/<mode>/.

    python3 -I tools/hypothesis_check.py runs/smoke      ->  runs/smoke/hypothesis_check.csv (+ markdown on stdout)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0] = ['/home/claude/s9']
from harness.metrics import aut
from scipy.stats import spearmanr

base = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/claude/s9/runs/smoke')
rows = []


def add(hid, statistic, value, verdict, note=''):
    rows.append({'id': hid, 'statistic': statistic, 'value': value, 'direction': verdict, 'note': note})


def read(rel):
    p = base / rel
    return pd.read_csv(p, low_memory=False) if p.exists() else None


def fmt(x, d=3):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


r2 = read('e2/results.csv')
if r2 is not None:
    r2['drop_block'] = r2['drop_block'].fillna('none')
    f2 = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01)]

    def cell(pool, rung, dedup, drop='none', pi=0.01):
        return f2[(f2.pool == pool) & (f2.rung == rung) & (f2.dedup == dedup) & (f2.drop_block == drop) & (f2.pi == pi)]

    # H-C1a: stump vs best model, AP at 1%
    gaps = {}
    for nm, c in (('literature', ('all', 'random', 'none')), ('strict', ('genuine', 'grouped', 'exact'))):
        m = cell(*c).groupby('model')['auc_pr'].mean()
        if 'stump' in m:
            gaps[nm] = (m.max() - m['stump'], m.idxmax(), m['stump'])
    if gaps:
        v = '; '.join(f'{k}: best {b} - stump = {fmt(g)} (stump AP {fmt(s)})' for k, (g, b, s) in gaps.items())
        fals = all(g > 0.10 for g, _, _ in gaps.values())
        add('H-C1a', 'best AP@1% minus stump AP@1%', v, 'inconsistent' if fals else 'consistent',
            'falsified if the gap exceeds 0.10 in both cells')
    # H-C1b: decay with window index when svcscan and synthetic rows are removed
    out = []
    for drop in ('svcscan', 'none'):
        g = cell('genuine', 'temporal', 'exact', drop)
        rh = []
        for m, h in g.groupby('model'):
            h = h.groupby('window')['recall'].mean()
            if len(h) >= 3 and h.nunique() > 1:
                rh.append(spearmanr(h.index, h.to_numpy())[0])
        out.append((drop, np.median(rh) if rh else np.nan, len(rh)))
    v = '; '.join(f'svcscan {"removed" if d == "svcscan" else "kept"}: median rho {fmt(r)} over {n} models'
                  for d, r, n in out)
    rho_removed = out[0][1]
    add('H-C1b', 'Spearman rho(window, recall at FPR<=1%)', v,
        'n/a' if not np.isfinite(rho_removed) else ('consistent' if rho_removed < 0 else 'inconsistent'),
        'falsified if rho >= 0 in the genuine / no-svcscan temporal cell')
    # H-C2b: random minus grouped AP at 1%
    for pool in ('all', 'genuine'):
        a = cell(pool, 'random', 'exact').groupby('model')['auc_pr'].mean()
        b = cell(pool, 'grouped', 'exact').groupby('model')['auc_pr'].mean()
        d = (a - b).dropna()
        if len(d):
            add('H-C2b', f'AP@1% random - grouped ({pool} pool, exact dedup)',
                f'median {fmt(d.median(), 4)}; positive for {int((d > 0).sum())}/{len(d)} models',
                'consistent' if d.median() > 0 else 'inconsistent', 'falsified if the paired difference <= 0')
    # H-C2c: precision at 1% in the strict cell
    s = cell('genuine', 'grouped', 'exact').groupby('model')[['precision', 'fpr']].mean()
    if len(s):
        n_low = int((s.precision < 0.90).sum()); n_fpr = int((s.fpr <= 0.0005).sum())
        add('H-C2c', 'models with precision@1% < 0.90 (strict cell)',
            f'{n_low}/{len(s)} models; {n_fpr}/{len(s)} reach FPR <= 0.05%',
            'consistent' if n_low >= 6 else 'inconsistent',
            'prediction: >= 6 of 9 classifiers below 0.90; falsified if all reach FPR <= 0.05%')
    # H-C5c: AUC-ROC invariant across pi, AP decreasing with pi (exact by construction)
    keys = ['pool', 'rung', 'dedup', 'drop_block', 'model', 'seed', 'fold', 'window', 'family', 'category',
            'family_trained']
    k = [c for c in keys if c in f2]
    g = f2.assign(**{c: f2[c].fillna(-1) for c in k}).groupby(k)
    auc_rng = g['auc_roc'].agg(lambda x: x.max() - x.min()).max()
    mono = g.apply(lambda h: bool(np.all(np.diff(h.sort_values('pi')['auc_pr'].to_numpy()) >= -1e-12)),
                   include_groups=False)
    add('H-C5c', 'max AUC-ROC range across pi; share of fits with AP non-decreasing in pi',
        f'{auc_rng:.2e}; {mono.mean():.3f} of {len(mono)} fits', 'consistent' if auc_rng <= 0.01 and mono.all()
        else 'inconsistent', 'holds by construction of the prevalence weights')

ra = read('stats/rank_agreement.csv')
if ra is not None and len(ra):
    ra = ra[ra.get('drop_block', pd.Series('none', index=ra.index)).fillna('none') == 'none']
    n_lt1 = int((ra.kendall_tau < 1 - 1e-12).sum())
    add('H-C2a', 'cells where the AUC-ROC ranking differs from the MCC@1% ranking (Kendall tau < 1)',
        f'{n_lt1}/{len(ra)} MalMem cells; median tau {fmt(ra.kendall_tau.median())}',
        'consistent' if n_lt1 > 0 else 'inconsistent', 'falsified if tau = 1 in every cell')

r3 = read('e3/results.csv')
if r3 is not None:
    f3 = r3[(r3.rule == 'fpr') & (r3.alpha == 0.01)]
    gaps3 = {}
    for ds in ('bodmas', 'ember'):
        for met, pi_, tag in (('f1', 0.1, 'F1@10% (primary)'), ('auc_pr', 0.01, 'AP@1% (secondary)')):
            d = []
            for m, g in f3[(f3.dataset == ds) & (f3.rung == 'temporal') & (f3.pi == pi_)].groupby('model'):
                if m == 'stump':                                                  # constant score: no decay to measure
                    continue
                a_ = aut(g.groupby('month')[met].mean().sort_index().to_numpy())
                rc = f3[(f3.dataset == ds) & (f3.rung == 'random_contrast') & (f3.pi == pi_) & (f3.model == m)][met].mean()
                d.append((m, rc - a_))
            if d:
                dd = pd.Series(dict(d)); gaps3[(ds, met)] = dd
                add('H-C2d', f'random-contrast {tag} minus AUT ({ds})',
                    f'median {fmt(dd.median())}; >= 0.02 for {int((dd >= 0.02).sum())}/{len(dd)} models',
                    'consistent' if dd.median() >= 0.02 else 'inconsistent', 'falsified if the difference <= 0.02')
    # H-C2f: BODMAS malware-side FNR increase across windows
    b = f3[(f3.dataset == 'bodmas') & (f3.rung == 'temporal') & (f3.pi == 0.01)]
    inc = []
    for m, g in b.groupby('model'):
        h = g.groupby('window')['fnr_all'].mean().sort_index()
        if len(h) >= 4:
            inc.append((m, h.iloc[-3:].mean() - h.iloc[0]))
    if inc:
        s3 = pd.Series(dict(inc))
        add('H-C2f', 'BODMAS FNR (mean of last 3 windows) minus FNR (first window), validation FPR-1% threshold',
            f'median {fmt(s3.median())}; positive for {int((s3 > 0).sum())}/{len(s3)} models',
            'consistent' if s3.median() > 0 else 'inconsistent', 'go/no-go for decay attribution; CI needed in S10')
    # H-C2e: temporal inflation static PE vs memory
    if r2 is not None and ('bodmas', 'f1') in gaps3:
        for met, pi_, tag in (('f1', 0.1, 'F1@10%'), ('auc_pr', 0.01, 'AP@1%')):
            mm = []
            for m, g in f2[(f2.pool == 'genuine') & (f2.rung == 'temporal') & (f2.dedup == 'exact')
                           & (f2.drop_block == 'none') & (f2.pi == pi_)].groupby('model'):
                a_ = aut(g.groupby('window')[met].mean().sort_index().to_numpy())
                rnd = f2[(f2.pool == 'genuine') & (f2.rung == 'random') & (f2.dedup == 'exact')
                         & (f2.drop_block == 'none') & (f2.pi == pi_) & (f2.model == m)]
                if len(rnd):
                    mm.append((m, rnd[met].mean() - a_))
            if mm:
                mg = pd.Series(dict(mm)); bg = gaps3[('bodmas', met)]; common = mg.index.intersection(bg.index)
                add('H-C2e', f'temporal gap (random {tag} - AUT {tag}): BODMAS vs MalMem (genuine)',
                    f'BODMAS median {fmt(bg[common].median())} vs MalMem median {fmt(mg[common].median())} '
                    f'({len(common)} common models)',
                    'consistent' if bg[common].median() > mg[common].median() else 'inconsistent',
                    'falsified if the BODMAS gap <= the MalMem gap')

r5 = read('e5/results.csv')
if r5 is not None:
    mia = r5[r5.part == 'mia']
    if len(mia):
        t = mia.groupby('model')['lira_tpr_at_fpr_0.01'].max()
        trees = t[t.index.isin(['rf', 'lgbm', 'xgb', 'et'])]
        add('H-C3a', 'LiRA TPR at FPR 1% (max over rebalancing variants), tree ensembles',
            '; '.join(f'{m} {fmt(v)}' for m, v in trees.items()),
            'consistent' if (trees > 0.01).any() else 'inconsistent', 'falsified if <= 0.01 for every model')
        p = mia.pivot_table(index='model', columns='variant', values='lira_tpr_at_fpr_0.01', aggfunc='mean')
        if {'smote', 'class_weight'} <= set(p.columns):
            d = (p['smote'] - p['class_weight']).dropna()
            add('H-C3b', 'LiRA TPR at FPR 1%: SMOTE minus class weighting',
                '; '.join(f'{m} {fmt(v, 4)}' for m, v in d.items()),
                'consistent' if (d > 0).all() else ('mixed' if (d > 0).any() else 'inconsistent'),
                'falsified if the paired difference <= 0')
    dp = r5[r5.part == 'dp']
    if len(dp):
        parts, ok = [], True
        var = dp['dp_variant'] if 'dp_variant' in dp else pd.Series('dp', index=dp.index)
        for u, g0 in dp.groupby('unit'):
            v = var[g0.index]
            g = g0[v != 'clip_only']
            g = g.assign(e=g.epsilon_target.replace(np.inf, 1e9)).groupby('e')['mcc_1pct'].mean().sort_index()
            mono = bool(np.all(np.diff(g.to_numpy()) >= -0.01))
            gap8 = g.iloc[-1] - g.get(8.0, np.nan)
            clip = g0[v == 'clip_only'].mcc_1pct.mean() if (v == 'clip_only').any() else np.nan
            ok &= mono and (np.isfinite(gap8) and gap8 <= 0.05)
            parts.append(f'{u}: monotone(+/-0.01) {mono}, MCC gap at eps 8 = {fmt(gap8)}'
                         + (f' (clipping alone: gap {fmt(g.iloc[-1] - clip)})' if np.isfinite(clip) else ''))
        add('H-C3c', 'DP-SGD MCC@1% vs epsilon', '; '.join(parts), 'consistent' if ok else 'inconsistent',
            'falsified by a monotonicity violation or an eps-8 gap > 0.05 MCC')
    fl = r5[r5.part == 'fl']
    if len(fl) and 'gap_mcc_vs_central' in fl:
        g = fl[fl.setting == 'fedavg'].sort_values('alpha')
        gaps_fl = g.gap_mcc_vs_central.to_numpy(); gaps_ap = g.gap_auc_pr_vs_central.to_numpy()
        add('H-C3d', 'FedAvg gap to centralised (MCC@1%; AP@1%) by alpha',
            '; '.join(f'alpha {a:g}: {fmt(x)}; {fmt(y, 4)}' for a, x, y in zip(g.alpha, gaps_fl, gaps_ap)),
            'consistent' if (np.all(np.diff(np.abs(gaps_fl)) <= 0.01) and abs(gaps_fl[-1]) <= 0.03) else 'inconsistent',
            'falsified if the gap does not shrink or exceeds 0.03 at alpha = 1')

r6 = read('e6/shap_shares.csv')
if r6 is not None:
    sh = r6.groupby(['pool', 'rung', 'model'])['share_svcscan'].mean()
    add('H-C4a', 'svcscan share of mean |SHAP| (min over cells x models)',
        f'min {fmt(sh.min())} ({" | ".join(sh.idxmin())}); median {fmt(sh.median())}',
        'consistent' if sh.min() >= 0.5 else 'inconsistent', 'falsified if < 0.5 in any cell')
    st = read('e6/shap_stability.csv')
    if st is not None:
        w = st[st.scope == 'within_cell'].kendall_tau.mean(); a_ = st[st.scope == 'across_cells'].kendall_tau.mean()
        add('H-C4b', 'Kendall tau of SHAP rankings: within cell vs across cells', f'{fmt(w)} vs {fmt(a_)}',
            'consistent' if w > a_ else 'inconsistent', 'falsified if within <= across')
    dl = read('e6/deletion.csv')
    if dl is not None:
        res = []
        for kk, g in dl.groupby('k'):
            if kk > 5:
                continue
            base_ = g.base.iloc[0]
            res.append((kk, base_ - g[g.kind == 'shap_top'].score.mean(), base_ - g[g.kind == 'random'].score.mean()))
        n_ok = sum(t > r for _, t, r in res)
        add('H-C4c', 'AP@1% drop after deleting top-k SHAP vs random-k features (k <= 5)',
            '; '.join(f'k={kk}: {fmt(t, 4)} vs {fmt(r, 4)}' for kk, t, r in res),
            'consistent' if n_ok == len(res) else ('mixed' if n_ok else 'inconsistent'),
            'falsified if the top-k drop <= the random-k drop')

c7 = read('e7/cost_vs_default.csv')
if c7 is not None and len(c7):
    c7['drop_block'] = c7['drop_block'].fillna('none')
    y7 = c7[(c7.pi == 0.01) & (c7.drop_block == 'none')]
    x = y7.rel_reduction
    pooled = 1 - y7.exp_cost_r10_cost.sum() / y7.exp_cost_r10_half.sum()
    add('H-C5a', 'relative expected-cost reduction, cost-optimal vs 0.5 (C_FN/C_FP = 10, pi = 1%)',
        f'pooled {fmt(pooled)} over {len(y7)} fits (primary); per-fit median {fmt(x.median())}, '
        f'{100 * (x.abs() < 1e-12).mean():.0f}% of fits with identical costs (secondary)',
        'consistent' if pooled >= 0.20 else ('inconsistent' if pooled < 0.05 else 'between rules'),
        'prediction >= 20%; falsified if the pooled reduction < 5%')
b7 = read('e7/base_rate_law.csv')
if b7 is not None and len(b7):
    ok = b7['consistent'] if 'consistent' in b7 else b7.inside_ci
    add('H-C5b', 'Eq. 58 at the realised prevalence vs empirical subsampled precision',
        f'{int(ok.sum())}/{len(b7)} (fit, pi) cells consistent ({int(b7.inside_ci.sum())} inside the 95% interval; '
        f'max |analytic - subsample mean| = {(b7.analytic - b7.emp_mean).abs().max():.4f})',
        'consistent' if ok.mean() >= 17 / 18 else 'inconsistent', 'falsified if violated in more than 1 of every 18 cells')
r8 = read('e8/results.csv')
if r8 is not None:
    le = r8[r8.part == 'linear_exactness']
    add('H-C5d', 'max |FGSM evasion - radius prediction| (rows)', f'{int(le.abs_diff_rows.max())} over {len(le)} cells',
        'consistent' if le.abs_diff_rows.max() <= 1 else 'inconsistent', 'falsified if > 1 row at any epsilon')
    g = r8[(r8.part == 'greedy') & (r8.capability == 'user')]
    if len(g):
        bmax = g.budget.max()
        e_b = g[(g.model == 'lgbm') & (g.budget == bmax)].evasion_rate.mean()
        e_m = g[(g.model == 'lgbm_mono') & (g.budget == bmax)].evasion_rate.mean()
        rc = 100 * (g[g.model == 'lgbm'].clean_recall.mean() - g[g.model == 'lgbm_mono'].clean_recall.mean())
        red = 1 - e_m / e_b if e_b > 0 else np.nan
        add('H-C5e', f'user-level evasion at budget {int(bmax)}: LightGBM vs monotone LightGBM',
            f'{fmt(e_b)} vs {fmt(e_m)} (reduction {fmt(red)}); clean recall cost {rc:.2f} pp',
            'consistent' if (red >= 0.5 and rc <= 1) else ('inconsistent' if (red < 0.2 or rc > 3) else 'between rules'),
            'prediction >= 50% reduction at <= 1 pp recall; falsified if < 20% or > 3 pp')
h4 = read('e9/h4_drift_decay.csv')
if h4 is not None and len(h4):
    parts = [f'{d}: median rho {fmt(g.rho_jsd_mcc.median())} ({len(g)} models)' for d, g in h4.groupby('dataset')]
    med = h4.groupby('dataset').rho_jsd_mcc.median()
    add('H-C2g', 'Spearman rho(window mean JSD, window MCC@1%)', '; '.join(parts),
        'consistent' if (med < 0).all() else ('mixed' if (med < 0).any() else 'inconsistent'),
        'falsified if rho >= 0 (one-sided p > 0.05)')

df = pd.DataFrame(rows)
order = ['H-C1a', 'H-C1b', 'H-C2a', 'H-C2b', 'H-C2c', 'H-C2d', 'H-C2e', 'H-C2f', 'H-C2g', 'H-C3a', 'H-C3b', 'H-C3c',
         'H-C3d', 'H-C4a', 'H-C4b', 'H-C4c', 'H-C5a', 'H-C5b', 'H-C5c', 'H-C5d', 'H-C5e']
df['o'] = df.id.map({h: i for i, h in enumerate(order)}); df = df.sort_values(['o'], kind='stable').drop(columns='o')
df.to_csv(base / 'hypothesis_check.csv', index=False)
print('| ID | Statistic (seed 0) | Value | Direction |\n|---|---|---|---|')
for _, r in df.iterrows():
    print(f"| {r['id']} | {r['statistic']} | {r['value']} | {r['direction']} |")

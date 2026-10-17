"""summarise_runs.py - markdown tables of the verification run for the S9 report (reads runs/<mode>/)."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0] = ['/home/claude/s9']
from harness.metrics import aut

base = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/claude/s9/runs/smoke')
out = []


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.{floatfmt}f}')
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def sel(d, rule='fpr', alpha=0.01):
    return d[(d.rule == rule) & ((d.alpha == alpha) if rule == 'fpr' else True)]


# ---------------------------------------------------------------- E2 ladder
p = base / 'e2' / 'results.csv'
if p.exists():
    d = pd.read_csv(p, low_memory=False)
    d['drop_block'] = d.get('drop_block', pd.Series('none', index=d.index)).fillna('none')
    f = sel(d)
    cell = f.pool + ' | ' + f.rung + ' | ' + f.dedup + np.where(f.drop_block != 'none', ' | -' + f.drop_block, '')
    f = f.assign(cell=cell)
    order = ['all | random | none', 'all | random | exact', 'all | grouped | none', 'all | grouped | exact',
             'genuine | random | none', 'genuine | random | exact', 'genuine | grouped | none',
             'genuine | grouped | exact', 'genuine | grouped | near', 'all | lofo | exact', 'genuine | lofo | exact',
             'genuine | loco | exact', 'all | single_family | none', 'genuine | single_family | exact',
             'all | temporal | exact', 'genuine | temporal | exact', 'genuine | temporal | exact | -svcscan',
             'all | temporal | exact | -svcscan', 'genuine | grouped | exact | -svcscan']
    rows = []
    for c in order:
        g = f[f.cell == c]
        if g.empty:
            continue
        r = {'cell (pool, split, dedup)': c}
        for m in ('stump', 'lr', 'rf', 'lgbm'):
            h1 = g[(g.model == m) & (g.pi == 0.01)]
            r[f'{m} AP@1%'] = h1.auc_pr.mean()
            r[f'{m} FPR@95'] = h1.fpr_at_95tpr.mean()
        rows.append(r)
    out.append('### E2 ladder (FPR-1% validation threshold; AP = AUC-PR at 1% prevalence; FPR@95 = FPR at 95% TPR)\n')
    out.append(md(pd.DataFrame(rows), 4))
    lit = f[(f.cell == 'all | random | none') & (f.pi == 0.5)]
    hf = d[(d.rule == 'half') & (d.pool == 'all') & (d.rung == 'random') & (d.dedup == 'none') & (d.pi == 0.5)]
    out.append('\n### E2 literature reproduction (all benign, random 5-fold, 0.5 threshold, 50% prevalence)\n')
    out.append(md(hf.groupby('model')[['accuracy', 'f1', 'auc_roc']].mean().reset_index(), 5))
    st = d[(d.model == 'stump') & d.stump_feature.notna()].groupby(['pool', 'rung', 'drop_block'])['stump_feature'].agg(
        lambda s: s.value_counts().index[0] + f' ({s.value_counts().iloc[0]}/{len(s)})').reset_index()
    out.append('\n### Stump feature chosen per rung\n'); out.append(md(st))
    g = pd.read_csv(base / 'e2' / 'gates.csv')
    g['drop_block'] = g['drop_block'].fillna('none') if 'drop_block' in g else 'none'
    gcols = [c for c in ['test_near_dup_share_malware', 'test_near_dup_share_benign', 'A1_group_overlap',
                         'A5_smote_straddlers', 'smote_purged_train', 'smote_purged_test', 'n_test'] if c in g]
    gg = g.groupby(['pool', 'rung', 'dedup', 'drop_block'])[gcols].mean().reset_index()
    out.append('\n### E2 gates (mean per fold): near-duplicate share of test rows and group overlap\n'); out.append(md(gg, 3))
    t = f[(f.rung == 'temporal') & (f.pi == 0.01) & f.model.isin(['lr', 'rf', 'lgbm', 'xgb'])]
    tt = t.pivot_table(index=['pool', 'drop_block', 'window'], columns='model', values='recall').reset_index()
    out.append('\n### E2 temporal (malware-side) recall per window at the FPR-1% threshold\n'); out.append(md(tt, 4))

# ---------------------------------------------------------------- E3 static PE
p = base / 'e3' / 'results.csv'
if p.exists():
    d = pd.read_csv(p, low_memory=False); f = sel(d)
    rows = []
    for (ds, m), g in f[f.rung.isin(['temporal', 'transfer', 'official'])].groupby(['dataset', 'model']):
        a10 = aut(g[g.pi == 0.1].groupby('month').f1.mean().sort_index().to_numpy())      # seed-mean per month
        a1 = aut(g[g.pi == 0.01].groupby('month').auc_pr.mean().sort_index().to_numpy())
        rc = f[(f.dataset == ds) & (f.rung == 'random_contrast') & (f.model == m)]
        rows.append({'dataset': ds, 'model': m, 'AUT F1@10%': a10,
                     'random F1@10%': rc[rc.pi == 0.1].f1.mean() if len(rc) else np.nan,
                     'AUT AP@1%': a1, 'random AP@1%': rc[rc.pi == 0.01].auc_pr.mean() if len(rc) else np.nan,
                     'mean AUC-ROC': g[g.pi == 0.5].auc_roc.mean(),
                     'max FPR (month)': g[g.pi == 0.01].fpr.max()})
    out.append('\n### E3 static PE: AUT over monthly windows vs random contrast\n'); out.append(md(pd.DataFrame(rows), 3))
    a = f[(f.rung == 'temporal') & (f.pi == 0.01) & (f.dataset == 'bodmas')]
    if 'fnr_seen' in a:
        aa = a.groupby(['model'])[['fnr_all', 'fnr_seen', 'fnr_unseen', 'fpr']].agg(['min', 'max']).round(4)
        aa.columns = [' '.join(c) for c in aa.columns]
        out.append('\n### E3 BODMAS decay attribution (range over 11 windows)\n'); out.append(md(aa.reset_index(), 4))
    fpr_m = f[(f.dataset == 'bodmas') & (f.rung == 'temporal') & (f.pi == 0.01)].pivot_table(
        index='month', columns='model', values='fpr').reset_index()
    out.append('\n### E3 BODMAS FPR per month at the validation FPR-1% threshold\n'); out.append(md(fpr_m, 4))

# ---------------------------------------------------------------- E4 NIDS
p = base / 'e4' / 'results.csv'
if p.exists():
    d = pd.read_csv(p, low_memory=False); f = sel(d)
    x = f[f.pi == 0.01].groupby(['dataset', 'rung', 'dedup', 'model'])[['auc_pr', 'fpr', 'recall']].mean().reset_index()
    x = x[x.model.isin(['stump', 'lr', 'rf', 'lgbm'])]
    out.append('\n### E4 NIDS ladder (AP and FPR/recall at 1% prevalence, FPR-1% threshold)\n'); out.append(md(x, 4))
    g = pd.read_csv(base / 'e4' / 'gates.csv')
    out.append('\n### E4 exact duplicates removed from test folds\n')
    out.append(md(g.groupby(['dataset', 'rung', 'dedup'])[['test_dropped_exact_dups', 'n_test']].sum().reset_index(), 0))

# ---------------------------------------------------------------- E5 privacy
p = base / 'e5' / 'results.csv'
if p.exists():
    d = pd.read_csv(p)
    m = d[d.part == 'mia']
    if len(m):
        cols = ['model', 'variant', 'loss_attack_auc', 'loss_tpr_at_fpr_0.01', 'lira_attack_auc', 'lira_advantage',
                'lira_tpr_at_fpr_0.001', 'lira_tpr_at_fpr_0.01', 'utility_auc_pr_1pct', 'n_shadow']
        out.append('\n### E5 membership inference on genuine benign records\n'); out.append(md(m[cols], 3))
    dp = d[d.part == 'dp']
    if len(dp):
        out.append('\n### E5 DP-SGD frontier\n')
        out.append(md(dp[[c for c in ['unit', 'dp_variant'] if c in dp] + ['epsilon_target', 'epsilon', 'noise_multiplier', 'auc_roc', 'auc_pr_1pct', 'mcc_1pct',
                          'mia_attack_auc', 'mia_advantage', 'seconds']], 3))
    fl = d[d.part == 'fl']
    if len(fl):
        out.append('\n### E5 FedAvg vs centralised\n')
        out.append(md(fl[[c for c in ['setting', 'alpha', 'auc_roc', 'auc_pr_1pct', 'mcc_1pct', 'gap_auc_pr_vs_central',
                                      'client_size_min', 'client_size_max'] if c in fl]], 3))

# ---------------------------------------------------------------- E6 SHAP
p = base / 'e6' / 'shap_shares.csv'
if p.exists():
    d = pd.read_csv(p)
    sh = d.groupby(['pool', 'rung', 'model'])[[c for c in d.columns if c.startswith('share_')]].mean().reset_index()
    out.append('\n### E6 SHAP block shares (mean over folds)\n'); out.append(md(sh, 3))
    top = d.groupby(['pool', 'rung', 'model'])['top1'].agg(lambda s: s.value_counts().index[0]).reset_index()
    out.append('\n### E6 most frequent top-1 feature\n'); out.append(md(top))
    stb = pd.read_csv(base / 'e6' / 'shap_stability.csv')
    out.append('\n### E6 stability (mean)\n')
    out.append(md(stb.groupby(['scope', 'model'])[['jaccard_at_k', 'kendall_tau', 'spearman_rho']].mean().reset_index(), 3))
    dl = pd.read_csv(base / 'e6' / 'deletion.csv')
    out.append('\n### E6 deletion test (AUC-PR at 1% after removing k features)\n')
    out.append(md(dl.groupby(['k', 'kind'])['score'].mean().unstack().reset_index(), 4))

# ---------------------------------------------------------------- E7 decision layer
p = base / 'e7' / 'cost_vs_default.csv'
if p.exists():
    c = pd.read_csv(p)
    cc = c[c.pi.isin([0.01, 0.1])].groupby(['pool', 'rung', 'pi'])['rel_reduction'].median().reset_index()
    out.append('\n### E7 median relative expected-cost reduction, cost-optimal vs 0.5 threshold (C_FN/C_FP = 10)\n')
    out.append(md(cc, 3))
    b = pd.read_csv(base / 'e7' / 'base_rate_law.csv')
    out.append(f"\n### E7 base-rate law (S8 H2): analytic precision inside the empirical 95% interval in "
               f"{int(b.inside_ci.sum())} of {len(b)} (fit, pi) cells\n")
    ab = pd.read_csv(base / 'e7' / 'abstention.csv')
    out.append('\n### E7 two-threshold triage per 10,000 events (mean over fits)\n')
    out.append(md(ab.groupby(['cell', 'model', 'pi'])[['alerts_per_10k', 'deferrals_per_10k', 'alert_precision',
                                                       'missed_malware_share']].mean().reset_index()
                  .query('pi in [0.001, 0.01, 0.1]'), 3))

# ---------------------------------------------------------------- E8 adversarial
p = base / 'e8' / 'results.csv'
if p.exists():
    d = pd.read_csv(p)
    le = d[d.part == 'linear_exactness']
    out.append(f"\n### E8 linear exactness (S8 H5): max |FGSM - radius prediction| = {int(le.abs_diff_rows.max())} rows over "
               f"{len(le)} (model, p, eps) cells\n")
    g = d[d.part == 'greedy']
    out.append('\n### E8 greedy invariant-preserving attack: evasion rate by action budget\n')
    out.append(md(g.pivot_table(index=['model', 'capability', 'clean_recall', 'clean_fpr'], columns='budget',
                                values='evasion_rate').reset_index(), 3))

# ---------------------------------------------------------------- E9 drift, E10 curves
p = base / 'e9' / 'h4_drift_decay.csv'
if p.exists():
    out.append('\n### E9 drift vs decay (Spearman rho, window JSD vs MCC/recall)\n'); out.append(md(pd.read_csv(p), 3))
p = base / 'e10' / 'learning_curves.csv'
if p.exists():
    d = pd.read_csv(p)
    out.append('\n### E10 learning curves (AUC-PR at 1%)\n')
    out.append(md(d.pivot_table(index=['dataset', 'frac'], columns='model', values='auc_pr').reset_index(), 4))
for e in ('e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e8', 'e10'):
    p = base / e / 'timing.csv'
    if p.exists():
        out.append(f'\n{e} wall-clock: {pd.read_csv(p).wall_s.sum():.0f} s')
print('\n'.join(out))

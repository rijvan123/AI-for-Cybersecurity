"""report_tables.py - compact markdown tables for section 6 of the S9 report, generated from runs/<mode>/ outputs.

Writes a JSON of {{TABLE_*}} placeholders that assemble_report.py substitutes, so no table number is hand-typed.
Usage: python3 -I tools/report_tables.py runs/smoke report/tables.json
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0] = ['/home/claude/s9']
from harness.metrics import aut

base = Path(sys.argv[1]); out = Path(sys.argv[2]); T = {}
LAB = {'stump': 'Stump', 'lr': 'LR', 'linsvm': 'Lin. SVM', 'gnb': 'GNB', 'knn': 'k-NN', 'dt': 'DT', 'rf': 'RF',
       'et': 'ET', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'stack_carrier': 'Carrier stack', 'xmal': 'XMal',
       'mlp_poc': 'MLP (PoC)', 'ember_lgbm_official': 'Official EMBER LightGBM', 'lgbm_mono': 'Monotone LightGBM',
       'lgbm_mono_noavg': 'Monotone LightGBM, no averages', 'lgbm_mono_admin': 'Monotone LightGBM, admin features'}


def md(df, digits=3, pct=()):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(('∞' if np.isinf(v) else '—') if not np.isfinite(v) else
                             (f'{100 * v:.1f}%' if c in pct else f'{v:.{digits}f}'))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def read(rel):
    p = base / rel
    return pd.read_csv(p, low_memory=False) if p.exists() else None


# ------------------------------------------------------------------ E2 ladder
r2 = read('e2/results.csv')
if r2 is not None:
    r2['drop_block'] = r2['drop_block'].fillna('none')
    f = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01) & (r2.pi == 0.01)]
    half = r2[(r2.rule == 'half') & (r2.pi == 0.5)]
    cells = [('all', 'random', 'none', 'none', 'Literature: all benign, random, no dedup'),
             ('all', 'random', 'exact', 'none', 'all, random, exact dedup'),
             ('all', 'grouped', 'exact', 'none', 'all, grouped, exact'),
             ('genuine', 'random', 'exact', 'none', 'genuine, random, exact'),
             ('genuine', 'grouped', 'exact', 'none', 'Strict: genuine, grouped, exact'),
             ('genuine', 'grouped', 'near', 'none', 'genuine, grouped, near-dup removed'),
             ('genuine', 'lofo', 'exact', 'none', 'genuine, leave-one-family-out'),
             ('genuine', 'loco', 'exact', 'none', 'genuine, leave-one-category-out'),
             ('genuine', 'single_family', 'exact', 'none', 'genuine, single-family training'),
             ('genuine', 'temporal', 'exact', 'none', 'genuine, temporal (malware-side)'),
             ('genuine', 'grouped', 'exact', 'svcscan', 'Strict without svcscan'),
             ('genuine', 'temporal', 'exact', 'svcscan', 'temporal without svcscan')]
    rows = []
    for p_, r_, d_, b_, name in cells:
        g = f[(f.pool == p_) & (f.rung == r_) & (f.dedup == d_) & (f.drop_block == b_)]
        h = half[(half.pool == p_) & (half.rung == r_) & (half.dedup == d_) & (half.drop_block == b_)]
        if g.empty:
            continue
        row = {'Cell': name}
        for m in ('lgbm', 'rf', 'lr', 'stump'):
            row[f'{LAB[m]} acc@0.5'] = h[h.model == m].accuracy.mean()
            row[f'{LAB[m]} AP@1%'] = g[g.model == m].auc_pr.mean()
        row['LightGBM FPR@95%TPR'] = g[g.model == 'lgbm'].fpr_at_95tpr.mean()
        rows.append(row)
    T['{{TABLE_E2_LADDER}}'] = md(pd.DataFrame(rows), 4)
    # literature vs strict, all models, three operating views
    rows = []
    for m in ['stump', 'lr', 'linsvm', 'gnb', 'knn', 'dt', 'rf', 'et', 'xgb', 'lgbm', 'stack_carrier', 'xmal', 'mlp_poc']:
        row = {'Model': LAB[m]}
        for tag, (p_, r_, d_) in (('Lit.', ('all', 'random', 'none')), ('Strict', ('genuine', 'grouped', 'exact'))):
            h = half[(half.pool == p_) & (half.rung == r_) & (half.dedup == d_) & (half.drop_block == 'none') & (half.model == m)]
            g = f[(f.pool == p_) & (f.rung == r_) & (f.dedup == d_) & (f.drop_block == 'none') & (f.model == m)]
            a3 = r2[(r2.rule == 'fpr') & (r2.alpha == 0.001) & (r2.pi == 0.01) & (r2.pool == p_) & (r2.rung == r_)
                    & (r2.dedup == d_) & (r2.drop_block == 'none') & (r2.model == m)]
            row[f'{tag} acc@0.5'] = h.accuracy.mean(); row[f'{tag} AP@1%'] = g.auc_pr.mean()
            row[f'{tag} prec@1% (α=1%)'] = g.precision.mean(); row[f'{tag} recall (α=0.1%)'] = a3.recall.mean()
        rows.append(row)
    T['{{TABLE_E2_LIT_STRICT}}'] = md(pd.DataFrame(rows), 4)
    g = read('e2/gates.csv'); g['drop_block'] = g['drop_block'].fillna('none')
    gg = g[g.drop_block == 'none'].groupby(['pool', 'rung', 'dedup']).agg(
        near_dup_malware=('test_near_dup_share_malware', 'mean'), near_dup_benign=('test_near_dup_share_benign', 'mean'),
        purged_train=('smote_purged_train', 'mean'), purged_val=('smote_purged_val', 'mean'),
        purged_test=('smote_purged_test', 'mean'), n_test=('n_test', 'mean')).reset_index()
    gg = gg[gg.rung.isin(['random', 'grouped', 'lofo', 'temporal'])]
    gg.columns = ['Pool', 'Split', 'Dedup', 'Malware test rows with near-dup', 'Benign test rows with near-dup',
                  'SMOTE purged (train)', 'SMOTE purged (val)', 'SMOTE purged (test)', 'Test rows']
    for c in ('SMOTE purged (train)', 'SMOTE purged (val)', 'SMOTE purged (test)', 'Test rows'):
        gg[c] = gg[c].map(lambda v: '—' if pd.isna(v) else f'{v:,.0f}')
    T['{{TABLE_E2_GATES}}'] = md(gg, 3, pct=('Malware test rows with near-dup', 'Benign test rows with near-dup'))

# ------------------------------------------------------------------ E1 block ablation
b = read('e1/block_ablation.csv')
if b is not None:
    x = b[(b.pi == 0.01)].groupby(['variant', 'model'])[['auc_roc', 'auc_pr']].mean().unstack('model')
    order = ['all', 'only:svcscan', 'only:pslist', 'only:handles', 'only:ldrmodules', 'only:dlllist', 'only:malfind',
             'only:callbacks', 'only:psxview', 'only:modules', 'drop:svcscan', 'drop:psxview', 'drop:pslist']
    x = x.reindex([o for o in order if o in x.index])
    t = pd.DataFrame({'Features': x.index, 'LightGBM AUC-ROC': x[('auc_roc', 'lgbm')].to_numpy(),
                      'LightGBM AP@1%': x[('auc_pr', 'lgbm')].to_numpy(), 'Stump AUC-ROC': x[('auc_roc', 'stump')].to_numpy(),
                      'Stump AP@1%': x[('auc_pr', 'stump')].to_numpy()})
    T['{{TABLE_E1_BLOCKS}}'] = md(t, 4)

# ------------------------------------------------------------------ E3 static PE
r3 = read('e3/results.csv')
if r3 is not None:
    f3 = r3[(r3.rule == 'fpr') & (r3.alpha == 0.01)]
    rows = []
    for ds, lab in (('bodmas', 'BODMAS'), ('ember', 'EMBER-2018'), ('ember->bodmas', 'EMBER → BODMAS'),
                    ('ember_official', 'Official model, EMBER 2018-12'), ('ember_official->bodmas', 'Official model → BODMAS')):
        for m, g in f3[(f3.dataset == ds) & f3.rung.isin(['temporal', 'transfer', 'official'])].groupby('model'):
            if m not in ('lr', 'rf', 'xgb', 'lgbm', 'ember_lgbm_official'):
                continue
            rc = f3[(f3.dataset == ds) & (f3.rung == 'random_contrast') & (f3.model == m)]
            t1 = g[g.pi == 0.01]
            rows.append({'Data': lab, 'Model': LAB[m],
                         'AUT F1@10%': aut(g[g.pi == 0.1].groupby('month').f1.mean().sort_index().to_numpy()),
                         'Random F1@10%': rc[rc.pi == 0.1].f1.mean() if len(rc) else np.nan,
                         'AUT AP@1%': aut(t1.groupby('month').auc_pr.mean().sort_index().to_numpy()),
                         'Random AP@1%': rc[rc.pi == 0.01].auc_pr.mean() if len(rc) else np.nan,
                         'FNR range': f"{t1.groupby('month').fnr.mean().min():.3f}–{t1.groupby('month').fnr.mean().max():.3f}",
                         'FPR range': f"{t1.groupby('month').fpr.mean().min():.3f}–{t1.groupby('month').fpr.mean().max():.3f}"})
    T['{{TABLE_E3}}'] = md(pd.DataFrame(rows), 3)

# ------------------------------------------------------------------ E4 NIDS
r4 = read('e4/results.csv')
if r4 is not None:
    f4 = r4[(r4.rule == 'fpr') & (r4.alpha == 0.01) & (r4.pi == 0.01)]
    x = f4.groupby(['dataset', 'rung', 'dedup', 'model'])[['auc_roc', 'auc_pr', 'recall', 'fpr']].mean().reset_index()
    x = x[x.model.isin(['stump', 'lr', 'rf', 'lgbm'])]
    x['model'] = x.model.map(LAB)
    x['dataset'] = x.dataset.map({'nsl_kdd': 'NSL-KDD', 'unsw_nb15': 'UNSW-NB15'})
    x['rung'] = x.rung.map({'random': 'random 5-fold', 'official': 'official partition',
                            'lofo': 'leave-one-attack-family-out'})
    x.columns = ['Dataset', 'Rung', 'Dedup', 'Model', 'AUC-ROC', 'AP@1%', 'Recall', 'FPR']
    T['{{TABLE_E4}}'] = md(x, 4)

# ------------------------------------------------------------------ E5 privacy
r5 = read('e5/results.csv')
if r5 is not None:
    m = r5[r5.part == 'mia'][['model', 'variant', 'lira_attack_auc', 'lira_tpr_at_fpr_0.001', 'lira_tpr_at_fpr_0.01',
                              'loss_attack_auc', 'loss_tpr_at_fpr_0.01', 'utility_auc_pr_1pct']].copy()
    m['model'] = m.model.map(LAB)
    m.columns = ['Model', 'Rebalancing', 'LiRA AUC', 'LiRA TPR@0.1%FPR', 'LiRA TPR@1%FPR', 'Loss-attack AUC',
                 'Loss-attack TPR@1%FPR', 'Utility AP@1%']
    T['{{TABLE_E5_MIA}}'] = md(m, 3)
    d = r5[r5.part == 'dp'].copy()
    if 'dp_variant' not in d:
        d['dp_variant'] = 'dp'
    d = d[['unit', 'dp_variant', 'epsilon_target', 'epsilon', 'noise_multiplier', 'auc_roc', 'auc_pr_1pct', 'mcc_1pct',
           'mia_attack_auc']]
    d['dp_variant'] = d.dp_variant.map({'dp': 'DP-SGD', 'clip_only': 'clipping only (σ = 0)',
                                        'non_private': 'non-private SGD'})
    d.columns = ['DP unit', 'Variant', 'ε target', 'ε spent', 'Noise σ', 'AUC-ROC', 'AP@1%', 'MCC@1%', 'Loss-attack AUC']
    T['{{TABLE_E5_DP}}'] = md(d, 3)
    fl = r5[r5.part == 'fl'][['setting', 'alpha', 'auc_roc', 'auc_pr_1pct', 'mcc_1pct', 'gap_auc_pr_vs_central',
                              'gap_mcc_vs_central']].copy()
    fl.columns = ['Setting', 'Dirichlet α', 'AUC-ROC', 'AP@1%', 'MCC@1%', 'AP gap vs central', 'MCC gap vs central']
    T['{{TABLE_E5_FL}}'] = md(fl, 4)

# ------------------------------------------------------------------ E6 SHAP
r6 = read('e6/shap_shares.csv')
if r6 is not None:
    sh = r6.groupby(['pool', 'rung', 'model'])[['share_svcscan', 'share_pslist', 'share_handles', 'share_dlllist',
                                                 'share_ldrmodules']].mean().reset_index()
    top = r6.groupby(['pool', 'rung', 'model'])['top1'].agg(lambda s: s.value_counts().index[0]).reset_index()
    sh = sh.merge(top, on=['pool', 'rung', 'model'])
    sh['model'] = sh.model.map(LAB)
    sh.columns = ['Pool', 'Rung', 'Model', 'svcscan', 'pslist', 'handles', 'dlllist', 'ldrmodules', 'Most frequent top-1']
    T['{{TABLE_E6}}'] = md(sh, 3)

# ------------------------------------------------------------------ E8 adversarial
r8 = read('e8/results.csv')
if r8 is not None:
    g = r8[r8.part == 'greedy']
    x = g.pivot_table(index=['model', 'capability'], columns='budget', values='evasion_rate').reset_index()
    cr = g.groupby('model')[['clean_recall', 'clean_fpr']].mean()
    x['Clean recall'] = x.model.map(cr.clean_recall); x['Clean FPR'] = x.model.map(cr.clean_fpr)
    x['model'] = x.model.map(LAB)
    x.columns = ['Model', 'Attacker'] + [f'k = {int(c)}' for c in x.columns[2:-2]] + ['Clean recall', 'Clean FPR']
    T['{{TABLE_E8}}'] = md(x, 3)

# ------------------------------------------------------------------ E10 learning curves
r10 = read('e10/learning_curves.csv')
if r10 is not None:
    x = r10.pivot_table(index=['dataset', 'frac'], columns='model', values='auc_pr').reset_index()
    x.columns = ['Data', 'Training fraction'] + [LAB.get(c, c) for c in x.columns[2:]]
    T['{{TABLE_E10}}'] = md(x, 3)

# ------------------------------------------------------------------ S10 CPU budget
fc = read('full_cost_estimate.csv')
if fc is not None:
    fc = fc.copy()
    fc['factor'] = fc['factor'].map(lambda v: '' if pd.isna(v) else f'{float(v):g}')
    fc['smoke_s'] = fc['smoke_s'].map(lambda v: f'{v:,.0f}')
    fc['full_h'] = fc['full_s_est'].map(lambda v: f'{v / 3600:.2f}')
    fc = fc[['experiment', 'smoke_s', 'factor', 'full_h', 'note']]
    fc.columns = ['Step', 'Timed smoke stages (s)', 'Factor', 'Full estimate (h)', 'Basis']
    T['{{TABLE_BUDGET}}'] = md(fc)

# ------------------------------------------------------------------ hypothesis direction check
hc = read('hypothesis_check.csv')
if hc is not None:
    hc = hc[['id', 'statistic', 'value', 'direction']].copy()
    hc.columns = ['ID', 'Statistic (seed 0)', 'Value', 'Direction']
    hc['Value'] = hc['Value'].astype(str).str.replace('|', '\\|', regex=False)          # escape table pipes
    hc['Statistic (seed 0)'] = hc['Statistic (seed 0)'].astype(str).str.replace('|', '\\|', regex=False)
    T['{{TABLE_HYP}}'] = md(hc)

out.write_text(json.dumps(T, indent=1, ensure_ascii=False))
print('\n\n'.join(f'{k}\n{v}' for k, v in T.items()))

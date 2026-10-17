"""E7 - C5 decision layer and deployment-readiness profile (post-hoc on E2/E3/E5/E6/E8 outputs; S5 I07, S6 P7).

Outputs: (1) cost_vs_default.csv - expected cost of the Bayes-optimal threshold vs the 0.5 default (H6.9);
(2) abstention.csv - two-threshold triage on the literature and strict cells (alert above the FPR-0.1% threshold,
defer between the 0.1% and 1% thresholds, pass below) with deferrals per 10,000 events at each prevalence; (3) base_rate_law.csv - analytic
precision (Eq. 58, at the realised subsample prevalence; nominal-pi value kept) vs empirical subsampled precision
(S8 H2); (4) readiness_profile.csv per model.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness.metrics import precision_at
from harness.splits import prevalence_subsample


def main():
    a = parse('E7 decision layer')
    out = outdir(a.mode, 'e7'); base = out.parent
    r2 = pd.read_csv(base / 'e2' / 'results.csv', low_memory=False)
    r2['drop_block'] = r2['drop_block'].fillna('none') if 'drop_block' in r2 else 'none'
    for c in ('fold', 'window', 'family', 'category', 'family_trained'):       # split identifiers (NaN-safe keys)
        r2[c] = r2[c].fillna(-1) if c in r2 else -1
    # (1) cost-optimal vs default 0.5 threshold, matched one-to-one on (cell, model, seed, split, pi)
    keys = ['pool', 'rung', 'dedup', 'drop_block', 'model', 'seed', 'fold', 'window', 'family', 'category',
            'family_trained', 'pi']
    cost = r2[(r2.rule == 'cost') & (r2.cost_ratio == 10)][keys + ['exp_cost_r10', 'precision', 'recall']]
    half = r2[r2.rule == 'half'][keys + ['exp_cost_r10', 'precision', 'recall']]
    j = cost.merge(half, on=keys, suffixes=('_cost', '_half'), how='inner', validate='one_to_one')
    j['rel_reduction'] = 1 - j['exp_cost_r10_cost'] / j['exp_cost_r10_half'].replace(0, np.nan)
    j.to_csv(out / 'cost_vs_default.csv', index=False)
    # (2) abstention band on stored scores of the literature and strict cells (full feature set)
    sc = load_scores(base / 'e2'); rows = []
    triage_cells = {('all', 'random', 'none'): 'literature', ('genuine', 'grouped', 'exact'): 'strict'}
    for k, (y, s, thr, _) in sc.items():
        f = parse_score_key(k)
        cell_name = triage_cells.get((f['pool'], f['rung'], f['dedup']))
        if cell_name is None or f['drop_block'] != 'none':
            continue
        t_lo, t_hi = thr.get('fpr|0.01|nan|0.5'), thr.get('fpr|0.001|nan|0.5')
        if t_lo is None or t_hi is None:
            continue
        for pi in C.PREVALENCES:
            w = np.where(y == 1, pi / max(y.mean(), 1e-12), (1 - pi) / max(1 - y.mean(), 1e-12)) / len(y)
            alert, defer = s >= t_hi, (s >= t_lo) & (s < t_hi)
            passed = s < t_lo
            rows.append({'key': k, 'cell': cell_name, 'model': f['model'], 'seed': f['seed'], 'fold': f['fold'],
                         'pi': pi,
                         'alerts_per_10k': 1e4 * w[alert].sum(), 'deferrals_per_10k': 1e4 * w[defer].sum(),
                         'alert_precision': (w[alert & (y == 1)].sum() / w[alert].sum()) if w[alert].sum() else np.nan,
                         'missed_malware_share': w[passed & (y == 1)].sum() / w[y == 1].sum() if (y == 1).any() else np.nan})
    pd.DataFrame(rows).to_csv(out / 'abstention.csv', index=False)
    # (3) S8 H2: analytic vs empirical precision at the FPR-1% threshold (40 fits in smoke, 200 in full)
    rng = np.random.default_rng(0); rows = []
    keys_h2 = [k for k in sc if parse_score_key(k)['drop_block'] == 'none']
    n_h2 = 40 if a.mode == 'smoke' else 200                                   # fits spread evenly over cells
    keys_h2 = keys_h2[::max(1, len(keys_h2) // n_h2)][:n_h2]
    for k in keys_h2:
        y, s, thr, _ = sc[k]
        t = thr.get('fpr|0.01|nan|0.5')
        if t is None or len(np.unique(y)) < 2:
            continue
        tpr, fpr = (s[y == 1] >= t).mean(), (s[y == 0] >= t).mean()
        for pi in (0.10, 0.05, 0.01):
            emp, pi_real = [], np.nan
            for _ in range(C.R_PREV if a.mode == 'full' else 50):
                sel = prevalence_subsample(y, pi, rng, min_pos=1)
                if sel is None:
                    break
                yy, ss = y[sel], s[sel] >= t
                pi_real = yy.mean()                       # integer row counts: realised prevalence, same each draw
                n_sub = len(yy)
                emp.append(yy[ss].mean() if ss.any() else np.nan)
            emp = np.asarray(emp, float)
            if len(emp) and np.isfinite(emp).any():
                lo, hi = np.nanquantile(emp, [0.025, 0.975])
                ana = precision_at(tpr, fpr, pi_real)     # Eq. 58 at the prevalence the subsamples actually have
                alerts = n_sub * (pi_real * tpr + (1 - pi_real) * fpr)   # expected alerts per subsample
                rows.append({'key': k, 'pi': pi, 'pi_realised': pi_real, 'tpr': tpr, 'fpr': fpr, 'analytic': ana,
                             'analytic_nominal_pi': precision_at(tpr, fpr, pi), 'emp_mean': np.nanmean(emp),
                             'emp_lo': lo, 'emp_hi': hi, 'interval_width': hi - lo, 'expected_alerts': alerts,
                             'inside_ci': bool(lo - 1e-9 <= ana <= hi + 1e-9),
                             'inside_ci_nominal_pi': bool(lo - 1e-9 <= precision_at(tpr, fpr, pi) <= hi + 1e-9),
                             # discrete subsamples: one alert moves precision by ~1/alerts, so a zero-width interval
                             # is judged against the subsample mean at that resolution
                             'consistent': bool((lo - 1e-9 <= ana <= hi + 1e-9)
                                                or abs(ana - np.nanmean(emp)) <= 1.0 / max(alerts, 1.0)),
                             'n_replicates': int(np.isfinite(emp).sum())})
    pd.DataFrame(rows).to_csv(out / 'base_rate_law.csv', index=False)
    # (4) readiness profile per cell (svcscan-removed cells kept separate via drop_block), pi = 1%, FPR-1% rule
    sel = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01) & (r2.pi == 0.01)]
    prof = (sel.groupby(['pool', 'rung', 'dedup', 'drop_block', 'model'])
            [['auc_roc', 'auc_pr', 'precision', 'recall', 'mcc', 'fpr', 'fpr_at_95tpr', 'alarms_per_10k',
              'false_alarms_per_10k', 'us_per_sample', 'fit_s']].mean().reset_index())
    for extra, fname in (('e6', 'shap_shares.csv'),):
        p = base / extra / fname
        if p.exists():
            sh = pd.read_csv(p).groupby(['pool', 'rung', 'model'])['share_svcscan'].mean().reset_index()
            prof = prof.merge(sh, on=['pool', 'rung', 'model'], how='left')
    p5 = base / 'e5' / 'results.csv'
    if p5.exists():
        m5 = pd.read_csv(p5); m5 = m5[m5.part == 'mia']
        mia = m5[m5.variant == 'class_weight'].groupby('model')[['lira_advantage', 'lira_tpr_at_fpr_0.01']].mean()
        prof = prof.merge(mia.reset_index(), on='model', how='left')
    prof.to_csv(out / 'readiness_profile.csv', index=False)
    print(prof[(prof.pool == 'genuine') & (prof.rung == 'grouped') & (prof.dedup == 'exact')
               & (prof.drop_block == 'none')].round(4).to_string())


if __name__ == '__main__':
    main()

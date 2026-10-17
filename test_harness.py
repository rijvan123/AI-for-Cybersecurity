"""Unit tests for the S9 harness (run: python3 -I tests/test_harness.py). Each test checks an identity from S8 or a
leakage gate on synthetic data, so a regression fails loudly before any experiment is trusted."""
import sys
sys.path.insert(0, '/home/claude/s9')
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from harness import metrics as MT, thresholds as TH, splits as S, stats as ST, drift as D, adversarial as A, privacy as P


def test_panel_matches_sklearn_at_empirical_prevalence():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 2000); s = y + rng.normal(0, 0.8, 2000)
    r = MT.panel(y, s, 0.5, None)
    pred = s >= 0.5
    assert abs(r['f1'] - f1_score(y, pred)) < 1e-12
    assert abs(r['mcc'] - matthews_corrcoef(y, pred)) < 1e-12
    assert abs(r['auc_roc'] - roc_auc_score(y, s)) < 1e-12
    assert abs(r['auc_pr'] - average_precision_score(y, s)) < 1e-12


def test_weighted_precision_equals_base_rate_law():
    """Eq. 58: precision of the re-weighted confusion matrix equals pi TPR / (pi TPR + (1 - pi) FPR)."""
    rng = np.random.default_rng(1); y = np.r_[np.ones(500), np.zeros(3000)].astype(int)
    s = np.r_[rng.normal(2, 1, 500), rng.normal(0, 1, 3000)]
    for pi in (0.5, 0.1, 0.01, 0.001):
        r = MT.panel(y, s, 1.0, pi)
        tpr, fpr = (s[y == 1] >= 1).mean(), (s[y == 0] >= 1).mean()
        assert abs(r['precision'] - MT.precision_at(tpr, fpr, pi)) < 1e-10
        assert abs(r['recall'] - tpr) < 1e-12 and abs(r['fpr'] - fpr) < 1e-12


def test_weighted_ap_matches_subsampling_expectation():
    """Prevalence-weighted AP approximates the mean AP of Eq. 7 subsamples (consistency of the weighting)."""
    rng = np.random.default_rng(2); y = np.r_[np.ones(4000), np.zeros(4000)].astype(int)
    s = np.r_[rng.normal(1.5, 1, 4000), rng.normal(0, 1, 4000)]
    pi = 0.1; w_ap = MT.panel(y, s, 0, pi)['auc_pr']
    aps = []
    for _ in range(200):
        sel = S.prevalence_subsample(y, pi, rng)
        aps.append(average_precision_score(y[sel], s[sel]))
    assert abs(w_ap - np.mean(aps)) < 0.01, (w_ap, np.mean(aps))


def test_fpr_threshold_respects_budget():
    rng = np.random.default_rng(3)
    for n in (50, 999, 5000):
        b = rng.normal(0, 1, n)
        for a in (1e-3, 1e-2, 0.1):
            t = TH.fpr_threshold(b, a)
            assert (b >= t).mean() <= a + 1e-12


def test_fpr_max_inverts_precision():
    for pi in (0.01, 0.05, 0.1):
        f = MT.fpr_max(0.95, pi, 0.9)
        assert abs(MT.precision_at(0.95, f, pi) - 0.9) < 1e-12


def test_prior_shift_inverse():
    p = np.linspace(0.01, 0.99, 50)
    q = TH.prior_shift(TH.prior_shift(p, 0.5, 0.01), 0.01, 0.5)
    assert np.allclose(p, q)


def test_aut_trapezoid():
    assert abs(MT.aut([1, 1, 1]) - 1) < 1e-12
    assert abs(MT.aut([1, 0]) - 0.5) < 1e-12
    assert abs(MT.aut([0.9, 0.8, 0.6, 0.5]) - ((0.85 + 0.7 + 0.55) / 3)) < 1e-12


def test_psi_is_jeffreys_and_jsd_bounded():
    rng = np.random.default_rng(4); p = rng.dirichlet(np.ones(10)); q = rng.dirichlet(np.ones(10))
    assert abs(D.psi(p, q) - (D.kl(p, q) + D.kl(q, p))) < 1e-12               # Eq. 71
    assert 0 <= D.jsd(p, q) <= 1 and D.jsd(p, p) < 1e-12                      # Eq. 72


def test_grouped_kfold_never_splits_a_group():
    rng = np.random.default_rng(5); g = rng.integers(0, 300, 6000).astype(str); y = (rng.random(6000) < 0.4).astype(int)
    for sp in S.grouped_kfold(y, g, 5, 0, 0.2):
        S.check_split(sp, y, groups=g, require_group_disjoint=True)


def test_lofo_purges_multilabel_groups():
    y = np.r_[np.ones(40), np.zeros(40)].astype(int)
    fam = np.array(['a'] * 20 + ['b'] * 20 + ['Benign'] * 40, dtype=object)
    g = np.array([f'm{i // 2}' for i in range(40)] + [f'b{i}' for i in range(40)], dtype=object)
    g[20] = g[0]                                                               # one binary labelled a and b
    for sp in S.leave_one_family_out(y, fam, g, 0, 5, 0.2):
        S.check_split(sp, y, groups=g, require_group_disjoint=True)


def test_temporal_gate_detects_leak():
    y = np.r_[np.ones(10), np.zeros(10)].astype(int); t = np.r_[np.arange(10), np.full(10, np.nan)]
    sp = S.Split('temporal', np.array([0, 1, 9]), np.array([2]), np.array([5, 6]), {})
    try:
        S.check_split(sp, y, t=t, require_temporal=True); raised = False
    except S.LeakageError:
        raised = True
    assert raised


def test_month_windows_merge_rule():
    import pandas as pd
    counts = pd.Series({157: 300, 160: 10, 163: 50, 166: 400, 181: 20})
    w, n = S.month_windows(157, 6, 5, counts, 200)
    assert sum(n) == counts.sum() and min(n) >= 200, (w, n)


def test_linear_fgsm_exactness():
    """S8 H5: FGSM evasion rate equals the radius-predicted fraction for a linear score."""
    rng = np.random.default_rng(6); Z = rng.normal(0, 1, (500, 8)); w = rng.normal(0, 1, 8); b = 0.3
    tau = 0.0; det = Z @ w + b >= tau
    for p in ('inf', '2'):
        df = A.evasion_rate_check(Z[det], w, b, tau, (0, 0.1, 0.5, 1, 2), p)
        assert (df['abs_diff_rows'] <= 1).all(), df


def test_delong_against_identity():
    rng = np.random.default_rng(7); y = rng.integers(0, 2, 400); s = y + rng.normal(0, 1, 400)
    a1, a2, z, p = ST.delong(y, s, s)
    assert abs(a1 - roc_auc_score(y, s)) < 1e-12 and p == 1.0


def test_cliffs_delta_extremes():
    assert ST.cliffs_delta([2, 3], [0, 1])[0] == 1.0 and ST.cliffs_delta([0, 1], [2, 3])[0] == -1.0


def test_smote_rows_are_interpolants():
    rng = np.random.default_rng(8); X = rng.normal(0, 1, (30, 4))
    Xs, base, mate = P.smote(X, 50, k=5, seed=0)
    lam = ((Xs - X[base]) / (X[mate] - X[base]))
    assert np.allclose(lam, lam[:, :1]) and (lam >= 0).all() and (lam <= 1).all()


def test_rdp_noise_calibration_monotone():
    s1 = P.calibrate_noise(1.0, 0.01, 1000, 1e-5); s8 = P.calibrate_noise(8.0, 0.01, 1000, 1e-5)
    assert s1 > s8 > 0 and P.rdp_epsilon(s1, 0.01, 1000, 1e-5) <= 1.0 + 1e-6


def test_invariants_on_constructed_rows():
    import pandas as pd
    from harness.data_malmem import invariant_table, INVARIANTS
    cols = sorted({c for v in INVARIANTS.values() for c in v[1:]})
    X = pd.DataFrame(np.zeros((2, len(cols))), columns=cols)
    X.loc[:, 'pslist.nproc'] = 9; X.loc[:, 'pslist.avg_threads'] = 95 / 9; X.loc[:, 'pslist.avg_handlers'] = 2000 / 9
    X.loc[:, 'handles.nhandles'] = 1800; X.loc[:, 'handles.avg_handles_per_proc'] = 200.0
    X.loc[:, 'dlllist.ndlls'] = 450; X.loc[:, 'dlllist.avg_dlls_per_proc'] = 50.0
    X.loc[1, 'pslist.avg_threads'] = 0.5 * (95 / 9 + 120 / 10)                # interpolated (SMOTE-like) row
    tab = invariant_table(X.round(8), 1e-3)
    assert tab.loc[0].all() and not tab.loc[1, 'I1']



# ---------------------------------------------------------------- regression tests added after the S9 code review
def test_fpr_at_tpr_keeps_collinear_points():
    """Ties of one malware and one benign row per score put ROC points on the diagonal; the exact FPR at 95% TPR
    is 0.95 and must not be read from a dropped (collinear) point."""
    s = np.repeat(np.arange(20, dtype=float), 2); y = np.tile([1, 0], 20)
    assert abs(MT.fpr_at_tpr(y, s, 0.95) - 0.95) < 1e-12


def test_fpr_threshold_refuses_empty_validation():
    try:
        TH.fpr_threshold(np.array([]), 0.01); raised = False
    except ValueError:
        raised = True
    assert raised


def test_temporal_both_dated_refuses_empty_validation():
    y = np.array([0, 1]); ym = np.array(['2019-08', '2019-09'], dtype=object)
    try:
        next(S.temporal_both_dated(y, ym, np.arange(2), ('2019-08',), ('2019-09',), 0.2)); raised = False
    except ValueError:
        raised = True
    assert raised


def test_smote_straddlers_purged_and_gated():
    """A child grouped with parent a (group A) whose parent b sits in group B must leave whichever side it is on
    when A and B land in different partitions; gate A5 raises before the purge and passes after it."""
    groups = np.array(['A', 'A', 'B', 'B', 'A', 'B'], dtype=object)            # rows 4, 5 are SMOTE children
    pg = np.empty((6, 2), dtype=object); pg[4] = ('A', 'B'); pg[5] = ('B', 'B')
    y = np.zeros(6, int)
    sp = S.Split('grouped', np.array([0, 1, 4]), np.array([], int), np.array([2, 3, 5]), {})
    try:
        S.check_split(sp, y, groups=groups, parent_groups=pg, require_group_disjoint=True); raised = False
    except S.LeakageError:
        raised = True
    assert raised
    sp = S.purge_smote_straddlers(sp, groups, pg)
    assert list(sp.train) == [0, 1] and list(sp.test) == [2, 3, 5] and sp.info['smote_purged_train'] == 1
    assert S.check_split(sp, y, groups=groups, parent_groups=pg, require_group_disjoint=True)['A5_smote_straddlers'] == 0


def test_score_store_roundtrip_reproduces_decisions():
    """Stored scores are float64, so the stored validation threshold reproduces in-process decisions exactly,
    including tied tree scores of the form k/300."""
    import tempfile, json
    from pathlib import Path
    sys.path.insert(0, '/home/claude/s9/experiments')
    from common import ScoreStore, load_scores
    rng = np.random.default_rng(9); s = rng.integers(0, 301, 4000) / 300.0; y = (rng.random(4000) < 0.3).astype(int)
    thr = TH.fpr_threshold(s[y == 0], 0.01)
    with tempfile.TemporaryDirectory() as d:
        st = ScoreStore(); st.add(('x', 0, 'rf'), np.arange(4000), y, s, {'fpr|0.01|nan|0.5': thr})
        st.flush(Path(d) / 'scores', 'cell')
        (yy, ss, tt, rid), = load_scores(Path(d)).values()
    assert np.array_equal(ss >= tt['fpr|0.01|nan|0.5'], s >= thr)


def test_done_keys_treat_missing_drop_block_as_none():
    import tempfile
    import pandas as pd
    from pathlib import Path
    sys.path.insert(0, '/home/claude/s9/experiments')
    from common import done_keys
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'r.csv'
        pd.DataFrame({'pool': ['all', 'genuine'], 'drop_block': [np.nan, 'svcscan'], 'seed': [0, 0]}).to_csv(p, index=False)
        k = done_keys(p, ['pool', 'drop_block', 'seed'])
    assert ('all', 'none', '0') in k and ('genuine', 'svcscan', '0') in k


def test_lira_uses_global_variance_with_few_out_shadows():
    rng = np.random.default_rng(10); phi_out = rng.normal(0, 1, (50, 4)); phi_t = rng.normal(0, 1, 50)
    z = P.lira_offline(phi_t, phi_out)
    expect = (phi_t - phi_out.mean(1)) / (np.std(phi_out) + 1e-6)
    assert np.allclose(z, expect)


def test_append_rows_union_and_atomic_replace():
    import tempfile
    import pandas as pd
    from pathlib import Path
    from harness.io_utils import append_rows
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'r.csv'
        append_rows([{'a': 1}], p); append_rows([{'a': 2, 'b': 3}], p)
        df = pd.read_csv(p)
        assert list(df.columns) == ['a', 'b'] and len(df) == 2 and not Path(str(p) + '.tmp').exists()


def test_lightgbm_importances_are_normalised_gain():
    from harness import models as M
    rng = np.random.default_rng(11); X = rng.normal(0, 1, (400, 5)); y = (X[:, 0] + 0.2 * rng.normal(0, 1, 400) > 0).astype(int)
    m = M.make_model('lgbm', 0); M.fit(m, X, y)
    imp = M.importances(m, [f'f{j}' for j in range(5)])
    g = m.named_steps['clf'].booster_.feature_importance(importance_type='gain')
    keep = m.named_steps['var'].get_support()
    assert abs(imp.sum() - 1) < 1e-12 and np.allclose(imp[keep], g / g.sum()) and imp.argmax() == 0


def test_rankstats_match_sklearn_with_ties_and_multiplicities():
    from harness.metrics import class_weights
    rng = np.random.default_rng(12)
    for trial in range(6):
        n = 1500; y = (rng.random(n) < 0.3).astype(int); s = np.round(y + rng.normal(0, 1, n), 1 + trial % 2)
        c = rng.integers(0, 3, n).astype(float); rs = ST.RankStats(y, s)
        idx = np.repeat(np.arange(n), c.astype(int)); yb, sb = y[idx], s[idx]
        assert abs(rs.auc(c) - roc_auc_score(yb, sb)) < 1e-12
        for pi in (None, 0.01):
            assert abs(rs.ap(c, pi) - average_precision_score(yb, sb, sample_weight=class_weights(yb, pi))) < 1e-10


def test_group_counts_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(13); g = np.repeat(np.arange(300), 5); y = (g % 3 == 0).astype(int)
    s = y + rng.normal(0, 1, len(y)); rs = ST.RankStats(y, s)
    ci = ST.group_counts_bootstrap(y, g, {'auc': rs.auc}, B=300, seed=0)
    pt = rs.auc(np.ones(len(y)))
    assert ci['auc'][0] < pt < ci['auc'][1] and ci['auc'][2] == 300

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t(); print('PASS', t.__name__)
        except Exception as e:                                                # noqa: BLE001
            failed += 1; print('FAIL', t.__name__, repr(e))
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)

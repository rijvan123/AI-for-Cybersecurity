"""stats.py - uncertainty and significance (S8 Eqs. 94-97; Demsar 2006; DeLong et al. 1988).

* group_bootstrap / group_counts_bootstrap: percentile CIs resampling sample-identity groups within each class
  (Eq. 94), so ten dumps of one binary never count as ten independent observations; the counts version evaluates
  AUC-ROC, AP and decision metrics from per-row multiplicities (RankStats) in O(n) per resample.
* paired tests across seeds x folds (or windows): Wilcoxon signed-rank (Eq. 95) + Cliff's delta (Eq. 97),
  Holm correction for families of comparisons.
* Friedman + Nemenyi critical difference over models x blocks (Eq. 96).
* DeLong test for correlated AUCs; McNemar exact test on paired decisions; Kendall tau for rankings (H2 of S2).
"""
import numpy as np
from scipy import stats as st

# Studentised range q_alpha / sqrt(2) for the Nemenyi test, alpha = 0.05 (Demsar 2006, Table 5), k = 2..20
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def group_bootstrap(y, s, groups, stat_fn, B=1000, seed=0, alpha=0.05):
    """Percentile CI of stat_fn(y_b, s_b) over B resamples of groups (stratified by class)."""
    y = np.asarray(y); s = np.asarray(s); g = np.asarray(groups)
    rng = np.random.default_rng(seed)
    idx_by_class = []
    for c in (0, 1):
        gc = g[y == c]; rows = np.flatnonzero(y == c)
        ug, inv = np.unique(gc, return_inverse=True)
        members = [rows[inv == i] for i in range(len(ug))]
        idx_by_class.append(members)
    vals = []
    for _ in range(B):
        take = []
        for members in idx_by_class:
            if not members:
                continue
            pick = rng.integers(0, len(members), len(members))
            take.append(np.concatenate([members[i] for i in pick]))
        b = np.concatenate(take)
        try:
            vals.append(stat_fn(y[b], s[b]))
        except ValueError:                                                    # degenerate resample
            continue
    vals = np.asarray(vals, float)
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), vals


class RankStats:
    """AUC-ROC and prevalence-weighted AP from per-row multiplicities (a bootstrap resample = counts per row).

    Scores are ranked once; each statistic is then O(n) in numpy, with ties handled exactly as scikit-learn does
    (AUC = Mann-Whitney with 1/2 per tie; AP = step-wise sum over distinct thresholds, Eq. 67)."""

    def __init__(self, y, s):
        self.y = np.asarray(y).astype(int)
        lv, self.level = np.unique(np.asarray(s, float), return_inverse=True)          # ascending score levels
        self.L = len(lv)

    def _agg(self, counts):
        c = np.asarray(counts, float)
        pos = np.bincount(self.level, weights=c * (self.y == 1), minlength=self.L)
        neg = np.bincount(self.level, weights=c * (self.y == 0), minlength=self.L)
        return pos, neg

    def auc(self, counts):
        pos, neg = self._agg(counts); P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0:
            return np.nan
        return float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (P * N))

    def ap(self, counts, pi=None):
        pos, neg = self._agg(counts); P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0:
            return np.nan
        if pi is not None:                                                              # metrics.class_weights
            n = P + N; pos = pos * (pi * n / P); neg = neg * ((1 - pi) * n / N)
        tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])                           # descending thresholds
        prec = tp / np.maximum(tp + fp, 1e-300)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def group_counts_bootstrap(y, groups, stat_fns, B=1000, seed=0, alpha=0.05):
    """Eq. (94) for several statistics at once: resample sample-identity groups within each class and represent each
    resample by per-row multiplicities (every row of a group drawn m times counts m times). stat_fns maps a name to
    f(counts) -> float. Returns {name: (lo, hi, n_valid)}. Same estimand as group_bootstrap, O(n) per resample."""
    y = np.asarray(y).astype(int); g = np.asarray(groups)
    rng = np.random.default_rng(seed)
    cls = [np.flatnonzero(y == c) for c in (0, 1)]
    gid, sizes = np.zeros(len(y), int), []
    for c in (0, 1):
        ug, inv = np.unique(g[cls[c]], return_inverse=True)
        gid[cls[c]] = inv; sizes.append(len(ug))
    vals = {k: [] for k in stat_fns}
    for _ in range(B):
        counts = np.zeros(len(y))
        for c in (0, 1):
            if sizes[c]:
                m = np.bincount(rng.integers(0, sizes[c], sizes[c]), minlength=sizes[c])
                counts[cls[c]] = m[gid[cls[c]]]
        for k, f in stat_fns.items():
            v = f(counts)
            if v is not None and np.isfinite(v):
                vals[k].append(v)
    out = {}
    for k, v in vals.items():
        v = np.asarray(v, float)
        out[k] = ((float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)), len(v)) if len(v)
                  else (np.nan, np.nan, 0))
    return out


def cliffs_delta(a, b):
    """Eq. (97): P(a > b) - P(a < b); magnitude labels after Romano et al. (0.147 / 0.33 / 0.474)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    d = (gt - lt) / (len(a) * len(b))
    mag = 'negligible' if abs(d) < 0.147 else 'small' if abs(d) < 0.33 else 'medium' if abs(d) < 0.474 else 'large'
    return float(d), mag


def paired_delta_effect(a, b):
    """Paired effect: Cliff's delta of the paired differences against zero (sign dominance), median difference."""
    d = np.asarray(a, float) - np.asarray(b, float)
    dom = (np.sum(d > 0) - np.sum(d < 0)) / len(d)
    return float(dom), float(np.median(d))


def wilcoxon(a, b):
    """Eq. (95): paired two-sided Wilcoxon signed-rank test (zero differences dropped, Pratt off)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a, b):
        return 0.0, 1.0
    r = st.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    return float(r.statistic), float(r.pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def friedman_nemenyi(M):
    """M: blocks x models matrix of a higher-is-better score. Returns chi2_F, p, mean ranks, CD (Eq. 96)."""
    M = np.asarray(M, float); N, k = M.shape
    ranks = np.apply_along_axis(lambda r: st.rankdata(-r), 1, M)              # rank 1 = best
    R = ranks.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    p = 1 - st.chi2.cdf(chi2, k - 1)
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * N))
    return {'chi2_F': float(chi2), 'p': float(p), 'mean_ranks': R, 'CD': float(cd), 'N_blocks': N, 'k': k}


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T
    return out


def delong(y, s1, s2):
    """Fast DeLong (Sun & Xu 2014) for two correlated AUCs on the same test rows: (auc1, auc2, z, p)."""
    y = np.asarray(y); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.concatenate([s[pos], s[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc); v10.append((tz[:m] - tx) / n); v01.append(1 - (tz[m:] - ty) / m)
    S10, S01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * st.norm.sf(abs(z)))


def mcnemar(y, pred1, pred2):
    """Exact McNemar test on discordant decisions of two classifiers on the same rows."""
    c1 = np.asarray(pred1) == np.asarray(y); c2 = np.asarray(pred2) == np.asarray(y)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(st.binomtest(min(b, c), b + c, 0.5).pvalue)


def kendall(rank_a, rank_b):
    r = st.kendalltau(rank_a, rank_b)
    return float(r.statistic), float(r.pvalue)


def spearman(a, b):
    r = st.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)

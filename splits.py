"""splits.py - protocol ladder (S3 constraint 2) with executable leakage gates A0-A5 (S7 M2; S8 Eqs. 5-9).

Every generator yields Split objects holding row indices into the analysis arrays of one dataset:
  train (fit), val (calibration/threshold, never used for fitting), test (evaluation only).
Rungs: 'random' (row-level stratified K-fold = literature protocol), 'grouped' (StratifiedGroupKFold on sample
identity), 'lofo' (leave-one-family-out), 'single_family' (train on one family, Madamidola et al.), 'temporal'
(TESSERACT-aware: C1 always; C2 when both classes are dated; C3 applied at evaluation through prevalence weights).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    rung: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    info: dict = field(default_factory=dict)


class LeakageError(AssertionError):
    """Raised when a split violates a gate; the run aborts instead of reporting a biased number (S7 M1)."""


# ------------------------------------------------------------------------------------------------ helpers
def _carve_val(idx, y, groups, frac, seed):
    """Group-intact, class-stratified validation slice of a training index set (no time information)."""
    if frac <= 0:
        return idx, np.array([], int)
    if groups is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        a, b = next(sss.split(idx, y[idx]))
    else:
        n = max(2, int(round(1 / frac)))
        sgk = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        a, b = next(sgk.split(idx, y[idx], groups[idx]))
    return idx[a], idx[b]


def check_split(sp, y, groups=None, hashes=None, t=None, require_group_disjoint=False,
                require_hash_disjoint=False, require_temporal=False, window=None, parent_groups=None):
    """Gates A0-A5; returns a dict of measured overlaps and raises LeakageError on a required violation.
    A5 (with parent_groups and require_group_disjoint): no SMOTE child sits in a different partition from the
    groups of its two genuine parents (see purge_smote_straddlers)."""
    fit = np.union1d(sp.train, sp.val)
    res = {'A0_index_overlap': int(len(np.intersect1d(fit, sp.test)))}
    if res['A0_index_overlap']:
        raise LeakageError(f'{sp.rung}: train/test index overlap')
    if groups is not None:                                                     # A1 sample identity (Eq. 8)
        ov = set(groups[fit]) & set(groups[sp.test])
        res['A1_group_overlap'] = len(ov)
        if require_group_disjoint and ov:
            raise LeakageError(f'{sp.rung}: {len(ov)} groups on both sides')
    if hashes is not None:                                                     # A2 exact duplicates (Eq. 9)
        ov = set(hashes[fit]) & set(hashes[sp.test])
        res['A2_hash_overlap'] = len(ov)
        if require_hash_disjoint and ov:
            raise LeakageError(f'{sp.rung}: {len(ov)} identical vectors on both sides')
    if t is not None and require_temporal:                                     # A3 = TESSERACT C1 (Eq. 6)
        tf, tt = t[fit], t[sp.test]
        tf, tt = tf[~np.isnan(tf)], tt[~np.isnan(tt)]
        res['A3_max_train_t'] = float(tf.max()) if len(tf) else None
        res['A3_min_test_t'] = float(tt.min()) if len(tt) else None
        if len(tf) and len(tt) and not tf.max() < tt.min():
            raise LeakageError(f'{sp.rung}: training data not strictly earlier than test data')
        if window is not None and len(tt):                                     # A4 = C2 single window
            lo, hi = window
            res['A4_outside_window'] = int(((tt < lo) | (tt >= hi)).sum())
            if res['A4_outside_window']:
                raise LeakageError(f'{sp.rung}: test rows outside their window')
    if parent_groups is not None and groups is not None and require_group_disjoint:   # A5 SMOTE parents (C1)
        n_bad = len(_straddlers(sp, groups, parent_groups)[0])                # (row-level rungs split every group,
        res['A5_smote_straddlers'] = n_bad                                    #  so A5 is defined on grouped rungs)
        if n_bad:
            raise LeakageError(f'{sp.rung}: {n_bad} SMOTE children straddle their parents\' partitions')
    return res


def _straddlers(sp, groups, parent_groups):
    """Rows (positions into the analysis arrays) whose known SMOTE parents' groups lie in another partition.
    parent_groups: object array (n, 2), None where the row is not an explained SMOTE child."""
    part = {}
    for name, idx in (('train', sp.train), ('val', sp.val), ('test', sp.test)):
        for g in set(groups[idx]):
            part.setdefault(g, set()).add(name)
    bad, where = [], []
    for name, idx in (('train', sp.train), ('val', sp.val), ('test', sp.test)):
        for r in idx:
            pg = parent_groups[r]
            if pg is None or pg[0] is None:
                continue
            others = set().union(*(part.get(g, set()) for g in pg))
            if others - {name}:
                bad.append(r); where.append(name)
    return np.asarray(bad, int), np.asarray(where, object)


def purge_smote_straddlers(sp, groups, parent_groups):
    """Remove explained SMOTE children whose two genuine parents' groups fall in a different partition.

    In the SMOTE-balanced release a synthetic benign row is x_a + lam (x_b - x_a) of two genuine rows; it is
    grouped with its nearest genuine row, but its second parent can sit in another pseudo-group. Merging both
    parents' groups is infeasible on CIC-MalMem-2022 (measured in S9: the union chains 62% of genuine rows into
    one component), so straddling children are purged per split instead (train, validation and test alike).
    Children whose parents are unexplained (no interpolation pair found) cannot be checked and are kept."""
    bad, where = _straddlers(sp, groups, parent_groups)
    for name in ('train', 'val', 'test'):
        drop = bad[where == name]
        sp.info[f'smote_purged_{name}'] = int(len(drop))
        if len(drop):
            setattr(sp, name, np.setdiff1d(getattr(sp, name), drop))
    return sp


def near_duplicate_keep(X_fit, X_test, theta, chunk=4096):
    """Eq. (9): keep test rows whose nearest training row (z-space fitted on training rows) is farther than theta."""
    sc = StandardScaler().fit(X_fit)
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(sc.transform(X_fit))
    d = np.concatenate([nn.kneighbors(sc.transform(X_test[i:i + chunk]))[0][:, 0]
                        for i in range(0, len(X_test), chunk)])
    return d > theta, d


def drop_hash_duplicates(sp, hashes):
    """Remove test rows whose exact vector also occurs in train/val (A2 enforced by construction)."""
    fit_h = set(hashes[np.union1d(sp.train, sp.val)])
    keep = np.array([h not in fit_h for h in hashes[sp.test]])
    sp.info['test_dropped_exact_dups'] = int((~keep).sum())
    sp.test = sp.test[keep]
    return sp


def purge_groups(train, val, test, groups):
    """Remove from train/val every row whose group also occurs in test (19 MalMem hashes carry two family
    labels, so a family hold-out would otherwise leak those binaries)."""
    tg = set(groups[test])
    keep = lambda idx: idx[~np.isin(groups[idx], list(tg))]
    return keep(train), keep(val)


# ------------------------------------------------------------------------------------------------ rungs
def random_kfold(y, n_folds, seed, val_frac=0.2):
    """Row-level stratified K-fold: the protocol most published CIC-MalMem-2022 results use (literature rung)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = _carve_val(tr, y, None, val_frac, seed + 101 * k)
        yield Split('random', tr, va, te, {'fold': k, 'seed': seed})


def grouped_kfold(y, groups, n_folds, seed, val_frac=0.2):
    """StratifiedGroupKFold on sample identity: all dumps of one binary (or one benign pseudo-group) together."""
    sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups)):
        tr, va = _carve_val(tr, y, groups, val_frac, seed + 101 * k)
        yield Split('grouped', tr, va, te, {'fold': k, 'seed': seed})


def leave_one_family_out(y, family, groups, seed, n_benign_folds=5, val_frac=0.2, key_name='family'):
    """Hold out one malware family (or category) per fold; benign rows rotate through grouped folds so that every
    benign group is tested once per n_benign_folds families and never trains the fold that tests it."""
    ben = np.flatnonzero(y == 0)
    sgk = StratifiedGroupKFold(n_splits=n_benign_folds, shuffle=True, random_state=seed)
    ben_folds = [ben[te] for _, te in sgk.split(ben, np.zeros(len(ben)), groups[ben])]
    fams = sorted(pd.unique(family[y == 1]))
    for k, f in enumerate(fams):
        b_te = ben_folds[k % n_benign_folds]
        te = np.concatenate([np.flatnonzero((y == 1) & (family == f)), b_te])
        tr = np.setdiff1d(np.arange(len(y)), te)
        tr = tr[~((y[tr] == 1) & (family[tr] == f))]
        tr, _ = purge_groups(tr, np.array([], int), te, groups)               # multi-label binaries stay out
        tr, va = _carve_val(tr, y, groups, val_frac, seed + 101 * k)
        yield Split('lofo', tr, va, np.sort(te), {key_name: f, 'fold': k, 'seed': seed})


def single_family_train(y, family, groups, seed, val_frac=0.2, test_frac_benign=0.3):
    """Madamidola et al. protocol: train on one family + benign, test on every other family + held-out benign."""
    ben = np.flatnonzero(y == 0)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac_benign, random_state=seed)
    b_tr, b_te = next(gss.split(ben, groups=groups[ben])); b_tr, b_te = ben[b_tr], ben[b_te]
    for k, f in enumerate(sorted(pd.unique(family[y == 1]))):
        tr = np.concatenate([np.flatnonzero((y == 1) & (family == f)), b_tr])
        te = np.concatenate([np.flatnonzero((y == 1) & (family != f)), b_te])
        te = te[~np.isin(groups[te], list(set(groups[tr])))]                  # a binary never on both sides
        tr, va = _carve_val(tr, y, groups, val_frac, seed + 101 * k)
        yield Split('single_family', tr, va, np.sort(te), {'family_trained': f, 'fold': k, 'seed': seed})


def month_windows(tau, delta, K, counts, m_min):
    """Eq. (5): windows [tau+(k-1)Delta, tau+k*Delta); forward-merge windows with < m_min malware rows,
    and merge a short final window backwards. counts: Series of malware rows indexed by month index t."""
    edges = [tau + k * delta for k in range(K + 1)]
    win = [[edges[k], edges[k + 1]] for k in range(K)]
    n = lambda w: int(counts[(counts.index >= w[0]) & (counts.index < w[1])].sum())
    k = 0
    while k < len(win) - 1:                                                    # forward merge
        if n(win[k]) < m_min:
            win[k + 1][0] = win[k][0]; win.pop(k)
        else:
            k += 1
    if len(win) > 1 and n(win[-1]) < m_min:                                    # backward merge of the tail
        win[-2][1] = win[-1][1]; win.pop()
    return [tuple(w) for w in win], [n(w) for w in win]


def temporal_malware_side(y, t_row, groups, tau, delta, K, m_min, seed, val_frac=0.2,
                          benign_train_frac=0.5, benign_val_frac=0.1, benign_test='shared',
                          benign_group_ids=None):
    """Temporal protocol when only malware is dated (CIC-MalMem-2022): C1 for malware, C2 impossible.

    Malware: training era t < tau; validation = the latest val_frac of training-era rows, cut at whole binaries;
    test windows from month_windows. Benign (undated): groups permuted with a fixed seed and partitioned into
    train / validation / test pools. benign_test='shared' evaluates every window against the same benign test pool
    (FPR is time-invariant by construction, so window differences isolate malware-side decay); 'disjoint' gives
    each window its own pool (S8 Eq. 7). Undated malware rows are excluded (reported by the caller)."""
    mal = np.flatnonzero((y == 1) & ~np.isnan(t_row))
    tr_m = mal[t_row[mal] < tau]
    gt = pd.Series(t_row[tr_m]).groupby(groups[tr_m]).max().sort_values()     # binaries ordered by time
    cum = pd.Series(t_row[tr_m]).groupby(groups[tr_m]).size().reindex(gt.index).cumsum()
    cut = cum.iloc[-1] * (1 - val_frac)
    val_groups = set(gt.index[cum.to_numpy() > cut])
    va_m = tr_m[np.isin(groups[tr_m], list(val_groups))]
    tr_m = tr_m[~np.isin(groups[tr_m], list(val_groups))]
    counts = pd.Series(1, index=t_row[mal]).groupby(level=0).sum()
    windows, n_w = month_windows(tau, delta, K, counts, m_min)
    ben = np.flatnonzero(y == 0)
    bg = benign_group_ids if benign_group_ids is not None else groups[ben]
    ug = np.unique(bg); rng = np.random.default_rng(seed); rng.shuffle(ug)
    n_tr, n_va = int(round(len(ug) * benign_train_frac)), int(round(len(ug) * benign_val_frac))
    pools = {'train': set(ug[:n_tr]), 'val': set(ug[n_tr:n_tr + n_va])}
    rest = ug[n_tr + n_va:]
    b_tr = ben[np.isin(bg, list(pools['train']))]; b_va = ben[np.isin(bg, list(pools['val']))]
    if benign_test == 'shared':
        b_te_list = [ben[np.isin(bg, rest)]] * len(windows)
    else:
        parts = np.array_split(rest, len(windows))
        b_te_list = [ben[np.isin(bg, p)] for p in parts]
    train = np.concatenate([tr_m, b_tr]); val = np.concatenate([va_m, b_va])
    for k, (w, nw) in enumerate(zip(windows, n_w)):
        te_m = mal[(t_row[mal] >= w[0]) & (t_row[mal] < w[1])]
        yield Split('temporal', np.sort(train), np.sort(val), np.sort(np.concatenate([te_m, b_te_list[k]])),
                    {'window': k, 'window_months': w, 'n_malware_window': nw, 'seed': seed,
                     'benign_test_mode': benign_test, 'tau': tau, 'delta': delta})


def temporal_both_dated(y, ym, ts_order, train_months, test_months, val_frac=0.2):
    """Full TESSERACT protocol (BODMAS, EMBER): C1 and C2 hold for both classes; C3 applied at evaluation.

    ts_order: sortable time key per row (exact timestamp if available, else month); validation = the latest
    val_frac of training rows by that key (still inside the training period)."""
    tr = np.flatnonzero(np.isin(ym, train_months))
    order = tr[np.argsort(ts_order[tr], kind='mergesort')]
    n_va = int(round(len(order) * val_frac))
    if n_va < 1 or n_va >= len(order):                                        # order[-0:] would select everything
        raise ValueError(f'temporal_both_dated: {len(order)} training rows cannot give a validation slice')
    val, train = np.sort(order[-n_va:]), np.sort(order[:-n_va])
    for k, m in enumerate(test_months):
        te = np.flatnonzero(ym == m)
        yield Split('temporal', train, val, te, {'window': k, 'month': m})


def random_contrast(y, pool_idx, n_train, n_val, seed):
    """Time-agnostic contrast for a temporal protocol: same pool, same train/val sizes, rows drawn at random."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool_idx)
    tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    return Split('random_contrast', np.sort(tr), np.sort(va), np.sort(te), {'seed': seed})


def prevalence_subsample(y_test, pi, rng, min_pos=1):
    """Eq. (7) empirical replicate at prevalence pi: keep all benign, draw malware without replacement; if
    malware is too scarce, downsample benign instead. Returns positions into y_test, or None if < min_pos malware."""
    pos, neg = np.flatnonzero(y_test == 1), np.flatnonzero(y_test == 0)
    n_pos = int(round(pi / (1 - pi) * len(neg)))
    if n_pos <= len(pos):
        if n_pos < min_pos:
            return None
        return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), neg]))
    n_neg = int(round(len(pos) * (1 - pi) / pi))
    return np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))

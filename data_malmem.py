"""data_malmem.py - CIC-MalMem-2022 ingestion and the C1 data-validity gate (S7 module M1; S8 Eqs. 1-9).

Measured facts this module encodes (S9 E0/E1): the official CSV holds 58,596 rows (29,298 benign, 29,298 malware);
malware rows are named "<category>-<family>-<hash>-<run>.raw" (2,887 distinct hash strings, median 10 dumps, max 20);
all benign rows are named "Benign" (no sample identity) and their Year/Month are empty in the dated file; malware
rows satisfy 17 arithmetic invariants (avg x count or count / avg integral) that most benign rows violate, the
signature of SMOTE interpolation (Carrier et al. state that the benign class was balanced with SMOTE).
"""
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

from .config import PATHS, GENEALOGY_K
from .io_utils import row_hashes

NAME_RE = re.compile(r'^(?P<category>[A-Za-z]+)-(?P<family>[A-Za-z0-9]+)-(?P<hash>[0-9A-Za-z]+)-(?P<run>\d+)\.raw$')

# Arithmetic invariants of genuinely extracted Volatility aggregates (S9 search over all 15x40 float/int pairs).
# kind='product': avg * count must be integral; kind='ratio': count / avg must be a positive integer (or 0/0).
INVARIANTS = {
    'I1': ('product', 'pslist.avg_threads', 'pslist.nproc'),            # total threads
    'I2': ('product', 'pslist.avg_handlers', 'pslist.nproc'),           # total handles (pslist view)
    'I3': ('ratio', 'handles.nhandles', 'handles.avg_handles_per_proc'),  # processes seen by the handles plugin
    'I4': ('ratio', 'dlllist.ndlls', 'dlllist.avg_dlls_per_proc'),      # processes seen by dlllist
    'I5a': ('ratio', 'ldrmodules.not_in_load', 'ldrmodules.not_in_load_avg'),
    'I5b': ('ratio', 'ldrmodules.not_in_init', 'ldrmodules.not_in_init_avg'),
    'I5c': ('ratio', 'ldrmodules.not_in_mem', 'ldrmodules.not_in_mem_avg'),
    'I6a': ('ratio', 'psxview.not_in_pslist', 'psxview.not_in_pslist_false_avg'),
    'I6b': ('ratio', 'psxview.not_in_ethread_pool', 'psxview.not_in_ethread_pool_false_avg'),
    'I6c': ('ratio', 'psxview.not_in_pspcid_list', 'psxview.not_in_pspcid_list_false_avg'),
    'I6d': ('ratio', 'psxview.not_in_csrss_handles', 'psxview.not_in_csrss_handles_false_avg'),
    'I6e': ('ratio', 'psxview.not_in_session', 'psxview.not_in_session_false_avg'),
    'I6f': ('ratio', 'psxview.not_in_deskthrd', 'psxview.not_in_deskthrd_false_avg'),
    'I7': ('ratio', 'malfind.ninjections', 'malfind.uniqueInjections'),
}
INVARIANT_SETS = {'min': ['I1', 'I2'],                     # the two checks used in the S3 probe
                  'core': ['I1', 'I2', 'I3'],              # primary set: the three strongest discriminators
                  'full': list(INVARIANTS)}                # every check (strictest notion of "genuine")

MONTHS = {'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3, 'marh': 3, 'apr': 4, 'april': 4,
          'arpil': 4, 'may': 5, 'msy': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
          'sep': 9, 'sept': 9, 'september': 9, 'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'novembe': 11,
          'novembber': 11, 'nomver': 11, 'dec': 12, 'december': 12, 'dember': 12, 'devember': 12, 'dececmber': 12}
# last eight keys repair spelling errors measured in the dated file ('Marh', 'Arpil', 'Msy', 'Novembber', ...)


def plugin_blocks(features):
    """Nine Volatility plugin blocks x^(b) (S8 Eq. 2), derived from the column prefixes."""
    blocks = {}
    for f in features:
        blocks.setdefault(f.split('.')[0], []).append(f)
    return blocks


def load_malmem(path=None):
    """Read the official CSV and return (X, meta): X = 55 float64 features; meta = label, identity and names."""
    df = pd.read_csv(path or PATHS['malmem'])
    feats = [c for c in df.columns if c not in ('Category', 'Class')]          # 55 Volatility aggregates
    X = df[feats].astype(np.float64)                                           # single dtype (S7 M1 step 2)
    meta = pd.DataFrame({'row_id': np.arange(len(df)),
                         'y': (df['Class'] == 'Malware').astype(int).values,  # 1 = malware (S8 Eq. 1)
                         'name': df['Category'].values})
    parsed = meta['name'].str.extract(NAME_RE)                                 # category / family / hash / run
    meta = pd.concat([meta, parsed], axis=1)
    meta.loc[meta['y'] == 0, ['category', 'family']] = 'Benign'
    meta['run'] = pd.to_numeric(meta['run'], errors='coerce')
    bad = meta['y'].eq(1) & meta['hash'].isna()
    if bad.any():                                                              # never silently drop identity
        raise ValueError(f'{int(bad.sum())} malware names failed to parse')
    meta['hash_len'] = meta['hash'].str.len()
    return X, meta


def invariant_table(X, tol):
    """Boolean pass/fail per row for every invariant at absolute tolerance tol; NaN-free by construction."""
    out = {}
    for name, (kind, a, b) in INVARIANTS.items():
        va, vb = X[a].to_numpy(), X[b].to_numpy()
        if kind == 'product':                                                  # avg * count integral
            v = va * vb
            out[name] = np.abs(v - np.round(v)) <= tol
        else:                                                                  # count / avg positive integer
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(vb > 0, va / vb, np.nan)
            ok_pos = (vb > 0) & (np.abs(r - np.round(r)) <= tol) & (np.round(r) >= 1)
            ok_zero = (vb == 0) & (va == 0)                                    # 0/0: nothing to aggregate
            out[name] = ok_pos | ok_zero
    return pd.DataFrame(out, index=X.index)


def validity_gate(X, meta, invariant_set='core', tol=1e-3):
    """C1 gate: flag benign rows that violate any invariant of the chosen set as synthetic (SMOTE) rows.

    Returns a dict of boolean masks plus counts. Malware rows are checked with the same test; a malware failure
    rate above zero would falsify the invariant (it is reported, never used to drop malware)."""
    tab = invariant_table(X, tol)[INVARIANT_SETS[invariant_set]]
    passes = tab.all(axis=1).to_numpy()
    is_mal = meta['y'].to_numpy() == 1
    res = {'genuine_benign': (~is_mal) & passes, 'synthetic_benign': (~is_mal) & ~passes,
           'malware_pass': is_mal & passes, 'malware_fail': is_mal & ~passes}
    res['counts'] = {k: int(v.sum()) for k, v in res.items()}
    res['counts'].update({'invariant_set': invariant_set, 'tol': tol})
    return res


def constant_features(X, near=0.999):
    """Global constants and near-constants (dominant value share >= near); reported, filtered inside folds."""
    const = [c for c in X.columns if X[c].nunique() == 1]
    nearc = {c: float(X[c].value_counts(normalize=True).iloc[0]) for c in X.columns
             if c not in const and X[c].value_counts(normalize=True).iloc[0] >= near}
    return const, nearc


def exact_duplicates(X, meta):
    """Exact-duplicate groups on the rounded vector: sizes, redundant copies per class, cross-class conflicts."""
    h = row_hashes(X.to_numpy())
    d = pd.DataFrame({'h': h, 'y': meta['y'].to_numpy()})
    sizes = d.groupby('h').size()
    dup_h = sizes[sizes > 1].index
    conflicts = int((d[d['h'].isin(dup_h)].groupby('h')['y'].nunique() > 1).sum())
    first = ~d.duplicated('h')                                                # keep first occurrence
    return {'hash': h, 'keep_first': first.to_numpy(),
            'n_rows_in_dup_groups': int(d['h'].isin(dup_h).sum()), 'n_dup_groups': int(len(dup_h)),
            'redundant_copies': int((~first).sum()),
            'redundant_by_class': d[~first].groupby('y').size().to_dict(), 'cross_class_conflicts': conflicts}


def sibling_structure(meta):
    """Sample-identity structure of the malware class (A3 leakage): dumps per binary, multi-family hashes."""
    m = meta[meta['y'] == 1]
    per = m.groupby('hash').size()
    fam = m.groupby('hash')['family'].nunique()
    runs = m.groupby(['hash', 'run']).size()
    return {'n_binaries': int(per.size), 'dumps_median': float(per.median()), 'dumps_max': int(per.max()),
            'dumps_hist': per.value_counts().sort_index().to_dict(),
            'hashes_multi_family': int((fam > 1).sum()), 'hash_run_pairs_repeated': int((runs > 1).sum()),
            'hash_length_rows': m['hash_len'].value_counts().to_dict(),
            'binaries_per_family': m.groupby('family')['hash'].nunique().to_dict()}


def genealogy(X, genuine, synthetic, k=GENEALOGY_K, int_cols=None, tol_float=1e-6, max_rows=None, seed=0):
    """Test whether each synthetic benign row is a SMOTE interpolant x_a + lam (x_b - x_a) of two genuine rows.

    lam is estimated by least squares on the float (average) columns, which are stored to 8 decimals; the
    interpolation must reproduce every float column within tol_float (relative to the column scale) and every
    integer column within 1 count (rounding or truncation after interpolation). Parents are searched among the
    k nearest genuine rows in z-space (scaler fitted on genuine rows only)."""
    cols = list(X.columns)
    int_cols = int_cols or [c for c in cols if np.allclose(X[c], np.round(X[c]))]
    flt_cols = [c for c in cols if c not in int_cols]
    G = X.loc[genuine].to_numpy(); S_idx = np.flatnonzero(synthetic)
    if max_rows and len(S_idx) > max_rows:                                     # optional fixed-seed subsample
        S_idx = np.sort(np.random.default_rng(seed).choice(S_idx, max_rows, replace=False))
    S = X.to_numpy()[S_idx]
    sc = StandardScaler().fit(G)
    nn = NearestNeighbors(n_neighbors=min(k, len(G))).fit(sc.transform(G))
    _, nbr = nn.kneighbors(sc.transform(S))
    fi = [cols.index(c) for c in flt_cols]; ii = [cols.index(c) for c in int_cols]
    scale = np.maximum(np.abs(G[:, fi]).max(axis=0), 1e-12)                    # per-column scale for tolerances
    pairs = np.array([(i, j) for i in range(nbr.shape[1]) for j in range(i + 1, nbr.shape[1])])
    found = np.zeros(len(S), bool); lam_out = np.full(len(S), np.nan); parents = np.full((len(S), 2), -1)
    gidx = np.flatnonzero(genuine)
    for r in range(len(S)):
        A = G[nbr[r, pairs[:, 0]]]; B = G[nbr[r, pairs[:, 1]]]; s = S[r]
        dF = (B - A)[:, fi]; num = ((s[fi] - A[:, fi]) * dF).sum(1); den = (dF ** 2).sum(1)
        lam = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)      # least-squares lambda per pair
        P = A + lam[:, None] * (B - A)                                          # interpolated candidate
        ok = (np.isfinite(lam) & (lam >= -1e-9) & (lam <= 1 + 1e-9)
              & (np.abs(P[:, fi] - s[fi]) / scale <= tol_float).all(1)
              & (np.abs(P[:, ii] - s[ii]) <= 1.0 + 1e-9).all(1))
        if ok.any():
            j = int(np.flatnonzero(ok)[0]); found[r] = True; lam_out[r] = lam[j]
            parents[r] = gidx[nbr[r, pairs[j]]]
    return {'rows_tested': int(len(S)), 'explained': int(found.sum()), 'share_explained': float(found.mean()),
            'lambda': lam_out[found], 'parents': parents[found], 'synthetic_row_index': S_idx[found],
            'n_float_cols': len(fi), 'n_int_cols': len(ii), 'k': k}


def identity_scaler(X, rows):
    """Label-free z-scaler shared by the sibling radius and the benign graph (fitted on the analysis rows)."""
    return StandardScaler().fit(X.to_numpy()[rows])


def sibling_radius(X, meta, scaler, mask=None):
    """Median z-distance from a malware dump to its nearest sibling dump of the same binary.

    Used as the linkage radius for benign pseudo-groups, so benign 'identity' is as coarse as malware identity.
    Distances are measured with the same label-free scaler that the benign graph uses."""
    m = (meta['y'] == 1).to_numpy() if mask is None else mask
    Z = scaler.transform(X.to_numpy()[m]); hs = meta['hash'].to_numpy()[m]
    d = []
    for h in np.unique(hs):
        idx = np.flatnonzero(hs == h)
        if len(idx) < 2:
            continue
        Zi = Z[idx]; D = np.sqrt(((Zi[:, None, :] - Zi[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
        d.extend(D.min(1))
    return float(np.median(d)), np.asarray(d)


def benign_pseudo_groups(Xb, radius, scaler):
    """Connected components of the radius-graph over benign rows in the shared identity z-space.

    Label-free, so computing it before the split leaks no label information (it plays the role of the hash)."""
    Z = scaler.transform(Xb)
    nn = NearestNeighbors(radius=radius).fit(Z)
    G = nn.radius_neighbors_graph(Z, mode='connectivity')
    n, lab = connected_components(G, directed=False)
    return lab, n


def normalise_month(s):
    """Map the free-text Month field (with measured misspellings) to 1..12; NaN if unknown."""
    return pd.Series(s, dtype='string').str.strip().str.lower().map(MONTHS)


def load_dated(meta, X):
    """Attach Year, Month and the monthly index t (S8 Eq. 3) to MalMem malware rows via the dated file.

    Join key: full sample name + rounded feature hash (the dated file re-annotates MalMem rows). Returns a frame
    aligned to meta with columns year, month, t, dated, conflict_binary (binary carries >1 year)."""
    D = pd.read_csv(PATHS['memmal_dated_mal'])
    D.columns = [c.strip() for c in D.columns]                                 # 'Year ' -> 'Year'
    feats = list(X.columns)
    D['month_n'] = normalise_month(D['Month'])
    if D['month_n'].isna().any():
        raise ValueError(f"unmapped months: {D.loc[D['month_n'].isna(), 'Month'].unique()}")
    keyD = D['Category'] + '|' + pd.Series(row_hashes(D[feats].astype(float).to_numpy(), 6))
    keyM = meta['name'] + '|' + pd.Series(row_hashes(X.to_numpy(), 6))
    ym = (D.assign(key=keyD).groupby('key')[['Year', 'month_n']]
          .agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan))           # conflicting duplicate keys -> NaN
    out = pd.DataFrame(index=meta.index)
    out['year'] = keyM.map(ym['Year']).to_numpy()
    out['month'] = keyM.map(ym['month_n']).to_numpy()
    out.loc[meta['y'] == 0, ['year', 'month']] = np.nan                        # benign: never dated
    out['t'] = 12 * (out['year'] - 2006) + out['month']                       # S8 Eq. 3, 1..192
    yrs = pd.DataFrame({'hash': meta['hash'], 'year': out['year']}).dropna().groupby('hash')['year'].nunique()
    out['conflict_binary'] = meta['hash'].map(yrs.gt(1)).fillna(False).astype(bool).to_numpy()
    out['dated'] = out['t'].notna() & ~out['conflict_binary']
    return out


def binary_time(meta, dated):
    """Group-level time stamp: earliest dated month of each binary (conflicting-year binaries excluded)."""
    d = pd.DataFrame({'hash': meta['hash'], 't': dated['t'], 'ok': dated['dated']})
    return d[d['ok']].groupby('hash')['t'].min()

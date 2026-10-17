"""common.py - analysis-set builders and the generic grid runner shared by experiments E1-E10.

Run every experiment as  python3 -I experiments/eX_*.py --mode smoke|full  (CPU only). -I keeps the data
directories off sys.path; the harness root is added explicitly below.
"""
import sys, argparse, json, time
from dataclasses import dataclass, field
from pathlib import Path
sys.path.insert(0, '/home/claude/s9')

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from harness import config as C
from harness import data_malmem as dm, splits as S, evaluate as E
from harness.io_utils import (set_seeds, TimingLog, environment, outdir, save_json, append_rows, row_hashes)


def parse(desc, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--models', nargs='*', default=None)
    for args, kw in (extra or []):
        ap.add_argument(*args, **kw)
    a = ap.parse_args()
    a.seeds = tuple(a.seeds) if a.seeds else (C.SEEDS_SMOKE if a.mode == 'smoke' else C.SEEDS_FULL)
    a.B = C.B_BOOT_SMOKE if a.mode == 'smoke' else C.B_BOOT
    return a


@dataclass
class AnalysisSet:
    name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    family: np.ndarray
    features: list
    hashes: np.ndarray
    row_id: np.ndarray
    t: np.ndarray = None                    # month index (NaN when undated)
    category: np.ndarray = None
    info: dict = field(default_factory=dict)
    parent_groups: np.ndarray = None        # (n, 2) groups of the two SMOTE parents (explained children only)


def malmem_set(pool='genuine', dedup=True, invariant_set=C.PRIMARY_INVARIANT_SET, tol=C.PRIMARY_TOLERANCE):
    """CIC-MalMem-2022 analysis set.

    pool: 'all' keeps every benign row (literature reproduction, ~94% synthetic); 'genuine' keeps benign rows that
    pass the invariant gate. Groups: malware -> binary hash; benign -> pseudo-group of the nearest genuine row
    (radius-graph components at the malware sibling radius), so SMOTE children travel with their nearest parent.
    For pool='all', parent_groups records the pseudo-groups of both reconstructed SMOTE parents (dm.genealogy) so
    that grouped rungs can purge children whose second parent falls on the other side (purge_smote_straddlers).
    Identity (groups, hashes, duplicates) is always defined on all 55 features, also when a model later sees a
    feature subset (E1 block ablation, E2 svcscan-removed cells), so paired contrasts share identical splits."""
    X, meta = dm.load_malmem()
    y = meta['y'].to_numpy(); is_mal = y == 1
    gate = dm.validity_gate(X, meta, invariant_set, tol)
    genuine = gate['genuine_benign']
    keep = is_mal | (genuine if pool == 'genuine' else ~is_mal)
    dup = dm.exact_duplicates(X, meta)
    if dedup:
        keep &= dup['keep_first']
    sc = dm.identity_scaler(X, is_mal | genuine)
    radius, _ = dm.sibling_radius(X, meta, sc)
    lab, n_comp = dm.benign_pseudo_groups(X.to_numpy()[genuine], radius, sc)
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(X.to_numpy()[genuine]))
    bj = nn.kneighbors(sc.transform(X.to_numpy()[~is_mal]), return_distance=False)[:, 0]
    bgroup = np.empty(len(y), dtype=object); bgroup[~is_mal] = lab[bj]
    hashes_m = meta['hash'].fillna('').astype(str).tolist()
    groups = np.array([('M:' + h) if m else ('B:' + str(b)) for m, h, b in zip(is_mal, hashes_m, bgroup)],
                      dtype=object)
    pgroups = None
    if pool == 'all':                                                          # SMOTE genealogy -> parents' groups
        gkey = (invariant_set, tol)
        if gkey not in _GENEALOGY:
            _GENEALOGY[gkey] = dm.genealogy(X, genuine, gate['synthetic_benign'], k=C.GENEALOGY_K)
        g = _GENEALOGY[gkey]
        gpos = -np.ones(len(y), int); gpos[np.flatnonzero(genuine)] = np.arange(int(genuine.sum()))
        pgroups = np.empty((len(y), 2), dtype=object)                         # None = not an explained child
        pa = g['parents']
        pgroups[g['synthetic_row_index'], 0] = ['B:' + str(v) for v in lab[gpos[pa[:, 0]]]]
        pgroups[g['synthetic_row_index'], 1] = ['B:' + str(v) for v in lab[gpos[pa[:, 1]]]]
    dated = dm.load_dated(meta, X); bt = dm.binary_time(meta, dated)
    t_row = np.array(meta['hash'].map(bt), dtype=float); t_row[~is_mal] = np.nan     # writable copy (pandas CoW)
    idx = np.flatnonzero(keep)
    info = {'pool': pool, 'dedup': dedup, 'invariant_set': invariant_set, 'tol': tol,
            'gate_counts': gate['counts'], 'sibling_radius': radius, 'benign_pseudo_groups': n_comp,
            'n_rows': int(len(idx)), 'n_malware': int(is_mal[idx].sum()), 'n_benign': int((~is_mal[idx]).sum()),
            'n_malware_dated': int(np.isfinite(t_row[idx]).sum()),
            'exact_dup_removed': int((~dup['keep_first'] & (is_mal | (genuine if pool == 'genuine' else ~is_mal))).sum()) if dedup else 0,
            'smote_children_explained': int(len(_GENEALOGY[(invariant_set, tol)]['parents'])) if pool == 'all' else 0}
    return AnalysisSet(name=f'malmem_{pool}', X=X.to_numpy()[idx], y=y[idx], groups=groups[idx],
                       family=meta['family'].fillna('Benign').to_numpy()[idx], features=list(X.columns),
                       hashes=dup['hash'][idx], row_id=meta['row_id'].to_numpy()[idx], t=t_row[idx],
                       category=meta['category'].fillna('Benign').to_numpy()[idx], info=info,
                       parent_groups=pgroups[idx] if pgroups is not None else None)


_GENEALOGY = {}                                                                # in-process cache (11 s per call)
GROUP_RUNGS = ('grouped', 'lofo', 'loco', 'single_family', 'temporal')


def malmem_cell_splits(ds, rung, seed, n_folds=C.N_FOLDS):
    """Split iterator for one MalMem rung; group-disjoint rungs on the 'all' pool purge SMOTE children whose
    second parent falls in another partition (no-op for the genuine pool, which holds no synthetic rows)."""
    it = _raw_cell_splits(ds, rung, seed, n_folds)
    if rung in GROUP_RUNGS and ds.parent_groups is not None:
        return (S.purge_smote_straddlers(sp, ds.groups, ds.parent_groups) for sp in it)
    return it


def _raw_cell_splits(ds, rung, seed, n_folds=C.N_FOLDS):
    if rung == 'random':
        return S.random_kfold(ds.y, n_folds, seed, C.VAL_FRAC)
    if rung == 'grouped':
        return S.grouped_kfold(ds.y, ds.groups, n_folds, seed, C.VAL_FRAC)
    if rung == 'lofo':
        return S.leave_one_family_out(ds.y, ds.family, ds.groups, seed, n_folds, C.VAL_FRAC)
    if rung == 'loco':                                                         # leave-one-category-out
        return S.leave_one_family_out(ds.y, ds.category, ds.groups, seed, n_folds, C.VAL_FRAC, key_name='category')
    if rung == 'single_family':
        return S.single_family_train(ds.y, ds.family, ds.groups, seed, C.VAL_FRAC)
    if rung == 'temporal':
        tau = 12 * (C.MALMEM_TAU_S[0] - 2006) + C.MALMEM_TAU_S[1]
        return S.temporal_malware_side(ds.y, ds.t, ds.groups, tau, C.MALMEM_DELTA, C.MALMEM_K, C.MALMEM_M_MIN,
                                       seed, C.VAL_FRAC, benign_test='shared')
    raise KeyError(rung)


def apply_dedup(ds, sp, level='exact', measure_near=True):
    """Eq. (9). level='exact': drop test rows identical to a training/validation row (A2 by construction);
    level='near': also drop test rows within THETA_DUP_NEAR (z-space) of a training row; level='none': keep all.
    The near-duplicate share is always measured and stored in sp.info, whatever the level."""
    if level in ('exact', 'near'):
        sp = S.drop_hash_duplicates(sp, ds.hashes)
    if (measure_near or level == 'near') and len(sp.test):
        keep, d = S.near_duplicate_keep(ds.X[np.union1d(sp.train, sp.val)], ds.X[sp.test], C.THETA_DUP_NEAR)
        sp.info['test_near_dup_share'] = float(np.mean(~keep))
        sp.info['test_near_dup_share_malware'] = float(np.mean(~keep[ds.y[sp.test] == 1])) if (ds.y[sp.test] == 1).any() else np.nan
        sp.info['test_near_dup_share_benign'] = float(np.mean(~keep[ds.y[sp.test] == 0])) if (ds.y[sp.test] == 0).any() else np.nan
        if level == 'near':
            sp.info['test_dropped_near_dups'] = int((~keep).sum()); sp.test = sp.test[keep]
    sp.info['dedup_level'] = level
    return sp


def gates_for(rung):
    return {'require_group_disjoint': rung in GROUP_RUNGS, 'require_temporal': rung == 'temporal'}


def gate_split(ds, sp, rung, dedup):
    """Run gates A0-A5 for one MalMem split (A4 uses the window of a temporal split; A5 the SMOTE parents)."""
    return S.check_split(sp, ds.y, groups=ds.groups, hashes=ds.hashes if dedup != 'none' else None, t=ds.t,
                         require_hash_disjoint=dedup != 'none', window=sp.info.get('window_months'),
                         parent_groups=ds.parent_groups, **gates_for(rung))


class ScoreStore:
    """Keeps test-row ids and float64 scores per fit (for curves, DeLong, bootstrap) and writes one .npz.
    Scores stay float64 so stored validation thresholds (nextafter of a float64 score) apply exactly."""

    def __init__(self):
        self.d = {}

    def add(self, key, row_id, y, s, thresholds):
        self.d[key] = {'row_id': np.asarray(row_id, np.int32), 'y': np.asarray(y, np.int8),
                       's': np.asarray(s, np.float64), 'thr': thresholds}

    def save(self, path):
        flat = {}
        for k, v in self.d.items():
            tag = '|'.join(str(x) for x in k)
            flat[tag + '|row_id'] = v['row_id']; flat[tag + '|y'] = v['y']; flat[tag + '|s'] = v['s']
        np.savez_compressed(path, **flat)
        with open(str(path).replace('.npz', '_thresholds.json'), 'w') as fh:
            json.dump({'|'.join(str(x) for x in k): v['thr'] for k, v in self.d.items()}, fh, default=float)

    def flush(self, directory, tag):
        """Write the entries collected so far to <directory>/<tag>.npz (+ thresholds) and clear memory, so a
        container restart loses at most one cell x seed (runs are resumable)."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        if self.d:
            safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in tag)
            self.save(Path(directory) / f'{safe}.npz')
        self.d = {}


def load_scores(exp_dir):
    """Merge scores.npz (legacy single file) and scores/*.npz into {key: (y, s, thresholds, row_id)}."""
    exp_dir = Path(exp_dir); files = []
    if (exp_dir / 'scores.npz').exists():
        files.append(exp_dir / 'scores.npz')
    files += sorted((exp_dir / 'scores').glob('*.npz')) if (exp_dir / 'scores').exists() else []
    out = {}
    for f in files:
        z = np.load(f); thr_p = Path(str(f).replace('.npz', '_thresholds.json'))
        thr = json.load(open(thr_p)) if thr_p.exists() else {}
        for k in sorted({k.rsplit('|', 1)[0] for k in z.files}):
            out[k] = (z[k + '|y'].astype(int), z[k + '|s'].astype(float), thr.get(k, {}), z[k + '|row_id'])
    return out


def done_keys(path, fields):
    """Set of tuples (fields) already present in a results CSV (resume support)."""
    if not Path(path).exists():
        return set()
    d = pd.read_csv(path, low_memory=False)
    for f in fields:
        if f not in d:
            d[f] = 'none'
    d = d[fields].fillna('none').astype(str).replace({'nan': 'none', 'None': 'none'})
    return set(map(tuple, d.drop_duplicates().to_numpy()))


def run_fit_eval(ds, sp, name, seed, cell, store=None, time_key=None, pis=C.PREVALENCES):
    """Fit one model on one split and return metric rows tagged with cell/split metadata."""
    r = E.fit_score(name, ds.X, ds.y, sp, seed, groups=ds.groups, time_key=time_key, feature_names=ds.features)
    rows, calkind = E.evaluate(ds.y[sp.test], r['s_test'], r['s_val'], ds.y[sp.val], r['has_proba'], pis=pis)
    tag = {k: v for k, v in sp.info.items() if not isinstance(v, (list, tuple, dict))}
    if 'window_months' in sp.info:
        tag['window_lo'], tag['window_hi'] = sp.info['window_months']
    base = {**cell, **tag, 'model': name, 'seed': seed, 'fit_s': r['fit_s'], 'us_per_sample': r['us_per_sample'],
            'n_train': len(sp.train), 'n_val': len(sp.val), 'n_test': len(sp.test),
            'n_test_pos': int(ds.y[sp.test].sum()), 'n_test_neg': int((ds.y[sp.test] == 0).sum()),
            'calibrator': calkind, 'stump_feature': r.get('stump_feature')}
    if store is not None:
        thr = {f'{x["rule"]}|{x["alpha"]}|{x["cost_ratio"]}|{x["pi"]}': x['threshold'] for x in rows
               if x['rule'] != 'cost'}                     # cost thresholds live on calibrated logits, not raw scores
        key = tuple([cell.get(k) for k in sorted(cell)] + [seed, sp.info.get('fold', sp.info.get('window')),
                                                            sp.info.get('family', ''), name])
        store.add(key, ds.row_id[sp.test], ds.y[sp.test], r['s_test'], thr)
    return [{**base, **row} for row in rows], r


# ------------------------------------------------------------------------------------------------ static-PE sets
def bodmas_set():
    """BODMAS rows inside the shared window (both classes dated), exact duplicates dropped (first kept).

    X stays a memmap over all 134,435 rows; ineligible rows get ym = 'out' so no split ever selects them."""
    from harness import data_bodmas as dbm
    X, meta = dbm.load_bodmas()
    dup = dbm.duplicate_report(X, meta)
    elig = meta['in_window'].to_numpy() & dup['keep_first']
    ym = np.where(elig, meta['ym'].to_numpy(), 'out')
    order = meta['ts'].astype('int64').to_numpy()                              # exact time for validation cut
    info = {'eligible': int(elig.sum()), 'benign_out_of_window': int(((meta.y == 0) & ~meta.in_window).sum()),
            'exact_dup_dropped': int((~dup['keep_first']).sum()),
            'train_months': C.BODMAS_TRAIN_MONTHS, 'test_months': C.BODMAS_TEST_MONTHS}
    return AnalysisSet(name='bodmas', X=X, y=meta['y'].to_numpy(), groups=meta['sha'].to_numpy(),
                       family=meta['family'].to_numpy(), features=[f'f{j}' for j in range(X.shape[1])],
                       hashes=dup['hash'], row_id=meta['row_id'].to_numpy(), category=meta['category'].to_numpy(),
                       info={**info, 'ym': ym, 'order': order})


def ember_set():
    """EMBER-2018 stratified subsample (2,500 per month x class); validation cut by month then sha256."""
    from harness import data_ember as dem
    X, meta = dem.load_ember()
    hs = row_hashes(np.asarray(X), 7)
    first = ~pd.Series(hs).duplicated().to_numpy()
    ym = np.where(first, meta['ym'].to_numpy(), 'out')
    order = pd.factorize(meta['ym'] + '|' + meta['sha256'], sort=True)[0]
    info = {'eligible': int(first.sum()), 'exact_dup_dropped': int((~first).sum()),
            'train_months': C.EMBER_TRAIN_MONTHS, 'test_months': C.EMBER_TEST_MONTHS}
    return AnalysisSet(name='ember', X=X, y=meta['y'].to_numpy(), groups=meta['sha256'].to_numpy(),
                       family=meta['family'].to_numpy(), features=[f'f{j}' for j in range(X.shape[1])],
                       hashes=hs, row_id=meta['row_id'].to_numpy(), info={**info, 'ym': ym, 'order': order})


def parse_score_key(k):
    """Field dict for a ScoreStore key (cell fields are stored in sorted-name order, then seed, fold/window,
    family, model). E2 keys carry drop_block; E3 keys are (dataset, rung, seed, month, model)."""
    p = k.split('|')
    if len(p) == 9:
        names = ['dataset', 'dedup', 'drop_block', 'pool', 'rung', 'seed', 'fold', 'family', 'model']
    elif len(p) == 8:
        names = ['dataset', 'dedup', 'pool', 'rung', 'seed', 'fold', 'family', 'model']
    elif len(p) == 5:
        names = ['dataset', 'rung', 'seed', 'month', 'model']
    else:                                                                      # E4: dataset, dedup, rung, seed, fold, family, model
        names = ['dataset', 'dedup', 'rung', 'seed', 'fold', 'family', 'model'][:len(p)]
    d = dict(zip(names, p))
    d.setdefault('drop_block', 'none')
    return d

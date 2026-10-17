"""data_bodmas.py - BODMAS ingestion (Yang et al., IEEE SPW/DLS 2021): EMBER-v2 features, timestamps, families.

Measured in S9: X is 134,435 x 2,381 float32 (57,293 malware, 77,142 benign); malware timestamps span
2019-08..2020-09; benign timestamps span 2007..2020 with placeholder-like spikes (e.g. 1,572 files in 2015-01,
~2,100 in 2018-02/03), so only benign files dated inside the malware window are used for temporal work (S5 item 7).
3,712 benign timestamps carry no timezone and are parsed as UTC with format='mixed'.
"""
import numpy as np
import pandas as pd
from .config import PATHS, CACHE, BODMAS_WINDOW
from .io_utils import row_hashes


def _cache_X():
    """Decompress X once to cache/bodmas_X.npy so later loads can memory-map it (7 GB RAM container)."""
    p = CACHE / 'bodmas_X.npy'
    if not p.exists():
        with np.load(PATHS['bodmas_npz']) as d:
            np.save(p, d['X'].astype(np.float32, copy=False))
            np.save(CACHE / 'bodmas_y.npy', d['y'].astype(np.int8))
    return p


def load_bodmas(mmap=True):
    """Return (X, meta). X is a read-only memmap (float32); meta aligns row-for-row with X."""
    X = np.load(_cache_X(), mmap_mode='r' if mmap else None)
    y = np.load(CACHE / 'bodmas_y.npy').astype(int)
    meta = pd.read_csv(PATHS['bodmas_meta'])
    if len(meta) != X.shape[0]:
        raise ValueError('metadata/feature row mismatch')
    meta['y'] = y
    meta['ts'] = pd.to_datetime(meta['timestamp'], utc=True, format='mixed')      # 3,712 rows lack a timezone
    meta['ym'] = meta['ts'].dt.strftime('%Y-%m')
    cat = pd.read_csv(PATHS['bodmas_cat']).rename(columns={'sha256': 'sha'})
    meta = meta.merge(cat, on='sha', how='left')                                   # malware category (14 values)
    meta['family'] = meta['family'].where(meta['y'] == 1, 'benign')
    meta['category'] = meta['category'].where(meta['y'] == 1, 'benign')
    lo, hi = BODMAS_WINDOW
    meta['in_window'] = (meta['ym'] >= lo) & (meta['ym'] <= hi)                    # both classes share the window
    meta['row_id'] = np.arange(len(meta))
    return X, meta


def duplicate_report(X, meta, block=20000):
    """Exact duplicate feature vectors (within/cross class); hashing in blocks keeps memory flat."""
    hs = np.concatenate([row_hashes(X[i:i + block], 7) for i in range(0, X.shape[0], block)])
    d = pd.DataFrame({'h': hs, 'y': meta['y'].to_numpy()})
    sizes = d.groupby('h').size(); dup = sizes[sizes > 1].index
    conflicts = int((d[d['h'].isin(dup)].groupby('h')['y'].nunique() > 1).sum())
    first = ~d.duplicated('h')
    return {'hash': hs, 'keep_first': first.to_numpy(), 'n_dup_groups': int(len(dup)),
            'redundant_copies': int((~first).sum()), 'redundant_by_class': d[~first].groupby('y').size().to_dict(),
            'cross_class_conflicts': conflicts}


def window_counts(meta):
    """Monthly class counts inside and outside the shared window (provenance table)."""
    t = pd.crosstab(meta['ym'], meta['y'])
    t.columns = ['benign', 'malware'][:t.shape[1]]
    return t

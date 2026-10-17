#!/usr/bin/env python3
"""make_ember_subsample.py -- S9 harness, EMBER-2018 stratified temporal subsample (runs on CPU, pandas only).
Pass 1 (--index): scan every EMBER-2018 raw-feature CSV for (sha256, appeared, label) and record the row number.
Pass 2 (--extract FILE): copy the selected rows of one source file into a gzip CSV shard.
Design: labelled rows only (label in {0,1}; EMBER's unlabelled label=-1 rows are excluded); months 2018-01..2018-12;
N_PER_CELL rows per (month, label) cell drawn uniformly without replacement with numpy default_rng(SEED)
after sorting candidates by sha256 (so the draw is independent of file order).
Executed in S9 on the author's computer (2 vCPU, 3 GB RAM VM): index 1,000,000 rows (400k benign, 400k malware,
200k unlabelled), selected 60,000 rows (2,500 per month x class), shards = 143 MB gzip. Usage:
  python3 make_ember_subsample.py --index train_features_0.csv ... test_features.csv
  python3 make_ember_subsample.py --select
  python3 make_ember_subsample.py --extract train_features_1.csv     (repeat per file)
"""
import sys, os, gzip, time
import numpy as np, pandas as pd
SRC = os.path.expanduser('~/mnt/Datasets/ember2018')              # author's EMBER-2018 raw-feature CSV conversion
FILES = ['train_features_%d.csv' % i for i in range(6)] + ['test_features.csv']
IDX = os.path.expanduser('~/s9_scratch/ember_index.csv')          # pass-1 index (scratch, outside the user's folder)
SEL = os.path.expanduser('~/s9_scratch/ember_selected.csv')       # selected (file, row) pairs
OUT = os.path.join(SRC, '_S9_shards')                              # derived shards, staged to the cloud later
MONTHS = ['2018-%02d' % m for m in range(1, 13)]
N_PER_CELL, SEED = 2500, 20181016


def index_file(fname):                                             # pass 1 for one file
    rows = []; start = 0
    for ch in pd.read_csv(os.path.join(SRC, fname), usecols=['sha256', 'appeared', 'label'], chunksize=50000):
        ch = ch.assign(file=fname, row=np.arange(start, start + len(ch))); start += len(ch)
        rows.append(ch)
    df = pd.concat(rows)
    df.to_csv(IDX, mode='a', header=not os.path.exists(IDX), index=False)
    return len(df)


def select():                                                       # stratified draw over the full index
    df = pd.read_csv(IDX)
    df = df[df['label'].isin([0, 1]) & df['appeared'].isin(MONTHS)]
    df = df.drop_duplicates('sha256')                               # guard: one row per file hash
    rng = np.random.default_rng(SEED); keep = []
    for (m, l), g in df.groupby(['appeared', 'label']):
        g = g.sort_values('sha256')
        k = min(N_PER_CELL, len(g)); keep.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    sel = pd.concat(keep).sort_values(['file', 'row'])
    sel.to_csv(SEL, index=False)
    print(sel.groupby(['appeared', 'label']).size().unstack().to_string())
    return sel


def extract(fname):                                                 # pass 2 for one file
    sel = pd.read_csv(SEL); rows = set(sel.loc[sel['file'] == fname, 'row'])
    os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, fname.replace('.csv', '.sel.csv.gz'))
    start, n = 0, 0
    with gzip.open(out, 'wt', compresslevel=6) as fh:
        for ch in pd.read_csv(os.path.join(SRC, fname), chunksize=5000, dtype=str, keep_default_na=False):
            idx = np.arange(start, start + len(ch)); m = np.isin(idx, list(rows)) if rows else np.zeros(len(ch), bool)
            if m.any():
                ch[m].to_csv(fh, header=(n == 0), index=False); n += int(m.sum())
            start += len(ch)
    print(fname, 'selected', len(rows), 'written', n, '->', out)


if __name__ == '__main__':
    t = time.time()
    if sys.argv[1] == '--index':
        for f in sys.argv[2:]: print(f, index_file(f), 'rows', round(time.time() - t, 1), 's')
    elif sys.argv[1] == '--select': select()
    elif sys.argv[1] == '--extract': extract(sys.argv[2]); print(round(time.time() - t, 1), 's')

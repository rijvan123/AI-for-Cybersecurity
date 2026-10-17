"""data_ember.py - EMBER-2018 raw-feature vectoriser (feature version 2, 2,381 dims) and loaders.

The vectoriser re-implements ember/features.py::process_raw_features (elastic/ember, master; SHA-256 of the
reference file db0d93bb...) without LIEF, because the author's copy of EMBER-2018 stores raw features only.
Fidelity notes:
  * SectionInfo hashes the entry-section name with FeatureHasher(...).transform([raw_obj['entry']]): with the
    scikit-learn of 2018 a bare string was iterated character by character. Current scikit-learn rejects a bare
    string, so the character iteration is reproduced explicitly with list(entry) ('chars' mode, default).
  * All other hashers are batched across rows (identical output, ~100x faster than per-row calls).
The official ember_model_2018.txt (LightGBM) scores the vectorised official test months as an acceptance test:
a wrong vectoriser collapses its AUC; E0 records the AUC for both entry-name modes.
"""
import ast
import gzip
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from .config import PATHS, CACHE, EMBER_DIM, EMBER_SHARD_SHA256

RAW_COLS = ['histogram', 'byteentropy', 'strings', 'general', 'header', 'section', 'imports', 'exports',
            'datadirectories']
DD_ORDER = ["EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"]
BLOCKS = [('histogram', 256), ('byteentropy', 256), ('strings', 104), ('general', 10), ('header', 62),
          ('section', 255), ('imports', 1280), ('exports', 128), ('datadirectories', 30)]   # sums to 2,381


def block_index():
    """Feature-block -> column slice (for drift / SHAP aggregation by block)."""
    out, s = {}, 0
    for name, d in BLOCKS:
        out[name] = (s, s + d); s += d
    assert s == EMBER_DIM
    return out


def _fh(n, kind, samples):
    """One batched FeatureHasher call (alternate_sign=True is the default in 2018 and today)."""
    return FeatureHasher(n, input_type=kind).transform(samples).toarray()


def vectorise(raw, entry_mode='chars'):
    """raw: DataFrame of parsed raw objects (one dict/list per cell). Returns float32 (n, 2381)."""
    n = len(raw)
    hist = np.array([r for r in raw['histogram']], dtype=np.float32)
    hist = hist / hist.sum(1, keepdims=True)                                      # ByteHistogram
    bent = np.array([r for r in raw['byteentropy']], dtype=np.float32)
    bent = bent / bent.sum(1, keepdims=True)                                      # ByteEntropyHistogram
    S = raw['strings']                                                             # StringExtractor (104)
    strs = np.zeros((n, 104), np.float32)
    for i, o in enumerate(S):
        div = float(o['printables']) if o['printables'] > 0 else 1.0
        strs[i] = np.hstack([o['numstrings'], o['avlength'], o['printables'],
                             np.asarray(o['printabledist']) / div, o['entropy'], o['paths'], o['urls'],
                             o['registry'], o['MZ']])
    gen = np.array([[o['size'], o['vsize'], o['has_debug'], o['exports'], o['imports'], o['has_relocations'],
                     o['has_resources'], o['has_signature'], o['has_tls'], o['symbols']]
                    for o in raw['general']], dtype=np.float32)                   # GeneralFileInfo (10)
    H = raw['header']                                                              # HeaderFileInfo (62)
    hdr = np.hstack([
        np.array([[o['coff']['timestamp']] for o in H], dtype=np.float64),
        _fh(10, 'string', [[o['coff']['machine']] for o in H]),
        _fh(10, 'string', [o['coff']['characteristics'] for o in H]),
        _fh(10, 'string', [[o['optional']['subsystem']] for o in H]),
        _fh(10, 'string', [o['optional']['dll_characteristics'] for o in H]),
        _fh(10, 'string', [[o['optional']['magic']] for o in H]),
        np.array([[o['optional'][k] for k in (
            'major_image_version', 'minor_image_version', 'major_linker_version', 'minor_linker_version',
            'major_operating_system_version', 'minor_operating_system_version', 'major_subsystem_version',
            'minor_subsystem_version', 'sizeof_code', 'sizeof_headers', 'sizeof_heap_commit')] for o in H],
            dtype=np.float64)]).astype(np.float32)
    Sec = raw['section']                                                           # SectionInfo (255)
    general = np.array([[len(o['sections']),
                         sum(1 for s in o['sections'] if s['size'] == 0),
                         sum(1 for s in o['sections'] if s['name'] == ""),
                         sum(1 for s in o['sections'] if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
                         sum(1 for s in o['sections'] if 'MEM_WRITE' in s['props'])] for o in Sec], np.float64)
    if entry_mode == 'chars':                                                      # 2018 behaviour (see header)
        entry = [list(o['entry']) for o in Sec]
    else:                                                                          # whole-string variant (check)
        entry = [[o['entry']] for o in Sec]
    sec = np.hstack([
        general,
        _fh(50, 'pair', [[(s['name'], s['size']) for s in o['sections']] for o in Sec]),
        _fh(50, 'pair', [[(s['name'], s['entropy']) for s in o['sections']] for o in Sec]),
        _fh(50, 'pair', [[(s['name'], s['vsize']) for s in o['sections']] for o in Sec]),
        _fh(50, 'string', entry),
        _fh(50, 'string', [[p for s in o['sections'] for p in s['props'] if s['name'] == o['entry']] for o in Sec]),
    ]).astype(np.float32)
    Imp = raw['imports']                                                           # ImportsInfo (1,280)
    imp = np.hstack([
        _fh(256, 'string', [list(set(l.lower() for l in o.keys())) for o in Imp]),
        _fh(1024, 'string', [[lib.lower() + ':' + e for lib, el in o.items() for e in el] for o in Imp]),
    ]).astype(np.float32)
    exp = _fh(128, 'string', [list(o) for o in raw['exports']]).astype(np.float32)  # ExportsInfo (128)
    dd = np.zeros((n, 30), np.float32)                                             # DataDirectories (30)
    for i, o in enumerate(raw['datadirectories']):
        for j in range(min(len(DD_ORDER), len(o))):
            dd[i, 2 * j] = o[j]['size']; dd[i, 2 * j + 1] = o[j]['virtual_address']
    X = np.hstack([hist, bent, strs, gen, hdr, sec, imp, exp, dd]).astype(np.float32)
    assert X.shape[1] == EMBER_DIM, X.shape
    return X


def _parse(cell):
    return ast.literal_eval(cell) if isinstance(cell, str) and cell else ([] if cell == '' else cell)


def read_shards(shard_dir=None):
    """Concatenate the gzip shards produced by tools/make_ember_subsample.py (raw columns kept as strings)."""
    d = shard_dir or PATHS['ember_shards']
    frames = []
    for name in EMBER_SHARD_SHA256:
        if name.endswith('.csv.gz'):
            with gzip.open(d / name, 'rt') as fh:
                frames.append(pd.read_csv(fh, dtype=str, keep_default_na=False))
    return pd.concat(frames, ignore_index=True)


def build_cache(entry_mode='chars', force=False):
    """Parse + vectorise the subsample once; cache X (float32 .npy) and meta (.csv)."""
    px, pm = CACHE / f'ember_X_{entry_mode}.npy', CACHE / 'ember_meta.csv'
    if px.exists() and pm.exists() and not force:
        return px, pm
    df = read_shards()
    raw = pd.DataFrame({c: [_parse(v) for v in df[c]] for c in RAW_COLS})
    X = vectorise(raw, entry_mode)
    meta = pd.DataFrame({'sha256': df['sha256'], 'appeared': df['appeared'], 'y': df['label'].astype(int),
                         'family': df['avclass'].replace('', np.nan)})
    order = np.lexsort((meta['sha256'].to_numpy(), meta['appeared'].to_numpy()))   # deterministic row order
    np.save(px, X[order]); meta.iloc[order].reset_index(drop=True).to_csv(pm, index=False)
    return px, pm


def load_ember(entry_mode='chars', mmap=True):
    px, pm = build_cache(entry_mode)
    X = np.load(px, mmap_mode='r' if mmap else None)
    meta = pd.read_csv(pm)
    meta['ym'] = meta['appeared']
    meta['family'] = meta['family'].where(meta['y'] == 1, 'benign').fillna('unknown')
    meta['row_id'] = np.arange(len(meta))
    return X, meta


def official_model():
    """Endgame's EMBER-2018 LightGBM benchmark model (trained on the official 2018 training months)."""
    import lightgbm as lgb
    return lgb.Booster(model_file=str(PATHS['ember_model']))

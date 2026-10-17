"""E0 - dataset provenance and acceptance tests (runs before any model; aborts on a failed check).

Checks: SHA-256 of every input, documented record counts, class balance, timestamp coverage, identity parsing,
MalMem variant equivalence, dated-file join, BODMAS window, EMBER vectoriser acceptance (official model AUC).
Output: runs/<mode>/e0/provenance.csv, checks.json, env.json.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']    # -I drops the script dir
from common import *                                                           # noqa: F401,F403
from harness import data_bodmas as dbm, data_ember as dem, data_nids as dn
from harness.io_utils import sha256_file
from sklearn.metrics import roc_auc_score


def main():
    a = parse('E0 data verification')
    out = outdir(a.mode, 'e0'); T = TimingLog(); checks = {}; prov = []
    with T.time('sha256'):
        for k, p in C.PATHS.items():
            if k == 'ember_shards':
                for name, h in C.EMBER_SHARD_SHA256.items():
                    checks[f'sha256:{name}'] = sha256_file(p / name) == h
            else:
                checks[f'sha256:{k}'] = sha256_file(p) == C.SHA256[k]
    with T.time('malmem'):
        X, meta = dm.load_malmem()
        checks['malmem_rows'] = len(meta) == C.EXPECTED_COUNTS['malmem_rows']
        checks['malmem_balance'] = int(meta.y.sum()) == C.EXPECTED_COUNTS['malmem_malware']
        alt = pd.read_csv(C.PATHS['malmem_alt'])
        h1 = pd.Series(row_hashes(X.to_numpy(), 9)).value_counts().sort_index()
        h2 = pd.Series(row_hashes(alt[list(X.columns)].astype(float).to_numpy(), 9)).value_counts().sort_index()
        checks['malmem_variant_same_multiset'] = h1.equals(h2)
        sib = dm.sibling_structure(meta)
        dated = dm.load_dated(meta, X)
        prov.append({'dataset': 'CIC-MalMem-2022', 'file': C.PATHS['malmem'].name, 'rows': len(meta),
                     'features': X.shape[1], 'malware': int(meta.y.sum()), 'benign': int((meta.y == 0).sum()),
                     'groups': f"{sib['n_binaries']} binaries (median {sib['dumps_median']:.0f} dumps)",
                     'timestamps': f"malware only: {int(dated['dated'].sum())} rows dated (VirusTotal first-submission month)",
                     'label_oracle': 'creator labels; VirusTotal-sourced binaries executed in a Windows 10 VM; benign SMOTE-balanced',
                     'source': 'CIC/UNB (Carrier et al., ICISSP 2022), author upload'})
        checks['dated_rows_joined'] = int(dated['t'].notna().sum()) == C.EXPECTED_COUNTS['memmal_dated_rows']
        checks['malmem_identity'] = sib
        checks['dated_conflict_rows'] = int(dated['conflict_binary'].sum())
    with T.time('bodmas'):
        Xb, mb = dbm.load_bodmas()
        checks['bodmas_rows'] = Xb.shape == (C.EXPECTED_COUNTS['bodmas_rows'], C.EMBER_DIM)
        checks['bodmas_classes'] = (int(mb.y.sum()), int((mb.y == 0).sum())) == (
            C.EXPECTED_COUNTS['bodmas_malware'], C.EXPECTED_COUNTS['bodmas_benign'])
        wc = dbm.window_counts(mb); wc.to_csv(out / 'bodmas_monthly_counts.csv')
        prov.append({'dataset': 'BODMAS', 'file': 'bodmas.npz + metadata + category', 'rows': len(mb),
                     'features': Xb.shape[1], 'malware': int(mb.y.sum()), 'benign': int((mb.y == 0).sum()),
                     'groups': f"{mb.loc[mb.y == 1, 'family'].nunique()} malware families",
                     'timestamps': f"both classes; in-window benign {int(mb[(mb.y == 0)].in_window.sum())} of {int((mb.y == 0).sum())}",
                     'label_oracle': 'dataset authors (vendor labels + aggregated AV verdicts)',
                     'source': 'Yang et al., IEEE SPW 2021; author upload'})
    with T.time('ember'):
        Xe, me = dem.load_ember()
        checks['ember_rows'] = len(me) == C.EXPECTED_COUNTS['ember_sub_rows']
        booster = dem.official_model()
        mask = me['ym'].isin(C.EMBER_OFFICIAL_TEST_MONTHS).to_numpy()
        p = booster.predict(np.asarray(Xe[mask]))
        auc_chars = roc_auc_score(me['y'][mask], p)
        checks['ember_official_model_auc_chars'] = float(auc_chars)
        checks['ember_vectoriser_accept'] = auc_chars > 0.99                    # acceptance threshold (pre-set)
        prov.append({'dataset': 'EMBER-2018 (subsample)', 'file': '_S9_shards (6 gzip CSV)', 'rows': len(me),
                     'features': Xe.shape[1], 'malware': int(me.y.sum()), 'benign': int((me.y == 0).sum()),
                     'groups': f"{me.loc[me.y == 1, 'family'].nunique()} AVClass families",
                     'timestamps': 'both classes (appeared month), 2,500 per month x class, 2018-01..12',
                     'label_oracle': 'EMBER labels (AV-threshold; unlabelled rows excluded)',
                     'source': 'Anderson & Roth 2018 (raw features, author copy); stratified seed 20181016'})
    with T.time('nids'):
        Fn, mn = dn.load_nsl(); Fu, mu = dn.load_unsw()
        checks['nsl_counts'] = (int((mn.partition == 'train').sum()), int((mn.partition == 'test').sum())) == (
            C.EXPECTED_COUNTS['nsl_train'], C.EXPECTED_COUNTS['nsl_test'])
        checks['unsw_counts'] = (int((mu.partition == 'train').sum()), int((mu.partition == 'test').sum())) == (
            C.EXPECTED_COUNTS['unsw_train'], C.EXPECTED_COUNTS['unsw_test'])
        for nm, F, m, src in (('NSL-KDD', Fn, mn, 'Tavallaee et al. 2009; GitHub mirror defcom17/NSL_KDD'),
                              ('UNSW-NB15', Fu, mu, 'Moustafa & Slay 2015; GitHub mirror Nir-J/ML-Projects')):
            prov.append({'dataset': nm, 'file': 'official train/test partitions', 'rows': len(m),
                         'features': F.shape[1], 'malware': int(m.y.sum()), 'benign': int((m.y == 0).sum()),
                         'groups': f"{m.loc[m.y == 1, 'family'].nunique()} attack families", 'timestamps': 'none',
                         'label_oracle': 'testbed attack schedule (published labels)', 'source': src})
    pd.DataFrame(prov).to_csv(out / 'provenance.csv', index=False)
    save_json(checks, out / 'checks.json'); save_json(environment(), out / 'env.json')
    T.frame().to_csv(out / 'timing.csv', index=False)
    failed = [k for k, v in checks.items() if v is False]
    print(pd.DataFrame(prov).to_string()); print('FAILED:', failed or 'none')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

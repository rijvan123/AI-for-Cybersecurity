"""E1 - C1 data-validity audit (S2 C1; S3 constraints 1, 5; S5 I03 + I02; S6 P1 conditions).

Parts: (a) invariant tolerance sweep, (b) SMOTE genealogy of synthetic rows, (c) constants, duplicates, sibling
identity, (d) neighbour leakage of synthetic rows, (e) univariate separability and plugin-block ablation on
genuine rows (shortcut audit), (f) the same stump / feature-group audit ported to NSL-KDD and UNSW-NB15.
Block ablation keeps one set of grouped, deduplicated folds for every variant (identity defined on all 55
features), so 'only:<block>' and 'drop:<block>' variants are compared on identical test rows.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import models as M, data_nids as dn
from harness.metrics import panel
from harness.thresholds import fpr_threshold
from sklearn.metrics import roc_auc_score

PORT_MAX_TRAIN = 50000      # training rows per NIDS ablation fit (CPU budget; the full ladder in E4 uses all rows)


def parent_components(X, meta, gate, g):
    """Why SMOTE children are purged rather than merged (common.malmem_set): link the benign pseudo-groups of each
    child's two parents and measure the connected components over genuine rows."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    gen = gate['genuine_benign']; is_mal = meta['y'].to_numpy() == 1
    sc = dm.identity_scaler(X, is_mal | gen); radius, _ = dm.sibling_radius(X, meta, sc)
    lab, n_comp = dm.benign_pseudo_groups(X.to_numpy()[gen], radius, sc)
    pos = -np.ones(len(X), int); pos[np.flatnonzero(gen)] = np.arange(int(gen.sum()))
    la, lb = lab[pos[g['parents'][:, 0]]], lab[pos[g['parents'][:, 1]]]
    nc, comp = connected_components(coo_matrix((np.ones(len(la)), (la, lb)), shape=(n_comp, n_comp)), directed=False)
    rows_per = np.bincount(comp[lab], minlength=nc)
    return {'pseudo_groups': int(n_comp), 'children_parents_same_group': float(np.mean(la == lb)),
            'merged_components': int(nc), 'largest_component_genuine_rows': int(rows_per.max()),
            'largest_component_share': float(rows_per.max() / gen.sum())}


def part_a_to_d(out, T, B):
    X, meta = dm.load_malmem(); y = meta.y.to_numpy()
    rows = []
    with T.time('invariant_sweep'):
        for s in dm.INVARIANT_SETS:
            for tol in C.INVARIANT_TOLERANCES:
                g = dm.validity_gate(X, meta, s, tol)['counts']; rows.append(g)
    pd.DataFrame(rows).to_csv(out / 'invariant_sweep.csv', index=False)
    tab = dm.invariant_table(X, C.PRIMARY_TOLERANCE)
    pr = pd.DataFrame({'malware_pass': tab[y == 1].mean(), 'benign_pass': tab[y == 0].mean()})
    pr['by_family_min'] = [tab[y == 1].groupby(meta.family[y == 1]).mean()[c].min() for c in tab.columns]
    pr.to_csv(out / 'invariant_pass_rates.csv')
    gate = dm.validity_gate(X, meta, C.PRIMARY_INVARIANT_SET, C.PRIMARY_TOLERANCE)
    const, nearc = dm.constant_features(X); dup = dm.exact_duplicates(X, meta); sib = dm.sibling_structure(meta)
    facts = {'gate': gate['counts'], 'constant': const, 'near_constant': nearc,
             'duplicates': {k: v for k, v in dup.items() if k not in ('hash', 'keep_first')}, 'siblings': sib}
    with T.time('genealogy'):
        gen = {}
        for k in (C.GENEALOGY_K, 30):
            g = dm.genealogy(X, gate['genuine_benign'], gate['synthetic_benign'], k=k)
            lam_min = np.minimum(g['lambda'], 1 - g['lambda'])
            parents = pd.Series(g['parents'].ravel()).value_counts()
            gen[f'k{k}'] = {'rows_tested': g['rows_tested'], 'explained': g['explained'],
                            'share_explained': g['share_explained'],
                            'genuine_rows_exposed_as_parent': int(parents.size),
                            'share_genuine_exposed': parents.size / int(gate['genuine_benign'].sum()),
                            'children_per_parent_median': float(parents.median()),
                            'children_per_parent_max': int(parents.max()),
                            'lambda_min_quantiles': np.quantile(lam_min, [.1, .25, .5, .75, .9]).tolist(),
                            'ks_uniform_0_0.5_p': float(__import__('scipy').stats.kstest(lam_min, 'uniform', args=(0, .5)).pvalue)}
            if k == C.GENEALOGY_K:
                np.save(out / 'genealogy_lambda.npy', g['lambda'])
                gen['parent_components'] = parent_components(X, meta, gate, g)
        facts['genealogy'] = gen
    with T.time('neighbour_leakage'):                                       # S3 probe, all-benign random split
        ds = malmem_set('all', dedup=False)
        sp = next(S.random_kfold(ds.y, C.N_FOLDS, 0, 0.0))
        gmask = np.isin(ds.row_id, np.flatnonzero(gate['genuine_benign']))
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(ds.X[sp.train]); Ztr = sc.transform(ds.X[sp.train])
        tr_gen = gmask[sp.train] & (ds.y[sp.train] == 0); tr_syn = (~gmask[sp.train]) & (ds.y[sp.train] == 0)
        te_gen = sp.test[gmask[sp.test] & (ds.y[sp.test] == 0)]
        d_syn = NearestNeighbors(n_neighbors=1).fit(Ztr[tr_syn]).kneighbors(sc.transform(ds.X[te_gen]))[0][:, 0]
        d_gen = NearestNeighbors(n_neighbors=1).fit(Ztr[tr_gen]).kneighbors(sc.transform(ds.X[te_gen]))[0][:, 0]
        facts['neighbour_leakage'] = {'n_genuine_test': int(len(te_gen)), 'median_d_synthetic': float(np.median(d_syn)),
                                      'median_d_genuine': float(np.median(d_gen)),
                                      'ratio_of_medians': float(np.median(d_gen) / np.median(d_syn)),
                                      'share_closer_to_synthetic': float(np.mean(d_syn < d_gen))}
    save_json(facts, out / 'malmem_facts.json')
    return X, meta, gate


def part_e(out, T, seeds, X, meta, gate):
    """Univariate separability (genuine pool) and plugin-block ablation with grouped CV (shortcut audit)."""
    ds = malmem_set('genuine', dedup=True)
    uni = []
    for j, f in enumerate(ds.features):
        if np.ptp(ds.X[:, j]) == 0:
            continue
        auc = roc_auc_score(ds.y, ds.X[:, j]); uni.append({'feature': f, 'auc': auc, 'auc_sym': max(auc, 1 - auc)})
    pd.DataFrame(uni).sort_values('auc_sym', ascending=False).to_csv(out / 'univariate_auc_genuine.csv', index=False)
    blocks = dm.plugin_blocks(ds.features); fidx = {f: i for i, f in enumerate(ds.features)}
    rows = []
    with T.time('block_ablation'):
        for seed in seeds:
            for sp in S.grouped_kfold(ds.y, ds.groups, C.N_FOLDS, seed, C.VAL_FRAC):
                sp = apply_dedup(ds, sp, 'exact', measure_near=False)
                variants = {'all': list(range(len(ds.features)))}
                for b, fs in blocks.items():
                    variants[f'only:{b}'] = [fidx[f] for f in fs]
                    variants[f'drop:{b}'] = [fidx[f] for f in ds.features if f not in fs]
                for v, cols in variants.items():
                    for name in ('stump', 'lgbm'):
                        m = M.make_model(name, seed); Xc = ds.X[:, cols]
                        M.fit(m, Xc[sp.train], ds.y[sp.train])
                        s_val, s_te = M.score(m, Xc[sp.val]), M.score(m, Xc[sp.test])
                        thr = fpr_threshold(s_val[ds.y[sp.val] == 0], C.PRIMARY_FPR_BUDGET)
                        for pi in (0.5, 0.01):
                            r = panel(ds.y[sp.test], s_te, thr, pi)
                            rows.append({'variant': v, 'model': name, 'seed': seed, 'fold': sp.info['fold'], 'pi': pi,
                                         'auc_roc': r['auc_roc'], 'auc_pr': r['auc_pr'], 'mcc': r['mcc'],
                                         'precision': r['precision'], 'recall': r['recall'], 'fpr': r['fpr']})
    pd.DataFrame(rows).to_csv(out / 'block_ablation.csv', index=False)


def part_f(out, T, seeds):
    """Port of the stump and feature-family audit to NSL-KDD and UNSW-NB15 (random and official rungs)."""
    rows = []
    for nm, loader, cats, groups_def in (('nsl_kdd', dn.load_nsl, dn.NSL_CAT, dn.NSL_GROUPS),
                                         ('unsw_nb15', dn.load_unsw, dn.UNSW_CAT, dn.UNSW_GROUPS)):
        F, meta = loader(); y = meta.y.to_numpy()
        with T.time(f'port_{nm}'):
            for seed in seeds:
                part = meta.partition.to_numpy()
                rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); cut = int(0.8 * len(y))
                rungs = {'official': (np.flatnonzero(part == 'train'), np.flatnonzero(part == 'test')),
                         'random': (np.sort(perm[:cut]), np.sort(perm[cut:]))}
                for rung, (tr, te) in rungs.items():
                    tr_fit, va = S._carve_val(tr, y, None, C.VAL_FRAC, seed)
                    if len(tr_fit) > PORT_MAX_TRAIN:                           # fixed-seed cap: ablation is relative
                        tr_fit = np.sort(np.random.default_rng(seed).choice(tr_fit, PORT_MAX_TRAIN, replace=False))
                    Xtr, Xte, names = dn.encode_split(F.iloc[tr], F.iloc[te], cats)
                    pos = {r: i for i, r in enumerate(tr)}
                    fit_rows = np.array([pos[r] for r in tr_fit]); val_rows = np.array([pos[r] for r in va])
                    gcols = dn.group_columns(names, groups_def)
                    variants = {'all': list(range(len(names)))}
                    for g, idx in gcols.items():
                        if idx:
                            variants[f'only:{g}'] = idx
                            variants[f'drop:{g}'] = [i for i in range(len(names)) if i not in set(idx)]
                    for v, cols in variants.items():
                        for name in ('stump', 'lgbm'):
                            m = M.make_model(name, seed)
                            M.fit(m, Xtr[fit_rows][:, cols], y[tr_fit])
                            s_val = M.score(m, Xtr[val_rows][:, cols]); s_te = M.score(m, Xte[:, cols])
                            thr = fpr_threshold(s_val[y[va] == 0], C.PRIMARY_FPR_BUDGET)
                            feat = M.stump_feature(m, np.asarray(names)[cols])[0] if name == 'stump' else None
                            for pi in (0.5, 0.01):
                                r = panel(y[te], s_te, thr, pi)
                                rows.append({'dataset': nm, 'rung': rung, 'variant': v, 'model': name, 'seed': seed,
                                             'pi': pi, 'stump_feature': feat, 'auc_roc': r['auc_roc'],
                                             'auc_pr': r['auc_pr'], 'mcc': r['mcc'], 'precision': r['precision'],
                                             'recall': r['recall'], 'fpr': r['fpr']})
    pd.DataFrame(rows).to_csv(out / 'nids_port.csv', index=False)


def main():
    a = parse('E1 data-validity audit')
    out = outdir(a.mode, 'e1'); T = TimingLog()
    X, meta, gate = part_a_to_d(out, T, a.B)
    part_e(out, T, a.seeds, X, meta, gate)
    part_f(out, T, a.seeds)
    T.frame().to_csv(out / 'timing.csv', index=False); save_json(environment(), out / 'env.json')
    print(T.frame().to_string())


if __name__ == '__main__':
    main()

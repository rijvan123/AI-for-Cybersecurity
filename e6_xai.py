"""E6 - C4: SHAP as a shortcut detector with stability and faithfulness checks (S2 C4; S3 constraint 9; S6 P6).

For each protocol cell (pool x rung) and fold, TreeSHAP global importance of the boosted models (LightGBM, XGBoost)
is computed on the test fold; outputs: block shares (share of mean |phi| per Volatility plugin; C4 predicts >= 50%
on svcscan), top-k sets, and stability (Jaccard@10, Kendall tau) within a cell (across folds/seeds), across cells
(protocols) and across temporal windows; deletion test on the strict cell (top-k SHAP vs random-k, retrained).
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from itertools import combinations
from harness import models as M, explain as XP
from harness.metrics import panel
from harness.thresholds import fpr_threshold

CELLS = [('all', 'random'), ('all', 'grouped'), ('genuine', 'random'), ('genuine', 'grouped'),
         ('genuine', 'lofo'), ('genuine', 'temporal'), ('all', 'temporal')]


def main():
    a = parse('E6 explanation audit')
    out = outdir(a.mode, 'e6'); T = TimingLog()
    smoke = a.mode == 'smoke'
    nexp = 500 if smoke else C.SHAP_EXPLAIN_N
    imps, rows = {}, []
    for pool, rung in CELLS:
        ds = malmem_set(pool, dedup=True); blocks = dm.plugin_blocks(ds.features)
        for seed in a.seeds:
            for sp in malmem_cell_splits(ds, rung, seed):
                if rung == 'lofo' and smoke and sp.info['fold'] > 4:            # smoke: first five families
                    continue
                sp = apply_dedup(ds, sp, 'exact', measure_near=False)
                for name in ('lgbm', 'xgb'):
                    with T.time('fit_shap', pool=pool, rung=rung, model=name, seed=seed):
                        m = M.make_model(name, seed); M.fit(m, ds.X[sp.train], ds.y[sp.train])
                        imp = XP.tree_shap(m, ds.X[sp.test], max_rows=nexp, seed=seed)
                    key = (pool, rung, name, seed, sp.info.get('fold', sp.info.get('window')))
                    imps[key] = imp
                    sh = XP.block_shares(imp, ds.features, blocks)
                    top = [ds.features[j] for j in np.argsort(-imp)[:C.SHAP_TOPK]]
                    rows.append({'pool': pool, 'rung': rung, 'model': name, 'seed': seed,
                                 'fold': key[-1], 'top1': top[0], 'top10': ';'.join(top),
                                 **{f'share_{b}': v for b, v in sh.items()}})
        print(f'{pool} {rung} done', flush=True)
    pd.DataFrame(rows).to_csv(out / 'shap_shares.csv', index=False)
    np.savez_compressed(out / 'shap_importance.npz', **{'|'.join(map(str, k)): v for k, v in imps.items()})
    # stability: within cell (pairs of folds/seeds), across cells (same model, fold-averaged vectors)
    st_rows = []
    keys = list(imps)
    for (pool, rung, name) in sorted({k[:3] for k in keys}):
        ks = [k for k in keys if k[:3] == (pool, rung, name)]
        for k1, k2 in combinations(ks, 2):
            st_rows.append({'scope': 'within_cell', 'a': '|'.join(map(str, k1)), 'b': '|'.join(map(str, k2)),
                            'pool': pool, 'rung': rung, 'model': name,
                            **XP.stability(imps[k1], imps[k2], C.SHAP_TOPK)})
    mean_imp = {k: np.mean([imps[x] for x in keys if x[:3] == k], axis=0) for k in {x[:3] for x in keys}}
    for (k1, k2) in combinations(sorted(mean_imp), 2):
        if k1[2] == k2[2]:
            st_rows.append({'scope': 'across_cells', 'a': '|'.join(k1), 'b': '|'.join(k2), 'model': k1[2],
                            **XP.stability(mean_imp[k1], mean_imp[k2], C.SHAP_TOPK)})
    pd.DataFrame(st_rows).to_csv(out / 'shap_stability.csv', index=False)
    # deletion (faithfulness) test on the strict cell, first split
    ds = malmem_set('genuine', dedup=True)
    sp = apply_dedup(ds, next(malmem_cell_splits(ds, 'grouped', a.seeds[0])), 'exact', measure_near=False)
    ranking = np.argsort(-mean_imp[('genuine', 'grouped', 'lgbm')])

    def fit_eval(cols):
        m = M.make_model('lgbm', a.seeds[0]); M.fit(m, ds.X[sp.train][:, cols], ds.y[sp.train])
        sv, st_ = M.score(m, ds.X[sp.val][:, cols]), M.score(m, ds.X[sp.test][:, cols])
        thr = fpr_threshold(sv[ds.y[sp.val] == 0], C.PRIMARY_FPR_BUDGET)
        return panel(ds.y[sp.test], st_, thr, 0.01)['auc_pr']
    with T.time('deletion'):
        dl = XP.deletion_test(fit_eval, ranking, C.DELETION_KS, len(ds.features), n_random=2 if smoke else 5,
                              seed=a.seeds[0])
    dl.to_csv(out / 'deletion.csv', index=False)
    T.frame().to_csv(out / 'timing.csv', index=False); save_json(environment(), out / 'env.json')


if __name__ == '__main__':
    main()

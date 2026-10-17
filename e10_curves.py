"""E10 - learning curves and feature importances for R5 (strict MalMem cell; BODMAS temporal).

Learning curves: fraction of training GROUPS kept in {2,5,10,20,40,70,100}% (MalMem, grouped folds) and fraction of
training rows (BODMAS, temporal; mean over the 11 monthly windows); metrics AUC-ROC, AUC-PR and MCC at pi = 1% with
the validation FPR-1% threshold. Importances: RF impurity, XGBoost gain and LightGBM gain (fold-averaged) on the
strict MalMem cell; permutation importance (AUC-PR drop) for RF on the test fold.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import models as M
from harness.metrics import panel
from harness.thresholds import fpr_threshold
from sklearn.inspection import permutation_importance

FRACS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)


def eval_fit(name, seed, Xtr, ytr, Xva, yva, Xte, yte):
    m = M.make_model(name, seed); M.fit(m, Xtr, ytr)
    sv, st_ = M.score(m, Xva), M.score(m, Xte)
    thr = fpr_threshold(sv[yva == 0], C.PRIMARY_FPR_BUDGET)
    r = panel(yte, st_, thr, 0.01)
    return m, {k: r[k] for k in ('auc_roc', 'auc_pr', 'mcc', 'recall', 'fpr')}


def main():
    a = parse('E10 learning curves and importances')
    out = outdir(a.mode, 'e10'); T = TimingLog(); rows = []; imp_rows = []
    ds = malmem_set('genuine', dedup=True)
    models = ('stump', 'lr', 'rf', 'lgbm') if a.mode == 'smoke' else ('stump', 'lr', 'gnb', 'rf', 'xgb', 'lgbm')
    for seed in a.seeds:
        for sp in malmem_cell_splits(ds, 'grouped', seed):
            if a.mode == 'smoke' and sp.info['fold'] > 1:
                continue
            sp = apply_dedup(ds, sp, 'exact', measure_near=False)
            ug = np.unique(ds.groups[sp.train]); rng = np.random.default_rng(seed)
            for f in FRACS:
                keep_g = set(rng.choice(ug, max(2, int(round(f * len(ug)))), replace=False))
                tr = sp.train[np.isin(ds.groups[sp.train], list(keep_g))]
                if len(np.unique(ds.y[tr])) < 2:
                    continue
                for name in models:
                    with T.time('lc_malmem', model=name, frac=f):
                        _, r = eval_fit(name, seed, ds.X[tr], ds.y[tr], ds.X[sp.val], ds.y[sp.val],
                                        ds.X[sp.test], ds.y[sp.test])
                    rows.append({'dataset': 'malmem_strict', 'model': name, 'seed': seed, 'fold': sp.info['fold'],
                                 'frac': f, 'n_train': len(tr), **r})
            for name in ('rf', 'xgb', 'lgbm'):                                 # importances (full training fold)
                m = M.make_model(name, seed); M.fit(m, ds.X[sp.train], ds.y[sp.train])
                imp = M.importances(m, ds.features)
                for j, v in enumerate(imp):
                    imp_rows.append({'model': name, 'kind': 'impurity_or_gain', 'seed': seed,
                                     'fold': sp.info['fold'], 'feature': ds.features[j], 'importance': v})
                if name == 'rf' and sp.info['fold'] == 0:
                    with T.time('permutation'):
                        pim = permutation_importance(m, ds.X[sp.test], ds.y[sp.test], scoring='average_precision',
                                                     n_repeats=3 if a.mode == 'smoke' else 10, random_state=seed,
                                                     n_jobs=1)
                    for j, v in enumerate(pim.importances_mean):
                        imp_rows.append({'model': 'rf', 'kind': 'permutation_ap', 'seed': seed, 'fold': 0,
                                         'feature': ds.features[j], 'importance': v})
    # BODMAS temporal learning curve (rows fraction), mean over monthly windows
    bod = bodmas_set()
    sp0 = next(S.temporal_both_dated(bod.y, bod.info['ym'], bod.info['order'], C.BODMAS_TRAIN_MONTHS,
                                     C.BODMAS_TEST_MONTHS, C.VAL_FRAC))
    wins = [np.flatnonzero(bod.info['ym'] == m) for m in C.BODMAS_TEST_MONTHS]
    if a.mode == 'smoke':
        wins = wins[::5]
    Xva = np.asarray(bod.X[sp0.val])
    for seed in a.seeds[:1]:
        rng = np.random.default_rng(seed)
        for f in (FRACS[1:] if a.mode == 'full' else (0.05, 0.2, 1.0)):
            tr = np.sort(rng.choice(sp0.train, max(50, int(round(f * len(sp0.train)))), replace=False))
            Xtr = np.asarray(bod.X[tr])
            for name in (('lr', 'lgbm') if a.mode == 'smoke' else ('lr', 'rf', 'lgbm')):
                with T.time('lc_bodmas', model=name, frac=f):
                    m = M.make_model(name, seed); M.fit(m, Xtr, bod.y[tr])
                    sv = M.score(m, Xva); thr = fpr_threshold(sv[bod.y[sp0.val] == 0], C.PRIMARY_FPR_BUDGET)
                    res = [panel(bod.y[w], M.score(m, np.asarray(bod.X[w])), thr, 0.01) for w in wins]
                rows.append({'dataset': 'bodmas_temporal', 'model': name, 'seed': seed, 'frac': f, 'n_train': len(tr),
                             **{k: float(np.mean([r[k] for r in res])) for k in ('auc_roc', 'auc_pr', 'mcc',
                                                                                 'recall', 'fpr')}})
    pd.DataFrame(rows).to_csv(out / 'learning_curves.csv', index=False)
    pd.DataFrame(imp_rows).to_csv(out / 'importances.csv', index=False)
    T.frame().to_csv(out / 'timing.csv', index=False); save_json(environment(), out / 'env.json')


if __name__ == '__main__':
    main()

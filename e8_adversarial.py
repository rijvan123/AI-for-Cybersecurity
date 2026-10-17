"""E8 - adversarial exposure (S5 I14, first to cut; S8 H5; S3 constraint 7). Feature-space only, stated as such.

(a) H5 exactness: for logistic regression and linear SVM in z-space, FGSM evasion rate == fraction of detected
    malware with radius r <= eps (Eq. 89-90), for p in {inf, 2}; add-only radii also reported.
(b) Greedy query attack with invariant-preserving memory actions (user vs admin capability) against LightGBM,
    random forest and monotone-constrained LightGBM (+1 on increase-only counts; variant without the averages).
    Evasion rate per action budget and clean recall at the validation threshold (FPR <= 1%).
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import models as M, adversarial as A
from harness.thresholds import fpr_threshold


def linear_check(ds, sp, out_rows):
    feats = np.asarray(ds.features)
    for name in ('lr', 'linsvm'):
        m = M.make_model(name, 0); M.fit(m, ds.X[sp.train], ds.y[sp.train])
        pre = m[:-1]; clf = m.named_steps['clf']
        Zv, Zt = pre.transform(ds.X[sp.val]), pre.transform(ds.X[sp.test])
        w, b = clf.coef_.ravel(), float(np.ravel(clf.intercept_)[0])
        tau = fpr_threshold((Zv @ w + b)[ds.y[sp.val] == 0], C.PRIMARY_FPR_BUDGET)
        det = (ds.y[sp.test] == 1) & ((Zt @ w + b) >= tau)
        for p in ('inf', '2'):
            df = A.evasion_rate_check(Zt[det], w, b, tau, C.ADV_EPS, p)
            for r in df.to_dict('records'):
                out_rows.append({'part': 'linear_exactness', 'model': name, 'n_detected': int(det.sum()), **r})
        kept = feats[m.named_steps['var'].get_support()]
        add = np.isin(kept, A.INCREASE_ONLY)
        r_add = A.linear_radius((Zt[det] @ w + b) - tau, w, 'inf', add)
        out_rows.append({'part': 'add_only_radius', 'model': name, 'n_detected': int(det.sum()),
                         'add_only_features_useful': int((add & (w < 0)).sum()),
                         'median_radius_add_only': float(np.median(r_add)),
                         'median_radius_unconstrained': float(np.median(A.linear_radius((Zt[det] @ w + b) - tau, w, 'inf')))})


def greedy_block(ds, sp, out_rows, budgets, max_samples):
    feats = list(ds.features)
    ben_ref = ds.X[sp.train][ds.y[sp.train] == 0]
    am = A.MemoryActionModel(feats, ben_ref)
    inc_idx = [feats.index(f) for f in A.INCREASE_ONLY]
    avg_cols = [i for i, f in enumerate(feats) if 'avg' in f]
    variants = {'lgbm': ('lgbm', None), 'rf': ('rf', None)}
    mono = np.zeros(len(feats), int); mono[inc_idx] = 1
    mono_admin = np.zeros(len(feats), int); mono_admin[[feats.index(f) for f in A.ADMIN_INCREASE_ONLY]] = 1
    monos = {}
    for vname, cols, mvec in (('lgbm_mono', list(range(len(feats))), mono),
                              ('lgbm_mono_noavg', [i for i in range(len(feats)) if i not in avg_cols], mono),
                              ('lgbm_mono_admin', list(range(len(feats))), mono_admin)):
        variants[vname] = ('lgbm_mono', cols); monos[vname] = mvec
    for vname, (mname, cols) in variants.items():
        cols = cols if cols is not None else list(range(len(feats)))
        m = M.make_model(mname, 0)
        if mname == 'lgbm_mono':
            m.set_params(monotone_constraints=[int(monos[vname][c]) for c in cols],
                         monotone_constraints_method='advanced')
            m.fit(ds.X[sp.train][:, cols], ds.y[sp.train])
            sfn = lambda Z, m=m, cols=cols: m.predict_proba(Z[:, cols])[:, 1]
        else:
            M.fit(m, ds.X[sp.train], ds.y[sp.train]); sfn = lambda Z, m=m: M.score(m, Z)
        tau = fpr_threshold(sfn(ds.X[sp.val])[ds.y[sp.val] == 0], C.PRIMARY_FPR_BUDGET)
        st_ = sfn(ds.X[sp.test]); mal = ds.y[sp.test] == 1
        recall = float((st_[mal] >= tau).mean()); fpr = float((st_[~mal] >= tau).mean())
        det_rows = sp.test[mal & (st_ >= tau)]
        if len(det_rows) > max_samples:
            det_rows = np.sort(np.random.default_rng(0).choice(det_rows, max_samples, replace=False))
        for cap, actions in (('user', A.USER_ACTIONS), ('admin', A.ADMIN_ACTIONS)):
            t0 = time.perf_counter()
            ev, _ = A.greedy_attack(sfn, ds.X[det_rows], tau, am, actions, max(budgets))
            for k in budgets:
                out_rows.append({'part': 'greedy', 'model': vname, 'capability': cap, 'budget': k,
                                 'evasion_rate': float(np.mean(ev <= k)), 'n_attacked': int(len(det_rows)),
                                 'clean_recall': recall, 'clean_fpr': fpr, 'seconds': time.perf_counter() - t0})
        print(f'greedy {vname} done', flush=True)


def main():
    a = parse('E8 adversarial exposure')
    out = outdir(a.mode, 'e8'); T = TimingLog(); rows = []
    ds = malmem_set('genuine', dedup=True)
    sp = apply_dedup(ds, next(malmem_cell_splits(ds, 'grouped', a.seeds[0])), 'exact', measure_near=False)
    with T.time('linear'):
        linear_check(ds, sp, rows)
    with T.time('greedy'):
        greedy_block(ds, sp, rows, C.ADV_ACTION_BUDGETS if a.mode == 'full' else (1, 5, 20),
                     2000 if a.mode == 'full' else 300)
    pd.DataFrame(rows).to_csv(out / 'results.csv', index=False)
    T.frame().to_csv(out / 'timing.csv', index=False); save_json(environment(), out / 'env.json')


if __name__ == '__main__':
    main()

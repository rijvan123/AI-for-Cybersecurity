"""E5 - C3 privacy audit on CIC-MalMem-2022 (genuine pool; S5 I16 + I18; S3 constraint 8; S8 H6, H7).

(a) SMOTE vs class weighting vs no rebalancing: membership inference on REAL minority (genuine benign) records,
    loss-threshold attack and offline LiRA with shadow models; TPR at FPR 0.1% / 1%, AUC, advantage; utility.
(b) DP-SGD logistic regression frontier, epsilon in {0.5,1,2,4,8} + non-private, row-level vs group-level units,
    RDP accountant; utility and loss-threshold MIA per epsilon.
(c) FedAvg over 10 Dirichlet non-IID clients (alpha in {0.1,0.5,1}) vs centralised SGD with the same optimiser.
Members / non-members: a grouped 50/50 partition of genuine benign pseudo-groups (seeded).
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import models as M, privacy as P
from harness.metrics import panel
from harness.thresholds import fpr_threshold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import expit


def partition(ds, seed):
    """Members (train) vs non-members among genuine benign groups; malware binaries split 50/50 too."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in (0, 1):
        idx = np.flatnonzero(ds.y == cls); ug = np.unique(ds.groups[idx]); rng.shuffle(ug)
        half = set(ug[: len(ug) // 2])
        inm = np.isin(ds.groups[idx], list(half))
        out[cls] = (idx[inm], idx[~inm])
    return out


def fit_variant(name, variant, X, y, seed, groups=None):
    """Train with no rebalancing, balanced class weights, or SMOTE on the training minority."""
    if variant == 'smote':
        mi = np.flatnonzero(y == 0); n_new = int((y == 1).sum() - (y == 0).sum())
        Xs, _, _ = P.smote(X[mi], max(n_new, 0), k=5, seed=seed)
        X = np.vstack([X, Xs]); y = np.concatenate([y, np.zeros(len(Xs), int)])
    m = M.make_model(name, seed)
    if variant == 'none':                                                     # strip class weights
        clf = m.named_steps['clf']
        if hasattr(clf, 'class_weight'):
            clf.set_params(class_weight=None)
        m.fit(X, y)
    else:
        M.fit(m, X, y, balanced=(variant == 'class_weight'))
    return m


def mia_block(ds, parts, seed, models, variants, n_shadow, out_rows):
    mem_b, non_b = parts[0]; tr_m, te_m = parts[1]
    cand = np.concatenate([mem_b, non_b]); is_mem = np.r_[np.ones(len(mem_b), bool), np.zeros(len(non_b), bool)]
    rng = np.random.default_rng(seed + 7)
    half = [rng.random(len(cand)) < 0.5 for _ in range(max(1, n_shadow // 2))]
    shadow_in = [m for h in half for m in (h, ~h)]                           # paired complements: every candidate
                                                                             # is OUT for exactly half the shadows
    for name in models:
        for variant in variants:
            tr = np.concatenate([mem_b, tr_m])
            m = fit_variant(name, variant, ds.X[tr], ds.y[tr], seed)
            p_c = m.predict_proba(ds.X[cand])[:, 1]                          # MIA needs probabilities, not margins
            conf = P.true_label_confidence(p_c, ds.y[cand])
            loss_attack = P.attack_report(np.log(conf[is_mem] + 1e-12), np.log(conf[~is_mem] + 1e-12))
            phi_t = P.logit_conf(conf)
            phi_out = np.full((len(cand), n_shadow), np.nan)
            for j, msk in enumerate(shadow_in):
                str_ = np.concatenate([cand[msk], tr_m])
                sm = fit_variant(name, variant, ds.X[str_], ds.y[str_], seed * 100 + j)
                cj = P.true_label_confidence(sm.predict_proba(ds.X[cand])[:, 1], ds.y[cand])
                phi_out[~msk, j] = P.logit_conf(cj)[~msk]                    # keep OUT observations only
            lira = P.lira_offline(phi_t, phi_out)
            lira_attack = P.attack_report(lira[is_mem], lira[~is_mem])
            te = np.concatenate([non_b, te_m]); s_te = M.score(m, ds.X[te])
            thr = fpr_threshold(M.score(m, ds.X[mem_b]), C.PRIMARY_FPR_BUDGET)   # train benign: optimistic, flagged
            util = panel(ds.y[te], s_te, thr, 0.01)
            out_rows.append({'part': 'mia', 'model': name, 'variant': variant, 'seed': seed,
                             **{f'loss_{k}': v for k, v in loss_attack.items()},
                             **{f'lira_{k}': v for k, v in lira_attack.items()},
                             'utility_auc_roc': util['auc_roc'], 'utility_auc_pr_1pct': util['auc_pr'],
                             'n_shadow': n_shadow})


def standardise(Xtr, *others):
    sc = StandardScaler().fit(M.signed_log1p(Xtr))                            # NOT privatised (limitation)
    f = lambda Z: np.clip(sc.transform(M.signed_log1p(Z)), -10, 10)
    return (f(Xtr),) + tuple(f(o) for o in others)


def dp_block(ds, parts, seed, eps_grid, out_rows, epochs):
    mem_b, non_b = parts[0]; tr_m, te_m = parts[1]
    tr = np.concatenate([mem_b, tr_m]); te = np.concatenate([non_b, te_m])
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
    a, b = next(gss.split(tr, groups=ds.groups[tr])); fit, val = tr[a], tr[b]
    Xf, Xv, Xt, Xm, Xn = standardise(ds.X[fit], ds.X[val], ds.X[te], ds.X[mem_b], ds.X[non_b])
    yf = ds.y[fit]; cw = dict(enumerate(compute_sample_weight('balanced', yf)[[np.argmax(yf == 0), np.argmax(yf == 1)]]))
    mem_in_fit = np.isin(mem_b, fit)
    for unit in ('row', 'group'):
        # 'clip_only' = the same Poisson-sampled, clipped SGD without noise: separates the clipping/sampling bias
        # from the cost of the Gaussian noise; 'non_private' = plain mini-batch SGD (reference)
        for eps, variant in [(e, 'dp') for e in eps_grid] + [(np.inf, 'clip_only'), (np.inf, 'non_private')]:
            n_units = len(np.unique(ds.groups[fit])) if unit == 'group' else len(fit)
            delta = 1.0 / n_units ** 2
            t0 = time.perf_counter()
            if variant in ('dp', 'clip_only'):
                w, b0, info = P.dp_sgd_logreg(Xf, yf, eps, delta, epochs=epochs, lot=C.DP_LOT, clip=C.DP_CLIP,
                                              groups=ds.groups[fit] if unit == 'group' else None,
                                              class_weight=cw, seed=seed)
            else:
                w, b0 = P.sgd_logreg(Xf, yf, epochs=epochs, batch=C.DP_LOT, class_weight=cw, seed=seed)
                info = {'epsilon': np.inf, 'noise_multiplier': 0.0, 'delta': delta, 'unit': unit, 'steps': None}
            sec = time.perf_counter() - t0
            sv, st_ = Xv @ w + b0, Xt @ w + b0
            thr = fpr_threshold(sv[ds.y[val] == 0], C.PRIMARY_FPR_BUDGET)
            u = panel(ds.y[te], st_, thr, 0.01)
            pm, pn = expit(Xm @ w + b0)[mem_in_fit], expit(Xn @ w + b0)
            att = P.attack_report(np.log(1 - pm + 1e-12), np.log(1 - pn + 1e-12))  # benign: true-label conf
            out_rows.append({'part': 'dp', 'unit': unit, 'dp_variant': variant, 'epsilon_target': eps, 'seed': seed, **info,
                             'auc_roc': u['auc_roc'], 'auc_pr_1pct': u['auc_pr'], 'mcc_1pct': u['mcc'],
                             'recall_1pct': u['recall'], 'fpr': u['fpr'], 'seconds': sec,
                             **{f'mia_{k}': v for k, v in att.items()}})


def fl_block(ds, parts, seed, out_rows, rounds):
    mem_b, non_b = parts[0]; tr_m, te_m = parts[1]
    tr = np.concatenate([mem_b, tr_m]); te = np.concatenate([non_b, te_m])
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
    a, b = next(gss.split(tr, groups=ds.groups[tr])); fit, val = tr[a], tr[b]
    Xf, Xv, Xt = standardise(ds.X[fit], ds.X[val], ds.X[te])
    yf = ds.y[fit]; cw = {0: float((yf == 1).sum() / max((yf == 0).sum(), 1)), 1: 1.0}
    def utility(w, b0):
        thr = fpr_threshold((Xv @ w + b0)[ds.y[val] == 0], C.PRIMARY_FPR_BUDGET)
        return panel(ds.y[te], Xt @ w + b0, thr, 0.01)
    wc, bc = P.sgd_logreg(Xf, yf, epochs=rounds * C.FL_LOCAL_EPOCHS, batch=64, class_weight=cw, seed=seed)
    uc = utility(wc, bc)
    out_rows.append({'part': 'fl', 'setting': 'centralised', 'alpha': np.nan, 'seed': seed,
                     'auc_roc': uc['auc_roc'], 'auc_pr_1pct': uc['auc_pr'], 'mcc_1pct': uc['mcc']})
    for alpha in C.FL_ALPHAS:
        client = P.dirichlet_partition(yf, ds.family[fit], ds.groups[fit], C.FL_CLIENTS, alpha, seed)
        sizes = np.bincount(client, minlength=C.FL_CLIENTS)
        w, b0 = P.fedavg_logreg(Xf, yf, client, rounds=rounds, local_epochs=C.FL_LOCAL_EPOCHS, batch=64,
                                class_weight=cw, seed=seed)
        u = utility(w, b0)
        out_rows.append({'part': 'fl', 'setting': 'fedavg', 'alpha': alpha, 'seed': seed,
                         'auc_roc': u['auc_roc'], 'auc_pr_1pct': u['auc_pr'], 'mcc_1pct': u['mcc'],
                         'client_size_min': int(sizes.min()), 'client_size_max': int(sizes.max()),
                         'gap_auc_pr_vs_central': uc['auc_pr'] - u['auc_pr'], 'gap_mcc_vs_central': uc['mcc'] - u['mcc']})


def main():
    a = parse('E5 privacy audit')
    out = outdir(a.mode, 'e5'); T = TimingLog(); rows = []
    ds = malmem_set('genuine', dedup=True)
    smoke = a.mode == 'smoke'
    models = tuple(a.models) if a.models else (('lgbm', 'rf') if smoke else ('lgbm', 'rf', 'xgb', 'lr'))
    for seed in a.seeds:
        set_seeds(seed); parts = partition(ds, seed)
        with T.time('mia', seed=seed):
            mia_block(ds, parts, seed, models, ('none', 'class_weight', 'smote'),
                      4 if smoke else C.LIRA_SHADOWS, rows)
        with T.time('dp', seed=seed):
            dp_block(ds, parts, seed, (1.0, 8.0) if smoke else C.DP_EPSILONS, rows, 5 if smoke else C.DP_EPOCHS)
        with T.time('fl', seed=seed):
            fl_block(ds, parts, seed, rows, 10 if smoke else C.FL_ROUNDS)
        pd.DataFrame(rows).to_csv(out / 'results.csv', index=False)
        print(f'seed {seed} done', flush=True)
    T.frame().to_csv(out / 'timing.csv', index=False); save_json(environment(), out / 'env.json')


if __name__ == '__main__':
    main()

"""explain.py - C4: SHAP as a shortcut detector, with stability and faithfulness checks (S8 Eqs. 91-93; S3 c. 9).

* TreeSHAP (path-dependent) on the boosted model of each protocol cell; global importance I_j = mean |phi_j|.
* Block shares: fraction of total mean |phi| on each plugin block (prediction of S2/C4: >= 50% on svcscan).
* Stability: Jaccard@k, Kendall tau and Spearman rho of global rankings across seeds, protocols and windows.
* Faithfulness (Warnecke-style descriptive accuracy): retrain without the top-k SHAP features vs k random ones.
"""
import numpy as np
import pandas as pd
from scipy import stats as st


def tree_shap(pipeline, X, max_rows=2000, seed=0):
    """Mean |phi| per input feature for a fitted tree pipeline (variance filter mapped back to all features)."""
    import shap
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), min(max_rows, len(X)), replace=False)
    keep = pipeline.named_steps['var'].get_support()
    Xk = pipeline.named_steps['var'].transform(X[rows])
    expl = shap.TreeExplainer(pipeline.named_steps['clf'], feature_perturbation='tree_path_dependent')
    phi = expl.shap_values(Xk)
    if isinstance(phi, list):                                                 # binary sklearn trees: class 1
        phi = phi[1]
    if phi.ndim == 3:
        phi = phi[:, :, 1]
    imp = np.zeros(X.shape[1]); imp[keep] = np.abs(phi).mean(0)
    return imp


def block_shares(imp, feature_names, blocks):
    tot = imp.sum()
    idx = {f: i for i, f in enumerate(feature_names)}
    return {b: float(imp[[idx[f] for f in fs]].sum() / tot) if tot > 0 else np.nan for b, fs in blocks.items()}


def topk(imp, k):
    return set(np.argsort(-imp)[:k])


def stability(imp_a, imp_b, k=10):
    """Eq. (93): agreement between two global importance vectors."""
    A, B = topk(imp_a, k), topk(imp_b, k)
    return {'jaccard_at_k': len(A & B) / len(A | B), 'kendall_tau': float(st.kendalltau(imp_a, imp_b).statistic),
            'spearman_rho': float(st.spearmanr(imp_a, imp_b).statistic)}


def deletion_test(fit_eval, ranking, ks, n_features, n_random=3, seed=0):
    """Retrain with the top-k features removed vs k random features removed; fit_eval(cols) -> AUC-PR."""
    rng = np.random.default_rng(seed); rows = []
    base = fit_eval(np.arange(n_features))
    for k in ks:
        top = set(ranking[:k])
        rows.append({'k': k, 'kind': 'shap_top', 'score': fit_eval(np.array([j for j in range(n_features)
                                                                             if j not in top])), 'base': base})
        for r in range(n_random):
            rnd = set(rng.choice(n_features, k, replace=False))
            rows.append({'k': k, 'kind': 'random', 'rep': r, 'base': base,
                         'score': fit_eval(np.array([j for j in range(n_features) if j not in rnd]))})
    return pd.DataFrame(rows)

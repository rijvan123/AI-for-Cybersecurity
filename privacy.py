"""privacy.py - C3 privacy audit (S8 Eqs. 75-86; S5 I16 + I18; S3 constraint 8).

1. SMOTE leakage: numpy SMOTE (k = 5) applied to training members only, compared with class weighting.
2. Membership inference on real minority records: loss-threshold attack (Yeom et al.) and offline LiRA
   (Carlini et al., S&P 2022) with shadow models; TPR reported at FPR 0.1% and 1% plus AUC and advantage.
3. DP-SGD logistic regression (Abadi et al., CCS 2016) with Poisson sampling, per-example or per-group clipping
   and an RDP accountant (Google dp_accounting); noise calibrated by bisection to the target epsilon.
4. FedAvg (McMahan et al., AISTATS 2017) over Dirichlet non-IID clients that never split a binary (Eqs. 83-86).
Limitation stated in the report: feature scaling statistics are computed non-privately; epsilon covers SGD only.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import expit

# --------------------------------------------------------------------------------------------- SMOTE
def smote(Xmin, n_new, k=5, seed=0):
    """Chawla et al. SMOTE: x_i + lam (x_nn - x_i), lam ~ U(0,1), nn among the k nearest minority rows."""
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    nbr = nn.kneighbors(Xmin, return_distance=False)[:, 1:]                  # drop self
    base = rng.integers(0, len(Xmin), n_new)
    mate = nbr[base, rng.integers(0, nbr.shape[1], n_new)]
    lam = rng.random((n_new, 1))
    return Xmin[base] + lam * (Xmin[mate] - Xmin[base]), base, mate


# --------------------------------------------------------------------------------------------- MIA
def _tpr_at(fpr, tpr, target):
    i = np.searchsorted(fpr, target, side='right') - 1
    return float(tpr[max(i, 0)])


def attack_report(score_members, score_nonmembers, fprs=(1e-3, 1e-2)):
    """ROC of an attack score (higher = member): AUC, membership advantage max(TPR-FPR), TPR at fixed FPRs."""
    y = np.r_[np.ones(len(score_members)), np.zeros(len(score_nonmembers))]
    s = np.r_[score_members, score_nonmembers]
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)                   # keep every achievable point
    out = {'attack_auc': float(roc_auc_score(y, s)), 'advantage': float(np.max(tpr - fpr)),
           'n_members': int(len(score_members)), 'n_nonmembers': int(len(score_nonmembers))}
    for f in fprs:
        out[f'tpr_at_fpr_{f:g}'] = _tpr_at(fpr, tpr, f)
    return out


def true_label_confidence(p1, y):
    """p(true label | x) from the malware probability."""
    return np.where(y == 1, p1, 1 - p1)


def logit_conf(p):
    """LiRA statistic phi = log(p / (1 - p)) of the true-label confidence (stable)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def lira_offline(phi_target, phi_out, min_out_var=None):
    """Offline LiRA score per candidate: (phi_target - mu_out) / sigma_out over shadow models that excluded it.
    phi_out: (n_candidates, n_shadows) with NaN where the candidate was IN that shadow's training set.
    sigma_out is per-example only when a candidate has >= min_out_var OUT shadows; otherwise the global (fixed)
    variance over all OUT observations is used, the variant Carlini et al. prefer when few shadow models exist
    (the 32-shadow switch point is a harness choice, config.LIRA_MIN_OUT_PER_EXAMPLE_VAR)."""
    from .config import LIRA_MIN_OUT_PER_EXAMPLE_VAR
    min_out_var = LIRA_MIN_OUT_PER_EXAMPLE_VAR if min_out_var is None else min_out_var
    n_out = np.sum(~np.isnan(phi_out), axis=1)
    mu = np.where(n_out > 0, np.nanmean(np.where(n_out[:, None] > 0, phi_out, 0.0), axis=1), np.nanmean(phi_out))
    sd_g = np.nanstd(phi_out)                                                  # global (fixed-variance LiRA)
    use_local = n_out >= max(min_out_var, 2)
    sd_l = np.nanstd(np.where(use_local[:, None], phi_out, 0.0), axis=1)
    sd = np.where(use_local, sd_l, sd_g) + 1e-6
    return (phi_target - mu) / sd


# --------------------------------------------------------------------------------------------- DP-SGD
def rdp_epsilon(noise_multiplier, q, steps, delta):
    from dp_accounting import dp_event
    from dp_accounting.rdp import rdp_privacy_accountant
    acc = rdp_privacy_accountant.RdpAccountant()
    acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
    return float(acc.get_epsilon(delta))


def calibrate_noise(target_eps, q, steps, delta, lo=0.3, hi=200.0, iters=40):
    """Smallest noise multiplier whose RDP epsilon <= target (bisection; epsilon decreases in sigma)."""
    if rdp_epsilon(hi, q, steps, delta) > target_eps:
        raise ValueError('target epsilon unreachable')
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if rdp_epsilon(mid, q, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def dp_sgd_logreg(X, y, eps, delta, epochs=30, lot=256, clip=1.0, lr=0.5, l2=1e-4, groups=None,
                  class_weight=None, seed=0):
    """DP-SGD for logistic regression (Eq. 80). Unit of privacy: one row, or one group when `groups` is given
    (per-group gradient sums are clipped, so neighbouring datasets differ in a whole binary). Returns (w, b, info)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    units = np.arange(n) if groups is None else np.unique(groups, return_inverse=True)[1]
    n_units = int(units.max() + 1)
    q = min(1.0, lot / n_units)                                               # Poisson sampling rate over units
    steps = int(np.ceil(epochs / q))
    sigma = calibrate_noise(eps, q, steps, delta) if np.isfinite(eps) else 0.0
    cw = np.ones(n) if class_weight is None else np.where(y == 1, class_weight[1], class_weight[0])
    members = [np.flatnonzero(units == u) for u in range(n_units)] if groups is not None else None
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        pick = np.flatnonzero(rng.random(n_units) < q)                        # Poisson-sampled lot of units
        if groups is None:
            rows = pick
            r = cw[rows] * (expit(X[rows] @ w + b) - y[rows])                 # per-example residual
            G = np.hstack([r[:, None] * X[rows], r[:, None]])                 # per-example gradients
        else:
            G = np.zeros((len(pick), d + 1))
            for j, u in enumerate(pick):
                rows = members[u]
                r = cw[rows] * (expit(X[rows] @ w + b) - y[rows])
                G[j] = np.r_[r @ X[rows], r.sum()]                            # per-group gradient sum
        if len(G):
            norms = np.linalg.norm(G, axis=1, keepdims=True)
            G = G / np.maximum(1.0, norms / clip)                             # clip to norm C
        g = G.sum(0) + rng.normal(0.0, sigma * clip, d + 1)                   # Gaussian mechanism
        g /= max(q * n_units, 1.0)                                            # divide by expected lot size
        w -= lr * (g[:d] + l2 * w); b -= lr * g[d]
    spent = rdp_epsilon(sigma, q, steps, delta) if sigma > 0 else np.inf
    return w, b, {'noise_multiplier': sigma, 'q': q, 'steps': steps, 'epsilon': spent, 'delta': delta,
                  'unit': 'row' if groups is None else 'group', 'n_units': n_units}


def sgd_logreg(X, y, epochs=30, batch=256, lr=0.5, l2=1e-4, class_weight=None, seed=0, w0=None, b0=0.0):
    """Non-private mini-batch SGD logistic regression (same optimiser as DP-SGD, for a fair utility gap)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d) if w0 is None else w0.copy(); b = b0
    cw = np.ones(n) if class_weight is None else np.where(y == 1, class_weight[1], class_weight[0])
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            rows = perm[i:i + batch]
            r = cw[rows] * (expit(X[rows] @ w + b) - y[rows])
            w -= lr * (r @ X[rows] / len(rows) + l2 * w); b -= lr * r.mean()
    return w, b


# --------------------------------------------------------------------------------------------- FedAvg
def dirichlet_partition(y, family, groups, n_clients, alpha, seed=0):
    """Eq. (86): benign groups spread uniformly; each malware family's binaries split by Dir(alpha) proportions.
    Binaries (groups) are never split across clients. Returns client id per row."""
    rng = np.random.default_rng(seed)
    client = np.full(len(y), -1)
    bg = np.unique(groups[y == 0]); rng.shuffle(bg)
    for i, g in enumerate(bg):
        client[groups == g] = i % n_clients
    for f in np.unique(family[y == 1]):
        gs = np.unique(groups[(y == 1) & (family == f)]); rng.shuffle(gs)
        p = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(p) * len(gs)).round().astype(int)[:-1]
        for c, part in enumerate(np.split(gs, cuts)):
            client[np.isin(groups, part)] = c
    return client


def fedavg_logreg(X, y, client, rounds=50, local_epochs=1, batch=64, lr=0.5, l2=1e-4, class_weight=None,
                  seed=0, frac=1.0):
    """Eqs. (83-85): synchronous FedAvg; each sampled client runs local SGD from the global model; the server
    averages client models weighted by n_k. No secure aggregation (not modelled)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]; w = np.zeros(d); b = 0.0
    ids = np.unique(client[client >= 0])
    for r in range(rounds):
        S = rng.choice(ids, max(1, int(round(frac * len(ids)))), replace=False)
        W, B, N = [], [], []
        for c in S:
            rows = np.flatnonzero(client == c)
            if len(rows) == 0:
                continue
            wc, bc = sgd_logreg(X[rows], y[rows], epochs=local_epochs, batch=batch, lr=lr, l2=l2,
                                class_weight=class_weight, seed=seed * 1000 + r * 37 + int(c), w0=w, b0=b)
            W.append(wc); B.append(bc); N.append(len(rows))
        N = np.asarray(N, float)
        w = np.average(np.vstack(W), axis=0, weights=N); b = float(np.average(B, weights=N))
    return w, b

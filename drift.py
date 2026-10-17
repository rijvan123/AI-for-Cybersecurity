"""drift.py - feature drift between a training reference and later windows (S8 Eqs. 17, 69-74).

Cut points are training-quantile edges (Eq. 17); per-feature PSI (= Jeffreys divergence, Eq. 71) and base-2
Jensen-Shannon divergence (Eq. 72) are aggregated over all features, per block and importance-weighted (Eq. 73).
scipy's jensenshannon returns the JS *distance*, so it is not used; JSD is computed directly.
"""
import numpy as np
import pandas as pd
from .config import DRIFT_BINS, DRIFT_EPS


def fit_edges(X_ref, bins=DRIFT_BINS):
    """Interior train-quantile cut points per feature (ties collapse to unique edges)."""
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return [np.unique(np.quantile(X_ref[:, j], qs)) for j in range(X_ref.shape[1])]


def binned(X, edges, eps=DRIFT_EPS):
    """Eq. (69): smoothed bin probabilities per feature (list of arrays, one per feature)."""
    out = []
    for j, e in enumerate(edges):
        b = np.searchsorted(e, X[:, j], side='right')                       # bin index b_j(x), Eq. 17
        c = np.bincount(b, minlength=len(e) + 1).astype(float)
        out.append((c + eps) / (c.sum() + eps * len(c)))
    return out


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def psi(p, q):
    """Eq. (71): PSI = sum (q - p) ln(q / p) = KL(q||p) + KL(p||q)."""
    return float(np.sum((q - p) * np.log(q / p)))


def jsd(p, q):
    """Eq. (72): base-2 Jensen-Shannon divergence in [0, 1]."""
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def window_drift(X_ref, X_win, edges=None, feature_names=None):
    """Per-feature PSI and JSD of a window against the reference; returns a DataFrame."""
    edges = edges or fit_edges(X_ref)
    P, Q = binned(X_ref, edges), binned(X_win, edges)
    rows = [{'feature': (feature_names[j] if feature_names is not None else j),
             'psi': psi(P[j], Q[j]), 'jsd': jsd(P[j], Q[j]), 'tv_bound': np.sqrt(psi(P[j], Q[j]) / 2)}  # Eq. 74
            for j in range(len(edges))]
    return pd.DataFrame(rows)


def aggregate(df, blocks=None, weights=None):
    """Eq. (73): mean JSD over features, per block, and importance-weighted (weights normalised to 1)."""
    out = {'jsd_mean': float(df['jsd'].mean()), 'psi_mean': float(df['psi'].mean()),
           'psi_alarm_features': int((df['psi'] > 0.25).sum())}
    if weights is not None:
        w = np.asarray(weights, float); w = w / w.sum() if w.sum() > 0 else w
        out['jsd_weighted'] = float(np.sum(w * df['jsd'].to_numpy()))
    if blocks:
        for b, idx in blocks.items():
            out[f'jsd_{b}'] = float(df['jsd'].to_numpy()[idx].mean())
    return out

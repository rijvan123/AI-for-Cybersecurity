"""metrics.py - full metric panel at any deployment base rate (S8 Eqs. 54, 58-59, 63-68; S3 constraint 4).

Prevalence handling: a test set with n+ malware and n- benign rows is re-weighted to prevalence pi with
w+ = pi * N / n+ and w- = (1 - pi) * N / n-. Weighted confusion counts equal the expectation of the counts
over Eq. (7) subsampling, so precision, F1, MCC, AUC-PR, alarms and expected cost at pi are computed without
subsampling noise; AUC-ROC and FPR@95%TPR are prevalence-invariant. Empirical subsampling replicates are kept
only to test that equivalence (prediction H2).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from .config import ALERT_UNIT


def class_weights(y, pi):
    """Per-row weights that move a test set to prevalence pi (None = keep the empirical prevalence)."""
    y = np.asarray(y)
    if pi is None:
        return np.ones(len(y))
    n, npos, nneg = len(y), (y == 1).sum(), (y == 0).sum()
    return np.where(y == 1, pi * n / max(npos, 1), (1 - pi) * n / max(nneg, 1))


def confusion(y, pred, w):
    tp = float(w[(y == 1) & pred].sum()); fn = float(w[(y == 1) & ~pred].sum())
    fp = float(w[(y == 0) & pred].sum()); tn = float(w[(y == 0) & ~pred].sum())
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def mcc(tp, fp, tn, fn):
    """Eq. (65); 0 when a marginal is empty (scikit-learn convention)."""
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def fpr_at_tpr(y, s, target=0.95):
    """FPR at the first ROC point reaching TPR >= target (threshold-free, prevalence-invariant).
    drop_intermediate=False keeps every achievable point; the default would drop collinear points and can
    overstate the FPR when the target TPR lies on a straight ROC segment."""
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    i = np.searchsorted(tpr, target, side='left')
    return float(fpr[min(i, len(fpr) - 1)])


def panel(y, s, thr, pi=None, cost_ratios=(1, 10, 100)):
    """Metric panel at threshold thr and prevalence pi (None = as sampled). Returns a flat dict."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    w = class_weights(y, pi)
    pred = s >= thr
    tp, fp, tn, fn = confusion(y, pred, w)
    prec, rec = safe_div(tp, tp + fp), safe_div(tp, tp + fn)                  # Eq. 63
    fpr, fnr = safe_div(fp, fp + tn), safe_div(fn, tp + fn)
    n = tp + fp + tn + fn
    out = {'precision': prec, 'recall': rec, 'f1': safe_div(2 * tp, 2 * tp + fp + fn),   # Eq. 64
           'mcc': mcc(tp, fp, tn, fn), 'fpr': fpr, 'fnr': fnr,
           'accuracy': safe_div(tp + tn, n), 'balanced_accuracy': 0.5 * (rec + (1 - fpr)),
           'alarms_per_10k': ALERT_UNIT * safe_div(tp + fp, n),
           'false_alarms_per_10k': ALERT_UNIT * safe_div(fp, n),
           'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    if len(np.unique(y)) == 2:
        out['auc_roc'] = float(roc_auc_score(y, s))                           # Eq. 66 (invariant to pi)
        out['auc_pr'] = float(average_precision_score(y, s, sample_weight=w))  # Eq. 67 at prevalence pi
        out['fpr_at_95tpr'] = fpr_at_tpr(y, s, 0.95)
    else:
        out['auc_roc'] = out['auc_pr'] = out['fpr_at_95tpr'] = np.nan
    pi_eff = safe_div(tp + fn, n)
    for r in cost_ratios:                                                     # Eq. 54 with C_FP = 1, C_FN = r
        out[f'exp_cost_r{r}'] = pi_eff * r * fnr + (1 - pi_eff) * fpr
    return out


def precision_at(tpr, fpr, pi):
    """Eq. (58): Bayesian detection rate (Axelsson)."""
    den = pi * tpr + (1 - pi) * fpr
    return pi * tpr / den if den > 0 else np.nan


def fpr_max(tpr, pi, p_star):
    """Eq. (59): largest FPR that still gives precision >= p_star at prevalence pi."""
    return pi * tpr * (1 - p_star) / ((1 - pi) * p_star)


def aut(values):
    """Eq. (68): TESSERACT area under time, trapezoid rule normalised by N - 1 slots."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v[0]) if len(v) else np.nan
    return float(np.sum((v[1:] + v[:-1]) / 2) / (len(v) - 1))


def expected_cost(fnr, fpr, pi, c_fn, c_fp=1.0):
    """Eq. (54)."""
    return pi * c_fn * fnr + (1 - pi) * c_fp * fpr

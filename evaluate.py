"""evaluate.py - shared fit / score / evaluate loop used by every experiment (S7 Algorithm 1, lines 10-17).

One call = one (model, split): fit on train only, thresholds and calibration from val only, then the metric panel
on test at every prevalence pi and every operating-point rule. Scores are returned so curves, DeLong tests and
group bootstraps can be recomputed later without refitting.
"""
import time
import numpy as np
from . import models as M
from .thresholds import operating_points, logit
from .metrics import panel
from .config import FPR_BUDGETS, PREVALENCES, COST_RATIOS


def fit_score(name, X, y, sp, seed, groups=None, time_key=None, feature_names=None):
    """Fit model `name` on sp.train; return scores on val/test, timings and model artefacts."""
    model = M.make_model(name, seed)
    g_tr = groups[sp.train] if groups is not None else None
    tk = time_key[sp.train] if time_key is not None else None
    fit_s = M.fit(model, X[sp.train], y[sp.train], groups=g_tr, time_key=tk)
    t0 = time.perf_counter(); s_val = M.score(model, X[sp.val]); s_test = M.score(model, X[sp.test])
    pred_s = time.perf_counter() - t0
    out = {'model': model, 's_val': s_val, 's_test': s_test, 'fit_s': fit_s,
           'us_per_sample': 1e6 * pred_s / max(len(sp.val) + len(sp.test), 1),
           'has_proba': M.is_probability(name)}
    if name == 'stump' and feature_names is not None:
        out['stump_feature'], out['stump_threshold'] = M.stump_feature(model, feature_names)
    return out


def evaluate(y_test, s_test, s_val, y_val, has_proba=True, pis=PREVALENCES, alphas=FPR_BUDGETS,
             cost_ratios=COST_RATIOS, rules=('fpr', 'cost', 'half')):
    """Metric rows for every (prevalence, rule). 'cost' thresholds are matched to the evaluated prevalence."""
    ops, cal = operating_points(s_val, y_val, alphas, pis, cost_ratios, half=0.5 if has_proba else 0.0)
    rows = []
    z_test = None
    for pi in pis:
        for (rule, a, r), thr in ops.items():
            if rule not in rules:
                continue
            if rule == 'cost':
                if a != pi:                                                    # cost rule targets this pi only
                    continue
                if z_test is None:
                    z_test = logit(cal(s_test))                                # Eq. 57 operates on calibrated logits
                m = panel(y_test, z_test, thr, pi)
                rows.append({'pi': pi, 'rule': 'cost', 'alpha': np.nan, 'cost_ratio': r, 'threshold': thr, **m})
            else:
                m = panel(y_test, s_test, thr, pi)
                rows.append({'pi': pi, 'rule': rule, 'alpha': a if rule == 'fpr' else np.nan,
                             'cost_ratio': np.nan, 'threshold': thr, **m})
    return rows, cal.kind

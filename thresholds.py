"""thresholds.py - operating points fixed on validation data only (S3 constraint 4; S8 Eqs. 30, 52-57).

Three rules are reported side by side so no result depends on a test-set threshold:
  'fpr'  : Neyman-Pearson threshold with validation FPR <= alpha (Eq. 30) - the primary rule;
  'cost' : Bayes-optimal threshold on calibrated probabilities after prior shift to the deployment base rate
           (Eqs. 55-57);
  'half' : 0.5 on the model's own probability (or 0 on an SVM margin) - the literature default, for comparison.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fpr_threshold(scores_benign_val, alpha):
    """Smallest threshold tau with #(benign validation scores >= tau) <= floor(alpha * n): decide malware iff s >= tau."""
    s = np.sort(np.asarray(scores_benign_val, dtype=np.float64))[::-1]        # descending
    if len(s) == 0:                                                           # no benign validation rows: refuse
        raise ValueError('fpr_threshold: empty benign validation set')
    k = int(np.floor(alpha * len(s)))                                         # benign alarms allowed
    if k >= len(s):
        return -np.inf
    return float(np.nextafter(s[k], np.inf))                                  # strictly above the (k+1)-th score


class Calibrator:
    """Isotonic when the validation slice is large and both classes are present (>= 50 each), else Platt."""

    def fit(self, s, y):
        s = np.asarray(s, float); y = np.asarray(y)
        if min((y == 0).sum(), (y == 1).sum()) >= 50 and len(y) >= 1000:
            self.kind = 'isotonic'
            self.m = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1 - 1e-6).fit(s, y)
        else:
            self.kind = 'platt'
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
        self.pi_cal = float(np.mean(y))                                       # prevalence of the calibration slice
        return self

    def __call__(self, s):
        s = np.asarray(s, float)
        if self.kind == 'isotonic':
            return np.clip(self.m.predict(s), 1e-6, 1 - 1e-6)
        return self.m.predict_proba(s.reshape(-1, 1))[:, 1]


def prior_shift(p, pi_from, pi_to):
    """Eq. (56): posterior under a new prior with unchanged class-conditional densities (Saerens et al.)."""
    a = (pi_to / pi_from) * p
    b = ((1 - pi_to) / (1 - pi_from)) * (1 - p)
    return a / (a + b)


def cost_threshold_logit(pi_cal, pi_dep, c_fp, c_fn):
    """Eq. (57): decide malware iff logit(p_cal) >= ln(C_FP/C_FN) + ln((1-pi)/pi) - ln((1-pi_cal)/pi_cal)."""
    return np.log(c_fp / c_fn) + np.log((1 - pi_dep) / pi_dep) - np.log((1 - pi_cal) / pi_cal)


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def operating_points(s_val, y_val, alphas, pis, cost_ratios, half=0.5):
    """All thresholds a model needs, computed from validation scores only."""
    ops = {('half', None, None): half}
    for a in alphas:
        ops[('fpr', a, None)] = fpr_threshold(s_val[y_val == 0], a)
    cal = Calibrator().fit(s_val, y_val)
    for pi in pis:
        for r in cost_ratios:                                                 # r = C_FN / C_FP
            ops[('cost', pi, r)] = cost_threshold_logit(cal.pi_cal, pi, 1.0, float(r))
    return ops, cal

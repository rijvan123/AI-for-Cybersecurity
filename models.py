"""models.py - CPU model zoo with leakage-free pipelines (S8 Eqs. 10-15, 25-51; S7 M4).

Every learner is an sklearn Pipeline whose preprocessing (signed log, variance filter, scaling, RFE) is fitted on
the training rows only. Hyperparameters are fixed, literature-standard values (no test-set tuning); the optional
tune() runs a grouped inner random search on the training rows only. Deep models are limited to a shallow MLP
proof of concept on a fixed-seed subsample, as the project rules require.
"""
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from .config import N_JOBS, KNN_MAX_TRAIN, MLP_MAX_TRAIN


def signed_log1p(X):
    """Eq. (11) extended to signed inputs (EMBER hashed features can be negative): sign(x) * ln(1 + |x|)."""
    return np.sign(X) * np.log1p(np.abs(X))


def _scaled(est):
    return Pipeline([('log', FunctionTransformer(signed_log1p, validate=False)),   # Eq. 11
                     ('var', VarianceThreshold(0.0)),                             # Eq. 15 (fitted on train)
                     ('sc', StandardScaler()),                                    # Eq. 13 (train mu, s)
                     ('clf', est)])


def _trees(est):
    return Pipeline([('var', VarianceThreshold(0.0)), ('clf', est)])              # trees are scale-invariant


class SubsampleFit(BaseEstimator, ClassifierMixin):
    """Fit the wrapped estimator on a fixed-seed, class-stratified subsample of at most max_n training rows."""

    def __init__(self, estimator, max_n, seed=0):
        self.estimator, self.max_n, self.seed = estimator, max_n, seed

    def fit(self, X, y, sample_weight=None):
        idx = np.arange(len(y))
        if len(y) > self.max_n:
            rng = np.random.default_rng(self.seed)
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                             int(round(self.max_n * np.mean(y == c))), replace=False)
                                  for c in np.unique(y)])
        self.est_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.est_.classes_
        self.n_fit_ = len(idx)
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def predict(self, X):
        return self.est_.predict(X)


class GroupedStack(BaseEstimator, ClassifierMixin):
    """Two-layer stack of Carrier et al. (NB, RF, DT -> LR), S8 Eq. (51), with leakage-safe out-of-fold inputs:
    the OOF folds are grouped by sample identity (or time-ordered when `time_key` is given), never row-random."""

    def __init__(self, seed=0, n_folds=5):
        self.seed, self.n_folds = seed, n_folds

    def _bases(self):
        return [('nb', _scaled(GaussianNB())),
                ('rf', _trees(RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=self.seed,
                                                     class_weight='balanced_subsample'))),
                ('dt', _trees(DecisionTreeClassifier(random_state=self.seed, class_weight='balanced')))]

    def fit(self, X, y, groups=None, time_key=None):
        if time_key is not None:                                                # forward-chaining blocks
            order = np.argsort(time_key, kind='mergesort'); blocks = np.array_split(order, self.n_folds + 1)
            folds = [(np.concatenate(blocks[:i]), blocks[i]) for i in range(1, len(blocks))]
            oof_rows = np.concatenate([te for _, te in folds])
        elif groups is not None:
            folds = list(GroupKFold(n_splits=self.n_folds).split(X, y, groups)); oof_rows = np.arange(len(y))
        else:
            folds = list(StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed).split(X, y))
            oof_rows = np.arange(len(y))
        Z = np.full((len(y), 3), np.nan)
        for tr, te in folds:
            for j, (_, b) in enumerate(self._bases()):
                Z[te, j] = clone(b).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        self.meta_ = LogisticRegression(max_iter=1000).fit(Z[oof_rows], y[oof_rows])
        self.bases_ = [clone(b).fit(X, y) for _, b in self._bases()]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Z = np.column_stack([b.predict_proba(X)[:, 1] for b in self.bases_])
        return self.meta_.predict_proba(Z)


def make_model(name, seed=0, n_features=None):
    """Factory for the model zoo. Every model exposes predict_proba or decision_function."""
    if name == 'stump':                                    # best single feature + threshold (shortcut probe)
        return _trees(DecisionTreeClassifier(max_depth=1, random_state=seed, class_weight='balanced'))
    if name == 'lr':                                       # Eq. 31/34: logistic regression, L2, balanced
        return _scaled(LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced'))
    if name == 'linsvm':                                   # Eq. 36-38: linear SVM (squared hinge, balanced)
        return _scaled(LinearSVC(C=0.1, class_weight='balanced', max_iter=10000, random_state=seed))
    if name == 'gnb':                                      # Eq. 27-28: Gaussian naive Bayes on log-scaled data
        return _scaled(GaussianNB(var_smoothing=1e-9))
    if name == 'knn':                                      # Eq. 25: k-NN, distance-weighted, capped reference set
        return _scaled(SubsampleFit(KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=N_JOBS),
                                    KNN_MAX_TRAIN, seed))
    if name == 'dt':                                       # Eq. 42: unpruned CART (stack component)
        return _trees(DecisionTreeClassifier(random_state=seed, class_weight='balanced'))
    if name == 'rf':                                       # Eq. 43: probability-averaging random forest
        return _trees(RandomForestClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=seed,
                                             class_weight='balanced_subsample'))
    if name == 'et':
        return _trees(ExtraTreesClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=seed,
                                           class_weight='balanced_subsample'))
    if name == 'xgb':                                      # Eq. 44-50: regularised boosting (hist splits)
        from xgboost import XGBClassifier
        return _trees(XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                    colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', n_jobs=N_JOBS,
                                    random_state=seed, eval_metric='logloss', verbosity=0))
    if name == 'lgbm':
        from lightgbm import LGBMClassifier
        return _trees(LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8, class_weight='balanced',
                                     n_jobs=N_JOBS, random_state=seed, verbose=-1))
    if name == 'lgbm_mono':                                # monotone-constrained variant (adversarial module)
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, class_weight='balanced', n_jobs=N_JOBS, random_state=seed,
                              verbose=-1)
    if name == 'stack_carrier':                            # Carrier et al. two-layer stack, grouped OOF
        return GroupedStack(seed=seed)
    if name == 'xmal':                                     # Alani et al. XMal: RFE to 5 features, then XGBoost
        from xgboost import XGBClassifier
        base = dict(tree_method='hist', n_jobs=N_JOBS, random_state=seed, eval_metric='logloss', verbosity=0)
        return Pipeline([('var', VarianceThreshold(0.0)),
                         ('rfe', RFE(XGBClassifier(n_estimators=100, max_depth=4, **base),
                                     n_features_to_select=5, step=5)),
                         ('clf', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, **base))])
    if name == 'mlp_poc':                                  # Eq. 31-34: shallow MLP proof of concept (subsample)
        return _scaled(SubsampleFit(MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                  max_iter=300, random_state=seed), MLP_MAX_TRAIN, seed))
    raise KeyError(name)


def fit(model, X, y, groups=None, time_key=None, balanced=True):
    """Fit with balanced sample weights where the final estimator has no class_weight (XGBoost, MLP excluded)."""
    t0 = time.perf_counter()
    if isinstance(model, GroupedStack):
        model.fit(X, y, groups=groups, time_key=time_key)
    elif isinstance(model, Pipeline) and type(model.steps[-1][1]).__name__ == 'XGBClassifier' and balanced:
        model.fit(X, y, clf__sample_weight=compute_sample_weight('balanced', y))
    else:
        model.fit(X, y)
    return time.perf_counter() - t0


SCORE_KIND = {'lr': 'logit', 'linsvm': 'margin', 'gnb': 'log_odds', 'lgbm': 'raw_margin', 'xgb': 'raw_margin',
              'lgbm_mono': 'raw_margin', 'xmal': 'raw_margin'}               # unsaturated scores (see score)


def is_probability(name):
    """True when score() returns a probability (so the literature 'half' rule is 0.5; otherwise it is 0)."""
    return name not in SCORE_KIND


def score(model, X):
    """Continuous, unsaturated malware score.

    Linear models return the decision value w.z + b (Eq. 36/40), Gaussian NB the log-odds
    log P(1|x) - log P(0|x) (Eq. 29) and boosted trees their raw additive margin (Eq. 44): on 2,381-dimensional
    EMBER/BODMAS features predict_proba saturates at exactly 0 or 1 for many rows, and ties at 1.0 make a
    validation FPR threshold degenerate (recall 0). Forests, trees, k-NN and the MLP return P(y=1|x)."""
    final = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(final, 'decision_function') and not hasattr(final, 'predict_proba') or \
            isinstance(final, LogisticRegression):
        return model.decision_function(X)
    if isinstance(final, GaussianNB):
        lp = model.predict_log_proba(X)
        return lp[:, 1] - lp[:, 0]
    kind = type(final).__name__
    if kind == 'LGBMClassifier':                                               # boosting: raw additive margin
        return model.predict(X, raw_score=True)
    if kind == 'XGBClassifier':
        return model.predict(X, output_margin=True)
    return model.predict_proba(X)[:, 1]


def stump_feature(model, feature_names):
    """Feature and threshold chosen by the depth-1 tree (reported by the shortcut audit)."""
    tree = model.named_steps['clf'].tree_; keep = model.named_steps['var'].get_support()
    names = np.asarray(feature_names)[keep]
    return (names[tree.feature[0]] if tree.feature[0] >= 0 else None), float(tree.threshold[0])


def importances(model, feature_names):
    """Impurity/gain importances mapped back through the variance filter (zeros for filtered features), sum 1.
    LightGBM's feature_importances_ defaults to split counts, so its total gain is read from the booster;
    XGBoost's sklearn wrapper reports average gain for tree boosters; forests report mean impurity decrease."""
    clf = model.named_steps['clf']; keep = model.named_steps['var'].get_support()
    if not hasattr(clf, 'feature_importances_'):
        return None
    vals = clf.feature_importances_
    if hasattr(clf, 'booster_') and hasattr(clf.booster_, 'feature_importance'):   # LightGBM
        vals = clf.booster_.feature_importance(importance_type='gain')
    imp = np.zeros(len(feature_names)); imp[keep] = vals
    return imp / imp.sum() if imp.sum() > 0 else imp                       # normalised to sum to 1 (comparable across models)


def tune(name, X, y, groups, seed, n_iter=20):
    """Optional inner random search on the training rows only (grouped inner CV), used for S10 sensitivity."""
    from sklearn.model_selection import RandomizedSearchCV
    grids = {'rf': {'clf__n_estimators': [200, 300, 500], 'clf__max_features': ['sqrt', 0.3, 0.5],
                    'clf__min_samples_leaf': [1, 2, 5]},
             'lgbm': {'clf__num_leaves': [15, 31, 63, 127], 'clf__learning_rate': [0.02, 0.05, 0.1],
                      'clf__n_estimators': [200, 400, 800], 'clf__min_child_samples': [5, 20, 50]},
             'xgb': {'clf__max_depth': [3, 6, 9], 'clf__learning_rate': [0.05, 0.1, 0.2],
                     'clf__n_estimators': [200, 300, 600]}}
    rs = RandomizedSearchCV(make_model(name, seed), grids[name], n_iter=n_iter, scoring='average_precision',
                            cv=GroupKFold(3), random_state=seed, n_jobs=1, refit=True)
    rs.fit(X, y, groups=groups)
    return rs.best_estimator_, rs.best_params_

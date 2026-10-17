"""adversarial.py - adversarial exposure (S8 Eqs. 87-90; S5 I14; S3 constraint 7).

(a) Exact linear evasion (prediction H5 of S8): for s(z) = w.z + b and threshold tau, the minimal L_p evasion
    radius is r_p = (s - tau) / ||w||_{p*}; FGSM along -sign(w) (p = inf) or -w/||w|| (p = 2) attains it, so
    the empirical evasion rate must equal the fraction with r <= eps. The add-only radius restricts the
    perturbation to increasable coordinates with negative weight.
(b) Problem-space-consistent greedy attack on CIC-MalMem-2022 features: actions an attacker's process can take
    (spawn benign-looking processes, open handles of a type, load DLLs; with admin rights, register services or
    drivers) update every affected count AND the dependent averages, preserving the arithmetic invariants of
    data_malmem.INVARIANTS. It is still a feature-space simulation: no binary is executed (stated as such).
"""
import numpy as np
import pandas as pd

HANDLE_TYPES = ['handles.nfile', 'handles.nevent', 'handles.ndesktop', 'handles.nkey', 'handles.nthread',
                'handles.ndirectory', 'handles.nsemaphore', 'handles.ntimer', 'handles.nsection', 'handles.nmutant']
USER_ACTIONS = ['spawn_process', 'load_dlls'] + [f'open_{h.split(".n", 1)[1]}' for h in HANDLE_TYPES]
ADMIN_ACTIONS = USER_ACTIONS + ['create_service_own', 'create_service_shared', 'load_driver']

# features an unprivileged process can only increase (counts); used for add-only radii and monotone defences
INCREASE_ONLY = ['pslist.nproc', 'dlllist.ndlls', 'handles.nhandles'] + HANDLE_TYPES
# additionally increasable with administrator rights (service / driver registration)
ADMIN_INCREASE_ONLY = INCREASE_ONLY + ['svcscan.nservices', 'svcscan.nactive', 'svcscan.process_services',
                                       'svcscan.shared_process_services', 'svcscan.kernel_drivers', 'modules.nmodules']


# --------------------------------------------------------------------------------------------- (a) linear
def linear_radius(s_minus_tau, w, p='inf', add_only_mask=None):
    """Eq. (89): minimal evasion radius in z-space for detected rows (s - tau > 0)."""
    if add_only_mask is not None:                                             # only increase coordinates j with w_j < 0
        useful = add_only_mask & (w < 0)
        denom = np.abs(w[useful]).sum() if p == 'inf' else np.sqrt((w[useful] ** 2).sum())
        return np.full_like(s_minus_tau, np.inf) if denom == 0 else s_minus_tau / denom
    denom = np.abs(w).sum() if p == 'inf' else np.sqrt((w ** 2).sum())        # dual norm ||w||_{p*}
    return s_minus_tau / denom


def fgsm_linear(Z, w, b, eps, p='inf'):
    """Eq. (88) without projection: the optimal one-step attack on a linear score."""
    step = np.sign(w) if p == 'inf' else w / np.linalg.norm(w)
    return Z - eps * step, (Z - eps * step) @ w + b


def evasion_rate_check(Z_det, w, b, tau, eps_grid, p='inf'):
    """H5: compare the FGSM empirical evasion rate with the radius-predicted fraction for every eps."""
    s = Z_det @ w + b
    r = linear_radius(s - tau, w, p)
    rows = []
    for e in eps_grid:
        _, s_adv = fgsm_linear(Z_det, w, b, e, p)
        emp = float(np.mean(s_adv < tau)); pred = float(np.mean(r < e))
        rows.append({'p': p, 'eps': e, 'evasion_empirical': emp, 'evasion_predicted': pred,
                     'abs_diff_rows': int(abs(np.sum(s_adv < tau) - np.sum(r < e)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- (b) problem space
class MemoryActionModel:
    """Consistent feature updates for attacker actions on one memory snapshot (raw feature space, 55 columns)."""

    def __init__(self, feature_names, benign_reference):
        self.f = list(feature_names); self.ix = {c: i for i, c in enumerate(self.f)}
        B = pd.DataFrame(benign_reference, columns=self.f)
        self.t_thr = float(B['pslist.avg_threads'].median())                  # threads of a typical process
        self.t_hdl = float(B['pslist.avg_handlers'].median())                 # handles of a typical process
        self.t_dll = float(B['dlllist.avg_dlls_per_proc'].median())           # DLLs of a typical process
        tot = B[HANDLE_TYPES].sum(1).replace(0, np.nan)
        self.mix = (B[HANDLE_TYPES].div(tot, axis=0)).median().fillna(0).to_numpy()
        self.mix = self.mix / self.mix.sum()                                   # benign median handle-type mix

    def _procs(self, x, avg_col, cnt_col):
        a = x[self.ix[avg_col]]
        return np.round(x[self.ix[cnt_col]] / a) if a > 0 else x[self.ix['pslist.nproc']]

    def apply(self, x, action, n=1):
        """Return a new feature vector after `action` (n repetitions)."""
        x = x.astype(float).copy(); ix = self.ix
        if action == 'spawn_process':
            p_old = x[ix['pslist.nproc']]; p_new = p_old + n
            x[ix['pslist.avg_threads']] = (x[ix['pslist.avg_threads']] * p_old + n * round(self.t_thr)) / p_new
            x[ix['pslist.avg_handlers']] = (x[ix['pslist.avg_handlers']] * p_old + n * round(self.t_hdl)) / p_new
            x[ix['pslist.nproc']] = p_new
            q = self._procs(x, 'dlllist.avg_dlls_per_proc', 'dlllist.ndlls')
            x[ix['dlllist.ndlls']] += n * round(self.t_dll)
            x[ix['dlllist.avg_dlls_per_proc']] = x[ix['dlllist.ndlls']] / (q + n)
            h = self._procs(x, 'handles.avg_handles_per_proc', 'handles.nhandles')
            add = np.floor(self.mix * n * round(self.t_hdl))
            for j, c in enumerate(HANDLE_TYPES):
                x[ix[c]] += add[j]
            x[ix['handles.nhandles']] += add.sum()
            x[ix['handles.avg_handles_per_proc']] = x[ix['handles.nhandles']] / (h + n)
        elif action == 'load_dlls':
            q = self._procs(x, 'dlllist.avg_dlls_per_proc', 'dlllist.ndlls')
            x[ix['dlllist.ndlls']] += 5 * n                                    # five DLLs per action
            x[ix['dlllist.avg_dlls_per_proc']] = x[ix['dlllist.ndlls']] / max(q, 1)
        elif action.startswith('open_'):
            c = 'handles.n' + action[5:]
            h = self._procs(x, 'handles.avg_handles_per_proc', 'handles.nhandles')
            x[ix[c]] += 10 * n; x[ix['handles.nhandles']] += 10 * n          # ten handles per action
            x[ix['handles.avg_handles_per_proc']] = x[ix['handles.nhandles']] / max(h, 1)
            x[ix['pslist.avg_handlers']] += 10 * n / max(x[ix['pslist.nproc']], 1)
        elif action in ('create_service_own', 'create_service_shared'):
            x[ix['svcscan.nservices']] += n; x[ix['svcscan.nactive']] += n
            key = 'svcscan.process_services' if action.endswith('own') else 'svcscan.shared_process_services'
            x[ix[key]] += n
        elif action == 'load_driver':
            x[ix['svcscan.nservices']] += n; x[ix['svcscan.kernel_drivers']] += n
            x[ix['modules.nmodules']] += n
        else:
            raise KeyError(action)
        return x


def greedy_attack(score_fn, X_det, tau, model, actions, budget):
    """Query-based greedy search: at each step apply, per sample, the action that lowers the score most.
    score_fn(X) -> scores; returns evasion step per sample (np.inf if not evaded within budget) and final X."""
    X = X_det.astype(float).copy(); n = len(X)
    evaded_at = np.full(n, np.inf); active = np.arange(n)
    s = score_fn(X); evaded_at[s < tau] = 0
    active = active[s >= tau]
    for step in range(1, budget + 1):
        if len(active) == 0:
            break
        cand = np.vstack([np.vstack([model.apply(X[i], a) for a in actions]) for i in active])
        sc = score_fn(cand).reshape(len(active), len(actions))
        best = sc.argmin(1)
        for k, i in enumerate(active):
            X[i] = cand[k * len(actions) + best[k]]
        s_new = sc[np.arange(len(active)), best]
        done = s_new < tau
        evaded_at[active[done]] = step
        active = active[~done]
    return evaded_at, X

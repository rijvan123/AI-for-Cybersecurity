"""E3 - C2 temporal protocol on static-PE data with full TESSERACT constraints (S5 I06; S6 BODMAS mandate).

Cells: BODMAS temporal (train 2019-08..10, monthly tests 2019-11..2020-09) and its random contrast; EMBER-2018
temporal (train Jan-Jun, monthly tests Jul-Dec) and its random contrast; EMBER-trained models transferred to BODMAS
months (feature-aligned cross-dataset test, 2019-08..2020-09); the official EMBER-2018 LightGBM model on the same
months (threshold fixed on the official test month 2018-11). Each model is fitted once per seed and scored on every
window. Also written: decay attribution (FNR on families seen vs unseen in training), feature drift per window
(PSI/JSD, Eqs. 69-73) and TreeSHAP importance per window (explanation drift, S8 H8).
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import models as M, drift as D, explain as X_
from harness.thresholds import fpr_threshold, operating_points
from harness.data_ember import block_index, official_model

E3_SEEDS_FULL = (0, 1, 2)       # time split is fixed; seeds vary model randomness and the random contrast


def windows_of(ds, months):
    ym = ds.info['ym']
    return [(m, np.flatnonzero(ym == m)) for m in months]


def train_val(ds, months):
    sp = next(S.temporal_both_dated(ds.y, ds.info['ym'], ds.info['order'], months, months[-1:], C.VAL_FRAC))
    return sp.train, sp.val


def attribution(s, y, fam, thr, seen):
    """Decay attribution at the validation threshold: FNR overall, on seen and on unseen families."""
    mal = y == 1
    known = mal & (fam != 'unknown')
    is_seen = np.isin(fam, list(seen))
    fn = s < thr
    out = {'fnr_all': float(fn[mal].mean()) if mal.any() else np.nan,
           'share_unseen': float((known & ~is_seen).sum() / max(known.sum(), 1))}
    for nm, m in (('seen', known & is_seen), ('unseen', known & ~is_seen)):
        out[f'fnr_{nm}'] = float(fn[m].mean()) if m.any() else np.nan
        out[f'n_{nm}'] = int(m.sum())
    return out


def eval_model_on(model, name, ds_eval, wins, s_val, y_val, has_proba, cell, seed, store, seen, res_path,
                  explain_n=0, shap_rows=None):
    thr1 = fpr_threshold(s_val[y_val == 0], C.PRIMARY_FPR_BUDGET)
    for k, (label, idx) in enumerate(wins):
        if len(idx) == 0 or len(np.unique(ds_eval.y[idx])) < 2:
            continue
        Xw = np.asarray(ds_eval.X[idx]); t0 = time.perf_counter(); s = M.score(model, Xw)
        us = 1e6 * (time.perf_counter() - t0) / len(idx)
        rows, _ = E.evaluate(ds_eval.y[idx], s, s_val, y_val, has_proba)
        att = attribution(s, ds_eval.y[idx], ds_eval.family[idx], thr1, seen) if seen is not None else {}
        base = {**cell, 'model': name, 'seed': seed, 'window': k, 'month': label, 'us_per_sample': us,
                'n_test': len(idx), 'n_test_pos': int(ds_eval.y[idx].sum()),
                'n_test_neg': int((ds_eval.y[idx] == 0).sum()), **att}
        res_path.extend({**base, **r} for r in rows)                          # caller writes once per fit
        store.add(tuple([cell['dataset'], cell['rung'], seed, label, name]), ds_eval.row_id[idx], ds_eval.y[idx], s,
                  {'fpr|0.01': thr1})
        if explain_n and shap_rows is not None and name == 'lgbm' and seed == 0:
            imp = X_.tree_shap(model, Xw, max_rows=explain_n, seed=k)
            shap_rows.append({'cell': f"{cell['dataset']}|{cell['rung']}", 'month': label, 'imp': imp})


def month_num(ym):
    """Month index 12 * year + month for 'YYYY-MM' labels; NaN for ineligible rows ('out')."""
    ym = np.asarray(ym, dtype=object); out_ = np.full(len(ym), np.nan); ok = ym != 'out'
    out_[ok] = [12 * int(s[:4]) + int(s[5:7]) for s in ym[ok]]
    return out_


def gate_windows(ds, tr, va, wins, label, out, rc=None):
    """Gates A0-A4 for every temporal window (TESSERACT C1 = A3, C2 = A4) and A0-A2 for the random contrast;
    raises LeakageError on a violation and writes gates_<label>.csv."""
    tn = month_num(ds.info['ym']); rows = []
    for k, (m, idx) in enumerate(wins):
        sp = S.Split('temporal', tr, va, idx, {'window': k, 'month': m})
        lo = 12 * int(m[:4]) + int(m[5:7])
        rows.append({'dataset': label, 'rung': 'temporal', 'month': m,
                     **S.check_split(sp, ds.y, groups=ds.groups, hashes=ds.hashes, t=tn, require_group_disjoint=True,
                                     require_hash_disjoint=True, require_temporal=True, window=(lo, lo + 1))})
    if rc is not None:
        rows.append({'dataset': label, 'rung': 'random_contrast', 'month': 'random',
                     **S.check_split(rc, ds.y, groups=ds.groups, hashes=ds.hashes, require_group_disjoint=True,
                                     require_hash_disjoint=True)})
    pd.DataFrame(rows).to_csv(out / f'gates_{label}.csv', index=False)


def run_dataset(ds, train_months, test_months, models, seeds, out, store, T, label, transfer_to=None,
                shap_rows=None, mode='smoke'):
    res_path = out / 'results.csv'
    tr, va = train_val(ds, train_months)
    wins = windows_of(ds, test_months)
    pool0 = np.flatnonzero(np.isin(ds.info['ym'], list(train_months) + list(test_months)))
    gate_windows(ds, tr, va, wins, label, out, S.random_contrast(ds.y, pool0, len(tr), len(va), seeds[0]))
    seen = set(pd.unique(ds.family[tr][ds.y[tr] == 1])) - {'unknown'}
    prog = progress(out)
    for seed in seeds:
        set_seeds(seed)
        # random contrast: same pool and sizes, rows drawn at random from train + test months
        pool = np.flatnonzero(np.isin(ds.info['ym'], list(train_months) + list(test_months)))
        rc = S.random_contrast(ds.y, pool, len(tr), len(va), seed)
        for name in models:
            if f'{label}|{name}|{seed}' in prog:
                print(f'skip {label} {name} seed {seed} (done)', flush=True); continue
            purge(res_path, [label, f'{label}->bodmas'], name, seed)                 # drop partial rows
            buf = []
            for rung, (a_tr, a_va, a_wins) in (('temporal', (tr, va, wins)),
                                                ('random_contrast', (rc.train, rc.val, [('random', rc.test)]))):
                cell = {'dataset': label, 'rung': rung}
                with T.time('fit', **cell, model=name, seed=seed):
                    model = M.make_model(name, seed)
                    fit_s = M.fit(model, np.asarray(ds.X[a_tr]), ds.y[a_tr])
                    s_val = M.score(model, np.asarray(ds.X[a_va]))
                has_proba = M.is_probability(name)
                with T.time('eval', **cell, model=name, seed=seed):
                    eval_model_on(model, name, ds, a_wins, s_val, ds.y[a_va], has_proba, {**cell, 'fit_s': fit_s},
                                  seed, store, seen if rung == 'temporal' else None, buf,
                                  explain_n=300 if mode == 'smoke' else 1000, shap_rows=shap_rows)
                if rung == 'temporal' and transfer_to is not None:            # feature-aligned transfer
                    tds, tmonths = transfer_to
                    with T.time('eval_transfer', model=name, seed=seed):
                        eval_model_on(model, name, tds, windows_of(tds, tmonths), s_val, ds.y[a_va], has_proba,
                                      {'dataset': f'{label}->{tds.name}', 'rung': 'transfer', 'fit_s': fit_s},
                                      seed, store, None, buf)
                if name == 'lgbm' and rung == 'temporal' and seed == seeds[0]:
                    np.save(out / f'{label}_lgbm_gain.npy', M.importances(model, ds.features))
            append_rows(buf, res_path); store.flush(out / 'scores', f'{label}_{name}_s{seed}')
            if shap_rows:                                                     # explanation drift (lgbm, seed 0)
                np.savez_compressed(out / f'shap_windows_{label}.npz',
                                    **{f"{r['cell']}|{r['month']}": r['imp'] for r in shap_rows})
            mark(out, f'{label}|{name}|{seed}')
            print(f'{label} {name} seed {seed} done', flush=True)
    return tr, va, wins


def progress(out):
    p = out / 'progress.json'
    return set(json.load(open(p))) if p.exists() else set()


def mark(out, key):
    done = progress(out) | {key}
    with open(out / 'progress.json', 'w') as fh:
        json.dump(sorted(done), fh)


def purge(res_path, datasets, model, seed):
    """Remove rows of an interrupted (dataset, model, seed) so a resumed run never duplicates them."""
    if not res_path.exists():
        return
    d = pd.read_csv(res_path, low_memory=False)
    bad = d['dataset'].isin(datasets) & (d['model'] == model) & (d['seed'] == seed)
    if bad.any():
        d[~bad].to_csv(res_path, index=False)


def drift_table(ds, tr, wins, label, out, weights=None):
    Xr = np.asarray(ds.X[tr]); edges = D.fit_edges(Xr); rows = []
    blocks = {b: list(range(a, z)) for b, (a, z) in block_index().items()}
    for m, idx in wins:
        for cls, name in ((None, 'all'), (1, 'malware'), (0, 'benign')):
            sel = idx if cls is None else idx[ds.y[idx] == cls]
            ref = Xr if cls is None else Xr[ds.y[tr] == cls]
            e = edges if cls is None else D.fit_edges(ref)
            df = D.window_drift(ref, np.asarray(ds.X[sel]), e)
            rows.append({'dataset': label, 'month': m, 'class': name, **D.aggregate(df, blocks, weights)})
    return rows


def main():
    a = parse('E3 temporal static-PE', [(('--fresh',), {'action': 'store_true'})])
    if a.mode == 'full' and a.seeds == C.SEEDS_FULL:
        a.seeds = E3_SEEDS_FULL
    out = outdir(a.mode, 'e3'); T = TimingLog(); store = ScoreStore(); shap_rows = []
    if a.fresh:
        import shutil
        for f in ('results.csv', 'progress.json', 'drift.csv'):
            if (out / f).exists():
                (out / f).unlink()
        shutil.rmtree(out / 'scores', ignore_errors=True)
    models = tuple(a.models) if a.models else C.HIGH_DIM_MODELS
    with T.time('load'):
        bod, emb = bodmas_set(), ember_set()
    trB, vaB, winsB = run_dataset(bod, C.BODMAS_TRAIN_MONTHS, C.BODMAS_TEST_MONTHS, models, a.seeds, out, store, T,
                                  'bodmas', shap_rows=shap_rows, mode=a.mode)
    shap_rows = []
    allB = tuple(C.BODMAS_TRAIN_MONTHS) + tuple(C.BODMAS_TEST_MONTHS)
    trE, vaE, winsE = run_dataset(emb, C.EMBER_TRAIN_MONTHS, C.EMBER_TEST_MONTHS, models, a.seeds, out, store, T,
                                  'ember', transfer_to=(bod, allB), shap_rows=shap_rows, mode=a.mode)
    if 'official' not in progress(out):                                       # Endgame EMBER-2018 benchmark model
        with T.time('official_model'):
            off = official_model(); buf = []
            nov = np.flatnonzero(emb.info['ym'] == '2018-11')                # threshold month = official test month 1
            s_val = off.predict(np.asarray(emb.X[nov]))
            cell = {'dataset': 'ember_official', 'rung': 'official', 'fit_s': np.nan}

            class _Off:                                                       # adapter: predict_proba interface
                def predict_proba(self, X):
                    p = off.predict(X); return np.column_stack([1 - p, p])
            adapter = _Off()
            eval_model_on(adapter, 'ember_lgbm_official', emb, windows_of(emb, ('2018-12',)), s_val, emb.y[nov],
                          True, cell, 0, store, None, buf)
            eval_model_on(adapter, 'ember_lgbm_official', bod, windows_of(bod, allB), s_val, emb.y[nov], True,
                          {**cell, 'dataset': 'ember_official->bodmas', 'rung': 'transfer'}, 0, store, None, buf)
            purge(out / 'results.csv', ['ember_official', 'ember_official->bodmas'], 'ember_lgbm_official', 0)
            append_rows(buf, out / 'results.csv'); store.flush(out / 'scores', 'ember_official')
            mark(out, 'official')
    if not (out / 'drift.csv').exists():
        with T.time('drift'):
            rows = []
            for ds, tr, wins, label in ((bod, trB, winsB, 'bodmas'), (emb, trE, winsE, 'ember')):
                gp = out / f'{label}_lgbm_gain.npy'
                rows += drift_table(ds, tr, wins, label, out, np.load(gp) if gp.exists() else None)
            pd.DataFrame(rows).to_csv(out / 'drift.csv', index=False)
    append_rows(T.rows, out / 'timing.csv')                                  # schema-tolerant, survives resumes
    save_json({'bodmas': {k: v for k, v in bod.info.items() if k not in ('ym', 'order')},
               'ember': {k: v for k, v in emb.info.items() if k not in ('ym', 'order')},
               'n_train_bodmas': len(trB), 'n_val_bodmas': len(vaB), 'n_train_ember': len(trE),
               'n_val_ember': len(vaE)}, out / 'sets.json')
    save_json(environment(), out / 'env.json')


if __name__ == '__main__':
    main()

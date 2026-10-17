"""E4 - C2 protocol ladder on network data (NSL-KDD, UNSW-NB15): random K-fold on the pooled data (literature),
official train->test partition (distribution shift), leave-one-attack-family-out; dedup {none, exact} on the random
rung. Categorical one-hot encoders are fitted on training rows only. NIDS rows carry no sample identity, so a
grouped rung does not apply. Training rows are capped (fixed seed) for CPU budget; the cap is reported.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import data_nids as dn, models as M

MAX_TRAIN = 60000
SMOKE_MODELS = ('stump', 'lr', 'gnb', 'rf', 'lgbm')


def raw_hash(F, cats):
    G = F.copy()
    for c in cats:
        G[c] = pd.factorize(G[c])[0]
    return row_hashes(G.to_numpy(np.float64), 9)


def splits_for(meta, y, rung, seed, folds):
    part = meta.partition.to_numpy()
    if rung == 'official':
        tr = np.flatnonzero(part == 'train'); te = np.flatnonzero(part == 'test')
        tr, va = S._carve_val(tr, y, None, C.VAL_FRAC, seed)
        yield S.Split('official', tr, va, te, {'fold': 0, 'seed': seed})
    elif rung == 'random':
        for sp in S.random_kfold(y, C.N_FOLDS, seed, C.VAL_FRAC):
            if sp.info['fold'] in folds:
                yield sp
    elif rung == 'lofo':
        fam = meta.family.to_numpy(); groups = np.arange(len(y)).astype(str)
        yield from S.leave_one_family_out(y, fam, groups, seed, C.N_FOLDS, C.VAL_FRAC)


def main():
    a = parse('E4 NIDS ladder')
    out = outdir(a.mode, 'e4'); T = TimingLog(); store = ScoreStore()
    res_path, gate_path = out / 'results.csv', out / 'gates.csv'
    for p in (res_path, gate_path):
        if p.exists():
            p.unlink()
    models = tuple(a.models) if a.models else (SMOKE_MODELS if a.mode == 'smoke' else C.CORE_MODELS)
    folds = (0,) if a.mode == 'smoke' else tuple(range(C.N_FOLDS))
    cells = [('random', 'none'), ('random', 'exact'), ('official', 'exact'), ('lofo', 'exact')]
    for nm, loader, cats in (('nsl_kdd', dn.load_nsl, dn.NSL_CAT), ('unsw_nb15', dn.load_unsw, dn.UNSW_CAT)):
        F, meta = loader(); y = meta.y.to_numpy(); hs = raw_hash(F, cats)
        for rung, dedup in cells:
            cell = {'dataset': nm, 'rung': rung, 'dedup': dedup}
            for seed in a.seeds:
                set_seeds(seed); rng = np.random.default_rng(seed)
                for sp in splits_for(meta, y, rung, seed, folds):
                    if dedup == 'exact':
                        sp = S.drop_hash_duplicates(sp, hs)
                    g = S.check_split(sp, y, hashes=hs if dedup == 'exact' else None,
                                      require_hash_disjoint=dedup == 'exact')
                    n_tr_full = len(sp.train)
                    if len(sp.train) > MAX_TRAIN:
                        sp.train = np.sort(rng.choice(sp.train, MAX_TRAIN, replace=False))
                    fit_idx = np.concatenate([sp.train, sp.val])
                    Xfit, Xte, names = dn.encode_split(F.iloc[fit_idx], F.iloc[sp.test], cats)
                    Xall = np.vstack([Xfit, Xte]); yall = np.concatenate([y[fit_idx], y[sp.test]])
                    loc = S.Split(sp.rung, np.arange(len(sp.train)), np.arange(len(sp.train), len(fit_idx)),
                                  np.arange(len(fit_idx), len(Xall)), sp.info)
                    ds = AnalysisSet(name=nm, X=Xall, y=yall, groups=np.arange(len(yall)).astype(str),
                                     family=np.concatenate([meta.family.to_numpy()[fit_idx],
                                                            meta.family.to_numpy()[sp.test]]),
                                     features=names, hashes=None,
                                     row_id=np.concatenate([fit_idx, sp.test]))
                    append_rows([{**cell, 'seed': seed, **{k: v for k, v in sp.info.items()
                                                          if not isinstance(v, (list, tuple, dict))}, **g,
                                  'n_train_before_cap': n_tr_full, 'n_train': len(sp.train), 'n_test': len(sp.test)}],
                                gate_path)
                    for name in models:
                        with T.time('fit_eval', **cell, model=name, seed=seed):
                            rows, _ = run_fit_eval(ds, loc, name, seed, cell, store)
                        append_rows(rows, res_path)
                print(f'{cell} seed {seed} done', flush=True)
    store.save(out / 'scores.npz'); T.frame().to_csv(out / 'timing.csv', index=False)
    save_json(environment(), out / 'env.json')


if __name__ == '__main__':
    main()

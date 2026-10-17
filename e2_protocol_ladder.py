"""E2 - C2 protocol-inflation ladder on CIC-MalMem-2022 (factorial; S6 P1 redesign; S3 constraints 2-5).

Factors: benign pool {all = literature, genuine = synthetic rows removed} x split {random, grouped} x dedup
{none, exact}; extra rungs: leave-one-family-out (lofo), leave-one-category-out (loco), single-family training
(Madamidola et al.), temporal (malware-side, dated binaries), near-duplicate sensitivity. Prevalence is a test-time
re-weighting (no refit). Grouped rungs on the 'all' pool purge SMOTE children whose second parent falls in another
partition (gate A5). svcscan-removed cells reuse the identity (groups, hashes) of all 55 features so they pair with
their full-feature cells; collisions in the reduced space are reported as test_reduced_space_dup_share.
Output: runs/<mode>/e2/results.csv, gates.csv, scores/*.npz, timing.csv.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403

LOFO_MODELS = ('stump', 'lr', 'gnb', 'rf', 'lgbm')
CELLS = [  # (pool, rung, dedup, model set)
    ('all', 'random', 'none', 'core+extra'),        # literature reproduction (CIC release, row-level CV)
    ('all', 'random', 'exact', 'core'),
    ('all', 'grouped', 'none', 'core'),
    ('all', 'grouped', 'exact', 'core'),
    ('genuine', 'random', 'none', 'core'),
    ('genuine', 'random', 'exact', 'core'),
    ('genuine', 'grouped', 'none', 'core'),
    ('genuine', 'grouped', 'exact', 'core+extra'),  # strict cell
    ('all', 'lofo', 'exact', 'lofo'),
    ('genuine', 'lofo', 'exact', 'lofo'),
    ('genuine', 'loco', 'exact', 'lofo'),
    ('all', 'single_family', 'none', 'lofo'),       # Madamidola et al. protocol as published (all benign)
    ('genuine', 'single_family', 'exact', 'lofo'),
    ('all', 'temporal', 'exact', 'core+extra'),
    ('genuine', 'temporal', 'exact', 'core+extra'),
    ('genuine', 'grouped', 'near', 'core'),         # near-duplicate removal sensitivity (Eq. 9, theta = 0.05)
    ('genuine', 'temporal', 'exact', 'core', 'svcscan'),   # S2 H4: environment block removed
    ('all', 'temporal', 'exact', 'core', 'svcscan'),
    ('genuine', 'grouped', 'exact', 'core', 'svcscan'),    # C1: does the ranking survive without the shortcut block?
]


def model_set(key, override):
    if override:
        return tuple(override)
    return {'core': C.CORE_MODELS, 'core+extra': C.CORE_MODELS + C.EXTRA_MODELS, 'lofo': LOFO_MODELS}[key]


def main():
    a = parse('E2 protocol ladder', [(('--cells',), {'type': int, 'nargs': '*', 'default': None}),
                                      (('--fresh',), {'action': 'store_true'})])
    out = outdir(a.mode, 'e2'); T = TimingLog(); store = ScoreStore()
    res_path, gate_path = out / 'results.csv', out / 'gates.csv'
    if a.fresh:                                                               # explicit clean start
        import shutil
        for p in (res_path, gate_path):
            if p.exists():
                p.unlink()
        shutil.rmtree(out / 'scores', ignore_errors=True)
    done = done_keys(res_path, ['pool', 'rung', 'dedup', 'drop_block', 'seed'])  # resume after a restart
    cache = {}
    cells = [CELLS[i] for i in a.cells] if a.cells else CELLS
    for spec in cells:
        pool, rung, dedup, mkey = spec[:4]
        drop = spec[4] if len(spec) > 4 else None                            # plugin block removed (S2 H4)
        key = (pool, dedup != 'none')
        if key not in cache:
            with T.time('build_set', pool=pool, dedup=dedup):
                cache[key] = malmem_set(pool, dedup=dedup != 'none')
        ds = cache[key]
        red_hashes = None
        if drop:                                       # model sees 48 features; identity stays on all 55 (paired)
            keepc = [i for i, f in enumerate(ds.features) if not f.startswith(drop + '.')]
            ds = AnalysisSet(**{**ds.__dict__, 'X': ds.X[:, keepc], 'features': [ds.features[i] for i in keepc]})
            red_hashes = row_hashes(ds.X)              # diagnostic: collisions in the reduced feature space
        cell = {'dataset': 'malmem', 'pool': pool, 'rung': rung, 'dedup': dedup, 'drop_block': drop or 'none'}
        models = model_set(mkey, a.models)
        for seed in a.seeds:
            if (pool, rung, dedup, drop or 'none', str(seed)) in done:
                print(f'skip {cell} seed {seed} (done)', flush=True); continue
            set_seeds(seed); buf, gbuf = [], []
            for sp in malmem_cell_splits(ds, rung, seed):
                with T.time('dedup_gate', **cell, seed=seed):
                    sp = apply_dedup(ds, sp, dedup, measure_near=(seed == a.seeds[0]))
                    g = gate_split(ds, sp, rung, dedup)                       # A0-A5 (A4 window, A5 SMOTE parents)
                    if red_hashes is not None and len(sp.test):
                        fit_h = set(red_hashes[np.union1d(sp.train, sp.val)])
                        g['test_reduced_space_dup_share'] = float(np.mean([h in fit_h for h in red_hashes[sp.test]]))
                gbuf.append({**cell, 'seed': seed, **{k: v for k, v in sp.info.items()
                                                     if not isinstance(v, (list, tuple, dict))}, **g,
                             'n_train': len(sp.train), 'n_val': len(sp.val), 'n_test': len(sp.test)})
                for name in models:
                    with T.time('fit_eval', **cell, model=name, seed=seed):
                        rows, _ = run_fit_eval(ds, sp, name, seed, cell, store)
                    buf += rows
            store.flush(out / 'scores', f'{pool}_{rung}_{dedup}_{drop or "none"}_s{seed}')   # scores and gates
            append_rows(gbuf, gate_path); append_rows(buf, res_path)   # first, results last: "done" implies all
            print(f'done {cell} seed {seed}', flush=True)
    if a.fresh and (out / 'timing.csv').exists():
        (out / 'timing.csv').unlink()
    append_rows(T.rows, out / 'timing.csv'); save_json(environment(), out / 'env.json')
    save_json({f'{k[0]}|dedup={k[1]}': v.info for k, v in cache.items()}, out / 'sets.json')


if __name__ == '__main__':
    main()

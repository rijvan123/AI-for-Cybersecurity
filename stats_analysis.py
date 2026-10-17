"""stats_analysis.py - S10 statistics over E2/E3/E4 outputs (S8 Eqs. 94-97; Demsar 2006; DeLong 1988).

Writes runs/<mode>/stats/:
  ci_pooled.csv     95% group-bootstrap CIs on pooled out-of-fold predictions per cell x model x seed
                    (AUC-ROC, AUC-PR at 1%; precision at 1%, recall and FPR at each fold's validation FPR-1%
                    threshold), computed from per-row multiplicities (stats.group_counts_bootstrap)
  paired_rungs.csv  Wilcoxon signed-rank + paired dominance + Cliff's delta + Holm, rung contrasts per model
  friedman.csv      Friedman chi2, p and Nemenyi CD across models per cell (blocks = seed x fold / window)
  rank_agreement.csv Kendall tau between the AUC-ROC ranking and the MCC-at-1% ranking per cell (H-C2a)
  delong.csv        DeLong tests between the two best models on identical test rows (first seed)
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import stats as ST

CONTRASTS = [  # (name, cell A, cell B): A is the more optimistic protocol
    ('synthetic_rows', ('all', 'random', 'none'), ('genuine', 'random', 'none')),
    ('grouping_all', ('all', 'random', 'exact'), ('all', 'grouped', 'exact')),
    ('grouping_genuine', ('genuine', 'random', 'exact'), ('genuine', 'grouped', 'exact')),
    ('dedup_genuine', ('genuine', 'grouped', 'none'), ('genuine', 'grouped', 'exact')),
    ('literature_vs_strict', ('all', 'random', 'none'), ('genuine', 'grouped', 'exact')),
]
METRICS = ['auc_roc', 'auc_pr', 'precision', 'recall', 'mcc', 'fpr', 'fpr_at_95tpr']


def cell_query(df, cell):
    p, r, d = cell
    q = (df['pool'] == p) & (df['rung'] == r) & (df['dedup'] == d)
    if 'drop_block' in df:
        q &= df['drop_block'].fillna('none').eq('none')
    return df[q]


def main():
    a = parse('statistics')
    base = Path(C.RUNS) / a.mode; out = base / 'stats'; out.mkdir(exist_ok=True)
    r2 = pd.read_csv(base / 'e2' / 'results.csv', low_memory=False)
    r2['drop_block'] = r2['drop_block'].fillna('none') if 'drop_block' in r2 else 'none'
    sel = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01) & (r2.pi == 0.01)]
    # paired rung contrasts (blocks = seed x fold)
    rows = []
    for name, ca, cb in CONTRASTS:
        A, B = cell_query(sel, ca), cell_query(sel, cb)
        for m in sorted(set(A.model) & set(B.model)):
            x = A[A.model == m].set_index(['seed', 'fold']); y = B[B.model == m].set_index(['seed', 'fold'])
            j = x[METRICS].join(y[METRICS], lsuffix='_a', rsuffix='_b', how='inner')
            for met in METRICS:
                va, vb = j[f'{met}_a'].to_numpy(), j[f'{met}_b'].to_numpy()
                stat, p = ST.wilcoxon(va, vb) if len(va) >= 6 else (np.nan, np.nan)
                dom, med = ST.paired_delta_effect(va, vb)
                cd, mag = ST.cliffs_delta(va, vb)
                rows.append({'contrast': name, 'model': m, 'metric': met, 'n_blocks': len(va), 'mean_a': va.mean(),
                             'mean_b': vb.mean(), 'diff_pp': 100 * (va.mean() - vb.mean()),
                             'rel_diff_pct': 100 * (va.mean() - vb.mean()) / vb.mean() if vb.mean() else np.nan,
                             'median_diff': med, 'paired_dominance': dom, 'cliffs_delta': cd, 'magnitude': mag,
                             'wilcoxon_W': stat, 'p': p})
    pr = pd.DataFrame(rows)
    if len(pr):
        for (c, met), g in pr.groupby(['contrast', 'metric']):
            ok = g['p'].notna()
            pr.loc[g.index[ok], 'p_holm'] = ST.holm(g.loc[ok, 'p'].to_numpy())
    pr.to_csv(out / 'paired_rungs.csv', index=False)
    # Friedman / Nemenyi and rank agreement per cell
    fr, ra = [], []
    for cell, g in sel.groupby(['pool', 'rung', 'dedup', 'drop_block']):    # ablation cells stay separate
        blk = 'fold' if g['fold'].notna().any() else 'window'
        if g.duplicated(['seed', blk, 'model']).any():                         # one value per block x model
            raise ValueError(f'friedman: duplicate (seed, {blk}, model) rows in cell {cell}')
        M = g.pivot_table(index=['seed', blk], columns='model', values='auc_pr', aggfunc='first').dropna(axis=1)
        cd = {'pool': cell[0], 'rung': cell[1], 'dedup': cell[2], 'drop_block': cell[3]}
        if M.shape[0] >= 2 and M.shape[1] >= 3:
            f = ST.friedman_nemenyi(M.to_numpy())
            fr.append({**cd, 'chi2_F': f['chi2_F'], 'p': f['p'], 'CD': f['CD'], 'N_blocks': f['N_blocks'], 'k': f['k'],
                       **{f'rank_{m}': r for m, r in zip(M.columns, f['mean_ranks'])}})
        h = r2[(r2.pool == cell[0]) & (r2.rung == cell[1]) & (r2.dedup == cell[2]) & (r2.drop_block == cell[3])
               & (r2.rule == 'fpr') & (r2.alpha == 0.01)]
        auc = h[h.pi == 0.5].groupby('model')['auc_roc'].mean(); mcc = h[h.pi == 0.01].groupby('model')['mcc'].mean()
        common = auc.index.intersection(mcc.index)
        if len(common) >= 3:
            tau, p = ST.kendall(auc[common].rank(), mcc[common].rank())
            ra.append({**cd, 'kendall_tau': tau, 'p': p, 'n_models': len(common),
                       'best_by_auc': auc[common].idxmax(), 'best_by_mcc1': mcc[common].idxmax()})
    pd.DataFrame(fr).to_csv(out / 'friedman.csv', index=False); pd.DataFrame(ra).to_csv(out / 'rank_agreement.csv', index=False)
    # pooled out-of-fold group-bootstrap CIs and DeLong (needs stored scores + group ids)
    sc = load_scores(base / 'e2'); keys = sorted(sc)
    sets = {('all', False): malmem_set('all', False), ('all', True): malmem_set('all', True),
            ('genuine', False): malmem_set('genuine', False), ('genuine', True): malmem_set('genuine', True)}
    gid = {k: dict(zip(v.row_id, v.groups)) for k, v in sets.items()}
    pooled = {}
    for k in keys:
        f = parse_score_key(k)
        dedup, pool, rung, model, seed = f['dedup'], f['pool'], f['rung'], f['model'], f['seed']
        if rung not in ('random', 'grouped') or f['drop_block'] != 'none':
            continue
        yv, s, thr_k, rid = sc[k]
        t1 = thr_k.get('fpr|0.01|nan|0.01')
        d = pooled.setdefault((pool, rung, dedup, model, seed), {'y': [], 's': [], 'pred': [], 'rid': []})
        d['y'].append(yv); d['s'].append(s); d['pred'].append(s >= t1); d['rid'].append(rid)
    ci_rows = []
    for (pool, rung, dedup, model, seed), d in pooled.items():
        y = np.concatenate(d['y']); s = np.concatenate(d['s']); pred = np.concatenate(d['pred'])
        rid = np.concatenate(d['rid']); g = np.array([gid[(pool, dedup != 'none')].get(r, f'r{r}') for r in rid])
        rs = ST.RankStats(y, s); yp, yn, pp = (y == 1), (y == 0), pred.astype(float)

        def dec(c, which):                                                    # decision metrics from multiplicities
            P, N = (c * yp).sum(), (c * yn).sum(); tp, fp = (c * yp * pp).sum(), (c * yn * pp).sum()
            if which == 'recall':
                return tp / P if P else np.nan
            if which == 'fpr':
                return fp / N if N else np.nan
            a_, b_ = 0.01 * tp / P if P else 0.0, 0.99 * fp / N if N else 0.0   # precision at pi = 1% (Eq. 58)
            return a_ / (a_ + b_) if a_ + b_ > 0 else 0.0
        fns = {'auc_roc': rs.auc, 'auc_pr_1pct': lambda c: rs.ap(c, 0.01),
               'precision_1pct': lambda c: dec(c, 'precision'), 'recall': lambda c: dec(c, 'recall'),
               'fpr': lambda c: dec(c, 'fpr')}
        ci = ST.group_counts_bootstrap(y, g, fns, B=a.B, seed=int(seed))
        rec = {'pool': pool, 'rung': rung, 'dedup': dedup, 'model': model, 'seed': seed, 'n': len(y),
               'n_groups': int(len(np.unique(g))), 'B': a.B}
        ones = np.ones(len(y))
        for nm, fn in fns.items():
            rec[nm] = fn(ones); rec[f'{nm}_lo'], rec[f'{nm}_hi'] = ci[nm][0], ci[nm][1]
        ci_rows.append(rec)
    pd.DataFrame(ci_rows).to_csv(out / 'ci_pooled.csv', index=False)
    # DeLong: two best models (by AUC-PR) on identical rows, strict and literature cells, first fold
    dl = []
    for cell in (('all', 'random', 'none'), ('genuine', 'grouped', 'exact')):
        g = cell_query(sel, cell)
        top = g.groupby('model')['auc_pr'].mean().sort_values(ascending=False).index[:3].tolist()
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                ka = [k for k in keys if (lambda f: (f['pool'], f['rung'], f['dedup']) == cell and f['model'] == top[i]
                                          and f['fold'] == '0' and f['drop_block'] == 'none')(parse_score_key(k))]
                kb = [k[: -len(top[i])] + top[j] for k in ka]
                if ka and kb[0] in sc:
                    a1, a2, zz, p = ST.delong(sc[ka[0]][0], sc[ka[0]][1], sc[kb[0]][1])
                    dl.append({'cell': '|'.join(cell), 'model_a': top[i], 'model_b': top[j], 'auc_a': a1, 'auc_b': a2,
                               'z': zz, 'p': p})
    pd.DataFrame(dl).to_csv(out / 'delong.csv', index=False)
    print(pr[pr.metric.isin(['auc_pr', 'fpr_at_95tpr'])].round(4).to_string()[:3000])


if __name__ == '__main__':
    main()

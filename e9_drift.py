"""E9 - drift diagnostics and the drift-decay link (S8 Eqs. 69-74, prediction H4; S5 I07 window alarms).

MalMem (malware-side): PSI/JSD of each test window's malware against the training-era malware, per feature and
per plugin block (benign is undated and therefore held fixed). Static PE: E3's drift.csv. H4: Spearman rho between
window drift (mean JSD) and window MCC (pi = 1%, FPR-1% rule) per dataset x model, pooled across windows x seeds.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import drift as D
from harness.stats import spearman


def main():
    a = parse('E9 drift')
    out = outdir(a.mode, 'e9'); base = out.parent; rows = []
    ds = malmem_set('genuine', dedup=True); blocks = dm.plugin_blocks(ds.features)
    bidx = {b: [ds.features.index(f) for f in fs] for b, fs in blocks.items()}
    sps = list(malmem_cell_splits(ds, 'temporal', a.seeds[0]))
    tr_m = sps[0].train[ds.y[sps[0].train] == 1]
    Xr = ds.X[tr_m]; edges = D.fit_edges(Xr)
    feat_rows = []
    for sp in sps:
        te_m = sp.test[ds.y[sp.test] == 1]
        df = D.window_drift(Xr, ds.X[te_m], edges, ds.features)
        df['window'] = sp.info['window']; feat_rows.append(df)
        rows.append({'dataset': 'malmem_genuine', 'window': sp.info['window'], 'months': str(sp.info['window_months']),
                     'class': 'malware', **D.aggregate(df, bidx)})
    pd.concat(feat_rows).to_csv(out / 'malmem_feature_drift.csv', index=False)
    dr = pd.DataFrame(rows); dr.to_csv(out / 'malmem_drift.csv', index=False)
    # H4: drift vs decay
    h4 = []
    r2 = pd.read_csv(base / 'e2' / 'results.csv', low_memory=False)
    nodrop = r2['drop_block'].fillna('none').eq('none') if 'drop_block' in r2 else True   # full feature set only
    t2 = r2[(r2.rung == 'temporal') & (r2.pool == 'genuine') & (r2.dedup == 'exact') & nodrop & (r2.rule == 'fpr')
            & (r2.alpha == 0.01) & (r2.pi == 0.01)]
    for model, g in t2.groupby('model'):
        m = g.merge(dr[['window', 'jsd_mean']], on='window')
        rho, p = spearman(m['jsd_mean'], m['mcc']) if m['jsd_mean'].nunique() > 1 else (np.nan, np.nan)
        rho_r, p_r = spearman(m['jsd_mean'], m['recall']) if m['jsd_mean'].nunique() > 1 else (np.nan, np.nan)
        h4.append({'dataset': 'malmem_genuine', 'model': model, 'rho_jsd_mcc': rho, 'p': p,
                   'rho_jsd_recall': rho_r, 'p_recall': p_r, 'n': len(m)})
    p3 = base / 'e3'
    if (p3 / 'drift.csv').exists():
        d3 = pd.read_csv(p3 / 'drift.csv'); r3 = pd.read_csv(p3 / 'results.csv')
        r3 = r3[(r3.rung == 'temporal') & (r3.rule == 'fpr') & (r3.alpha == 0.01) & (r3.pi == 0.01)]
        for (dsn, model), g in r3.groupby(['dataset', 'model']):
            dd = d3[(d3.dataset == dsn) & (d3['class'] == 'all')][['month', 'jsd_mean']]
            m = g.merge(dd, on='month')
            if len(m) > 2:
                rho, p = spearman(m['jsd_mean'], m['mcc'])
                rho_r, p_r = spearman(m['jsd_mean'], m['recall'])
                h4.append({'dataset': dsn, 'model': model, 'rho_jsd_mcc': rho, 'p': p, 'rho_jsd_recall': rho_r,
                           'p_recall': p_r, 'n': len(m)})
    pd.DataFrame(h4).to_csv(out / 'h4_drift_decay.csv', index=False)
    print(pd.DataFrame(h4).round(3).to_string())


if __name__ == '__main__':
    main()

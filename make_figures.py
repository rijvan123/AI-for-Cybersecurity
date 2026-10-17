"""make_figures.py - render every R5 figure (and the C1-C5 diagnostics) from runs/<mode>/ outputs.

python3 -I experiments/make_figures.py --mode smoke|full   ->  runs/<mode>/figures/*.png|pdf
Figures whose inputs are missing are skipped with a message, so partial runs still render.
"""
import sys; sys.path[:0] = ['/home/claude/s9', '/home/claude/s9/experiments']
from common import *                                                           # noqa: F401,F403
from harness import plots as PL
import matplotlib.pyplot as plt

DATASET_LABEL = {'malmem_lit': 'MalMem\nliterature', 'malmem_strict': 'MalMem\nstrict', 'malmem_temp': 'MalMem\ntemporal',
                 'bodmas': 'BODMAS\ntemporal', 'ember': 'EMBER-18\ntemporal', 'nsl_kdd': 'NSL-KDD\nofficial',
                 'unsw_nb15': 'UNSW-NB15\nofficial'}


def load(base, exp, name='results.csv', ablations=False):
    """Read one output table. E2 results are restricted to the full feature set unless ablations=True, so the
    svcscan-removed cells (drop_block = svcscan) never mix into the base cells of a figure."""
    p = base / exp / name
    if not p.exists():
        return None
    d = pd.read_csv(p, low_memory=False)
    if exp == 'e2' and name == 'results.csv' and 'drop_block' in d:
        d['drop_block'] = d['drop_block'].fillna('none')
        if not ablations:
            d = d[d.drop_block == 'none']
    return d


def scores(base, exp):
    return {k: (v[0], v[1]) for k, v in load_scores(base / exp).items()}


def headline_frame(base):
    """One row per (dataset, model): mean metrics at pi=10% / 1% with the FPR-1% rule."""
    rows = []
    r2, r3, r4 = load(base, 'e2'), load(base, 'e3'), load(base, 'e4')
    sel = lambda d: d[(d.rule == 'fpr') & (d.alpha == 0.01)]
    if r2 is not None:
        for tag, q in (('malmem_lit', "pool=='all' & rung=='random' & dedup=='none'"),
                       ('malmem_strict', "pool=='genuine' & rung=='grouped' & dedup=='exact'"),
                       ('malmem_temp', "pool=='genuine' & rung=='temporal' & dedup=='exact'")):
            d = sel(r2).query(q)
            for (m, pi), g in d.groupby(['model', 'pi']):
                rows.append({'dataset': tag, 'model': m, 'pi': pi, **g[['auc_roc', 'auc_pr', 'f1', 'mcc', 'precision',
                                                                       'recall', 'fpr']].mean().to_dict()})
    if r3 is not None:
        d = sel(r3)[sel(r3).rung == 'temporal']
        for (ds_, m, pi), g in d.groupby(['dataset', 'model', 'pi']):
            rows.append({'dataset': ds_, 'model': m, 'pi': pi, **g[['auc_roc', 'auc_pr', 'f1', 'mcc', 'precision',
                                                                    'recall', 'fpr']].mean().to_dict()})
    if r4 is not None:
        d = sel(r4)[sel(r4).rung == 'official']
        for (ds_, m, pi), g in d.groupby(['dataset', 'model', 'pi']):
            rows.append({'dataset': ds_, 'model': m, 'pi': pi, **g[['auc_roc', 'auc_pr', 'f1', 'mcc', 'precision',
                                                                    'recall', 'fpr']].mean().to_dict()})
    return pd.DataFrame(rows)


def fig_bars(base, figs, H):
    if H.empty:
        return print('skip bars')
    datasets = [d for d in DATASET_LABEL if d in set(H.dataset)]
    models = PL.ordered(set(H.model) & set(['stump'] + PL.MODEL_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(PL.FULL_W, 5.2), sharex=True)
    h10 = H[H.pi == 0.1].assign(dataset=lambda d: d.dataset.map(DATASET_LABEL))
    PL.bar_metric_by_dataset(h10, 'auc_roc', axes[0], [DATASET_LABEL[d] for d in datasets], models,
                             ylabel='AUC-ROC', ylim=(0.5, 1.0))
    PL.bar_metric_by_dataset(h10, 'f1', axes[1], [DATASET_LABEL[d] for d in datasets], models,
                             ylabel='F1 at 10% prevalence', ylim=(0, 1.0))
    axes[0].set_title('(a) AUC-ROC (prevalence-invariant)', loc='left')
    axes[1].set_title('(b) F1 at 10% malware prevalence, validation threshold FPR <= 1%', loc='left')
    axes[0].legend(ncol=5, loc='lower left', bbox_to_anchor=(0, 1.12))
    PL.save(fig, figs / 'fig_r5_01_bars_auc_f1')


def pick_curves(sc, want, models):
    """First stored test fold/window per model whose key fields match `want` (dict of field -> value)."""
    out = {}
    for k, v in sorted(sc.items()):
        f = parse_score_key(k)
        if all(f.get(a) == b for a, b in want.items()) and f['drop_block'] == 'none' and f['model'] in models \
                and f['model'] not in out:
            out[f['model']] = v
    return out


def fig_roc_pr(base, figs):
    sc2, sc3, sc4 = scores(base, 'e2'), scores(base, 'e3'), scores(base, 'e4')
    models = ['stump'] + PL.MODEL_ORDER
    panels = [('MalMem literature (all benign, random)',
               pick_curves(sc2, {'pool': 'all', 'rung': 'random', 'dedup': 'none', 'seed': '0', 'fold': '0'}, models)),
              ('MalMem strict (genuine, grouped)',
               pick_curves(sc2, {'pool': 'genuine', 'rung': 'grouped', 'dedup': 'exact', 'seed': '0', 'fold': '0'}, models)),
              ('BODMAS temporal, 2020-09', pick_curves(sc3, {'dataset': 'bodmas', 'rung': 'temporal', 'month': '2020-09'}, models)),
              ('EMBER-2018 temporal, 2018-12', pick_curves(sc3, {'dataset': 'ember', 'rung': 'temporal', 'month': '2018-12'}, models)),
              ('NSL-KDD official partition', pick_curves(sc4, {'dataset': 'nsl_kdd', 'rung': 'official'}, models)),
              ('UNSW-NB15 official partition', pick_curves(sc4, {'dataset': 'unsw_nb15', 'rung': 'official'}, models))]
    panels = [p for p in panels if p[1]]
    for kind in ('roc', 'pr'):
        n = len(panels)
        if not n:
            return print('skip roc/pr')
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(PL.FULL_W, 2.6 * ((n + 1) // 2)))
        for ax, (title, curves) in zip(axes.ravel(), panels):
            if kind == 'roc':
                PL.roc_overlay(ax, curves, title)
            else:
                PL.pr_overlay(ax, curves, 0.01, title)
        for ax in axes.ravel()[n:]:
            ax.axis('off')
        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.03))
        fig.tight_layout()
        PL.save(fig, figs / f'fig_r5_0{2 if kind == "roc" else 3}_{kind}_overlays')


def fig_confusion(base, figs, H):
    d = H[(H.dataset == 'malmem_strict') & (H.pi == 0.01)]
    r2 = load(base, 'e2')
    if d.empty or r2 is None:
        return print('skip confusion')
    top = d.sort_values('mcc', ascending=False).model.head(3).tolist()
    sel = r2[(r2.pool == 'genuine') & (r2.rung == 'grouped') & (r2.dedup == 'exact') & (r2.rule == 'fpr') &
             (r2.alpha == 0.01) & (r2.pi == 0.01)]
    fig, axes = plt.subplots(1, 3, figsize=(PL.FULL_W, 2.4))
    for ax, m in zip(axes, top):
        g = sel[sel.model == m]
        n = (g.tp + g.fp + g.tn + g.fn).mean()
        k = 1e4 / n
        PL.confusion_heatmap(ax, g.tp.mean() * k, g.fp.mean() * k, g.tn.mean() * k, g.fn.mean() * k,
                             f'{PL.MODEL_LABEL.get(m, m)}: MCC {g.mcc.mean():.3f}')
    fig.suptitle('Strict MalMem cell, 1% prevalence, validation FPR <= 1% threshold', x=0.01, ha='left', fontsize=8.5)
    fig.tight_layout(); PL.save(fig, figs / 'fig_r5_04_confusion_top3')


def fig_importance(base, figs):
    d = load(base, 'e10', 'importances.csv')
    if d is None:
        return print('skip importance')
    fig, axes = plt.subplots(1, 3, figsize=(PL.FULL_W, 3.4))
    for ax, (m, kind, title, c) in zip(axes, (('rf', 'impurity_or_gain', 'Random forest (impurity)', PL.SLOTS[4]),
                                               ('xgb', 'impurity_or_gain', 'XGBoost (gain)', PL.SLOTS[6]),
                                               ('rf', 'permutation_ap', 'RF permutation (AP drop)', PL.SLOTS[0]))):
        g = d[(d.model == m) & (d.kind == kind)].groupby('feature')['importance'].mean()
        if len(g):
            PL.hbar_importance(ax, g.index.to_numpy(), g.to_numpy(), title, color=c)
    fig.tight_layout(); PL.save(fig, figs / 'fig_r5_05_feature_importance')


def fig_learning(base, figs):
    d = load(base, 'e10', 'learning_curves.csv')
    if d is None:
        return print('skip learning curves')
    sets = [s for s in ('malmem_strict', 'bodmas_temporal') if s in set(d.dataset)]
    fig, axes = plt.subplots(len(sets), 2, figsize=(PL.FULL_W, 2.5 * len(sets)), squeeze=False)
    for i, s in enumerate(sets):
        g = d[d.dataset == s]
        for j, met in enumerate(('auc_pr', 'mcc')):
            PL.line_by_model(axes[i, j], g, 'frac', met, set(g.model), 'Fraction of training data (log)',
                             {'auc_pr': 'AUC-PR at 1%', 'mcc': 'MCC at 1%'}[met],
                             {'malmem_strict': 'MalMem strict cell (genuine, grouped)',
                              'bodmas_temporal': 'BODMAS temporal (mean over windows)'}.get(s, s), band=True,
                             logx=True)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(); PL.save(fig, figs / 'fig_r5_06_learning_curves')


def fig_violins(base, figs):
    r2 = load(base, 'e2')
    if r2 is None:
        return print('skip violins')
    sel = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01) & (r2.pi == 0.01)]
    cells = (("pool=='all' & rung=='random' & dedup=='none'", 'Literature protocol (all benign, random split)'),
             ("pool=='genuine' & rung=='grouped' & dedup=='exact'", 'Strict protocol (genuine benign, grouped split)'))
    models = PL.ordered(set(sel.model) & set(['stump'] + PL.MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(PL.FULL_W, 3.0), sharey=True)
    for ax, (q, title) in zip(axes, cells):
        PL.violin_by_model(ax, sel.query(q), 'auc_pr', models, title, 'AUC-PR at 1% prevalence')
    fig.tight_layout(); PL.save(fig, figs / 'fig_r5_07_cv_violins')


def fig_temporal(base, figs):
    r3, r2 = load(base, 'e3'), load(base, 'e2')
    panels = []
    if r3 is not None:
        sel = r3[(r3.rule == 'fpr') & (r3.alpha == 0.01)]
        for ds_, title in (('bodmas', 'BODMAS (train 2019-08..10)'), ('ember', 'EMBER-2018 (train Jan-Jun)'),
                           ('ember->bodmas', 'EMBER-trained, tested on BODMAS')):
            d = sel[(sel.dataset == ds_) & sel.rung.isin(['temporal', 'transfer'])]   # no random-contrast rows
            if len(d):
                panels.append((title, d, 'month'))
    if r2 is not None:
        d = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01) & (r2.rung == 'temporal') & (r2.pool == 'genuine')].copy()
        if len(d):                                                            # month index t = 12 (year - 2006) + month
            ym = lambda t_: f'{(int(t_) - 1) // 12 + 2006 - 2000:02d}-{(int(t_) - 1) % 12 + 1:02d}'
            d['window_label'] = [f'{ym(lo)}..{ym(hi - 1)}' for lo, hi in zip(d.window_lo, d.window_hi)]
            panels.append(('MalMem, malware-side half-year windows', d, 'window_label'))
    if not panels:
        return print('skip temporal')
    fig, axes = plt.subplots(len(panels), 2, figsize=(PL.FULL_W, 2.3 * len(panels)), squeeze=False)
    for i, (title, d, xcol) in enumerate(panels):
        models = PL.ordered(set(d.model) & set(['stump'] + PL.MODEL_ORDER))
        for j, (met, pi, lab) in enumerate((('f1', 0.1, 'F1 at 10%'), ('auc_pr', 0.01, 'AUC-PR at 1%'))):
            g = d[d.pi == pi]
            xs = sorted(g[xcol].unique()); pos = {x: k for k, x in enumerate(xs)}
            g = g.assign(xpos=g[xcol].map(pos))
            PL.line_by_model(axes[i, j], g, 'xpos', met, models, '', lab, f'{title}')
            axes[i, j].set_xticks(range(len(xs)))
            axes[i, j].set_xticklabels([str(x)[2:] if xcol == 'month' else str(x) for x in xs], rotation=45,
                                       ha='right', fontsize=6.5)
            if r3 is not None and xcol == 'month' and not title.startswith('EMBER-trained'):
                ds_ = 'bodmas' if title.startswith('BODMAS') else 'ember'
                rc = r3[(r3.dataset == ds_) & (r3.rung == 'random_contrast') & (r3.rule == 'fpr') &
                        (r3.alpha == 0.01) & (r3.pi == pi) & (r3.model == 'lgbm')]
                if len(rc):
                    axes[i, j].axhline(rc[met].mean(), color=PL.MUTED, lw=0.9)
                    axes[i, j].text(0, rc[met].mean(), 'LightGBM, random split', fontsize=6.5, color=PL.INK2,
                                    va='bottom')
    hl = {}                                                                   # legend over every panel's models
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(list(hl.values()), list(hl.keys()), ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(); PL.save(fig, figs / 'fig_r5_08_temporal_decay')


def fig_ladder(base, figs):
    r2 = load(base, 'e2')
    if r2 is None:
        return print('skip ladder')
    sel = r2[(r2.rule == 'fpr') & (r2.alpha == 0.01) & (r2.pi == 0.01) & r2.model.isin(['stump', 'lr', 'rf', 'lgbm'])]
    sel = sel.assign(cell=sel.pool + ' | ' + sel.rung + ' | ' + sel.dedup)
    order = [c for c in ('all | random | none', 'all | random | exact', 'all | grouped | exact',
                         'genuine | random | none', 'genuine | random | exact', 'genuine | grouped | exact',
                         'genuine | grouped | near', 'genuine | lofo | exact', 'genuine | loco | exact',
                         'genuine | single_family | exact', 'genuine | temporal | exact', 'all | temporal | exact')
             if c in set(sel.cell)]
    fig, axes = plt.subplots(1, 2, figsize=(PL.FULL_W, 3.6), sharey=True)
    for ax, met, lab in ((axes[0], 'precision', 'Precision at 1% prevalence'), (axes[1], 'auc_pr', 'AUC-PR at 1% prevalence')):
        for m in ('stump', 'lr', 'rf', 'lgbm'):
            g = sel[sel.model == m].groupby('cell')[met].mean().reindex(order)
            ax.plot(g.to_numpy(), np.arange(len(order)), color=PL.MODEL_COLOR[m], marker=PL.MARKER[m], ms=4, lw=1,
                    ls=PL.LINESTYLE[m], label=PL.MODEL_LABEL[m])
        ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order, fontsize=6.5); ax.invert_yaxis()
        ax.set_xlabel(lab); ax.set_xlim(0, 1.02)
    axes[0].legend(loc='lower left', fontsize=6.5)
    axes[0].set_title('Protocol ladder (pool | split | dedup), FPR <= 1% threshold', loc='left')
    fig.tight_layout(); PL.save(fig, figs / 'fig_x_09_protocol_ladder')


def fig_validity(base, figs):
    sw = load(base, 'e1', 'invariant_sweep.csv'); uni = load(base, 'e1', 'univariate_auc_genuine.csv')
    lam_p = base / 'e1' / 'genealogy_lambda.npy'
    if sw is None:
        return print('skip validity')
    fig, axes = plt.subplots(1, 3, figsize=(PL.FULL_W, 2.6))
    for i, s in enumerate(('min', 'core', 'full')):
        g = sw[sw.invariant_set == s]
        axes[0].plot(g.tol, g.genuine_benign, marker='os^'[i], color=PL.SLOTS[i], ls=['-', '--', ':'][i], label=s)
    axes[0].set_xscale('log'); axes[0].set_xlabel('Tolerance'); axes[0].set_ylabel('Benign rows passing (genuine)')
    axes[0].set_title('(a) Invariant gate', loc='left'); axes[0].legend(title='invariant set', fontsize=6.5)
    if lam_p.exists():
        lam = np.load(lam_p); lm = np.minimum(lam, 1 - lam)
        axes[1].hist(lm, bins=25, color=PL.SLOTS[0], edgecolor=PL.SURF)
        axes[1].set_xlabel('min(lambda, 1 - lambda)'); axes[1].set_ylabel('Synthetic rows')
        axes[1].set_title('(b) SMOTE interpolation position', loc='left')
    if uni is not None:
        u = uni.head(10)[::-1]
        axes[2].barh(np.arange(len(u)), u.auc_sym, color=PL.SLOTS[2], edgecolor=PL.SURF, height=0.7)
        axes[2].set_yticks(np.arange(len(u))); axes[2].set_yticklabels(u.feature, fontsize=6)
        axes[2].set_xlim(0.9, 1.0); axes[2].set_xlabel('Single-feature AUC'); axes[2].grid(axis='y', visible=False)
        axes[2].set_title('(c) One feature separates the classes', loc='left')
    fig.tight_layout(); PL.save(fig, figs / 'fig_x_10_data_validity')


def fig_shap(base, figs):
    d = load(base, 'e6', 'shap_shares.csv')
    if d is None:
        return print('skip shap')
    blocks = [c for c in d.columns if c.startswith('share_')]
    g = d[d.model == 'lgbm'].groupby(['pool', 'rung'])[blocks].mean()
    fig, ax = plt.subplots(figsize=(PL.FULL_W, 2.8))
    left = np.zeros(len(g))
    pal = PL.SLOTS + [PL.MUTED]
    for i, b in enumerate(blocks):
        ax.barh(np.arange(len(g)), g[b], left=left, color=pal[i % len(pal)], edgecolor=PL.SURF, height=0.6,
                label=b.replace('share_', ''))
        left += g[b].to_numpy()
    ax.set_yticks(np.arange(len(g))); ax.set_yticklabels([f'{p} | {r}' for p, r in g.index]); ax.invert_yaxis()
    ax.set_xlabel('Share of mean |SHAP| (LightGBM)'); ax.set_xlim(0, 1); ax.grid(axis='y', visible=False)
    ax.legend(ncol=5, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    PL.save(fig, figs / 'fig_x_11_shap_block_shares')


def fig_privacy(base, figs):
    d = load(base, 'e5')
    if d is None:
        return print('skip privacy')
    fig, axes = plt.subplots(1, 2, figsize=(PL.FULL_W, 2.7))
    dp = d[d.part == 'dp']
    if 'dp_variant' in dp:                                                    # frontier: DP points + non-private end
        dp = dp[dp.dp_variant != 'clip_only']
    for i, u in enumerate(('row', 'group')):
        g = dp[dp.unit == u].copy(); g['eps'] = g['epsilon_target'].replace(np.inf, 64.0)
        g = g.groupby('eps')['auc_pr_1pct'].mean()
        axes[0].plot(g.index, g.values, marker='os'[i], color=PL.SLOTS[i], ls=['-', '--'][i], label=f'{u}-level DP')
    axes[0].set_xscale('log', base=2); axes[0].set_xlabel('epsilon (64 = non-private)')
    axes[0].set_ylabel('AUC-PR at 1%'); axes[0].legend(); axes[0].set_title('(a) DP-SGD utility frontier', loc='left')
    mia = d[d.part == 'mia']
    if len(mia):
        g = mia.groupby(['model', 'variant'])['lira_tpr_at_fpr_0.01'].mean().unstack()
        x = np.arange(len(g.index)); w = 0.8 / max(len(g.columns), 1)
        for j, v in enumerate(g.columns):
            axes[1].bar(x + (j - (len(g.columns) - 1) / 2) * w, g[v], width=w, color=PL.SLOTS[j], edgecolor=PL.SURF,
                        label=v)
        axes[1].set_xticks(x); axes[1].set_xticklabels([PL.MODEL_LABEL.get(m, m) for m in g.index])
        axes[1].axhline(0.01, color=PL.MUTED, lw=0.8); axes[1].set_ylabel('LiRA TPR at FPR 1%')
        axes[1].legend(title='rebalancing', fontsize=6.5); axes[1].set_title('(b) Membership leakage', loc='left')
        axes[1].grid(axis='x', visible=False)
    fig.tight_layout(); PL.save(fig, figs / 'fig_x_12_privacy')


def fig_evasion(base, figs):
    d = load(base, 'e8')
    if d is None:
        return print('skip evasion')
    g = d[d.part == 'greedy']
    fig, axes = plt.subplots(1, 2, figsize=(PL.FULL_W, 2.6), sharey=True)
    cmap = {'lgbm': PL.MODEL_COLOR['lgbm'], 'rf': PL.MODEL_COLOR['rf'], 'lgbm_mono': PL.SLOTS[0],
            'lgbm_mono_noavg': PL.SLOTS[5]}
    for ax, cap in zip(axes, ('user', 'admin')):
        for i, (m, h) in enumerate(g[g.capability == cap].groupby('model')):
            ax.plot(h.budget, h.evasion_rate, marker='osD^'[i % 4], color=cmap.get(m, PL.MUTED), label=m)
        ax.set_xscale('log'); ax.set_xlabel('Action budget k'); ax.set_title(f'{cap}-level attacker', loc='left')
    axes[0].set_ylabel('Evasion rate'); axes[0].legend(fontsize=6.5)
    fig.tight_layout(); PL.save(fig, figs / 'fig_x_13_evasion')


def main():
    a = parse('render figures')
    base = Path(C.RUNS) / a.mode; figs = base / 'figures'; figs.mkdir(parents=True, exist_ok=True)
    PL.style()
    H = headline_frame(base); H.to_csv(base / 'headline_table.csv', index=False)
    for f in (lambda: fig_bars(base, figs, H), lambda: fig_roc_pr(base, figs), lambda: fig_confusion(base, figs, H),
              lambda: fig_importance(base, figs), lambda: fig_learning(base, figs), lambda: fig_violins(base, figs),
              lambda: fig_temporal(base, figs), lambda: fig_ladder(base, figs), lambda: fig_validity(base, figs),
              lambda: fig_shap(base, figs), lambda: fig_privacy(base, figs), lambda: fig_evasion(base, figs)):
        try:
            f()
        except Exception as e:                                                # keep rendering the others
            print('figure failed:', repr(e))
    print(sorted(p.name for p in figs.glob('*.png')))


if __name__ == '__main__':
    main()

"""plots.py - R5 figure functions (matplotlib, static journal figures; Springer single column <= 174 mm wide).

Palette: the validated 8-slot categorical palette (light mode), assigned to models in a FIXED display order
(colour follows the entity, never its rank); the depth-1 stump is a neutral grey reference. Secondary encoding:
line styles / markers, so figures stay legible in greyscale and for colour-vision deficiency. One y-axis per
panel, hairline solid grid, thin marks, legend whenever there are >= 2 series.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MODEL_ORDER = ['lr', 'linsvm', 'gnb', 'knn', 'rf', 'et', 'xgb', 'lgbm']           # slot i -> model i
MODEL_COLOR = {m: SLOTS[i] for i, m in enumerate(MODEL_ORDER)}
MODEL_COLOR.update({'stump': MUTED, 'dt': '#5f5e5a', 'stack_carrier': '#3d3d3a', 'xmal': '#7a7974',
                    'mlp_poc': '#a6a59f', 'ember_lgbm_official': '#3d3d3a'})
MODEL_LABEL = {'lr': 'LR', 'linsvm': 'Linear SVM', 'gnb': 'Gaussian NB', 'knn': 'k-NN', 'rf': 'Random forest',
               'et': 'Extra trees', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'stump': 'Stump (1 feature)',
               'dt': 'Decision tree', 'stack_carrier': 'Stack (Carrier et al.)', 'xmal': 'XMal (5 features)',
               'mlp_poc': 'MLP (PoC)', 'ember_lgbm_official': 'EMBER official LGBM'}
LINESTYLE = {m: s for m, s in zip(MODEL_ORDER, ['-', '--', '-.', ':', '-', '--', '-.', '-'])}
LINESTYLE['stump'] = ':'
MARKER = {m: k for m, k in zip(MODEL_ORDER, ['o', 's', '^', 'v', 'D', 'P', 'X', 'o'])}
MARKER['stump'] = '.'
MM = 1 / 25.4
FULL_W = 174 * MM                                                                # single-column text width


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.edgecolor': AXIS,
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round', 'figure.facecolor': SURF,
        'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False, 'pdf.fonttype': 42})


def save(fig, path_stem):
    fig.savefig(f'{path_stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{path_stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def ordered(models):
    order = ['stump'] + MODEL_ORDER + ['dt', 'stack_carrier', 'xmal', 'mlp_poc', 'ember_lgbm_official']
    return [m for m in order if m in set(models)]


def bar_metric_by_dataset(df, metric, ax, datasets, models, err=None, ylabel=None, ylim=None):
    """Grouped bars: x = dataset, hue = model (fixed colours); 1px surface gap between adjacent bars."""
    n = len(models); width = 0.8 / n
    for i, m in enumerate(models):
        vals = [df.loc[(df['dataset'] == d) & (df['model'] == m), metric].mean() for d in datasets]
        x = np.arange(len(datasets)) + (i - (n - 1) / 2) * width
        e = None
        if err is not None:
            lo = [df.loc[(df['dataset'] == d) & (df['model'] == m), err[0]].mean() for d in datasets]
            hi = [df.loc[(df['dataset'] == d) & (df['model'] == m), err[1]].mean() for d in datasets]
            e = np.vstack([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
        ax.bar(x, vals, width=width, color=MODEL_COLOR.get(m, MUTED), edgecolor=SURF, linewidth=0.6,
               label=MODEL_LABEL.get(m, m), yerr=e, error_kw={'elinewidth': 0.6, 'ecolor': INK2, 'capsize': 0})
    ax.set_xticks(np.arange(len(datasets))); ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel or metric); ax.grid(axis='x', visible=False)
    if ylim:
        ax.set_ylim(*ylim)


def roc_overlay(ax, curves, title, log_fpr=True):
    """curves: dict model -> (y, s). FPR on a log axis exposes the low-FPR regime that deployment needs."""
    for m in ordered(curves):
        y, s = curves[m]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(np.clip(fpr, 1e-5, 1), tpr, color=MODEL_COLOR.get(m, MUTED), ls=LINESTYLE.get(m, '-'),
                label=MODEL_LABEL.get(m, m))
    if log_fpr:
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1)
    ax.set_xlabel('False-positive rate' + (' (log)' if log_fpr else '')); ax.set_ylabel('True-positive rate')
    ax.set_title(title, loc='left')


def pr_overlay(ax, curves, pi, title):
    """PR curves re-weighted to prevalence pi (expected precision under Eq. 7 subsampling); chance line = pi."""
    for m in ordered(curves):
        y, s = curves[m]
        w = np.where(y == 1, pi / y.mean(), (1 - pi) / (1 - y.mean()))
        prec, rec, _ = precision_recall_curve(y, s, sample_weight=w)
        ax.plot(rec, prec, color=MODEL_COLOR.get(m, MUTED), ls=LINESTYLE.get(m, '-'), label=MODEL_LABEL.get(m, m))
    ax.axhline(pi, color=MUTED, lw=0.8); ax.text(0.02, pi, f'chance = {pi:g}', color=INK2, va='bottom', fontsize=7)
    ax.set_xlabel('Recall'); ax.set_ylabel(f'Precision at prevalence {pi:g}'); ax.set_ylim(0, 1.02)
    ax.set_title(title, loc='left')


def confusion_heatmap(ax, tp, fp, tn, fn, title, unit='per 10,000 events'):
    M = np.array([[tn, fp], [fn, tp]], float)
    ax.imshow(np.log10(M + 1), cmap='Blues', vmin=0, vmax=4.1)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f'{v:,.1f}', ha='center', va='center', color=INK if v < 300 else SURF, fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred. benign', 'Pred. malware'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Benign', 'Malware']); ax.grid(False)
    ax.set_title(f'{title}\n({unit})', loc='left')


def hbar_importance(ax, names, values, title, color=SLOTS[0], top=15):
    o = np.argsort(values)[::-1][:top][::-1]
    ax.barh(np.arange(len(o)), np.asarray(values)[o], color=color, edgecolor=SURF, height=0.7)
    ax.set_yticks(np.arange(len(o))); ax.set_yticklabels(np.asarray(names)[o], fontsize=6.5)
    ax.grid(axis='y', visible=False); ax.set_title(title, loc='left')


def line_by_model(ax, df, x, y, models, xlabel, ylabel, title, band=None, logx=False):
    for m in ordered(models):
        d = df[df['model'] == m].groupby(x)[y].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(d[x], d['mean'], color=MODEL_COLOR.get(m, MUTED), ls=LINESTYLE.get(m, '-'),
                marker=MARKER.get(m, 'o'), ms=3.5, label=MODEL_LABEL.get(m, m))
        if band and len(d):
            ax.fill_between(d[x], d['min'], d['max'], color=MODEL_COLOR.get(m, MUTED), alpha=0.10, lw=0)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc='left')


def violin_by_model(ax, df, metric, models, title, ylabel):
    data = [df.loc[df['model'] == m, metric].dropna().to_numpy() for m in models]
    keep = [i for i, d in enumerate(data) if len(d)]
    parts = ax.violinplot([data[i] for i in keep], positions=keep, showmedians=True, widths=0.7)
    for k, b in zip(keep, parts['bodies']):
        b.set_facecolor(MODEL_COLOR.get(models[k], MUTED)); b.set_edgecolor(SURF); b.set_alpha(0.55)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        if key in parts:
            parts[key].set_color(INK2); parts[key].set_linewidth(0.8)
    for k in keep:
        ax.scatter(np.full(len(data[k]), k) + np.random.default_rng(0).uniform(-0.12, 0.12, len(data[k])),
                   data[k], s=4, color=INK2, zorder=3, linewidths=0)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], rotation=30,
                                                          ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title, loc='left'); ax.grid(axis='x', visible=False)

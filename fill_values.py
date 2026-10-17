"""fill_values.py - computed placeholder values for the S9 report (assemble_report.py reads the JSON it writes).

Every value is read from runs/<mode>/ outputs, runs/chain.log or the source tree, so the report never carries a
hand-typed number for these fields. Usage: python3 -I tools/fill_values.py runs/smoke report/fill.json
"""
import json, re, sys
from datetime import datetime
from pathlib import Path
import pandas as pd

base = Path(sys.argv[1]); out = Path(sys.argv[2]); ROOT = Path('/home/claude/s9')
fill = {}

# wall-clock per step from the last chain run (start/exit pairs in runs/chain.log)
log = (ROOT / 'runs' / 'chain.log').read_text().splitlines()
start = max(i for i, l in enumerate(log) if l.startswith('== chain v4 start'))
t0 = {}
dur = {}
for l in log[start:]:
    m = re.match(r'== (\S+) start .*?(\d\d:\d\d:\d\d)$', l)
    if m:
        t0[m.group(1)] = datetime.strptime(m.group(2), '%H:%M:%S')
    m = re.match(r'== (\S+) exit (\d+) (\d\d:\d\d:\d\d)$', l)
    if m and m.group(1) in t0:
        dur[m.group(1)] = (datetime.strptime(m.group(3), '%H:%M:%S') - t0[m.group(1)]).seconds


def fmt_s(s):
    return f'{s / 60:.1f} min' if s >= 90 else f'{s:.0f} s'


e0 = pd.read_csv(base / 'e0' / 'timing.csv').wall_s.sum()
fill['{{T_E0}}'] = fmt_s(e0)
for key, script in (('E1', 'e1_validity_audit'), ('E2', 'e2_protocol_ladder'), ('E3', 'e3_temporal_static'),
                    ('E4', 'e4_nids'), ('E5', 'e5_privacy'), ('E6', 'e6_xai'), ('E7', 'e7_decision_layer'),
                    ('E8', 'e8_adversarial'), ('E9', 'e9_drift'), ('E10', 'e10_curves'), ('STATS', 'stats_analysis')):
    fill['{{T_' + key + '}}'] = fmt_s(dur[script]) if script in dur else 'n/a'
total = e0 + sum(dur.values())
fill['{{T_TOTAL}}'] = f'{total / 3600:.1f} h'

# S10 CPU budget (tools/estimate_full_cost.py writes full_cost_estimate.csv)
fc = pd.read_csv(base / 'full_cost_estimate.csv')
fill['{{FULL_HOURS}}'] = f"{fc.loc[fc.experiment == 'TOTAL', 'full_s_est'].iloc[0] / 3600:.0f}"

# number of source files in the bundle (same rule as tools/bundle_code.py)
files = (['README.md', 'requirements.txt', 'run_all.py'] + list((ROOT / 'harness').glob('*.py'))
         + list((ROOT / 'experiments').glob('*.py')) + ['tests/test_harness.py'] + list((ROOT / 'tools').glob('*.py')))
fill['{{N_FILES}}'] = str(len(files))

# official EMBER model on BODMAS months
r3 = pd.read_csv(base / 'e3' / 'results.csv', low_memory=False)
o = r3[(r3.dataset == 'ember_official->bodmas') & (r3.rule == 'fpr') & (r3.alpha == 0.01) & (r3.pi == 0.5)].auc_roc
fill['{{OFFICIAL_BODMAS_AUC}}'] = f'{o.min():.4f}–{o.max():.4f}'

# benign near-duplicate shares (E2 gates, seed 0)
g = pd.read_csv(base / 'e2' / 'gates.csv'); g['drop_block'] = g['drop_block'].fillna('none')
g = g[(g.drop_block == 'none') & (g.dedup == 'exact')]
sh = g.groupby(['pool', 'rung'])['test_near_dup_share_benign'].mean()
fill['{{BENIGN_NEARDUP_SENTENCE}}'] = (
    f"the share of benign test rows with a training row within 0.05 z-units is {100 * sh[('genuine', 'grouped')]:.1f}% "
    f"in grouped folds of the genuine pool vs {100 * sh[('genuine', 'random')]:.1f}% in random folds "
    f"({100 * sh[('all', 'grouped')]:.1f}% vs {100 * sh[('all', 'random')]:.1f}% in the all-benign pool).")

tl = ROOT / 'report' / 'tldr_signals.md'
if tl.exists():
    lines = [l.strip() for l in tl.read_text().splitlines() if l.strip()]
    fill['{{TLDR_SIGNALS}}'] = '\n' + '\n'.join('  ' + l for l in lines)    # nested list under the TL;DR bullet
out.write_text(json.dumps(fill, indent=1, ensure_ascii=False))
print(json.dumps(fill, indent=1, ensure_ascii=False)[:3000])

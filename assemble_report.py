"""assemble_report.py - build the S9 report from its parts, filling code excerpts and the manifest verbatim."""
import hashlib, inspect, sys, textwrap
from pathlib import Path
sys.path.insert(0, '/home/claude/s9')
from harness import data_malmem, splits, metrics, thresholds, evaluate   # noqa: E402

R = Path('/home/claude/s9/report'); ROOT = Path('/home/claude/s9')


def src(*objs):
    return '\n\n'.join(inspect.getsource(o).rstrip() for o in objs)


def manifest():
    files = (['README.md', 'requirements.txt', 'run_all.py'] + sorted(f'harness/{p.name}' for p in (ROOT / 'harness').glob('*.py'))
             + sorted(f'experiments/{p.name}' for p in (ROOT / 'experiments').glob('*.py'))
             + ['tests/test_harness.py'] + sorted(f'tools/{p.name}' for p in (ROOT / 'tools').glob('*.py')))
    rows = ['| File | Lines | SHA-256 (16 hex) |', '|---|---|---|']
    tot = 0
    for f in files:
        t = (ROOT / f).read_text(); n = t.count('\n') + 1; tot += n
        rows.append(f'| `{f}` | {n} | `{hashlib.sha256(t.encode()).hexdigest()[:16]}` |')
    rows.append(f'| **total ({len(files)} files)** | **{tot}** | |')
    return '\n'.join(rows)


def main(fill):
    text = ''.join((R / p).read_text() for p in ('S9_part1.md', 'S9_part2.md', 'S9_part3.md', 'S9_part4_template.md',
                                                 'S9_part5.md', 'S9_part6_tail.md', 'S9_part7_refs.md'))
    rep = {'{{MANIFEST}}': manifest(),
           '{{EXCERPT_INVARIANTS}}': src(data_malmem.invariant_table, data_malmem.validity_gate),
           '{{EXCERPT_GATES}}': src(splits.check_split, splits._straddlers, splits.purge_smote_straddlers),
           '{{EXCERPT_PANEL}}': src(metrics.class_weights, metrics.panel),
           '{{EXCERPT_EVALUATE}}': src(thresholds.fpr_threshold, evaluate.evaluate)}
    import pandas as pd
    h = pd.read_csv(R / 'hypotheses_register.csv')
    lines = ['| ID | Origin | Prediction | Falsified if | Experiment | Output |', '|---|---|---|---|---|---|']
    for _, r in h.iterrows():
        lines.append(f"| {r['id']} | {r['origin']} | {r['prediction']} | {r['decision_rule_falsified_if']} | "
                     f"{r['experiment']} | `{r['output_file']}` |")
    rep['{{HYPOTHESES}}'] = '\n'.join(lines)
    rep.update(fill)
    for k, v in rep.items():
        text = text.replace(k, v)
    left = [l for l in text.splitlines() if '{{' in l]
    if left:
        print('UNFILLED:', left[:5])
    (R / 'S9_Experimental_Harness.md').write_text(text)
    print('written', len(text), 'chars')


if __name__ == '__main__':
    import json
    main(json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {})

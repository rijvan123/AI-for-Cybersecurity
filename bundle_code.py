"""bundle_code.py - write the full harness source into one Markdown file (project persistence) and a zip archive."""
import hashlib, zipfile, sys
from pathlib import Path
ROOT = Path('/home/claude/s9')
FILES = (['README.md', 'requirements.txt', 'run_all.py'] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'harness').glob('*.py'))
         + sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'experiments').glob('*.py'))
         + ['tests/test_harness.py'] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'tools').glob('*.py')))
out_md, out_zip = Path(sys.argv[1]), Path(sys.argv[2])
lines = ['# S9 harness - full source code', '',
         'Every file of the CPU-only experimental harness, verbatim. Recreate the tree with these paths under one root',
         '(the code expects /home/claude/s9; change ROOT/RAW in harness/config.py elsewhere).', '',
         '| File | Lines | SHA-256 (first 16 hex) |', '|---|---|---|']
body = []
for f in FILES:
    p = ROOT / f
    if not p.exists():
        continue
    txt = p.read_text()
    h = hashlib.sha256(txt.encode()).hexdigest()[:16]
    lines.append(f'| `{f}` | {txt.count(chr(10)) + 1} | `{h}` |')
    lang = 'python' if f.endswith('.py') else ('text' if f.endswith('.txt') else 'markdown')
    body += ['', f'## `{f}`', '', f'```{lang}', txt.rstrip('\n'), '```']
out_md.write_text('\n'.join(lines + body) + '\n')
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        if (ROOT / f).exists():
            z.write(ROOT / f, f's9_harness/{f}')
print(out_md, out_md.stat().st_size, out_zip, out_zip.stat().st_size)

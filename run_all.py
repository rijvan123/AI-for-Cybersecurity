#!/usr/bin/env python3
"""run_all.py - S9 orchestrator: unit tests, then E0..E10 in dependency order, statistics and figures.

    python3 -I run_all.py --mode smoke      # verification run (seed 0, reduced grids)  ~2-3 h on 2 vCPU
    python3 -I run_all.py --mode full       # S10 run (seeds 0-4; E3 seeds 0-2)         see S9 report for budget
Each experiment writes runs/<mode>/<exp>/ and a timing.csv; E0 aborts the run if any input hash or count fails.
"""
import subprocess, sys, time, argparse
from pathlib import Path
ROOT = Path('/home/claude/s9')
ORDER = ['e0_verify_data', 'e1_validity_audit', 'e2_protocol_ladder', 'e3_temporal_static', 'e4_nids',
         'e5_privacy', 'e6_xai', 'e8_adversarial', 'e10_curves', 'e7_decision_layer', 'e9_drift', 'stats_analysis']
ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='smoke'); ap.add_argument('--only', nargs='*')
a = ap.parse_args()
t_all = time.time()
r = subprocess.run([sys.executable, '-I', str(ROOT / 'tests/test_harness.py')])
if r.returncode:
    sys.exit('unit tests failed')
for e in (a.only or ORDER):
    t0 = time.time(); print(f'== {e}', flush=True)
    r = subprocess.run([sys.executable, '-I', str(ROOT / 'experiments' / f'{e}.py'), '--mode', a.mode])
    print(f'== {e} exit {r.returncode} in {time.time() - t0:.0f}s', flush=True)
    if r.returncode and e == 'e0_verify_data':
        sys.exit('data verification failed: aborting')
subprocess.run([sys.executable, '-I', str(ROOT / 'experiments/make_figures.py'), '--mode', a.mode])
print(f'total {time.time() - t_all:.0f}s')

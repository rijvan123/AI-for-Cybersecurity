# S9 experimental harness (CPU-only)

Harness for the Discover Artificial Intelligence manuscript *AI for Cybersecurity: Innovations in Threat Detection,
Prevention, and Privacy*. It implements the S2 lock (fused framework; CIC-MalMem-2022 data-validity flagship;
contributions C1-C5), the S3 binding constraints 1-12, the S5 survivors, the S6 redesign conditions, the S7 modules
and the S8 equations. Everything runs on 2 vCPU / 7 GB RAM without a GPU.

## Layout

| Path | Role |
|---|---|
| `harness/config.py` | paths, SHA-256 pins, documented counts, every design constant |
| `harness/data_*.py` | loaders: MalMem (+ invariant gate, genealogy, identity), BODMAS, EMBER (v2 vectoriser), NSL-KDD, UNSW-NB15 |
| `harness/splits.py` | protocol ladder + leakage gates A0-A5 (index, group, hash, temporal C1/C2, SMOTE parents) |
| `harness/models.py` | model zoo, leakage-free pipelines, grouped stacking, XMal, MLP proof of concept |
| `harness/thresholds.py`, `metrics.py` | validation-only operating points; metric panel at any prevalence |
| `harness/stats.py`, `drift.py`, `privacy.py`, `adversarial.py`, `explain.py` | statistics, drift, privacy audit, evasion, SHAP |
| `harness/plots.py`, `experiments/make_figures.py` | R5 figures (validated palette, journal sizing) |
| `experiments/e0..e10_*.py` | experiments (see the S9 report for the design table) |
| `tests/test_harness.py` | 29 unit tests: S8 identities, leakage gates and regression tests from the code review |
| `tools/make_ember_subsample.py` | stratified EMBER-2018 subsample (run where the raw EMBER CSVs live) |
| `tools/summarise_runs.py`, `estimate_full_cost.py` | verification tables; S10 CPU budget from smoke timings |
| `tools/assemble_report.py`, `bundle_code.py` | report assembly; source bundle (Markdown + zip) |

## Run

```bash
pip install -r requirements.txt
python3 -I tests/test_harness.py               # must print 29/29 passed
python3 -I run_all.py --mode smoke             # verification run (seed 0)
python3 -I run_all.py --mode full              # S10 run
```

Inputs are expected under `/home/claude/data_raw/<source>/` (one folder per source; see `harness/config.py`);
E0 re-hashes every file and stops if anything differs from the S9 pins.

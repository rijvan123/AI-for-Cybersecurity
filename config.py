"""config.py - single source of truth for paths, expected file hashes and every design constant of the S9 harness.

Constants marked [S8] come from the S8 hand-off; [S3] from the S3 binding constraints; [S5]/[S6] from the
venue filter and Reviewer-#2 redesign; [S9] are decisions taken here after measuring the data (see the S9 report).
Nothing in this file is fitted to data: changing a constant changes the experiment, never a result silently.
"""
from pathlib import Path

# ----------------------------------------------------------------------------------------------- paths
ROOT = Path('/home/claude/s9')                       # harness root (code + caches + runs)
RAW = Path('/home/claude/data_raw')                  # untrusted downloaded / staged data, one folder per source
CACHE = ROOT / 'cache'                               # derived arrays (decompressed BODMAS, vectorised EMBER, ...)
RUNS = ROOT / 'runs'                                 # experiment outputs: runs/<mode>/<experiment>/

PATHS = {
    # CIC-MalMem-2022, official CIC release (Category = "<cat>-<family>-<hash>-<run>.raw" or "Benign")
    'malmem': RAW / 'malmem/Obfuscated-MalMem2022.csv',
    # author's re-layout of the same 58,596 rows (Category split into family + Filename); verified identical multiset
    'malmem_alt': RAW / 'malmem/MalMem2022.csv',
    # MemMal-D2024-style dated malware file (Year/Month per row; VirusTotal first-submission month per Maniriho et al.)
    'memmal_dated_mal': RAW / 'memmal_dated/All Malware Samples-Combined2006-2021.csv',
    'memmal_dated_ben': RAW / 'memmal_dated/All Benign Samples-combined.csv',
    # BODMAS (Yang et al., IEEE SPW 2021): EMBER-v2 feature matrix + metadata + malware categories
    'bodmas_npz': RAW / 'bodmas/bodmas.npz',
    'bodmas_meta': RAW / 'bodmas/bodmas_metadata.csv',
    'bodmas_cat': RAW / 'bodmas/bodmas_malware_category.csv',
    # EMBER-2018: stratified temporal subsample of the raw features (tools/make_ember_subsample.py) + official model
    'ember_shards': RAW / 'ember_shards',
    'ember_model': RAW / 'ember/ember_model_2018.txt',
    # NSL-KDD (GitHub mirror defcom17/NSL_KDD) and UNSW-NB15 official partitions (GitHub mirror Nir-J/ML-Projects)
    'nsl_train': RAW / 'nslkdd/KDDTrain+.txt',
    'nsl_test': RAW / 'nslkdd/KDDTest+.txt',
    'unsw_train': RAW / 'unswnb15/UNSW_NB15_training-set.csv',
    'unsw_test': RAW / 'unswnb15/UNSW_NB15_testing-set.csv',
}

# SHA-256 of every input as verified in S9 (E0 re-checks them before any experiment runs)  [S3 constraint 10]
SHA256 = {
    'malmem': '624ecb6c9229cf62bd3b8f65d23a2b132760e6bada458aa46eee134a313840c4',
    'malmem_alt': 'cc7a637a174ffe797e0af0375bce3c09561f0dc8b8115c0a6292718034f5012a',
    'memmal_dated_mal': '9ab9f23db1c6308a0567a61e5bc8ca5bad2fc86ac0038ab4afb02287958ccdc8',
    'memmal_dated_ben': '2304f0fb724517f2e83400a75ed6dcf86700254ef0db42c567205b736406b980',
    'bodmas_npz': '0a833e46c22c096920c8068d4fb54b868067777c277c0a6eecf527c98d7e5a9e',
    'bodmas_meta': '5f1dc793cac4a540ef9e0c40fd7897142095621790d3c6f2d6637971806f4c50',
    'bodmas_cat': '617fda69347ec907a79f198add431c8bb39102781de41cc03b7cd1718cff2198',
    'ember_model': '509de4a83e2d76b69fb3025ced0429e5f40f6a95ca40122b9eedd03317fff5e2',
    'nsl_train': '1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95',
    'nsl_test': 'fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84',
    'unsw_train': 'bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa',
    'unsw_test': '734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559',
}
EMBER_SHARD_SHA256 = {
    'train_features_1.sel.csv.gz': 'beadd31b406f3cc9bbc3b898c1044138a730140cd1973e7d13ccdfd0e0a9fd50',
    'train_features_2.sel.csv.gz': '303547564b38e325e057e433250e6515c934355e7f0f510e7c9cd023582b3196',
    'train_features_3.sel.csv.gz': 'b1148279b89a8c33580ff16b11f1cffd19a616d18dc45efbe42a1de61f577805',
    'train_features_4.sel.csv.gz': '4b8c621c9e4aac1b3c36da39a8014a4ce7a73cb425550261cd46408185b5b311',
    'train_features_5.sel.csv.gz': 'e9b7dc7bd924cdf3879896a97f5bccd798ac3678df19ac244640cc9aa29f2b5b',
    'test_features.sel.csv.gz': '3031e87fcd1bc35f5eba092edaca7ba9cad839abf40cf461ae21dec5a1fd3f96',
    'ember_selected_manifest.csv': 'b9b77fb7bd522f85f09a129b22ef26b08a906419951499f51064b0df513cca57',
}

# Documented record counts used as acceptance tests in E0 (dataset papers / official pages)
EXPECTED_COUNTS = {
    'malmem_rows': 58596, 'malmem_benign': 29298, 'malmem_malware': 29298,     # Carrier et al., ICISSP 2022
    'memmal_dated_rows': 28870,                                                # S8/S3 measurement
    'bodmas_rows': 134435, 'bodmas_malware': 57293, 'bodmas_benign': 77142,    # Yang et al., IEEE SPW 2021
    'nsl_train': 125973, 'nsl_test': 22544,                                    # Tavallaee et al., CISDA 2009
    'unsw_train': 175341, 'unsw_test': 82332,                                  # Moustafa & Slay, MilCIS 2015 / ISJ 2016
    'ember_sub_rows': 60000,                                                   # 2,500 x 12 months x 2 classes
}

# ----------------------------------------------------------------------------------------------- reproducibility
SEEDS_FULL = (0, 1, 2, 3, 4)          # [S6/S7] five seeds for every refit grid in S10
SEEDS_SMOKE = (0,)                    # [S9] verification run uses one seed
N_FOLDS = 5                           # [S6] 5-fold (grouped) CV for non-temporal rungs
N_JOBS = 2                            # container has 2 cores; all estimators share this cap
B_BOOT = 1000                         # [S8 Eq. 94] group-level bootstrap resamples (S10); smoke uses 200
B_BOOT_SMOKE = 200
R_PREV = 200                          # [S8 H2] prevalence-subsampling replicates per (window, pi)

# ----------------------------------------------------------------------------------------------- base rates & costs
PREVALENCES = (0.5, 0.10, 0.01, 0.001)          # [S3 constraint 3] 50% (literature), 10%, 1%, 0.1%
PREVALENCE_LABELS = {0.5: '50%', 0.10: '10%', 0.01: '1%', 0.001: '0.1%'}
FPR_BUDGETS = (1e-3, 1e-2)                      # [S8] alpha for Neyman-Pearson thresholds (Eq. 30)
PRIMARY_FPR_BUDGET = 1e-2                       # [S9] point metrics reported at the validation threshold FPR<=1%
COST_RATIOS = (1, 10, 100)                      # [S8] C_FN / C_FP (Eq. 52-57)
TARGET_PRECISION = (0.9, 0.5)                   # [S8 Eq. 59] P* for FPR_max tables
ALERT_UNIT = 10_000                             # alarms reported per 10,000 events [S3 constraint 3]

# ----------------------------------------------------------------------------------------------- CIC-MalMem-2022 audit
INVARIANT_TOLERANCES = (1e-6, 1e-4, 1e-3, 1e-2)  # [S3 constraint 1] tolerance sweep
PRIMARY_TOLERANCE = 1e-3                          # [S9] fixed before modelling; sweep reported as sensitivity
PRIMARY_INVARIANT_SET = 'core'                    # [S9] {I1,I2,I3}; 'min' = {I1,I2} (S3), 'full' = all 17 checks
SVCSCAN_BLOCK = 'svcscan'                         # [S2 C4] block whose SHAP share is tested
THETA_DUP_EXACT = 1e-6                            # [S8] exact-duplicate radius in z-space (Eq. 9)
THETA_DUP_NEAR = 0.05                             # [S8] near-duplicate radius in z-space (Eq. 9)
GENEALOGY_K = 15                                  # [S9] genuine neighbours searched per synthetic row (SMOTE parents)

# Temporal protocol for the dated MalMem malware (malware-side only; benign undated)  [S8 hand-off]
MALMEM_TAU_S = (2019, 1)        # split month tau_s
MALMEM_DELTA = 6                # window length Delta (months); sensitivity Delta = 3 in S10
MALMEM_K = 6                    # nominal windows 2019H1..2021H2
MALMEM_M_MIN = 200              # minimum malware rows per window (Eq. 5 merge rule)
VAL_FRAC = 0.20                 # validation = last 20% of the training era by time, group-intact [S8 Eq. 10]

# ----------------------------------------------------------------------------------------------- static-PE temporal axes
BODMAS_WINDOW = ('2019-08', '2020-09')            # malware collection window; benign outside it is dropped [S5]
BODMAS_TRAIN_MONTHS = ('2019-08', '2019-09', '2019-10')
BODMAS_TEST_MONTHS = ('2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04',
                      '2020-05', '2020-06', '2020-07', '2020-08', '2020-09')
EMBER_TRAIN_MONTHS = tuple('2018-%02d' % m for m in range(1, 7))     # Jan-Jun 2018
EMBER_TEST_MONTHS = tuple('2018-%02d' % m for m in range(7, 13))     # Jul-Dec 2018
EMBER_OFFICIAL_TEST_MONTHS = ('2018-11', '2018-12')                   # official EMBER-2018 test period
EMBER_DIM = 2381

# ----------------------------------------------------------------------------------------------- privacy / FL / adversarial
DP_EPSILONS = (0.5, 1.0, 2.0, 4.0, 8.0)          # [S8 H6]
DP_CLIP = 1.0                                    # [S8] C_clip
DP_EPOCHS = 30                                   # passes over the data (expected) for DP-SGD
DP_LOT = 256                                     # expected lot size L (Poisson sampling rate q = L / n)
FL_CLIENTS = 10                                  # [S8 H7] N_c
FL_ALPHAS = (0.1, 0.5, 1.0)                      # Dirichlet concentration alpha_D
FL_ROUNDS = 50                                   # R
FL_LOCAL_EPOCHS = 1                              # E
MIA_FPRS = (1e-3, 1e-2)                          # [S3 constraint 8] TPR reported at FPR 0.1% and 1%
LIRA_SHADOWS = 16                                # offline LiRA shadow models (CPU-feasible on MalMem)
LIRA_MIN_OUT_PER_EXAMPLE_VAR = 32               # [S9] per-example LiRA variance only with >= 32 OUT shadows, else global (fixed) variance
ADV_EPS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)       # [S8] L_inf / L_2 budgets in z-space for the linear exactness test
ADV_ACTION_BUDGETS = (1, 2, 5, 10, 20, 50)       # [S9] problem-space action budget k for the greedy attack

# ----------------------------------------------------------------------------------------------- drift
DRIFT_BINS = 10                 # [S8] B_bin train-quantile bins
DRIFT_EPS = 1e-6                # [S8] additive smoothing epsilon_s
PSI_ALARM = 0.25                # conventional PSI alarm (credit-scoring origin; S7), reported not tuned

# ----------------------------------------------------------------------------------------------- explanation
SHAP_TOPK = 10                  # top-k for Jaccard / Kendall stability
SHAP_EXPLAIN_N = 2000           # rows explained per (model, split) to bound TreeSHAP cost
DELETION_KS = (1, 2, 3, 5, 10)  # features removed in the deletion (faithfulness) test

# ----------------------------------------------------------------------------------------------- model zoo selection
CORE_MODELS = ('stump', 'lr', 'linsvm', 'gnb', 'rf', 'et', 'xgb', 'lgbm')          # every cell of every grid
EXTRA_MODELS = ('knn', 'dt', 'stack_carrier', 'xmal', 'mlp_poc')                  # key cells only (CPU budget)
HIGH_DIM_MODELS = ('stump', 'lr', 'linsvm', 'gnb', 'rf', 'et', 'xgb', 'lgbm', 'knn')  # BODMAS / EMBER (2,381 dims)
KNN_MAX_TRAIN = 20000           # k-NN reference set cap (fixed-seed subsample) for CPU feasibility
MLP_MAX_TRAIN = 10000           # shallow MLP proof-of-concept trains on a fixed-seed subsample [project rule]

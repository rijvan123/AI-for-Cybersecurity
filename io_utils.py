"""io_utils.py - hashing, timing hooks (T1-T6 of S7 Algorithm 1), environment capture, seeded RNG and result I/O."""
import hashlib, json, os, platform, random, time, contextlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd


def sha256_file(path, chunk=1 << 20):
    """SHA-256 of a file, streamed (used by E0 to pin every input) [S3 constraint 10]."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def row_hashes(X, decimals=9):
    """Content hash h(x) per row on the rounded feature vector (S7 M1 step 5); exact duplicates share a hash."""
    Xr = np.round(np.asarray(X, dtype=np.float64), decimals)           # rounding removes float-print noise
    Xr[Xr == 0] = 0.0                                                  # -0.0 and 0.0 must hash identically
    return np.array([hashlib.blake2b(r.tobytes(), digest_size=12).hexdigest() for r in Xr])


def set_seeds(seed):
    """Fix python, numpy and hash seeds; estimators receive random_state explicitly as well."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return np.random.default_rng(seed)


class TimingLog:
    """Collects CPU wall-clock per stage (S7 hooks T1-T6); written next to every experiment's results."""

    def __init__(self):
        self.rows = []

    @contextlib.contextmanager
    def time(self, stage, **tags):
        t0 = time.perf_counter()                                      # wall clock (single process, CPU only)
        c0 = time.process_time()                                      # CPU time of this process
        try:
            yield
        finally:
            self.rows.append({'stage': stage, 'wall_s': time.perf_counter() - t0,
                              'cpu_s': time.process_time() - c0, **tags})

    def frame(self):
        return pd.DataFrame(self.rows)


def environment():
    """Hardware and software fingerprint reported with every run (CPU-only claim, reproducibility)."""
    import sklearn, scipy
    info = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'n_cpu': os.cpu_count()}
    from importlib import metadata
    for mod, dist in (('lightgbm', 'lightgbm'), ('xgboost', 'xgboost'), ('shap', 'shap'),
                      ('dp_accounting', 'dp-accounting'), ('matplotlib', 'matplotlib')):
        try:
            info[mod] = metadata.version(dist)                         # installed distribution version
        except metadata.PackageNotFoundError:                          # optional dependency missing
            info[mod] = None
    try:                                                               # CPU model and RAM from /proc (Linux)
        cpu = [l for l in open('/proc/cpuinfo') if l.startswith('model name')]
        info['cpu_model'] = cpu[0].split(':', 1)[1].strip() if cpu else None
        mem = [l for l in open('/proc/meminfo') if l.startswith('MemTotal')]
        info['ram_gb'] = round(int(mem[0].split()[1]) / 1024 ** 2, 2) if mem else None
    except OSError:
        pass
    info['gpu'] = 'none (CPU-only by design)'
    return info


def outdir(mode, experiment):
    """runs/<mode>/<experiment>/, created on demand."""
    from .config import RUNS
    p = Path(RUNS) / mode / experiment
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    def default(o):                                                    # numpy scalars / arrays -> python
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, default=default)


def append_rows(rows, path):
    """Append result rows (list of dicts) to a CSV with a schema-tolerant union of columns (rows from different
    rungs carry different split-info keys). The table is written to a temporary file and moved over the old one
    with os.replace (atomic on POSIX), so a crash mid-write leaves the previous complete table in place."""
    if not rows:
        return
    import os
    df = pd.DataFrame(rows)
    if Path(path).exists():
        df = pd.concat([pd.read_csv(path, low_memory=False), df], ignore_index=True, sort=False)
    tmp = Path(str(path) + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

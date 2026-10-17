"""data_nids.py - NSL-KDD and UNSW-NB15 loaders (network ports of the C1 audit and the C2 shift rung).

NSL-KDD: KDDTrain+ (125,973) and KDDTest+ (22,544) from the defcom17/NSL_KDD mirror; counts equal the official
release (Tavallaee et al., CISDA 2009). KDDTest+ contains attack types absent from training, so the official
partition is a distribution-shift rung, never evidence about modern traffic (S3 section 3.1).
UNSW-NB15: official training (175,341) and testing (82,332) partitions (Moustafa & Slay) from a GitHub mirror.
Categorical columns are one-hot encoded inside each split by `encode_split` (fit on training rows only).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from .config import PATHS

NSL_COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty']
NSL_CAT = ['protocol_type', 'service', 'flag']
NSL_GROUPS = {'basic': NSL_COLS[0:9], 'content': NSL_COLS[9:22], 'time_traffic': NSL_COLS[22:31],
              'host_traffic': NSL_COLS[31:41]}                                    # KDD'99 feature families
NSL_ATTACK_CAT = {
    **{a: 'DoS' for a in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'mailbomb',
                          'processtable', 'udpstorm']},
    **{a: 'Probe' for a in ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']},
    **{a: 'R2L' for a in ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient',
                          'warezmaster', 'named', 'sendmail', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop',
                          'worm']},
    **{a: 'U2R' for a in ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel', 'ps', 'sqlattack',
                          'xterm']},
    'normal': 'Normal'}

UNSW_CAT = ['proto', 'service', 'state']
UNSW_GROUPS = {                                                                    # Moustafa & Slay feature families
    'flow': ['proto'],
    'basic': ['state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
              'spkts', 'dpkts', 'rate'],
    'content': ['swin', 'dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth', 'response_body_len'],
    'time': ['sjit', 'djit', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat'],
    'additional': ['is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
                   'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                   'ct_dst_src_ltm']}


def load_nsl():
    """Return (frame, meta): frame = 41 raw features; meta = y, attack, family (DoS/Probe/R2L/U2R), partition."""
    parts = []
    for part, key in (('train', 'nsl_train'), ('test', 'nsl_test')):
        d = pd.read_csv(PATHS[key], header=None, names=NSL_COLS)
        d['partition'] = part
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    meta = pd.DataFrame({'y': (d['label'] != 'normal').astype(int), 'attack': d['label'],
                         'family': d['label'].map(NSL_ATTACK_CAT).fillna('Unknown'),
                         'partition': d['partition'], 'row_id': np.arange(len(d))})
    return d[NSL_COLS[:41]], meta


def load_unsw():
    parts = []
    for part, key in (('train', 'unsw_train'), ('test', 'unsw_test')):
        d = pd.read_csv(PATHS[key], encoding='utf-8-sig')                        # file starts with a BOM
        d['partition'] = part
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    feats = [c for c in d.columns if c not in ('id', 'attack_cat', 'label', 'partition')]
    meta = pd.DataFrame({'y': d['label'].astype(int), 'attack': d['attack_cat'].str.strip(),
                         'family': d['attack_cat'].str.strip(), 'partition': d['partition'],
                         'row_id': np.arange(len(d))})
    return d[feats], meta


def encode_split(F_train, F_test, cat_cols, min_frequency=None):
    """One-hot categoricals fitted on training rows only; returns float arrays + feature names."""
    num = [c for c in F_train.columns if c not in cat_cols]
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=min_frequency)
    enc.fit(F_train[cat_cols].astype(str))
    tr = np.hstack([F_train[num].to_numpy(np.float64), enc.transform(F_train[cat_cols].astype(str))])
    te = np.hstack([F_test[num].to_numpy(np.float64), enc.transform(F_test[cat_cols].astype(str))])
    names = num + list(enc.get_feature_names_out(cat_cols))
    return tr, te, names


def group_columns(names, groups):
    """Map engineered column names (incl. one-hot expansions) back to their feature family."""
    out = {g: [] for g in groups}
    for i, n in enumerate(names):
        for g, cols in groups.items():
            if any(n == c or n.startswith(c + '_') for c in cols):
                out[g].append(i)
                break
    return out

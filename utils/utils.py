# utils.py
import numpy as np

EPS = 1e-12

def total_variation(p_counts, q_counts):
    p = np.array(p_counts, dtype=float); q = np.array(q_counts, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return None
    p /= p.sum(); q /= q.sum()
    return 0.5 * np.sum(np.abs(p - q))

def kl_divergence(p_counts, q_counts):
    p = np.array(p_counts, dtype=float); q = np.array(q_counts, dtype=float)
    if p.sum() == 0:
        return None
    p /= p.sum(); q /= q.sum()
    q = np.clip(q, EPS, None); p = np.clip(p, EPS, None)
    return float(np.sum(p * np.log(p / q)))

def bhattacharyya_fidelity(p_counts, q_counts):
    p = np.array(p_counts, dtype=float); q = np.array(q_counts, dtype=float)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(np.sqrt(p * q)))
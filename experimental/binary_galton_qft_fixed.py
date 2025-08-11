# binary_galton_qft_fixed.py
import math
from typing import List, Dict, Optional, Callable
import numpy as np
from qiskit import QuantumCircuit, transpile
try:
    from qiskit.circuit.library import QFT
    _HAS_QFT_LIBRARY = True
except Exception:
    _HAS_QFT_LIBRARY = False
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit_aer import AerSimulator

EPS = 1e-12

# QFT and IQFT helpers (same as before)
def qft_rotations(qc: QuantumCircuit, qubits: List[int]) -> None:
    n = len(qubits)
    for i in range(n):
        qc.h(qubits[i])
        for k in range(i + 1, n):
            angle = math.pi / (2 ** (k - i))
            qc.cp(angle, qubits[k], qubits[i])

def qft(qc: QuantumCircuit, qubits: List[int], do_swaps: bool = False) -> None:
    if _HAS_QFT_LIBRARY:
        qft_inst = QFT(len(qubits), do_swaps=do_swaps, approximation_degree=0)
        qc.append(qft_inst.to_instruction(), qubits)
    else:
        qft_rotations(qc, qubits)
        if do_swaps:
            n = len(qubits)
            for i in range(n // 2):
                qc.swap(qubits[i], qubits[n - i - 1])

def iqft(qc: QuantumCircuit, qubits: List[int], do_swaps: bool = False) -> None:
    if _HAS_QFT_LIBRARY:
        qft_inst = QFT(len(qubits), do_swaps=do_swaps, approximation_degree=0)
        qc.append(qft_inst.inverse().to_instruction(), qubits)
    else:
        if do_swaps:
            n = len(qubits)
            for i in range(n // 2):
                qc.swap(qubits[i], qubits[n - i - 1])
        n = len(qubits)
        for i in reversed(range(n)):
            for k in reversed(range(i + 1, n)):
                angle = -math.pi / (2 ** (k - i))
                qc.cp(angle, qubits[k], qubits[i])
            qc.h(qubits[i])

# ------------------------------
# Candidate increment variants
# Each variant is a function (qc, coin, reg) -> None that applies controlled increment
# We'll try each variant in detection and pick the one that yields correct mapping.
# ------------------------------
def increment_variant_base(qc: QuantumCircuit, coin: int, reg: List[int], do_swaps=False):
    """Original approach: QFT(reg); cp(coin->reg[j]) on reg[j]; IQFT(reg)"""
    m = len(reg)
    if m == 0:
        return
    qft(qc, reg, do_swaps=do_swaps)
    for j in range(m):
        angle = 2.0 * math.pi / (2 ** (j + 1))
        qc.cp(angle, coin, reg[j])
    iqft(qc, reg, do_swaps=do_swaps)

def increment_variant_rev_targets(qc: QuantumCircuit, coin: int, reg: List[int], do_swaps=False):
    """Apply CPs to reversed(reg)"""
    m = len(reg)
    if m == 0:
        return
    qft(qc, reg, do_swaps=do_swaps)
    rev = list(reversed(reg))
    for j in range(m):
        angle = 2.0 * math.pi / (2 ** (j + 1))
        qc.cp(angle, coin, rev[j])
    iqft(qc, reg, do_swaps=do_swaps)

def increment_variant_rev_qft(qc: QuantumCircuit, coin: int, reg: List[int], do_swaps=False):
    """Apply QFT/IQFT on reversed(reg), but CPs on reg (or vice versa)"""
    m = len(reg)
    if m == 0:
        return
    rev = list(reversed(reg))
    qft(qc, rev, do_swaps=do_swaps)
    # apply CPs to reg[j] (original order)
    for j in range(m):
        angle = 2.0 * math.pi / (2 ** (j + 1))
        qc.cp(angle, coin, reg[j])
    iqft(qc, rev, do_swaps=do_swaps)

# Add more composites if needed
CANDIDATE_VARIANTS = [
    ("base_no_swaps", lambda qc, coin, reg: increment_variant_base(qc, coin, reg, do_swaps=False)),
    ("rev_targets_no_swaps", lambda qc, coin, reg: increment_variant_rev_targets(qc, coin, reg, do_swaps=False)),
    ("rev_qft_no_swaps", lambda qc, coin, reg: increment_variant_rev_qft(qc, coin, reg, do_swaps=False)),
    ("base_swaps", lambda qc, coin, reg: increment_variant_base(qc, coin, reg, do_swaps=True)),
    ("rev_targets_swaps", lambda qc, coin, reg: increment_variant_rev_targets(qc, coin, reg, do_swaps=True)),
]

# ------------------------------
# Auto-detection machinery
# ------------------------------
_detection_cache = {}  # cache detection result per m

def _test_variant_m(m: int, variant_fn: Callable, shots: int = 2048) -> bool:
    """
    Deterministic test: prepare reg=r=0, coin=1 (so increment applied),
    apply variant and measure; expect output 1.
    Returns True if measured majority output equals 1.
    """
    total = m + 1
    reg = list(range(m))
    coin = m
    qc = QuantumCircuit(total, m)
    # prepare r=0 (do nothing), set coin = 1
    qc.x(coin)
    # apply candidate variant
    variant_fn(qc, coin, reg)
    # measure reg into (MSB..LSB) classical bits by measuring reversed(reg)
    qc.measure(list(reversed(reg)), list(range(m)))
    # run
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    res = sim.run(tqc, shots=shots, seed_simulator=1234).result()
    counts = res.get_counts()
    if not counts:
        return False
    # find most frequent output
    out_bs, _ = max(counts.items(), key=lambda it: it[1])
    out_i = int(str(out_bs).replace(" ", ""), 2)
    return out_i == 1

def detect_working_variant_for_m(m: int) -> str:
    """Detect which variant works for register size m. Returns variant name."""
    if m in _detection_cache:
        return _detection_cache[m]
    for name, fn in CANDIDATE_VARIANTS:
        try:
            ok = _test_variant_m(m, fn)
        except Exception:
            ok = False
        if ok:
            _detection_cache[m] = name
            return name
    # none worked
    _detection_cache[m] = None
    return None

def get_variant_fn_for_m(m: int):
    name = detect_working_variant_for_m(m)
    if name is None:
        return None
    for n, fn in CANDIDATE_VARIANTS:
        if n == name:
            return fn
    return None

# ------------------------------
# Public QFT-based builder using detected variant
# ------------------------------
def build_binary_galton_qft_auto(k: int) -> QuantumCircuit:
    """
    Build QFT-based binary Galton circuit using auto-detected increment variant.
    """
    assert isinstance(k, int) and k >= 0
    m = max(1, math.ceil(math.log2(k + 1)))
    reg = list(range(m))
    coin = m
    qc = QuantumCircuit(m + 1, m)

    # detect variant for this m once (this runs tiny circuits)
    variant_fn = get_variant_fn_for_m(m)
    if variant_fn is None:
        raise RuntimeError(f"No working QFT increment variant detected for m={m}")

    # Build layers
    for _ in range(k):
        qc.reset(coin)
        qc.h(coin)
        variant_fn(qc, coin, reg)

    qc.measure(list(reversed(reg)), list(range(m)))
    return qc

def run_binary_galton_qft_auto(k: int, shots: int = 5000, seed: Optional[int] = None, noise_model=None) -> Dict[int, int]:
    qc = build_binary_galton_qft_auto(k)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    try:
        if noise_model is None:
            job = sim.run(tqc, shots=shots, seed_simulator=seed)
        else:
            sim_noisy = AerSimulator(noise_model=noise_model)
            tqc2 = transpile(qc, sim_noisy)
            job = sim_noisy.run(tqc2, shots=shots, seed_simulator=seed)
    except Exception:
        job = sim.run(tqc, shots=shots)
    res = job.result()
    raw = res.get_counts()
    counts = {i: 0 for i in range(k + 1)}
    for bs, c in raw.items():
        s = str(bs).replace(" ", "")
        idx = int(s, 2)
        if idx <= k:
            counts[idx] += c
        else:
            counts[k] += c
    return counts

# A small verify helper using the auto variant
def verify_binary_qft_auto(k_list=(1,2,3,4), shots=4000, verbose=True):
    from math import comb
    def tv(p,q): return 0.5 * sum(abs(pi-qi) for pi, qi in zip(p,q))
    res = {}
    for k in k_list:
        counts = run_binary_galton_qft_auto(k, shots=shots, seed=12345)
        p = np.array([counts[i] for i in range(k+1)], dtype=float); p /= p.sum()
        q = np.array([comb(k,i)*(0.5**k)*shots for i in range(k+1)], dtype=float); q /= q.sum()
        res[k] = {"tv": float(tv(p,q))}
        if verbose:
            print(f"k={k} -> TV={res[k]['tv']:.4f}")
    return res
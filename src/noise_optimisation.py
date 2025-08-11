# noise_optimisation.py
import math
import numpy as np
from typing import Dict, Optional, Tuple
from qiskit import transpile
from qiskit import QuantumCircuit
# Aer simulator import may differ by qiskit versions; try robustly
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error
except Exception:
    # fallback names used by some versions
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error

EPS = 1e-12

# -------------------------
# Noise model builder
# -------------------------
def build_noise_model(gate_error_1q: float = 0.001,
                      gate_error_2q: float = 0.01,
                      readout_error: float = 0.02,
                      t1: float = 50e-6,
                      t2: float = 70e-6,
                      gate_time_1q: float = 50e-9,
                      gate_time_2q: float = 200e-9) -> NoiseModel:
    """
    Construct a simple NoiseModel with depolarizing errors on single- and
    two-qubit gates and a readout error on measurements. Also injects a
    thermal relaxation error model for single- and two-qubit gates.
    This is *parametric* and useful as a first approximation.
    """
    nm = NoiseModel()

    # depolarizing (1q + 2q)
    err1 = depolarizing_error(gate_error_1q, 1)
    err2 = depolarizing_error(gate_error_2q, 2)

    # thermal relaxation (approx): wrap as quantum errors on 1q and 2q
    # Build single-qubit thermal error if thermal_relaxation_error is available
    try:
        therm1 = thermal_relaxation_error(t1, t2, gate_time_1q)
        therm2 = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(thermal_relaxation_error(t1, t2, gate_time_2q))
        # combine depolarizing + thermal (compose)
        err1_full = err1.compose(therm1)
        err2_full = err2.compose(therm2)
    except Exception:
        # if thermal helpers not available, fall back to depolarizing only
        err1_full = err1
        err2_full = err2

    # Add to noise model for typical gates (cx, u3/ry/rz/h/x)
    one_qubit_gates = ['u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'sx']
    two_qubit_gates = ['cx', 'cz', 'swap', 'iswap', 'rzz']  # removed duplicate 'cx'

    # ensure uniqueness (defensive)
    one_qubit_gates = list(dict.fromkeys(one_qubit_gates))
    two_qubit_gates = list(dict.fromkeys(two_qubit_gates))

    for g in one_qubit_gates:
        nm.add_all_qubit_quantum_error(err1_full, g)
    for g in two_qubit_gates:
        nm.add_all_qubit_quantum_error(err2_full, g)


    # Readout error (bit-flip biased)
    ro_matrix = [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    nm.add_all_qubit_readout_error(ReadoutError(ro_matrix))

    return nm

# -------------------------
# Simulation runners
# -------------------------
def simulate_qc(qc: QuantumCircuit, shots: int = 5000, seed: Optional[int] = None,
                noise_model: Optional[NoiseModel] = None, backend_name: Optional[str] = None) -> Dict[str, int]:
    """
    Simulate a circuit using AerSimulator. If noise_model is provided, try to
    run with noise. The function attempts a few API styles for compatibility.
    Returns counts dict (bitstring -> counts).
    """
    # choose simulator
    sim = AerSimulator()
    # transpile for simulator
    tqc = transpile(qc, sim)
    # Try different run APIs depending on Aer version
    if noise_model is not None:
        try:
            # Preferred: construct simulator with noise model
            sim_noisy = AerSimulator(noise_model=noise_model)
            tqc2 = transpile(qc, sim_noisy)
            job = sim_noisy.run(tqc2, shots=shots, seed_simulator=seed)
            res = job.result()
            return res.get_counts()
        except Exception:
            try:
                # Alternative: pass noise_model in run call
                job = sim.run(tqc, shots=shots, noise_model=noise_model, seed_simulator=seed)
                res = job.result()
                return res.get_counts()
            except Exception:
                # Worst-case: fall back to noiseless sim and warn
                print("Warning: couldn't apply noise_model to Aer API; returning noiseless results.")
                job = sim.run(tqc, shots=shots, seed_simulator=seed)
                res = job.result()
                return res.get_counts()
    else:
        # No noise case
        job = sim.run(tqc, shots=shots, seed_simulator=seed)
        res = job.result()
        return res.get_counts()

# -------------------------
# Transpile / optimization helper
# -------------------------
def optimize_circuit(qc: QuantumCircuit, basis_gates: Optional[list] = None, optimization_level: int = 3,
                     backend_sim: Optional[AerSimulator] = None) -> QuantumCircuit:
    """
    Transpile/optimize the circuit for AerSimulator; returns optimized circuit.
    If backend_sim provided, transpile for that backend.
    """
    if backend_sim is None:
        backend_sim = AerSimulator()
    opt_qc = transpile(qc, backend_sim, optimization_level=optimization_level, basis_gates=basis_gates)
    return opt_qc

# -------------------------
# Metrics
# -------------------------
def counts_to_prob_vector(counts, n_bins: int) -> np.ndarray:
    """
    Convert counts into a probability vector length n_bins.

    Accepts:
      - counts: dict where keys are bitstrings (e.g. '010') -> counts
      - OR keys are ints (0..n_bins-1) -> counts

    Returns normalized probability vector (sums to 1) of length n_bins.
    """
    vec = np.zeros(n_bins, dtype=float)

    # If keys are integers:
    any_int_key = any(isinstance(k, (int, np.integer)) for k in counts.keys())
    if any_int_key:
        for k, c in counts.items():
            if 0 <= int(k) < n_bins:
                vec[int(k)] += c
    else:
        # assume bitstring keys (MSB..LSB)
        for bs, c in counts.items():
            s = str(bs).replace(' ', '')
            try:
                idx = int(s, 2)
            except Exception:
                # If bitstring contains non-binary chars, try stripping or skip
                continue
            if 0 <= idx < n_bins:
                vec[idx] += c

    if vec.sum() > 0:
        return vec / vec.sum()
    return vec

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.sum(np.abs(p - q))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum(p * np.log(p / q)))

def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)))

def compare_metrics(counts, target_counts, n_bins):
    """
    counts: dict (int->counts) or (bitstring->counts) returned by simulator
    target_counts: either list/array of expected counts (length n_bins) OR dict int->counts
    n_bins: length
    Returns dict {tv, kl, fidelity}
    """
    p = counts_to_prob_vector(counts, n_bins)

    # Normalize target in flexible ways
    if isinstance(target_counts, dict):
        q_raw = np.array([target_counts.get(i, 0) for i in range(n_bins)], dtype=float)
    else:
        q_raw = np.array(list(target_counts), dtype=float)

    if q_raw.sum() > 0:
        q = q_raw / q_raw.sum()
    else:
        q = q_raw

    # TV
    tv = 0.5 * np.sum(np.abs(p - q))
    # KL (p||q) with epsilon
    p_safe = np.clip(p, EPS, None)
    q_safe = np.clip(q, EPS, None)
    kl = float(np.sum(p_safe * np.log(p_safe / q_safe)))
    # Bhattacharyya fidelity-like
    fidelity = float(np.sum(np.sqrt(p * q)))

    return {"tv": float(tv), "kl": float(kl), "fidelity": float(fidelity)}

# test_core_galton.py
import math
import numpy as np
import pytest

from core_galton import (
    GaltonEngine,
    build_galton_circuit,
    simulate_engine,
    run_galton
)

# --------- Helpers ---------
def analytic_expected(k, shots):
    """Analytic expected counts list for Binomial(k, 1/2)."""
    return [math.comb(k, i) * (0.5 ** k) * shots for i in range(k + 1)]


def total_variation(obs_counts, exp_counts):
    """Total variation distance between two count vectors."""
    obs = np.array(obs_counts, dtype=float)
    exp = np.array(exp_counts, dtype=float)
    if obs.sum() == 0 or exp.sum() == 0:
        return 1.0
    obs /= obs.sum()
    exp /= exp.sum()
    return 0.5 * np.sum(np.abs(obs - exp))


def no_duplicate_qubit_ops(qc):
    """Return True if no operation in qc has duplicate qubit args."""
    for instr, qargs, cargs in qc.data:
        # qargs are Qubit objects; test their indices
        idxs = [q.index for q in qargs]
        if len(set(idxs)) != len(idxs):
            return False
    return True


# --------- Tests ---------

def test_circuit_sizes():
    """Circuits for k=1,2 should have expected qubit / classical bit counts
    for the 'spaced' peg layout (total_qubits = 2*k + 2, clbits = k+1)."""
    for k in (1, 2, 4):
        qc = build_galton_circuit(k, mode='full')
        # spaced layout uses total_qubits = 2*k + 2 (control + 2k+1 pegs)
        assert qc.num_qubits == 2 * k + 2
        assert qc.num_clbits == k + 1


def test_simulation_counts_sum():
    """Simulate a circuit and ensure the returned raw counts sum to shots."""
    qc = build_galton_circuit(3, mode='full')
    shots = 600
    raw = simulate_engine(qc, shots=shots, seed=42)
    # simulate_engine returns a dict: bitstring -> counts
    assert sum(raw.values()) == shots


def test_run_reproducible_with_seed():
    """Using the same seed yields identical bin_counts from GaltonEngine.run."""
    eng = GaltonEngine(layers=4, mode='full')
    shots = 1000
    seed = 123456
    c1 = eng.run(shots=shots, seed=seed)
    c2 = eng.run(shots=shots, seed=seed)
    assert c1 == c2
    # different seed usually gives different draws (not strictly required)
    c3 = eng.run(shots=shots, seed=seed + 1)
    assert isinstance(c3, dict) and sum(c3.values()) == shots


def test_binomial_tv_small_k():
    """For small k the sample should be close to analytic in TV distance.
    This is a probabilistic test — thresholds are intentionally generous.
    """
    k = 2
    shots = 2000
    eng = GaltonEngine(layers=k, mode='full')
    # deterministic seed for CI stability
    seed = 20240901
    sample = eng.run(shots=shots, seed=seed)
    obs = [sample.get(i, 0) for i in range(k + 1)]
    exp = analytic_expected(k, shots)
    tv = total_variation(obs, exp)
    # Threshold: TV <= 0.12 (empirically generous for 2k)
    assert tv <= 0.12, f"TV too large: {tv:.4f}; obs={obs}, exp_approx={[int(round(e)) for e in exp]}"


def test_binomial_tv_medium_k():
    """For medium k check TV distance threshold (more liberal)."""
    k = 4
    shots = 4000
    eng = GaltonEngine(layers=k, mode='full')
    seed = 314159
    sample = eng.run(shots=shots, seed=seed)
    obs = [sample.get(i, 0) for i in range(k + 1)]
    exp = analytic_expected(k, shots)
    tv = total_variation(obs, exp)
    # TV threshold a bit larger for moderate k and finite shots
    assert tv <= 0.15, f"TV too large: {tv:.4f}"


def test_no_duplicate_qubit_arguments_in_circuit():
    """Ensure generated circuits don't include duplicate qubit arguments in any op."""
    for k in (1, 2, 4, 8):
        qc = build_galton_circuit(k, mode='full')
        assert no_duplicate_qubit_ops(qc), f"Found duplicate qubit args in QC for k={k}"
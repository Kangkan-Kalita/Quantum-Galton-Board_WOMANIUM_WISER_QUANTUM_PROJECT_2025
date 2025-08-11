# src/binary_galton_qft.py
import math
from typing import Dict, Optional, List
import numpy as np
from qiskit import QuantumCircuit, transpile
try:
    # preferred: use QFT from qiskit circuit library if available
    from qiskit.circuit.library import QFT
    _HAS_QFT_LIBRARY = True
except Exception:
    _HAS_QFT_LIBRARY = False
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit_aer import AerSimulator

EPS = 1e-12

# ---------------------------
# QFT helpers (fallback if QFT not available)
# ---------------------------
def qft_rotations(qc: QuantumCircuit, qubits: List[int]) -> None:
    """Apply QFT rotations on qubits list (qubits[0] is the first target).
    Uses the common textbook pattern:
        for i in range(n):
            H(i)
            for k in range(i+1, n):
                CP(pi / 2^(k-i), control=qubits[k], target=qubits[i])
    """
    import math
    n = len(qubits)
    for i in range(n):
        qc.h(qubits[i])
        for k in range(i + 1, n):
            angle = math.pi / (2 ** (k - i))
            qc.cp(angle, qubits[k], qubits[i])


def qft(qc: QuantumCircuit, qubits: List[int], do_swaps: bool = False) -> None:
    """Apply QFT on the provided qubits (in-place)."""
    if _HAS_QFT_LIBRARY:
        # use library QFT (do_swaps controls final swap layer)
        qft_inst = QFT(len(qubits), do_swaps=do_swaps, approximation_degree=0)
        qc.append(qft_inst.to_instruction(), qubits)
    else:
        qft_rotations(qc, qubits)
        if do_swaps:
            n = len(qubits)
            for i in range(n // 2):
                qc.swap(qubits[i], qubits[n - i - 1])


def iqft(qc: QuantumCircuit, qubits: List[int], do_swaps: bool = False) -> None:
    """Inverse QFT on qubits."""
    if _HAS_QFT_LIBRARY:
        qft_inst = QFT(len(qubits), do_swaps=do_swaps, approximation_degree=0)
        qc.append(qft_inst.inverse().to_instruction(), qubits)
    else:
        if do_swaps:
            n = len(qubits)
            for i in range(n // 2):
                qc.swap(qubits[i], qubits[n - i - 1])
        # inverse of qft_rotations: reverse order, conjugate angles
        n = len(qubits)
        for i in reversed(range(n)):
            for k in reversed(range(i + 1, n)):
                angle = -math.pi / (2 ** (k - i))
                qc.cp(angle, qubits[k], qubits[i])
            qc.h(qubits[i])


# ---------------------------
# Controlled increment via QFT
# ---------------------------
def _controlled_increment_qft(qc: QuantumCircuit, coin: int, reg: List[int]) -> None:
    """
    Coherently add 1 to 'reg' (binary register) controlled on 'coin' using QFT trick.
    reg: list of qubit indices for the binary register; we treat reg[0] as LSB in the
         arithmetic convention inside this routine (we'll apply QFT on this order).
    The implementation:
      QFT(reg)
      For j=0..m-1: apply controlled-phase with angle 2*pi / 2^{j+1} from coin -> reg[j]
      IQFT(reg)
    """
    import math
    m = len(reg)
    if m == 0:
        return
    # Apply QFT on the register (LSB..MSB ordering)
    qft(qc, reg, do_swaps=False)
    rev = list(reversed(reg))
    # Controlled phase rotations: angle = 2*pi / 2^{j+1}
    # We apply control from 'coin' to target rev[j] for j=0..m-1
    for j in range(m):
        angle = 2.0 * math.pi / (2 ** (j + 1))
        qc.cp(angle, coin, rev[j])

    # inverse QFT
    iqft(qc, reg, do_swaps=False)


# ---------------------------
# Build the full sequential binary Galton circuit (QFT-based increment)
# ---------------------------
def build_binary_galton_qft_circuit(k: int) -> QuantumCircuit:
    """
    Build a coherent binary Galton engine using QFT-based controlled increments.
    - k: number of layers
    Returns QuantumCircuit with:
      - m = ceil(log2(k+1)) qubits for binary register (LSB at reg[0])
      - 1 coin qubit (index = m)
      - no ancillas required
      - measures the register into m classical bits (we measure reg in MSB..LSB order)
    Notes on measurement ordering:
      - We measure reg in reversed order when creating classical bits so the measured
        bitstring maps directly with int(bitstring, 2) == register value.
    """
    assert isinstance(k, int) and k >= 0
    m = max(1, math.ceil(math.log2(k + 1)))  # at least 1 qubit
    reg = list(range(0, m))  # we will treat reg[0] as LSB
    coin = m
    total_qubits = m + 1
    qc = QuantumCircuit(total_qubits, m)

    # For each layer: reset coin, coin flip (H), controlled increment via QFT
    for _ in range(k):
        qc.reset(coin)
        qc.h(coin)
        _controlled_increment_qft(qc, coin, reg)

    # Measurement: measure reg qubits into classical bits.
    # We'll measure reg in reversed order (MSB -> classical highest index) so that
    # the resulting bitstring from Qiskit (MSB..LSB) corresponds to int(bitstring,2).
    qc.measure(list(reversed(reg)), list(range(m)))
    return qc


# ---------------------------
# Runner wrapper
# ---------------------------
def run_binary_galton_qft(k: int, shots: int = 5000, seed: Optional[int] = None, noise_model=None) -> Dict[int, int]:
    """
    Build and run the QFT-based binary Galton circuit, returning dict of bin->counts.
    """
    qc = build_binary_galton_qft_circuit(k)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    try:
        if noise_model is None:
            job = sim.run(tqc, shots=shots, seed_simulator=seed)
        else:
            try:
                sim_noisy = AerSimulator(noise_model=noise_model)
                tqc2 = transpile(qc, sim_noisy)
                job = sim_noisy.run(tqc2, shots=shots, seed_simulator=seed)
            except Exception:
                job = sim.run(tqc, shots=shots, noise_model=noise_model, seed_simulator=seed)
    except TypeError:
        job = sim.run(tqc, shots=shots)
    result = job.result()
    raw = result.get_counts()

    # raw keys are bitstrings (MSB..LSB) because we measured reversed(reg)
    counts = {i: 0 for i in range(k + 1)}
    for bs, c in raw.items():
        s = str(bs).replace(" ", "")
        try:
            idx = int(s, 2)
        except Exception:
            continue
        if idx <= k:
            counts[idx] += c
        else:
            # clamp improbable values to last bin
            counts[k] += c
    return counts


# ---------------------------
# Simple verify helper (small k tests)
# ---------------------------
def analytic_binomial_counts(k: int, shots: int):
    from math import comb
    return [comb(k, i) * (0.5 ** k) * shots for i in range(k + 1)]


def verify_binary_qft(k_list=(1, 2, 3, 4), shots=5000, noise_model=None, verbose=True):
    """
    Run small-k verification tests and print TV/KL/fidelity vs analytic.
    """
    from math import comb
    from collections import defaultdict
    # local copies of metric helpers to avoid extra imports
    def tv(p, q): return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))
    def kl(p, q):
        p_arr = np.clip(np.array(p), EPS, None)
        q_arr = np.clip(np.array(q), EPS, None)
        return float(np.sum(p_arr * np.log(p_arr / q_arr)))
    def bhat(p, q): return float(sum((pi * qi) ** 0.5 for pi, qi in zip(p, q)))

    results = {}
    for k in k_list:
        counts = run_binary_galton_qft(k, shots=shots, seed=12345, noise_model=noise_model)
        p = np.array([counts[i] for i in range(k + 1)], dtype=float)
        p /= (p.sum() + EPS)
        q = np.array(analytic_binomial_counts(k, shots), dtype=float)
        q /= (q.sum() + EPS)
        results[k] = {
            "tv": float(tv(p, q)),
            "kl": float(kl(p, q)),
            "fidelity": float(bhat(p, q))
        }
        if verbose:
            print(f"k={k} -> TV={results[k]['tv']:.4f}, KL={results[k]['kl']:.4f}, fidelity={results[k]['fidelity']:.6f}")
    return results
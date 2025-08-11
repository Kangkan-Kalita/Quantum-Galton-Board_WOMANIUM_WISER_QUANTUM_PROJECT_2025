# quantum_galton_board.py
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit.providers.fake_provider import GenericBackendV2


def build_fast_galton_circuit(k: int) -> QuantumCircuit:
    """
    Build the Galton board circuit:
      - Use k coin qubits (Hadamard on each) and measure them.
      - The bin index is the number of '1's measured (0..k).
    Returns: QuantumCircuit (k qubits, k classical bits)
    """
    assert k >= 0
    qc = QuantumCircuit(k, k)
    for i in range(k):
        qc.h(i)
    qc.measure(list(range(k)), list(range(k)))
    return qc


def simulate_counts_fast(k: int, shots: int = 5000, seed: Optional[int] = None, noise_model=None) -> Dict[int, int]:
    """
    Simulate the circuit and return counts binned by number of ones (0..k).
    Use AerSimulator; if noise_model provided, use it.
    """
    qc = build_fast_galton_circuit(k)
    try:
        sim = AerSimulator()
    except Exception:
        from qiskit.providers.fake_provider import GenericBackendV2
        sim = AerSimulator()
    tqc = transpile(qc, sim)
    if noise_model is None:
        job = sim.run(tqc, shots=shots, seed_simulator=seed)
    else:
        sim_n = AerSimulator(noise_model=noise_model)
        tqc2 = transpile(qc, sim_n)
        job = sim_n.run(tqc2, shots=shots, seed_simulator=seed)
    res = job.result()
    raw = res.get_counts()
    # Map bitstrings (MSB..LSB) to counts per bin index = popcount
    bin_counts = {i: 0 for i in range(k + 1)}
    for bs, c in raw.items():
        s = str(bs).replace(" ", "")
        ones = s.count('1')
        bin_counts[ones] += c
    return bin_counts


def analytic_binomial(k: int) -> List[float]:
    """Return analytic probabilities for Binomial(k, 0.5) as list length k+1."""
    probs = [math.comb(k, i) * (0.5 ** k) for i in range(k + 1)]
    return probs


def plot_binomial_counts(bin_counts: Dict[int, int], k: int, shots: int, show_expected=True, figsize=(7, 4)):
    """Plot histogram of bin_counts; optionally overlay analytic curve (scaled to shots)."""
    bins = list(range(0, k + 1))
    counts = [bin_counts.get(i, 0) for i in bins]
    plt.figure(figsize=figsize)
    plt.bar(bins, counts, width=0.6, label='Observed', edgecolor='k')
    plt.xticks(bins)
    plt.xlabel('Bin index (number of 1s)')
    plt.ylabel('Counts')
    plt.title(f'Quantum Galton Board (k={k}, shots={shots})')
    if show_expected:
        probs = analytic_binomial(k)
        expected = [p * shots for p in probs]
        plt.plot(bins, expected, marker='o', linestyle='--', label='Analytic (scaled)')
        # Also overlay expected as thin bars
        plt.bar(bins, expected, width=0.25, alpha=0.4, label='Expected (scaled)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def run_many_ks(ks: List[int], shots: int = 5000, seed: Optional[int] = None) -> Dict[int, Dict[int, int]]:
    """
    Run Q-Galton for multiple k values and return dict k -> bin_counts.
    Use different random seeds per k if seed is None (so you get fresh randomness).
    """
    results = {}
    for k in ks:
        # use fresh seed for each run to see different samples if seed is None
        s = None if seed is None else (seed + k)
        bc = simulate_counts_fast(k, shots=shots, seed=s)
        results[k] = bc
    return results
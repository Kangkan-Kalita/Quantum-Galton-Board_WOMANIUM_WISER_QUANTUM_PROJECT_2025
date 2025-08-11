# exponential_sampler.py
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def build_exponential_circuit(n_bins: int, lam: float):
    """
    Build a circuit that prepares an exponential distribution over n_bins
    using state initialization on m=ceil(log2(n_bins)) qubits.
    Returns QuantumCircuit (measured).
    """
    m = math.ceil(math.log2(n_bins))
    N = 2 ** m
    # target probabilities for first n_bins indices
    probs = np.array([math.exp(-lam * i) for i in range(n_bins)], dtype=float)
    probs /= probs.sum()
    # create amplitudes vector length N
    amps = np.zeros(N, dtype=complex)
    amps[:n_bins] = np.sqrt(probs)
    # normalize again for numerical stability
    amps /= np.linalg.norm(amps)
    qc = QuantumCircuit(m, m)
    # Qiskit initialize expects statevector of length 2^m
    qc.initialize(amps, qc.qubits)
    qc.measure(range(m), range(m))
    return qc

def run_exponential(n_bins, lam, shots=5000, seed=None):
    qc = build_exponential_circuit(n_bins, lam)
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed) if seed is not None else sim.run(transpile(qc, sim), shots=shots)
    raw = job.result().get_counts()
    # Convert measured bitstrings to integers and aggregate only 0..n_bins-1
    counts = {i: 0 for i in range(n_bins)}
    for bs, c in raw.items():
        idx = int(bs.replace(' ', ''), 2)
        if idx < n_bins:
            counts[idx] += c
    return counts

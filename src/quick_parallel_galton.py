# quick_parallel_galton.py
from qiskit import QuantumCircuit, transpile
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit_aer import AerSimulator
from typing import Dict, Optional
import numpy as np

def build_parallel_coin_circuit(k: int) -> QuantumCircuit:
    """
    Build a circuit with k coin qubits, each H, measured.
    Returns a QuantumCircuit with k qubits and k classical bits.
    """
    qc = QuantumCircuit(k, k)
    for i in range(k):
        qc.h(i)
    qc.measure(range(k), range(k))
    return qc

def run_parallel_galton(k: int, shots: int = 5000, seed: Optional[int] = None, noise_model=None) -> Dict[int, int]:
    qc = build_parallel_coin_circuit(k)
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
    res = job.result()
    raw = res.get_counts()

    # Map bitstrings to integer counts (bitstring MSB..LSB)
    counts = {i: 0 for i in range(k + 1)}
    for bs, c in raw.items():
        s = bs.replace(' ', '')
        # int(s,2) is the integer representing the bitstring (MSB..LSB).
        idx = int(s, 2)
        # idx can range 0..2^k-1; but number of ones is bin index.
        ones = bin(idx).count("1")
        if ones <= k:
            counts[ones] += c
    return counts
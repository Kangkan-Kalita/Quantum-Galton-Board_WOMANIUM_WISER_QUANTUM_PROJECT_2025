# adamard_walk.py
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

def build_hadamard_walk_circuit(steps: int):
    """
    Build a discrete-time Hadamard quantum walk with `steps` steps using
    the same spaced-position layout.
    Returns a measured QuantumCircuit.
    """
    n = steps
    total_qubits = 2 * n + 2
    qc = QuantumCircuit(total_qubits, n + 1)

    coin = 0
    centre = n + 1   # odd index
    # init walker at centre
    qc.x(centre)

    for _ in range(n):
        # coin toss
        qc.h(coin)
        # conditional shift: sweep controlled-swaps across positions
        # We will apply cswap(coin, middle, right) then cswap(coin, middle, left)
        # but avoid duplicating indices — this pattern moves amplitude conditionally
        for j in range(_ + 1):  # for layer i do i+1 swaps? This follows Galton layering
            middle = centre - _ + 2 * j
            left = middle - 1
            right = middle + 1
            # shift right conditional on coin
            if 0 <= right < total_qubits:
                qc.cswap(coin, middle, right)
            # shift left conditional on coin
            if 0 <= left < total_qubits:
                qc.cswap(coin, middle, left)
        # Note: we deliberately do not reset the coin or add extra cx entangling steps
    # measure position wires
    qc.measure([2 * k + 1 for k in range(n + 1)], list(range(n + 1)))
    return qc

def run_hadamard_walk(steps, shots=5000, seed=None):
    qc = build_hadamard_walk_circuit(steps)
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed) if seed is not None else sim.run(transpile(qc, sim), shots=shots)
    raw = job.result().get_counts()
    # convert raw bitstrings -> bins (0..steps)
    counts = {i: 0 for i in range(steps + 1)}
    for bs, c in raw.items():
        s = bs.replace(' ','')[-(steps+1):].zfill(steps+1)
        idx = s[::-1].find('1')
        if idx >= 0 and idx <= steps:
            counts[idx] += c
    return counts
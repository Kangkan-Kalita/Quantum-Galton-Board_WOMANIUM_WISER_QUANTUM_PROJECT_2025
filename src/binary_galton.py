# src/binary_galton.py
import math
import numpy as np
from typing import Dict, Optional, List
from qiskit import QuantumCircuit, transpile
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit.providers.aer import AerSimulator

EPS = 1e-12

def _controlled_increment(qc: QuantumCircuit, coin: int, reg: List[int], ancillas: List[int]) -> None:
    """
    Controlled increment by 1: add `coin` to binary register `reg` (LSB = reg[0]).
    Implemented as ripple-carry using ancillas to store carries:
      - compute carries with CCX (Toffoli)
      - flip bits with CX using carries
      - uncompute carries
    ancillas must have length at least len(reg)-1 (or empty if reg length 1).
    """
    m = len(reg)
    if m == 0:
        return
    if m == 1:
        # Single-bit: just CNOT from coin -> reg[0]
        qc.cx(coin, reg[0])
        return

    # compute carries: c0 = coin & a0, c1 = c0 & a1, ...
    # ancillas[i] holds carry for position i (i from 0 to m-2)
    # compute first carry
    qc.ccx(coin, reg[0], ancillas[0])
    for i in range(1, m - 1):
        qc.ccx(ancillas[i - 1], reg[i], ancillas[i])

    # apply result flips: a0 ^= coin; a1 ^= c0; a2 ^= c1; ...
    qc.cx(coin, reg[0])
    for i in range(1, m):
        qc.cx(ancillas[i - 1], reg[i])

    # uncompute carries (reverse order)
    for i in reversed(range(1, m - 1)):
        qc.ccx(ancillas[i - 1], reg[i], ancillas[i])
    qc.ccx(coin, reg[0], ancillas[0])


def build_binary_galton_circuit(k: int) -> QuantumCircuit:
    """
    Build a binary-encoded Galton board circuit with k layers.
    - Binary register size m = ceil(log2(k+1)) to encode values 0..k.
    - One coin qubit.
    - Ancillas: m-1 Toffoli ancillas for ripple-carry.
    Circuit measures the binary register (m classical bits) and returns counts.
    """
    assert k >= 0 and isinstance(k, int)
    # number of bins = k+1 -> need m qubits
    m = math.ceil(math.log2(k + 1)) if k >= 1 else 1
    reg_qubits = list(range(0, m))            # position register (LSB at index 0)
    coin = m                                  # coin qubit index
    anc_count = max(0, m - 1)
    ancillas = list(range(m + 1, m + 1 + anc_count))
    total_qubits = m + 1 + anc_count

    qc = QuantumCircuit(total_qubits, m)

    # Initialize register to 0 (already |0>)
    # Loop over layers: reset coin, coin flip (H), conditional increment
    for layer in range(k):
        qc.reset(coin)
        qc.h(coin)
        if anc_count > 0:
            _controlled_increment(qc, coin, reg_qubits, ancillas)
        else:
            # m == 1: just single CNOT
            qc.cx(coin, reg_qubits[0])

    # Measure register bits into classical bits.
    # We will measure reg_qubits in *reversed* order so that the resulting
    # bitstring from Qiskit (MSB..LSB) maps naturally to integer via int(bs,2)
    qc.measure(list(reversed(reg_qubits)), list(range(m)))
    return qc


def run_binary_galton(k: int, shots: int = 5000, seed: Optional[int] = None,
                      noise_model=None) -> Dict[int, int]:
    """
    Build & run binary Galton circuit and return counts mapped to integer bin indices.
    If noise_model provided, it will be used by AerSimulator (if supported).
    """
    qc = build_binary_galton_circuit(k)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    # Run with or without noise_model depending on Aer API available
    try:
        if noise_model is None:
            job = sim.run(tqc, shots=shots, seed_simulator=seed)
        else:
            # some Aer versions accept noise_model in constructor or run call
            try:
                sim_noisy = AerSimulator(noise_model=noise_model)
                tqc2 = transpile(qc, sim_noisy)
                job = sim_noisy.run(tqc2, shots=shots, seed_simulator=seed)
            except Exception:
                job = sim.run(tqc, shots=shots, noise_model=noise_model, seed_simulator=seed)
    except TypeError:
        # Fallback: simple run without seed/noise
        job = sim.run(tqc, shots=shots)
    result = job.result()
    raw = result.get_counts()

    # Map bitstrings to integer indices. Qiskit returns bitstrings MSB..LSB
    counts = {i: 0 for i in range(k + 1)}
    for bs, c in raw.items():
        s = bs.replace(' ', '')
        # bs is MSB..LSB (left->right). We mapped reg_qubits reversed when measuring
        # so int(s,2) gives the integer index directly.
        idx = int(s, 2)
        if idx <= k:
            counts[idx] += c
        else:
            # If for some reason index > k (should not happen), clamp or ignore
            # We clamp to nearest valid bin to be conservative:
            counts[k] += c
    return counts
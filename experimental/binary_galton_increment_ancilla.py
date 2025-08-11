# src/binary_galton_increment_ancilla.py
from typing import Dict, List, Optional
import math
from qiskit import QuantumCircuit, transpile
try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit.providers.fake_provider import GenericBackendV2


EPS = 1e-12

def controlled_increment_with_ancillas(qc: QuantumCircuit, coin: int, reg: List[int], ancillas: List[int]):
    """
    Correct ancilla-chain controlled increment:
      if coin == 1 then reg := reg + 1 (LSB at reg[0]).
    Requirements:
      - ancillas length >= m-1
      - reg is a list of indices LSB-first
      - coin is index of control qubit
    Algorithm:
      1) Compute prefix ancillas (anc[i] = coin & reg[0] & ... & reg[i])
         - anc[0] = coin & reg[0]
         - anc[1] = anc[0] & reg[1], ...
      2) Flip reg[0] with cx(coin, reg[0])
      3) For i=1..m-1 flip reg[i] with cx(anc[i-1], reg[i])
      4) Uncompute ancillas in reverse order
    """
    m = len(reg)
    if m == 0:
        return
    if (m - 1) > len(ancillas):
        raise ValueError("Need at least m-1 ancillas for m register bits.")

    # 1) Compute ancilla prefix chain using original reg bits
    for i in range(1, m):
        if i == 1:
            # anc0 = coin & reg[0]
            qc.ccx(coin, reg[0], ancillas[0])
        else:
            # anc[i-1] = anc[i-2] & reg[i-1]
            qc.ccx(ancillas[i-2], reg[i-1], ancillas[i-1])

    # 2) Flip LSB (uses coin only)
    qc.cx(coin, reg[0])

    # 3) Use ancillas to flip higher bits
    for i in range(1, m):
        qc.cx(ancillas[i-1], reg[i])

    # 4) Uncompute ancillas in reverse order
    for i in reversed(range(1, m)):
        if i == 1:
            qc.ccx(coin, reg[0], ancillas[0])
        else:
            qc.ccx(ancillas[i-2], reg[i-1], ancillas[i-1])

    return

def build_binary_galton_increment_circuit(k: int) -> QuantumCircuit:
    """
    Build a binary Galton engine of k layers using ancilla-chain controlled increment.
    Uses layout:
      qubits: [reg0 ... reg(m-1), coin, anc0 ... anc(m-2)]
    Measure reg in classical bits (MSB..LSB)
    """
    assert k >= 0
    m = max(1, math.ceil(math.log2(k + 1)))
    # register indices: 0..m-1 (LSB at 0)
    reg = list(range(0, m))
    coin = m
    ancillas = list(range(m + 1, m + 1 + max(0, m - 1)))  # enough ancillas
    total = m + 1 + max(0, m - 1)
    qc = QuantumCircuit(total, m)

    # run k layers
    for _ in range(k):
        qc.reset(coin)
        qc.h(coin)
        controlled_increment_with_ancillas(qc, coin, reg, ancillas)

    # measure register into classical bits (MSB..LSB)
    qc.measure(list(reversed(reg)), list(range(m)))
    return qc

def run_binary_galton_increment(k: int, shots: int = 5000, seed: Optional[int] = None, noise_model=None) -> Dict[int, int]:
    """
    Build and simulate the above circuit, returning bin counts (0..k).
    """
    qc = build_binary_galton_increment_circuit(k)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    try:
        if noise_model is None:
            job = sim.run(tqc, shots=shots, seed_simulator=seed)
        else:
            sim_noisy = AerSimulator(noise_model=noise_model)
            tqc2 = transpile(qc, sim_noisy)
            job = sim_noisy.run(tqc2, shots=shots, seed_simulator=seed)
    except TypeError:
        job = sim.run(tqc, shots=shots)  # fallback
    res = job.result()
    raw = res.get_counts()
    # map bitstrings to integer indices (Qiskit returns MSB..LSB)
    counts = {i: 0 for i in range(k + 1)}
    for bs, c in raw.items():
        s = str(bs).replace(" ", "")
        idx = int(s, 2)
        if idx <= k:
            counts[idx] += c
        else:
            counts[k] += c
    return counts

# Deterministic mapping test (exhaustive for given m)
def mapping_test_increment(max_m: int = 5):
    """
    Perform mapping tests for m=1..max_m, returning dict m->list of failures.
    """
    from qiskit import transpile
    try:
        from qiskit_aer import AerSimulator
    except Exception:
        from qiskit.providers.fake_provider import GenericBackendV2
    sim = AerSimulator()
    failures = {}
    for m in range(1, max_m + 1):
        fails = []
        # build reg/anc layout consistent with build function
        reg = list(range(0, m))
        coin = m
        ancillas = list(range(m + 1, m + 1 + max(0, m - 1)))
        total = m + 1 + max(0, m - 1)
        for r in range(2 ** m):
            qc = QuantumCircuit(total, m)
            # prepare r in reg (LSB-first)
            for i in range(m):
                if (r >> i) & 1:
                    qc.x(reg[i])
            qc.x(coin)  # coin=1 to apply increment
            controlled_increment_with_ancillas(qc, coin, reg, ancillas)
            qc.measure(list(reversed(reg)), list(range(m)))
            tqc = transpile(qc, sim)
            res = sim.run(tqc, shots=2048, seed_simulator=1234).result()
            counts = res.get_counts()
            out_bs, _ = max(counts.items(), key=lambda kv: kv[1])
            out_i = int(str(out_bs).replace(" ", ""), 2)
            expected = (r + 1) % (2 ** m)
            if out_i != expected:
                fails.append((m, r, out_i, expected, counts))
        failures[m] = fails
    return failures
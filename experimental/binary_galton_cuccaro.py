# binary_galton_cuccaro.py
from typing import Dict, List
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# CDKM ripple-carry adder class (Cuccaro-style) in recent Qiskit
try:
    from qiskit.circuit.library import CDKMRippleCarryAdder
    CDKM_AVAILABLE = True
except Exception:
    CDKM_AVAILABLE = False


def _make_regs_for_increment(m: int):
    """
    Create (carry_in, A_reg, B_reg, out_zero) registers layout:
    we'll use the CDKM 'full' layout: c0 | A[m] | B[m] | z(1)
    """
    c0 = QuantumRegister(1, name="c0")
    a = QuantumRegister(m, name="A")   # input to be incremented
    b = QuantumRegister(m, name="B")   # will hold A+coin after add
    z = QuantumRegister(1, name="Z")   # extra target bit (unused)
    creg = ClassicalRegister(m, name="out")
    return c0, a, b, z, creg


def controlled_increment_via_cdkm(qc: QuantumCircuit,
                                  coin_qubit: int,
                                  a_qubits: List[int],
                                  b_qubits: List[int],
                                  c0_qubit: int):
    """
    On the provided circuit qc, perform: if coin==1 then B <- A + B + 1 (with carry-in)
    We assume qc already has the right registers placed and that:
      - coin_qubit is an index in qc.qubits for the coin control
      - a_qubits is list of qubit indices for A (LSB-first)
      - b_qubits is list of qubit indices for B (LSB-first)
      - c0_qubit is the index for carry-in qubit (single qubit)
    Implementation:
      - copy coin -> c0 (CNOT)
      - append CDKM adder (full) acting on (c0, A, B, Z)
      - uncompute c0 (CNOT coin -> c0) to restore it to 0 (optional if you need reuse)
    """
    # copy coin into carry-in
    qc.cx(coin_qubit, c0_qubit)

    # Build CDKM adder and append
    if not CDKM_AVAILABLE:
        raise RuntimeError("CDKMRippleCarryAdder not available in Qiskit install.")
    m = len(a_qubits)
    adder = CDKMRippleCarryAdder(num_state_qubits=m, kind="full", name=f"cdkm_add_{m}")
    # gate expects ordering: [c0] + A + B + [z]
    # We'll need to identify a single z qubit; here we assume caller reserves one.
    # The adder returns a Gate; append directly with appropriate qubits.
    # Build ordered list of qubits for the adder:
    # NOTE: CDKM expects qubit objects (QuantumRegister elements) if constructed from regs.
    ordered = [qc.qubits[c0_qubit]] + [qc.qubits[i] for i in a_qubits] + [qc.qubits[i] for i in b_qubits] + [qc.qubits[c0_qubit]]  # placeholder Z - replace if you provide separate Z
    # The CDKM signature in some Qiskit versions needs a separate Z qubit.
    # To be robust, we will create a temporary small subcircuit using registers instead of indices when possible.
    # Instead of trying to append via raw indices (different Qiskit versions differ), we will let user call helper wrapper
    qc.append(adder, [qc.qubits[c0_qubit]] + [qc.qubits[i] for i in a_qubits] + [qc.qubits[i] for i in b_qubits] + [qc.qubits[c0_qubit]])

    # uncompute c0 so it's back to zero
    qc.cx(coin_qubit, c0_qubit)
    return qc


def demo_increment(m: int = 3, shots: int = 2000, seed: int = 42):
    """
    Demo usage:
      - A register of m qubits (holds number in binary, LSB-first)
      - B register of m qubits (init zero)
      - c0 carry-in (single qubit)
      - coin qubit (control)
    This will produce B <- A + coin, leaving A unchanged.
    """
    if not CDKM_AVAILABLE:
        raise RuntimeError("CDKM adder not available in this Qiskit install. "
                           "Install qiskit>=0.30 with CDKM or use the MAJ/UMA fallback (not included here).")

    # build registers
    c0, A, B, Z, out = _make_regs_for_increment(m)
    coin = QuantumRegister(1, name="coin")
    qc = QuantumCircuit(coin, c0, A, B, Z, out)

    # Example: prepare A = some small number (like decimal 3 -> binary 11)
    # LSB-first mapping: A[0] is least-significant bit
    init_val = 3
    for i in range(m):
        if (init_val >> i) & 1:
            qc.x(A[i])

    # Prepare B = 0 (already zero); coin = 1 (control)
    qc.x(coin[0])  # set coin=1 to test increment; comment this line to test coin=0

    # copy coin -> c0
    qc.cx(coin[0], c0[0])

    # append adder
    adder = CDKMRippleCarryAdder(num_state_qubits=m, kind="full")
    qc.append(adder, [c0[0]] + [A[i] for i in range(m)] + [B[i] for i in range(m)] + [Z[0]])

    # optionally uncompute coin->c0 (not necessary here, but do for cleanliness)
    qc.cx(coin[0], c0[0])

    # measure B (the result A+coin)
    qc.measure([B[i] for i in range(m)], list(range(m)))
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    job = sim.run(tqc, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    # convert bitstrings (Qiskit MSB-first) into integer indices, assuming measured qubits order matches
    out_counts = {}
    for bs, c in counts.items():
        # bs format like 'b_{m-1} ... b_0' — Qiskit returns MSB-first, so we reverse
        idx = int(bs.replace(" ", "")[::-1], 2)
        out_counts[idx] = out_counts.get(idx, 0) + c
    return out_counts
# core_galton.py
"""
Peg-DSL Galton engine (fixed: methods properly defined inside the class).
Provides:
 - GaltonEngine class with `.run()`, `.draw()`, `.plot()`
 - build_galton_circuit(layers, mode) -> QuantumCircuit (convenience)
 - simulate_engine(qc, shots, seed) -> raw_counts (convenience)
 - run_galton(layers, shots, mode, seed) -> bin_counts (convenience)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Optional noisy backend import (if available in your environment)
try:
    from qiskit.providers.fake_provider import GenericBackendV2
except Exception:
    GenericBackendV2 = None


@dataclass
class Peg:
    coin: int
    left: Optional[int]
    mid: int
    right: Optional[int]


class GaltonEngine:
    """Galton engine built from a peg-DSL (no Enums)."""

    def __init__(self, layers: int, mode: str = 'full'):
        assert isinstance(layers, int) and layers >= 0, "layers must be >= 0"
        assert mode in ('full', 'fast'), "mode must be 'full' or 'fast'"
        self.layers = layers
        self.mode = mode
        self._build_pegs()
        self._build_circuit()

    # ----------------------------
    # Peg layout builders & apply
    # ----------------------------
    def _build_pegs(self) -> None:
        P = self.layers + 1
        center = P // 2
        self.pegs: List[List[Peg]] = []
        for layer in range(self.layers):
            row: List[Peg] = []
            for j in range(layer + 1):
                mid = center - layer + 2 * j
                left = mid - 1 if (mid - 1) >= 0 else None
                right = mid + 1 if (mid + 1) <= (P - 1) else None
                row.append(Peg(coin=P, left=left, mid=mid, right=right))
            self.pegs.append(row)

    def _apply_peg(self, qc: QuantumCircuit, peg: Peg) -> None:
        coin = peg.coin
        mid = peg.mid
        # cswap to right if valid
        if peg.right is not None and peg.right != mid and peg.right != coin:
            qc.cswap(coin, mid, peg.right)
        # entangle
        qc.cx(mid, coin)
        # cswap to left if valid
        if peg.left is not None and peg.left != mid and peg.left != coin:
            qc.cswap(coin, mid, peg.left)

    def _apply_layer(self, qc: QuantumCircuit, row: List[Peg]) -> None:
        if not row:
            return
        coin = row[0].coin
        qc.reset(coin)
        qc.h(coin)
        for peg in row:
            self._apply_peg(qc, peg)

    # ----------------------------
    # Circuit builder
    # ----------------------------
    def _build_circuit(self) -> None:
        """
        Build circuit with spaced position wires (robust, reference-style).
        total_qubits = 2*layers + 2
        position wires at indices 1,3,5,...,2*layers+1 (there are layers+1 of them).
        control coin is qubit 0.
        """
        n = self.layers
        total_qubits = 2 * n + 2
        # number of position bins = n + 1
        self.qc = QuantumCircuit(total_qubits, n + 1)

        control = 0
        centre = n + 1         # this is the 'ball' starting qubit (odd index)
        # initialize ball at the centre peg
        self.qc.x(centre)

        # loop over layers (i = 0 .. n-1)
        for i in range(n):
            # reset and coin flip
            self.qc.reset(control)
            self.qc.h(control)

            # j runs from 0..i (i+1 pegs this layer)
            for j in range(i + 1):
                middle = centre - i + 2 * j    # lands on odd index
                left   = middle - 1            # even index
                right  = middle + 1            # even index

                # Controlled swaps & entangling CNOT (peg gadget)
                # cswap(control, middle, right)
                if 0 <= right < total_qubits and right != control and middle != control:
                  self.qc.cswap(control, middle, right)
                # cx(middle, control)
                self.qc.cx(middle, control)
                # cswap(control, middle, left)
                if 0 <= left < total_qubits and left != control and middle != control:
                  self.qc.cswap(control, middle, left)
                # optional inter-peg connection (as in many references)
                if j < i:
                  # middle+1 is the even index between pegs; ensure safe
                  inter = middle + 1
                  if 0 <= inter < total_qubits and inter != control:
                    self.qc.cx(inter, control)

        # measure position pegs (odd indices) into classical bits 0..n
        for k in range(n + 1):
          self.qc.measure(2 * k + 1, k)

    # ----------------------------
    # Runner / helpers
    # ----------------------------
    def run(self, shots: int = 5000, seed: Optional[int] = None, noise: bool = False) -> Dict[int, int]:
        """Simulate and return bin_counts: {bin_index: count}."""
        # choose backend
        if noise and GenericBackendV2 is not None:
            backend = GenericBackendV2(num_qubits=self.qc.num_qubits)
        else:
            backend = AerSimulator()

        tqc = transpile(self.qc, backend)
        # some Aer versions accept seed_simulator; we pass it if non-None
        if seed is None:
            job = backend.run(tqc, shots=shots)
        else:
            job = backend.run(tqc, shots=shots, seed_simulator=seed)

        raw = job.result().get_counts()
        # convert raw bitstrings (MSB->LSB) to LSB-first bin index
        P = self.layers + 1
        bins: Dict[int, int] = {i: 0 for i in range(P)}
        for bs, c in raw.items():
            s = bs.replace(' ', '')[-P:].zfill(P)
            idx = s[::-1].find('1')
            if idx >= 0 and idx < P:
                bins[idx] += c
        return bins

    def draw(self, **kwargs) -> str:
        """Return a QC drawing (text or mpl)."""
        return self.qc.draw(**kwargs)

    def plot(self, counts: Dict[int, int], *, figsize=(6, 4), title: Optional[str] = None) -> None:
        x = sorted(counts.keys())
        y = [counts[i] for i in x]
        plt.figure(figsize=figsize)
        plt.bar(x, y, edgecolor='k')
        plt.xticks(x)
        plt.xlabel('Bin index')
        plt.ylabel('Counts')
        plt.title(title or f'Galton k={self.layers}')
        plt.show()


# ----------------------------
# Convenience top-level wrappers (keeps old notebook code working)
# ----------------------------
def build_galton_circuit(layers: int, mode: str = 'full') -> QuantumCircuit:
    """Return a QuantumCircuit for the given layers and mode."""
    eng = GaltonEngine(layers=layers, mode=mode)
    return eng.qc


def simulate_engine(qc: QuantumCircuit, shots: int = 5000, seed: Optional[int] = None, noise: bool = False) -> Dict[str, int]:
    """Simulate the provided circuit and return raw counts (bitstrings->counts)."""
    # Use AerSimulator (no wrappers here) - returns raw counts
    backend = AerSimulator() if not (noise and GenericBackendV2 is not None) else GenericBackendV2(num_qubits=qc.num_qubits)
    tqc = transpile(qc, backend)
    if seed is None:
        job = backend.run(tqc, shots=shots)
    else:
        job = backend.run(tqc, shots=shots, seed_simulator=seed)
    return job.result().get_counts()


def run_galton(layers: int, shots: int = 5000, mode: str = 'full', seed: Optional[int] = None, noise: bool = False) -> Dict[int, int]:
    """Convenience: build engine, run it, return bin_counts."""
    eng = GaltonEngine(layers=layers, mode=mode)
    return eng.run(shots=shots, seed=seed, noise=noise)

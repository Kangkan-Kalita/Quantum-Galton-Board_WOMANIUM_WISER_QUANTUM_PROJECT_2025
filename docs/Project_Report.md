# **Building a Noise-Aware Quantum Galton Board in Qiskit — A NISQ-Era Quantum Computing Project**
---

**Author : Kangkan Kalita**

**Affiliation : Gauhati University**, *Graduated*

**Country : India**

**Email :** *kalitakangkan.239@gmail.com*

**Date :** `2025-08-10`

---

**Abstract**

This report documents the design, implementation, and validation of a Quantum Galton Board prototype developed during the Womanium & WISER Global Quantum Program 2025. The project produces a modular k-layer Galton engine, compares one-hot and binary encodings (QFT and Cuccaro/ripple), evaluates circuits under noiseless and noisy simulators, and verifies outputs using statistical metrics (Total Variation, KL divergence, fidelity). Code and notebooks are available at : [Github Repo](https://github.com/Kangkan-Kalita/Quantum-Galton-Board_WOMANIUM_WISER_QUANTUM_PROJECT_2025)

---

**1. Introduction**

The classical Galton board (bean machine) produces binomial / Gaussian-like distributions by routing balls across rows of pegs. Its quantum analogue is a compact, illustrative platform for quantum sampling, walks, and Monte Carlo. This project builds a reusable circuit generator for arbitrary layer count , implements different encodings for the walker, and evaluates performance under realistic noise models and transpiler optimizations.

**2. Background**

One-hot (CSWAP) representation: position wires encode the walker location directly; pegs are implemented as controlled-swap gadgets. This is straightforward to measure and map to bins.

Binary representations: encode position in a compact binary register and implement coherent incrementers. QFT-based incrementers are compact but fragile cross-SDK; Cuccaro (MAJ/UMA) ripple adders are more robust for demonstrations.

Verification: we compare sampled distributions to analytic targets using Total Variation (TV), KL divergence, and a fidelity-like overlap metric.


**3. Design & methods**

The k-layer generator builds a circuit with 2·k+2 qubits for the full one-hot CSWAP model (control coin + positions) and measures the position wires to produce k+1 output bins.

Binary encodings are implemented as alternate engines: a QFT-based controlled increment (experimental) and a Cuccaro ripple-adder fallback.

Simulators: Qiskit AerSimulator for noiseless runs; fake backends / noise models (GenericBackendV2 or custom noise parameters) for noisy runs.

Metrics: each run produces sampled counts which are mapped into bin indices. From these we compute TV, KL, and fidelity values for comparison against analytic binomial or target distributions.


**4. Implementation notes**

Main files (high-level):

src/core_galton.py — k-layer engine and helper functions.

src/binary_galton.py & src/binary_galton_qft.py — binary encodings (Cuccaro & QFT variants).

src/noise_optimisation.py — noise-model construction, transpiler options, metrics computation.

notebooks/2_k_layer_demo.ipynb — demo for k=1,2,4,8 and verification cells.

requirements.txt — pinned dependencies (recommended environment).


Reproducibility: notebooks include the seeds, shot counts, and SDK versions used in experiments. Use the requirements.txt provided in the repo root to reproduce runs.


**5. Experiments performed**

Noiseless verification for k = 1, 2, 4, 8 to confirm analytic agreement (one-hot generator).

Binary-encoding experiments (QFT and Cuccaro) with deterministic mapping tests (small registers) and sampling tests.

Noisy experiments using a simple NISQ noise model and multiple optimization levels via noise_optimisation.py.

Resource profiling (CNOT count, circuit depth) and metric collection for several optimization levels.


**6. Results (key outputs)**

Below are the representative outputs from the main experiments (values rounded where indicated).

k = 4 (noisy run)

TV = 0.3814, KL = 9.7598, Fidelity = 0.7851


Noise-optimization (imported noise_optimisation.py) — same noise model, multiple optimization levels

level 0 : TV = 0.3677, KL = 9.3863, Fidelity = 0.7942

level 1 : TV = 0.3677, KL = 9.3863, Fidelity = 0.7942

level 2 : TV = 0.3677, KL = 9.3863, Fidelity = 0.7942

level 3 : TV = 0.3677, KL = 9.3863, Fidelity = 0.7942


Binary Galton (results after importing & running binary_galton)

Noiseless : p sum = 1.0, q sum = 1.0

Noiseless metrics : TV = 0.7702, KL = 3.4806, Fidelity = 0.4043

Noisy metrics : TV = 0.6241, KL = 2.5270, Fidelity = 0.6531


Final summary (per optimisation level, rounded)

lvl 0 → TV = 0.3807, KL = 0.3948, Fidelity = 0.8887, CNOTs = 144, Depth = 219

lvl 1 → TV = 0.3810, KL = 0.3972, Fidelity = 0.8884, CNOTs = 144, Depth = 219

lvl 2 → TV = 0.3743, KL = 0.3873, Fidelity = 0.8920, CNOTs = 112, Depth = 206

lvl 3 → TV = 0.3743, KL = 0.3873, Fidelity = 0.8920, CNOTs = 112, Depth = 206


Short interpretation (summary):

The one-hot k-layer generator reproduces the expected distribution shape in noiseless simulations.

Binary encodings show varied behavior: the QFT-based increment was fragile in larger registers and across SDK differences; the Cuccaro/ripple adder gave more reliable behavior for demos.

Noise-model runs (k = 4) show degraded fidelity and higher TV as expected; applying transpiler optimizations reduced CNOTs and depth in some levels (lvl 2–3) and produced modest fidelity improvements.


**7. Discussion & limitations**

What worked well: modular generator for one-hot circuits; clear verification pipeline; reproducible notebooks; ability to compare encodings easily.

Limitations & failures: QFT increments required careful bit-order handling and were sensitive to SDK/runtime variants. Depth and two-qubit gate counts cause large fidelity losses under realistic noise; hardware runs require small-depth variants or partner sandbox credits.

Practical lessons: include deterministic mapping tests for any coherent arithmetic; use robust adders for reliable demos; include metric-based checks in CI for reproducibility.


**8. Future work**

Produce a hardware-ready k=4 demo (requires partner sandbox or small hardware credits).

Add automated CI tests (unit tests that verify deterministic mapping and run quick sampling checks).

Package a short teaching Colab version of the k=4 demo for outreach and reproducibility.


**9. Conclusion**

The project demonstrates a practical path from concept to a noise-aware, reproducible prototype for a quantum Galton board. For NISQ devices, careful encoding choices and depth/CX optimization are essential for preserving fidelity; robust arithmetic primitives (Cuccaro) are preferable for demonstrations across toolchains.

---

**Acknowledgements**

Thanks to the Womanium & WISER Global Quantum Program 2025 and partner organisations (QWorld, PennyLane, Xanadu, PsiQuantum, Fraunhofer) for the curated content and mentorship that made this work possible.

---

**References & resources**

Carney & Varcoe, *Universal Statistical Simulator* (2022). [arXiv:2202.01735](https://arxiv.org/pdf/2202.01735)

* Qiskit documentation : [https://qiskit.org](https://qiskit.org)

---

**Appendix** — quick reproducibility & run commands

1. Create and activate a virtual environment, then install dependencies:

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


2. Run the k=4 noiseless demo (example script / notebook):

Option A (script):
python -m src.run_demo --k 4 --shots 5000 --noisy False

Option B (notebook): Open notebooks/2_k_layer_demo.ipynb and run the cells for k = 1,2,4,8.


3. Noise & optimisation example (notebook / script):

In the notebook or script, import noise_optimisation.py and run the optimization levels (0..3) to reproduce the metrics in this report.

---

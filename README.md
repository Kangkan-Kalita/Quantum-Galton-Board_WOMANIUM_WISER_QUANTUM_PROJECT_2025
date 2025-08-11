# Quantum Galton Board — WOMANIUM & WISER QUANTUM PROGRAM 2025

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Notebooks](https://img.shields.io/badge/notebooks-4-orange.svg)](#notebooks)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#tests)

**Program :** **WOMANIUM & WISER QUANTUM PROGRAM 2025**

**Author :** **Kangkan Kalita**

**Email :** *kalitakangkan.239@gmail.com*

**Team Name :** **QBlitz**

**Enrollment ID :** `gst-LYB7DR8Qbmc59CD`

---

## Project summary :

This repository implements and analyses quantum Galton-board style circuits (a.k.a. Quantum Galton Board, QGB) with two complementary implementations:

- **Full one-hot Galton engine** (`src/core_galton.py`) — explicit CSWAP/coin design that tracks the “ball” position across `k+1` position qubits (one-hot encoding). Useful for correct modelling and small-k experiments.
- **Fast parallel-coin sampler** (`src/parallel_galton.py`) — Hadamard-based sampler useful for larger `k` when resource constraints make the full engine impractical.

We also implemented:
- **Alternate target samplers**: exponential sampler and Hadamard quantum walk (`src/quantum_walk.py`) with verification.
- **Noise modelling and optimisation** (`src/noise_optimisation.py`): parameterised noise models, transpiler-level comparison, and resource/metric reporting (TV, KL, fidelity).
- **Experimental** proofs-of-concept: QFT-based incrementers and Cuccaro ripple-adders for binary-encoded Galton engines (archived in `experimental/` due to instability across Qiskit versions).

Key deliverables: notebooks, scripts, plots, and a short report summarizing results and limitations.

---

## Repo layout

```

├── docs/ ← paper summary (2-page)
├── experimental/ ← exploratory notebooks and experimental encoders
│ ├── advanced_noise_optimisation.ipynb
│ ├── binary_galton_cuccaro.py
│ ├── binary_galton_increment_ancilla.py
│ ├── binary_galton_qft.py
│ ├── binary_galton_qft_fixed.py
│ └── notes_experimental.md
├── notebooks/ ← main reproducible notebooks/demos
│ ├── galton_demo.ipynb
│ ├── noise_optimisation_demo.ipynb
│ └── quantum_walk_demo.ipynb
├── results/ ← generated figures and CSV outputs
│ ├── cct_k1.png
│ ├── cct_k2.png
│ ├── cct_k4.png
│ ├── cct_k8.png
│ ├── cct_k16.png
│ ├── HW_plot_1.png
│ ├── HW_plot_2.png
│ ├── QGB_k1.png
│ ├── QGB_k2.png
│ ├── QGB_k4.png
│ ├── QGB_k16.png
│ └── noise_opt_plot_histo.png
├── src/ ← core modules (importable)
│ ├── binary_galton.py
│ ├── core_galton.py
│ ├── exponential_sampler.py
│ ├── hadamard_walk.py
│ ├── noise_optimisation.py
│ ├── quantum_galton_board
│ └── quick_parallel_galton.py
├── tests/ ← unit + distribution tests
│ ├── test_core_galton.py
│ └── test_distributions.py
├── utils/ ← helper utilities and lightweight scripts
│ └── utils.py
├── requirements.txt ← pinned packages for reproducibility
├── README.md ← this file
└── LICENSE ← MIT

````

---

## Quickstart (Colab)

If you want to run everything quickly in Google Colab, use the following first cell in a Colab notebook:

```bash
# Recommended minimal installs (Colab / fresh environment)
!pip install -q qiskit==0.46.1 qiskit-aer==0.10.4 numpy matplotlib plotly pytest

````

Then upload the repo to Colab (or mount Google Drive) and open the notebooks under `/notebooks/`.

> Colab notes: Colab’s Python environment may already have incompatible qiskit versions; after installing, restart the runtime to ensure imports load correctly.

---

## Quickstart (Local)

1. Clone the repo:

```bash
git clone https://github.com/<Kangkan-Kalita>/quantum-galton-board.git
cd quantum-galton-board
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows (PowerShell)
```

3. Install dependencies (pinned to tested versions):

```bash
pip install -r requirements.txt
# OR (if you want explicit single-line)
pip install qiskit==0.46.1 qiskit-aer==0.10.4 numpy matplotlib plotly pytest
```

4. Start Jupyter Lab / Notebook and open notebooks:

```bash
jupyter lab
# then open notebooks/galton_demo.ipynb, noise_optimisation_demo.ipynb, ...
```

---

## Usage examples

### Run the demo notebook

Open and run `notebooks/galton_demo.ipynb`:

* Builds circuits for k=1,2,4,8
* Simulates on `AerSimulator`
* Produces histograms and computes TV/KL/fidelity vs analytic binomial

### Run noise optimisation

Open `notebooks/noise_optimisation_demo.ipynb`:

* Builds simple noise model
* Runs circuits under different transpiler optimization levels
* Produces metrics vs optimization level and before/after histograms

### Run unit tests

From repo root:

```bash
pytest -q tests/test_core_galton.py
pytest -q tests/test_distributions.py::test_hadamard_walk_ballistic
```

> Note: Some tests are stochastic; tests call `run_*` helpers with fixed random seeds to reduce flakiness.

---

## Metrics & outputs

We compute the following metrics to quantify match with analytic targets :

* **Total Variation (TV) distance**
* **Kullback-Leibler (KL) divergence**
* **Bhattacharyya / classical fidelity** (sum of sqrt of probabilities)

Plots and metric summaries are saved in `assets/figures/`. See `project_report.pdf` for a two-page summary of selected results.

---

## Known issues & troubleshooting

* **Aer imports vary by version**: newer `qiskit-aer` exposes `qiskit_aer.AerSimulator`; older setups use `qiskit.providers.aer.AerSimulator`. The notebooks include import fallbacks — if you get import errors, install `qiskit-aer` matching the pinned version.
* **Bitstring ordering**: Qiskit classical bitstrings are MSB..LSB. Our helpers attempt to robustly translate to a bin index (LSB-first). If you changed measurement wire order, adjust `counts_to_bin_counts()` accordingly.
* **Deterministic outputs / no jitter**: If repeated runs give identical counts, check you didn’t pass a fixed simulator seed or use a deterministic backend. Remove `seed_simulator` to get fresh randomness (or pass different seeds).
* **Experimental folder**: QFT & ripple-adder experiments are archived; they may fail deterministically on different Qiskit versions because of decompositions and Incrementer availability. See `experimental/notes_experimental.md`.

---

## Reproducibility & CI

* Recommended CI: GitHub Actions workflow that installs pinned dependencies and runs:

  * `pytest -q`
  * Optionally run a headless script to regenerate figures.
* Add `requirements.txt` with pinned versions for reproducibility.

---

## References & resources

* Carney & Varcoe, *Universal Statistical Simulator* (2022). [arXiv:2202.01735](https://arxiv.org/pdf/2202.01735)
* Qiskit docs: [https://qiskit.org](https://qiskit.org)

---

## License

This project is released under the **MIT License** — see `LICENSE` for details.

---

## Acknowledgements

Thanks to the WISER & Womanium Quantum Program 2025 for sponsoring this project. Additional thanks to the Qiskit and PennyLane communities for tools and examples.

---

## Contact

If you find a bug or want to collaborate, please open an issue on the repository or contact : **kalitakangkan.239@gmail.com**.

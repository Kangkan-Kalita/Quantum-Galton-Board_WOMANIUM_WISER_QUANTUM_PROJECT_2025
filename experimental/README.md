## `experimental/README.md`


# experimental/

This folder contains experimental and exploratory code that extends the core Galton engine with binary encodings and alternate incrementer schemes.

**Contents**
- `advanced_noise_optimisation.ipynb` — notebook exploring aggressive transpiler/optimization and alternate noise parameters.
- `binary_galton_cuccaro.py` — Cuccaro-style MAJ/UMA adder based binary increment experiments.
- `binary_galton_increment_ancilla.py` — ancilla-assisted incrementer experimental implementation.
- `binary_galton_qft.py` — QFT-based binary incrementer attempt.
- `binary_galton_qft_fixed.py` — patched QFT variant with register ordering fixes.
- `notes_experimental.md` — diagnostics, failing-cases, version-specific notes, and recommended next steps.

**Important**
- Experimental scripts are *not* guaranteed to be stable across Qiskit/Aer versions. See `notes_experimental.md` for reproduction hints and pinned-version recommendations.
- Keep experimental work here separate from the main `src/` modules until validated and tested.

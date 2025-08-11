# notebooks/

This folder contains interactive notebooks used for development, demonstration and verification.

**Primary notebooks**
- `galton_demo.ipynb` — core Galton engine build & demo. Draws 1- and 2-layer circuits, builds general k-layer circuits, simulates, plots histograms and computes analytic comparisons.
- `noise_optimisation_demo.ipynb` — demonstrates the noise model pipeline: create noise model, run circuits at transpiler levels (0..3), plot TV/KL/fidelity and resource counts, and produce before/after histograms for the selected level.
- `quantum_walk_demo.ipynb` — implements and visualises the Hadamard quantum walk and an exponential-like sampler.

**Notes**
- Run notebooks top-to-bottom after installing dependencies.
- Use Colab for quick runs (see root README).

# src/

Core, importable Python modules for the project.

**Files**
- `binary_galton.py` — for better noise optimisation method 
- `core_galton.py` — main GaltonEngine / build_galton_circuit implementations (one-hot CSWAP model).
- `exponential_sampler.py` — exponential sampler functions.
- `hadamard_walk.py` — quantum/hadamard_walk functions.
- `noise_optimisation.py` — noise model builder and evaluation helpers.
- `quantum_galton_board.py` — effective for higher k values and prompt output
- `quick_parallel_galton.py` — FAST/quick parallel-coin sampler (Hadamard per position).

**Usage**
Import these modules in notebooks:
```python
from src.core_galton import GaltonEngine, build_galton_circuit
from src.noise_optimisation import build_simple_noise_model, evaluate_transpile_levels

## `tests/README.md`


# tests/

Unit and statistical tests used to validate correctness.

**Files**
- `test_core_galton.py` — tests for circuit shape, qubit counts and basic op counts.
- `test_distributions.py` — tests that verify distributions (analytical comparisons, variance checks for quantum walk).

**Run tests**
From repository root:
```bash
pip install -r requirements.txt
pytest -q

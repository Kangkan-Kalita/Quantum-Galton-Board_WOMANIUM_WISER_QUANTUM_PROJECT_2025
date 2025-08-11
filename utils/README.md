
## `utils/README.md`


# utils/

Lightweight utility functions used across notebooks and scripts.

**Files**
- `utils.py` — helpers for:
  - `counts_to_bin_counts`
  - `analytic_binomial_probs`
  - plotting wrappers for histograms, comparative plots
  - metric helpers (TV, KL, fidelity)

**Recommendation**
Import from `utils` in notebooks, and keep single source of truth to avoid bitstring-order mismatches.

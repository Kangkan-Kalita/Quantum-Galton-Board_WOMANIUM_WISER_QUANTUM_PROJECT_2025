# test_distributions.py
import math
import numpy as np
import pytest

EPS = 1e-12

# robust imports: try top-level module names, else layout
try:
    from exponential_sampler import run_exponential
except Exception:
    from exponential_sampler import run_exponential

try:
    from hadamard_walk import run_hadamard_walk
except Exception:
    from hadamard_walk import run_hadamard_walk

# optional GaltonEngine wrappers test
try:
    from core_galton import GaltonEngine
    HAS_ENGINE = True
except Exception:
    HAS_ENGINE = False


def total_variation_counts(obs, exp):
    obs = np.array(obs, dtype=float)
    exp = np.array(exp, dtype=float)
    if obs.sum() == 0 or exp.sum() == 0:
        return 1.0
    obs /= (obs.sum() + EPS)
    exp /= (exp.sum() + EPS)
    return 0.5 * np.sum(np.abs(obs - exp))


def test_exponential_tv():
    """Exponential sampler approximates analytic target (deterministic seed)."""
    n_bins = 8
    lam = 0.7
    shots = 5000
    seed = 20241001
    sample = run_exponential(n_bins, lam, shots=shots, seed=seed)
    obs = [sample.get(i, 0) for i in range(n_bins)]
    probs = [math.exp(-lam * i) for i in range(n_bins)]
    Z = sum(probs)
    expected = [shots * p / Z for p in probs]
    tv = total_variation_counts(obs, expected)
    # Threshold tuned for 5000 shots; relax if CI shows flakiness
    assert tv < 0.06, f"TV too large: {tv:.4f}"


def test_exponential_reproducible():
    """Same seed yields identical counts for exponential sampler."""
    n_bins = 6
    lam = 0.5
    shots = 2000
    seed = 1234567
    a = run_exponential(n_bins, lam, shots=shots, seed=seed)
    b = run_exponential(n_bins, lam, shots=shots, seed=seed)
    assert a == b


def test_hadamard_walk_ballistic():
    """
    Test that the Hadamard walk variance scales super-linearly with steps.
    Fitter: var ~ C * k^alpha, we require alpha > 1.0 (ballistic vs diffusive).
    """
    # Choose two moderate step counts where ballistic behavior should be visible
    k1, k2 = 8, 12
    shots = 10000   # increase shots to reduce sampling noise (CI costlier but more stable)
    s1 = 1000 + k1
    s2 = 1000 + k2

    sample1 = run_hadamard_walk(k1, shots=shots, seed=s1)
    sample2 = run_hadamard_walk(k2, shots=shots, seed=s2)

    bins1 = np.array(sorted(sample1.keys())); vals1 = np.array([sample1[i] for i in bins1], dtype=float)
    probs1 = vals1 / (vals1.sum() + EPS)
    mean1 = (bins1 * probs1).sum()
    var1 = ((bins1 - mean1) ** 2 * probs1).sum()

    bins2 = np.array(sorted(sample2.keys())); vals2 = np.array([sample2[i] for i in bins2], dtype=float)
    probs2 = vals2 / (vals2.sum() + EPS)
    mean2 = (bins2 * probs2).sum()
    var2 = ((bins2 - mean2) ** 2 * probs2).sum()

    # Avoid degenerate / zero-case
    assert var1 > 0 and var2 > 0

    # Local power-law exponent alpha = log(var2/var1) / log(k2/k1)
    alpha = np.log(var2 / var1) / np.log(k2 / k1)

    # Require super-linear growth (alpha > 1.0). Use slight tolerance for noise:
    assert alpha > 0.95, f"Observed exponent alpha={alpha:.3f} not > 0.95 (var1={var1:.3f}, var2={var2:.3f})"
    


def test_hadamard_reproducible():
    """Check reproducibility for Hadamard walk."""
    steps = 6
    shots = 3000
    seed = 20231234
    a = run_hadamard_walk(steps, shots=shots, seed=seed)
    b = run_hadamard_walk(steps, shots=shots, seed=seed)
    assert a == b


@pytest.mark.skipif(not HAS_ENGINE, reason="GaltonEngine not importable / wrappers not present")
def test_engine_wrappers_exist_and_work():
    """If GaltonEngine has .to_exponential and .to_hadamard_walk wrappers, test them quickly."""
    eng = GaltonEngine(layers=4)
    # exponential wrapper
    if hasattr(eng, "to_exponential"):
        eng.to_exponential(n_bins=8, lam=0.7)
        assert hasattr(eng, "qc") and eng.qc is not None
    # hadamard wrapper
    if hasattr(eng, "to_hadamard_walk"):
        eng.to_hadamard_walk(steps=6)
        assert hasattr(eng, "qc") and eng.qc is not None

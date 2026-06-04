import numpy as np
import pytest

from easygrad import numeric


def test_finite_difference_matches_analytic(scalar_case):
    f, fp, x = scalar_case
    approx = numeric.finite_difference(f, x, h=1e-6, scheme="central")
    assert approx == pytest.approx(fp(x), rel=1e-6, abs=1e-9)


def test_forward_and_backward_schemes_agree_roughly(scalar_case):
    f, fp, x = scalar_case
    fwd = numeric.finite_difference(f, x, h=1e-7, scheme="forward")
    bwd = numeric.finite_difference(f, x, h=1e-7, scheme="backward")
    assert fwd == pytest.approx(fp(x), rel=1e-4, abs=1e-5)
    assert bwd == pytest.approx(fp(x), rel=1e-4, abs=1e-5)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        numeric.finite_difference(lambda x: x, 1.0, 1e-6, scheme="nope")


# A numpy-based suite (the conftest functions use `math.*`, which reject complex
# inputs); these accept complex so the complex step can be applied to them.
COMPLEX_SUITE = [
    ("poly", lambda x: x**3 - 2 * x + 1, lambda x: 3 * x**2 - 2, 1.3),
    ("exp", lambda x: np.exp(0.5 * x), lambda x: 0.5 * np.exp(0.5 * x), 0.7),
    ("sin", lambda x: np.sin(x) * x, lambda x: np.sin(x) + x * np.cos(x), 1.1),
    ("tanh", lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2, 0.9),
    ("ratio", lambda x: 1.0 / (1.0 + x**2), lambda x: -2 * x / (1 + x**2) ** 2, 0.6),
    ("softplus", lambda x: np.log(1 + np.exp(x)), lambda x: 1 / (1 + np.exp(-x)), -0.4),
]


@pytest.mark.parametrize("f,fp,x", [c[1:] for c in COMPLEX_SUITE], ids=[c[0] for c in COMPLEX_SUITE])
def test_complex_step_matches_analytic(f, fp, x):
    # an absurdly small h still gives full-precision derivatives (no cancellation)
    approx = numeric.complex_step(f, x, h=1e-200)
    assert approx == pytest.approx(float(fp(x)), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("f,fp,x", [c[1:] for c in COMPLEX_SUITE], ids=[c[0] for c in COMPLEX_SUITE])
def test_complex_step_beats_finite_difference_at_tiny_h(f, fp, x):
    # at h = 1e-30 the central difference has fully cancelled, but the complex
    # step is still exact -- the whole point of the method, checked per function.
    h = 1e-30
    cs_err = abs(numeric.complex_step(f, x, h) - float(fp(x)))
    fd_err = abs(numeric.finite_difference(f, x, h, "central") - float(fp(x)))
    assert cs_err < 1e-9
    assert fd_err > cs_err


def test_error_curve_is_U_shaped():
    """Central-difference error must rise again for tiny h (cancellation wall)."""
    f = lambda x: np.exp(0.5 * x)
    fp = lambda x: 0.5 * np.exp(0.5 * x)
    hs = np.logspace(-1, -14, 40)
    err = numeric.error_curve(f, fp, 0.7, hs, scheme="central")
    # the minimum is interior, not at the smallest h -> a genuine U
    argmin = int(np.argmin(err))
    assert 0 < argmin < len(hs) - 1
    # smallest-h error is far worse than the best
    assert err[-1] > 100 * err[argmin]


def test_complex_step_has_no_cancellation_wall():
    """Complex-step error stays flat as h -> 0, unlike finite differences."""
    f = lambda x: np.exp(0.5 * x)
    fp = lambda x: 0.5 * np.exp(0.5 * x)
    hs = np.logspace(-1, -200, 60)
    cs_err = numeric.complex_step_error_curve(f, fp, 0.7, hs)
    fd_err = numeric.error_curve(f, fp, 0.7, np.logspace(-1, -14, 60), scheme="central")
    # complex step is accurate at the tiniest h ...
    assert cs_err[-1] < 1e-10
    # ... whereas finite differencing has blown up there
    assert fd_err[-1] > cs_err[-1]

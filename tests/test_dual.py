import math

import pytest

from easygrad import dual
from easygrad.dual import Dual


# functions written against easygrad.dual so they run on Dual or float
DUAL_SUITE = [
    (lambda x: x * x * x - 2 * x + 1, lambda x: 3 * x**2 - 2, 1.3),
    (lambda x: dual.exp(0.5 * x), lambda x: 0.5 * math.exp(0.5 * x), 0.7),
    (lambda x: dual.sin(x) * x, lambda x: math.sin(x) + x * math.cos(x), 1.1),
    (lambda x: 1.0 / (1.0 + x * x), lambda x: -2 * x / (1 + x**2) ** 2, 0.6),
    (lambda x: dual.log(1 + dual.exp(x)), lambda x: 1 / (1 + math.exp(-x)), -0.4),
    (lambda x: dual.tanh(x), lambda x: 1 - math.tanh(x) ** 2, 0.9),
    (lambda x: dual.sqrt(x), lambda x: 0.5 / math.sqrt(x), 2.0),
]


@pytest.mark.parametrize("f,fp,x", DUAL_SUITE)
def test_dual_matches_analytic(f, fp, x):
    assert dual.derivative(f, x) == pytest.approx(fp(x), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("f,fp,x", DUAL_SUITE)
def test_value_and_derivative(f, fp, x):
    val, der = dual.value_and_derivative(f, x)
    assert der == pytest.approx(fp(x), rel=1e-12, abs=1e-12)
    # primal equals plain evaluation
    assert val == pytest.approx(f(x), rel=1e-12, abs=1e-12)


def test_eps_squared_is_zero():
    # (1 + eps)*(1 + eps) should be 1 + 2 eps, not 1 + 2 eps + eps^2
    z = Dual(1.0, 1.0) * Dual(1.0, 1.0)
    assert z.primal == 1.0 and z.tangent == 2.0


def test_one_tangent_at_a_time():
    # derivative wrt x of x*y at (x=3,y=5) is y=5 (seed only x)
    f = lambda x: x * 5.0
    assert dual.derivative(f, 3.0) == pytest.approx(5.0)

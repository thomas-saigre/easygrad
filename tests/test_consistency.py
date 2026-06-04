import numpy as np
import pytest

from easygrad import forward, numeric, reverse
from easygrad.forward import jvp
from easygrad.reverse import grad


# functions available in both the forward and reverse namespaces
SHARED = [
    ("sum_sq", lambda mod, x: (x * x).sum(), lambda x: (x * x).sum()),
    ("exp_sum", lambda mod, x: mod.exp(x).sum(), lambda x: np.exp(x).sum()),
    ("tanh_sum", lambda mod, x: mod.tanh(x * 0.5).sum(), lambda x: np.tanh(x * 0.5).sum()),
]

POINTS = [np.array([0.2, -0.5, 1.3]), np.array([1.0, 2.0, -1.0])]


@pytest.mark.parametrize("name,fn,_np", SHARED, ids=[s[0] for s in SHARED])
@pytest.mark.parametrize("x", POINTS, ids=["pt0", "pt1"])
def test_reverse_equals_forward_full_gradient(name, fn, _np, x):
    # reverse mode: full gradient in one pass
    g_rev = grad(lambda z: fn(reverse, z))(x)

    # forward mode: assemble the gradient column by column (one JVP per basis e_i)
    n = x.size
    g_fwd = np.array([
        jvp(lambda z: fn(forward, z), [x], [np.eye(n)[i]])[1]
        for i in range(n)
    ])
    assert np.allclose(g_rev, g_fwd, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name,fn,f_np", SHARED, ids=[s[0] for s in SHARED])
@pytest.mark.parametrize("x", POINTS, ids=["pt0", "pt1"])
def test_reverse_equals_numeric(name, fn, f_np, x):
    g_rev = grad(lambda z: fn(reverse, z))(x)
    # numeric directional gradient via central differences along each axis
    n = x.size
    g_num = np.array([
        numeric.finite_difference(lambda t, i=i: f_np(x + t * np.eye(n)[i]), 0.0, 1e-6, "central")
        for i in range(n)
    ])
    assert np.allclose(g_rev, g_num, rtol=1e-5, atol=1e-7)

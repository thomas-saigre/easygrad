import numpy as np
import pytest

from easygrad import forward, reverse
from easygrad.forward import jvp
from easygrad.reverse import Node, grad


def test_jvp_matches_finite_difference():
    """JVP tangent == directional derivative (f(x+eps u) - f(x-eps u)) / 2eps."""
    def f(x):
        return forward.tanh(x * x).sum()

    x = np.array([0.3, -0.7, 1.2])
    u = np.array([1.0, 0.5, -2.0])
    primal, tangent = jvp(f, [x], [u])

    eps = 1e-6
    f_np = lambda z: np.tanh(z * z).sum()
    fd = (f_np(x + eps * u) - f_np(x - eps * u)) / (2 * eps)
    assert primal == pytest.approx(f_np(x))
    assert tangent == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_jvp_seeds_one_input_direction():
    # f(x,y) = x*y ; jvp with tangent (1,0) gives df along x = y
    def f(x, y):
        return x * y

    p, t = jvp(f, [np.array(3.0), np.array(5.0)], [np.array(1.0), np.array(0.0)])
    assert p == pytest.approx(15.0)
    assert t == pytest.approx(5.0)


def test_jvp_vjp_transpose_identity():
    """<v, J u> == <J^T v, u> for f: R^n -> R^m, f(x) = tanh(W x).

    J u is the JVP tangent (forward mode); J^T v is obtained by reverse mode as
    the gradient of x -> <v, f(x)>.
    """
    rng = np.random.default_rng(1)
    W = rng.standard_normal((4, 3))
    x = rng.standard_normal(3)
    u = rng.standard_normal(3)          # input tangent
    v = rng.standard_normal(4)          # output cotangent

    # forward: J u
    _, Ju = jvp(lambda z: forward.tanh(W @ z), [x], [u])

    # reverse: J^T v  (grad of <v, f(x)>)
    def vf(z):
        return (Node(v) * reverse.tanh(W @ z)).sum()

    JTv = grad(vf)(x)

    lhs = float(v @ Ju)
    rhs = float(JTv @ u)
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-12)

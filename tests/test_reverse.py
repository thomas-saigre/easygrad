import numpy as np
import pytest

from easygrad import reverse
from easygrad.reverse import Node, grad, value_and_grad, unbroadcast


def complex_step_grad(f_np, x, h=1e-200):
    """Numeric gradient of a scalar numpy function via per-coordinate complex step."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        z = x.astype(complex)
        z[idx] += 1j * h
        g[idx] = np.imag(f_np(z)) / h
    return g


# (reverse-mode f on Nodes, numpy f on complex arrays, x)
REVERSE_SUITE = [
    (lambda x: (x * x).sum(),
     lambda x: (x * x).sum(),
     np.array([1.0, -2.0, 3.0])),
    (lambda x: reverse.exp(x * 2.0).sum(),
     lambda x: np.exp(x * 2.0).sum(),
     np.array([0.1, 0.2, -0.3])),
    (lambda x: reverse.tanh(x).sum(),
     lambda x: np.tanh(x).sum(),
     np.array([0.5, -1.0, 2.0])),
    (lambda x: reverse.log((x * x + 1.0)).sum(),
     lambda x: np.log(x * x + 1.0).sum(),
     np.array([0.3, 1.5, -0.7])),
]


@pytest.mark.parametrize("f_node,f_np,x", REVERSE_SUITE)
def test_grad_passes_gradcheck(f_node, f_np, x):
    g = grad(f_node)(x)
    g_num = complex_step_grad(f_np, x)
    assert np.allclose(g, g_num, rtol=1e-8, atol=1e-10)
    assert g.shape == x.shape


def test_value_and_grad_returns_value():
    def f(x: np.ndarray) -> float:
        return (x * x).sum()
    val, g = value_and_grad(f)(np.array([1.0, 2.0, 3.0]))
    assert val == pytest.approx(14.0)
    assert np.allclose(g, [2.0, 4.0, 6.0])


def test_mlp_loss_multiarg_grad():
    """A 1-layer MLP squared-error loss, gradients wrt W and b via argnums."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 3))
    y = rng.standard_normal((4, 2))

    def loss(W, b):
        pred = reverse.tanh(Node(X) @ W + b)
        diff = pred - Node(y)
        return (diff * diff).mean()

    gW, gb = grad(loss, argnums=(0, 1))(np.zeros((3, 2)), np.zeros((2,)))
    assert gW.shape == (3, 2)
    assert gb.shape == (2,)

    # cross-check against complex step over the flattened parameters
    def loss_np(W, b):
        pred = np.tanh(X @ W + b)
        return ((pred - y) ** 2).mean()

    gW_num = complex_step_grad(lambda W: loss_np(W, np.zeros((2,), complex)), np.zeros((3, 2)))
    gb_num = complex_step_grad(lambda b: loss_np(np.zeros((3, 2), complex), b), np.zeros((2,)))
    assert np.allclose(gW, gW_num, atol=1e-9)
    assert np.allclose(gb, gb_num, atol=1e-9)


def test_unbroadcast_restores_shapes():
    # broadcast (3,) into (2,3): cotangent (2,3) must collapse to (3,)
    g = np.ones((2, 3))
    assert unbroadcast(g, (3,)).shape == (3,)
    assert np.allclose(unbroadcast(g, (3,)), [2.0, 2.0, 2.0])
    # size-1 axis stretched
    assert unbroadcast(np.ones((2, 3)), (1, 3)).shape == (1, 3)
    assert np.allclose(unbroadcast(np.ones((2, 3)), (1, 3)), [[2.0, 2.0, 2.0]])


def test_broadcasting_vjp_in_backprop():
    """Adding a (3,) bias to a (2,3) matrix: the bias grad must sum over rows."""
    def f(b):
        return (Node(np.ones((2, 3))) + b).sum()

    gb = grad(f)(np.zeros(3))
    assert gb.shape == (3,)
    assert np.allclose(gb, [2.0, 2.0, 2.0])  # two rows each contribute 1

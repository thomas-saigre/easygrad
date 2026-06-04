import numpy as np

from easygrad import reverse
from easygrad.reverse import Node, value_and_grad
from easygrad.train import gradient_descent


def _make_data(n=64):
    X = np.linspace(-1.0, 1.0, n).reshape(n, 1)
    y = np.sin(np.pi * X)
    return X, y


def _init(H, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((1, H)),          # W1
        rng.standard_normal(H),               # b1
        rng.standard_normal((H, 1)) / np.sqrt(H),  # W2
        np.zeros(1),                          # b2
    ]


def _mlp_loss_factory(X, y):
    def loss(W1, b1, W2, b2):
        h = reverse.relu(Node(X) @ W1 + b1)
        pred = h @ W2 + b2
        return ((pred - Node(y)) ** 2).mean()

    return loss


def test_mlp_training_reduces_loss():
    X, y = _make_data()
    loss = _mlp_loss_factory(X, y)

    params = _init(H=16)
    # plain gradient descent on a gentle target (see docs/train-mlp.md)
    params, history = gradient_descent(loss, params, lr=0.1, steps=1500)

    initial = history[0]
    final, _ = value_and_grad(loss, argnums=(0, 1, 2, 3))(*params)

    # this 2-layer ReLU net should cut the loss by >10x
    assert final < 0.1 * initial
    assert final < 0.05


def test_gradient_descent_returns_history():
    """``gradient_descent`` returns one loss reading per step, monotone-ish down."""
    X, y = _make_data(n=8)
    loss = _mlp_loss_factory(X, y)

    params = _init(H=4)
    steps = 20
    _, history = gradient_descent(loss, params, lr=0.05, steps=steps)

    assert len(history) == steps
    assert all(isinstance(L, float) for L in history)
    # gentle target + small lr: the last step's loss should beat the first
    assert history[-1] < history[0]


def test_training_step_matches_manual_gradient():
    """One value_and_grad call returns both the loss and a usable gradient."""
    X, y = _make_data(n=8)
    loss = _mlp_loss_factory(X, y)

    params = _init(H=4)
    L, grads = value_and_grad(loss, argnums=(0, 1, 2, 3))(*params)
    assert isinstance(L, float)
    assert [g.shape for g in grads] == [(1, 4), (4,), (4, 1), (1,)]
    # a tiny step along -grad must not increase the loss
    stepped = [p - 1e-4 * g for p, g in zip(params, grads)]
    L2, _ = value_and_grad(loss, argnums=(0, 1, 2, 3))(*stepped)
    assert L2 <= L

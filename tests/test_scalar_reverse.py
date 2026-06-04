import math

import pytest

from easygrad import dual
from easygrad.scalar_reverse import Value, topo_sort, is_valid_topo_order


def test_grad_matches_analytic_simple():
    # f(a,b) = a*b + a   ->  df/da = b + 1, df/db = a
    a, b = Value(2.0), Value(3.0)
    out = a * b + a
    out.backward()
    assert a.grad == pytest.approx(3.0 + 1.0)
    assert b.grad == pytest.approx(2.0)


def test_grad_with_elementary_functions():
    # f(x) = tanh(exp(x) + x^2)
    x = Value(0.5)
    out = (x.exp() + x**2).tanh()
    out.backward()
    xv = 0.5
    inner = math.exp(xv) + xv**2
    expected = (1 - math.tanh(inner) ** 2) * (math.exp(xv) + 2 * xv)
    assert x.grad == pytest.approx(expected, rel=1e-12)


def test_reverse_matches_dual_forward():
    # same scalar function via reverse tape and via forward dual numbers
    def f_value(x):
        return (x * x + Value(1.0)).log()

    def f_dual(x):
        return dual.log(x * x + 1.0)

    xv = 1.7
    x = Value(xv)
    out = f_value(x)
    out.backward()
    assert x.grad == pytest.approx(dual.derivative(f_dual, xv), rel=1e-12)


def test_reused_node_accumulates():
    # f(x) = x * x via a *single reused node* -> df/dx = 2x (needs += not =)
    x = Value(3.0)
    out = x * x
    out.backward()
    assert x.grad == pytest.approx(6.0)

    # f(x) = x + x + x -> 3
    y = Value(1.0)
    out2 = y + y + y
    out2.backward()
    assert y.grad == pytest.approx(3.0)


def test_topo_sort_is_valid_linearization():
    a = Value(1.0)
    b = Value(2.0)
    c = a * b
    d = c + a  # 'a' reused -> diamond
    order = topo_sort(d)
    assert is_valid_topo_order(order)
    # every reachable node appears exactly once
    assert len(order) == len({id(n) for n in order})
    assert d is order[-1]


def test_backward_is_idempotent():
    # calling backward twice should give the same grads (it re-zeros first)
    x = Value(2.0)
    out = x**3
    out.backward()
    g1 = x.grad
    out.backward()
    assert x.grad == pytest.approx(g1) == pytest.approx(12.0)

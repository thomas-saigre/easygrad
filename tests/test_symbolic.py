import sympy as sp

from easygrad import symbolic


def test_logistic_map_is_one_step():
    x, r = sp.symbols("x r")
    assert sp.expand(symbolic.logistic_map(x, r) - (r * x - r * x ** 2)) == 0


def test_expression_size_counts_nodes():
    x = sp.symbols("x")
    assert symbolic.expression_size(x) == 1
    assert symbolic.expression_size(x + 1) > 1


def test_swell_demo_grows_exponentially():
    data = symbolic.swell_demo(max_iter=5)
    orders = [n for n, _ in data]
    sizes = [s for _, s in data]

    assert orders == [0, 1, 2, 3, 4, 5]
    assert sizes[0] == 1                       # d/dx of x is 1
    assert all(b > a for a, b in zip(sizes, sizes[1:]))   # strictly increasing

    # super-linear: later steps add far more than earlier ones (a linear curve
    # would have roughly constant differences)
    diffs = [b - a for a, b in zip(sizes, sizes[1:])]
    assert diffs[-1] > 5 * diffs[0]

import math

import pytest


# Closed-form (f, f') pairs evaluated with plain math, for the numeric tests.
SCALAR_SUITE = [
    ("poly", lambda x: x**3 - 2 * x + 1, lambda x: 3 * x**2 - 2, 1.3),
    ("exp", lambda x: math.exp(0.5 * x), lambda x: 0.5 * math.exp(0.5 * x), 0.7),
    ("sin", lambda x: math.sin(x) * x, lambda x: math.sin(x) + x * math.cos(x), 1.1),
    ("ratio", lambda x: 1.0 / (1.0 + x**2), lambda x: -2 * x / (1 + x**2) ** 2, 0.6),
    ("composite", lambda x: math.log(1 + math.exp(x)), lambda x: 1 / (1 + math.exp(-x)), -0.4),
]


@pytest.fixture(params=SCALAR_SUITE, ids=[s[0] for s in SCALAR_SUITE])
def scalar_case(request):
    _, f, fp, x0 = request.param
    return f, fp, x0

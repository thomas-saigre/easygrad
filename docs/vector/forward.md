# Forward mode over arrays

**Objective.** Lift dual numbers to arrays to compute JVPs.

## Recap

Forward mode over arrays is [dual numbers](../scalar/forward-dual.md), lifted to arrays.
Each value carries a `primal` and a `tangent` of the same shape, and they propagate together.
One forward pass computes one **JVP**:

$$
\text{jvp}(f, x, v) = \big(f(x),\; J\,v\big),
$$

i.e., the directional derivative of $f$ along $v$.
The rules are the dual-number rules with arrays substituted in.
For instance, for matmul, the product rule gives $(Ax)' = A'x + Ax'$.

Remark that forward and reverse are transposes operations.
For $f : \mathbb{R}^n \to \mathbb{R}^m$:

- JVP (forward): $v \mapsto J v$ push a tangent forward.
- VJP (reverse): $\bar y \mapsto J^\top \bar y$  pull a cotangent backward.

They satisfy the adjoint identity

$$
\langle \bar y,\; J v \rangle = \langle J^\top \bar y,\; v \rangle,
$$

which the test suite checks numerically: it builds $Jv$ with `jvp` and $J^\top \bar y$ as the [reverse-mode](reverse.md) gradient of $x \mapsto \langle \bar y, f(x)\rangle$, then confirms the two inner products agree.

If tangent, cotangent, pull and push start to sound a bit geometrical or categorical to you (and you like it), go read this nice paper [The simple essence of automatic differentiation](https://arxiv.org/abs/1804.00746) by C. Elliot.

## Exercise

Implement the forward-mode arithmetic in [`src/easygrad/forward.py`](https://github.com/svaiter/easygrad/blob/main/src/easygrad/forward.py).
The `DualArray` operators (`+`, `-`, `*`, `/`, `**`, `@`) and the elementwise `exp`/`log`/`tanh`/`relu`, each propagating the **tangent** the same way the [scalar dual numbers](../scalar/forward-dual.md).

```python
import numpy as np
from easygrad import forward
from easygrad.forward import jvp

def f(x):
    return forward.tanh(x * x).sum()

primal, tangent = jvp(f, [np.array([0.3, -0.7, 1.2])],
                          [np.array([1.0,  0.5, -2.0])])
# primal  = f(x)
# tangent = directional derivative  J @ v
```

`jvp(f, primals, tangents)` (given) pairs each primal with its seed tangent and reads the output tangent in one pass.

Validate with `uv run pytest tests/test_forward.py`.

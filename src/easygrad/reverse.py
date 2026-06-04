"""Reverse mode over numpy arrays: VJPs and backprop."""

from typing import Callable, Sequence

import numpy as np

type ArrayLike = Node | np.ndarray | float | int


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` so it matches ``shape`` after numpy broadcasting.

    EXERCISE -- see docs/vector/reverse.md.
    """
    raise NotImplementedError("Exercise: unbroadcast (docs/vector/reverse.md)")


class Node:
    """A node wrapping a numpy array in the reverse-mode tape."""

    # see https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    __array_ufunc__ = None

    def __init__(
        self, value: ArrayLike, _children: tuple["Node", ...] = (), _op: str = ""
    ) -> None:
        self.value: np.ndarray = np.asarray(value, dtype=float)
        self.grad: np.ndarray = np.zeros_like(self.value)
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple["Node", ...] = tuple(_children)
        self._op: str = _op

    # -- helpers
    @staticmethod
    def _coerce(other: ArrayLike) -> "Node":
        return other if isinstance(other, Node) else Node(other)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Node(shape={self.value.shape}, op={self._op!r})"

    # binary ops (EXERCISE)
    # Compute `out` with the forward value + children, a `_backward` that
    # accumulates VJPs into each parent's .grad (use `unbroadcast`),
    # and return out.
    def __add__(self, other: ArrayLike) -> "Node":
        other = self._coerce(other)
        raise NotImplementedError("Exercise: Node.__add__ (docs/vector/reverse.md)")

    __radd__ = __add__

    def __mul__(self, other: ArrayLike) -> "Node":
        other = self._coerce(other)
        raise NotImplementedError("Exercise: Node.__mul__ (docs/vector/reverse.md)")

    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> "Node":
        other = self._coerce(other)
        raise NotImplementedError("Exercise: Node.__truediv__ (docs/vector/reverse.md)")

    def __pow__(self, p: float) -> "Node":
        raise NotImplementedError("Exercise: Node.__pow__ (docs/vector/reverse.md)")

    def __matmul__(self, other: ArrayLike) -> "Node":
        # use the given _matmul_lhs_grad / _matmul_rhs_grad helpers in _backward
        other = self._coerce(other)
        raise NotImplementedError("Exercise: Node.__matmul__ (docs/vector/reverse.md)")

    # derived / reflected ops (given)
    def __sub__(self, other: ArrayLike) -> "Node":
        return self + (-self._coerce(other))

    def __rsub__(self, other: ArrayLike) -> "Node":
        return self._coerce(other) + (-self)

    def __neg__(self) -> "Node":
        return self * -1

    def __rmatmul__(self, other: ArrayLike) -> "Node":
        return self._coerce(other).__matmul__(self)

    # reductions
    def sum(self, axis: int | tuple[int, ...] | None = None) -> "Node":
        """Sum reduction. VJP: broadcast the cotangent back over the summed axes.

        You should use :func:`_restore_axes` helper

        EXERCISE -- see docs/vector/reverse.md.
        """
        raise NotImplementedError("Exercise: Node.sum (docs/vector/reverse.md)")

    def mean(self, axis: int | tuple[int, ...] | None = None) -> "Node":
        n = (
            self.value.size
            if axis is None
            else np.prod(np.take(self.value.shape, np.atleast_1d(axis)))
        )
        return self.sum(axis=axis) * (1.0 / float(n))

    # elementwise functions (EXERCISE)
    def exp(self) -> "Node":
        raise NotImplementedError("Exercise: Node.exp (docs/vector/reverse.md)")

    def log(self) -> "Node":
        raise NotImplementedError("Exercise: Node.log (docs/vector/reverse.md)")

    def tanh(self) -> "Node":
        raise NotImplementedError("Exercise: Node.tanh (docs/vector/reverse.md)")

    def relu(self) -> "Node":
        raise NotImplementedError("Exercise: Node.relu (docs/vector/reverse.md)")

    # reverse pass (EXERCISE)
    def backward(self) -> None:
        """Seed the (scalar) output with 1 and accumulate gradients.

        Use the given :func:`_topo` to order the graph. Zero every node's grad,
        seed ``self.grad = ones_like(self.value)``, then call each ``_backward``

        EXERCISE -- see docs/vector/reverse.md.
        """
        raise NotImplementedError("Exercise: Node.backward (docs/vector/reverse.md)")


# matrix helpers
def _matmul_lhs_grad(g: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = np.asarray(g)
    if a.ndim == 1 and b.ndim == 1:  # vector dot -> scalar
        return g * b
    if a.ndim == 1:  # (k,) @ (k,n) -> (n,)
        return g @ b.T
    if b.ndim == 1:  # (m,k) @ (k,) -> (m,)
        return np.outer(g, b)
    return g @ b.T  # (m,k) @ (k,n) -> (m,n)


def _matmul_rhs_grad(g: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = np.asarray(g)
    if a.ndim == 1 and b.ndim == 1:
        return g * a
    if a.ndim == 1:  # (k,) @ (k,n) -> (n,)
        return np.outer(a, g)
    if b.ndim == 1:  # (m,k) @ (k,) -> (m,)
        return a.T @ g
    return a.T @ g


def _restore_axes(
    grad: np.ndarray, shape: tuple[int, ...], axis: int | tuple[int, ...] | None
) -> np.ndarray:
    """Reinsert reduced axes (as size 1) so the cotangent can broadcast back."""
    if axis is None:
        return np.asarray(grad)
    grad = np.asarray(grad)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a % len(shape) for a in axes)
    return np.expand_dims(grad, axes)


def _topo(root: Node) -> list[Node]:
    order: list[Node] = []
    visited: set[int] = set()

    def build(node: Node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for child in node._prev:
            build(child)
        order.append(node)

    build(root)
    return order


# elementwise free functions
def exp(x: Node) -> Node:
    return x.exp()


def log(x: Node) -> Node:
    return x.log()


def tanh(x: Node) -> Node:
    return x.tanh()


def relu(x: Node) -> Node:
    return x.relu()

# jax-like interface
def grad(
    f: Callable[..., Node], argnums: int | Sequence[int] = 0
):
    """Return a function computing the gradient of scalar-output ``f``.

    ``f`` takes one or more numpy arrays and returns a scalar :class:`Node`.
    ``grad(f)(x)`` returns ``df/dx`` with the same shape as ``x``. Pass
    ``argnums`` to differentiate with respect to other / several positional
    arguments (mirrors ``jax.grad``).
    """
    single = isinstance(argnums, int)
    nums: tuple[int, ...] = (argnums,) if single else tuple(argnums)

    def grad_fn(*args: np.ndarray):
        nodes = [Node(a) if i in nums else a for i, a in enumerate(args)]
        node_args = [n if isinstance(n, Node) else Node(n) for n in nodes]
        out = f(*node_args)
        if out.value.size != 1:
            raise ValueError("grad requires a scalar-output function")
        out.backward()
        grads = [node_args[i].grad for i in nums]
        return grads[0] if single else tuple(grads)

    return grad_fn


def value_and_grad(f: Callable[..., Node], argnums: int | Sequence[int] = 0):
    """Like :func:`grad` but also returns the scalar value of ``f``."""
    single = isinstance(argnums, int)
    nums: tuple[int, ...] = (argnums,) if single else tuple(argnums)

    def vg(*args: np.ndarray):
        node_args = [Node(a) for a in args]
        out = f(*node_args)
        if out.value.size != 1:
            raise ValueError("value_and_grad requires a scalar-output function")
        out.backward()
        grads = [node_args[i].grad for i in nums]
        return float(out.value), (grads[0] if single else tuple(grads))

    return vg

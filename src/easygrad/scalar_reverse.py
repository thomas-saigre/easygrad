"""Scalar reverse mode."""

from typing import Callable, Iterable


class Value:
    """A node in a scalar computational graph."""

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = "") -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        # internal autograd
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple["Value", ...] = tuple(_children)
        self._op: str = _op

    # helpers
    @staticmethod
    def _coerce(other: "Value | float") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # primitive ops (EXERCISE)
    # Each should:
    # - compute `out` with the forward value and (self[, other]) as children,
    # - define a `_backward` that accumulates into the parents .grad
    # - assign it to out._backward, and return out.
    def __add__(self, other: "Value | float") -> "Value":
        other = self._coerce(other)
        raise NotImplementedError("Exercise: Value.__add__ (docs/scalar/reverse-tape.md)")

    __radd__ = __add__

    def __mul__(self, other: "Value | float") -> "Value":
        other = self._coerce(other)
        raise NotImplementedError("Exercise: Value.__mul__ (docs/scalar/reverse-tape.md)")

    __rmul__ = __mul__

    def __pow__(self, p: float) -> "Value":
        assert isinstance(p, (int, float)), "only constant real powers supported"
        raise NotImplementedError("Exercise: Value.__pow__ (docs/scalar/reverse-tape.md)")

    # derived ops
    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-self._coerce(other))

    def __rsub__(self, other: "Value | float") -> "Value":
        return self._coerce(other) + (-self)

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: "Value | float") -> "Value":
        return self._coerce(other) * self ** -1

    # elementary functions (EXERCISE)
    def exp(self) -> "Value":
        raise NotImplementedError("Exercise: Value.exp (docs/scalar/reverse-tape.md)")

    def log(self) -> "Value":
        raise NotImplementedError("Exercise: Value.log (docs/scalar/reverse-tape.md)")

    def tanh(self) -> "Value":
        raise NotImplementedError("Exercise: Value.tanh (docs/scalar/reverse-tape.md)")

    def relu(self) -> "Value":
        raise NotImplementedError("Exercise: Value.relu (docs/scalar/reverse-tape.md)")

    # reverse pass (EXERCISE)
    def backward(self) -> None:
        """Run reverse-mode autodiff seeded at ``self`` (assumed scalar output). """
        raise NotImplementedError("Exercise: Value.backward (docs/scalar/reverse-tape.md)")


def topo_sort(root: Value) -> list[Value]:
    """Return a topological ordering of the graph ending at ``root``.

    EXERCISE -- see docs/scalar/reverse-tape.md.
    """
    raise NotImplementedError("Exercise: topo_sort (docs/scalar/reverse-tape.md)")


def is_valid_topo_order(order: Iterable[Value]) -> bool:
    """Check that no node precedes one of its parents in ``order`` (given)."""
    seen: set[int] = set()
    for node in order:
        for parent in node._prev:
            if id(parent) not in seen:
                return False
        seen.add(id(node))
    return True

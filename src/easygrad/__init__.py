"""easygrad: a teaching implementation of automatic differentiation."""

from . import (
    dual,
    forward,
    numeric,
    reverse,
    scalar_reverse,
    symbolic,
    train,
)

__all__ = [
    "numeric",
    "symbolic",
    "dual",
    "scalar_reverse",
    "reverse",
    "forward",
    "train",
]

__version__ = "0.1.0"

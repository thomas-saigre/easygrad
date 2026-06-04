"""easygrad: a teaching implementation of automatic differentiation. """

from . import (
    dual,
    forward,
    numeric,
    reverse,
    scalar_reverse,
    symbolic,
    tracing,
    train,
)

__all__ = [
    "numeric",
    "symbolic",
    "dual",
    "scalar_reverse",
    "reverse",
    "forward",
    "tracing",
    "train",
]

__version__ = "0.1.0"

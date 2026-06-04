# An introduction to automatic differentiation

Automatic differentiation (AD) is at the core of every modern ML framework.
This site/repository proposes you to *implement* each idea in minimal Python [`easygrad`](https://github.com/svaiter/easygrad) package.

We assume you are comfortable with calculus, linear algebra, `numpy`, but have not looked inside an AD framework before.

There are two "obvious" ways to differentiate a function on a computer:

- **Numerical (finite differences):** scales linearly with the dimension and no good $h$: see the [numerical primer](primer/numerical.md).
- **Symbolic:** use a CAS and differentiate the formula. Exact, but the expression for a nested function may [swells](primer/symbolic.md).

**Automatic differentiation** is a third option.
It is exact and cheap (proportional to the original program, also known as the Baur-Strassen theorem).
It applies the chain rule to the *program itself*, elementary operation by operation, reusing intermediate values.

## How to read this site

This site is not a full course by itself on automatic differentiation but rather a quick recap of concepts + exercises to implement them.

1. **Primer**: shortcomings of [numerical](primer/numerical.md) and [symbolic](primer/symbolic.md) differentiation + the idea of [complex step](primer/complex-step.md).
2. **Scalar case**: [dual numbers](scalar/forward-dual.md) (forward) and the
   [tape](scalar/reverse-tape.md) (reverse).
3. **Vector case**: [array backprop / VJPs](vector/reverse.md) and
   [array JVPs](vector/forward.md).
4. **[Let's train an MLP](train-mlp.md)**: check your implementation with a example.


The `easygrad` package in `src/` ships with the core functions left unimplemented, each raises `NotImplementedError`.
Your job is to fill them in until the test suite goes green:

```bash
uv run pytest
```

You can also run just one file while iterating, e.g.
```bash
uv run pytest tests/test_dual.py
```

## Ressources

These repository is a companion to a lecture I gave at the [Workshop on Machine Learning and Automatic Differentiation in JAX for Scientific Computing](https://majsc2026.pages.math.unistra.fr/) in Strasbourg in June 2026.

Other ressources include:

- [micrograd](https://github.com/karpathy/micrograd): a PyTorch-like minimal engine.
- [autodidact](https://github.com/mattjj/autodidact): a JAX-like minimal engine.
- [autodidax](https://docs.jax.dev/en/latest/autodidax.html): JAX's own from-scratch tutorial.
- [minitorch](https://minitorch.github.io): broader scope (module 1 is close to this course).

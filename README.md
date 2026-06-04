# easygrad

Automatic differentiation (AD) is at the core of every modern ML framework.
This site/repository proposes you to implement each idea in a minimal Python package.

## Exercices

It is recommended to use [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv sync
uv run pytest
uv run pytest tests/test_dual.py
```

Each `src/easygrad/*.py` module marks its exercises with `raise NotImplementedError(...)` and a pointer to the docs page that explains the task.

## Docs site

```bash
uv run zensical serve   # preview the docs site locally
uv run zensical build   # build the static site into site/
```

## Ressources

- [micrograd](https://github.com/karpathy/micrograd): a PyTorch-like minimal engine.
- [autodidact](https://github.com/mattjj/autodidact): a JAX-like minimal engine.
- [autodidax](https://docs.jax.dev/en/latest/autodidax.html): JAX's own from-scratch tutorial.
- [minitorch](https://minitorch.github.io): broader scope (module 1 is close to this course).

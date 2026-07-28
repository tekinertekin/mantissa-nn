# Contributing to mantissa-nn

`mantissa-nn` provides dense neural-network building blocks (Dense, Flatten,
and the training loop) on top of the
[mantissa](https://github.com/tekinertekin/mantissa) low-precision C core, with
a thin, NumPy-friendly Python API. Contributions of all sizes are welcome — bug
fixes, new layers, benchmarks, examples, docs.

**Anyone is welcome here** — no prior involvement needed. Open an issue to report
a bug, ask a question, or propose an idea, and open a pull request when you have a
change. First-time contributors are especially welcome.

## Getting started

```sh
git clone https://github.com/tekinertekin/mantissa-nn
cd mantissa-nn
pip install -e ".[dev]"     # installs mantissa-core + pytest
pytest                      # run the test suite
```

## Backends

Layers run either in the mantissa C core (the default engine) or on a pure-NumPy
reference backend used for debugging and for running without the compiled core.
The two are kept numerically in step, so a test can assert one against the other.

## Project layout

```
mantissa_nn/    the package (layers.py, engine, _engine.py, _numpy_backend)
tests/          pytest suite (layers, engine, backends)
```

## Making a change

1. **Open an issue first** for anything non-trivial, so we can agree on the
   approach. Small, obvious fixes can go straight to a PR.
2. Keep the diff focused — one logical change per PR.
3. **Add or update a test.** `pytest` must pass. For a numerical fix, add a case
   that fails before and passes after; new layers should check gradients and
   agree across the engine and numpy backends.
4. Match the surrounding style: clear NumPy-style code, comments only where the
   *why* isn't obvious.

## AI-assisted contributions

Using AI to help write code is **completely fine**. But you are **responsible for
every single line** you submit — you must understand it, be able to explain it,
and stand behind its correctness. Please mention in the PR that AI helped, so
reviewers have the full picture.

## License

By contributing, you agree that your contributions are licensed under the
project's MIT License (see [LICENSE](LICENSE)).

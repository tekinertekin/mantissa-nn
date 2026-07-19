# mantissa-nn

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)
[![Engine](https://img.shields.io/badge/engine-mantissa-00599C.svg)](https://github.com/tekinertekin/mantissa)

**The shared neural-net base for the mantissa family.**

`mantissa-nn` holds the primitives every other package in the family reuses,
so that the specialized packages depend on this base rather than on each
other:

- the **C-engine binding** (`mantissa_nn._engine.engine`) — loads
  [mantissa](https://github.com/tekinertekin/mantissa) once and gates on the
  dense primitive every model needs;
- the **pure-numpy reference backend** (`mantissa_nn._numpy_backend`) — the
  identical call signatures in numpy, the correctness oracle;
- the **fully-connected layers** `Dense` / `Flatten` and the `Layer` contract
  (`mantissa_nn.layers`).

Datasets are deliberately *not* here — each package owns and downloads its own,
so they stay independent.

Built on top of it:

| package | adds |
|---------|------|
| [mantissa-cnn](https://github.com/tekinertekin/mantissa-cnn) | convolution + pooling layers, a CNN model zoo |
| [mantissa-mlp](https://github.com/tekinertekin/mantissa-mlp) | the tabular training loop + tabular datasets |
| [mantissa-autoencoder](https://github.com/tekinertekin/mantissa-autoencoder) | encoder/decoder model + decoder layers |

```python
from mantissa_nn import layers
from mantissa_nn._engine import engine     # the shared mantissa.Mantissa
```

The default compute path is the mantissa C engine (>= 0.2.1); `backend="numpy"`
in the model packages selects the pure-numpy reference implementation.

## Install

```
pip install mantissa-nn
```

For the development layout, clone the family repos as siblings and
`pip install -e .` — a sibling `mantissa/` checkout is picked up automatically
when `mantissa-core` is not installed.

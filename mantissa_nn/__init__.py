"""mantissa-nn: the shared neural-net primitives for the mantissa family.

The base every other package builds on — the C-engine binding, the pure-numpy
reference backend, and the fully-connected ``Dense`` / ``Flatten`` layers with
the ``Layer`` contract. mantissa-cnn adds convolution and pooling; mantissa-mlp
the tabular training loop; and mantissa-autoencoder the encoder/decoder model —
all on top of these. Datasets are each package's own concern, not shared here.

>>> from mantissa_nn import layers
>>> from mantissa_nn._engine import engine   # the shared mantissa.Mantissa

The default compute path is the mantissa C engine (>= 0.2.1); the pure-numpy
backend (:mod:`mantissa_nn._numpy_backend`) has the identical call signatures
and serves as the correctness oracle.
"""
from ._engine import MANTISSA_MIN_VERSION, MANTISSA_PIP_NAME, engine, load_mantissa
from .layers import Dense, Flatten, Layer
from . import layers

__version__ = "0.1.0"
__all__ = ["Layer", "Dense", "Flatten", "layers",
           "engine", "load_mantissa", "MANTISSA_PIP_NAME", "MANTISSA_MIN_VERSION"]

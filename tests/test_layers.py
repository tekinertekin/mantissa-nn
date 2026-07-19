"""Shape propagation and scratch reuse for the base layers (Dense, Flatten).
All on the numpy backend — no engine needed."""
import numpy as np
import pytest

from mantissa_nn import _numpy_backend as B
from mantissa_nn.layers import Dense, Flatten, Layer


def _built(layer, in_shape, seed=0):
    layer.build(in_shape, np.random.default_rng(seed))
    return layer


def test_flatten_roundtrip():
    fl = _built(Flatten(), (16, 5, 5))
    assert fl.out_shape == (400,)
    X = np.random.default_rng(3).normal(size=(2, 16, 5, 5)).astype(np.float32)
    Y = fl.forward(X, B)
    assert Y.shape == (2, 400)
    assert np.array_equal(fl.backward(Y, B), X)


def test_dense_shapes_and_flat_input_required():
    d = _built(Dense(120, act="relu"), (400,))
    assert d.W.shape == (120, 400) and d.out_shape == (120,)
    X = np.random.default_rng(4).normal(size=(5, 400)).astype(np.float32)
    assert d.forward(X, B).shape == (5, 120)
    with pytest.raises(ValueError, match="Flatten"):
        Dense(10).build((16, 5, 5), np.random.default_rng(0))


def test_dense_backward_shapes_and_param_count():
    d = _built(Dense(8), (5,))
    assert d.param_count() == 8 * 5 + 8
    X = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    Y = d.forward(X, B)
    dX = d.backward(np.ones_like(Y), B)
    assert dX.shape == (4, 5)
    assert d.dW.shape == d.W.shape and d.db.shape == d.b.shape


def test_dense_rejects_bad_activation():
    with pytest.raises(ValueError, match="act must be one of"):
        Dense(4, act="softmax")


def test_scratch_reused_across_batches():
    # Design requirement: Z/Y/grad buffers allocated once per batch shape.
    d = _built(Dense(6), (8,))
    X = np.random.default_rng(5).normal(size=(6, 8)).astype(np.float32)
    assert d.forward(X, B) is d.forward(X, B)          # same Y buffer
    g1 = d.backward(d._scratch[6]["Y"], B)
    g2 = d.backward(d._scratch[6]["Y"], B)
    assert g1 is g2                                    # same dX buffer


def test_layer_base_is_abstract():
    with pytest.raises(NotImplementedError):
        Layer().build((3,), np.random.default_rng(0))

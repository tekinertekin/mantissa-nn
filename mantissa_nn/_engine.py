"""Locate and load the mantissa engine — the shared base binding.

Resolution order (same as the rest of the family):
1. an installed ``mantissa`` package (pip: ``mantissa-core``),
2. the sibling development checkout ``../mantissa/python`` (the repo layout
   used before the PyPI release).

Every engine call across the mantissa-nn family goes through the object
returned by :func:`engine` — no other module imports mantissa. This base
binding gates on the *dense* primitive (``linear_forward_batch``), which every
model built on this package needs; the CNN feature gate
(``conv2d_forward``) lives in :mod:`mantissa_cnn._engine`, layered on top.
:mod:`mantissa_nn._numpy_backend` implements the identical call signatures in
pure numpy, so a model is backend-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The PyPI distribution name for mantissa. Update this one constant (and
# pyproject.toml) if it changes; nothing else references the pip name.
MANTISSA_PIP_NAME = "mantissa-core"

# Minimum engine release for the primitives this family relies on.
MANTISSA_MIN_VERSION = "0.2.1"

# Sibling checkout when this repo lives next to mantissa/:
#   <parent>/mantissa-nn/mantissa_nn/_engine.py  ->  <parent>/mantissa/python
_DEV_PYTHON_DIR = Path(__file__).resolve().parents[2] / "mantissa" / "python"

_tk = None  # process-wide engine singleton (one dylib load)


def load_mantissa():
    """Import and return the ``mantissa`` module.

    Raises ImportError with the exact install command if it cannot be found.
    """
    try:
        import mantissa
        return mantissa
    except ImportError:
        pass
    # The checkout ships either a module (mantissa.py) or a package
    # (mantissa/__init__.py) depending on the packaging work — accept both.
    if (_DEV_PYTHON_DIR / "mantissa.py").is_file() \
            or (_DEV_PYTHON_DIR / "mantissa" / "__init__.py").is_file():
        p = str(_DEV_PYTHON_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
        import mantissa
        return mantissa
    raise ImportError(
        f"mantissa is not installed — run: pip install {MANTISSA_PIP_NAME}\n"
        f"(dev fallback also checked: {_DEV_PYTHON_DIR})"
    )


def engine():
    """Return the shared ``mantissa.Mantissa`` instance, gated on the dense API.

    Feature detection: ``hasattr(tk, "linear_forward_batch")`` — the fully-
    connected primitive Dense builds on. Raises ImportError when mantissa is
    absent entirely, RuntimeError when the installed engine predates the
    primitives this package needs.
    """
    global _tk
    if _tk is None:
        tk = load_mantissa().Mantissa()
        if not hasattr(tk, "linear_forward_batch"):
            raise RuntimeError(
                f"mantissa >= {MANTISSA_MIN_VERSION} required — "
                f"run: pip install --upgrade {MANTISSA_PIP_NAME}"
            )
        _tk = tk
    return _tk

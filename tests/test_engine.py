"""Engine resolution errors (verbatim) for the shared base binding."""
import sys
import types

import pytest

import mantissa_nn._engine as eng

MISSING_MSG = "mantissa is not installed — run: pip install mantissa-core"
TOO_OLD_MSG = ("mantissa >= 0.2.1 required — "
               "run: pip install --upgrade mantissa-core")


def test_missing_mantissa_message_verbatim(monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "_tk", None)
    monkeypatch.setattr(eng, "_DEV_PYTHON_DIR", tmp_path / "nowhere")
    monkeypatch.setitem(sys.modules, "mantissa", None)   # import -> ImportError
    with pytest.raises(ImportError) as exc:
        eng.engine()
    lines = str(exc.value).splitlines()
    assert lines[0] == MISSING_MSG
    assert "dev fallback also checked" in lines[1]


def test_too_old_mantissa_message_verbatim(monkeypatch):
    fake = types.ModuleType("mantissa")
    fake.Mantissa = type("Mantissa", (), {})             # no linear_forward_batch
    monkeypatch.setattr(eng, "_tk", None)
    monkeypatch.setitem(sys.modules, "mantissa", fake)
    with pytest.raises(RuntimeError) as exc:
        eng.engine()
    assert str(exc.value) == TOO_OLD_MSG


def test_engine_memoizes_singleton(monkeypatch):
    calls = {"n": 0}

    class FakeMantissa:
        def linear_forward_batch(self, *a, **k):
            pass

    fake = types.ModuleType("mantissa")

    def _factory():
        calls["n"] += 1
        return FakeMantissa()

    fake.Mantissa = _factory
    monkeypatch.setattr(eng, "_tk", None)
    monkeypatch.setitem(sys.modules, "mantissa", fake)
    a = eng.engine()
    b = eng.engine()
    assert a is b and calls["n"] == 1

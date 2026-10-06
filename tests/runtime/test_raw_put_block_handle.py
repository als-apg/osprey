"""The put symbols on the ``libca`` handle a permitted pyepics load returns.

A readonly run lets pyepics load ``libca`` through ``ctypes`` and closes the
put entry points on the handle it gets back. Every spelling of a Channel
Access put that the handle exposes must be closed, the synchronous-group put
included: ``ca_sg_array_put`` writes the channel as ``ca_array_put`` does, only
deferred until the group is flushed.

``install`` is called directly with the ``ctypes`` rows alone, so the only
process-wide objects it patches are the ``ctypes`` loaders — restored after
each test together with the import-hook finder it registers.
"""

import ctypes
import sys
from types import ModuleType

import pytest

from osprey.runtime import raw_put_block

_REFUSAL = "refused by the handle test"

_LOADER_ROWS = (
    ("ctypes", ("CDLL", "PyDLL", "WinDLL", "OleDLL")),
    ("ctypes.LibraryLoader", ("LoadLibrary", "__getattr__")),
)


@pytest.fixture(autouse=True)
def _restore_ctypes():
    saved = []
    for dotted, attrs in _LOADER_ROWS:
        target = ctypes if dotted == "ctypes" else ctypes.LibraryLoader
        for attr in attrs:
            if attr in vars(target):
                saved.append((target, attr, vars(target)[attr], True))
            elif hasattr(target, attr):
                saved.append((target, attr, None, False))
    yield
    for target, attr, value, own in saved:
        if own:
            setattr(target, attr, value)
        elif attr in vars(target):
            delattr(target, attr)
    sys.meta_path[:] = [
        f for f in sys.meta_path if not getattr(f, raw_put_block._GUARD_ATTR, False)
    ]


def _install_fake_pyepics_loader(monkeypatch):
    """A stand-in ``epics.ca`` whose ``initialize_libca`` loads a library as pyepics does."""
    ca = ModuleType("epics.ca")

    def initialize_libca():
        return ctypes.cdll.LoadLibrary(None)

    ca.initialize_libca = initialize_libca
    package = ModuleType("epics")
    package.ca = ca
    monkeypatch.setitem(sys.modules, "epics", package)
    monkeypatch.setitem(sys.modules, "epics.ca", ca)
    return ca


def _install_readonly():
    raw_put_block.install(
        "readonly", eager_targets=_LOADER_ROWS, deferred_targets=(), refusal=_REFUSAL
    )


def test_handle_put_symbols_name_the_synchronous_group_put():
    assert "ca_sg_array_put" in raw_put_block._HANDLE_PUT_SYMBOLS


@pytest.mark.parametrize("symbol", ["ca_array_put", "ca_array_put_callback", "ca_sg_array_put"])
def test_readonly_refuses_each_put_symbol_on_the_pyepics_handle(monkeypatch, symbol):
    ca = _install_fake_pyepics_loader(monkeypatch)
    _install_readonly()

    handle = ca.initialize_libca()

    assert handle is not None
    with pytest.raises(RuntimeError, match=_REFUSAL):
        getattr(handle, symbol)(0, 1, object(), object())


def test_readonly_refuses_a_load_outside_pyepics(monkeypatch):
    _install_fake_pyepics_loader(monkeypatch)
    _install_readonly()

    with pytest.raises(RuntimeError, match=_REFUSAL):
        ctypes.cdll.LoadLibrary(None)

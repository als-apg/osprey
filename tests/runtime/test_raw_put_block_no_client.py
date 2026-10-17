"""The ``no_client`` mode of the raw-put block: no control-system client at all.

``install("no_client", ...)`` refuses a client package at import and a client
shared library at load, for reads and writes alike. Every in-process contract
here names fake packages, fake libraries and fake symbols, never the real
ones, so the test process itself never loses a client; the cases that need a
real client library run in a subprocess. The fixture from
:mod:`tests.runtime._patch_restore` puts back every loader the block replaced
and drops its finders.
"""

from __future__ import annotations

import _ctypes
import ctypes
import ctypes.util
import importlib
import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from osprey.runtime import raw_put_block
from osprey_connectors.errors import ClientLibraryRefusedError
from tests.runtime._patch_restore import restore_patches  # noqa: F401

# Every test restores what the block patched and drops its finders.
pytestmark = pytest.mark.usefixtures("restore_patches")

_FAKE = "osprey_fake_client"
_FAKE_LIBRARY = "libosprey_fake_client"


def _contract(**overrides):
    contract = {
        "denied_packages": (_FAKE,),
        "library_names": (_FAKE_LIBRARY,),
        "library_symbols": ("osprey_no_such_symbol",),
        "loader_targets": (),
    }
    contract.update(overrides)
    return contract


@pytest.fixture
def installed(restore_patches):  # noqa: ARG001, F811 - snapshot taken before the install
    raw_put_block.install("no_client", **_contract())


class _Recorder:
    """Stands in for a loader: records the call, then fails like a missing file."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, name, *args, **kwargs):
        self.calls.append(name)
        raise OSError(f"{name}: no such file")


@pytest.fixture
def recorded_loaders(monkeypatch, restore_patches):  # noqa: ARG001, F811 - snapshot first
    """Replace the ``ctypes`` loaders with recorders before the block wraps them."""
    recorder = _Recorder()
    monkeypatch.setattr(ctypes, "_dlopen", recorder)
    monkeypatch.setattr(_ctypes, "dlopen", recorder)
    return recorder


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=180,
    )


_INSTALL = """
    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import no_client_contract

    raw_put_block.install("no_client", **no_client_contract())
"""


def test_refusal_is_a_runtime_error_not_an_import_error():
    error = ClientLibraryRefusedError(package="epics")
    assert isinstance(error, RuntimeError)
    assert not isinstance(error, ImportError)
    assert error.reason == "RAW_CLIENT_IMPORT"
    assert error.package == "epics"
    assert error.library is None


def test_refusal_message_names_the_runtime_route():
    by_package = str(ClientLibraryRefusedError(package="p4p"))
    by_library = str(ClientLibraryRefusedError(library="libca.so.7"))
    for message in (by_package, by_library):
        assert message.startswith(ClientLibraryRefusedError.MARKER)
        assert "osprey.runtime" in message
        for call in ("read_channel(address)", "write_channel(address, value)", "write_channels("):
            assert call in message
    assert "'p4p'" in by_package
    assert "'libca.so.7'" in by_library
    assert "client library" in by_library


def test_contract_is_literals_from_the_one_table(monkeypatch):
    from osprey.services.python_executor.execution import wrapper
    from osprey.services.python_executor.write_surface import (
        _CLIENT_LIBRARY_NAMES,
        _CLIENT_LIBRARY_SYMBOLS,
        _CLIENT_LOADER_ROWS,
        CLIENT_PACKAGES,
    )

    contract = wrapper.no_client_contract()
    assert contract == {
        "denied_packages": tuple(sorted(CLIENT_PACKAGES)),
        "library_names": _CLIENT_LIBRARY_NAMES,
        "library_symbols": _CLIENT_LIBRARY_SYMBOLS,
        "loader_targets": _CLIENT_LOADER_ROWS,
    }
    # Literals only: the contract survives a round trip through its repr.
    assert eval(repr(contract)) == contract

    # Read at call time, so a test can narrow the table.
    monkeypatch.setattr(wrapper, "CLIENT_PACKAGES", frozenset({"osprey_fake_client"}))
    assert wrapper.no_client_contract()["denied_packages"] == ("osprey_fake_client",)


# --- import refusal ---------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(lambda: exec(f"import {_FAKE}", {}), id="import"),
        pytest.param(lambda: importlib.import_module(f"{_FAKE}.sub"), id="import_module-submodule"),
        pytest.param(lambda: __import__(_FAKE), id="__import__"),
        pytest.param(lambda: importlib.util.find_spec(_FAKE), id="find_spec"),
    ],
)
@pytest.mark.usefixtures("installed")
def test_every_denied_name_and_submodule_is_refused(spelling):
    with pytest.raises(ClientLibraryRefusedError) as info:
        spelling()
    assert info.value.package == _FAKE
    assert _FAKE not in sys.modules


@pytest.mark.usefixtures("installed")
def test_a_non_denied_package_imports(tmp_path, monkeypatch):
    allowed = "osprey_fake_allowed"
    (tmp_path / allowed).mkdir()
    (tmp_path / allowed / "__init__.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, allowed, raising=False)
    try:
        assert importlib.import_module(allowed).VALUE == 1
    finally:
        sys.modules.pop(allowed, None)


def test_install_raises_when_a_denied_package_is_already_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, f"{_FAKE}.sub", ModuleType(f"{_FAKE}.sub"))
    with pytest.raises(ClientLibraryRefusedError) as info:
        raw_put_block.install("no_client", **_contract())
    assert info.value.package == _FAKE


# --- shared-library refusal -------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(lambda path: ctypes.CDLL(path), id="CDLL"),
        pytest.param(lambda path: ctypes.PyDLL(path), id="PyDLL"),
        pytest.param(lambda path: ctypes.cdll.LoadLibrary(path), id="cdll.LoadLibrary"),
        pytest.param(lambda path: ctypes.cdll.libca, id="cdll.attr"),
        pytest.param(lambda path: ctypes.cdll._dlltype(path), id="cdll._dlltype"),
        pytest.param(lambda path: _ctypes.dlopen(path, ctypes.RTLD_LOCAL), id="_ctypes.dlopen"),
    ],
)
def test_ctypes_spellings_are_refused_by_name_before_loading(recorded_loaders, tmp_path, spelling):
    raw_put_block.install("no_client", **_contract(library_names=("libca",)))
    path = str(tmp_path / "libca.so.7")
    with pytest.raises(ClientLibraryRefusedError) as info:
        spelling(path)
    assert info.value.library in ("libca.so.7", "libca")
    assert recorded_loaders.calls == []


@pytest.mark.parametrize("basename", ["libcairo.so", "libcap.so.2", "libcalc.dylib"])
def test_a_library_sharing_a_prefix_loads(recorded_loaders, tmp_path, basename):
    raw_put_block.install("no_client", **_contract(library_names=("libca",)))
    path = str(tmp_path / basename)
    with pytest.raises(OSError, match="no such file"):
        ctypes.CDLL(path)
    assert recorded_loaders.calls == [path]


@pytest.mark.parametrize(
    "basename",
    ["libca.so", "libca.so.3.15", "libca.3.15.dylib", "LIBCA.DYLIB", "ca.dll", "libpvAccess.so.7"],
)
def test_a_stem_matches_its_versioned_basenames(recorded_loaders, tmp_path, basename):
    raw_put_block.install(
        "no_client", **_contract(library_names=("libca", "ca.dll", "libpvAccess"))
    )
    with pytest.raises(ClientLibraryRefusedError):
        ctypes.CDLL(str(tmp_path / basename))
    assert recorded_loaders.calls == []


def _libc() -> str:
    return ctypes.util.find_library("c") or "libSystem.dylib"


@pytest.mark.usefixtures("installed")
def test_an_unrelated_library_loads():
    assert ctypes.CDLL(_libc()).strlen(b"abc") == 3


def test_a_library_exporting_a_client_symbol_is_refused():
    raw_put_block.install("no_client", **_contract(library_symbols=("strlen",)))
    with pytest.raises(ClientLibraryRefusedError) as info:
        ctypes.CDLL(_libc())
    assert info.value.library is not None


def test_a_second_install_refuses_the_same_and_wraps_nothing_twice():
    raw_put_block.install("no_client", **_contract())
    first = (ctypes._dlopen, _ctypes.dlopen)
    raw_put_block.install("no_client", **_contract())
    assert (ctypes._dlopen, _ctypes.dlopen) == first
    finders = [f for f in sys.meta_path if type(f).__name__ == "_OspreyNoClientFinder"]
    assert len(finders) == 2
    for finder in finders:
        with pytest.raises(ClientLibraryRefusedError):
            finder.find_spec(_FAKE, None)
    with pytest.raises(ClientLibraryRefusedError):
        importlib.import_module(_FAKE)


# --- real client libraries, in a subprocess ---------------------------------


def test_a_renamed_libca_is_refused_by_symbol(tmp_path):
    epicscorelibs_path = pytest.importorskip("epicscorelibs.path")
    libca = Path(epicscorelibs_path.get_lib("ca"))
    # libca resolves libCom beside itself, so the copy takes it along.
    for sibling in libca.parent.iterdir():
        if sibling.name.startswith("libCom"):
            shutil.copy2(sibling, tmp_path / sibling.name)
    copy = tmp_path / f"libnothing{libca.suffix}"
    shutil.copy2(libca, copy)

    result = _run(
        _INSTALL
        + f"""
    import ctypes
    from osprey_connectors.errors import ClientLibraryRefusedError

    try:
        ctypes.CDLL({str(copy)!r})
    except ClientLibraryRefusedError as error:
        print("REFUSED", error.library)
    else:
        print("LOADED")
    """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"REFUSED {copy.name}", result.stdout


def test_cffi_dlopen_is_gated(tmp_path):
    pytest.importorskip("cffi")
    result = _run(
        _INSTALL
        + f"""
    import cffi
    from osprey_connectors.errors import ClientLibraryRefusedError

    try:
        cffi.FFI().dlopen({str(tmp_path / "libca.so.7")!r})
    except ClientLibraryRefusedError as error:
        print("REFUSED", error.library)
    """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REFUSED libca.so.7", result.stdout


def test_osprey_layer_imports_and_client_frameworks_are_refused():
    result = _run(
        _INSTALL
        + """
    import importlib
    import sys

    from osprey.services.python_executor.write_surface import CLIENT_PACKAGES
    from osprey_connectors.errors import ClientLibraryRefusedError

    for name in (
        "osprey.runtime",
        "osprey_connectors.ipc.pool",
        "osprey.registry",
        "osprey.services.bluesky_bridge.devices.connector",
        "osprey.jupyter_kernel",
        "ipykernel.kernelapp",
        "bluesky",
        "ophyd_async.core",
    ):
        importlib.import_module(name)
        print("IMPORTED", name)

    for name in ("ophyd", "ophyd_async.epics.core"):
        try:
            importlib.import_module(name)
        except ClientLibraryRefusedError as error:
            assert error.package in CLIENT_PACKAGES, error.package
            print("REFUSED", name)
        else:
            print("LOADED", name)

    from IPython.core.completerlib import module_completion

    print("COMPLETION", module_completion("from epics import "))
    loaded = sorted(
        name for name in sys.modules if name.split(".")[0] in CLIENT_PACKAGES
    )
    print("CLIENTS", loaded)
    """
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for name in (
        "osprey.runtime",
        "osprey_connectors.ipc.pool",
        "osprey.registry",
        "osprey.services.bluesky_bridge.devices.connector",
        "osprey.jupyter_kernel",
        "ipykernel.kernelapp",
        "bluesky",
        "ophyd_async.core",
    ):
        assert f"IMPORTED {name}" in lines, result.stdout
    assert "REFUSED ophyd" in lines, result.stdout
    assert "REFUSED ophyd_async.epics.core" in lines, result.stdout
    assert "COMPLETION []" in lines, result.stdout
    assert "CLIENTS []" in lines, result.stdout

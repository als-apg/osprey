"""The shared-library loaders a readonly run gates, against the real ``libca``.

A readonly run gates the loader every ``ctypes`` spelling funnels into and
cffi's ``FFI.dlopen`` beside it, and every load through them refuses. Each case
installs the block the executor emits for a readonly run, in a subprocess, so
``libca`` never loads in the test process itself.
"""

from __future__ import annotations

import ctypes.util
import subprocess
import sys
import textwrap

import pytest

from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

_INSTALL = """
    import os
    import sys

    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    cffi_before = "cffi" in sys.modules
    exec(ExecutionWrapper(execution_mode="readonly")._get_readonly_guard(), {})


    def attempt(label, call):
        try:
            call()
        except RuntimeError as error:
            print(label, "REFUSED", str(error))
        else:
            print(label, "ALLOWED")
"""

# The process leaves without running exit handlers: a Channel Access client's
# own would tear a context down, which has nothing to do with these checks.
_EXIT = """
    sys.stdout.flush()
    os._exit(0)
"""


def _run(script: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_INSTALL + script + _EXIT)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()


def _libca_path() -> str:
    epicscorelibs_path = pytest.importorskip("epicscorelibs.path")
    return str(epicscorelibs_path.get_lib("ca"))


def _assert_refused(line: str, label: str) -> None:
    assert line.startswith(f"{label} REFUSED"), line
    assert READONLY_REFUSAL_MARKER in line, line


def test_readonly_refuses_ctypes_dlopen_of_libca():
    libca = _libca_path()

    lines = _run(
        f"""
    import _ctypes

    attempt("dlopen", lambda: _ctypes.dlopen({libca!r}))
    """
    )

    _assert_refused(lines[-1], "dlopen")


def test_readonly_refuses_cffi_dlopen_of_libca():
    pytest.importorskip("cffi")
    libca = _libca_path()

    lines = _run(
        f"""
    import cffi

    attempt("cffi", lambda: cffi.FFI().dlopen({libca!r}))
    """
    )

    _assert_refused(lines[-1], "cffi")


def test_readonly_install_does_not_import_cffi():
    lines = _run(
        """
    print("BEFORE", cffi_before, "AFTER", "cffi" in sys.modules)
    """
    )

    assert lines[-1] == "BEFORE False AFTER False", lines


_HOLDS_A_LOADER = """
    import importlib.util

    from osprey.runtime import raw_put_block


    def report_loader_rows():
        rows = raw_put_block._CTYPES_LOADER_ROWS | raw_put_block._FFI_LOADER_ROWS
        for dotted, attr in sorted(rows):
            if importlib.util.find_spec(dotted.split(".")[0]) is None:
                continue
            owner = raw_put_block._resolve(dotted)
            if owner is None or not hasattr(owner, attr):
                continue
            loader = getattr(owner, attr)
            gated = bool(getattr(loader, "_osprey_loader_gate", False))
            holds = any(callable(cell.cell_contents) for cell in loader.__closure__ or ())
            print("ROW", f"{dotted}.{attr}", "GATED", gated, "HOLDS", holds)
"""


def _loader_rows(lines: list[str]) -> dict[str, tuple[str, str]]:
    rows = {}
    for line in lines:
        if line.startswith("ROW "):
            _, row, _, gated, _, holds = line.split()
            rows[row] = (gated, holds)
    assert "_ctypes.dlopen" in rows, lines
    return rows


def test_no_client_wraps_the_loader_while_readonly_holds_none():
    """The no_client gate lets other libraries load; a readonly run lets none."""
    libc = ctypes.util.find_library("c") or "libSystem.dylib"

    no_client = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
    import ctypes

    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import no_client_contract

    raw_put_block.install("no_client", **no_client_contract())
"""
                + _HOLDS_A_LOADER
                + f"""
    print("LOADS", ctypes.CDLL({libc!r}).strlen(b"abc") == 3)
    report_loader_rows()
"""
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert no_client.returncode == 0, no_client.stderr
    no_client_lines = no_client.stdout.strip().splitlines()
    assert no_client_lines[0] == "LOADS True", no_client.stdout
    for row, state in _loader_rows(no_client_lines).items():
        assert state == ("True", "True"), (row, no_client.stdout)

    lines = _run(
        _HOLDS_A_LOADER
        + f"""
    import _ctypes

    attempt("dlopen", lambda: _ctypes.dlopen({libc!r}))
    report_loader_rows()
    """
    )

    _assert_refused(next(line for line in lines if line.startswith("dlopen ")), "dlopen")
    for row, state in _loader_rows(lines).items():
        assert state == ("True", "False"), (row, lines)

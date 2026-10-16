"""The shared-library loaders a readonly run gates, against the real ``libca``.

A readonly run gates the loader every ``ctypes`` spelling funnels into and
cffi's ``FFI.dlopen`` beside it: every load refuses except pyepics' own load of
``libca``, whose put entry points stay closed. Each case installs the block the
executor emits for a readonly run, in a subprocess, so ``libca`` never loads in
the test process itself.
"""

from __future__ import annotations

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

# The process leaves without running exit handlers: pyepics' own would tear a
# Channel Access context down, which has nothing to do with these checks.
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


def test_pyepics_reads_through_the_funnel():
    pytest.importorskip("epics")

    lines = _run(
        """
    import epics.ca

    print("HANDLE", epics.ca.initialize_libca() is not None)
    attempt("put", lambda: epics.ca.libca.ca_array_put)
    """
    )

    assert lines[-2] == "HANDLE True", lines
    _assert_refused(lines[-1], "put")

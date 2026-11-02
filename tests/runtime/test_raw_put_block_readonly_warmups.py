"""The stdlib resolutions a readonly run warms before its gate closes.

A readonly run refuses every shared-library load and every subprocess once it is
installed. Some stdlib lookups load native code or spawn a process lazily, on
first use, and libraries a readonly script may import (the simulation stack
among them) make those lookups while they initialise. The guard resolves each
row of ``_STDLIB_WARMUPS`` before it installs, so none of them refuses later.
Each case installs the block the executor emits for a readonly run, in a
subprocess, so the gate never closes in the test process itself.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

_INSTALL = """
    import os
    import sys

    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    exec(ExecutionWrapper(execution_mode="readonly")._get_readonly_guard(), {})


    def attempt(label, call):
        try:
            call()
        except Exception as error:
            print(label, "REFUSED", type(error).__name__, str(error))
        else:
            print(label, "ALLOWED")
"""

# The process leaves without running exit handlers, so nothing an exit handler
# does under the gate decides the result.
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


def _assert_allowed(lines: list[str], label: str) -> None:
    line = next((line for line in lines if line.startswith(f"{label} ")), None)
    assert line == f"{label} ALLOWED", lines
    assert not any(READONLY_REFUSAL_MARKER in line for line in lines), lines


def test_readonly_gate_lets_ctypes_util_import():
    lines = _run(
        """
    def import_ctypes_util():
        import ctypes.util

        if sys.platform == "darwin":
            assert ctypes.util.find_library("c")

    attempt("ctypes.util", import_ctypes_util)
    """
    )

    _assert_allowed(lines, "ctypes.util")


def test_readonly_gate_lets_at_import():
    lines = _run(
        """
    import importlib.util

    def import_simulation_stack():
        import at
        import scipy.io

    if importlib.util.find_spec("at") is None:
        print("SKIP")
    else:
        attempt("at", import_simulation_stack)
    """
    )

    if lines[-1] == "SKIP":
        return
    _assert_allowed(lines, "at")


# Each warm-up row, and how a script under the gate observes that it resolved.
# A processor lookup that was not resolved before install shells out to
# ``uname -p``, which the gate refuses; one that was answers from its cache.
_RESOLVED = """
    import importlib

    probes = {
        "platform.processor": lambda: importlib.import_module("platform").processor(),
        "ctypes.util": lambda: sys.modules["ctypes.util"],
    }
    for label, _warm in raw_put_block._STDLIB_WARMUPS:
        attempt(label, probes[label])
    print("LABELS", sorted(label for label, _ in raw_put_block._STDLIB_WARMUPS))
    print("PROBES", sorted(probes))
"""


def test_every_warmup_runs_before_the_gate():
    lines = _run(_RESOLVED)

    labels = next(line for line in lines if line.startswith("LABELS "))
    probes = next(line for line in lines if line.startswith("PROBES "))
    assert labels.removeprefix("LABELS ") == probes.removeprefix("PROBES "), lines
    for line in lines:
        if line.startswith(("LABELS ", "PROBES ")):
            continue
        assert line.endswith(" ALLOWED"), lines

"""The sandbox process ends when its script and persistence are done.

A control-system client holds native state whose interpreter-shutdown hooks
can block or crash the process — pyepics' ``finalize_libca`` wedges after
Channel Access was used from a worker thread, which the EPICS connector always
does. The executor only sees a child that never closes its pipes, and reports a
run that finished in a second as a timeout ten minutes later. The wrapper
therefore leaves the way ``osprey_connectors.ipc.host`` does: streams flushed,
outputs persisted, then ``os._exit`` — no shutdown hook gets a say.

The stand-in for the wedging hook is an ``atexit`` handler registered by the
user code itself: exactly as blocking, and it needs no control system.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

_SRC_ROOT = Path(__file__).resolve().parents[4] / "src"

#: Longer than the assertion below, shorter than any patience a CI runner has.
_HOOK_BLOCK_SECONDS = 45


def _run_wrapper(
    tmp_path: Path, user_code: str, *, mode: str
) -> tuple[subprocess.CompletedProcess, Path]:
    execution_folder = tmp_path / "exec"
    execution_folder.mkdir()
    script = execution_folder / "wrapped_script.py"
    script.write_text(
        ExecutionWrapper(execution_mode=mode).create_wrapper(user_code, execution_folder),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC_ROOT), env.get("PYTHONPATH")]))
    proc = subprocess.run(  # noqa: S603 - fixed argv, generated script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=_HOOK_BLOCK_SECONDS + 30,
    )
    return proc, execution_folder


_BLOCKING_SHUTDOWN_HOOK = f"""
import atexit, time
atexit.register(lambda: time.sleep({_HOOK_BLOCK_SECONDS}))
print("script finished")
results = {{"answer": 42}}
"""


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
def test_sandbox_exits_without_waiting_for_shutdown_hooks(tmp_path, mode):
    started = time.monotonic()
    proc, folder = _run_wrapper(tmp_path, _BLOCKING_SHUTDOWN_HOOK, mode=mode)
    elapsed = time.monotonic() - started

    assert proc.returncode == 0, proc.stderr
    assert elapsed < _HOOK_BLOCK_SECONDS, f"sandbox waited on a shutdown hook for {elapsed:.0f}s"
    # The abrupt exit costs nothing the executor reads: outputs are persisted
    # and the pipes carry the script's own output.
    assert "script finished" in proc.stdout
    metadata = json.loads((folder / "execution_metadata.json").read_text())
    assert metadata["success"] is True
    assert metadata["stdout"] == "script finished\n"
    assert json.loads((folder / "results.json").read_text()) == {"answer": 42}


def test_sandbox_still_reports_a_failed_script(tmp_path):
    """The abrupt exit must not swallow the failure record of a script that raised."""
    proc, folder = _run_wrapper(tmp_path, "raise ValueError('boom')", mode="readonly")

    assert proc.returncode == 0, proc.stderr
    metadata = json.loads((folder / "execution_metadata.json").read_text())
    assert metadata["success"] is False
    assert "boom" in metadata["error"]
    assert "ValueError" in proc.stderr

"""A readwrite run holds its control target's lock from before its code to its saved record.

The sandbox wrapper of a ``readwrite`` run enters
``osprey.runtime.guarded_run.lock(None)`` before the first line of user code and
leaves it after the run's record is saved; a ``readonly`` run never touches the
lock. Each case runs a real wrapped script in a deployment repo whose
``osprey.runtime`` reads and writes a dict machine kept in a JSON file (the
sandbox of ``test_approved_call_forwarding``).

Covered elsewhere and kept green by this module's gate set: a pyAML tool run
inside a readwrite run joins the run's lock
(``tests/pyaml_cs_osprey/test_run_tool.py``), and a killed readwrite run frees
its target
(``tests/runtime/test_guarded_run_cases.py::test_a_killed_readwrite_execute_frees_the_target``).
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from tests._guarded_run import held_lock, lock_path
from tests.mcp_server.python_executor.test_approved_call_forwarding import (
    _SRC_ROOT,
    CHILD_TIMEOUT_S,
    _Sandbox,
)

#: Where user code finds the target's lock file and the file it would write.
ENV_LOCK_PATH = "HOLDS_LOCK_PATH"
ENV_SENTINEL = "HOLDS_LOCK_SENTINEL"

#: User code that asks a second process for the lock and reads the holder record.
_PROBE_THE_LOCK = textwrap.dedent(
    f"""
    import json, os, subprocess, sys

    _try = (
        "import fcntl, os, sys\\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\\n"
        "try:\\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\\n"
        "except BlockingIOError:\\n"
        "    print('BlockingIOError')\\n"
        "else:\\n"
        "    print('acquired')\\n"
    )
    _lock_path = os.environ[{ENV_LOCK_PATH!r}]
    _second = subprocess.run(
        [sys.executable, "-c", _try, _lock_path], capture_output=True, text=True, check=True
    )
    with open(_lock_path) as handle:
        _holder = json.load(handle)
    with open(os.environ["APPROVED_CALL_VALUES"]) as handle:
        _seen = json.load(handle)
    results = {{
        "second_process": _second.stdout.strip(),
        "holder_pid": _holder["pid"],
        "run_pid": os.getpid(),
        "seen": _seen,
    }}
    """
)

#: User code that leaves a mark that it ran.
_WRITE_SENTINEL = textwrap.dedent(
    f"""
    import os

    with open(os.environ[{ENV_SENTINEL!r}], "w") as handle:
        handle.write("ran")
    results = {{"ran": True}}
    """
)


def _lock_path(sandbox: _Sandbox) -> Path:
    return lock_path("live", sandbox.root)


def _run(
    sandbox: _Sandbox, mode: str, code: str, env_extra: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run *code* wrapped for *mode* in *sandbox*; return its record, with any results."""
    script = sandbox.folder / "wrapped_script.py"
    script.write_text(
        ExecutionWrapper(execution_mode=mode).create_wrapper(code, sandbox.folder),
        encoding="utf-8",
    )
    for name in ("execution_metadata.json", "results.json"):
        (sandbox.folder / name).unlink(missing_ok=True)
    env = os.environ.copy()
    for name in ("OSPREY_CONFIG", "CONFIG_FILE"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(sandbox.site), str(_SRC_ROOT), env.get("PYTHONPATH")])
    )
    env["APPROVED_CALL_VALUES"] = str(sandbox.values_path)
    env["OSPREY_EXECUTION_MODE"] = mode
    env["OSPREY_CONTROL_TARGET"] = "live"
    env["OSPREY_CONTROL_TARGET_GENERATION"] = "4"
    env["OSPREY_AGENT_DATA_ROOT"] = str(sandbox.root / "agent_data")
    env[ENV_LOCK_PATH] = str(_lock_path(sandbox))
    env[ENV_SENTINEL] = str(sandbox.root / "sentinel")
    env.update(env_extra or {})
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(sandbox.root),
        env=env,
        capture_output=True,
        timeout=CHILD_TIMEOUT_S,
        check=False,
    )
    metadata = json.loads((sandbox.folder / "execution_metadata.json").read_text())
    results_path = sandbox.folder / "results.json"
    if results_path.exists():
        metadata["_results"] = json.loads(results_path.read_text())
    return metadata


def test_a_readwrite_runs_code_finds_the_lock_held_by_the_run(tmp_path: Path) -> None:
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 3.0})

    metadata = _run(sandbox, "readwrite", _PROBE_THE_LOCK)

    assert metadata["success"] is True, metadata.get("traceback")
    results = metadata["_results"]
    assert results["second_process"] == "BlockingIOError"
    assert results["holder_pid"] == results["run_pid"]


def test_the_lock_is_free_after_the_run_and_a_second_run_takes_it(tmp_path: Path) -> None:
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 3.0})

    first = _run(sandbox, "readwrite", _PROBE_THE_LOCK)
    assert first["success"] is True, first.get("traceback")

    fd = os.open(_lock_path(sandbox), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.pread(fd, 4096, 0) == b"", "the holder record is emptied on release"
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

    second = _run(sandbox, "readwrite", _PROBE_THE_LOCK)
    assert second["success"] is True, second.get("traceback")
    assert second["_results"]["holder_pid"] == second["_results"]["run_pid"]


def test_a_readonly_run_never_touches_the_lock(tmp_path: Path) -> None:
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 3.0})

    metadata = _run(sandbox, "readonly", _WRITE_SENTINEL)

    assert metadata["success"] is True, metadata.get("traceback")
    assert (sandbox.root / "sentinel").exists()
    assert not _lock_path(sandbox).exists()


def test_a_readwrite_run_on_a_held_target_fails_before_its_code(tmp_path: Path) -> None:
    sandbox = _Sandbox(tmp_path / "repo", {"Q": 3.0})
    with held_lock("live", root=sandbox.root) as holder_process:
        holder = json.loads(_lock_path(sandbox).read_text(encoding="utf-8"))

        metadata = _run(sandbox, "readwrite", _WRITE_SENTINEL)

    assert metadata["success"] is False
    assert metadata["error_type"] == "OspreyRunBusy"
    assert not (sandbox.root / "sentinel").exists(), "the user code never started"
    assert f"pid {holder_process.pid}" in metadata["error"]
    assert holder["started"] in metadata["error"]

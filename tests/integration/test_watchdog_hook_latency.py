"""The watchdog trigger costs a prompt at most 100 ms, even with a hung server.

Each hook run is paired with a no-op hook run started by the same interpreter, so
the interpreter's own start-up cancels out. The median of the paired differences
is the claim. A wall-clock budget, so it runs alone (``wallclock``).
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_watchdog_trigger as trigger

BUDGET_SECONDS = 0.100
PAIRS = 15

NOOP_HOOK = "import json, sys\njson.load(sys.stdin)\nsys.exit(0)\n"

#: A record entry that does nothing, so the lingering children stay cheap.
EMPTY_RECORD_ENTRY = ""


def _timed(argv: list[str], stdin: str, env: dict[str, str]) -> float:
    started = time.perf_counter()
    done = subprocess.run(argv, input=stdin, capture_output=True, text=True, env=env, timeout=30)
    elapsed = time.perf_counter() - started
    assert done.returncode == 0
    assert done.stdout == ""
    return elapsed


@pytest.mark.wallclock
def test_the_hook_adds_at_most_100_ms_with_a_hung_server(tmp_path: Path) -> None:
    watchdog = tmp_path / "stub" / "osprey" / "services" / "watchdog"
    watchdog.mkdir(parents=True)
    for package in (watchdog, watchdog.parent, watchdog.parent.parent):
        (package / "__init__.py").write_text("")
    (watchdog / "__main__.py").write_text(EMPTY_RECORD_ENTRY)
    noop = tmp_path / "noop.py"
    noop.write_text(NOOP_HOOK)
    scratch = tmp_path / "tmp"
    scratch.mkdir()

    stdin = json.dumps({"session_id": "s", "prompt": "raise the limit"})
    # Listening and never accepting: a connection lands in the backlog and hangs.
    with socket.socket() as hung:
        hung.bind(("127.0.0.1", 0))
        hung.listen(128)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "TMPDIR": str(scratch),
            "PYTHONPATH": str(tmp_path / "stub"),
            "CLAUDE_PROJECT_DIR": str(tmp_path),
            "OSPREY_WEB_PORT": str(hung.getsockname()[1]),
            "OSPREY_SESSION_ID": "pool-key-1",
        }
        hook_argv = [sys.executable, trigger.__file__]
        noop_argv = [sys.executable, str(noop)]
        deltas = [
            _timed(hook_argv, stdin, env) - _timed(noop_argv, stdin, env) for _ in range(PAIRS)
        ]

    assert statistics.median(deltas) <= BUDGET_SECONDS, sorted(deltas)

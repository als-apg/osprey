"""Every connector handle is a ``ControlSystem`` to the type checker.

A consumer types the connector it holds as
:class:`osprey_connectors.control_system.ControlSystem`, and is handed one of
three objects: the in-process :class:`ControlSystemConnector`, the
connector-host :class:`ConnectorHostProxy`, or a pool's
:class:`PooledConnector`. The runtime half of that promise (the same
parameters, kinds and defaults) is the parametrised signature test beside the
proxy's. This is the static half: mypy accepts each of the three where a
``ControlSystem`` is asked for, a ``ControlSystem`` where settle's
``ChannelReader`` is asked for, and types the batch read as a
``ChannelReadBatch`` through every one of them. A class that only looks like a
connector is refused, which is what keeps the accepting snippet from passing
for the wrong reason.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import osprey_connectors

_CONNECTORS_SRC = Path(osprey_connectors.__file__).parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]

_ACCEPTED = """\
from typing import assert_type

from osprey_connectors.control_system import (
    ChannelReadBatch,
    ControlSystem,
    ControlSystemConnector,
)
from osprey_connectors.ipc import ConnectorHostProxy, PooledConnector
from osprey_connectors.settle import ChannelReader


async def every_handle(
    connector: ControlSystemConnector, proxy: ConnectorHostProxy, pooled: PooledConnector
) -> None:
    handles: list[ControlSystem] = [connector, proxy, pooled]
    readers: list[ChannelReader] = [*handles]
    for handle in handles:
        assert_type(await handle.read_multiple_channels(["SR:A"]), ChannelReadBatch)
    assert_type(await pooled.read_multiple_channels(["SR:A"]), ChannelReadBatch)
    assert_type(await proxy.read_multiple_channels(["SR:A"]), ChannelReadBatch)
    del readers
"""

_REFUSED = """\
from osprey_connectors.control_system import ControlSystem


class OnlyReads:
    async def read_channel(self, address: str, *, timeout_s: float | None = None) -> object:
        return None


handle: ControlSystem = OnlyReads()
"""


@pytest.fixture(scope="module")
def mypy_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Both snippets checked in one mypy pass; the process start dominates the cost."""
    snippet_dir = tmp_path_factory.mktemp("control_system_typing")
    (snippet_dir / "accepted.py").write_text(_ACCEPTED)
    (snippet_dir / "refused.py").write_text(_REFUSED)

    env = dict(os.environ)
    env["MYPYPATH"] = str(_CONNECTORS_SRC)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(_REPO_ROOT / "pyproject.toml"),
            "--follow-imports=silent",
            str(snippet_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
        timeout=600,
    )
    assert "No module named mypy" not in result.stderr, (
        "mypy is not installed in this environment, so the static half of this "
        f"contract cannot be checked at all:\n{result.stderr}"
    )
    return result.stdout + result.stderr


def _errors_in(output: str, snippet: str) -> list[str]:
    return [line for line in output.splitlines() if snippet in line and "error:" in line]


def test_every_handle_is_a_control_system_and_reads_a_batch(mypy_output: str) -> None:
    assert _errors_in(mypy_output, "accepted.py") == [], mypy_output


def test_an_object_that_only_reads_is_no_control_system(mypy_output: str) -> None:
    errors = _errors_in(mypy_output, "refused.py")
    assert errors, f"the refused snippet type-checked clean:\n{mypy_output}"
    assert any('"ControlSystem"' in line for line in errors), mypy_output

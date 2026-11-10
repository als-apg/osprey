"""An in-flight marker writer that reaches the identity's state directory first makes it shared.

The three marker writers (an execution, a notebook cell, a session write) put
their ``0640`` marker in :func:`target_state.state_dir`, the directory the
control-context record and the session ledger share. A CLI or a script with no
controls server can be the first to reach it; a directory left at the umask
then shuts out every other uid's reader, and the record that comes later finds
it existing and sets no mode.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import osprey.runtime as runtime
from osprey.audit import posture
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor
from osprey.runtime.session import ControlSession


def _markers(directory: Path) -> list[Path]:
    return sorted(directory.glob(target_state.INFLIGHT_FILE_GLOB))


def _execution(directory: Path) -> list[Path]:
    # The execution's marker lives only while the run does.
    with executor._in_flight_marker("live"):
        return _markers(directory)


def _cell(directory: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    monkeypatch.setenv(runtime.ENV_IN_CELL, "1")
    monkeypatch.setattr(posture, "posture_session", lambda: f"{runtime.KERNEL_SESSION_PREFIX}k1")
    monkeypatch.setattr(runtime, "_stamped_target", lambda: "live")
    monkeypatch.setattr(runtime, "_cell_marker", None)
    runtime._claim_cell()
    return _markers(directory)


def _session_write(directory: Path) -> list[Path]:
    session = SimpleNamespace(
        _audit_session=lambda: "s1",
        _surface="panel",
        _client_label="a panel write",
        _binding=SimpleNamespace(target="live"),
    )
    ControlSession._write_marker(session, "SR:BEAM")  # type: ignore[arg-type]
    return _markers(directory)


@pytest.mark.parametrize("writer", ["execution", "cell", "session_write"])
def test_a_marker_first_identity_directory_is_group_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    tree = tmp_path / "state"
    tree.mkdir()
    os.chmod(tree, 0o2770)
    identity = tree / "alice"
    monkeypatch.setattr(target_state, "state_dir", lambda: identity)

    previous = os.umask(0o077)
    try:
        if writer == "execution":
            markers = _execution(identity)
        elif writer == "cell":
            markers = _cell(identity, monkeypatch)
        else:
            markers = _session_write(identity)
    finally:
        os.umask(previous)

    assert markers, "the writer wrote its marker"
    assert stat.S_IMODE(identity.stat().st_mode) == 0o2770
    assert identity.stat().st_gid == tree.stat().st_gid

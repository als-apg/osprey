"""Opt-in approval stamps for tests that call a stage-1 tool directly.

A tool whose approval policy asks runs only on the record the approval hook
filed for the call (``osprey.mcp_server.approval_record``). A test that calls
such a tool without the hook writes that record itself::

    from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir

    @pytest.fixture
    def approval_stamps(tmp_path, monkeypatch) -> ApprovalStamps:
        return use_stamp_dir(monkeypatch, tmp_path / "state")

    def test_an_approved_write(approval_stamps):
        with approval_stamps.approved("channel_write", operations=ops):
            ...  # call the tool

:func:`approved` files ``approval_ask_<tool_use_id>.json`` and, for
``channel_write`` given its operations, the ``write_approval_<slug>_<key>.json``
stamp that tool reads, then binds the call in :func:`~osprey.audit.call.call_scope`.
Nothing here is autouse. :func:`approval_stamps` is the same fixture, ready to
import by a module that prefers it.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

#: The MCP server each short tool name is served by, for the full name the
#: approval hook records in an ask stamp.
TOOL_SERVERS: dict[str, str] = {
    "execute": "python",
    "execute_file": "python",
    "pyaml_measure": "python",
    "channel_write": "controls",
    "queue_start": "bluesky",
    "queue_add": "bluesky",
    "phoebus_drive": "phoebus",
}

#: The tool-use id :meth:`ApprovalStamps.approved` binds when given none.
DEFAULT_TOOL_USE_ID = "toolu_stamped"


def full_tool_name(tool: str) -> str:
    """The ``mcp__<server>__<tool>`` name the hook records for *tool*.

    A name that already starts with ``mcp__`` is returned as given.
    """
    if tool.startswith("mcp__"):
        return tool
    return f"mcp__{TOOL_SERVERS[tool]}__{tool}"


@dataclass
class ApprovalStamps:
    """Writes approval stamps into one state directory."""

    state: Path

    def ask(
        self,
        tool: str,
        *,
        tool_use_id: str = DEFAULT_TOOL_USE_ID,
        approver: str = "human",
        **fields: Any,
    ) -> Path:
        """File the ask stamp for *tool_use_id*, recording *tool* and *fields*."""
        from osprey.mcp_server.approval_record import APPROVAL_ASK_PREFIX, APPROVAL_ASK_SUFFIX

        payload = {
            "tool_use_id": tool_use_id,
            "tool": full_tool_name(tool),
            "approver": approver,
            **fields,
        }
        path = self.state / f"{APPROVAL_ASK_PREFIX}{tool_use_id}{APPROVAL_ASK_SUFFIX}"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_approval(
        self,
        operations: list[dict],
        confirm: bool | None = None,
        *,
        target: str | None = None,
        generation: int | None = None,
    ) -> Path:
        """File the stamp ``channel_write`` reads for *operations* under this session."""
        from osprey.audit.posture import posture_session
        from osprey.mcp_server.control_system.tools import channel_write as cw

        key = cw._approval_stamp_key(operations, confirm)
        assert key is not None, "a write-approval stamp needs operations"
        payload = {
            "tool": "channel_write",
            "key": key,
            "target": target,
            "generation": generation,
            "session": posture_session(),
            "rendered_at": "2026-01-01T00:00:00+00:00",
        }
        path = self.state / cw._approval_stamp_name(key)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @contextlib.contextmanager
    def approved(
        self,
        tool: str,
        *,
        tool_use_id: str = DEFAULT_TOOL_USE_ID,
        session_id: str | None = None,
        approver: str = "human",
        operations: list[dict] | None = None,
        confirm: bool | None = None,
        target: str | None = None,
        generation: int | None = None,
        **fields: Any,
    ) -> Iterator[None]:
        """Stamp an approved call of *tool* and bind it for the body.

        With *operations* (``channel_write``), the write-approval stamp is
        filed too, bound to *target* and *generation*.
        """
        from osprey.audit.call import call_scope

        self.ask(tool, tool_use_id=tool_use_id, approver=approver, **fields)
        if operations is not None:
            self.write_approval(operations, confirm, target=target, generation=generation)
        with call_scope(tool_use_id, session_id):
            yield


def use_stamp_dir(monkeypatch: pytest.MonkeyPatch, state: Path) -> ApprovalStamps:
    """Point the identity state directory at *state* for this test, and stamp into it."""
    from osprey.mcp_server.control_system import target_state

    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(target_state, "state_dir", lambda: state)
    return ApprovalStamps(state)


@pytest.fixture
def approval_stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApprovalStamps:
    """Approval stamps filed in a fresh state directory under *tmp_path*."""
    return use_stamp_dir(monkeypatch, tmp_path / "state")

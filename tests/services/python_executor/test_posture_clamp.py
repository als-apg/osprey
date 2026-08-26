"""Tests for the per-session posture clamp in the executor's execution gates.

A web-terminal session switched to the sandbox posture spawns its child with
``OSPREY_EXECUTION_MODE=readonly``, and the MCP servers launched under that
session inherit it. The clamp is what makes the executor honour that posture:
an agent asking for ``execution_mode="readwrite"`` inside a sandboxed session
must be refused even though the *deployment* allows writes.

The semantics pinned here mirror ``osprey_connectors``' ``is_readonly_run``:
a **value** comparison, never a presence check. Any other value of the
variable — including ``"readwrite"`` and garbage — leaves both modes alone,
so the only thing that can sandbox a session is the sandbox posture itself.
"""

from __future__ import annotations

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.mcp_server.python_executor.tools._execution_gates import enforce_posture_clamp

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _audit_zone(tmp_path, monkeypatch):
    """Redirect the audit zone into ``tmp_path`` for every test in this file.

    The clamp *records* its refusal, and ``writer.audit_dir`` resolves against
    the real project root — so without this a plain ``pytest`` run appends
    refusals that never happened to the deployment's own ledger, where an
    operator cannot tell them from the real ones.

    Autouse rather than requested by the two tests that fire the clamp today: a
    test added here later inherits the redirect instead of rediscovering the
    leak the hard way.
    """
    from osprey.audit import writer

    monkeypatch.setattr(writer, "audit_dir", lambda: tmp_path / "var" / "audit")
    return tmp_path / "var" / "audit"


def _envelope(exc_info) -> dict:
    """The structured error envelope ``make_error`` packed into the ToolError."""
    return json.loads(str(exc_info.value))


def test_readwrite_refused_under_sandbox_posture(monkeypatch):
    """The whole point: writes asked for inside a sandboxed session are refused."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    envelope = _envelope(exc_info)
    assert envelope["error"] is True
    assert envelope["error_type"] == "safety_error"
    assert envelope["suggestions"]


def test_posture_refusal_names_the_posture_not_the_deployment(monkeypatch):
    """Two-vocabulary rule: nothing is wrong with the deployment config.

    Mirror of ``test_readonly_refusal_message_does_not_blame_deployment`` on the
    connector side — a posture refusal that mentions ``writes_enabled`` sends
    the operator to edit a config file that is not the gate.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    envelope = _envelope(exc_info)
    text = envelope["error_message"] + " " + " ".join(envelope["suggestions"])
    assert "writes_enabled" not in text
    assert "posture" in text.lower()
    # And it says how to leave the sandbox, from where the operator can do it.
    assert "terminal card" in text.lower()


def test_readonly_passes_under_sandbox_posture(monkeypatch):
    """A readonly run is exactly what the sandbox posture permits."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    assert enforce_posture_clamp("readonly", tool="execute") is None


def test_no_mode_var_means_not_a_sandbox_run(monkeypatch):
    """Outside a postured session the variable is unset and the clamp is inert.

    Pins the same semantics as the connector test of this name: with no
    variable, the deployment gates alone decide, and *both* modes pass here.
    """
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    assert enforce_posture_clamp("readonly", tool="execute") is None
    assert enforce_posture_clamp("readwrite", tool="execute") is None


@pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
def test_other_values_leave_both_modes_unchanged(monkeypatch, value):
    """Value comparison, not presence: only the exact ``readonly`` string clamps.

    ``readwrite`` is the writes posture and must not be mistaken for a sandbox;
    the remaining values are the ones a typo or a stale env would produce, and
    a presence check would sandbox the session on every one of them.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", value)

    assert enforce_posture_clamp("readonly", tool="execute") is None
    assert enforce_posture_clamp("readwrite", tool="execute") is None

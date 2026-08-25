"""A readonly sandbox run is refused at the connector, whatever the deployment posture.

``control_system.writes_enabled`` is the launch-time deployment posture. The
python executor additionally exports ``OSPREY_EXECUTION_MODE`` into its
sandbox subprocess; when that says ``readonly`` the connector base class
refuses every write before any connector-specific code runs — so
``osprey.runtime.write_channel`` and direct ``connector.write_channel`` calls
alike are blocked in a readonly run even on a write-enabled deployment.
"""

import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from osprey.connectors.control_system.base import (
    ChannelWriteResult,
    ControlSystemConnector,
)


class _WriteEnabledConnector(ControlSystemConnector):
    """Concrete connector whose deployment posture says writes are on."""

    def __init__(self):
        self.writes: list[tuple[str, Any]] = []

    async def connect(self, config: dict[str, Any]) -> None: ...
    async def disconnect(self) -> None: ...

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str = "callback",
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        self.writes.append((channel_address, value))
        return ChannelWriteResult(
            channel_address=channel_address, value_written=value, success=True
        )

    async def write_multiple_channels(self, operations, timeout=None):
        return [await self.write_channel(addr, val) for addr, val in operations]

    async def subscribe(self, channel_address, callback):
        raise NotImplementedError

    async def unsubscribe(self, channel_address):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:
        return True


@pytest.fixture
def writes_enabled_deployment(monkeypatch):
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: True if key == "control_system.writes_enabled" else default,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_run_refuses_write(monkeypatch, writes_enabled_deployment):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 150.0)

    assert result.blocked is True
    assert result.success is False
    assert result.refusal_reason == "WRITES_DISABLED"
    assert "readonly" in result.error_message
    assert connector.writes == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_run_refuses_multi_write(monkeypatch, writes_enabled_deployment):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

    assert all(r.blocked for r in results)
    assert connector.writes == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_refusal_message_does_not_blame_deployment(
    monkeypatch, writes_enabled_deployment
):
    """The operator-facing text must not send anyone to flip writes_enabled —
    the deployment allows writes; this *run* was declared readonly."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" not in result.error_message
    assert "readwrite" in result.error_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readwrite_run_passes_through(monkeypatch, writes_enabled_deployment):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.success is True
    assert connector.writes == [("A:SP", 1.0)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_mode_var_means_not_a_sandbox_run(monkeypatch, writes_enabled_deployment):
    """Outside the sandbox (e.g. the controls MCP server) the variable is unset
    and the deployment posture alone decides."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.success is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_refusal_message_carries_the_shared_marker(
    monkeypatch, writes_enabled_deployment
):
    """The connector's refusal must stay recognisable to the tool layer.

    A write refused here reaches the executor tools only as a traceback on the
    subprocess's stderr, where it is matched by substring so that the operator
    alert and the audit record fire for this layer too. Rewording either
    message without the other would silently stop that, so the two are pinned
    to one constant.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert READONLY_REFUSAL_MARKER in result.error_message


# --- Session posture: the same variable, a different story to tell ----------
#
# A session switched to the sandbox posture carries ``OSPREY_EXECUTION_MODE=
# readonly`` in the environment of every process it launched, so the controls
# MCP server refuses ``channel_write`` through exactly the code path above —
# but there is no script there, and nothing to resubmit. The connector tells
# the two apart by whether it is running inside an OSPREY MCP server process
# (see ``_in_mcp_server_process``): the executor's sandbox subprocess, which
# runs a script the operator really can resubmit, keeps the older text.


def _controls_server_main() -> Path:
    """``argv[0]`` of a live controls MCP server: its ``__main__.py``.

    Built from the installed package rather than a made-up string, so the
    discriminator is pinned against the real launch layout
    (``python -m osprey.mcp_server.control_system``, registry/mcp.py) instead
    of against a path this test invented.
    """
    spec = find_spec("osprey.mcp_server")
    assert spec is not None and spec.origin is not None
    main_py = Path(spec.origin).parent / "control_system" / "__main__.py"
    assert main_py.exists(), f"controls MCP server entry point moved: {main_py}"
    return main_py


@pytest.fixture
def sandbox_posture_session(monkeypatch):
    """A controls MCP server process inside a session in the sandbox posture."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setattr(sys, "argv", [str(_controls_server_main())])


@pytest.fixture
def executor_sandbox_script(monkeypatch, tmp_path):
    """The executor's sandbox subprocess: a real script, running readonly."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "wrapped_script.py")])


@pytest.mark.unit
def test_mcp_server_process_is_detected_from_the_real_entry_point(monkeypatch):
    """The discriminator, on its own: server entry point yes, script no.

    The posture and a readonly script run are indistinguishable by environment
    — the posture is env-only by design and the executor's subprocess inherits
    its parent's whole environment — so this is the only thing separating the
    two messages. Pinning it here means a move of the MCP server package
    fails loudly rather than silently reverting the posture text.
    """
    from osprey.connectors.control_system.base import _in_mcp_server_process

    monkeypatch.setattr(sys, "argv", [str(_controls_server_main())])
    assert _in_mcp_server_process() is True

    monkeypatch.setattr(sys, "argv", ["/tmp/exec_001/wrapped_script.py"])
    assert _in_mcp_server_process() is False

    monkeypatch.setattr(sys, "argv", [])
    assert _in_mcp_server_process() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posture_refusal_names_the_posture(
    writes_enabled_deployment, sandbox_posture_session
):
    """The operator is told what actually refused: the session's posture."""
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 150.0)

    assert result.blocked is True
    assert result.success is False
    assert "posture" in result.error_message.lower()
    assert "sandbox posture" in result.error_message.lower()
    assert connector.writes == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posture_refusal_does_not_blame_deployment_or_a_script(
    writes_enabled_deployment, sandbox_posture_session
):
    """Neither of the other two stories applies here.

    ``writes_enabled`` is not the gate (the deployment allows writes), and
    there is no script to resubmit — ``channel_write`` came straight from the
    agent through the controls MCP server. Sending the operator to either one
    is a dead end.
    """
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" not in result.error_message
    assert "resubmit" not in result.error_message.lower()
    assert "execution_mode" not in result.error_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posture_refusal_says_where_to_change_it(
    writes_enabled_deployment, sandbox_posture_session
):
    """A refusal that names no way out is a dead end; the toggle lives on the
    terminal card, which is also where the hook and the executor gate point."""
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "terminal card" in result.error_message.lower()
    assert "writes posture" in result.error_message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posture_refusal_carries_the_shared_marker(
    writes_enabled_deployment, sandbox_posture_session
):
    """The posture message keeps the marker the script message carries.

    It costs nothing — the posture is readonly execution mode, held for a
    whole session instead of one run — and it means the stderr matcher still
    recognises this refusal if it is ever produced inside a subprocess, which
    is exactly the case a future launch-shape change would create.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert READONLY_REFUSAL_MARKER in result.error_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posture_refusal_keeps_the_shared_refusal_reason(
    writes_enabled_deployment, sandbox_posture_session
):
    """Only the message forks. ``refusal_reason`` is the machine-readable
    contract every caller of ``raise_for_write_result`` already handles, and a
    posture refusal is the same kind of refusal: writes are off for this
    caller, and the control system was never asked."""
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.refusal_reason == "WRITES_DISABLED"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_posture_refusal_covers_the_multi_write_path(
    writes_enabled_deployment, sandbox_posture_session
):
    """Both guarded entry points build their result the same way."""
    connector = _WriteEnabledConnector()

    results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

    assert all(r.blocked for r in results)
    assert all("sandbox posture" in r.error_message.lower() for r in results)
    assert connector.writes == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sandbox_script_run_keeps_the_script_shaped_message(
    writes_enabled_deployment, executor_sandbox_script
):
    """Inside the executor's subprocess a script genuinely exists.

    The posture text would be wrong here: the run is readonly because *this
    run* was declared readonly, and resubmitting it as readwrite is the real
    remedy. Pins that the discriminator did not swallow the older branch.
    """
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "readwrite" in result.error_message
    assert "posture" not in result.error_message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deployment_refusal_is_unchanged_inside_an_mcp_server(monkeypatch):
    """The posture branch is reached only when the posture is what refused.

    A write-disabled deployment refuses in the controls MCP server too, and
    that one really is ``writes_enabled`` — telling the operator to change
    their session posture would send them somewhere that cannot help.
    """
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: False if key == "control_system.writes_enabled" else default,
    )
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.setattr(sys, "argv", [str(_controls_server_main())])
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" in result.error_message
    assert "posture" not in result.error_message.lower()

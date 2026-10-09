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

import pytest

from osprey.connectors.control_system.base import WriteOutcome
from tests.connectors._write_fakes import RecordingConnector, writes_enabled_config


@pytest.fixture
def writes_enabled_deployment(monkeypatch):
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_readonly_run_refuses_write(monkeypatch):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = RecordingConnector()

    result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 150.0)

    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "WRITES_DISABLED"
    assert "readonly" in result.error_message
    assert connector.writes == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_readonly_run_refuses_multi_write(monkeypatch):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = RecordingConnector()

    results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

    assert all(r.outcome is WriteOutcome.REFUSED for r in results)
    assert connector.writes == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_readonly_refusal_message_does_not_blame_deployment(monkeypatch):
    """The operator-facing text must not send anyone to flip writes_enabled —
    the deployment allows writes; this *run* was declared readonly."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" not in result.error_message
    assert "readwrite" in result.error_message


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_readwrite_run_passes_through(monkeypatch):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.outcome is WriteOutcome.CONFIRMED
    assert connector.writes == [("A:SP", 1.0)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_no_mode_var_means_not_a_sandbox_run(monkeypatch):
    """Outside the sandbox (e.g. the controls MCP server) the variable is unset
    and the deployment posture alone decides."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.outcome is WriteOutcome.CONFIRMED


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_readonly_refusal_message_carries_the_shared_marker(monkeypatch):
    """The connector's refusal must stay recognisable to the tool layer.

    A write refused here reaches the executor tools only as a traceback on the
    subprocess's stderr, where it is matched by substring so that the operator
    alert and the audit record fire for this layer too. Rewording either
    message without the other would silently stop that, so the two are pinned
    to one constant.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert READONLY_REFUSAL_MARKER in result.error_message


# --- A deployment-wide read-only run: same variable, a different story ------
#
# ``OSPREY_EXECUTION_MODE=readonly`` on the deployment itself puts the whole
# run in readonly execution mode, so the controls MCP server refuses
# ``channel_write`` through exactly the code path above — but there is no
# script there, and nothing to resubmit. The connector tells the two apart by
# whether it is running inside an OSPREY MCP server process (see
# ``_in_mcp_server_process``): the executor's sandbox subprocess, which runs a
# script the operator really can resubmit, keeps the older text. Neither story
# is the narrowing the control-target chip makes — that never
# lived in the environment, and is read from the posture store.


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
def readonly_run_in_an_mcp_server(monkeypatch):
    """A controls MCP server process inside a deployment-wide read-only run."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setattr(sys, "argv", [str(_controls_server_main())])


@pytest.fixture
def executor_sandbox_script(monkeypatch, tmp_path):
    """The executor's sandbox subprocess: a real script, running readonly."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "wrapped_script.py")])


def test_mcp_server_process_is_detected_from_the_real_entry_point(monkeypatch):
    """The discriminator, on its own: server entry point yes, script no.

    A deployment-wide read-only run and a readonly script run are
    indistinguishable by environment — both are the same variable at the same
    value, and the executor's subprocess inherits its parent's whole
    environment — so this is the only thing separating the two messages.
    Pinning it here means a move of the MCP server package fails loudly rather
    than silently reverting the deployment-wide text.
    """
    from osprey.connectors.control_system.base import _in_mcp_server_process

    monkeypatch.setattr(sys, "argv", [str(_controls_server_main())])
    assert _in_mcp_server_process() is True

    monkeypatch.setattr(sys, "argv", ["/tmp/exec_001/wrapped_script.py"])
    assert _in_mcp_server_process() is False

    monkeypatch.setattr(sys, "argv", [])
    assert _in_mcp_server_process() is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "readonly_run_in_an_mcp_server")
async def test_readonly_run_refusal_names_the_deployment_wide_run():
    """The operator is told what actually refused: the whole run is read-only.

    Not a posture, and not this one session — the variable is on the
    deployment, so every session it serves refuses writes while it is set.
    """
    connector = RecordingConnector()

    result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 150.0)

    assert result.outcome is WriteOutcome.REFUSED
    assert "readonly execution mode" in result.error_message.lower()
    assert "OSPREY_EXECUTION_MODE=readonly" in result.error_message
    assert "every session" in result.error_message.lower()
    assert connector.writes == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "readonly_run_in_an_mcp_server")
async def test_readonly_run_refusal_does_not_blame_config_or_a_script():
    """Neither of the other stories applies here.

    ``writes_enabled`` is not the gate (the deployment config allows writes),
    and there is no script to resubmit — ``channel_write`` came straight from
    the agent through the controls MCP server. Sending the operator to either
    one is a dead end. The variable is named, but as the run's own switch,
    never as an ``execution_mode`` argument the caller could resubmit with.
    """
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" not in result.error_message
    assert "resubmit" not in result.error_message.lower()
    assert "execution_mode=" not in result.error_message
    assert "execution_mode='" not in result.error_message


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "readonly_run_in_an_mcp_server")
async def test_readonly_run_refusal_says_the_chip_cannot_lift_it():
    """The one remedy that does NOT work is named, because it is the one an
    operator would reach for: the control-target chip in the header already
    reads writes here and cannot lift a deployment-wide read-only run. Sending
    them there is the dead end this wording exists to close."""
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    message = result.error_message.lower()
    assert "control-target chip in the header" in message
    assert "cannot lift it" in message


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "readonly_run_in_an_mcp_server")
async def test_readonly_run_refusal_carries_the_shared_marker():
    """This message keeps the marker the script message carries.

    It costs nothing — both are readonly execution mode, one held for the
    whole deployment and one for a single run — and it means the stderr
    matcher still recognises this refusal if it is ever produced inside a
    subprocess, which is exactly the case a future launch-shape change would
    create.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert READONLY_REFUSAL_MARKER in result.error_message


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "readonly_run_in_an_mcp_server")
async def test_readonly_run_refusal_keeps_the_shared_refusal_reason():
    """Only the message forks. ``refusal_reason`` is the machine-readable
    contract every caller of ``raise_for_write_result`` already handles, and a
    read-only run's refusal is the same kind of refusal: writes are off for
    this caller, and the control system was never asked."""
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.refusal_reason == "WRITES_DISABLED"


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "readonly_run_in_an_mcp_server")
async def test_readonly_run_refusal_covers_the_multi_write_path():
    """Both guarded entry points build their result the same way."""
    connector = RecordingConnector()

    results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

    assert all(r.outcome is WriteOutcome.REFUSED for r in results)
    assert all("readonly execution mode" in r.error_message.lower() for r in results)
    assert connector.writes == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes_enabled_deployment", "executor_sandbox_script")
async def test_sandbox_script_run_keeps_the_script_shaped_message():
    """Inside the executor's subprocess a script genuinely exists.

    The posture text would be wrong here: the run is readonly because *this
    run* was declared readonly, and resubmitting it as readwrite is the real
    remedy. Pins that the discriminator did not swallow the older branch.
    """
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "readwrite" in result.error_message
    # The MCP-server branch's distinctive text must not leak into the script's.
    assert "every session" not in result.error_message.lower()


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
    connector = RecordingConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" in result.error_message
    # Not a readonly run, so neither readonly story may be told.
    assert "readonly execution mode" not in result.error_message.lower()

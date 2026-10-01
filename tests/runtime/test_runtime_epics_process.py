"""The runtime against the EPICS connector's per-process facts.

Two of them, both about pvapy:

* It binds each provider to the ``EPICS_CA_*`` / ``EPICS_PVA_*`` environment
  in force at its first channel, for the life of the process. A process that
  is not a notebook kernel and whose stamp moved rebuilds its connector in the
  SAME process, so a rebuild needing another gateway is refused — the runtime
  raises :class:`ControlTargetUnreachableError` and holds no connector,
  rather than reaching the old gateway under the new target's name. (A kernel
  serves Channel Access from connector-host children instead; see
  ``test_runtime_kernel_pool.py``.)
* On macOS a thread that called pvapy hangs forever when it exits. The
  runtime's notebook branch runs its coroutine on a ``ThreadPoolExecutor``
  thread that is joined at once, and the limits net's ``max_step`` read is
  made from there — so that read must run on the connector's own workers.
  The subprocess test below hangs (and fails on its budget) if it does not.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import osprey.runtime as runtime
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.runtime import ControlTargetChangedError, ControlTargetUnreachableError
from osprey_connectors.errors import ClientEndpointConflictError
from tests.connectors._epics_fakes import (
    EPICS_CA_VARS,
    EPICS_PVA_VARS,
    install_fake_pvaccess,
    record,
)
from tests.connectors._epics_fakes import patch_writes_enabled as _patch_writes_enabled

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The gateway each control target's block names.
GATEWAYS = {"va": "va-gw", "live": "live-gw"}


@pytest.fixture
def kernel_like(monkeypatch):
    """A process with no runtime state, stamped like a kernel, building real EPICS connectors.

    The factory is stood in for only so far as choosing the block: each target
    gets an :class:`EPICSConnector` connected for its own gateway, over the
    fake pvapy client.
    """
    for var in EPICS_CA_VARS + EPICS_PVA_VARS:
        monkeypatch.delenv(var, raising=False)
    for name in ("_runtime_connector", "_connector_stamp", "_cell_marker", "_limits_validator"):
        monkeypatch.setattr(runtime, name, None)
    monkeypatch.delenv(runtime.ENV_CONTROL_TARGET_REFUSAL, raising=False)
    monkeypatch.delenv(runtime.ENV_IN_CELL, raising=False)
    _patch_writes_enabled(monkeypatch, False)
    pvaccess = install_fake_pvaccess(monkeypatch)
    pvaccess.serve("SR:CH", record(3.0))

    async def create(config=None, control_target=None):  # noqa: ARG001 - called by keyword
        connector = EPICSConnector()
        await connector.connect(
            {"gateways": {"read_only": {"address": GATEWAYS[control_target], "port": 5064}}}
        )
        return connector

    monkeypatch.setattr(runtime, "_target_connector_config", lambda: None)
    monkeypatch.setattr(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector", create
    )
    yield monkeypatch
    runtime._runtime_connector = None
    runtime._connector_stamp = None


def _stamp(monkeypatch, target: str, generation: int) -> None:
    monkeypatch.setenv(runtime.ENV_CONTROL_TARGET, target)
    monkeypatch.setenv(runtime.ENV_CONTROL_TARGET_GENERATION, str(generation))


class TestATargetMoveInOneProcess:
    def test_a_rebuild_needing_another_gateway_is_refused(self, kernel_like):
        _stamp(kernel_like, "va", 1)
        assert runtime.read_channel("SR:CH") == 3.0

        _stamp(kernel_like, "live", 2)
        with pytest.raises(ControlTargetUnreachableError) as caught:
            runtime.read_channel("SR:CH")

        message = str(caught.value)
        assert message.startswith("This process cannot reach control target 'live'.")
        assert "EPICS_CA_ADDR_LIST=live-gw" in message
        assert "EPICS_CA_ADDR_LIST=va-gw" in message
        assert "fresh process" in message
        assert isinstance(caught.value, ControlTargetChangedError)
        assert isinstance(caught.value.__cause__, ClientEndpointConflictError)
        # Nothing is held, and the environment still names the bound gateway.
        assert runtime._runtime_connector is None
        assert os.environ["EPICS_CA_ADDR_LIST"] == "va-gw"

    def test_every_later_call_is_refused_the_same_way(self, kernel_like):
        """No call after the refusal reaches the old gateway, a write least of all."""
        _stamp(kernel_like, "va", 1)
        runtime.read_channel("SR:CH")
        _stamp(kernel_like, "live", 2)
        with pytest.raises(ControlTargetUnreachableError):
            runtime.read_channel("SR:CH")
        pvaccess = sys.modules["pvaccess"]
        gets = len(pvaccess.calls("get"))

        with pytest.raises(ControlTargetUnreachableError):
            runtime.read_channel("SR:CH")

        assert len(pvaccess.calls("get")) == gets

    def test_a_move_that_keeps_the_gateway_rebuilds(self, kernel_like):
        """A new generation on the same target needs no new endpoint."""
        _stamp(kernel_like, "va", 1)
        runtime.read_channel("SR:CH")
        first = runtime._runtime_connector

        _stamp(kernel_like, "va", 2)

        assert runtime.read_channel("SR:CH") == 3.0
        assert runtime._runtime_connector is not first


# ---------------------------------------------------------------------------
# The notebook branch of _run_async, against real pvapy
# ---------------------------------------------------------------------------

#: How long the scenario may take, start to exit. The step read's own budget
#: is 0.2 s (backstop 1.6 s); a regression is a process that never exits.
SCENARIO_BUDGET_S = 30.0

_SCENARIO = textwrap.dedent(
    """
    import asyncio
    import pvaccess

    import osprey.runtime as runtime
    from osprey_connectors.control_system.epics_connector import EPICSConnector
    from osprey_connectors.control_system.limits_validator import (
        ChannelLimitsConfig,
        LimitsValidator,
    )
    from osprey_connectors.errors import ChannelLimitsViolationError

    connector = EPICSConnector()
    connector._pvaccess = pvaccess
    connector._timeout = 0.2
    connector._step_read_timeout = 0.2
    connector._connected = True
    runtime._runtime_connector = connector
    runtime._connector_stamp = (None, None)
    runtime._limits_validator = LimitsValidator(
        {
            "NOPE:CH": ChannelLimitsConfig(
                channel_address="NOPE:CH", min_value=0.0, max_value=100.0, max_step=1.0
            )
        },
        {"allow_unlisted_channels": False},
        {},
    )

    async def cell():
        # A notebook cell: synchronous runtime code inside a running loop.
        try:
            runtime.write_channel("NOPE:CH", 1.0)
        except ChannelLimitsViolationError as exc:
            print("REFUSED", repr(exc), flush=True)

    asyncio.run(cell())
    print("DONE", flush=True)
    """
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_notebook_branch_step_read_leaves_the_process_free_to_exit(tmp_path):
    """The ``max_step`` read made from ``_run_async``'s pool thread never hangs it.

    Real pvapy, no IOC: the step read times out, the write is refused for want
    of a reading, and the process must then exit. Had the read run on the
    pool thread itself, joining that thread on macOS would never return.
    """
    pytest.importorskip("pvaccess")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("EPICS_", "OSPREY_", "PYTEST_"))
    }
    env.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(REPO_ROOT),
                str(REPO_ROOT / "src"),
                str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
            ]
        ),
        EPICS_CA_ADDR_LIST="127.0.0.1",
        EPICS_CA_SERVER_PORT=str(_free_port()),
        EPICS_CA_AUTO_ADDR_LIST="NO",
    )

    start = time.monotonic()
    try:
        done = subprocess.run(
            [sys.executable, "-c", _SCENARIO],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=SCENARIO_BUDGET_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the scenario did not exit within {SCENARIO_BUDGET_S}s — a pvapy call ran on "
            f"a joined thread. stdout: {exc.stdout!r} stderr: {exc.stderr!r}"
        )
    elapsed = time.monotonic() - start

    assert done.returncode == 0, done.stderr
    assert "REFUSED" in done.stdout, done.stdout
    assert "DONE" in done.stdout
    assert "can't proceed" not in done.stderr
    assert elapsed < SCENARIO_BUDGET_S

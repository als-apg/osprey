"""What CANOPY's PV backend says it is bound to, in each binding state.

The backend exposes ``label``, ``notice`` and ``binding`` for CANOPY to
announce to its clients. Each test opens a real
:class:`~osprey.runtime.session.ControlSession` against a control-context
directory in ``tmp_path``, in one of the three states a session can be in —
bound to a record, falling back to the deployment's baseline, or unservable —
and checks the three answers while it is open and that all three are ``None``
before ``start()`` and after ``close()``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from canopy.epics.config import EpicsConfig
from canopy.epics.gateway import build_connector

from osprey.integrations.canopy import backend as backend_module
from osprey.runtime import session as session_module
from osprey.runtime.session import STATE_BOUND, STATE_FALLBACK, STATE_UNSERVABLE
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries

# start() arms the real raw-put block in this process; put back what it patched.
pytestmark = pytest.mark.usefixtures("raw_put_block_restored")

BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: A deployment whose baseline is the virtual accelerator.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


class _RefusingConnector(FakeConnector):
    """A connector whose control system is not there."""

    async def connect(self, config):  # noqa: ARG002 - the connector connect signature
        raise ConnectionError("no gateway answers")


def _write_record(state_dir: Path) -> None:
    control_context.write_record(
        control_context.ControlContext(target="va", generation=3, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A session process's environment, serving :data:`VA_SECTION`, with no record yet."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith(("OSPREY_CONTROL_TARGET", "EPICS_", "CANOPY_EPICS__")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.chdir(tmp_path)

    def get_config_value(path, default=None, _config_path=None):
        return VA_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)

    def no_exit(status):
        raise AssertionError(f"the backend tried to end the test process with status {status}")

    monkeypatch.setattr(backend_module.os, "_exit", no_exit)
    control_context.invalidate_cache()
    yield directory
    control_context.invalidate_cache()


def _identity(connector) -> tuple:
    return (connector.label, connector.notice, connector.binding)


async def _check_lifecycle(connector_class, *, state: str, notice, binding_id: str) -> None:
    """Start a backend over *connector_class*, check its identity, close it."""
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, connector_class)
        built = build_connector(EpicsConfig(backend=BACKEND_PATH, connect_report_s=5.0))
        assert _identity(built) == (None, None, None)
        await built.start()
        try:
            binding = built._session.binding
            assert binding.state == state
            assert built.label == binding.display_name
            assert isinstance(built.label, str) and built.label
            assert built.notice == notice
            assert built.binding == binding.id == binding_id
        finally:
            await built.close()
        assert _identity(built) == (None, None, None)


class TestIdentity:
    """One binding state per test."""

    async def test_a_bound_backend_names_its_record_and_gives_no_notice(
        self, state_dir, monkeypatch
    ):
        """Bound to ``va@3``: the target's name, no notice, the record as its id."""
        _write_record(state_dir)
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
        await _check_lifecycle(FakeConnector, state=STATE_BOUND, notice=None, binding_id="va@3")

    async def test_a_fallback_backend_says_it_is_falling_back(self, state_dir, monkeypatch):  # noqa: ARG002 - the fixture sets the environment
        """No record: the baseline's name, notice and id both ``fallback``."""
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, backend_module.NO_RECORD)
        await _check_lifecycle(
            FakeConnector, state=STATE_FALLBACK, notice="fallback", binding_id="fallback"
        )

    async def test_an_unservable_backend_says_it_is_unservable(self, state_dir, monkeypatch):
        """A record whose control system refuses: notice and id both ``unservable``."""
        _write_record(state_dir)
        monkeypatch.setenv(backend_module.RENDERED_FOR_ENV, "va@3")
        await _check_lifecycle(
            _RefusingConnector,
            state=STATE_UNSERVABLE,
            notice="unservable",
            binding_id="unservable",
        )

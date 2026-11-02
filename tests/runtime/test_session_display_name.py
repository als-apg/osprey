"""The name a session gives its target is the one every other surface renders.

:attr:`~osprey.runtime.session.Binding.display_name` comes from
:func:`~osprey.mcp_server.control_system.connector_host_manager.target_display_metadata`
over the **full** config the file ``OSPREY_CONFIG`` names, so a panel badge, the
web terminal's chip and the prompt hook cannot call one machine two things.
Anything that goes wrong on the way falls back to the target's own name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.runtime import session as session_module
from osprey.runtime.session import ControlSession
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries

#: A deployment whose baseline is the virtual accelerator.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}

#: A stand-in with no simulator beside it and no real machine named: ``live`` has no answer.
STANDIN_ONLY_SECTION = {"type": types.LIVE_STANDIN, "connector": {types.LIVE_STANDIN: {}}}


def _serve(monkeypatch, section):
    """Serve *section* as ``control_system`` without touching the config file."""

    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)


def _write_record(state_dir: Path, *, target="va", generation=3) -> None:
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """The config file ``OSPREY_CONFIG`` names; ``{}`` until a test rewrites it."""
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
        if name.startswith("OSPREY_CONTROL_TARGET") or name.startswith("EPICS_"):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    control_context.invalidate_cache()
    yield config
    control_context.invalidate_cache()


@pytest.fixture
def va_session(config_file, monkeypatch):
    """A ``va`` record served by the fake; yields the config file to rewrite first."""
    _serve(monkeypatch, VA_SECTION)
    _write_record(config_file.parent / "control_target", target="va")
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, FakeConnector)
        yield config_file


async def _display_name() -> str:
    session = await ControlSession.open(surface="panel")
    try:
        return session.binding.display_name
    finally:
        await session.close()


class TestDisplayName:
    """The binding's name is the deployment's, rendered by the one describer."""

    @pytest.mark.usefixtures("va_session")
    async def test_the_virtual_accelerator_is_the_simulator(self):
        assert await _display_name() == "Simulator"

    async def test_the_configured_name_in_the_full_config_wins(self, va_session):
        # The section the session builds its connector from carries no names;
        # only the file does, so this passes only when the full file is read.
        va_session.write_text(
            "control_system:\n  target_display_names:\n    va: Booster twin\n",
            encoding="utf-8",
        )

        assert await _display_name() == "Booster twin"

    async def test_an_unservable_target_is_still_named(self, config_file, monkeypatch):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        _write_record(config_file.parent / "control_target", target="live")

        session = await ControlSession.open(surface="panel")

        assert session.binding.state == "unservable"
        assert session.binding.display_name == "Live"
        await session.close()

    async def test_an_unreadable_config_falls_back_to_the_target(self, va_session):
        va_session.write_text("control_system: [unclosed\n", encoding="utf-8")

        assert await _display_name() == "va"

    async def test_a_missing_config_file_falls_back_to_the_target(self, va_session):
        va_session.unlink()

        assert await _display_name() == "va"

    @pytest.mark.usefixtures("va_session")
    async def test_a_describer_that_raises_falls_back_to_the_target(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("describer broke")

        monkeypatch.setattr(
            "osprey.mcp_server.control_system.connector_host_manager.target_display_metadata",
            broken,
        )

        assert await _display_name() == "va"

    def test_the_session_keeps_no_name_table_of_its_own(self):
        assert not hasattr(session_module, "_DISPLAY_NAMES")

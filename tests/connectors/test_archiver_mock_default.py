"""An unset ``archiver.type`` must resolve to the mock archiver, never to EPICS.

A config that omits the ``archiver:`` section yields ``{}``, which is
under-specified rather than a declaration that a facility archiver exists.
Defaulting such a config to ``epics_archiver`` sent it straight into
``ValueError: archiver URL is required`` on the first ``archiver_read``, so the
fallback resolves to the mock archiver and says so at WARNING level — the same
fail-closed shape ``control_system.type`` uses.

The shipped presets are pinned here too: the fallback is a safety net, not a
substitute for a config that states what it uses.
"""

from __future__ import annotations

import logging

import pytest

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.connectors import types
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries

FACTORY_LOGGER = "connector_factory"

#: Every bundled preset that grants ``archiver_read`` an approval policy, and
#: so has to say which archiver that tool reaches. The framework template
#: renders no ``archiver`` block at all — this comes from the preset's own
#: ``config:``.
PRESETS_WITH_ARCHIVER = ["hello-world", "control-assistant"]


def _cfg(preset: str) -> dict:
    """The preset's resolved ``config:`` block, dotted keys folded in."""
    profile, _profile_dir = resolve_build_profile(None, preset)
    return _expand_dotted(profile.config)


class _LiveArchiverTripwire:
    """Registered as EPICS so a regressed fallback fails loudly instead of dialing out."""

    def __init__(self) -> None:
        raise AssertionError(
            "archiver.type fallback selected the EPICS archiver; "
            "an unset key must resolve to the mock archiver"
        )


@pytest.fixture(autouse=True)
def registered_archivers():
    """Mock plus an EPICS tripwire, so 'not epics_archiver' is proven, not assumed."""
    with isolated_connector_registries(clear=True):
        ConnectorFactory.register_archiver(types.MOCK_ARCHIVER, MockArchiverConnector)
        ConnectorFactory.register_archiver(types.EPICS_ARCHIVER, _LiveArchiverTripwire)
        yield


class TestArchiverTypeFallback:
    @pytest.mark.asyncio
    async def test_empty_config_creates_mock_archiver(self, caplog):
        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_archiver_connector({})

        assert isinstance(connector, MockArchiverConnector)
        assert "archiver.type" in caplog.text
        assert types.MOCK_ARCHIVER in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_blank_type_is_treated_as_unset(self, caplog):
        # A commented-out or emptied YAML value parses as None; it must take the
        # same fail-closed path as a missing key rather than raising later.
        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_archiver_connector({"type": None})

        assert isinstance(connector, MockArchiverConnector)
        assert "archiver.type" in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_config_none_loads_global_config_and_falls_back(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {},
        )

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_archiver_connector(None)

        assert isinstance(connector, MockArchiverConnector)
        assert "archiver.type" in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_explicit_type_is_honoured_without_warning(self, caplog):
        config = {"type": types.MOCK_ARCHIVER, types.MOCK_ARCHIVER: {"sample_rate_hz": 1.0}}

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_archiver_connector(config)

        assert isinstance(connector, MockArchiverConnector)
        assert "archiver.type is not set" not in caplog.text

        await connector.disconnect()


class TestShippedPresetsDeclareAnArchiver:
    @pytest.mark.parametrize("preset", PRESETS_WITH_ARCHIVER)
    def test_preset_names_the_archiver_its_approval_policy_covers(self, preset: str):
        # A preset that grants archiver_read an approval policy has to say which
        # archiver that tool reaches rather than leaning on the fallback.
        config = _cfg(preset)

        assert "archiver_read" in config["approval"]["tools"], (
            f"{preset} no longer grants archiver_read a policy — drop it from this matrix"
        )
        assert config["archiver"]["type"], f"{preset} grants archiver_read but names no archiver"

    def test_the_onboarding_preset_ships_the_mock_archiver(self):
        """hello-world runs no archive service, so it states the mock explicitly."""
        assert _cfg("hello-world")["archiver"]["type"] == types.MOCK_ARCHIVER

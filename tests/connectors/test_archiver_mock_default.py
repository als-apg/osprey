"""An unset ``archiver.type`` must resolve to the mock archiver, never to EPICS.

A config that omits the ``archiver:`` section yields ``{}``, which is
under-specified rather than a declaration that a facility archiver exists.
Defaulting such a config to ``epics_archiver`` sent it straight into
``ValueError: archiver URL is required`` on the first ``archiver_read``, so the
fallback resolves to the mock archiver and says so at WARNING level — the same
fail-closed shape ``control_system.type`` uses.

The shipped templates are pinned here too: the fallback is a safety net, not a
substitute for a config that states what it uses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.connectors import types
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries

FACTORY_LOGGER = "connector_factory"

TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
TEMPLATES_WITH_ARCHIVER = [
    TEMPLATE_ROOT / "apps" / "hello_world" / "config.yml.j2",
    TEMPLATE_ROOT / "project" / "config.yml.j2",
]


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


class TestShippedTemplatesDeclareAnArchiver:
    @pytest.mark.parametrize("template", TEMPLATES_WITH_ARCHIVER, ids=lambda p: p.parent.name)
    def test_template_ships_an_archiver_block(self, template: Path):
        # Both templates grant archiver_read an approval policy, so both must say
        # which archiver that tool reaches rather than leaning on the fallback.
        text = template.read_text()

        assert "archiver:" in text
        assert f"type: {types.MOCK_ARCHIVER}" in text

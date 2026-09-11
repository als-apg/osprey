"""``osprey ariel``'s choice options resolve from the registry, not a literal.

``--adapter``, ``--module`` and ``--mode`` used to freeze the framework's own
names into ``click.Choice`` at import time, so a deployment that registered its
own ingestion adapter could not name it on the command line. Each option now
resolves its options from the registry when the command is parsed.

The second half covers the default that went with that literal: ``ingest``
shipped ``--adapter generic_json``, which silently outranked an
``ariel.ingestion.adapter`` the project had authored. Passing nothing now leaves
the configured adapter in place, the rule ``watch`` already followed.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.ariel import (
    _ENHANCEMENT_MODULES_ATTR,
    _INGESTION_ADAPTERS_ATTR,
    _SEARCH_MODULES_ATTR,
    _RegistryChoice,
    ariel_group,
)
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.config import framework_ariel_names

_DB = {"database": {"uri": "postgresql://ariel:ariel@localhost:5432/ariel"}}


@pytest.fixture
def registered_ingest(monkeypatch) -> list[dict[str, Any]]:
    """Stub ``osprey ariel ingest``'s work and record what it was called with."""
    calls: list[dict[str, Any]] = []

    async def _fake_ingest(config_dict, source, adapter, since, limit, dry_run, progress=None):
        calls.append({"config": config_dict, "adapter": adapter})
        return ops.IngestResult(count=0, enhanced_count=0, failed_count=0, dry_run=True)

    async def _no_resync(config_dict, progress=None):
        return None

    monkeypatch.setattr(ops, "run_ingest", _fake_ingest)
    monkeypatch.setattr(ops, "resync_qmd_mirror_best_effort", _no_resync)
    return calls


class TestChoicesComeFromTheRegistry:
    """Every option's option list is the registry's list."""

    @pytest.mark.parametrize(
        "attribute",
        [_SEARCH_MODULES_ATTR, _ENHANCEMENT_MODULES_ATTR, _INGESTION_ADAPTERS_ATTR],
    )
    def test_choices_match_the_registered_names(self, attribute: str) -> None:
        assert set(_RegistryChoice(attribute).choices) >= set(framework_ariel_names(attribute))

    def test_every_registered_adapter_parses(self, monkeypatch, registered_ingest) -> None:
        """A name the registry carries is accepted, one it does not is refused."""
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)

        for name in framework_ariel_names(_INGESTION_ADAPTERS_ATTR):
            result = CliRunner().invoke(ariel_group, ["ingest", "-s", "entries.json", "-a", name])
            assert result.exit_code == 0, result.output

        refused = CliRunner().invoke(
            ariel_group, ["ingest", "-s", "entries.json", "-a", "no_such_adapter"]
        )
        assert refused.exit_code != 0

    def test_a_registry_added_adapter_is_accepted(self, monkeypatch, registered_ingest) -> None:
        """The point of the change: a facility's own adapter needs no code edit."""
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)
        monkeypatch.setattr(
            "osprey.cli.ariel._registered_names",
            lambda attribute: ("facility_logbook",),
        )

        result = CliRunner().invoke(
            ariel_group, ["ingest", "-s", "entries.json", "-a", "facility_logbook"]
        )

        assert result.exit_code == 0, result.output
        assert registered_ingest[-1]["adapter"] == "facility_logbook"


class TestIngestHonoursTheConfiguredAdapter:
    """``--adapter`` is an override, not a value the CLI always supplies."""

    def test_no_flag_passes_no_override(self, monkeypatch, registered_ingest) -> None:
        monkeypatch.setattr("osprey.cli.ariel.get_config_value", lambda key, default=None: _DB)

        result = CliRunner().invoke(ariel_group, ["ingest", "-s", "entries.json"])

        assert result.exit_code == 0, result.output
        assert registered_ingest[-1]["adapter"] is None

    async def test_run_ingest_keeps_the_configured_adapter(self, monkeypatch) -> None:
        """No override leaves ``ingestion.adapter`` as config.yml authored it."""
        config_dict = {"ingestion": {"adapter": "als_logbook"}}
        seen: list[str] = []

        class _Adapter:
            source_system_name = "als_logbook"

            async def fetch_entries(self, since=None, limit=None):
                return
                yield  # pragma: no cover — makes this an async generator

        def _fake_get_adapter(config):
            seen.append(config.ingestion.adapter)
            return _Adapter()

        monkeypatch.setattr("osprey.services.ariel_search.ingestion.get_adapter", _fake_get_adapter)
        monkeypatch.setattr(
            "osprey.services.ariel_search.enhancement.create_enhancers_from_config",
            lambda config: [],
        )

        result = await ops.run_ingest(config_dict, "entries.json", None, None, None, True)

        assert result.dry_run
        assert seen == ["als_logbook"]
        assert config_dict["ingestion"]["adapter"] == "als_logbook"

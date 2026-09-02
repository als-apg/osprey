"""The build advisory for a remote logbook nothing in the stack ingests."""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_reach import ariel_ingestion_advisories


def _config(*, source_url: str | None = None, deployed: list[str] | None = None) -> dict[str, Any]:
    """A rendered config carrying the two facts the advisory reads."""
    config: dict[str, Any] = {}
    if deployed is not None:
        config["deployed_services"] = deployed
    if source_url is not None:
        config["ariel"] = {"ingestion": {"source_url": source_url}}
    return config


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_remote_source_with_no_ingesting_service_is_advised(scheme: str) -> None:
    """A deploying render pointed at a remote logbook mirrors nothing."""
    config = _config(
        source_url=f"{scheme}://logbook.example.org/api/entries",
        deployed=["postgresql", "ariel"],
    )

    advisories = ariel_ingestion_advisories(config)

    assert len(advisories) == 1
    assert "template: osprey.ariel_sync" in advisories[0]
    assert f"{scheme}://logbook.example.org/api/entries" in advisories[0]


def test_deployed_ariel_sync_is_silent() -> None:
    """The service that ingests it is in the stack, so there is nothing to say."""
    config = _config(
        source_url="https://logbook.example.org/api/entries",
        deployed=["postgresql", "ariel", "ariel_sync"],
    )

    assert ariel_ingestion_advisories(config) == []


def test_attached_render_is_silent() -> None:
    """An attached render deploys nothing, so it ingests nothing by design."""
    config = _config(source_url="https://logbook.example.org/api/entries", deployed=[])

    assert ariel_ingestion_advisories(config) == []


def test_missing_deployed_services_key_is_silent() -> None:
    """A render with no ``deployed_services`` at all is the same case."""
    config = _config(source_url="https://logbook.example.org/api/entries")

    assert ariel_ingestion_advisories(config) == []


def test_local_file_source_is_silent() -> None:
    """The bundled ARIEL app ingests a seed file from disk, not over the network."""
    config = _config(
        source_url="data/logbook_seed/demo_logbook.json",
        deployed=["postgresql", "ariel"],
    )

    assert ariel_ingestion_advisories(config) == []


def test_no_ariel_block_is_silent() -> None:
    """A deployment that runs no ARIEL at all has no source to mirror."""
    assert ariel_ingestion_advisories(_config(deployed=["postgresql"])) == []

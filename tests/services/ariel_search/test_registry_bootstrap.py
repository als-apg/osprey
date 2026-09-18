"""Tests that the search surfaces fill the registry they read.

The search surfaces resolve module objects rather than names, so each one fills
the registry it reads rather than depending on what else the process has
touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.registry import get_registry
from osprey.services.ariel_search.capabilities import get_capabilities
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.service import ARIELSearchService

_ARIEL_SECTION = {
    "database": {"uri": "postgresql://localhost:5432/test"},
    "search_modules": {"keyword": {"enabled": True}},
}


def _config_file(tmp_path: Path) -> Path:
    """Write the config.yml a registry is built from, and return its path."""
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump({"ariel": _ARIEL_SECTION}))
    return path


@pytest.fixture(autouse=True)
def _mock_ariel_registry():
    """This module reads the real registry.

    The package's stand-in is handed back already populated, so it cannot show
    whether a caller fills the registry it reads or finds one somebody else
    filled.
    """
    yield


def _prime(tmp_path: Path):
    """Build the process registry from a config, uninitialized, and return it."""
    registry = get_registry(config_path=str(_config_file(tmp_path)))
    assert registry.list_ariel_search_modules() == []
    return registry


def test_a_fresh_registry_resolves_the_framework_search_modes(tmp_path: Path):
    """The service lists every framework module without a prior initialize."""
    _prime(tmp_path)

    assert set(ARIELSearchService._registered_descriptors()) == {
        "keyword",
        "semantic",
        "hybrid",
    }


@pytest.mark.asyncio
async def test_a_named_mode_routes_in_a_process_that_never_initialized_the_registry(
    tmp_path: Path,
):
    """A configured mode routes in a process that only ever searched."""
    _prime(tmp_path)

    repository = MagicMock()
    repository.health_check = AsyncMock(return_value=(True, "OK"))
    repository.validate_search_model_table = AsyncMock()
    service = ARIELSearchService(
        config=ARIELConfig.from_dict(_ARIEL_SECTION),
        pool=MagicMock(),
        repository=repository,
    )

    with patch(
        "osprey.services.ariel_search.search.keyword.keyword_search",
        AsyncMock(return_value=[]),
    ) as keyword_search:
        result = await service.search("beam current", mode="keyword")

    keyword_search.assert_awaited_once()
    assert result.search_modes_used == ("keyword",)


def test_the_capabilities_report_lists_a_mode_a_fresh_registry_can_route(tmp_path: Path):
    """The report and the router read one listing, so they offer one set."""
    _prime(tmp_path)

    capabilities = get_capabilities(ARIELConfig.from_dict(_ARIEL_SECTION))

    assert [mode["name"] for mode in capabilities["categories"]["direct"]["modes"]] == ["keyword"]

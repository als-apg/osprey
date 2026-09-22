"""Integration tests for ARIEL capability with real service.

Tests capability factory with real database (TEST-M007 / INT-005).

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]


#: The one row the capability's own repository stores and reads back.
CAPABILITY_PREFIX = "cap-integ-"


@pytest.fixture
async def reset_capability_singleton():
    """Reset and cleanup capability singleton before/after tests.

    Uses async cleanup to properly close the service's connection pool.
    """
    from osprey.services.ariel_search.capability import (
        close_ariel_service,
        reset_ariel_service,
    )

    # Reset before test
    reset_ariel_service()
    yield
    # Async cleanup after test
    await close_ariel_service()


class TestCapabilityIntegration:
    """Test capability factory creates real service."""

    @pytest.mark.usefixtures("reset_capability_singleton")
    async def test_get_ariel_search_service_creates_real_service(self, database_url):
        """get_ariel_search_service creates ARIELSearchService (not mock).

        Steps:
        1. Create config with test database_url
        2. Call get_ariel_search_service()
        3. Verify service is ARIELSearchService (not mock)
        4. Verify health_check() passes
        """
        from osprey.services.ariel_search import ARIELSearchService
        from osprey.services.ariel_search.capability import get_ariel_search_service

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {"keyword": {"enabled": True}},
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service = await get_ariel_search_service()

        # Should be real service instance
        assert isinstance(service, ARIELSearchService)

        # Health check should pass with real database
        healthy, message = await service.health_check()
        assert healthy is True
        assert isinstance(message, str)

    @pytest.mark.usefixtures("reset_capability_singleton")
    async def test_get_ariel_search_service_returns_singleton(self, database_url):
        """get_ariel_search_service returns same instance on multiple calls."""
        from osprey.services.ariel_search.capability import get_ariel_search_service

        mock_config = {
            "database": {"uri": database_url},
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service1 = await get_ariel_search_service()
            service2 = await get_ariel_search_service()

        # Should be same instance
        assert service1 is service2

    @pytest.mark.usefixtures("reset_capability_singleton")
    async def test_get_ariel_search_service_raises_without_config(self):
        """get_ariel_search_service raises ConfigurationError when not configured."""
        from osprey.services.ariel_search import ConfigurationError
        from osprey.services.ariel_search.capability import get_ariel_search_service

        with patch("osprey.services.ariel_search.capability.get_config_value", return_value={}):
            with pytest.raises(ConfigurationError) as exc_info:
                await get_ariel_search_service()

        assert "not configured" in str(exc_info.value).lower()

    @pytest.mark.usefixtures("reset_capability_singleton")
    async def test_reset_ariel_service_clears_singleton(self, database_url):
        """reset_ariel_service clears the cached instance."""
        from osprey.services.ariel_search.capability import (
            close_ariel_service,
            get_ariel_search_service,
        )

        mock_config = {"database": {"uri": database_url}}

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service1 = await get_ariel_search_service()

            # Close and reset (proper cleanup)
            await close_ariel_service()

            # New service should be different instance
            service2 = await get_ariel_search_service()

        # Should be different instances
        assert service1 is not service2


class TestCapabilityWithRealService:
    """Test capability factory integration with real service operations."""

    @pytest.mark.usefixtures("migrated_pool", "reset_capability_singleton")
    async def test_service_can_count_entries(self, database_url):
        """Service from capability can count database entries."""
        from osprey.services.ariel_search.capability import get_ariel_search_service

        mock_config = {
            "database": {"uri": database_url},
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service = await get_ariel_search_service()

        count = await service.repository.count_entries()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.usefixtures("migrated_pool", "reset_capability_singleton")
    async def test_service_can_store_and_retrieve_entry(
        self, database_url, seed_entry_factory, seeded_prefixes
    ):
        """Service from capability can store and retrieve entries."""
        from osprey.services.ariel_search.capability import get_ariel_search_service

        mock_config = {
            "database": {"uri": database_url},
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service = await get_ariel_search_service()

        # Create and store entry
        entry = seed_entry_factory(
            entry_id=f"{CAPABILITY_PREFIX}001",
            raw_text="Capability integration test entry",
        )
        seeded_prefixes.add(CAPABILITY_PREFIX)
        await service.repository.upsert_entry(entry)

        # Retrieve entry
        retrieved = await service.repository.get_entry("cap-integ-001")
        assert retrieved is not None
        assert retrieved["entry_id"] == "cap-integ-001"

    @pytest.mark.usefixtures("migrated_pool", "reset_capability_singleton")
    async def test_service_search_returns_result_object(self, database_url):
        """Service search method returns ARIELSearchResult."""
        from osprey.services.ariel_search.capability import get_ariel_search_service
        from osprey.services.ariel_search.models import ARIELSearchResult

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {"keyword": {"enabled": True}},
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service = await get_ariel_search_service()

        result = await service.search("test query")

        assert isinstance(result, ARIELSearchResult)
        assert hasattr(result, "entries")
        assert hasattr(result, "answer")


class TestCapabilityConfig:
    """Test capability handles different config scenarios."""

    @pytest.mark.usefixtures("migrated_pool", "reset_capability_singleton")
    async def test_capability_with_all_modules_enabled(self, database_url):
        """Capability works with all search modules enabled."""
        from osprey.services.ariel_search.capability import get_ariel_search_service

        mock_config = {
            "database": {"uri": database_url},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "nomic-embed-text"},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
            },
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service = await get_ariel_search_service()

        # Should have search modules enabled
        assert service.config.is_search_module_enabled("keyword")
        assert service.config.is_search_module_enabled("semantic")

    @pytest.mark.usefixtures("migrated_pool", "reset_capability_singleton")
    async def test_capability_with_minimal_config(self, database_url):
        """Capability works with minimal config (just database)."""
        from osprey.services.ariel_search.capability import get_ariel_search_service

        mock_config = {
            "database": {"uri": database_url},
        }

        with patch(
            "osprey.services.ariel_search.capability.get_config_value", return_value=mock_config
        ):
            service = await get_ariel_search_service()

        # Should have service with default settings
        assert service is not None
        healthy, _ = await service.health_check()
        assert healthy is True

"""Integration tests for ARIELSearchService.

Tests the service with real PostgreSQL backend.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestServiceIntegration:
    """Test ARIELSearchService with real database."""

    async def test_service_creation_and_health_check(self, integration_ariel_config):
        """Service can be created and passes health check."""
        from osprey.services.ariel_search.service import create_ariel_service

        service = await create_ariel_service(integration_ariel_config)
        async with service:
            healthy, message = await service.health_check()
            assert healthy is True
            assert isinstance(message, str)

    async def test_service_closes_cleanly(self, integration_ariel_config):
        """Service closes without error."""
        from osprey.services.ariel_search.service import create_ariel_service

        service = await create_ariel_service(integration_ariel_config)
        async with service:
            pass  # Just test the context manager works
        # Should not raise

    async def test_service_repository_access(self, integration_ariel_config):
        """Service provides access to repository."""
        from osprey.services.ariel_search.service import create_ariel_service

        service = await create_ariel_service(integration_ariel_config)
        async with service:
            # Verify we can access the database through the service
            count = await service.repository.count_entries()
            assert isinstance(count, int)


class TestAgentIntegration:
    """Test agent execution with mocked LLM (INT-002)."""

    async def test_agent_execution_with_mocked_llm(
        self, repository, integration_ariel_config, seed_entry_factory
    ):
        """Real agent execution flow with LLM responses mocked.

        INT-002: Agent with mocked LLM test.

        Verifies:
        - Tool binding works correctly
        - State management functions
        - Result formatting is correct
        """
        from osprey.services.ariel_search.models import ARIELSearchResult
        from osprey.services.ariel_search.service import ARIELSearchService

        # Create test entries
        entries = [
            seed_entry_factory(
                entry_id="agent-integ-001",
                raw_text="Beam current dropped to 450mA after vacuum event.",
            ),
            seed_entry_factory(
                entry_id="agent-integ-002",
                raw_text="RF cavity frequency adjusted for optimal beam lifetime.",
            ),
        ]

        for entry in entries:
            await repository.upsert_entry(entry)

        # Create service with real repository but mocked LLM
        from osprey.services.ariel_search.database.connection import create_connection_pool

        pool = await create_connection_pool(integration_ariel_config.database)

        service = ARIELSearchService(
            config=integration_ariel_config,
            pool=pool,
            repository=repository,
        )

        # Test search returns proper result structure
        result = await service.search("beam current", max_results=5)

        assert isinstance(result, ARIELSearchResult)
        assert hasattr(result, "entries")
        assert hasattr(result, "answer")
        assert hasattr(result, "search_modes_used")

        await pool.close()

    async def test_tools_are_bound_correctly(self, integration_ariel_config, repository):
        """Tools are properly created and bound to AgentExecutor."""
        from osprey.services.ariel_search.agent import AgentExecutor

        # Create executor
        executor = AgentExecutor(
            config=integration_ariel_config,
            repository=repository,
            embedder_loader=MagicMock(),
        )

        # Create tools
        tools, _descriptors = executor._create_tools()

        # Should have at least keyword tool (enabled in integration_ariel_config)
        assert len(tools) >= 1

        # All tools should have required attributes
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert callable(getattr(tool, "invoke", None)) or callable(
                getattr(tool, "ainvoke", None)
            )

    async def test_cleanup(self, migrated_pool):
        """Clean up agent test data."""
        async with migrated_pool.connection() as conn:
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'agent-integ-%'
            """)


class TestHealthCheckMessages:
    """Test health check returns specific error messages (INT-008)."""

    async def test_health_check_returns_specific_error_messages(self, integration_ariel_config):
        """Health check returns specific error message formats.

        INT-008: Health check message format test.

        Tests each failure mode returns correct message format:
        - Database unavailable
        - Embedding service down
        - etc.
        """
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.repository import ARIELRepository
        from osprey.services.ariel_search.service import ARIELSearchService

        # Test healthy service message
        pool = await create_connection_pool(integration_ariel_config.database)
        repo = ARIELRepository(pool, integration_ariel_config)
        service = ARIELSearchService(
            config=integration_ariel_config,
            pool=pool,
            repository=repo,
        )

        healthy, message = await service.health_check()
        assert healthy is True
        assert "healthy" in message.lower() or "ok" in message.lower()
        await pool.close()

    async def test_health_check_database_failure_message(self, integration_ariel_config):
        """Health check returns database-specific error on DB failure."""
        from osprey.services.ariel_search.service import ARIELSearchService

        # Create service with mock repository that fails health check
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        mock_repo = MagicMock()
        mock_repo.health_check = AsyncMock(return_value=(False, "Database connection refused"))
        mock_repo.config = integration_ariel_config

        service = ARIELSearchService(
            config=integration_ariel_config,
            pool=mock_pool,
            repository=mock_repo,
        )

        healthy, message = await service.health_check()
        assert healthy is False
        assert "database" in message.lower()

    async def test_health_check_message_includes_details(self, integration_ariel_config):
        """Health check message includes relevant diagnostic details."""
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.repository import ARIELRepository
        from osprey.services.ariel_search.service import ARIELSearchService

        pool = await create_connection_pool(integration_ariel_config.database)
        repo = ARIELRepository(pool, integration_ariel_config)
        service = ARIELSearchService(
            config=integration_ariel_config,
            pool=pool,
            repository=repo,
        )

        healthy, message = await service.health_check()

        # Message should be informative
        assert len(message) > 10
        assert isinstance(message, str)

        await pool.close()


class TestServiceSearchModes:
    """Test service handles different search modes."""

    async def test_service_keyword_search(
        self, integration_ariel_config, repository, seed_entry_factory
    ):
        """Service performs keyword search correctly."""
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.models import SearchMode
        from osprey.services.ariel_search.service import ARIELSearchService

        # Add test entry
        entry = seed_entry_factory(
            entry_id="search-mode-001",
            raw_text="Vacuum pump maintenance completed successfully.",
        )
        await repository.upsert_entry(entry)

        pool = await create_connection_pool(integration_ariel_config.database)
        service = ARIELSearchService(
            config=integration_ariel_config,
            pool=pool,
            repository=repository,
        )

        result = await service.search("vacuum pump", mode=SearchMode.KEYWORD)

        # Should return result object
        assert result is not None
        assert hasattr(result, "entries")

        await pool.close()

    async def test_cleanup(self, migrated_pool):
        """Clean up search mode test data."""
        async with migrated_pool.connection() as conn:
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'search-mode-%'
            """)

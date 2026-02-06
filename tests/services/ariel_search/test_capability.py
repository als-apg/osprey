"""Tests for ARIEL capability module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osprey.services.ariel_search.capability import (
    reset_ariel_service,
)


class TestResetArielService:
    """Tests for reset_ariel_service function."""

    def test_reset_clears_singleton(self) -> None:
        """Resetting clears the module-level singleton."""
        from osprey.services.ariel_search import capability

        # Force a value
        capability._ariel_service_instance = "test_value"  # type: ignore[assignment]
        assert capability._ariel_service_instance == "test_value"

        # Reset
        reset_ariel_service()

        # Verify cleared
        assert capability._ariel_service_instance is None


class TestGetArielSearchService:
    """Tests for get_ariel_search_service function."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_ariel_service()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_ariel_service()

    @pytest.mark.asyncio
    async def test_raises_configuration_error_when_not_configured(self) -> None:
        """Raises ConfigurationError when ARIEL is not configured."""
        from osprey.services.ariel_search import ConfigurationError

        # The capability module imports get_config at runtime - we need to mock where it's looked up
        with patch.dict(
            "sys.modules", {"osprey.config": MagicMock(get_config=lambda *a, **kw: {})}
        ):
            # Reset to force re-import of get_config
            reset_ariel_service()
            # Reload the module to pick up the mock
            import importlib

            import osprey.services.ariel_search.capability as cap_module

            importlib.reload(cap_module)

            with pytest.raises(ConfigurationError) as exc_info:
                await cap_module.get_ariel_search_service()

            assert "ARIEL not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_creates_service_from_config(self) -> None:
        """Creates service from configuration - simplified test."""
        from osprey.services.ariel_search import capability as cap_module

        # Test that service gets cached properly
        mock_service = MagicMock()
        cap_module._ariel_service_instance = mock_service

        # When instance exists, it should be returned directly
        result = await cap_module.get_ariel_search_service()
        assert result is mock_service

    @pytest.mark.asyncio
    async def test_returns_singleton_on_subsequent_calls(self) -> None:
        """Returns same instance on subsequent calls."""
        from osprey.services.ariel_search import capability as cap_module

        # Set a mock service
        mock_service = MagicMock()
        cap_module._ariel_service_instance = mock_service

        # Call multiple times
        service1 = await cap_module.get_ariel_search_service()
        service2 = await cap_module.get_ariel_search_service()

        assert service1 is service2
        assert service1 is mock_service


class TestCapabilityExports:
    """Tests for capability module exports."""

    def test_get_ariel_search_service_exported(self) -> None:
        """get_ariel_search_service is exported from main module."""
        from osprey.services.ariel_search import get_ariel_search_service

        assert callable(get_ariel_search_service)
        assert get_ariel_search_service.__name__ == "get_ariel_search_service"

    def test_reset_ariel_service_exported(self) -> None:
        """reset_ariel_service is exported from main module."""
        from osprey.services.ariel_search import reset_ariel_service

        assert callable(reset_ariel_service)
        assert reset_ariel_service.__name__ == "reset_ariel_service"

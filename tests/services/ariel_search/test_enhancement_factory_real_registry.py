"""Regression tests for enhancement factory with real (non-mock) registry.

These tests verify that the enhancement factory works with a real
RegistryManager that has NOT been initialized — reproducing the bug
where `osprey ariel enhance` fails because `initialize_registry()` was
never called by the CLI commands.

Unlike test_enhancement.py, these tests do NOT use the autouse
`_mock_ariel_registry` fixture from conftest.py.
"""

from unittest.mock import patch

import pytest

from osprey.registry.manager import RegistryManager
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.enhancement.factory import (
    create_enhancers_from_config,
    get_enhancer_names,
)


@pytest.fixture
def real_registry():
    """Create a real RegistryManager (framework-only, NOT initialized).

    This reproduces the state the registry is in when CLI commands like
    `osprey ariel enhance` run — constructed but never initialized.
    """
    registry = RegistryManager(registry_path=None)
    # Patch get_registry globally so the factory uses our real instance
    with patch("osprey.registry.get_registry", return_value=registry):
        yield registry


@pytest.fixture
def ariel_config_with_text_embedding():
    """ARIELConfig with text_embedding enabled."""
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
            },
        }
    )


@pytest.fixture
def ariel_config_with_both():
    """ARIELConfig with both enhancement modules enabled."""
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "enhancement_modules": {
                "semantic_processor": {"enabled": True},
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
            },
        }
    )


class TestFactoryWithRealRegistry:
    """Tests that the factory works without registry.initialize()."""

    def test_create_enhancers_uses_real_registry(
        self, real_registry, ariel_config_with_text_embedding
    ):
        """Factory creates enhancers from un-initialized registry.

        This was the bug: create_enhancers_from_config used
        registry.list_ariel_enhancement_modules() which needs initialize(),
        but the CLI never calls initialize().
        """
        enhancers = create_enhancers_from_config(ariel_config_with_text_embedding)
        assert len(enhancers) == 1
        assert enhancers[0].name == "text_embedding"

    def test_create_enhancers_respects_execution_order(
        self, real_registry, ariel_config_with_both
    ):
        """Factory returns enhancers in execution_order (semantic_processor=10 before text_embedding=20)."""
        enhancers = create_enhancers_from_config(ariel_config_with_both)
        assert len(enhancers) == 2
        assert enhancers[0].name == "semantic_processor"
        assert enhancers[1].name == "text_embedding"

    def test_get_enhancer_names_uses_real_registry(self, real_registry):
        """get_enhancer_names works without registry.initialize()."""
        names = get_enhancer_names()
        assert "semantic_processor" in names
        assert "text_embedding" in names

    def test_create_enhancers_configures_module(
        self, real_registry, ariel_config_with_text_embedding
    ):
        """Factory configures the enhancer with module config."""
        enhancers = create_enhancers_from_config(ariel_config_with_text_embedding)
        assert len(enhancers) == 1
        # text_embedding should have been configured with the models list
        assert len(enhancers[0]._models) == 1
        assert enhancers[0]._models[0]["name"] == "nomic-embed-text"
        assert enhancers[0]._models[0]["dimension"] == 768

    def test_create_enhancers_skips_disabled(self, real_registry):
        """Factory skips modules that are not enabled in config."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost:5432/test"},
                "enhancement_modules": {
                    "text_embedding": {
                        "enabled": False,
                        "models": [{"name": "nomic-embed-text", "dimension": 768}],
                    },
                },
            }
        )
        enhancers = create_enhancers_from_config(config)
        assert enhancers == []

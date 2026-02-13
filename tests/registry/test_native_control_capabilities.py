"""Tests for native control system capabilities in the framework registry.

Validates that all 4 control capabilities, 4 context types, and 1 service
are properly registered in the framework registry.
"""

import pytest

from osprey.registry.registry import FrameworkRegistryProvider


class TestNativeControlCapabilities:
    """Test that control system capabilities are registered in framework registry."""

    @pytest.fixture
    def framework_config(self):
        """Get framework registry config."""
        provider = FrameworkRegistryProvider()
        return provider.get_registry_config()

    def test_channel_finding_registered(self, framework_config):
        """Test channel_finding capability is in framework registry."""
        cap_names = [c.name for c in framework_config.capabilities]
        assert "channel_finding" in cap_names

        cap = next(c for c in framework_config.capabilities if c.name == "channel_finding")
        assert cap.module_path == "osprey.capabilities.channel_finding"
        assert cap.class_name == "ChannelFindingCapability"
        assert "CHANNEL_ADDRESSES" in cap.provides
        assert cap.requires == []

    def test_channel_read_registered(self, framework_config):
        """Test channel_read capability is in framework registry."""
        cap_names = [c.name for c in framework_config.capabilities]
        assert "channel_read" in cap_names

        cap = next(c for c in framework_config.capabilities if c.name == "channel_read")
        assert cap.module_path == "osprey.capabilities.channel_read"
        assert cap.class_name == "ChannelReadCapability"
        assert "CHANNEL_VALUES" in cap.provides
        assert "CHANNEL_ADDRESSES" in cap.requires

    def test_channel_write_registered(self, framework_config):
        """Test channel_write capability is in framework registry."""
        cap_names = [c.name for c in framework_config.capabilities]
        assert "channel_write" in cap_names

        cap = next(c for c in framework_config.capabilities if c.name == "channel_write")
        assert cap.module_path == "osprey.capabilities.channel_write"
        assert cap.class_name == "ChannelWriteCapability"
        assert "CHANNEL_WRITE_RESULTS" in cap.provides
        assert "CHANNEL_ADDRESSES" in cap.requires

    def test_archiver_retrieval_registered(self, framework_config):
        """Test archiver_retrieval capability is in framework registry."""
        cap_names = [c.name for c in framework_config.capabilities]
        assert "archiver_retrieval" in cap_names

        cap = next(c for c in framework_config.capabilities if c.name == "archiver_retrieval")
        assert cap.module_path == "osprey.capabilities.archiver_retrieval"
        assert cap.class_name == "ArchiverRetrievalCapability"
        assert "ARCHIVER_DATA" in cap.provides
        assert "CHANNEL_ADDRESSES" in cap.requires


class TestNativeControlContextTypes:
    """Test that control system context types are registered."""

    @pytest.fixture
    def framework_config(self):
        provider = FrameworkRegistryProvider()
        return provider.get_registry_config()

    def test_channel_addresses_context_registered(self, framework_config):
        """Test CHANNEL_ADDRESSES context type is registered."""
        ctx_types = [c.context_type for c in framework_config.context_classes]
        assert "CHANNEL_ADDRESSES" in ctx_types

        ctx = next(
            c for c in framework_config.context_classes if c.context_type == "CHANNEL_ADDRESSES"
        )
        assert ctx.module_path == "osprey.capabilities.channel_finding"
        assert ctx.class_name == "ChannelAddressesContext"

    def test_channel_values_context_registered(self, framework_config):
        """Test CHANNEL_VALUES context type is registered."""
        ctx_types = [c.context_type for c in framework_config.context_classes]
        assert "CHANNEL_VALUES" in ctx_types

        ctx = next(
            c for c in framework_config.context_classes if c.context_type == "CHANNEL_VALUES"
        )
        assert ctx.module_path == "osprey.capabilities.channel_read"
        assert ctx.class_name == "ChannelValuesContext"

    def test_channel_write_results_context_registered(self, framework_config):
        """Test CHANNEL_WRITE_RESULTS context type is registered."""
        ctx_types = [c.context_type for c in framework_config.context_classes]
        assert "CHANNEL_WRITE_RESULTS" in ctx_types

        ctx = next(
            c for c in framework_config.context_classes if c.context_type == "CHANNEL_WRITE_RESULTS"
        )
        assert ctx.module_path == "osprey.capabilities.channel_write"
        assert ctx.class_name == "ChannelWriteResultsContext"

    def test_archiver_data_context_registered(self, framework_config):
        """Test ARCHIVER_DATA context type is registered."""
        ctx_types = [c.context_type for c in framework_config.context_classes]
        assert "ARCHIVER_DATA" in ctx_types

        ctx = next(c for c in framework_config.context_classes if c.context_type == "ARCHIVER_DATA")
        assert ctx.module_path == "osprey.capabilities.archiver_retrieval"
        assert ctx.class_name == "ArchiverDataContext"


class TestNativeControlPromptBuilders:
    """Test that channel finder prompt builders are registered."""

    @pytest.fixture
    def framework_config(self):
        provider = FrameworkRegistryProvider()
        return provider.get_registry_config()

    def test_prompt_builders_registered(self, framework_config):
        """Test channel finder prompt builders in framework prompt provider."""
        prompt_providers = framework_config.framework_prompt_providers
        assert len(prompt_providers) > 0

        # Find the framework prompt provider
        provider = prompt_providers[0]
        builders = provider.prompt_builders

        assert "channel_finder_in_context" in builders
        assert "channel_finder_hierarchical" in builders
        assert "channel_finder_middle_layer" in builders

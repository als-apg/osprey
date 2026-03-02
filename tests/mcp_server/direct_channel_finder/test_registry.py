"""Tests for DirectChannelFinderRegistry."""

import pytest
import yaml

from osprey.mcp_server.direct_channel_finder.registry import (
    DirectChannelFinderRegistry,
    get_dcf_registry,
    initialize_dcf_registry,
    reset_dcf_registry,
)
from osprey.services.channel_finder.backends.mock import MockPVInfoBackend


class TestDirectChannelFinderRegistry:
    def test_initialize_with_mock_backend(self, tmp_path):
        config = {"channel_finder": {"direct": {"backend": "mock"}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_registry()
            assert isinstance(registry.backend, MockPVInfoBackend)
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_initialize_with_default_backend(self, tmp_path):
        config = {"channel_finder": {"direct": {}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_registry()
            assert isinstance(registry.backend, MockPVInfoBackend)
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_backend_raises_when_not_configured(self, tmp_path):
        config = {"channel_finder": {"direct": {"backend": "nonexistent"}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_registry()
            with pytest.raises(RuntimeError, match="PV info backend not available"):
                _ = registry.backend
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_facility_name(self, tmp_path):
        config = {
            "facility": {"name": "ALS"},
            "channel_finder": {"direct": {"backend": "mock"}},
        }
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_registry()
            assert registry.facility_name == "ALS"
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_get_registry_before_init_raises(self):
        reset_dcf_registry()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_dcf_registry()

    def test_get_registry_after_init(self, tmp_path):
        config = {"channel_finder": {"direct": {"backend": "mock"}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            initialize_dcf_registry()
            registry = get_dcf_registry()
            assert registry is not None
        finally:
            del os.environ["OSPREY_CONFIG"]

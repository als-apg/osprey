"""Tests for DirectChannelFinderContext."""

import pytest
import yaml

from osprey.mcp_server.direct_channel_finder.server_context import (
    get_dcf_context,
    initialize_dcf_context,
    reset_dcf_context,
)
from osprey.services.channel_finder.backends.als_channel_finder import (
    ALSChannelFinderBackend,
)
from osprey.services.channel_finder.backends.mock import MockPVInfoBackend


class TestDirectChannelFinderContext:
    def test_initialize_with_mock_backend(self, tmp_path):
        config = {"channel_finder": {"direct": {"backend": "mock"}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_context()
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
            registry = initialize_dcf_context()
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
            registry = initialize_dcf_context()
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
            registry = initialize_dcf_context()
            assert registry.facility_name == "ALS"
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_initialize_with_als_backend(self, tmp_path):
        config = {
            "channel_finder": {
                "direct": {
                    "backend": "als_channel_finder",
                    "backend_url": "https://localhost:9999/ChannelFinder",
                }
            }
        }
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_context()
            assert isinstance(registry.backend, ALSChannelFinderBackend)
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_als_backend_default_url(self, tmp_path):
        config = {"channel_finder": {"direct": {"backend": "als_channel_finder"}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            registry = initialize_dcf_context()
            backend = registry.backend
            assert isinstance(backend, ALSChannelFinderBackend)
            assert "localhost:8443" in backend._base_url
        finally:
            del os.environ["OSPREY_CONFIG"]

    def test_get_context_before_init_raises(self):
        reset_dcf_context()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_dcf_context()

    def test_get_context_after_init(self, tmp_path):
        config = {"channel_finder": {"direct": {"backend": "mock"}}}
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(config))

        import os

        os.environ["OSPREY_CONFIG"] = str(config_path)
        try:
            initialize_dcf_context()
            registry = get_dcf_context()
            assert registry is not None
        finally:
            del os.environ["OSPREY_CONFIG"]

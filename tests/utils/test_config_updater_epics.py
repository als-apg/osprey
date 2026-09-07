"""Tests for EPICS gateway configuration in config_writer."""

from textwrap import dedent

import pytest

from osprey.utils.config_writer import get_epics_gateway_config


@pytest.fixture
def sample_config_content():
    """Sample config.yml content for testing."""
    return dedent(
        """
        control_system:
          type: mock
          connector:
            epics:
              timeout: 5.0
              gateways:
                read_only:
                  address: gw.example.org
                  port: 5064
                  use_name_server: false
                write_access:
                  address: gw.example.org
                  port: 5084
                  use_name_server: false
    """
    )


def test_get_epics_gateway_config(tmp_path, sample_config_content):
    """Test reading EPICS gateway config from file."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    gateways = get_epics_gateway_config(config_path)

    assert gateways is not None
    assert "read_only" in gateways
    assert "write_access" in gateways
    assert gateways["read_only"]["address"] == "gw.example.org"
    assert gateways["read_only"]["port"] == 5064


def test_get_epics_gateway_config_without_a_gateway_block(tmp_path):
    """A config that declares no gateways reads as None, not as an error."""
    config_path = tmp_path / "config.yml"
    config_path.write_text("control_system:\n  type: mock\n")

    assert get_epics_gateway_config(config_path) is None

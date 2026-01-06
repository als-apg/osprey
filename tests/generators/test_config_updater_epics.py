"""Tests for EPICS gateway configuration in config_updater."""

from textwrap import dedent

import pytest

from osprey.generators.config_updater import (
    get_epics_gateway_config,
    get_facility_from_gateway_config,
    set_epics_gateway_config,
)


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
                  address: cagw-alsdmz.als.lbl.gov
                  port: 5064
                  use_name_server: false
                write_access:
                  address: cagw-alsdmz.als.lbl.gov
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
    assert gateways["read_only"]["address"] == "cagw-alsdmz.als.lbl.gov"
    assert gateways["read_only"]["port"] == 5064


def test_get_facility_from_gateway_config_als(tmp_path, sample_config_content):
    """Test detecting ALS facility from gateway config."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    facility = get_facility_from_gateway_config(config_path)

    assert facility == "ALS (Lawrence Berkeley National Laboratory)"


def test_set_epics_gateway_config_aps(tmp_path, sample_config_content):
    """Test setting APS gateway configuration."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    new_content, preview = set_epics_gateway_config(config_path, "aps")

    # Verify preview contains APS information
    assert "APS" in preview or "Argonne" in preview
    assert "pvgatemain1.aps4.anl.gov" in new_content

    # Verify gateway was updated
    config_path.write_text(new_content)
    gateways = get_epics_gateway_config(config_path)
    assert gateways["read_only"]["address"] == "pvgatemain1.aps4.anl.gov"
    assert gateways["read_only"]["port"] == 5064


def test_set_epics_gateway_config_custom(tmp_path, sample_config_content):
    """Test setting custom gateway configuration."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(sample_config_content)

    custom_config = {
        "read_only": {
            "address": "custom-gateway.example.com",
            "port": 6064,
            "use_name_server": True,
        },
        "write_access": {
            "address": "custom-gateway.example.com",
            "port": 6084,
            "use_name_server": True,
        },
    }

    new_content, preview = set_epics_gateway_config(config_path, "custom", custom_config)

    # Verify custom settings in content
    assert "custom-gateway.example.com" in new_content
    assert "6064" in new_content
    assert "true" in new_content.lower()  # use_name_server: true

    # Verify gateway was updated
    config_path.write_text(new_content)
    gateways = get_epics_gateway_config(config_path)
    assert gateways["read_only"]["address"] == "custom-gateway.example.com"
    assert gateways["read_only"]["port"] == 6064
    assert gateways["read_only"]["use_name_server"] is True

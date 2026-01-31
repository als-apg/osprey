"""Tests for EPICS facility presets."""

from osprey.templates.data.facility_presets import (
    FACILITY_PRESETS,
    get_facility_choices,
    get_facility_config,
    list_facilities,
)


def test_facility_presets_structure():
    """Test that facility presets have correct structure."""
    assert "aps" in FACILITY_PRESETS
    assert "als" in FACILITY_PRESETS
    assert "simulation" in FACILITY_PRESETS

    # Check APS structure
    aps = FACILITY_PRESETS["aps"]
    assert "name" in aps
    assert "description" in aps
    assert "gateways" in aps
    assert "read_only" in aps["gateways"]
    assert "write_access" in aps["gateways"]

    # Check gateway has required fields
    assert "address" in aps["gateways"]["read_only"]
    assert "port" in aps["gateways"]["read_only"]
    assert "use_name_server" in aps["gateways"]["read_only"]


def test_get_facility_config():
    """Test getting facility configuration."""
    # Valid facility
    aps_config = get_facility_config("aps")
    assert aps_config is not None
    assert aps_config["name"] == "APS (Argonne National Laboratory)"
    assert aps_config["gateways"]["read_only"]["address"] == "pvgatemain1.aps4.anl.gov"
    assert aps_config["gateways"]["read_only"]["port"] == 5064

    als_config = get_facility_config("als")
    assert als_config is not None
    assert als_config["name"] == "ALS (Lawrence Berkeley National Laboratory)"
    assert als_config["gateways"]["read_only"]["address"] == "cagw-alsdmz.als.lbl.gov"
    assert als_config["gateways"]["read_only"]["port"] == 5064
    assert als_config["gateways"]["write_access"]["port"] == 5084

    # Invalid facility
    invalid_config = get_facility_config("invalid")
    assert invalid_config is None


def test_list_facilities():
    """Test listing available facilities."""
    facilities = list_facilities()
    assert isinstance(facilities, list)
    assert "aps" in facilities
    assert "als" in facilities
    assert "simulation" in facilities


def test_get_facility_choices():
    """Test getting facility choices for CLI."""
    choices = get_facility_choices()
    assert isinstance(choices, list)
    assert len(choices) >= 3  # aps, als, simulation

    # Each choice is a tuple of (display_name, facility_id)
    for display_name, facility_id in choices:
        assert isinstance(display_name, str)
        assert isinstance(facility_id, str)
        assert facility_id in FACILITY_PRESETS


def test_aps_gateway_addresses():
    """Test APS gateway addresses are correct."""
    aps = get_facility_config("aps")
    assert aps["gateways"]["read_only"]["address"] == "pvgatemain1.aps4.anl.gov"
    assert aps["gateways"]["write_access"]["address"] == "pvgatemain1.aps4.anl.gov"
    assert aps["gateways"]["read_only"]["port"] == 5064
    assert aps["gateways"]["write_access"]["port"] == 5064


def test_als_gateway_addresses():
    """Test ALS gateway addresses are correct."""
    als = get_facility_config("als")
    assert als["gateways"]["read_only"]["address"] == "cagw-alsdmz.als.lbl.gov"
    assert als["gateways"]["write_access"]["address"] == "cagw-alsdmz.als.lbl.gov"
    assert als["gateways"]["read_only"]["port"] == 5064
    assert als["gateways"]["write_access"]["port"] == 5084  # Different port for write


def test_simulation_preset():
    """Test simulation preset for local soft IOC."""
    assert "simulation" in FACILITY_PRESETS

    sim = get_facility_config("simulation")
    assert sim is not None
    assert sim["name"] == "Local Simulation"
    assert "caproto" in sim["description"]
    assert sim["gateways"]["read_only"]["address"] == "localhost"
    assert sim["gateways"]["write_access"]["address"] == "localhost"
    assert sim["gateways"]["read_only"]["port"] == 5064
    assert sim["gateways"]["write_access"]["port"] == 5064  # Same port for simulation

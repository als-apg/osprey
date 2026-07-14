"""Assert the shipped 'simulation' facility-gateway preset matches the exact
host<->container CA configuration proven by the PyAT Virtual Accelerator
probe (task 1.1's empirical finding): CA name-server mode over TCP, not
broadcast discovery, which does not reliably cross the macOS<->container VM
boundary. This is the shape `osprey config set-epics-gateway` hands out for
the "Local Simulation" choice, so it must not mislead users into a broadcast
configuration.
"""

from osprey.templates.data.facility_gateways import FACILITY_GATEWAYS, get_facility_config

PROBE_PROVEN_GATEWAY_SHAPE = {
    "address": "localhost",
    "port": 5064,
    "use_name_server": True,
}


def test_simulation_preset_exists():
    assert "simulation" in FACILITY_GATEWAYS


def test_simulation_preset_matches_probe_proven_shape():
    """read_only and write_access must both be CA name-server mode against
    localhost:5064 -- the one host<->container configuration the probe
    confirmed works independent of container runtime."""
    sim = get_facility_config("simulation")
    assert sim is not None

    assert sim["gateways"]["read_only"] == PROBE_PROVEN_GATEWAY_SHAPE
    assert sim["gateways"]["write_access"] == PROBE_PROVEN_GATEWAY_SHAPE


def test_simulation_preset_does_not_rely_on_broadcast_discovery():
    """use_name_server must be True on both gateways: False would mean
    EPICSConnector configures EPICS_CA_ADDR_LIST/EPICS_CA_AUTO_ADDR_LIST
    broadcast discovery, which the probe found unreliable across the
    macOS<->container VM boundary depending on container runtime."""
    sim = get_facility_config("simulation")
    assert sim["gateways"]["read_only"]["use_name_server"] is True
    assert sim["gateways"]["write_access"]["use_name_server"] is True


def test_simulation_preset_description_names_va_container():
    """The description must name the VA container as this preset's intended
    target, not just describe a generic unspecified soft IOC."""
    sim = get_facility_config("simulation")
    description = sim["description"]
    assert "soft IOC" in description  # preserved for test_facility_gateways.py's assertion
    assert "Virtual Accelerator" in description

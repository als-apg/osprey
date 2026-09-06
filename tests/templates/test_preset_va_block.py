"""The `virtual_accelerator` connector block the Control Assistant preset writes.

The preset's ``config:`` block is where a deployment's ``control_system`` comes
from — the framework template renders no connector at all — so the shape is
pinned on the preset's resolved config. Asserts the connector blocks the preset
carries (mock | virtual_accelerator | epics) all co-exist beside the
`live_standin` type it selects, the virtual_accelerator block uses
the probe-proven CA name-server gateway shape (mirrored in the "simulation"
facility-gateway preset — see tests/templates/test_gateway_presets.py) plus a
`simulation_file` key at the exact path
(`connector.virtual_accelerator.simulation_file`) that the type-aware simulation
lookup resolves, and the `epics` block ships no gateway values — authoring them
is the go-live edit.
"""

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile

PRESET = "control-assistant"

# No `port`: it is derived from services.virtual_accelerator.port at
# config-load time rather than rendered out (see
# tests/connectors/test_va_gateway_port_fill.py).
PROBE_PROVEN_GATEWAY_SHAPE = {
    "address": "localhost",
    "use_name_server": True,
}

# The `epics` block a stock deployment ships: the timeout and nothing else. The
# gateways, `probe_channel` and the operator acknowledgment are all commented
# out — a facility's machine cannot be guessed, so authoring them is the
# go-live edit and a fresh deployment's live target reads "not configured".
SHIPPED_EPICS_BLOCK = {"timeout": 5.0}


def _control_system_config():
    """The preset's resolved ``control_system`` block, dotted keys folded in."""
    profile, _profile_dir = resolve_build_profile(None, PRESET)
    return _expand_dotted(profile.config)["control_system"]


def test_all_connector_blocks_coexist():
    connector = _control_system_config()["connector"]
    assert "mock" in connector
    assert "virtual_accelerator" in connector
    assert "epics" in connector


def test_type_selects_the_live_standin():
    """The preset's connector type is the live stand-in.

    Its own connector block is not spelled here: the build assembles
    ``connector.live_standin`` for a deploying render, while the three blocks
    above are the ones the preset carries verbatim for the operator to switch
    to.
    """
    control_system = _control_system_config()
    assert control_system["type"] == "live_standin"
    assert "live_standin" not in control_system["connector"]


def test_virtual_accelerator_block_matches_probe_proven_gateway_shape():
    va = _control_system_config()["connector"]["virtual_accelerator"]
    assert va["gateways"]["read_only"] == PROBE_PROVEN_GATEWAY_SHAPE
    assert va["gateways"]["write_access"] == PROBE_PROVEN_GATEWAY_SHAPE


def test_virtual_accelerator_block_does_not_rely_on_broadcast_discovery():
    va = _control_system_config()["connector"]["virtual_accelerator"]
    assert va["gateways"]["read_only"]["use_name_server"] is True
    assert va["gateways"]["write_access"]["use_name_server"] is True


def test_virtual_accelerator_simulation_file_matches_mock_connector():
    """Contract with the type-aware simulation lookup: the virtual_accelerator
    block's simulation_file must be present at
    connector.virtual_accelerator.simulation_file and match the mock
    connector's value so both types resolve the same machine model."""
    connector = _control_system_config()["connector"]
    assert "simulation_file" in connector["virtual_accelerator"]
    assert (
        connector["virtual_accelerator"]["simulation_file"] == connector["mock"]["simulation_file"]
    )


def test_epics_block_ships_no_gateway_values():
    epics = _control_system_config()["connector"]["epics"]
    assert epics == SHIPPED_EPICS_BLOCK

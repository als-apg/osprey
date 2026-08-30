"""Render test for the `virtual_accelerator` connector block added to the
Control Assistant preset (task 4.4).

Asserts the three-state `control_system.connector` switch (mock default |
virtual_accelerator | epics untouched) renders correctly: all three blocks
co-exist, the default `type` still selects mock, the virtual_accelerator
block uses the probe-proven CA name-server gateway shape (task 1.1's
empirical finding, mirrored in the "simulation" facility-gateway preset —
see tests/templates/test_gateway_presets.py) plus a `simulation_file` key at
the exact path (`connector.virtual_accelerator.simulation_file`) that the
type-aware simulation lookup (task 4.2) resolves, and the `epics` block ships
no gateway values — authoring them is the go-live edit.
"""

import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

TEMPLATE_PATH = "apps/control_assistant/config.yml.j2"

# No `port`: it is derived from services.virtual_accelerator.port at
# config-load time rather than rendered out (see
# tests/connectors/test_va_gateway_port_fill.py).
PROBE_PROVEN_GATEWAY_SHAPE = {
    "address": "localhost",
    "use_name_server": True,
}

# The `epics` block a stock render ships: the timeout and nothing else. The
# gateways, `probe_channel` and the operator acknowledgment are all commented
# out — a facility's machine cannot be guessed, so authoring them is the
# go-live edit and a fresh deployment's live target reads "not configured".
SHIPPED_EPICS_BLOCK = {"timeout": 5.0}


def _render_connector_config():
    manager = TemplateManager()
    template = manager.jinja_env.get_template(TEMPLATE_PATH)
    # The gateway commentary derives its example ports from `osprey_ports`,
    # which the real render builds in TemplateManager._project_context. This
    # helper reaches the environment directly, so it supplies the table itself.
    rendered = template.render(
        port_base=DEFAULT_PORT_BASE,
        osprey_ports=layout_ports(DEFAULT_PORT_BASE),
    )
    return yaml.safe_load(rendered)["control_system"]


def test_all_three_connector_blocks_coexist():
    connector = _render_connector_config()["connector"]
    assert "mock" in connector
    assert "virtual_accelerator" in connector
    assert "epics" in connector


def test_default_type_still_selects_mock():
    control_system = _render_connector_config()
    assert control_system["type"] == "mock"


def test_virtual_accelerator_block_matches_probe_proven_gateway_shape():
    va = _render_connector_config()["connector"]["virtual_accelerator"]
    assert va["gateways"]["read_only"] == PROBE_PROVEN_GATEWAY_SHAPE
    assert va["gateways"]["write_access"] == PROBE_PROVEN_GATEWAY_SHAPE


def test_virtual_accelerator_block_does_not_rely_on_broadcast_discovery():
    va = _render_connector_config()["connector"]["virtual_accelerator"]
    assert va["gateways"]["read_only"]["use_name_server"] is True
    assert va["gateways"]["write_access"]["use_name_server"] is True


def test_virtual_accelerator_simulation_file_matches_mock_connector():
    """Contract with the type-aware simulation lookup (task 4.2): the
    virtual_accelerator block's simulation_file must be present at
    connector.virtual_accelerator.simulation_file and match the mock
    connector's value so both types resolve the same machine model."""
    connector = _render_connector_config()["connector"]
    assert "simulation_file" in connector["virtual_accelerator"]
    assert (
        connector["virtual_accelerator"]["simulation_file"] == connector["mock"]["simulation_file"]
    )


def test_epics_block_ships_no_gateway_values():
    epics = _render_connector_config()["connector"]["epics"]
    assert epics == SHIPPED_EPICS_BLOCK

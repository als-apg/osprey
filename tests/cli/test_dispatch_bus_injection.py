"""The transport axis of the ``_inject_dispatch`` build step (``dispatch.transport``).

Like the network axis next door, one profile knob is written into both service
blocks, and only off its default: an HTTP deployment's ``config.yml`` stays the
bytes ``test_dispatch_network_injection`` pins.
"""

from __future__ import annotations

from osprey.cli.build_cmd import _inject_dispatch
from osprey.port_layout import default_port
from tests.cli.test_dispatch_network_injection import (
    BRIDGE_RENDER_CONFIG_YML,
    _dispatch,
    _project,
    _services,
)


def test_the_bus_is_written_into_both_halves(tmp_path):
    project_path, profile_dir = _project(tmp_path)
    _inject_dispatch(_dispatch(transport="bus"), profile_dir, project_path, facility_name="ERF")

    services = _services(project_path)
    assert services["event_dispatcher"]["transport"] == "bus"
    assert services["event_dispatcher"]["bus_port"] == default_port("dispatch_bus")
    assert services["dispatch_worker"]["transport"] == "bus"
    assert "bus_port" not in services["dispatch_worker"]


def test_http_is_the_default_and_writes_nothing_new(tmp_path):
    project_path, profile_dir = _project(tmp_path)
    _inject_dispatch(_dispatch(transport="http"), profile_dir, project_path, facility_name="ERF")

    assert (project_path / "config.yml").read_text() == BRIDGE_RENDER_CONFIG_YML

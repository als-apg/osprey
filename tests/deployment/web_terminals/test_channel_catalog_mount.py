"""The web terminals mount the build's channel catalog read-only, only when it exists.

``osprey build`` stages ``build/web_terminals/channels.json`` when the
deployment has channels to suggest. ``render_web_terminals`` then binds it
read-only at :data:`osprey.integrations.canopy.CHANNELS_FILE` (the path the
per-user CANOPY panel is given as ``CANOPY_CHANNELS_FILE``) into every web
container. Without the file there is no bind at all: a bind to a missing file
would make the runtime create an empty directory in its place.
"""

from __future__ import annotations

import yaml

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.integrations.canopy import CHANNELS_FILE

from .test_golden_render import EXAMPLE_CONFIG

MOUNT = f"./build/web_terminals/channels.json:{CHANNELS_FILE}:ro"


def _services() -> dict[str, dict]:
    compose = yaml.safe_load(render_web_terminals(EXAMPLE_CONFIG)["docker-compose.web.yml"])
    return compose["services"]


def _catalog_volumes(service: dict) -> list[str]:
    return [v for v in service.get("volumes", []) if isinstance(v, str) and CHANNELS_FILE in v]


def test_the_constant_path_is_the_one_the_panel_reads():
    assert CHANNELS_FILE == "/etc/osprey/channels.json"


def test_no_catalog_no_mount(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, service in _services().items():
        assert _catalog_volumes(service) == [], name


def test_a_staged_catalog_is_mounted_read_only_into_every_web_container(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "build" / "web_terminals" / "channels.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text('["SR:BPM:01:X"]')
    services = _services()
    terminals = {name for name in services if name != "nginx"}
    assert terminals, "the example config renders web containers"
    for name in terminals:
        assert _catalog_volumes(services[name]) == [MOUNT], name
    assert _catalog_volumes(services.get("nginx", {})) == []


def test_a_directory_at_the_catalog_path_is_not_mounted(tmp_path, monkeypatch):
    """Only a regular file counts; a directory left in its place mounts nothing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build" / "web_terminals" / "channels.json").mkdir(parents=True)
    for name, service in _services().items():
        assert _catalog_volumes(service) == [], name

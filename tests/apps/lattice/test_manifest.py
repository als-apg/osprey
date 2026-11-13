"""The application's declaration: a read-only panel, three read tools, one skill."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import lattice_app
import lattice_app.tools as tools
import yaml
from lattice_app import APP_NAME

from osprey.apps.boundary import check_package
from osprey.apps.loader import _refuse_missing_skills, load
from osprey.apps.manifest import AppManifest, parse_manifest
from osprey.mcp_server.startup import application_server
from osprey.profiles import presets

PRESETS_ROOT = Path(presets.__file__).parent
APP_DIR = PRESETS_ROOT / "apps" / APP_NAME
PACKAGE = Path(lattice_app.__file__).resolve().parent
SKILL = APP_DIR / "skills" / APP_NAME / "SKILL.md"


def _manifest() -> AppManifest:
    return parse_manifest(yaml.safe_load((APP_DIR / "app.yml").read_text()), app=APP_NAME)


def test_every_app_scoped_identifier_is_the_applications_name() -> None:
    manifest = _manifest()
    assert manifest.name == APP_NAME
    assert list(manifest.containers) == [APP_NAME]
    assert manifest.containers[APP_NAME].state == f"var/{APP_NAME}"
    assert manifest.containers[APP_NAME].runtime == "osprey"
    assert manifest.panel is not None
    assert manifest.panel.container == APP_NAME
    assert manifest.skills == [APP_NAME]


def test_the_agent_reaches_exactly_the_three_read_routes() -> None:
    manifest = _manifest()
    assert manifest.panel is not None
    assert [(route.method, route.path) for route in manifest.panel.agent_routes] == [
        ("GET", lattice_app.STATE_ROUTE),
        ("GET", lattice_app.RESULT_ROUTE),
        ("GET", lattice_app.EXPORT_ROUTE),
    ]
    # The panel door holds a result read at least as long as the backend waits for one.
    assert manifest.panel.read_timeout_s is not None
    assert manifest.panel.read_timeout_s >= lattice_app.MAX_WAIT_S + tools.MARGIN_S


def test_the_application_writes_nothing() -> None:
    manifest = _manifest()
    assert manifest.tools is not None
    assert manifest.tools.approval == []
    assert manifest.vendor == []


def test_the_tool_server_registers_exactly_the_declared_tools() -> None:
    manifest = _manifest()
    assert manifest.tools is not None
    server = application_server(APP_NAME)
    tools.register(server)
    registered = {tool.name for tool in asyncio.run(server.list_tools())}
    assert registered == set(manifest.tools.read)
    body = SKILL.read_text(encoding="utf-8")
    for name in registered:
        assert f"`{name}" in body, name


def test_the_loader_accepts_the_application_with_its_skill() -> None:
    manifests = load(PRESETS_ROOT, [APP_NAME])
    assert [manifest.name for manifest in manifests] == [APP_NAME]
    _refuse_missing_skills(PRESETS_ROOT, tuple(manifests))
    head = SKILL.read_text(encoding="utf-8")[4:].partition("\n---\n")[0]
    meta = yaml.safe_load(head)
    assert meta["name"] == APP_NAME
    for phrase in ("model tunes", "dynamic aperture", "read-only"):
        assert phrase in meta["description"]


def test_the_package_stays_inside_the_boundary() -> None:
    assert check_package(PACKAGE) == []


def test_the_tool_server_loads_neither_the_backend_nor_the_numerics() -> None:
    tree = ast.parse((PACKAGE / "tools.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"lattice_app.server", "lattice_app.jobs", "numpy", "at"}

"""The modules DATA's read key names are the ones FRONT calls that DATA mounts.

Needs CANOPY installed: the gate installs the pinned export. The sidecar
restates which CANOPY modules FRONT (``canopy_mcp``) calls on DATA, because it
imports nothing from CANOPY; this file derives that set from ``canopy_mcp``'s
own code — every upstream path literal in the package, its forward route table
and its tool table — and pins the restatement against it. A tool or route that
starts calling another module fails here first, instead of as a 403
``insufficient_module_access`` in a running panel.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import canopy_mcp
from canopy.core import config as canopy_config
from canopy.core import modules as canopy_modules
from canopy.core import security
from canopy_mcp import config as mcp_config
from canopy_mcp import forward
from canopy_mcp import server as canopy_server

from osprey.integrations.canopy import EXPECTED_FRONT_CANOPY_VARS, MODULES
from osprey.integrations.canopy import sidecar as sidecar_module

#: Every tool canopy_mcp registers with authoring on. A new tool may call a new
#: module: when this set changes, re-derive ``FRONT_CALLED_MODULES`` from what
#: the new tool asks of DATA before updating it.
TOOLS = frozenset(
    {
        "check",
        "read_pv",
        "search_pvs",
        "screenshot",
        "get_screen",
        "list_drafts",
        "list_widgets",
        "get_widget",
        "get_schema",
        "new_screen",
        "patch_node",
        "add_node",
        "remove_node",
        "delete_draft",
        "convert_legacy",
        "request_push",
        "list_collections",
        "get_workspace",
        "set_workspace",
    }
)

_UPSTREAM = re.compile(r"^/([a-z][a-z-]*)/v1/")


def _upstream_modules(path: str) -> set[str]:
    match = _UPSTREAM.match(path)
    if match and match.group(1) in canopy_config.ALL_MODULES:
        return {match.group(1)}
    return set()


def _package_upstream_modules() -> dict[str, set[str]]:
    """Module -> the canopy_mcp files whose string literals name one of its paths.

    Docstrings and prose never start with ``/<module>/v1/``; a path literal or
    the leading part of an f-string path does.
    """
    root = Path(canopy_mcp.__file__).parent
    found: dict[str, set[str]] = {}
    for source in sorted(root.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for module in _upstream_modules(node.value):
                    found.setdefault(module, set()).add(str(source.relative_to(root)))
    return found


def _forwarded_modules() -> set[str]:
    """The modules of the routes FRONT relays upstream (the 405 stubs excluded)."""
    settings = mcp_config.Settings.model_construct()
    modules: set[str] = set()
    for route in forward.build_forward_routes(settings):
        if getattr(getattr(route, "endpoint", None), "__name__", "") == "not_allowed":
            continue
        modules |= _upstream_modules(route.path)
    return modules


def test_the_restated_called_modules_are_the_ones_canopy_mcps_code_calls() -> None:
    found = _package_upstream_modules()
    assert set(found) == set(sidecar_module.FRONT_CALLED_MODULES), found


def test_the_forward_routes_relay_hare_formula_engine_and_mole() -> None:
    assert _forwarded_modules() == {"hare", "formula-engine", "mole"}
    assert _forwarded_modules() <= set(sidecar_module.FRONT_CALLED_MODULES)


def test_the_tool_path_constants_are_among_the_called_modules() -> None:
    from canopy_mcp import static
    from canopy_mcp.check import lint
    from canopy_mcp.tools import convert, pvs

    paths = {
        lint.RACCOON_VALIDATE_PATH: "raccoon",
        convert.CONVERSIONS_PATH: "converter",
        static.SUMMARY_PATH: "hare",
        pvs._pv_path("SR:C01:X"): "hare",
    }
    for path, module in paths.items():
        assert _upstream_modules(path) == {module}, path
        assert module in sidecar_module.FRONT_CALLED_MODULES


def test_the_tool_table_is_the_one_the_called_modules_were_derived_from() -> None:
    canopy_server.register_all()
    canopy_server.set_authoring(True)
    assert set(canopy_server.registered_tools()) == TOOLS


def test_the_widget_catalog_calls_no_module() -> None:
    """The catalog is the viewer's ir-map.json, read in FRONT: no Beaver, no DATA call.

    Beaver is retired upstream, so it is neither a module CANOPY knows nor one
    FRONT's key may name, and canopy_mcp's catalog code names no upstream path.
    """
    from canopy_mcp import catalog

    assert "beaver" not in sidecar_module.FRONT_CALLED_MODULES
    assert "beaver" not in canopy_config.ALL_MODULES
    assert "catalog.py" not in {f for files in _package_upstream_modules().values() for f in files}
    assert not hasattr(catalog, "WIDGETS_PATH")


def test_the_read_key_names_the_called_modules_data_mounts() -> None:
    mounted = set(MODULES.split(","))
    assert set(sidecar_module.FRONT_MODULES) == set(sidecar_module.FRONT_CALLED_MODULES) & mounted
    assert set(sidecar_module.FRONT_MODULES) <= canopy_modules.MODULES
    known, unknown = canopy_modules.split_modules(set(sidecar_module.FRONT_MODULES))
    assert not unknown and known == frozenset(sidecar_module.FRONT_MODULES)


def test_canopy_loads_the_data_keys_the_sidecar_writes(tmp_path: Path) -> None:
    read_key = "r" * 43
    keys_file = tmp_path / "data.keys"
    keys_file.write_text(sidecar_module.data_keys_text(read_key), encoding="utf-8")
    registry = security.load_keys_file(keys_file)
    assert len(keys_file.read_text(encoding="utf-8").splitlines()) == 1
    key = asyncio.run(registry.get(read_key))
    assert key is not None
    assert key.read and not key.write and not key.admin
    assert key.modules == frozenset(sidecar_module.FRONT_MODULES)
    for module in sidecar_module.FRONT_MODULES:
        assert key.allows_module(module)
    for module in set(MODULES.split(",")) - set(sidecar_module.FRONT_MODULES):
        assert not key.allows_module(module)


def _settings_env_names() -> set[str]:
    prefix = mcp_config.Settings.model_config["env_prefix"]
    return {prefix + field.upper() for field in mcp_config.Settings.model_fields}


def test_every_rendered_front_name_is_one_canopy_reads() -> None:
    """Settings ignores unknown names, so a misspelt or retired name would be
    rendered into FRONT's environment and silently dropped; pin each to a field."""
    settings_names = _settings_env_names()
    unread = {name for name in EXPECTED_FRONT_CANOPY_VARS if name not in settings_names}
    assert not unread, unread


def test_the_collection_stores_are_settings_fields() -> None:
    fields = mcp_config.Settings.model_fields
    assert "collections_dir" in fields
    assert "facility_collections_dir" in fields
    assert {"CANOPY_COLLECTIONS_DIR", "CANOPY_FACILITY_COLLECTIONS_DIR"} <= (
        EXPECTED_FRONT_CANOPY_VARS
    )


def test_the_waveform_default_query_is_a_setting_osprey_renders() -> None:
    assert "waveform_default_query" in mcp_config.Settings.model_fields
    assert "CANOPY_WAVEFORM_DEFAULT_QUERY" in EXPECTED_FRONT_CANOPY_VARS

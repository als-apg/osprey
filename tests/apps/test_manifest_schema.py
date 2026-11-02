"""The application manifest schema (`app.yml`) and its loader."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.errors import AppManifestError
from osprey.apps.loader import load
from osprey.apps.manifest import (
    AppManifest,
    is_write_method,
    launch_token_env,
    parse_manifest,
)

FULL: dict[str, Any] = {
    "name": "knob-demo",
    "summary": "Turn one knob.",
    "containers": {
        "knob-demo": {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_demo.server"],
            "port": 8100,
            "health": "/health",
            "drives_target": True,
            "control_identity": True,
            "state": "var/knob-demo",
            "env": ["KNOB_DEMO_MODE"],
        },
        "knob-cache": {
            "image": "redis:7.2",
            "fragment": "compose/knob-cache/",
        },
    },
    "panel": {
        "label": "KNOB",
        "container": "knob-demo",
        "entry": "/ui/",
        "launch_token": True,
        "read_timeout_s": 30,
        "agent_routes": [["GET", "/api/state"], ["POST", "/api/write"]],
        "roster_grant": {"single_user_alias": True},
    },
    "tools": {
        "module": "knob_demo.tools",
        "shown_when": "panel",
        "deny_in_jobs": True,
        "read": ["read_knob"],
        "approval": [
            {
                "tool": "set_knob",
                "prompt": "Set {{ input.value }}",
                "state_path": "/api/state",
                "display_lines_key": "lines",
                "bind": {
                    "state_fields": [{"field": "approved_readback", "key": "readback"}],
                    "input_digests": [{"field": "approved_digest", "over": ["value"]}],
                },
                "require": {
                    "all_of": ["approved_target", "approved_digest"],
                    "one_of": [["approved_readback"]],
                },
            }
        ],
        "errors": {"unreachable": "Open the panel.", "refused": "Re-read the knob."},
    },
    "audit_identity": "knob-audit",
    "vendor": [
        {"url": "https://example.org/lib.js", "into": "panel/vendor/lib.js", "sha256": "a" * 64},
        {
            "package": "uplot",
            "version": "1.6.30",
            "file": "dist/uPlot.iife.min.js",
            "into": "panel/vendor/uplot.js",
            "sha256": "b" * 64,
        },
    ],
}


def _with(path: str, value: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of `base` (FULL) with the dotted `path` set to `value`."""
    data = copy.deepcopy(FULL if base is None else base)
    *parents, leaf = path.split(".")
    node = data
    for part in parents:
        node = node[part]
    node[leaf] = value
    return data


def _without(path: str) -> dict[str, Any]:
    data = copy.deepcopy(FULL)
    *parents, leaf = path.split(".")
    node = data
    for part in parents:
        node = node[part]
    del node[leaf]
    return data


def _refusal(data: dict[str, Any]) -> AppManifestError:
    with pytest.raises(AppManifestError) as info:
        parse_manifest(data, app="knob-demo")
    return info.value


# --- the full shape ----------------------------------------------------------


def test_full_manifest_parses() -> None:
    m = parse_manifest(FULL, app="knob-demo")
    assert isinstance(m, AppManifest)
    assert m.name == "knob-demo"
    assert m.tools is not None and m.tools.module == "knob_demo.tools"
    entry = m.tools.approval[0]
    assert entry.state_path == "/api/state"
    assert entry.display_lines_key == "lines"
    assert entry.bind.state_fields[0].field == "approved_readback"
    assert entry.bind.state_fields[0].key == "readback"
    assert entry.bind.input_digests[0].over == ["value"]
    assert entry.require.all_of == ["approved_target", "approved_digest"]
    assert entry.require.one_of == [["approved_readback"]]
    assert m.containers["knob-demo"].env == ["KNOB_DEMO_MODE"]
    assert m.containers["knob-cache"].fragment == "compose/knob-cache/"
    assert m.panel is not None and m.panel.entry == "/ui/"
    assert [(r.method, r.path) for r in m.panel.agent_routes] == [
        ("GET", "/api/state"),
        ("POST", "/api/write"),
    ]
    assert m.audit_identity == "knob-audit"


def test_minimal_manifest_takes_defaults() -> None:
    m = parse_manifest({"name": "tiny", "containers": {"tiny": {"runtime": "osprey"}}}, app="tiny")
    assert m.summary is None and m.panel is None and m.tools is None
    assert m.vendor == []
    assert m.audit_identity == "tiny"
    c = m.containers["tiny"]
    assert c.drives_target is False and c.control_identity is False
    assert c.env == [] and c.port is None


def test_panel_id_defaults_to_the_name_and_entry_to_root() -> None:
    m = parse_manifest(
        {
            "name": "tiny",
            "containers": {"tiny": {"runtime": "osprey", "port": 8000}},
            "panel": {"label": "TINY", "container": "tiny"},
        },
        app="tiny",
    )
    assert m.panel is not None
    assert m.panel.id == "tiny"
    assert m.panel.entry == "/"
    assert m.panel.agent_routes == []
    assert m.panel.launch_token is False


def test_read_timeout_unset_is_distinguishable_from_explicit_null() -> None:
    base = {
        "name": "tiny",
        "containers": {"tiny": {"runtime": "osprey"}},
        "panel": {"label": "TINY", "container": "tiny"},
    }
    unset = parse_manifest(base, app="tiny")
    assert unset.panel is not None
    assert "read_timeout_s" not in unset.panel.model_fields_set
    explicit = parse_manifest(_with("panel.read_timeout_s", None, base), app="tiny")
    assert explicit.panel is not None
    assert "read_timeout_s" in explicit.panel.model_fields_set
    assert explicit.panel.read_timeout_s is None


# --- launch token and write routes: the one definition each -------------------


@pytest.mark.parametrize(
    ("app", "env"),
    [
        ("knob-demo", "KNOB_DEMO_LAUNCH_TOKEN"),
        ("tiny", "TINY_LAUNCH_TOKEN"),
        ("a-b-c", "A_B_C_LAUNCH_TOKEN"),
    ],
)
def test_launch_token_env_is_derived_from_the_name(app: str, env: str) -> None:
    assert launch_token_env(app) == env


def test_launch_token_env_refuses_a_non_application_name() -> None:
    with pytest.raises(ValueError):
        launch_token_env("Not An App")


@pytest.mark.parametrize(
    ("method", "write"),
    [
        ("GET", False),
        ("HEAD", False),
        ("OPTIONS", False),
        ("POST", True),
        ("PUT", True),
        ("PATCH", True),
        ("DELETE", True),
    ],
)
def test_a_write_route_is_any_method_but_get_head_options(method: str, write: bool) -> None:
    assert is_write_method(method) is write
    m = parse_manifest(_with("panel.agent_routes", [[method, "/x"]]), app="knob-demo")
    assert m.panel is not None and m.panel.agent_routes[0].is_write is write


# --- name ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["Knob", "knob_demo", "knob demo", ""])
def test_name_must_be_a_slug(bad: str) -> None:
    err = _refusal(_with("name", bad))
    assert err.field == "name"


def test_name_and_containers_are_required() -> None:
    assert _refusal(_without("name")).field == "name"
    assert _refusal(_without("containers")).field == "containers"
    assert _refusal(_with("containers", {})).field == "containers"


# --- unknown keys are refused at every level ------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "colour",
        "containers.knob-demo.colour",
        "panel.colour",
        "panel.roster_grant.colour",
        "tools.colour",
        "tools.approval",
        "tools.errors.colour",
    ],
)
def test_unknown_keys_are_refused_at_every_level(path: str) -> None:
    if path == "tools.approval":
        data = copy.deepcopy(FULL)
        data["tools"]["approval"][0]["colour"] = 1
        field = "tools.approval.0.colour"
    else:
        data = _with(path, 1)
        field = path
    err = _refusal(data)
    assert err.field == field
    assert err.rule == "extra_forbidden"


def test_pre_amendment_keys_are_refused() -> None:
    assert _refusal(_with("panel.path", "/panel/knob-demo")).field == "panel.path"
    assert _refusal(_with("tools.container", "knob-demo")).field == "tools.container"


# --- compose keys name their home -------------------------------------------------


@pytest.mark.parametrize("key", ["binds", "volumes", "env_file", "depends_on", "ports"])
def test_a_compose_key_on_a_container_names_the_fragment(key: str) -> None:
    err = _refusal(_with(f"containers.knob-cache.{key}", ["x"]))
    assert err.field == f"containers.knob-cache.{key}"
    assert err.rule == "compose_key"
    assert "compose/knob-cache/docker-compose.yml.j2" in err.message


def test_network_names_the_profile_axis_key() -> None:
    err = _refusal(_with("containers.knob-demo.network", "bridge"))
    assert err.field == "containers.knob-demo.network"
    assert err.rule == "compose_key"
    assert "profile.yml" in err.message
    assert "apps.knob-demo.services.knob-demo.network" in err.message


def test_command_on_an_image_container_names_the_fragment() -> None:
    err = _refusal(_with("containers.knob-cache.command", ["redis-server"]))
    assert err.field == "containers.knob-cache.command"
    assert err.rule == "compose_key"
    assert "compose/knob-cache/docker-compose.yml.j2" in err.message


# --- containers -------------------------------------------------------------------


def test_runtime_and_image_are_exactly_one() -> None:
    both = _with("containers.knob-demo.image", "busybox")
    assert _refusal(both).field == "containers.knob-demo"
    neither = _with("containers.knob-demo", {"port": 8000})
    assert _refusal(neither).field == "containers.knob-demo"


def test_runtime_is_only_osprey() -> None:
    assert _refusal(_with("containers.knob-demo.runtime", "python")).field == (
        "containers.knob-demo.runtime"
    )


def test_env_is_a_list_of_host_variable_names() -> None:
    err = _refusal(_with("containers.knob-demo.env", {"KNOB_DEMO_MODE": "fast"}))
    assert err.field == "containers.knob-demo.env"
    assert err.rule == "env_names"
    assert "list of host variable names" in err.message
    assert _refusal(_with("containers.knob-demo.env", ["not a name"])).field.startswith(
        "containers.knob-demo.env"
    )


def test_fragment_names_the_container_directory() -> None:
    m = parse_manifest(
        _with("containers.knob-cache.fragment", "compose/knob-cache"), app="knob-demo"
    )
    assert m.containers["knob-cache"].fragment == "compose/knob-cache/"
    err = _refusal(_with("containers.knob-cache.fragment", "compose/knob-cache.yml.j2"))
    assert err.field == "containers.knob-cache.fragment"
    assert err.rule == "fragment_directory"
    assert "compose/knob-cache/" in err.message
    assert "docker-compose.yml.j2" in err.message
    other = _refusal(_with("containers.knob-cache.fragment", "compose/elsewhere/"))
    assert other.rule == "fragment_directory"


def test_fragment_is_for_image_containers_only() -> None:
    err = _refusal(_with("containers.knob-demo.fragment", "compose/knob-demo/"))
    assert err.field == "containers.knob-demo.fragment"


@pytest.mark.parametrize("port", [0, 70000])
def test_port_is_a_tcp_port(port: int) -> None:
    assert _refusal(_with("containers.knob-demo.port", port)).field == "containers.knob-demo.port"


# --- panel -------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["ui/", "/panel/knob-demo/ui/"])
def test_panel_entry_is_backend_relative(entry: str) -> None:
    assert _refusal(_with("panel.entry", entry)).field == "panel.entry"


def test_agent_routes_are_method_path_pairs() -> None:
    assert _refusal(_with("panel.agent_routes", [["get", "/x"]])).field.startswith(
        "panel.agent_routes.0"
    )
    assert _refusal(_with("panel.agent_routes", [["FETCH", "/x"]])).field.startswith(
        "panel.agent_routes.0"
    )
    assert _refusal(_with("panel.agent_routes", [["GET"]])).field.startswith("panel.agent_routes.0")
    assert _refusal(_with("panel.agent_routes", [["GET", "/panel/knob-demo/x"]])).field.startswith(
        "panel.agent_routes.0"
    )


# --- tools ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["knob-demo.tools", "", "1knob"])
def test_tools_module_is_a_python_module(module: str) -> None:
    assert _refusal(_with("tools.module", module)).field == "tools.module"


def test_shown_when_is_always_or_panel() -> None:
    assert _refusal(_with("tools.shown_when", "sometimes")).field == "tools.shown_when"


def test_state_path_is_backend_relative() -> None:
    data = copy.deepcopy(FULL)
    data["tools"]["approval"][0]["state_path"] = "/panel/knob-demo/api/state"
    assert _refusal(data).field == "tools.approval.0.state_path"


def test_input_digest_needs_at_least_one_input_key() -> None:
    data = copy.deepcopy(FULL)
    data["tools"]["approval"][0]["bind"]["input_digests"][0]["over"] = []
    assert _refusal(data).field == "tools.approval.0.bind.input_digests.0.over"


def test_old_bind_shape_is_refused() -> None:
    data = copy.deepcopy(FULL)
    data["tools"]["approval"][0]["bind"] = {"state": "/api/state", "fields": ["x"]}
    assert _refusal(data).field.startswith("tools.approval.0.bind.")


# --- vendor --------------------------------------------------------------------------


def test_package_vendor_entry_resolves_through_the_fixed_template() -> None:
    m = parse_manifest(FULL, app="knob-demo")
    by_url, by_package = m.vendor
    assert by_url.source_url == "https://example.org/lib.js"
    assert by_package.source_url == (
        "https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js"
    )


@pytest.mark.parametrize("version", ["^1.6.30", "~1.6", "1.x", ">=1.0", "latest", "1.6"])
def test_package_version_must_be_exact(version: str) -> None:
    data = copy.deepcopy(FULL)
    data["vendor"][1]["version"] = version
    assert _refusal(data).field == "vendor.1.package.version"


def test_vendor_sha256_is_a_hex_digest() -> None:
    data = copy.deepcopy(FULL)
    data["vendor"][0]["sha256"] = "abc"
    assert _refusal(data).field == "vendor.0.url.sha256"


def test_vendor_entry_names_one_source() -> None:
    data = copy.deepcopy(FULL)
    data["vendor"][0]["package"] = "uplot"
    assert _refusal(data).field.startswith("vendor.0")
    data = copy.deepcopy(FULL)
    data["vendor"][0] = {"into": "x", "sha256": "a" * 64}
    assert _refusal(data).field.startswith("vendor.0")


# --- loader ---------------------------------------------------------------------------


def _write_app(root: Path, name: str, data: dict[str, Any]) -> Path:
    app_dir = root / "apps" / name
    app_dir.mkdir(parents=True)
    (app_dir / "app.yml").write_text(yaml.safe_dump(data))
    return app_dir


def test_load_discovers_app_directories_sorted_by_name(tmp_path: Path) -> None:
    _write_app(tmp_path, "zeta", {"name": "zeta", "containers": {"zeta": {"runtime": "osprey"}}})
    _write_app(tmp_path, "knob-demo", FULL)
    manifests = load(tmp_path)
    assert [m.name for m in manifests] == ["knob-demo", "zeta"]


def test_load_without_an_apps_dir_is_empty(tmp_path: Path) -> None:
    assert load(tmp_path) == []


def test_load_ignores_the_provenance_sidecar(tmp_path: Path) -> None:
    app_dir = _write_app(tmp_path, "knob-demo", FULL)
    (app_dir / ".provenance").write_text("preset: control-assistant\n")
    assert [m.name for m in load(tmp_path)] == ["knob-demo"]


def test_load_refuses_a_name_that_differs_from_the_directory(tmp_path: Path) -> None:
    _write_app(tmp_path, "other", FULL)
    with pytest.raises(AppManifestError) as info:
        load(tmp_path)
    assert info.value.app == "other"
    assert info.value.field == "name"


def test_load_refuses_a_plain_app_file_and_points_at_revalidate(tmp_path: Path) -> None:
    (tmp_path / "apps").mkdir()
    (tmp_path / "apps" / "logbook.yml").write_text("name: logbook\n")
    with pytest.raises(AppManifestError) as info:
        load(tmp_path)
    assert info.value.app == "logbook"
    assert "osprey init --revalidate" in info.value.message


def test_load_refuses_a_directory_without_app_yml(tmp_path: Path) -> None:
    (tmp_path / "apps" / "knob-demo").mkdir(parents=True)
    with pytest.raises(AppManifestError) as info:
        load(tmp_path)
    assert info.value.app == "knob-demo"
    assert "app.yml" in info.value.message


def test_load_refuses_a_manifest_that_is_not_a_mapping(tmp_path: Path) -> None:
    app_dir = tmp_path / "apps" / "knob-demo"
    app_dir.mkdir(parents=True)
    (app_dir / "app.yml").write_text("- a\n- b\n")
    with pytest.raises(AppManifestError) as info:
        load(tmp_path)
    assert info.value.app == "knob-demo"


def test_refusal_carries_app_field_rule_and_message() -> None:
    err = _refusal(_with("containers.knob-demo.colour", 1))
    assert (err.app, err.field, err.rule) == (
        "knob-demo",
        "containers.knob-demo.colour",
        "extra_forbidden",
    )
    assert "knob-demo" in str(err) and "containers.knob-demo.colour" in str(err)


def test_refusal_carries_every_violation() -> None:
    data = _with("containers.knob-demo.colour", 1)
    data["panel"]["shade"] = 2
    data["tools"]["shown_when"] = "sometimes"
    err = _refusal(data)
    fields = [v.field for v in err.violations]
    assert fields == ["containers.knob-demo.colour", "panel.shade", "tools.shown_when"]
    assert all(v.app == "knob-demo" for v in err.violations)
    assert (err.field, err.rule) == ("containers.knob-demo.colour", "extra_forbidden")
    assert "panel.shade" in str(err) and "tools.shown_when" in str(err)


def test_a_single_refusal_carries_one_violation() -> None:
    err = _refusal(_with("panel.colour", 1))
    assert len(err.violations) == 1
    assert err.violations[0].field == err.field == "panel.colour"

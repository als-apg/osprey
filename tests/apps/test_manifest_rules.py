"""The render-independent application rules and the framework pin check."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest

from osprey import bluesky_tool_names
from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import AppManifest, launch_token_env, parse_manifest
from osprey.apps.rules import (
    HOOK_DISPATCHED_TOOLS,
    QUEUE_TOOL_PREFIX,
    AppSelection,
    LoginSelection,
    Reserved,
    builtin_credential_names,
    check_pins,
    deployment_packages,
    fixed_names,
    framework_pins,
    validate,
)

HOOKS = Path(__file__).resolve().parents[2] / "src/osprey/templates/claude_code/claude/hooks"

KNOB: dict[str, Any] = {
    "name": "knob",
    "containers": {
        "knob": {
            "runtime": "osprey",
            "port": 8100,
            "drives_target": True,
            "control_identity": True,
        },
        "knob-cache": {"image": "redis:7.2"},
    },
    "panel": {
        "label": "KNOB",
        "container": "knob",
        "launch_token": True,
        "agent_routes": [["GET", "/api/state"], ["POST", "/api/write"]],
    },
    "tools": {
        "module": "knob.tools",
        "shown_when": "panel",
        "deny_in_jobs": True,
        "read": ["read_knob"],
        "approval": [
            {
                "tool": "set_knob",
                "prompt": "Set {{ input.value }}",
                "state_path": "/api/state",
                "bind": {
                    "state_fields": [{"field": "approved_readback", "key": "readback"}],
                    "input_digests": [{"field": "approved_digest", "over": ["value"]}],
                },
                "require": {
                    "all_of": ["approved_target"],
                    "one_of": [["approved_digest", "approved_readback"]],
                },
            }
        ],
    },
    "vendor": [
        {"url": "https://example.org/lib.js", "into": "panel/vendor/lib.js", "sha256": "a" * 64}
    ],
}

EMPTY = Reserved(services=frozenset(), panels=frozenset(), servers=frozenset(), names=frozenset())


def _manifest(data: dict[str, Any] | None = None, **changes: Any) -> AppManifest:
    raw = copy.deepcopy(KNOB if data is None else data)
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = raw
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if value is _DROP:
            del node[leaf]
        else:
            node[leaf] = value
    return parse_manifest(raw, app=raw["name"])


_DROP = object()


def _person(tier: str = "allowed", **apps: bool) -> LoginSelection:
    return LoginSelection(
        audience="person",
        apps={n: AppSelection(shown=s) for n, s in apps.items()},
        tier=tier,  # type: ignore[arg-type]
        shared_cards=(),
    )


def _refused(
    manifests: list[AppManifest],
    *,
    logins: dict[str, LoginSelection] | None = None,
    reserved: Reserved = EMPTY,
) -> AppManifestError:
    with pytest.raises(AppManifestError) as info:
        validate(manifests, logins=logins or {}, reserved=reserved)
    return info.value


def _rules(err: AppManifestError) -> list[str]:
    return [v.rule for v in err.violations]


# --- types -------------------------------------------------------------------


def test_the_types_are_frozen() -> None:
    selection = _person(knob=True)
    with pytest.raises(AttributeError):
        selection.tier = "never"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        AppSelection(shown=True).shown = False  # type: ignore[misc]
    with pytest.raises(AttributeError):
        EMPTY.names = frozenset()  # type: ignore[misc]


def test_a_login_arms_an_app_it_shows_under_tier_allowed() -> None:
    assert _person("allowed", knob=True).arms("knob")
    assert not _person("never", knob=True).arms("knob")
    assert not _person("allowed", knob=False).arms("knob")
    assert not _person("allowed").arms("knob")


def test_a_valid_set_passes() -> None:
    validate([_manifest()], logins={"ops": _person(knob=True)}, reserved=EMPTY)


# --- containers ----------------------------------------------------------------


def test_drives_target_needs_the_osprey_runtime() -> None:
    data = copy.deepcopy(KNOB)
    data["containers"]["knob-cache"]["drives_target"] = True
    err = _refused([_manifest(data)])
    assert _rules(err) == ["drives_target_runtime"]
    assert err.field == "containers.knob-cache.drives_target"


def test_control_identity_needs_drives_target() -> None:
    err = _refused([_manifest(containers__knob__drives_target=False)])
    assert _rules(err) == ["control_identity_drives_target"]
    assert err.field == "containers.knob.control_identity"


def test_container_keys_are_unique_across_apps_and_services() -> None:
    other = _manifest({"name": "other", "containers": {"knob-cache": {"image": "redis:7.2"}}})
    err = _refused([_manifest(), other])
    assert _rules(err) == ["container_key_unique"]
    assert err.app == "other" and "knob" in err.message
    err = _refused(
        [_manifest()],
        reserved=Reserved(frozenset({"knob-cache"}), frozenset(), frozenset(), frozenset()),
    )
    assert _rules(err) == ["container_key_unique"]


# --- panel -----------------------------------------------------------------------


def test_shown_when_panel_needs_a_panel() -> None:
    m = _manifest(panel=_DROP, tools__approval=[])
    err = _refused([m])
    assert _rules(err) == ["shown_when_panel"]
    assert err.field == "tools.shown_when"


def test_the_panel_is_served_by_one_of_the_apps_containers() -> None:
    err = _refused([_manifest(panel__container="elsewhere")])
    assert _rules(err) == ["panel_container"]
    assert "elsewhere" in err.message


def test_a_panel_container_needs_no_port_pin() -> None:
    validate([_manifest(containers__knob__port=_DROP)], logins={}, reserved=EMPTY)


def test_panel_ids_are_unique_across_apps_and_web_panels() -> None:
    other = _manifest(
        {
            "name": "other",
            "containers": {"other": {"runtime": "osprey"}},
            "panel": {"id": "knob", "label": "O", "container": "other"},
        }
    )
    assert _rules(_refused([_manifest(), other])) == ["panel_id_unique"]
    err = _refused(
        [_manifest()],
        reserved=Reserved(frozenset(), frozenset({"knob"}), frozenset(), frozenset()),
    )
    assert _rules(err) == ["panel_id_unique"]
    assert err.field == "panel.id"


# --- vendor --------------------------------------------------------------------------


@pytest.mark.parametrize("into", ["/etc/lib.js", "../lib.js", "panel/../../lib.js", ".", "a\\b"])
def test_vendor_into_stays_inside_the_app_directory(into: str) -> None:
    err = _refused([_manifest(vendor__0__into=into)])
    assert _rules(err) == ["vendor_into"]
    assert err.field == "vendor.0.into"


def test_vendor_into_may_walk_inside_the_app_directory() -> None:
    validate([_manifest(vendor__0__into="panel/x/../lib.js")], logins={}, reserved=EMPTY)


# --- names -----------------------------------------------------------------------------


def test_fixed_names_cover_servers_panels_and_services() -> None:
    from osprey.cli.build_profile_load import bundled_service_names
    from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    names = fixed_names()
    assert set(FRAMEWORK_SERVERS) <= names
    assert BUILTIN_PANELS | UNIVERSAL_PANELS <= names
    assert bundled_service_names() <= names


def test_an_app_named_like_a_framework_server_is_refused_without_enabling_it() -> None:
    reserved = Reserved(frozenset(), frozenset(), frozenset(), fixed_names())
    phoebus = _manifest({"name": "phoebus", "containers": {"phoebus": {"runtime": "osprey"}}})
    err = _refused([phoebus], reserved=reserved)
    assert _rules(err) == ["reserved_name"]
    assert err.app == "phoebus" and err.field == "name"


def test_a_tool_server_name_is_unique_against_mcp_servers() -> None:
    err = _refused(
        [_manifest()],
        reserved=Reserved(frozenset(), frozenset(), frozenset({"knob"}), frozenset()),
    )
    assert _rules(err) == ["server_name_unique"]
    assert err.field == "tools"


def test_an_app_name_appears_once() -> None:
    assert "name_unique" in _rules(_refused([_manifest(), _manifest()]))


# --- tool names -------------------------------------------------------------------------


def test_hook_dispatched_names_are_the_hooks_own() -> None:
    text = (HOOKS / "osprey_approval.py").read_text(encoding="utf-8")
    for name in HOOK_DISPATCHED_TOOLS:
        assert f'short_name == "{name}"' in text, name


def test_the_queue_prefix_is_the_hooks_own() -> None:
    text = (HOOKS / "osprey_approval.py").read_text(encoding="utf-8")
    assert re.search(rf'^_QUEUE_TOOL_PREFIX = "{QUEUE_TOOL_PREFIX}"$', text, re.MULTILINE)
    assert all(n.startswith(QUEUE_TOOL_PREFIX) for n in bluesky_tool_names.QUEUE_TOOLS)


@pytest.mark.parametrize("name", HOOK_DISPATCHED_TOOLS)
def test_an_app_tool_may_not_take_a_hook_dispatched_name(name: str) -> None:
    err = _refused([_manifest(tools__approval__0__tool=name)])
    assert _rules(err) == ["builtin_tool_name"]
    assert err.field == "tools.approval.0.tool" and name in err.message


@pytest.mark.parametrize("name", [*bluesky_tool_names.QUEUE_CONTROL_TOOLS, "queue_anything"])
def test_an_app_tool_may_not_take_a_queue_name(name: str) -> None:
    err = _refused([_manifest(tools__read=[name])])
    assert _rules(err) == ["queue_tool_name"]
    assert err.field == "tools.read.0"


# --- approval entries ----------------------------------------------------------------------


def test_state_path_needs_a_panel() -> None:
    m = _manifest(panel=_DROP, tools__shown_when="always")
    assert _rules(_refused([m])) == ["state_path_panel"]


@pytest.mark.parametrize("routes", [[], [["POST", "/api/state"]], [["GET", "/api/other"]]])
def test_state_path_needs_a_get_agent_route(routes: list[list[str]]) -> None:
    err = _refused([_manifest(panel__agent_routes=routes)])
    assert _rules(err) == ["state_path_route"]
    assert err.field == "tools.approval.0.state_path"


def test_a_tool_that_binds_fetched_state_may_not_spell_deny_in_jobs_false() -> None:
    err = _refused([_manifest(tools__deny_in_jobs=False)])
    assert _rules(err) == ["state_path_in_jobs"]
    assert "set_knob" in err.message and "binds fetched state" in err.message
    validate([_manifest(tools__deny_in_jobs=_DROP)], logins={}, reserved=EMPTY)


def test_require_names_approved_target_or_a_bound_field() -> None:
    m = _manifest(
        tools__approval__0__require={
            "all_of": ["approved_target", "approved_setpoint"],
            "one_of": [["approved_digest", "approved_other"]],
        }
    )
    err = _refused([m])
    assert [v.field for v in err.violations] == [
        "tools.approval.0.require.all_of.1",
        "tools.approval.0.require.one_of.0",
    ]
    assert set(_rules(err)) == {"approved_fields"}
    assert "approved_setpoint" in err.message


def test_a_bound_field_is_named_approved() -> None:
    m = _manifest(tools__approval__0__bind__state_fields__0__field="readback")
    m2 = _manifest(
        tools__approval__0__bind__state_fields__0__field="readback",
        tools__approval__0__require__one_of=[["approved_digest"]],
    )
    assert "approved_fields" in _rules(_refused([m]))
    err = _refused([m2])
    assert _rules(err) == ["bound_field_name"]
    assert err.field == "tools.approval.0.bind.state_fields.0.field"


def test_a_field_is_bound_once() -> None:
    m = _manifest(tools__approval__0__bind__input_digests__0__field="approved_readback")
    err = _refused(
        [
            _manifest(
                tools__approval__0__bind__input_digests__0__field="approved_readback",
                tools__approval__0__require__one_of=[["approved_readback"]],
            )
        ]
    )
    assert _rules(err) == ["bound_twice"]
    assert err.field == "tools.approval.0.bind.input_digests.0.field"
    assert "bound_twice" in _rules(_refused([m]))


def test_the_tool_server_declares_no_separate_field_list() -> None:
    with pytest.raises(AppManifestError) as info:
        _manifest(tools__stamp_fields=["approved_target"])
    assert info.value.rule == "extra_forbidden"


# --- launch token -------------------------------------------------------------------------


def test_builtin_credential_names_include_the_lane_tokens() -> None:
    from osprey.bluesky_bridge_connection import SECOND_LANE_KEYS, lane_env_prefix

    names = builtin_credential_names()
    assert "BLUESKY_LAUNCH_TOKEN" in names
    for lane in SECOND_LANE_KEYS.values():
        assert f"{lane_env_prefix(lane)}_LAUNCH_TOKEN" in names


def test_a_launch_token_may_not_collide_with_a_builtin_credential() -> None:
    m = _manifest(
        {
            "name": "bluesky",
            "containers": {"bluesky-app": {"runtime": "osprey"}},
            "panel": {"label": "B", "container": "bluesky-app", "launch_token": True},
        }
    )
    assert launch_token_env("bluesky") in builtin_credential_names()
    err = _refused([m])
    assert _rules(err) == ["launch_token_env"]
    assert "BLUESKY_LAUNCH_TOKEN" in err.message
    m = _manifest({"name": "bluesky", "containers": {"bluesky-app": {"runtime": "osprey"}}})
    validate([m], logins={}, reserved=EMPTY)


# --- logins ---------------------------------------------------------------------------------


def test_a_shared_login_may_not_be_tier_allowed() -> None:
    shared = LoginSelection(
        audience="shared",
        apps={"knob": AppSelection(shown=True)},
        tier="allowed",
        shared_cards=("control-room", "night-shift"),
    )
    err = _refused([_manifest()], logins={"ops": shared})
    assert _rules(err) == ["shared_login_tier"]
    assert err.app == "" and err.field == "logins.ops"
    assert "control-room" in err.message and "night-shift" in err.message
    sealed = LoginSelection("shared", {"knob": AppSelection(shown=True)}, "never", ("c",))
    validate([_manifest()], logins={"ops": sealed}, reserved=EMPTY)


# --- one refusal ------------------------------------------------------------------------------


def test_every_violation_is_collected_into_one_refusal() -> None:
    data = copy.deepcopy(KNOB)
    data["containers"]["knob-cache"]["drives_target"] = True
    data["panel"]["container"] = "elsewhere"
    data["vendor"][0]["into"] = "../x"
    err = _refused([_manifest(data)])
    assert _rules(err) == ["drives_target_runtime", "panel_container", "vendor_into"]
    assert all(v.app == "knob" for v in err.violations)
    text = str(err)
    assert "drives_target" in text and "elsewhere" in text and "../x" in text


# --- pins ---------------------------------------------------------------------------------------


def _pyproject(directory: Path, *deps: str, name: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pyproject.toml"
    listed = ", ".join(f'"{d}"' for d in deps)
    path.write_text(f'[project]\nname = "{name or directory.name}"\ndependencies = [{listed}]\n')
    return path


def test_framework_pins_are_the_exact_requirements_of_every_extra() -> None:
    import importlib.metadata

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pins = framework_pins()
    exact = {}
    for raw in importlib.metadata.requires("osprey-framework") or ():
        req = Requirement(raw)
        specs = list(req.specifier)
        if len(specs) == 1 and specs[0].operator == "==":
            exact[canonicalize_name(req.name)] = specs[0].version
    assert pins == exact
    assert pins, "osprey-framework pins at least one requirement"


def test_a_compatible_requirement_passes(tmp_path: Path) -> None:
    path = _pyproject(tmp_path / "apps" / "knob", "numpy>=1", "lume-base>=0.4", "lume-base")
    check_pins([path], {"lume-base": "0.5.0"})


def test_a_requirement_that_moves_a_pin_is_refused(tmp_path: Path) -> None:
    knob = _pyproject(tmp_path / "apps" / "knob", "Lume_Base>=0.6", "numpy")
    extra = _pyproject(tmp_path / "libs" / "helper", "claude-agent-sdk==0.1.0", name="helper")
    with pytest.raises(AppManifestError) as info:
        check_pins([knob, extra], {"lume-base": "0.5.0", "claude-agent-sdk": "0.2.136"})
    err = info.value
    assert _rules(err) == ["pin_moved", "pin_moved"]
    first, second = err.violations
    assert first.app == "knob" and first.source == str(knob)
    assert "lume-base" in first.message.lower().replace("_", "-")
    assert "0.5.0" in first.message and ">=0.6" in first.message
    assert second.app == "helper"
    assert "0.2.136" in second.message and "==0.1.0" in second.message


def test_a_direct_reference_to_a_pinned_package_is_refused(tmp_path: Path) -> None:
    path = _pyproject(tmp_path / "apps" / "knob", "lume-base @ https://example.org/x.whl")
    with pytest.raises(AppManifestError) as info:
        check_pins([path], {"lume-base": "0.5.0"})
    assert info.value.rule == "pin_moved"


def test_an_unreadable_pyproject_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "apps" / "knob" / "pyproject.toml"
    path.parent.mkdir(parents=True)
    path.write_text("[project\n")
    with pytest.raises(AppManifestError) as info:
        check_pins([path], {})
    assert info.value.rule == "pyproject"


def test_deployment_packages_enumerates_apps_and_local_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "deploy"
    zeta = _pyproject(root / "apps" / "zeta")
    alpha = _pyproject(root / "apps" / "alpha")
    (root / "apps" / "no-python").mkdir(parents=True)
    rel = _pyproject(root / "libs" / "helper")
    absolute = _pyproject(tmp_path / "elsewhere" / "tool")
    url = _pyproject(tmp_path / "by-url")
    found = deployment_packages(
        root,
        [
            "numpy>=1",
            "./libs/helper",
            str(absolute.parent),
            f"by-url @ {url.parent.as_uri()}",
            "./missing",
            "-e ./libs/helper",
        ],
    )
    assert found == [alpha, zeta, rel.resolve(), absolute.resolve(), url.resolve()]


def test_deployment_packages_without_apps_or_dependencies_is_empty(tmp_path: Path) -> None:
    assert deployment_packages(tmp_path, []) == []

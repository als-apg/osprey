"""Tests for :meth:`BuildProfile.validate` — the profile consistency checks.

Pins the error-accumulation contract: ``validate()`` never fails fast. Every
check appends to one ``errors`` list and, only after all of them have run, a
single :class:`~osprey.errors.BuildProfileError` is raised whose message is
``"Build profile validation failed:\\n  - "`` followed by every failure joined
by ``"\\n  - "``. A profile with four unrelated faults therefore reports all
four in one raise, so a build author fixes them in one pass instead of one
rebuild per typo. The structural checks that ``continue`` (a service with a
bundled ``osprey.``-prefixed template, a ``panel_presets`` entry that is not a
list, a ``categories`` entry that is not a mapping) skip only their own
remaining sub-checks — they never short-circuit the rest of the validator.

Complements the block-scoped suites (``test_profile_schema.py`` for
``dispatch:``, ``test_build_profile.py`` for ``bluesky_web:``) by covering
the per-field branches those files leave untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from osprey.cli.build_profile import (
    BlueskyConfig,
    BuildProfile,
    DispatchConfig,
    EnvConfig,
    LifecycleConfig,
    LifecycleStep,
    McpServerDef,
    ServiceDef,
    VAConfig,
    _parse_profile,
)
from osprey.errors import BuildProfileError


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    message = str(exc.value)
    header, _, body = message.partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _graph_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """The validation errors about the graph store alone, or ``[]`` if it validates.

    An attached profile fails the qmd-sidecar rule on the same run — its render
    keeps hybrid search on over ``services: {}`` — and these tests are about
    the store rule, not the count of rules.
    """
    try:
        profile.validate(profile_dir)
    except BuildProfileError as exc:
        _, _, body = str(exc).partition(":\n  - ")
        return [error for error in body.split("\n  - ") if "services.graphdb" in error]
    return []


def _write_triggers(tmp_path: Path, name: str = "trig.yml") -> str:
    """Write a minimal triggers file into ``tmp_path`` and return its name."""
    (tmp_path / name).write_text("triggers: []", encoding="utf-8")
    return name


# --- error-accumulation contract ------------------------------------------


def test_validate_accumulates_every_failure_into_one_error(tmp_path: Path) -> None:
    """Four unrelated faults are reported together, not one raise per fault."""
    profile = BuildProfile(
        name="",
        deploy_services="yes",  # type: ignore[arg-type]
        tier=2,
        dependencies=["   "],
    )
    errors = _errors(profile, tmp_path)
    assert errors == [
        "Profile 'name' is required",
        "deploy_services must be a boolean (got str)",
        "tier must be 1 or 3 (got 2)",
        "Dependency must be a non-empty string: '   '",
    ]


def test_valid_profile_validates_silently(tmp_path: Path) -> None:
    """A profile with no faults returns None rather than raising."""
    assert BuildProfile(name="x").validate(tmp_path) is None


# --- scalar fields: name, deploy_services, tier, channel_finder_mode -------


def test_missing_name_is_rejected(tmp_path: Path) -> None:
    """An empty 'name' is a validation failure."""
    assert _errors(BuildProfile(name=""), tmp_path) == ["Profile 'name' is required"]


def test_non_boolean_deploy_services_is_rejected(tmp_path: Path) -> None:
    """deploy_services must be a bool; the message names the offending type."""
    profile = BuildProfile(name="x", deploy_services=1)  # type: ignore[arg-type]
    assert _errors(profile, tmp_path) == ["deploy_services must be a boolean (got int)"]


def test_tier_outside_one_or_three_is_rejected(tmp_path: Path) -> None:
    """Only tiers 1 and 3 ship a channel database."""
    assert _errors(BuildProfile(name="x", tier=2), tmp_path) == ["tier must be 1 or 3 (got 2)"]


def test_tier_one_with_hierarchical_mode_is_rejected(tmp_path: Path) -> None:
    """Tier 1 ships only the in_context DB, so a paradigm mismatch fails here."""
    profile = BuildProfile(name="x", tier=1, channel_finder_mode="hierarchical")
    errors = _errors(profile, tmp_path)
    assert errors == [
        "tier 1 requires channel_finder_mode: in_context (got channel_finder_mode: 'hierarchical')"
    ]


def test_explicit_tier_with_graph_mode_is_rejected(tmp_path: Path) -> None:
    """``graph``'s store is a seeded service, so no tier selects anything for it.

    The rule is checked ahead of the tier-1/in_context rule, so a ``tier: 1``
    graph profile reports the graph message rather than being told to switch to
    in_context — the fix is to drop ``tier``, not to change the paradigm.
    """
    for tier in (1, 3):
        profile = BuildProfile(name="x", tier=tier, channel_finder_mode="graph")
        assert _errors(profile, tmp_path) == [
            f"channel_finder_mode: graph has no tiered artifacts; omit tier (got tier: {tier})"
        ]


def test_graph_mode_without_tier_validates(tmp_path: Path) -> None:
    """Omitting ``tier`` is the supported way to build the graph paradigm."""
    BuildProfile(name="x", channel_finder_mode="graph").validate(tmp_path)


def test_unknown_channel_finder_mode_is_rejected(tmp_path: Path) -> None:
    """A channel_finder_mode outside the known paradigms is a typo, not a mode."""
    profile = BuildProfile(name="x", channel_finder_mode="in-context")
    (error,) = _errors(profile, tmp_path)
    assert "channel_finder_mode must be one of" in error
    assert "'in-context'" in error


def test_resolved_tier_returns_explicit_tier() -> None:
    """An explicit tier wins over the paradigm-aware default."""
    assert BuildProfile(name="x", tier=1, channel_finder_mode="in_context").resolved_tier() == 1


def test_resolved_tier_defaults_from_paradigm() -> None:
    """With no explicit tier, the paradigm picks the default (in_context -> 1)."""
    assert BuildProfile(name="x", channel_finder_mode="in_context").resolved_tier() == 1
    assert BuildProfile(name="x", channel_finder_mode="hierarchical").resolved_tier() == 3


# --- convention directories -----------------------------------------------


def test_wellformed_convention_dirs_validate(tmp_path: Path) -> None:
    """A profile whose convention directories are well shaped passes."""
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "facility-ops.md").write_text("# ops", encoding="utf-8")
    (tmp_path / "project" / "docs").mkdir(parents=True)
    (tmp_path / "project" / "docs" / "runbook.md").write_text("# run", encoding="utf-8")
    BuildProfile(name="x").validate(tmp_path)


def test_misshapen_convention_source_is_rejected(tmp_path: Path) -> None:
    """A loose file where a directory-shaped convention expects a directory."""
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "loose.md").write_text("hi", encoding="utf-8")
    (error,) = _errors(BuildProfile(name="x"), tmp_path)
    assert "skills/loose.md" in error
    assert "one directory per skill" in error


def test_reserved_mirror_path_is_rejected_naming_its_channel(tmp_path: Path) -> None:
    """The project/ mirror may not write a path the build itself owns."""
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / ".mcp.json").write_text("{}", encoding="utf-8")
    (error,) = _errors(BuildProfile(name="x"), tmp_path)
    assert "project/.mcp.json" in error
    assert "`mcp_servers:`" in error


# --- mcp_servers ----------------------------------------------------------


def test_mcp_server_without_command_or_url_is_rejected(tmp_path: Path) -> None:
    """A server with neither transport declared cannot be launched."""
    profile = BuildProfile(name="x", mcp_servers={"empty": McpServerDef()})
    assert _errors(profile, tmp_path) == ["MCP server 'empty' missing 'command' or 'url'"]


def test_mcp_server_with_url_only_validates(tmp_path: Path) -> None:
    """An HTTP server needs no command."""
    profile = BuildProfile(
        name="x", mcp_servers={"http": McpServerDef(url="http://localhost:8020/mcp")}
    )
    profile.validate(tmp_path)


# --- services -------------------------------------------------------------


def test_service_without_template_is_rejected(tmp_path: Path) -> None:
    """A service must name a template dir (or a bundled osprey.* template)."""
    profile = BuildProfile(name="x", services={"svc": ServiceDef(template="")})
    assert _errors(profile, tmp_path) == ["Service 'svc' missing 'template'"]


def test_service_template_dir_not_found_is_rejected(tmp_path: Path) -> None:
    """A profile-relative template that is not a directory fails."""
    profile = BuildProfile(name="x", services={"svc": ServiceDef(template="services/svc")})
    (error,) = _errors(profile, tmp_path)
    assert error == f"Service 'svc' template dir not found: {tmp_path / 'services/svc'}"


def test_service_template_dir_without_compose_is_rejected(tmp_path: Path) -> None:
    """A template dir that exists must still carry docker-compose.yml.j2."""
    (tmp_path / "services" / "svc").mkdir(parents=True)
    profile = BuildProfile(name="x", services={"svc": ServiceDef(template="services/svc")})
    assert _errors(profile, tmp_path) == [
        "Service 'svc' template dir missing docker-compose.yml.j2"
    ]


def test_service_template_dir_with_compose_validates(tmp_path: Path) -> None:
    """A complete template dir produces no failure."""
    svc_dir = tmp_path / "services" / "svc"
    svc_dir.mkdir(parents=True)
    (svc_dir / "docker-compose.yml.j2").write_text("services: {}", encoding="utf-8")
    BuildProfile(name="x", services={"svc": ServiceDef(template="services/svc")}).validate(tmp_path)


# --- lifecycle steps ------------------------------------------------------


def test_lifecycle_step_missing_name_and_run_is_rejected(tmp_path: Path) -> None:
    """Both required step fields are reported, and the phase is named."""
    profile = BuildProfile(
        name="x",
        lifecycle=LifecycleConfig(pre_build=[LifecycleStep(name="", run="")]),
    )
    assert _errors(profile, tmp_path) == [
        "Lifecycle pre_build step missing 'name'",
        "Lifecycle pre_build step missing 'run'",
    ]


def test_lifecycle_steps_are_checked_in_every_phase(tmp_path: Path) -> None:
    """pre_build, post_build and validate steps all go through the same checks."""
    bad = [LifecycleStep(name="", run="echo hi")]
    profile = BuildProfile(
        name="x",
        lifecycle=LifecycleConfig(pre_build=list(bad), post_build=list(bad), validate=list(bad)),
    )
    assert _errors(profile, tmp_path) == [
        "Lifecycle pre_build step missing 'name'",
        "Lifecycle post_build step missing 'name'",
        "Lifecycle validate step missing 'name'",
    ]


def test_lifecycle_step_absolute_cwd_is_rejected(tmp_path: Path) -> None:
    """A step cwd is resolved inside the built project, so it must be relative."""
    profile = BuildProfile(
        name="x",
        lifecycle=LifecycleConfig(post_build=[LifecycleStep(name="s", run="echo hi", cwd="/tmp")]),
    )
    assert _errors(profile, tmp_path) == [
        "Lifecycle post_build step 's' cwd must be relative without '..': /tmp"
    ]


def test_lifecycle_step_parent_traversal_cwd_is_rejected(tmp_path: Path) -> None:
    """A '..' component in a step cwd escapes the project dir."""
    profile = BuildProfile(
        name="x",
        lifecycle=LifecycleConfig(
            validate=[LifecycleStep(name="s", run="echo hi", cwd="../elsewhere")]
        ),
    )
    assert _errors(profile, tmp_path) == [
        "Lifecycle validate step 's' cwd must be relative without '..': ../elsewhere"
    ]


def test_lifecycle_step_relative_cwd_validates(tmp_path: Path) -> None:
    """A plain relative cwd is accepted."""
    profile = BuildProfile(
        name="x",
        lifecycle=LifecycleConfig(pre_build=[LifecycleStep(name="s", run="echo hi", cwd="sub")]),
    )
    profile.validate(tmp_path)


def test_lifecycle_step_non_positive_timeout_is_rejected(tmp_path: Path) -> None:
    """A zero or negative timeout would abort the step before it starts."""
    profile = BuildProfile(
        name="x",
        lifecycle=LifecycleConfig(pre_build=[LifecycleStep(name="s", run="echo hi", timeout=0)]),
    )
    assert _errors(profile, tmp_path) == [
        "Lifecycle pre_build step 's' timeout must be positive: 0"
    ]


# --- env ------------------------------------------------------------------


def test_invalid_env_var_name_is_rejected(tmp_path: Path) -> None:
    """Required env var names must be upper-snake shell identifiers."""
    profile = BuildProfile(name="x", env=EnvConfig(required=["OK_VAR", "not-a-var"]))
    assert _errors(profile, tmp_path) == ["Invalid env var name: not-a-var"]


def test_pinned_env_var_names_are_held_to_the_required_pattern(tmp_path: Path) -> None:
    """``pinned`` names the same kind of thing as ``required``, one message per name."""
    profile = BuildProfile(name="x", env=EnvConfig(pinned=["OK_VAR", "not-a-var", "also bad"]))
    assert _errors(profile, tmp_path) == [
        "Invalid env.pinned var name: 'not-a-var'",
        "Invalid env.pinned var name: 'also bad'",
    ]


def test_pinned_env_entries_that_are_not_strings_are_rejected(tmp_path: Path) -> None:
    """A YAML author who writes a bare number gets a name error, not a crash."""
    profile = BuildProfile(name="x", env=EnvConfig(pinned=[7]))  # type: ignore[list-item]
    assert _errors(profile, tmp_path) == ["Invalid env.pinned var name: 7"]


def test_pinned_env_block_that_is_not_a_list_is_rejected(tmp_path: Path) -> None:
    """A scalar where a list belongs would otherwise validate character by character."""
    profile = BuildProfile(name="x", env=EnvConfig(pinned="OSPREY_TOKEN"))  # type: ignore[arg-type]
    assert _errors(profile, tmp_path) == ["env.pinned must be a list of env var names (got str)"]


def test_pinned_env_names_reach_the_profile_from_yaml() -> None:
    """The key is parsed, not silently dropped — the validator has to see it."""
    profile = _parse_profile({"name": "x", "env": {"pinned": ["OSPREY_TOKEN"]}})
    assert profile.env.pinned == ["OSPREY_TOKEN"]
    assert profile.env.required == []


def test_a_profile_without_pinned_env_names_declares_none() -> None:
    """Absent means empty, so nothing changes for a profile written before pins."""
    assert _parse_profile({"name": "x", "env": {"required": ["OSPREY_TOKEN"]}}).env.pinned == []


def test_missing_env_file_is_rejected(tmp_path: Path) -> None:
    """env.file names a profile-relative file to copy; it must exist."""
    profile = BuildProfile(name="x", env=EnvConfig(file="env.template"))
    (error,) = _errors(profile, tmp_path)
    assert error.startswith("env.file not found: env.template")
    assert str(tmp_path / "env.template") in error


def test_present_env_file_validates(tmp_path: Path) -> None:
    """An env.file that exists produces no failure."""
    (tmp_path / "env.template").write_text("FOO=bar", encoding="utf-8")
    BuildProfile(name="x", env=EnvConfig(file="env.template")).validate(tmp_path)


# --- dependencies ---------------------------------------------------------


def test_blank_and_non_string_dependencies_are_rejected(tmp_path: Path) -> None:
    """Each dependency must be a non-empty string spec."""
    profile = BuildProfile(name="x", dependencies=["numpy", "", 7])  # type: ignore[list-item]
    assert _errors(profile, tmp_path) == [
        "Dependency must be a non-empty string: ''",
        "Dependency must be a non-empty string: 7",
    ]


# --- requires_osprey_version ----------------------------------------------


def test_invalid_requires_osprey_version_is_rejected(tmp_path: Path) -> None:
    """A non-PEP-440 specifier is caught at validate time, not at install time."""
    profile = BuildProfile(name="x", requires_osprey_version="latest")
    (error,) = _errors(profile, tmp_path)
    assert error.startswith("Invalid requires_osprey_version specifier: 'latest'")
    assert "PEP 440" in error


def test_valid_requires_osprey_version_validates(tmp_path: Path) -> None:
    """A PEP 440 specifier set passes."""
    BuildProfile(name="x", requires_osprey_version=">=0.12.0,<1.0").validate(tmp_path)


# --- default_panel / panel_presets membership -----------------------------


def test_unknown_default_panel_is_rejected(tmp_path: Path) -> None:
    """A default_panel typo would silently fall back at runtime, so it fails here."""
    profile = BuildProfile(name="x", default_panel="areil")
    (error,) = _errors(profile, tmp_path)
    assert error.startswith("Unknown default_panel 'areil'")
    assert "'web.panels.areil.url' config override" in error


def test_default_panel_declared_in_web_panels_is_known(tmp_path: Path) -> None:
    """A url-backed custom panel listed in web_panels is a valid default_panel."""
    profile = BuildProfile(
        name="x",
        web_panels=["ops"],
        default_panel="ops",
        config={"web.panels.ops.url": "http://localhost:8099/"},
    )
    profile.validate(tmp_path)
    assert profile._is_known_panel_id("ops") is True


def test_is_known_panel_id_accepts_builtin_and_rejects_typo() -> None:
    """The shared membership predicate takes built-ins and rejects unbacked ids."""
    profile = BuildProfile(name="x")
    assert profile._is_known_panel_id("ariel") is True
    assert profile._is_known_panel_id("areil") is False


def test_unknown_panel_presets_member_is_rejected(tmp_path: Path) -> None:
    """Preset members resolve through the same predicate as default_panel."""
    profile = BuildProfile(name="x", panel_presets={"Ops": ["ariel", "areil"]})
    (error,) = _errors(profile, tmp_path)
    assert error.startswith("Unknown panel_presets['Ops'] member 'areil'")


def test_non_list_panel_presets_entry_is_rejected(tmp_path: Path) -> None:
    """A preset whose value is not a list is reported once and skipped."""
    profile = _parse_profile({"name": "x", "panel_presets": {"Ops": "ariel"}})
    assert _errors(profile, tmp_path) == [
        "panel_presets['Ops'] must be a list of panel ids (got str)"
    ]


# --- artifact_server ------------------------------------------------------


def test_non_mapping_category_is_rejected(tmp_path: Path) -> None:
    """A category that is not a mapping is reported once, then skipped."""
    profile = _parse_profile(
        {"name": "x", "artifact_server": {"categories": {"ops": ["label", "color"]}}}
    )
    assert _errors(profile, tmp_path) == ["Category 'ops' must be a mapping with label and color"]


def test_category_missing_label_is_rejected(tmp_path: Path) -> None:
    """A category needs a string 'label'."""
    profile = BuildProfile(name="x", artifact_server={"categories": {"ops": {"color": "#aabbcc"}}})
    assert _errors(profile, tmp_path) == ["Category 'ops' missing or invalid 'label'"]


def test_category_with_non_hex_color_is_rejected(tmp_path: Path) -> None:
    """Category colors must be #RRGGBB hex."""
    profile = BuildProfile(
        name="x", artifact_server={"categories": {"ops": {"label": "Ops", "color": "red"}}}
    )
    assert _errors(profile, tmp_path) == [
        "Category 'ops' missing or invalid 'color' (must be #RRGGBB)"
    ]


def test_well_formed_category_validates(tmp_path: Path) -> None:
    """A label + #RRGGBB color pair passes."""
    BuildProfile(
        name="x", artifact_server={"categories": {"ops": {"label": "Ops", "color": "#A1B2C3"}}}
    ).validate(tmp_path)


def test_artifact_server_unknown_subkey_rejected(tmp_path: Path) -> None:
    """Only host/port/auto_launch/categories are supported under artifact_server."""
    profile = BuildProfile(name="x", artifact_server={"prot": "http"})
    assert _errors(profile, tmp_path) == [
        "artifact_server.prot is not a supported key "
        "(must be one of ['auto_launch', 'categories', 'host', 'port'])"
    ]


def test_artifact_server_scalar_overrides_validate(tmp_path: Path) -> None:
    """host/port/auto_launch overrides pass through validation."""
    BuildProfile(
        name="x", artifact_server={"host": "0.0.0.0", "port": 9086, "auto_launch": False}
    ).validate(tmp_path)


# --- dispatch numeric bounds ----------------------------------------------


def test_dispatch_port_bounds_are_rejected(tmp_path: Path) -> None:
    """dispatcher_port and worker_port_base must be inside the TCP port range."""
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(
        name="x",
        dispatch=DispatchConfig(triggers=triggers, dispatcher_port=0, worker_port_base=70000),
    )
    assert _errors(profile, tmp_path) == [
        "dispatch.dispatcher_port must be in 1..65535 (got 0)",
        "dispatch.worker_port_base must be in 1..65535 (got 70000)",
    ]


def test_dispatch_queue_and_timeout_bounds_are_rejected(tmp_path: Path) -> None:
    """The concurrency, queue-depth and timeout knobs all reject non-positive values."""
    triggers = _write_triggers(tmp_path)
    profile = BuildProfile(
        name="x",
        dispatch=DispatchConfig(
            triggers=triggers,
            max_concurrent_runs=0,
            max_queue_depth=0,
            timeout_sec=0,
        ),
    )
    assert _errors(profile, tmp_path) == [
        "dispatch.max_concurrent_runs must be >= 1 (got 0)",
        "dispatch.max_queue_depth must be >= 1 (got 0)",
        "dispatch.timeout_sec must be > 0 (got 0)",
    ]


# --- bluesky / virtual_accelerator ports ----------------------------------


def test_bluesky_port_out_of_range_is_rejected(tmp_path: Path) -> None:
    """The bridge port must be a usable TCP port."""
    profile = BuildProfile(name="x", bluesky=BlueskyConfig(port=0))
    assert _errors(profile, tmp_path) == ["bluesky.port must be in 1..65535 (got 0)"]


def test_bluesky_tiled_port_out_of_range_is_rejected(tmp_path: Path) -> None:
    """The tiled port is only checked when tiled is enabled."""
    profile = BuildProfile(name="x", bluesky=BlueskyConfig(tiled_enabled=True, tiled_port=70000))
    assert _errors(profile, tmp_path) == ["bluesky.tiled_port must be in 1..65535 (got 70000)"]


def test_bluesky_tiled_port_colliding_with_bridge_port_is_rejected(tmp_path: Path) -> None:
    """Tiled and the bridge cannot bind the same port in one container."""
    profile = BuildProfile(
        name="x", bluesky=BlueskyConfig(port=10080, tiled_enabled=True, tiled_port=10080)
    )
    assert _errors(profile, tmp_path) == [
        "bluesky.tiled_port must differ from bluesky.port (both 10080)"
    ]


def test_out_of_range_tiled_port_is_ignored_when_tiled_disabled(tmp_path: Path) -> None:
    """With tiled disabled the tiled_port value is inert."""
    BuildProfile(name="x", bluesky=BlueskyConfig(tiled_port=70000)).validate(tmp_path)


def test_virtual_accelerator_port_out_of_range_is_rejected(tmp_path: Path) -> None:
    """The soft-IOC Channel Access port must be a usable TCP port."""
    profile = BuildProfile(name="x", virtual_accelerator=VAConfig(port=0))
    assert _errors(profile, tmp_path) == ["virtual_accelerator.port must be in 1..65535 (got 0)"]


# --- graph mode's store prerequisite ---------------------------------------


def test_graph_mode_on_a_store_deploying_app_template_validates(tmp_path: Path) -> None:
    """The app templates that render a ``services.graphdb`` block need nothing else."""
    for bundle in ("control_assistant", "ariel_standalone"):
        BuildProfile(name="x", data_bundle=bundle, channel_finder_mode="graph").validate(tmp_path)


def test_graph_mode_on_a_storeless_app_template_is_rejected(tmp_path: Path) -> None:
    """``channel_finder_standalone`` renders no store, so graph has nothing to read.

    The refusal names the missing block rather than the paradigm alone: the fix
    is to configure a graph store, not to abandon the mode.
    """
    profile = BuildProfile(
        name="x", data_bundle="channel_finder_standalone", channel_finder_mode="graph"
    )
    (error,) = _errors(profile, tmp_path)
    assert "channel_finder_mode: graph" in error
    assert "services.graphdb" in error
    assert "channel_finder_standalone" in error


def test_graph_mode_with_an_external_store_uri_validates(tmp_path: Path) -> None:
    """Naming a store the facility runs is the other half of the rule.

    ``services.graphdb.uri`` on the ``config:`` overlay creates the block the
    app template omits, so a storeless template plus an external store passes —
    no local Neo4j is deployed and none is required.
    """
    profile = BuildProfile(
        name="x",
        data_bundle="channel_finder_standalone",
        channel_finder_mode="graph",
        config={
            "services.graphdb.uri": "bolt://graph.facility.org:7687",
            "services.graphdb.username": "neo4j",
        },
    )
    profile.validate(tmp_path)


def test_graph_mode_on_an_attached_project_follows_its_hosts_template(tmp_path: Path) -> None:
    """``deploy_services: false`` renders ``services: {}`` — and is then told the
    store's address by the build, from the hosting deployment's render.

    The attached profile IS the hosting profile plus a delta, so whether a
    store will be there to project is the hosting template's question, answered
    the same way: ``control_assistant`` deploys one, so no refusal; a storeless
    template refuses exactly as it does for a deploying profile. An attached
    profile built with no host in its repo is caught after the render instead
    (``osprey.deployment.reach.reach_errors``), on the config it actually wrote.
    """
    assert (
        _graph_errors(
            BuildProfile(name="x", channel_finder_mode="graph", deploy_services=False), tmp_path
        )
        == []
    )
    profile = BuildProfile(
        name="x", data_bundle="hello_world", channel_finder_mode="graph", deploy_services=False
    )
    (error,) = _graph_errors(profile, tmp_path)
    assert "services.graphdb" in error
    assert "deploy_services: false" in error


def test_graph_mode_on_an_attached_project_with_an_external_store_validates(
    tmp_path: Path,
) -> None:
    """An attached project reaches a shared stack's store by naming its uri."""
    profile = BuildProfile(
        name="x",
        channel_finder_mode="graph",
        deploy_services=False,
        config={"services.graphdb.uri": "bolt://127.0.0.1:7687"},
    )
    assert _graph_errors(profile, tmp_path) == []


def test_graph_mode_with_the_template_block_overridden_away_is_rejected(tmp_path: Path) -> None:
    """A bare ``services.graphdb:`` override deletes the block the template rendered."""
    profile = BuildProfile(name="x", channel_finder_mode="graph", config={"services.graphdb": None})
    (error,) = _errors(profile, tmp_path)
    assert "services.graphdb" in error


def test_graph_mode_with_a_profile_declared_graph_service_validates(tmp_path: Path) -> None:
    """A profile that declares the service itself carries the block into the render."""
    profile = BuildProfile(
        name="x",
        data_bundle="channel_finder_standalone",
        channel_finder_mode="graph",
        services={"graphdb": ServiceDef(template="osprey.graphdb")},
    )
    profile.validate(tmp_path)


def test_a_malformed_graph_store_override_is_not_reported_as_a_missing_block(
    tmp_path: Path,
) -> None:
    """A store named with a bad port is still a store this profile means to dial.

    The resolver raises about the port where it can be acted on — the deploy
    preflight — so this validator must not turn that into "no block at all".
    """
    profile = BuildProfile(
        name="x",
        data_bundle="channel_finder_standalone",
        channel_finder_mode="graph",
        config={"services.graphdb.port_host": "not-a-port"},
    )
    profile.validate(tmp_path)


def test_non_graph_modes_need_no_graph_store(tmp_path: Path) -> None:
    """The prerequisite belongs to graph alone; the file-database paradigms pass."""
    for mode in ("in_context", "hierarchical", "middle_layer", None):
        BuildProfile(
            name="x", data_bundle="channel_finder_standalone", channel_finder_mode=mode
        ).validate(tmp_path)


def test_graph_mode_skips_the_store_rule_when_the_channel_finder_is_off(
    tmp_path: Path,
) -> None:
    """A persona with no channel finder inherits the mode but not the rule.

    The logbook persona of a graph deployment switches the channel-finder server
    off and still inherits ``channel_finder_mode: graph`` from the profile it
    narrows — the mode belongs to the deployment. With no channel finder in that
    render there is no toolless agent to prevent, so the store is not required.
    Both spellings of the switch are honoured: the dotted one the renderer
    applies, and the nested one a hand-written profile can reach for.
    """
    dotted = {"claude_code.servers.channel-finder.enabled": False}
    nested = {"claude_code": {"servers": {"channel-finder": {"enabled": False}}}}
    mixed = {"claude_code.servers": {"channel-finder": {"enabled": False}}}
    for overlay in (dotted, nested, mixed):
        profile = BuildProfile(
            name="x",
            channel_finder_mode="graph",
            deploy_services=False,
            config=overlay,
        )
        assert _graph_errors(profile, tmp_path) == []


def test_graph_mode_still_needs_a_store_when_the_channel_finder_is_on(
    tmp_path: Path,
) -> None:
    """The carve-out is an explicit ``false``, not any mention of the key.

    A persona that leaves the channel finder on — or spells the switch ``true``
    — is exactly the render the rule protects, so on a template that deploys
    no store (``hello_world``; ``control_assistant`` would answer the store
    question itself) it is still refused.
    """
    for overlay in (
        {},
        {"claude_code.servers.channel-finder.enabled": True},
        {"claude_code": {"servers": {"channel-finder": {"enabled": True}}}},
        {"claude_code.servers.controls.enabled": False},
    ):
        profile = BuildProfile(
            name="x",
            data_bundle="hello_world",
            channel_finder_mode="graph",
            deploy_services=False,
            config=overlay,
        )
        (error,) = _graph_errors(profile, tmp_path)
        assert "services.graphdb" in error


# ---------------------------------------------------------------------------
# Hybrid logbook search needs the qmd sidecar it dials
# ---------------------------------------------------------------------------


def _qmd_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """The validation errors about the qmd sidecar alone.

    An attached profile can fail the graph-store rule at the same time, and
    these tests are about the sidecar rule, not the count of rules.
    """
    return [e for e in _errors(profile, profile_dir) if "services.qmd" in e]


def test_hybrid_search_on_a_sidecar_deploying_app_template_validates(tmp_path: Path) -> None:
    """The app templates that render a ``services.qmd`` block need nothing else."""
    for bundle in ("control_assistant", "ariel_standalone"):
        BuildProfile(name="x", data_bundle=bundle).validate(tmp_path)


def test_hybrid_search_on_an_attached_project_follows_its_hosts_template(tmp_path: Path) -> None:
    """``deploy_services: false`` renders ``services: {}`` — and is then told the
    sidecar's port by the build, from the hosting deployment's render.

    The template still switches ``ariel.search_modules.hybrid`` on, and the
    hosting template deploys the sidecar the module dials, so nothing is
    refused here; the build copies ``services.qmd.port`` into the attached
    render (``osprey.deployment.reach``). What IS refused is the same shape a
    deploying profile is refused for — a template that deploys no sidecar while
    the module stays on — and, after the render, an attached profile built with
    no host to be told by (``reach_errors``, on the config it actually wrote).
    """
    BuildProfile(name="x", deploy_services=False).validate(tmp_path)
    profile = BuildProfile(name="x", deploy_services=False, config={"services.qmd": None})
    (error,) = _qmd_errors(profile, tmp_path)
    assert "ariel.search_modules.hybrid" in error
    assert "services.qmd.port" in error
    assert "deploy_services: false" in error


def test_hybrid_search_on_an_attached_project_with_the_sidecar_port_validates(
    tmp_path: Path,
) -> None:
    """An attached project with no host names a shared stack's sidecar itself."""
    profile = BuildProfile(name="x", deploy_services=False, config={"services.qmd.port": 8180})
    profile.validate(tmp_path)


def test_hybrid_search_switched_off_needs_no_sidecar(tmp_path: Path) -> None:
    """A profile that turns the module off has nothing to dial.

    Only an explicit ``false`` counts, the same way the channel-finder switch
    reads for the graph rule: the key is absent from every profile that keeps
    the template's default, so absence reads as on. Both spellings of the
    switch are honoured.
    """
    dotted = {"ariel.search_modules.hybrid.enabled": False}
    nested = {"ariel": {"search_modules": {"hybrid": {"enabled": False}}}}
    for overlay in (dotted, nested):
        BuildProfile(name="x", deploy_services=False, config=overlay).validate(tmp_path)


def test_hybrid_search_with_the_sidecar_block_overridden_away_is_rejected(
    tmp_path: Path,
) -> None:
    """A bare ``services.qmd:`` override deletes the block the template rendered.

    The template's own comment tells an operator to drop the block, the
    ``deployed_services`` entry AND the two ariel modules together; dropping
    the block alone leaves hybrid search enabled with nothing behind it.
    """
    profile = BuildProfile(name="x", config={"services.qmd": None})
    (error,) = _qmd_errors(profile, tmp_path)
    assert "ariel.search_modules.hybrid" in error


def test_hybrid_search_with_the_sidecar_block_and_the_module_dropped_together_validates(
    tmp_path: Path,
) -> None:
    """Dropping the block together with the module is the supported no-sidecar path."""
    BuildProfile(
        name="x",
        config={"services.qmd": None, "ariel.search_modules.hybrid.enabled": False},
    ).validate(tmp_path)


def test_a_malformed_sidecar_port_override_is_not_reported_as_a_missing_block(
    tmp_path: Path,
) -> None:
    """A sidecar named with a bad port is still one this profile means to dial.

    The resolver raises about the port where it can be acted on — the deploy
    preflight — so this validator must not turn that into "no block at all".
    """
    BuildProfile(
        name="x", deploy_services=False, config={"services.qmd.port": "not-a-port"}
    ).validate(tmp_path)


def test_an_app_template_without_ariel_needs_no_sidecar(tmp_path: Path) -> None:
    """The prerequisite belongs to the hybrid module; a template with no ARIEL passes."""
    BuildProfile(name="x", data_bundle="channel_finder_standalone").validate(tmp_path)
    BuildProfile(name="x", data_bundle="channel_finder_standalone", deploy_services=False).validate(
        tmp_path
    )


# ---------------------------------------------------------------------------
# The build's render-side limits-block refusal
# ---------------------------------------------------------------------------
#
# The profile-side lint reads what a `config:` block spelled; this one reads
# the config a deployment actually runs, so a half-written per-type block that
# no profile spelling explains — one an injector assembled, one an app template
# shipped — is still refused before it reaches a machine.


def _write_render(tmp_path: Path, control_system: object) -> Path:
    """A render directory holding nothing but the ``control_system:`` section.

    The build reads its own ``config.yml`` back after the injectors have run,
    which is the only file this check touches.
    """
    import yaml

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    (render_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": "demo", "control_system": control_system}),
        encoding="utf-8",
    )
    return render_dir


def _render_limits_errors(render_dir: Path) -> list[str]:
    """The build's own render-side limits check, as ``_render_project`` runs it."""
    from osprey.cli.build_cmd import _incomplete_limits_errors

    return _incomplete_limits_errors(render_dir)


def test_render_with_a_half_written_limits_block_is_unrunnable(tmp_path: Path) -> None:
    """A per-type block stating only ``enabled`` answers no posture at all.

    It overrides the deployment-wide pair whole, so the deployment silently
    falls back to refusing unlisted channels — the refusal names the leaf an
    operator has to add rather than letting a build ship that surprise.
    """
    render_dir = _write_render(
        tmp_path, {"connector": {"virtual_accelerator": {"limits_checking": {"enabled": True}}}}
    )

    (error,) = _render_limits_errors(render_dir)

    assert (
        "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
        in error
    )


def test_render_with_a_complete_limits_block_is_runnable(tmp_path: Path) -> None:
    """Both leaves stated is the supported way to relax a simulator alone."""
    render_dir = _write_render(
        tmp_path,
        {
            "connector": {
                "virtual_accelerator": {
                    "limits_checking": {"enabled": True, "allow_unlisted_channels": True}
                }
            }
        },
    )

    assert _render_limits_errors(render_dir) == []


def test_render_without_a_per_type_limits_block_is_runnable(tmp_path: Path) -> None:
    """Silence is every deployment predating per-type blocks; only a half-written one is refused."""
    render_dir = _write_render(
        tmp_path,
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"prefix": "SR:"}},
            "limits_checking": {"enabled": True},
        },
    )

    assert _render_limits_errors(render_dir) == []


# ---------------------------------------------------------------------------
# The build's profile-side limits-block refusal
# ---------------------------------------------------------------------------
#
# `osprey validate` is the verb an operator runs by hand; `osprey build` is the
# one that has to refuse, because a deployment reaches a machine through the
# build and not through the verb somebody remembered to run. Both ask
# `limits_block_errors` about the same `config:` block, so a profile refused by
# one is refused by the other with the same words.
#
# Profile-side, not render-side: raising here means a build stops before the
# render, so the one-leaf case is reported once — by the check that can name
# the profile key an author has to fix — instead of twice.


def _build_with_config(tmp_path: Path, config: dict[str, Any], name: str) -> Result:
    """Run ``osprey build`` against a repo whose profile states *config*.

    The profile names no provider, so a build that gets past the profile-side
    lint stops at the next refusal instead of rendering — which is what makes
    the passing case below cheap and what its assertion reads.
    """
    import yaml

    from osprey.cli.build_cmd import build

    repo = tmp_path / name
    repo.mkdir()
    (repo / "profile.yml").write_text(
        yaml.safe_dump(
            {"name": "Demo Facility", "data_bundle": "hello_world", "config": config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_build_refuses_a_profile_writing_only_one_limits_leaf(tmp_path: Path) -> None:
    """One leaf states a posture nothing answers: the per-type block overrides
    the deployment-wide pair whole, so the missing leaf falls back to refusing
    unlisted channels rather than to what the profile meant."""
    result = _build_with_config(
        tmp_path,
        {"control_system.connector.virtual_accelerator.limits_checking.enabled": True},
        "half-a-block",
    )

    assert result.exit_code != 0, result.output
    assert "Profile validation failed" in result.output
    assert "allow_unlisted_channels" in result.output
    assert "virtual_accelerator" in result.output


def test_build_refuses_a_flat_dotted_custom_type(tmp_path: Path) -> None:
    """Flattened, the dots in ``mypkg.TangoConnector`` are indistinguishable
    from path separators, so the emitter renders a key no connector reads. The
    build names the offending key instead of rendering the dead spelling."""
    key = "control_system.connector.mypkg.TangoConnector.limits_checking.enabled"

    result = _build_with_config(tmp_path, {key: True}, "dotted-type")

    assert result.exit_code != 0, result.output
    assert "Profile validation failed" in result.output
    assert key in result.output


def test_build_passes_a_complete_per_type_limits_block(tmp_path: Path) -> None:
    """The supported spelling reaches the checks past the lint untouched.

    Pinned beside the two refusals because a lint that refuses every profile
    passes both of those on its own. The provider refusal is the marker: it is
    the next thing the build asks after the profile-side lint.
    """
    result = _build_with_config(
        tmp_path,
        {
            "control_system.connector.virtual_accelerator.limits_checking.enabled": True,
            "control_system.connector.virtual_accelerator.limits_checking."
            "allow_unlisted_channels": True,
        },
        "whole-block",
    )

    assert "Profile validation failed" not in result.output
    assert "names no provider" in result.output

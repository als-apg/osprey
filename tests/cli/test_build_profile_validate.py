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

import pytest

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
        name="x", bluesky=BlueskyConfig(port=8090, tiled_enabled=True, tiled_port=8090)
    )
    assert _errors(profile, tmp_path) == [
        "bluesky.tiled_port must differ from bluesky.port (both 8090)"
    ]


def test_out_of_range_tiled_port_is_ignored_when_tiled_disabled(tmp_path: Path) -> None:
    """With tiled disabled the tiled_port value is inert."""
    BuildProfile(name="x", bluesky=BlueskyConfig(tiled_port=70000)).validate(tmp_path)


def test_virtual_accelerator_port_out_of_range_is_rejected(tmp_path: Path) -> None:
    """The soft-IOC Channel Access port must be a usable TCP port."""
    profile = BuildProfile(name="x", virtual_accelerator=VAConfig(port=0))
    assert _errors(profile, tmp_path) == ["virtual_accelerator.port must be in 1..65535 (got 0)"]

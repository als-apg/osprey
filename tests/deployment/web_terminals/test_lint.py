"""Tests for modules.web_terminals validation (osprey.deployment.web_terminals.lint)."""

from __future__ import annotations

import copy

from osprey.deployment.web_terminals.lint import Finding, lint_web_terminals

_CLEAN_CONFIG = {
    "facility": {"prefix": "test"},
    "ports": {
        "matlab": 8001,
        "web_terminal_nginx": 9080,
        "event_dispatcher": 8010,
    },
    "modules": {
        "web_terminals": {
            "enabled": True,
            "nginx_port": 9080,
            "web_base_port": 9091,
            "artifact_base_port": 9291,
            "ariel_base_port": 9391,
            "lattice_base_port": 9491,
            "users": ["thellert", "gmartino"],
        },
        "event_dispatcher": {
            "enabled": True,
            "port": 8010,
            "sidecar_count": 5,
            "sidecar_port_base": 9190,
        },
    },
}


def _errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def _warnings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "warn"]


def _infos(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "info"]


def test_lint_clean_config_reports_no_error_findings() -> None:
    """A well-formed, non-colliding config must produce zero error findings."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []


def test_lint_duplicate_user_is_an_error() -> None:
    """Repeating a username breaks the one-service-per-user invariant."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["thellert", "thellert"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.duplicate_user" for f in errors)


def test_lint_port_family_overlap_with_event_dispatcher_sidecars_is_an_error() -> None:
    """artifact_base_port's per-user range must not collide with sidecar ports."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    # artifact_base_port range is [9291, 9292]; force it into the sidecar range
    # [9190, 9194] by lowering artifact_base_port so index 1 lands on 9191.
    config["modules"]["web_terminals"]["artifact_base_port"] = 9190

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    overlap_findings = [f for f in errors if f.code == "web_terminals.port_overlap"]
    assert overlap_findings
    assert any("artifact_base_port" in f.message for f in overlap_findings)
    assert any("sidecar_port_base" in f.message for f in overlap_findings)


def test_lint_reserved_name_nginx_is_an_error() -> None:
    """A user named 'nginx' collides with the always-present reverse-proxy service."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["nginx", "gmartino"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.reserved_name" for f in errors)


def test_lint_enabled_with_empty_users_is_a_single_warning_not_an_error() -> None:
    """Zero users is valid (renders nginx + an empty landing group) but worth flagging."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = []

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []
    warnings = _warnings(findings)
    assert len(warnings) == 1
    assert warnings[0].code == "web_terminals.empty_users"


def test_lint_empty_users_with_benchmarks_enabled_is_an_error() -> None:
    """benchmarks.runs_in_container needs a first user; zero users can't resolve one."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = []
    config["modules"]["benchmarks"] = {"enabled": True}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.empty_users_with_benchmarks" for f in errors)
    # The plain empty-users warning must not also fire alongside the hard failure.
    assert not any(f.code == "web_terminals.empty_users" for f in findings)


def test_lint_disabled_module_reports_nothing() -> None:
    """When web_terminals is off, none of the rules above should even run."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["enabled"] = False
    config["modules"]["web_terminals"]["users"] = ["nginx", "nginx"]  # would else double-fault

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert findings == []


def test_lint_missing_web_base_port_is_an_error() -> None:
    """A user list that can't fully resolve allocate_ports() is a consistency
    error. `web` is the only family without a registry default, so it is the
    one whose absence still fails."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    del config["modules"]["web_terminals"]["web_base_port"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.incomplete_port_families" for f in errors)


def test_lint_missing_companion_base_port_is_not_an_error() -> None:
    """A companion family's base port falls back to its registry default — a
    config written before that panel existed must keep linting clean."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    del config["modules"]["web_terminals"]["lattice_base_port"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.incomplete_port_families" for f in _errors(findings))


def test_lint_username_bad_charset_is_an_error() -> None:
    """Usernames become nginx `location` keys and URL path segments — must be
    ``^[a-z0-9][a-z0-9_-]*$``."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["Bad_User", "gmartino"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_username_charset" for f in errors)


def test_lint_username_charset_rejects_leading_dash_underscore_space_and_uppercase() -> None:
    """Each of these must be rejected: leading `-`, leading `_`, an embedded space,
    and an uppercase-only name."""
    # Arrange / Act / Assert
    for bad_name in ("-x", "_x", "a b", "A"):
        config = copy.deepcopy(_CLEAN_CONFIG)
        config["modules"]["web_terminals"]["users"] = [bad_name]

        findings = lint_web_terminals(config)

        errors = _errors(findings)
        assert any(f.code == "web_terminals.invalid_username_charset" for f in errors), (
            f"expected {bad_name!r} to be rejected"
        )


def test_lint_username_charset_accepts_leading_digit() -> None:
    """A leading digit is fine — only the character class matters, not digit-first."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["1abc"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.invalid_username_charset" for f in _errors(findings))


def test_lint_tls_enabled_adds_443_to_port_overlap_set() -> None:
    """When the TLS seam is enabled, port 443 (the `listen 443 ssl` port) joins the
    S1-S4 collision set and collides with an existing ports.* literal of 443."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["ports"]["conflicting"] = 443
    config["modules"]["web_terminals"]["tls"] = {"enabled": True}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    overlap_findings = [f for f in errors if f.code == "web_terminals.port_overlap"]
    assert any("443" in f.message for f in overlap_findings)


def test_lint_tls_disabled_does_not_add_443_to_port_overlap_set() -> None:
    """With the TLS seam left at its default (off), port 443 is just an ordinary
    ports.* value and must not be treated as a second, colliding source."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["ports"]["conflicting"] = 443
    # tls.enabled defaults to False; no web_terminals.tls stanza at all here.

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    overlap_findings = [f for f in errors if f.code == "web_terminals.port_overlap"]
    assert not any("443" in f.message for f in overlap_findings)


# --- Task 1.3: index-related findings for object-form users -----------------


def test_lint_duplicate_explicit_index_is_an_error() -> None:
    """Two object-form users sharing an index would collide on every port family."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "gmartino", "index": 0},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.duplicate_index" for f in errors)


def test_lint_distinct_explicit_indices_are_not_a_duplicate_index_error() -> None:
    """Distinct explicit indices must not falsely trigger the duplicate-index check."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "gmartino", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.duplicate_index" for f in errors)


def test_lint_missing_index_on_object_form_user_is_an_error() -> None:
    """An object-form entry with no `index` key at all is invalid."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert"}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_index" for f in errors)


def test_lint_non_integer_index_is_an_error() -> None:
    """A string index (e.g. from a hand-edited YAML) is not a valid port offset."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": "0"}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_index" for f in errors)


def test_lint_boolean_index_is_an_error() -> None:
    """`bool` is an `int` subclass in Python, but `index: true`/`false` is invalid."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": True}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_index" for f in errors)


def test_lint_negative_index_is_an_error() -> None:
    """A negative index can't resolve to a real port offset."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": -1}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_index" for f in errors)


def test_lint_valid_object_form_users_report_no_index_errors() -> None:
    """A well-formed, explicit-index roster must not trip either index error check."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "gmartino", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.invalid_index" for f in errors)
    assert not any(f.code == "web_terminals.duplicate_index" for f in errors)


def test_lint_bare_multi_user_list_warns_about_port_drift_risk() -> None:
    """A legacy bare list with >1 user risks positional port drift on decommission."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["thellert", "gmartino"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    warnings = _warnings(findings)
    assert any(f.code == "web_terminals.bare_list_port_drift_risk" for f in warnings)


def test_lint_bare_single_user_list_does_not_warn_about_port_drift_risk() -> None:
    """A single-user bare list has no positional drift risk to warn about."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["thellert"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    warnings = _warnings(findings)
    assert not any(f.code == "web_terminals.bare_list_port_drift_risk" for f in warnings)


def test_lint_explicit_index_roster_does_not_warn_about_port_drift_risk() -> None:
    """A roster already using explicit indices is exempt from the drift warning."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "gmartino", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    warnings = _warnings(findings)
    assert not any(f.code == "web_terminals.bare_list_port_drift_risk" for f in warnings)


def test_lint_mixed_roster_does_not_crash_and_does_not_warn_about_port_drift_risk() -> None:
    """A mixed bare/object-form roster is odd but must not crash the linter, and
    is exempt from the bare-list drift warning (it isn't a pure legacy list)."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["thellert", {"name": "gmartino", "index": 1}]

    # Act
    findings = lint_web_terminals(config)

    # Assert (no crash by construction; assert the drift warn specifically)
    warnings = _warnings(findings)
    assert not any(f.code == "web_terminals.bare_list_port_drift_risk" for f in warnings)


def test_lint_object_form_reserved_name_is_an_error() -> None:
    """Object-form entries must be held to the same reserved-name rule as bare
    strings — the schema change must not open a validation gap."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "nginx", "index": 0},
        {"name": "gmartino", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.reserved_name" for f in errors)


def test_lint_object_form_bad_charset_is_an_error() -> None:
    """Object-form entries must be held to the same charset rule as bare
    strings — usernames still become nginx location keys either way."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [{"name": "Bad_User", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_username_charset" for f in errors)


def test_lint_object_form_valid_name_reports_no_reserved_or_charset_errors() -> None:
    """A well-formed object-form name must not trip either name-validation check."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "gmartino", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.reserved_name" for f in errors)
    assert not any(f.code == "web_terminals.invalid_username_charset" for f in errors)


def test_lint_object_form_duplicate_name_is_still_a_duplicate_user_error() -> None:
    """Object-form users must still be caught by the pre-existing duplicate-name
    check (a dict is unhashable, so this exercises the name-based comparison)."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "thellert", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.duplicate_user" for f in errors)


# --- Task 2.3: persona catalog identity/reference checks ---------------------


def test_lint_clean_persona_catalog_reports_no_error_findings() -> None:
    """A well-formed catalog with a valid default_persona and a matching
    explicit reference must not trip any of the new persona checks."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["registry"] = {"url": "registry.example.org:5050"}
    config["modules"]["web_terminals"]["personas"] = {
        "assistant": {"project": "als-assistant"},
        "analysis": {"project": "als-analysis", "build_profile": "profiles/analysis.yml"},
    }
    config["modules"]["web_terminals"]["default_persona"] = "assistant"
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0},
        {"name": "gmartino", "index": 1, "persona": "analysis"},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []


def test_lint_unknown_explicit_persona_reference_is_an_error() -> None:
    """A roster entry's own `persona:` key must name a catalog entry."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"assistant": {"project": "als-assistant"}}
    config["modules"]["web_terminals"]["default_persona"] = "assistant"
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "persona": "ghost"}
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    unknown_ref = [f for f in errors if f.code == "web_terminals.unknown_persona_reference"]
    assert unknown_ref
    assert any("thellert" in f.message and "ghost" in f.message for f in unknown_ref)


def test_lint_unknown_inherited_default_persona_reference_is_an_error() -> None:
    """A user with no `persona:` of its own inherits `default_persona`; if that
    name isn't in the catalog, the inherited reference is unresolvable too."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"assistant": {"project": "als-assistant"}}
    config["modules"]["web_terminals"]["default_persona"] = "ghost"
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.unknown_persona_reference" for f in errors)


def test_lint_default_persona_not_in_catalog_is_an_error() -> None:
    """`default_persona` must itself name a catalog entry."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"assistant": {"project": "als-assistant"}}
    config["modules"]["web_terminals"]["default_persona"] = "ghost"
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.unknown_default_persona" for f in errors)
    assert any(
        "ghost" in f.message for f in errors if f.code == "web_terminals.unknown_default_persona"
    )


def test_lint_default_persona_in_catalog_reports_no_error() -> None:
    """A `default_persona` that does name a catalog entry must not be flagged."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"assistant": {"project": "als-assistant"}}
    config["modules"]["web_terminals"]["default_persona"] = "assistant"
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.unknown_default_persona" for f in errors)


def test_lint_persona_catalog_bad_charset_is_an_error() -> None:
    """A persona catalog key becomes an image-tag suffix and a path component;
    it's held to the same charset as usernames."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"Bad Persona": {"project": "als-x"}}
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_persona_charset" for f in errors)


def test_lint_persona_catalog_reserved_name_is_an_error() -> None:
    """A persona named 'nginx' collides with the same reserved-name closed set
    held over usernames."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"nginx": {"project": "als-x"}}
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_reserved_name" for f in errors)


def test_lint_no_personas_catalog_reports_no_persona_findings() -> None:
    """A config predating persona catalogs (no `personas:` block, no `persona:`
    keys, no `default_persona`) must resolve every entry as zero-migration and
    trip none of the new persona checks."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    persona_codes = {
        "web_terminals.invalid_persona_charset",
        "web_terminals.persona_reserved_name",
        "web_terminals.unknown_default_persona",
        "web_terminals.unknown_persona_reference",
    }
    assert not any(f.code in persona_codes for f in findings)


# --- Task 2.4: mode-coherence checks -----------------------------------------


def _persona_config(**overrides: object) -> dict:
    """A minimal config with a persona catalog in effect, for the mode-coherence
    tests below. Callers override/add keys under `modules.web_terminals` and/or
    the top-level `registry` section via the two supported override kwargs."""
    config = copy.deepcopy(_CLEAN_CONFIG)
    web_terminals_overrides = overrides.pop("web_terminals", {})
    registry_overrides = overrides.pop("registry", None)
    assert not overrides, f"unsupported override keys: {sorted(overrides)}"
    config["modules"]["web_terminals"]["personas"] = {
        "assistant": {"project": "als-assistant"},
    }
    config["modules"]["web_terminals"]["default_persona"] = "assistant"
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]
    config["modules"]["web_terminals"].update(web_terminals_overrides)
    if registry_overrides is not None:
        config["registry"] = registry_overrides
    return config


def test_lint_unknown_image_source_is_an_error() -> None:
    """An `image_source` value that is neither `registry` nor `local` is an error."""
    # Arrange
    config = _persona_config(
        web_terminals={"image_source": "s3"}, registry={"url": "registry.example.org:5050"}
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.unknown_image_source" for f in errors)
    assert any("s3" in f.message for f in errors if f.code == "web_terminals.unknown_image_source")


def test_lint_registry_mode_without_registry_url_is_an_error() -> None:
    """`image_source: registry` (the default) needs registry.url to pull images."""
    # Arrange
    config = _persona_config()  # image_source unset -> registry; no registry.url

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.registry_mode_missing_url" for f in errors)


def test_lint_registry_mode_with_registry_url_reports_no_missing_url_error() -> None:
    """A registry.url that is actually set clears the coherence error."""
    # Arrange
    config = _persona_config(registry={"url": "registry.example.org:5050"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.registry_mode_missing_url" for f in errors)


def test_lint_no_persona_catalog_does_not_require_registry_url() -> None:
    """Zero-migration path: a config with no personas catalog at all never
    triggers the registry.url coherence check, even with zero registry.url."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.registry_mode_missing_url" for f in errors)


def test_lint_local_mode_with_registry_url_is_a_warning() -> None:
    """`image_source: local` never reads registry.url; setting it anyway warns."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {"assistant": {"project": "als-assistant", "project_path": "/nonexistent"}},
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    warnings = _warnings(findings)
    assert any(f.code == "web_terminals.local_mode_unused_registry_url" for f in warnings)


def test_lint_local_mode_without_catalog_is_an_error() -> None:
    """`image_source: local` requires both a catalog and a default_persona —
    the lint-side mirror of resolve_personas()'s strict ValueError guard."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["image_source"] = "local"

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.local_mode_requires_catalog" for f in errors)


def test_lint_local_mode_without_default_persona_is_an_error() -> None:
    """A catalog alone isn't enough for local mode; default_persona is also required."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["image_source"] = "local"
    config["modules"]["web_terminals"]["personas"] = {
        "assistant": {"project": "als-assistant", "project_path": "/nonexistent"}
    }

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.local_mode_requires_catalog" for f in errors)


def test_lint_local_mode_missing_project_path_is_an_error() -> None:
    """A referenced persona with no `project_path` set can't be built locally."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {"assistant": {"project": "als-assistant"}},
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_project_path" for f in errors)


def test_lint_local_mode_project_path_not_a_directory_is_an_error(tmp_path) -> None:
    """A `project_path` that doesn't exist (or isn't a directory) can't be built
    when the entry has no build_profile to auto-render it from."""
    # Arrange (basename matches `project` so the name invariant passes and we
    # exercise the existence check itself, not the name-mismatch check)
    missing_path = tmp_path / "als-assistant"
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(missing_path)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_project_path_not_dir" for f in errors)


def test_lint_local_mode_missing_dockerfile_is_an_error(tmp_path) -> None:
    """A project_path directory with no Dockerfile can't be built."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_dockerfile" for f in errors)


def test_lint_local_mode_missing_config_yml_is_an_error(tmp_path) -> None:
    """A project_path directory with no config.yml can't be existence-checked
    for its own project_name, and can't confirm the persona's identity."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_config_yml" for f in errors)


def test_lint_local_mode_well_formed_project_path_reports_no_error(tmp_path) -> None:
    """A project_path with a Dockerfile, a config.yml, and a matching
    project_name must not trip any of the local-mode existence checks."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []


def test_lint_local_mode_project_name_mismatch_is_an_error(tmp_path) -> None:
    """The catalog's `project` must equal the project_path's own config.yml
    `project_name` — a mismatch would silently mount/path against the wrong
    directory at runtime."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    (project_dir / "config.yml").write_text("project_name: something-else\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    mismatch = [f for f in errors if f.code == "web_terminals.persona_project_mismatch"]
    assert mismatch
    assert any("als-assistant" in f.message and "something-else" in f.message for f in mismatch)


def test_lint_local_mode_unreferenced_persona_project_path_is_not_checked(tmp_path) -> None:
    """A catalog entry nobody references (no user's `persona:`, not
    `default_persona`) is outside the local-mode existence checks — an unused
    draft entry never blocks a deploy since only referenced personas are ever
    built."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)},
                "unused": {"project": "als-unused", "project_path": "/nonexistent"},
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []


# --- Task 3.2: auto-render demotion + project-path name invariant ------------


def test_lint_local_mode_missing_project_path_with_build_profile_is_auto_renderable(
    tmp_path,
) -> None:
    """A referenced persona whose project_path does not exist yet but carries a
    usable build_profile is only an informational finding — deploy up will
    render it before building, so it must not block a deploy."""
    # Arrange (project_path basename matches `project`; directory not created)
    missing_path = tmp_path / "als-assistant"
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(missing_path),
                    "build_profile": "profiles/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []
    infos = _infos(findings)
    assert any(f.code == "web_terminals.persona_project_path_auto_renderable" for f in infos)
    # The hard "not a directory" error must not fire alongside the info note.
    assert not any(f.code == "web_terminals.persona_project_path_not_dir" for f in findings)


def test_lint_local_mode_missing_project_path_without_build_profile_stays_an_error(
    tmp_path,
) -> None:
    """Without a build_profile there is nothing to auto-render from, so a
    non-existent project_path remains the pre-existing hard error."""
    # Arrange
    missing_path = tmp_path / "als-assistant"
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(missing_path)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_project_path_not_dir" for f in errors)
    assert not any(f.code == "web_terminals.persona_project_path_auto_renderable" for f in findings)


def test_lint_local_mode_partial_render_missing_dockerfile_stays_an_error(tmp_path) -> None:
    """A build_profile does NOT rescue a directory that already exists but is
    incomplete — auto-render never overwrites an existing directory, so a
    missing Dockerfile inside it is still a hard error (partial render)."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(project_dir),
                    "build_profile": "profiles/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_dockerfile" for f in errors)
    assert not any(f.code == "web_terminals.persona_project_path_auto_renderable" for f in findings)


def test_lint_local_mode_partial_render_missing_config_yml_stays_an_error(tmp_path) -> None:
    """Same partial-render rule for a missing config.yml inside an existing dir:
    the build_profile does not demote it."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(project_dir),
                    "build_profile": "profiles/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_config_yml" for f in errors)
    assert not any(f.code == "web_terminals.persona_project_path_auto_renderable" for f in findings)


def test_lint_local_mode_project_path_basename_not_matching_project_is_an_error(tmp_path) -> None:
    """The auto-render invariant: project_path's basename must equal the catalog
    `project`, since auto-render derives its output dir from `project`. A
    disagreement is an error even with a build_profile present."""
    # Arrange (basename "wrong-name" != project "als-assistant"; dir absent)
    project_path = tmp_path / "wrong-name"
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(project_path),
                    "build_profile": "profiles/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    mismatch = [f for f in errors if f.code == "web_terminals.persona_project_path_name_mismatch"]
    assert mismatch
    assert any("wrong-name" in f.message and "als-assistant" in f.message for f in mismatch)
    # The name-mismatch supersedes the auto-renderable demotion.
    assert not any(f.code == "web_terminals.persona_project_path_auto_renderable" for f in findings)


def test_lint_local_mode_name_invariant_fires_for_an_otherwise_wellformed_dir(tmp_path) -> None:
    """The name invariant also fails an existing, otherwise-complete directory
    whose basename disagrees with `project`, and supersedes the inner
    Dockerfile/config.yml checks (which never run for it)."""
    # Arrange
    project_dir = tmp_path / "wrong-name"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_project_path_name_mismatch" for f in errors)
    assert not any(f.code == "web_terminals.persona_missing_dockerfile" for f in errors)


def test_lint_local_mode_matching_basename_reports_no_name_invariant_error(tmp_path) -> None:
    """A project_path whose basename equals `project` must never trip the name
    invariant — the well-formed, on-disk case stays clean."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_dir)}
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.persona_project_path_name_mismatch" for f in findings)
    assert _errors(findings) == []


def test_lint_registry_mode_does_not_run_project_path_name_invariant(tmp_path) -> None:
    """The name invariant is a local-mode build concern; registry mode pulls
    images and must not evaluate project_path basenames at all."""
    # Arrange (basename disagrees with project, but image_source is registry)
    project_path = tmp_path / "wrong-name"
    config = _persona_config(
        web_terminals={
            "personas": {
                "assistant": {"project": "als-assistant", "project_path": str(project_path)},
            },
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.persona_project_path_name_mismatch" for f in findings)


def test_lint_registry_mode_non_default_persona_without_build_profile_is_an_error() -> None:
    """In registry mode, a non-default persona has no local project_path to
    build from — `build_profile` is its only route to a CI build job."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "personas": {
                "assistant": {"project": "als-assistant"},
                "analysis": {"project": "als-analysis"},
            },
            "users": [
                {"name": "thellert", "index": 0},
                {"name": "gmartino", "index": 1, "persona": "analysis"},
            ],
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    missing_profile = [f for f in errors if f.code == "web_terminals.persona_missing_build_profile"]
    assert missing_profile
    assert any("analysis" in f.message for f in missing_profile)


def test_lint_registry_mode_default_persona_is_exempt_from_build_profile() -> None:
    """The default persona's image is built by the core CI job, not a
    per-persona one — it never needs `build_profile`."""
    # Arrange
    config = _persona_config(registry={"url": "registry.example.org:5050"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.persona_missing_build_profile" for f in errors)


def test_lint_registry_mode_non_default_persona_with_build_profile_reports_no_error() -> None:
    """Setting build_profile on the non-default persona clears the error."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "personas": {
                "assistant": {"project": "als-assistant"},
                "analysis": {
                    "project": "als-analysis",
                    "build_profile": "profiles/analysis.yml",
                },
            },
            "users": [
                {"name": "thellert", "index": 0},
                {"name": "gmartino", "index": 1, "persona": "analysis"},
            ],
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.persona_missing_build_profile" for f in errors)


def test_lint_unknown_mcp_topology_is_an_error() -> None:
    """`shared_http` (and any other unrecognized value) is fail-closed at lint
    time too — the lint-side mirror of render.py's ValueError."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "shared_http"}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.unknown_mcp_topology" for f in errors)
    assert any("shared_http" in f.message for f in errors)


def test_lint_per_container_stdio_topology_reports_no_error() -> None:
    """The one wired, default topology value must never be flagged."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "per_container_stdio"}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.unknown_mcp_topology" for f in errors)


# --- Task 4.5: empty facility.prefix (web container-name prefix) -------------


def test_lint_users_with_absent_facility_prefix_is_an_error() -> None:
    """Web container names are `<facility.prefix>-nginx`/`<...>-web-<user>`, so a
    configured roster with no facility section at all renders leading-dash names
    like `-nginx`, which Docker rejects only at `deploy up`. Catch it at lint."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config.pop("facility", None)  # no facility section -> empty effective prefix

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.empty_facility_prefix" for f in errors)


def test_lint_users_with_empty_string_facility_prefix_is_an_error() -> None:
    """An explicit empty-string prefix derives the same broken `-nginx` name as an
    absent one (`facility.get("prefix") or ""`), so it is equally an error."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["facility"] = {"prefix": ""}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.empty_facility_prefix" for f in errors)


def test_lint_users_with_nonempty_facility_prefix_reports_no_prefix_error() -> None:
    """A non-empty prefix yields valid `<prefix>-nginx` names, so the check is silent."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["facility"] = {"prefix": "als"}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.empty_facility_prefix" for f in findings)


def test_lint_no_users_with_absent_facility_prefix_reports_no_prefix_error() -> None:
    """With no users configured there are no per-user services to name, so an empty
    prefix is not this check's concern (empty users[] is `_check_empty_users`')."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config.pop("facility", None)
    config["modules"]["web_terminals"]["users"] = []

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.empty_facility_prefix" for f in findings)


def test_lint_omitted_mcp_topology_reports_no_error() -> None:
    """No `mcp:` stanza at all (the common case) must never be flagged."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code == "web_terminals.unknown_mcp_topology" for f in errors)

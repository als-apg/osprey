"""Tests for modules.web_terminals validation (osprey.deployment.web_terminals.lint)."""

from __future__ import annotations

import copy

import pytest

from osprey.deployment.web_terminals import lint
from osprey.deployment.web_terminals.lint import (
    Finding,
    lint_profile_config,
    lint_web_terminals,
    profile_config_errors,
)

_CLEAN_CONFIG = {
    "facility": {"prefix": "test"},
    "services": {
        "openobserve": {"port": 5080},
        "postgresql": {"port_host": 5432},
        "event_dispatcher": {"port": 8020},
        # Reaches the dispatcher over the compose network and publishes nothing
        # on the host, so it is deliberately outside the collision set.
        "dispatch_worker": {"worker_port_base": 9190, "worker_count": 1},
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
    },
}


def _errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def _warnings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "warn"]


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


def test_lint_port_family_overlap_with_a_deployed_service_is_an_error() -> None:
    """The web stack runs on the host netns, so a per-user family collides with
    any port a deployed service publishes."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    # artifact_base_port's range is [9291, 9292]; move openobserve onto 9292,
    # the port user index 1 would bind.
    config["services"]["openobserve"]["port"] = 9292

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    overlap_findings = [f for f in errors if f.code == "web_terminals.port_overlap"]
    assert overlap_findings
    assert any("artifact_base_port" in f.message for f in overlap_findings)
    assert any("services.openobserve.port" in f.message for f in overlap_findings)


def test_lint_nginx_port_colliding_with_a_user_port_is_an_error() -> None:
    """nginx's listener lost its `ports.*` mirror, so it joins the collision set
    on its own — nothing else declares it any more."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["nginx_port"] = 9091  # web_base_port index 0

    # Act
    findings = lint_web_terminals(config)

    # Assert
    overlap_findings = [f for f in _errors(findings) if f.code == "web_terminals.port_overlap"]
    assert any("web_terminals.nginx_port" in f.message for f in overlap_findings)


def test_lint_container_internal_worker_port_is_not_in_the_collision_set() -> None:
    """A port published on no host interface cannot collide with one that is."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    # The dispatch worker's internal listener, set to a port a user also binds.
    config["services"]["dispatch_worker"]["worker_port_base"] = 9091

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.port_overlap" for f in _errors(findings))


def test_lint_username_matching_a_service_name_is_not_an_error() -> None:
    """A user's compose service key is `web-<user>`, so a name like 'nginx' has
    no service key to collide with — rejecting it would be a false failure.

    Carve-out: the two names that DO collide with something a user's container
    actually holds — the audit subdirectories of the auth sidecar and of a
    dispatch worker, which are bound read-write per identity — are refused by
    `_check_reserved_audit_identities` (see the two tests below). This test is
    about compose *service keys*, which a username never becomes.
    """
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["nginx", "gmartino"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []


@pytest.mark.parametrize("name", ["sidecar", "dispatch-worker-1"])
def test_lint_username_matching_a_service_audit_identity_is_an_error(name: str) -> None:
    """Each user's container binds `var/audit/<user>/` read-write, so a user
    named after a service's audit identity would read and rewrite the trail of
    the component that records them. The render refuses it outright; this is the
    same rejection at scaffold time, before a deploy is attempted."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [name, "gmartino"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.reserved_audit_identity" for f in _errors(findings))


def test_lint_username_resembling_a_service_audit_identity_is_not_an_error() -> None:
    """The rule matches the identity exactly (`dispatch-worker-<digits>`), not
    anything that starts like one: `dispatch-worker` and `dispatch-worker-a`
    name no subdirectory any service writes."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["dispatch-worker", "dispatch-worker-a"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.reserved_audit_identity" for f in findings)


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


def test_lint_username_charset_rejects_a_trailing_newline() -> None:
    """The charset check is a `fullmatch`, not a `match`.

    Python's `$` also matches *before* a trailing newline, so a `match` would
    report "alice\\n" as clean — a name that renders into an nginx location key
    mid-directive.
    """
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = ["alice\n"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_username_charset" for f in errors)


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
    collision set and collides with a service already publishing 443."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["services"]["conflicting"] = {"port": 443}
    config["modules"]["web_terminals"]["tls"] = {"enabled": True}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    overlap_findings = [f for f in errors if f.code == "web_terminals.port_overlap"]
    assert any("443" in f.message for f in overlap_findings)


def test_lint_tls_disabled_does_not_add_443_to_port_overlap_set() -> None:
    """With the TLS seam left at its default (off), 443 is just one service's
    port and must not be treated as a second, colliding source."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["services"]["conflicting"] = {"port": 443}
    # tls.enabled defaults to False; no web_terminals.tls stanza at all here.

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    overlap_findings = [f for f in errors if f.code == "web_terminals.port_overlap"]
    assert not any("443" in f.message for f in overlap_findings)


# --- index-related findings for object-form users ---------------------------


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


def test_lint_non_string_display_name_is_an_error() -> None:
    """A non-string `display_name` (a config typo) is rejected — the renderer would
    otherwise drop it silently."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "display_name": ["not", "a", "string"]}
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_display_name" for f in errors)


def test_lint_string_display_name_reports_no_error() -> None:
    """A well-formed string `display_name` is accepted."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "display_name": "Operations"},
        {"name": "gmartino", "index": 1},  # no display_name at all is equally fine
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.invalid_display_name" for f in findings)


def test_lint_non_boolean_user_login_is_an_error() -> None:
    """A non-boolean `login` deploys fail-closed as "login required", which is
    the opposite of what the author who wrote it believes — so the typo is an
    ERROR here rather than a silent lock-out."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "login": "false"}
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.invalid_user_login" for f in _errors(findings))


def test_lint_login_false_without_auth_is_an_inert_key_warning() -> None:
    """`login: false` with `auth.method: none` changes nothing — there is no
    login wall to be exempt from — and the config should not claim otherwise."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0, "login": False}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.user_login_inert" for f in _warnings(findings))


def test_lint_login_false_with_auth_on_reports_nothing() -> None:
    """The intended use — a public entry in an authenticated deployment — is
    clean; and explicit `login: true` is a well-formed (default) spelling."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": "password", "allow_insecure_http": True}
    web_terminals["users"] = [
        {"name": "thellert", "index": 0, "login": True},
        {"name": "ariel", "index": 1, "login": False},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.invalid_user_login" for f in findings)
    assert not any(f.code == "web_terminals.user_login_inert" for f in findings)


def test_lint_non_string_user_theme_is_an_error() -> None:
    """A non-string `theme` (a config typo) is rejected — the renderer would
    otherwise drop it silently."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "theme": {"family": "desy"}}
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_user_theme" for f in errors)


def test_lint_string_user_theme_reports_no_error() -> None:
    """A well-formed string `theme` is accepted — as a family or a concrete id."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "theme": "desy"},
        {"name": "gmartino", "index": 1, "theme": "desy-light"},
        {"name": "aallezy", "index": 2},  # no theme at all is equally fine
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.invalid_user_theme" for f in findings)


def test_lint_does_not_validate_the_theme_name_itself() -> None:
    """Lint checks the TYPE, never whether the name resolves.

    The theme registry ships with the image, not with this config, and the web
    terminal already warns and falls back at startup on an unknown value. Failing
    a build over a name this module cannot authoritatively resolve would be worse
    than that warning.
    """
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "theme": "no-such-theme"}
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.invalid_user_theme" for f in findings)


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


def test_lint_object_form_valid_name_reports_no_charset_error() -> None:
    """A well-formed object-form name must not trip the name-validation check."""
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


# --- persona catalog identity/reference checks -------------------------------


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


def test_lint_persona_charset_rejects_a_trailing_newline() -> None:
    """Same `fullmatch`-not-`match` rule as the username charset check: a persona
    key with a trailing newline is not clean."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"ops\n": {"project": "als-x"}}
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_persona_charset" for f in errors)


def test_lint_persona_named_after_a_service_is_not_an_error() -> None:
    """A persona name becomes an image-tag suffix and a path component, never a
    compose service key, so it has no service name to collide with."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["registry"] = {"url": "registry.example.com:5050/als"}
    config["modules"]["web_terminals"]["personas"] = {"nginx": {"project": "als-x"}}
    config["modules"]["web_terminals"]["default_persona"] = "nginx"
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "persona": "nginx"}
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []


def test_lint_persona_seed_base_non_bool_is_an_error() -> None:
    """A persona `seed_base` that isn't a boolean (e.g. the YAML string
    "false", which is truthy and would silently defeat the opt-out) is an
    error."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {
        "standalone": {"project": "als-x", "seed_base": "false"}
    }
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_invalid_seed_base" for f in errors)


def test_lint_persona_seed_base_bool_is_accepted() -> None:
    """A boolean `seed_base` (either value) trips no seed_base finding."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {
        "keep": {"project": "als-keep", "seed_base": True},
        "drop": {"project": "als-drop", "seed_base": False},
    }
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.persona_invalid_seed_base" for f in findings)


def test_lint_persona_seed_base_absent_is_accepted() -> None:
    """A persona entry with no `seed_base` key at all trips no seed_base finding."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["personas"] = {"plain": {"project": "als-x"}}
    config["modules"]["web_terminals"]["users"] = [{"name": "thellert", "index": 0}]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.persona_invalid_seed_base" for f in findings)


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
        "web_terminals.persona_invalid_seed_base",
        "web_terminals.unknown_default_persona",
        "web_terminals.unknown_persona_reference",
    }
    assert not any(f.code in persona_codes for f in findings)


# --- mode-coherence checks ---------------------------------------------------


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
    when the entry names no build_profile for `osprey build` to render it from."""
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


# --- not-rendered-yet demotion + project-path name invariant -----------------


def test_lint_local_mode_missing_project_path_with_build_profile_is_a_warning(
    tmp_path,
) -> None:
    """A referenced persona whose project_path does not exist yet but carries a
    usable build_profile is a WARNING, not an error and not a mere note.

    Not an error, because nothing is misconfigured — this is the ordinary state
    of a persona added since the last build, and `osprey build` clears it. Not
    informational, because `osprey up` refuses to start until the render is
    there, so the message has to say both halves: what renders it, and that a
    start will not run meanwhile."""
    # Arrange (project_path basename matches `project`; directory not created)
    missing_path = tmp_path / "als-assistant"
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(missing_path),
                    "build_profile": "personas/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []
    not_rendered = [
        f
        for f in _warnings(findings)
        if f.code == "web_terminals.persona_project_path_not_rendered_yet"
    ]
    assert not_rendered, findings
    # The message must name the command that renders it AND the refusal that
    # stands until then. A "osprey up will render it" promise here is the exact
    # thing that outlived the behaviour it described.
    message = not_rendered[0].message
    assert "osprey build" in message
    assert "REFUSES" in message
    assert "osprey up will render" not in message
    # The hard "not a directory" error must not fire alongside the warning.
    assert not any(f.code == "web_terminals.persona_project_path_not_dir" for f in findings)


def test_lint_local_mode_missing_project_path_without_build_profile_stays_an_error(
    tmp_path,
) -> None:
    """Without a build_profile there is no delta for `osprey build` to render
    from, so a non-existent project_path remains the pre-existing hard error."""
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
    assert not any(
        f.code == "web_terminals.persona_project_path_not_rendered_yet" for f in findings
    )


def test_lint_local_mode_partial_render_missing_dockerfile_stays_an_error(tmp_path) -> None:
    """A build_profile does NOT rescue a directory that already exists but is
    incomplete. The warning is for a render that has not happened yet; a
    directory that is there but missing its Dockerfile is a partial render, and
    that stays a hard error."""
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
                    "build_profile": "personas/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_dockerfile" for f in errors)
    assert not any(
        f.code == "web_terminals.persona_project_path_not_rendered_yet" for f in findings
    )


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
                    "build_profile": "personas/assistant.yml",
                }
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_missing_config_yml" for f in errors)
    assert not any(
        f.code == "web_terminals.persona_project_path_not_rendered_yet" for f in findings
    )


def _local_mode_build_profile_config(tmp_path, build_profile: str) -> dict:
    """Local-mode config whose one persona carries `build_profile`, project_path absent."""
    return _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(tmp_path / "als-assistant"),
                    "build_profile": build_profile,
                }
            },
        }
    )


@pytest.mark.parametrize(
    "build_profile",
    [
        "control-assistant",  # bundled preset name -- the pre-delta spelling
        "control_assistant",  # same, underscore spelling
        "/abs/personas/assistant.yml",  # absolute: could name any profile on the host
        "../elsewhere/personas/assistant.yml",  # climbs out of the profile
        "personas/../assistant.yml",  # climbs back out through the right directory
        "profiles/assistant.yml",  # a sibling directory of personas/
        "assistant.yml",  # the profile root itself, not personas/
        "personas/nested/assistant.yml",  # deeper than one level: never read as a delta
    ],
)
def test_lint_local_mode_build_profile_that_deploy_rejects_is_an_error(
    tmp_path, build_profile
) -> None:
    """`osprey up` never runs lint, so a value lint blesses and deploy
    then refuses is a gate that promised the problem away. Both sides share one
    predicate, so every shape rejected at deploy time is an ERROR here — and the
    entry is never also reported as merely awaiting a render, which it is not."""
    # Act
    findings = lint_web_terminals(_local_mode_build_profile_config(tmp_path, build_profile))

    # Assert
    errors = _errors(findings)
    bad = [f for f in errors if f.code == "web_terminals.persona_build_profile_not_a_delta"]
    assert bad, f"{build_profile!r} must be an error"
    assert any("personas/assistant.yml" in f.message for f in bad)  # names the fix
    assert not any(
        f.code == "web_terminals.persona_project_path_not_rendered_yet" for f in findings
    )


def test_lint_local_mode_build_profile_shape_is_checked_even_when_rendered(tmp_path) -> None:
    """A rendered directory makes an unusable build_profile harmless only until
    someone removes it. A verdict that depended on local filesystem state would
    not be a gate, so the shape error fires for a complete render too."""
    # Arrange
    project_dir = tmp_path / "als-assistant"
    project_dir.mkdir()
    (project_dir / "Dockerfile").write_text("FROM scratch\n")
    (project_dir / "config.yml").write_text("project_name: als-assistant\n")
    config = _local_mode_build_profile_config(tmp_path, "control-assistant")

    # Act
    errors = _errors(lint_web_terminals(config))

    # Assert
    assert any(f.code == "web_terminals.persona_build_profile_not_a_delta" for f in errors)


def test_lint_local_mode_delta_valued_build_profile_is_accepted(tmp_path) -> None:
    """The one accepted shape -- what `osprey init` emits -- draws no
    shape finding at all. Lint cannot check the file exists (it holds a config,
    not the deployed project), and must not pretend otherwise."""
    # Act
    findings = lint_web_terminals(
        _local_mode_build_profile_config(tmp_path, "personas/assistant.yml")
    )

    # Assert
    assert not any(f.code == "web_terminals.persona_build_profile_not_a_delta" for f in findings)
    assert any(f.code == "web_terminals.persona_project_path_not_rendered_yet" for f in findings)


def test_lint_registry_mode_keeps_its_own_build_profile_vocabulary(tmp_path) -> None:
    """The delta rule is local-mode only. Registry mode feeds `build_profile` to
    a generated CI job as a committed profile path, so a `profiles/*.yml` value
    stays valid there and must not inherit the local-mode shape error."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "image_source": "registry",
            "default_persona": "assistant",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "build_profile": "profiles/assistant.yml",
                },
                "analysis": {"project": "als-analysis", "build_profile": "profiles/analysis.yml"},
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.persona_build_profile_not_a_delta" for f in findings)


def test_lint_local_mode_project_path_basename_not_matching_project_is_an_error(tmp_path) -> None:
    """The render-location invariant: project_path's basename must equal the
    catalog `project`, since that basename is where `osprey build` puts the
    render. A disagreement is an error even with a build_profile present."""
    # Arrange (basename "wrong-name" != project "als-assistant"; dir absent)
    project_path = tmp_path / "wrong-name"
    config = _persona_config(
        web_terminals={
            "image_source": "local",
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "project_path": str(project_path),
                    "build_profile": "personas/assistant.yml",
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
    # The name-mismatch supersedes the awaiting-a-render demotion.
    assert not any(
        f.code == "web_terminals.persona_project_path_not_rendered_yet" for f in findings
    )


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


# --- persona extra_mounts syntax ---------------------------------------------


def test_lint_valid_persona_extra_mounts_reports_no_error() -> None:
    """Well-formed compose volume strings (2 or 3 non-empty colon parts) pass."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    "extra_mounts": ["/opt/site-data:/app/site-data:ro", "cache:/app/cache"],
                },
            },
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any(f.code.startswith("web_terminals.persona_") for f in errors)
    assert not any("extra_mount" in f.code for f in errors)


def test_lint_malformed_persona_extra_mount_string_is_an_error() -> None:
    """An entry that isn't a 2-or-3-part colon string renders a broken volume line."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "personas": {
                "assistant": {
                    "project": "als-assistant",
                    # no colon at all, empty part, and too many parts
                    "extra_mounts": ["no-colon", "/a::ro", "/a:/b:ro:extra"],
                },
            },
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = [f for f in _errors(findings) if f.code == "web_terminals.persona_invalid_extra_mount"]
    assert len(errors) == 3
    assert any("no-colon" in f.message for f in errors)


def test_lint_non_list_persona_extra_mounts_is_an_error() -> None:
    """`extra_mounts` must be a list; a scalar is a distinct, reported error."""
    # Arrange
    config = _persona_config(
        web_terminals={
            "personas": {
                "assistant": {"project": "als-assistant", "extra_mounts": "/a:/b:ro"},
            },
        },
        registry={"url": "registry.example.org:5050"},
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.persona_extra_mounts_not_list" for f in errors)


def test_lint_absent_persona_extra_mounts_reports_no_error() -> None:
    """A persona that omits `extra_mounts` is never flagged — the key is optional."""
    # Arrange
    config = _persona_config(registry={"url": "registry.example.org:5050"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert not any("extra_mount" in f.code for f in errors)


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


# --- empty facility.prefix (web container-name prefix) -----------------------


def test_lint_users_with_absent_facility_prefix_is_an_error() -> None:
    """Web container names are `<facility.prefix>-nginx`/`<...>-web-<user>`, so a
    configured roster with no facility section at all renders leading-dash names
    like `-nginx`, which Docker rejects only at `osprey up`. Catch it at lint."""
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


def test_lint_default_image_tag_reports_no_warning() -> None:
    """The default (unset) `image_tag` resolves to `latest`, so the empty-tag
    check stays silent for an ordinary registry-mode config."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.empty_image_tag" for f in findings)


def test_lint_empty_expanded_image_tag_is_a_warning(monkeypatch) -> None:
    """A registry-mode `image_tag` referencing an unset env var expands to empty
    and must produce a (non-fatal) warning, never an error."""
    # Arrange
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["image_tag"] = "${IMAGE_TAG}"

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _errors(findings) == []
    assert any(f.code == "web_terminals.empty_image_tag" for f in _warnings(findings))


def test_lint_empty_image_tag_in_local_mode_reports_no_warning(monkeypatch) -> None:
    """Local mode builds `:local` images and never reads `image_tag`, so an empty
    tag there is not this check's concern."""
    # Arrange
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    config = copy.deepcopy(_CLEAN_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["image_source"] = "local"
    web_terminals["default_persona"] = "ops"
    web_terminals["personas"] = {"ops": {"project": "ops-app", "project_path": "profiles/ops"}}
    web_terminals["image_tag"] = "${IMAGE_TAG}"

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.empty_image_tag" for f in findings)


# --- nginx_image override -----------------------------------------------------


def test_lint_omitted_nginx_image_reports_nothing() -> None:
    """No `nginx_image` key (the common case) applies the render-time default and
    must never be flagged."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    assert "nginx_image" not in config["modules"]["web_terminals"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any("nginx_image" in f.code for f in findings)


def test_lint_valid_nginx_image_string_reports_nothing() -> None:
    """A non-empty string image reference is accepted silently."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["nginx_image"] = (
        "registry.example.com:5050/mirrors/nginx:1.27-alpine"
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any("nginx_image" in f.code for f in findings)


def test_lint_non_string_nginx_image_is_an_error() -> None:
    """A non-string `nginx_image` cannot be an image reference — fail closed."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["nginx_image"] = 1234

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_nginx_image" for f in errors)


def test_lint_empty_nginx_image_is_a_warning() -> None:
    """An empty/whitespace-only `nginx_image` is inert (the default applies) but
    almost certainly a mistake — a non-fatal warning."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["nginx_image"] = "   "

    # Act
    findings = lint_web_terminals(config)

    # Assert
    warnings = _warnings(findings)
    assert any(f.code == "web_terminals.empty_nginx_image" for f in warnings)
    assert not any(f.code == "web_terminals.invalid_nginx_image" for f in _errors(findings))


# --- external_origin override -------------------------------------------------


def test_lint_omitted_external_origin_reports_nothing() -> None:
    """No `external_origin` key (the common case) derives the origin from
    `deploy.fqdn` and must never be flagged."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    assert "external_origin" not in config["modules"]["web_terminals"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any("external_origin" in f.code for f in findings)


def test_lint_valid_external_origin_reports_nothing() -> None:
    """scheme://host[:port] is exactly what a browser sends as `Origin`."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["external_origin"] = "https://terminals.example.org"

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any("external_origin" in f.code for f in findings)


def test_lint_external_origin_with_a_trailing_slash_is_an_error() -> None:
    """The quiet failure this exists to catch.

    The value is compared against the browser's `Origin` header as a whole
    string, and no browser sends a trailing slash — so the deployment's pages
    all load and its every write answers 403, with the only signal inside a
    container.
    """
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["external_origin"] = "https://terminals.example.org/"

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.invalid_external_origin" for f in _errors(findings))


def test_lint_non_string_external_origin_is_an_error() -> None:
    """A non-string cannot be an origin — fail closed, as render does."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["external_origin"] = 9080

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.invalid_external_origin" for f in _errors(findings))


def test_lint_empty_external_origin_is_a_warning() -> None:
    """Inert (the derivation applies) but almost certainly a mistake."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["external_origin"] = "  "

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.empty_external_origin" for f in _warnings(findings))
    assert not any(f.code == "web_terminals.invalid_external_origin" for f in _errors(findings))


# --- auth seam checks ---------------------------------------------------------


def test_username_charset_re_is_public_for_auth_credentials() -> None:
    """`auth_credentials` imports this regex so the deploy-time charset gate and
    the lint-time check cannot drift; it is part of this module's public surface."""
    # Act / Assert
    assert lint.USERNAME_CHARSET_RE.pattern == r"^[a-z0-9][a-z0-9_-]*$"


_AUTH_CODES = frozenset(
    {
        "web_terminals.invalid_auth_stanza",
        "web_terminals.invalid_auth_method_type",
        "web_terminals.unknown_auth_method",
        "web_terminals.auth_requires_tls",
        "web_terminals.auth_insecure_http",
        "web_terminals.auth_oidc_missing_issuer",
        "web_terminals.auth_oidc_invalid_client_env",
        "web_terminals.auth_oidc_unresolvable_origin",
        "web_terminals.auth_oidc_subject_unsafe",
        "web_terminals.auth_credential_collision",
    }
)


def _auth_config(auth: object, *, tls: bool = True, fqdn: str | None = "web.example.org") -> dict:
    """A clean config with an `auth` stanza, TLS on and an origin derivable.

    Those two defaults keep each test to the one thing it is about: with TLS
    off every auth-on config also reports the transport ERROR, and without
    `deploy.fqdn` every `oidc` config also reports the origin ERROR.
    """
    config = copy.deepcopy(_CLEAN_CONFIG)
    if fqdn is not None:
        config["deploy"] = {"fqdn": fqdn}
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["auth"] = auth
    if tls:
        web_terminals["tls"] = {
            "enabled": True,
            "cert": "/etc/osprey/tls/facility.crt",
            "key": "/etc/osprey/tls/facility.key",
        }
    return config


def _auth_findings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.code in _AUTH_CODES]


def test_lint_clean_password_auth_config_reports_no_auth_findings() -> None:
    """Password auth over TLS with an unambiguous roster is a valid deployment."""
    # Arrange
    config = _auth_config({"method": "password"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _auth_findings(findings) == []
    assert _errors(findings) == []


def test_lint_absent_auth_stanza_reports_no_auth_findings() -> None:
    """The inert default (no `auth` stanza at all) must stay silent."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _auth_findings(findings) == []


def test_lint_auth_method_none_reports_no_auth_findings() -> None:
    """`method: none` is the explicit spelling of the default, over plain HTTP."""
    # Arrange
    config = _auth_config({"method": "none"}, tls=False, fqdn=None)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _auth_findings(findings) == []


def test_lint_auth_method_written_with_no_value_reports_no_auth_findings() -> None:
    """`method:` with nothing after it parses to None — an omitted value, which
    is the documented default, not a typo to reject."""
    # Arrange
    config = _auth_config({"method": None}, tls=False, fqdn=None)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _auth_findings(findings) == []


def test_lint_unknown_auth_method_is_an_error() -> None:
    """A method the sidecar cannot serve would emit an auth seam nothing answers."""
    # Arrange
    config = _auth_config({"method": "basic"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.unknown_auth_method" for f in errors)
    assert any("'basic'" in f.message for f in errors)


def test_lint_empty_auth_method_string_is_an_error() -> None:
    """An empty `method` silently falls back to 'none' at render time (auth off),
    so lint is where an operator learns the stanza is inert."""
    # Arrange
    config = _auth_config({"method": ""})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.unknown_auth_method" for f in _errors(findings))


def test_lint_non_string_auth_method_is_an_error() -> None:
    """render reads `method` defensively, so a wrong-typed one renders auth
    silently OFF — lint is the only surface that catches the type mistake."""
    # Arrange
    config = _auth_config({"method": {"password": True}})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.invalid_auth_method_type" for f in errors)
    assert not any(f.code == "web_terminals.unknown_auth_method" for f in errors)


def test_lint_non_mapping_auth_stanza_is_an_error() -> None:
    """`auth: password` (a scalar where the mapping belongs) is read as no auth
    stanza at all — the same silent-off failure, one level up."""
    # Arrange
    config = _auth_config("password")

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.invalid_auth_stanza" for f in _errors(findings))


def test_lint_unknown_auth_method_suppresses_downstream_auth_findings() -> None:
    """A method render cannot parse makes every method-keyed check meaningless;
    only the unknown-method ERROR is reported, not confused follow-ons."""
    # Arrange — no TLS and no fqdn, which would otherwise add two more findings
    config = _auth_config({"method": "basic"}, tls=False, fqdn=None)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert [f.code for f in _auth_findings(findings)] == ["web_terminals.unknown_auth_method"]


def test_lint_auth_port_joins_the_port_overlap_set() -> None:
    """With auth enabled, the sidecar's listener is a real published port and
    collides with any other source claiming it."""
    # Arrange — put the sidecar on nginx's own published port
    config = _auth_config({"method": "password", "port": 9080})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    overlap_findings = [f for f in _errors(findings) if f.code == "web_terminals.port_overlap"]
    assert any("web_terminals.auth.port" in f.message for f in overlap_findings)
    assert any("9080" in f.message for f in overlap_findings)


def test_lint_default_auth_port_joins_the_port_overlap_set() -> None:
    """The default sidecar port (9070) is claimed just as an explicit one is."""
    # Arrange
    config = _auth_config({"method": "password"})
    config["services"] = {"conflicting": {"port": 9070}}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    overlap_findings = [f for f in _errors(findings) if f.code == "web_terminals.port_overlap"]
    assert any("web_terminals.auth.port" in f.message for f in overlap_findings)


def test_lint_auth_port_absent_from_overlap_set_when_method_is_none() -> None:
    """No sidecar is rendered for `method: none`, so its port must not be
    reserved against an ordinary config that happens to use 9070."""
    # Arrange
    config = _auth_config({"method": "none", "port": 9070}, tls=False, fqdn=None)
    config["services"] = {"conflicting": {"port": 9070}}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    overlap_findings = [f for f in _errors(findings) if f.code == "web_terminals.port_overlap"]
    assert not any("web_terminals.auth.port" in f.message for f in overlap_findings)


def test_lint_auth_without_tls_is_an_error() -> None:
    """Session cookies over cleartext HTTP is refused at render time; lint says
    so at scaffold time."""
    # Arrange
    config = _auth_config({"method": "password"}, tls=False)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.auth_requires_tls" for f in errors)
    assert not any(f.code == "web_terminals.auth_insecure_http" for f in _warnings(findings))


def test_lint_auth_without_tls_and_allow_insecure_http_is_a_warning() -> None:
    """The escape hatch makes the config renderable — the risk is restated as a
    warning at every lint, not silently accepted once."""
    # Arrange
    config = _auth_config({"method": "password", "allow_insecure_http": True}, tls=False)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.auth_insecure_http" for f in _warnings(findings))
    assert not any(f.code == "web_terminals.auth_requires_tls" for f in _errors(findings))


def test_lint_auth_insecure_http_warning_is_withheld_on_loopback() -> None:
    """With `deploy.fqdn` naming loopback the deployment advertises itself as
    same-host-only, so its cookies cross no network path — the exact case the
    escape hatch exists for (and the control-assistant preset's demo posture).
    A real hostname brings the warning back with the exposure."""
    # Arrange
    config = _auth_config(
        {"method": "password", "allow_insecure_http": True}, tls=False, fqdn="127.0.0.1"
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_insecure_http" for f in _warnings(findings))
    assert not any(f.code == "web_terminals.auth_requires_tls" for f in _errors(findings))


def test_lint_auth_with_tls_reports_no_transport_finding() -> None:
    """With TLS on, `allow_insecure_http` is inert and nothing is reported."""
    # Arrange
    config = _auth_config({"method": "password", "allow_insecure_http": True})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_insecure_http" for f in _warnings(findings))
    assert not any(f.code == "web_terminals.auth_requires_tls" for f in _errors(findings))


def test_lint_auth_method_none_without_tls_reports_no_transport_finding() -> None:
    """Plain HTTP is only a finding once there is a session cookie to protect."""
    # Arrange
    config = _auth_config({"method": "none"}, tls=False, fqdn=None)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_requires_tls" for f in _errors(findings))


def test_lint_clean_oidc_auth_config_reports_no_auth_findings() -> None:
    """A complete OIDC stanza over TLS with a derivable origin is valid."""
    # Arrange
    config = _auth_config(
        {
            "method": "oidc",
            "oidc": {
                "issuer": "https://idp.example.org/realms/osprey",
                "client_id_env": "FACILITY_OIDC_CLIENT_ID",
                "client_secret_env": "FACILITY_OIDC_CLIENT_SECRET",
            },
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _auth_findings(findings) == []


def test_lint_auth_oidc_without_issuer_is_an_error() -> None:
    """The issuer has no default: without it there is no IdP to redirect to."""
    # Arrange
    config = _auth_config({"method": "oidc"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.auth_oidc_missing_issuer" for f in _errors(findings))


def test_lint_auth_oidc_with_empty_issuer_is_an_error() -> None:
    """An empty issuer is as unusable as an absent one."""
    # Arrange
    config = _auth_config({"method": "oidc", "oidc": {"issuer": ""}})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.auth_oidc_missing_issuer" for f in _errors(findings))


def test_lint_auth_oidc_omitted_client_env_names_report_no_error() -> None:
    """Both env-var names carry a documented OSPREY_AUTH_OIDC_* default, so
    omitting them is a valid config, not a missing field."""
    # Arrange
    config = _auth_config({"method": "oidc", "oidc": {"issuer": "https://idp.example.org"}})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(
        f.code == "web_terminals.auth_oidc_invalid_client_env" for f in _errors(findings)
    )


def test_lint_auth_oidc_empty_client_id_env_is_an_error() -> None:
    """An unusable name silently restores the default variable, which this
    deployment has not set — the sidecar would read an unset credential."""
    # Arrange
    config = _auth_config(
        {"method": "oidc", "oidc": {"issuer": "https://idp.example.org", "client_id_env": "  "}}
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.auth_oidc_invalid_client_env" for f in errors)
    assert any("client_id_env" in f.message for f in errors)


def test_lint_auth_oidc_non_string_client_secret_env_is_an_error() -> None:
    """The field names an environment variable; a non-string is the same silent
    fallback as an empty one."""
    # Arrange
    config = _auth_config(
        {
            "method": "oidc",
            "oidc": {"issuer": "https://idp.example.org", "client_secret_env": ["A", "B"]},
        }
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.auth_oidc_invalid_client_env" for f in errors)
    assert any("client_secret_env" in f.message for f in errors)


def test_lint_auth_oidc_without_deploy_fqdn_is_an_error() -> None:
    """The redirect_uri is built from the deployment's external origin; without
    `deploy.fqdn` there is no origin, so no login can complete."""
    # Arrange
    config = _auth_config(
        {"method": "oidc", "oidc": {"issuer": "https://idp.example.org"}}, fqdn=None
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.auth_oidc_unresolvable_origin" for f in _errors(findings))


def test_lint_auth_password_mode_without_deploy_fqdn_reports_no_origin_error() -> None:
    """Only the OIDC callback needs an absolute origin; password mode's flow is
    same-origin and relative throughout."""
    # Arrange
    config = _auth_config({"method": "password"}, fqdn=None)

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(
        f.code == "web_terminals.auth_oidc_unresolvable_origin" for f in _errors(findings)
    )


def test_lint_auth_password_mode_reports_no_oidc_findings() -> None:
    """The whole OIDC block is unread in password mode."""
    # Arrange
    config = _auth_config({"method": "password"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code.startswith("web_terminals.auth_oidc") for f in findings)


def test_lint_auth_oidc_subject_with_dollar_is_an_error() -> None:
    """A ``$`` in ``oidc_subject`` is mangled by compose-document interpolation.

    The subject travels through the compose *document* (an ``environment:``
    entry), not an env_file, so the deploy-time ``$`` scan over the env files
    never sees it — the interpolated value silently maps the user to an
    identity the IdP never issues, and that one user can never log in.
    """
    # Arrange
    config = _auth_config(
        {"method": "oidc", "oidc": {"issuer": "https://idp.example.org"}},
    )
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "oidc_subject": "user$123@example.org"},
        {"name": "gmartino", "oidc_subject": "gmartino@example.org"},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    offenders = [f for f in errors if f.code == "web_terminals.auth_oidc_subject_unsafe"]
    assert len(offenders) == 1
    assert "'thellert'" in offenders[0].message
    assert "gmartino" not in offenders[0].message


def test_lint_auth_oidc_clean_subjects_report_no_subject_finding() -> None:
    """UUIDs and emails — the common subject shapes — pass untouched."""
    # Arrange
    config = _auth_config(
        {"method": "oidc", "oidc": {"issuer": "https://idp.example.org"}},
    )
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "oidc_subject": "8f14e45f-ceea-4a5c-9c76-01dd8f7f56a2"},
        {"name": "gmartino", "oidc_subject": "gmartino@example.org"},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_oidc_subject_unsafe" for f in findings)


def test_lint_auth_oidc_subject_check_is_mode_gated() -> None:
    """A leftover subject from an earlier oidc posture is unread in password
    mode — the template emits no OIDC settings there, so there is nothing to
    mangle and nothing to flag."""
    # Arrange
    config = _auth_config({"method": "password"})
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "oidc_subject": "user$123@example.org"},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_oidc_subject_unsafe" for f in findings)


def test_lint_auth_credential_collision_is_an_error() -> None:
    """`alice-b` and `alice_b` normalize onto one OSPREY_AUTH_PW_HASH_ALICE_B
    entry — one operator's password would open the other's terminal."""
    # Arrange
    config = _auth_config({"method": "password"})
    config["modules"]["web_terminals"]["users"] = ["alice-b", "alice_b"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    collisions = [f for f in errors if f.code == "web_terminals.auth_credential_collision"]
    assert collisions
    assert "OSPREY_AUTH_PW_HASH_ALICE_B" in collisions[0].message
    assert "'alice-b'" in collisions[0].message and "'alice_b'" in collisions[0].message


def test_lint_auth_credential_collision_covers_object_form_users() -> None:
    """The roster's object form keys credentials exactly as the bare form does."""
    # Arrange
    config = _auth_config({"method": "password"})
    config["modules"]["web_terminals"]["users"] = [
        {"name": "alice-b", "index": 0},
        {"name": "alice_b", "index": 1},
    ]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert any(f.code == "web_terminals.auth_credential_collision" for f in _errors(findings))


def test_lint_distinct_usernames_report_no_auth_credential_collision() -> None:
    """A roster whose names map onto distinct suffixes is unambiguous."""
    # Arrange
    config = _auth_config({"method": "password"})

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_credential_collision" for f in _errors(findings))


def test_lint_auth_credential_collision_not_reported_in_oidc_mode() -> None:
    """OIDC deployments hold no per-user credential variable to collide on."""
    # Arrange
    config = _auth_config({"method": "oidc", "oidc": {"issuer": "https://idp.example.org"}})
    config["modules"]["web_terminals"]["users"] = ["alice-b", "alice_b"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_credential_collision" for f in _errors(findings))


def test_lint_auth_credential_collision_not_reported_when_auth_is_off() -> None:
    """No auth means no credential file; the roster keys nothing."""
    # Arrange
    config = _auth_config({"method": "none"}, tls=False, fqdn=None)
    config["modules"]["web_terminals"]["users"] = ["alice-b", "alice_b"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert not any(f.code == "web_terminals.auth_credential_collision" for f in _errors(findings))


# --- profile altitude: the same engine over a build profile's `config:` block -


def _profile_config(**web_terminals_overrides: object) -> dict:
    """A profile `config:` block in the shape the shipped presets use: dotted
    keys, with the whole module subtree under one literal
    `modules.web_terminals` key."""
    web_terminals: dict = {
        "enabled": True,
        "image_source": "local",
        "nginx_port": 9080,
        "web_base_port": 9091,
        "default_persona": "readonly",
        "users": [
            {"name": "alice", "index": 0, "persona": "readonly"},
            {"name": "bob", "index": 1, "persona": "readonly"},
        ],
        "personas": {
            "readonly": {
                "project": "ca-readonly",
                "project_path": "../ca-readonly",
                "build_profile": "control-assistant-readonly",
            }
        },
    }
    web_terminals.update(web_terminals_overrides)
    return {
        "facility.prefix": "ca",
        "deploy.fqdn": "127.0.0.1",
        "modules.web_terminals": web_terminals,
    }


def test_lint_profile_config_accepts_a_well_formed_config_block() -> None:
    """A profile whose roster and catalog are sound reports nothing."""
    # Act
    findings = lint_profile_config(_profile_config())

    # Assert
    assert findings == []


def test_lint_profile_config_skips_the_checks_a_profile_cannot_answer() -> None:
    """`project_path` points at a directory the build has not rendered yet, and
    `build_profile` still holds the preset name `osprey init` will rewrite. The
    deploy-time pass reports that; the profile-time pass must not."""
    # Arrange
    config = _profile_config()

    # Act
    profile_findings = lint_profile_config(config)
    rendered_findings = lint_web_terminals(
        {
            "facility": {"prefix": "ca"},
            "modules": {"web_terminals": config["modules.web_terminals"]},
        }
    )

    # Assert
    assert profile_findings == []
    assert {f.code for f in _errors(rendered_findings)} == {
        "web_terminals.persona_build_profile_not_a_delta"
    }


def test_lint_profile_config_reports_a_duplicate_roster_entry() -> None:
    """Multi-user checks are the point of running at this altitude at all."""
    # Arrange
    config = _profile_config(users=[{"name": "alice", "index": 0}, {"name": "alice", "index": 1}])

    # Act
    findings = lint_profile_config(config)

    # Assert
    assert any(f.code == "web_terminals.duplicate_user" for f in _errors(findings))


def test_lint_profile_config_reports_a_port_collision_with_a_declared_service() -> None:
    """A `config:` block that sets a service port puts it in the collision set."""
    # Arrange
    config = _profile_config()
    config["services.openobserve.port"] = 9092  # web_base_port index 1

    # Act
    findings = lint_profile_config(config)

    # Assert
    overlap = [f for f in _errors(findings) if f.code == "web_terminals.port_overlap"]
    assert any("services.openobserve.port" in f.message for f in overlap)


def test_lint_profile_config_lets_a_deeper_dotted_key_refine_the_subtree() -> None:
    """`modules.web_terminals.enabled: false` beside a whole
    `modules.web_terminals:` subtree wins, the way `osprey build` applies them —
    a profile that switches the module off must lint as switched off. This is
    exactly how the bundled persona presets inherit their parent's catalog."""
    # Arrange
    config = _profile_config(users=[{"name": "alice", "index": 0}, {"name": "alice", "index": 1}])
    config["modules.web_terminals.enabled"] = False

    # Act
    findings = lint_profile_config(config)

    # Assert
    assert findings == []


def test_lint_profile_config_ignores_a_config_block_that_omits_the_module() -> None:
    """A profile that never mentions the module lints clean."""
    # Act
    findings = lint_profile_config({"control_system.type": "mock", "facility.prefix": "ca"})

    # Assert
    assert findings == []


def _shipped_preset_names() -> list[str]:
    """Every bundled preset name, so a newly shipped one is covered without an
    edit here."""
    from osprey.cli.build_profile import list_presets

    return sorted(list_presets())


@pytest.mark.parametrize("preset", _shipped_preset_names())
def test_every_shipped_preset_lints_clean_at_profile_altitude(preset: str) -> None:
    """The engine runs on every profile `osprey profile validate` and
    `osprey build` check, so a shipped preset that failed it would reject our
    own reference build."""
    # Arrange
    import yaml

    from osprey.cli.build_profile import _presets_dir

    profile = yaml.safe_load((_presets_dir() / f"{preset}.yml").read_text())

    # Act
    findings = lint_profile_config(profile.get("config") or {})

    # Assert
    assert [f.message for f in findings] == []


def test_profile_config_errors_returns_only_blocking_messages() -> None:
    """The gate the commands call: warnings and notes are advisory and must not
    reach it, or an inert `nginx_image` typo would fail a build."""
    # Arrange
    config = _profile_config(nginx_image="   ")  # an empty image is a WARN
    config["modules.web_terminals"]["users"] = [
        {"name": "alice", "index": 0},
        {"name": "alice", "index": 1},
    ]

    # Act
    messages = profile_config_errors(config)

    # Assert
    assert any("duplicate name(s)" in message for message in messages)
    assert not any("nginx_image" in message for message in messages)


#: Every name that reaches the web-stack lint engine, directly or through a
#: wrapper. A new entry point into the engine adds one line here — which is the
#: point: the guard below is only as good as this list is complete.
LINT_ENTRY_POINTS = (
    "lint_profile_config",
    "profile_config_errors",
    # The command surfaces' shared merge-then-lint pairing
    # (osprey.cli.build_profile_deploy). Calling it from `validate()` would
    # wire the engine in just as surely as calling the lint directly.
    "deploy_aware_config_errors",
)


def test_the_engine_is_not_wired_into_build_profile_validate() -> None:
    """`BuildProfile.validate()` also runs during profile RESOLUTION, which
    `osprey init` goes through — running the engine there pre-empts that
    command's own persona validator, which reports every unusable catalog entry
    at once. The engine belongs to the commands; this pins that it stayed there.
    """
    # Arrange
    import inspect

    from osprey.cli.build_profile_model import BuildProfile

    source = inspect.getsource(BuildProfile.validate)

    # Assert
    for entry_point in LINT_ENTRY_POINTS:
        assert entry_point not in source, f"{entry_point} reached BuildProfile.validate()"


# ---------------------------------------------------------------------------
# Roles resolved into personas (Task 4.2 wiring)
#
# Once `effective_persona` routes a roster `role:` into a persona at lint's two
# roster walks, the EXISTING persona rules answer the role's persona too. That
# is why the authorization parser deliberately added no "a role's persona is in
# the catalog" rule of its own — it would report the same config twice.
# ---------------------------------------------------------------------------


def _role_config(role_persona: str, *, catalog: dict | None = None) -> dict:
    """A clean config whose sole roster entry reaches its persona through a role."""
    config = copy.deepcopy(_CLEAN_CONFIG)
    web = config["modules"]["web_terminals"]
    web["users"] = [{"name": "thellert", "index": 0, "role": "operator"}]
    web["authorization"] = {"roles": {"operator": {"persona": role_persona}}}
    if catalog is not None:
        web["personas"] = catalog
        config["registry"] = {"url": "registry.example.org"}
    return config


def test_lint_role_naming_an_uncatalogued_persona_is_the_unknown_persona_finding() -> None:
    """THE wiring proof: a role bound to a persona the catalog has never heard of
    is caught by the existing unknown-persona rule, reached through
    `effective_persona`. No new rule, and no duplicate report."""
    # Arrange
    config = _role_config("ghost", catalog={"cli": {"project": "cli", "build_profile": "p.yml"}})

    # Act
    codes = [f.code for f in _errors(lint_web_terminals(config))]

    # Assert
    assert "web_terminals.unknown_persona_reference" in codes
    assert codes.count("web_terminals.unknown_persona_reference") == 1


def test_lint_role_bound_persona_is_checked_like_a_pinned_one() -> None:
    """lint's `_referenced_persona_names` walk reaches a role-bound persona, so
    the per-persona rules (here: registry mode's `build_profile` requirement)
    apply to it. A role must not be a way to skip a check a pin cannot skip."""
    # Arrange
    config = _role_config("gui", catalog={"gui": {"project": "als-gui"}})

    # Act
    codes = [f.code for f in _errors(lint_web_terminals(config))]

    # Assert
    assert "web_terminals.persona_missing_build_profile" in codes


def test_lint_role_bound_persona_in_the_catalog_is_clean() -> None:
    """The feature's working shape must not itself be a finding."""
    # Arrange
    config = _role_config(
        "gui", catalog={"gui": {"project": "als-gui", "build_profile": "personas/gui.yml"}}
    )

    # Act / Assert
    assert _errors(lint_web_terminals(config)) == []


def test_lint_entry_carrying_both_a_persona_and_a_role_is_an_error() -> None:
    """Both bind the same slot, so which one governs is unwritten — and a later
    edit to the role's persona would silently not reach this entry. Lint is the
    friendly half of the pair `effective_persona` raises on."""
    # Arrange
    config = _role_config(
        "gui", catalog={"gui": {"project": "als-gui", "build_profile": "personas/gui.yml"}}
    )
    config["modules"]["web_terminals"]["users"][0]["persona"] = "cli"

    # Act
    codes = [f.code for f in _errors(lint_web_terminals(config))]

    # Assert
    assert "web_terminals.conflicting_user_persona_and_role" in codes
    # The ambiguity IS the finding: the entry is not also reported for whichever
    # of the two the lint pass might otherwise have preferred.
    assert "web_terminals.unknown_role_reference" not in codes


def test_lint_conflicting_persona_and_role_survives_an_undeclared_role() -> None:
    """The conflict is a roster shape, so it is reported whether or not the role
    itself is declared — and it replaces the undeclared-role finding."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "persona": "cli", "role": "nope"}
    ]

    # Act
    codes = [f.code for f in _errors(lint_web_terminals(config))]

    # Assert
    assert "web_terminals.conflicting_user_persona_and_role" in codes
    assert "web_terminals.unknown_role_reference" not in codes


# ---------------------------------------------------------------------------
# The one inert role source left in the stanza
#
# Its former sibling — a roster `role:` under `oidc` — is NOT inert any more:
# the render resolves that entry's persona from it and the sidecar cross-checks
# the ID token's role against it. `claims` has no such second job.
# ---------------------------------------------------------------------------

_CLAIMS_STANZA = {
    "roles": {"operator": {"persona": "cli"}},
    "claims": {"claim": "groups", "map": {"als-operators": "operator"}},
}


def _claims_config(method: str) -> dict:
    """A clean auth config carrying a full `claims:` block under ``method``."""
    config = _auth_config({"method": method} if method != "none" else {"method": "none"})
    config["modules"]["web_terminals"]["authorization"] = copy.deepcopy(_CLAIMS_STANZA)
    return config


@pytest.mark.parametrize("method", ["password", "none"])
def test_lint_a_claims_block_without_single_sign_on_is_a_warning(method: str) -> None:
    """No ID token arrives under either method, so the map is never read: the
    roles it names are silently never granted, and the config looks live."""
    # Act
    findings = lint_web_terminals(_claims_config(method))

    # Assert
    codes = [f.code for f in _warnings(findings)]
    assert codes.count("web_terminals.authorization_claims_without_oidc") == 1
    # Inert, not wrong — a facility staging a move to SSO writes exactly this.
    assert "web_terminals.authorization_claims_without_oidc" not in [
        f.code for f in _errors(findings)
    ]


def test_lint_a_claims_block_under_oidc_is_silent() -> None:
    """The control: under single sign-on the block is exactly what it claims."""
    # Act
    findings = lint_web_terminals(_claims_config("oidc"))

    # Assert
    assert "web_terminals.authorization_claims_without_oidc" not in [f.code for f in findings]


def test_lint_a_roster_role_under_oidc_is_not_reported_as_inert() -> None:
    """The asymmetry, pinned. A roster `role:` under `oidc` binds the persona
    the container is built from AND is the cross-check the sidecar refuses a
    disagreeing ID token against, so it is emitted under both methods and is
    never an inert source."""
    # Arrange
    config = _auth_config({"method": "oidc"})
    web = config["modules"]["web_terminals"]
    web["users"] = [{"name": "thellert", "index": 0, "role": "operator"}]
    web["authorization"] = {"roles": {"operator": {"persona": "cli"}}}

    # Act
    codes = [f.code for f in lint_web_terminals(config)]

    # Assert — nothing calls the roster role inert, under any spelling.
    assert [code for code in codes if "inert" in code] == []
    assert "web_terminals.authorization_claims_without_oidc" not in codes


def test_lint_does_not_raise_on_an_incoherent_authorization_stanza_with_roles() -> None:
    """A report must show every finding rather than die on the first bad entry:
    lint's roster walks resolve leniently, and the stanza is its own finding."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    config["modules"]["web_terminals"]["users"] = [
        {"name": "thellert", "index": 0, "role": "operator"}
    ]
    config["modules"]["web_terminals"]["authorization"] = {"roles": {"operator": {}}}

    # Act
    codes = [f.code for f in _errors(lint_web_terminals(config))]

    # Assert
    assert "web_terminals.invalid_authorization" in codes

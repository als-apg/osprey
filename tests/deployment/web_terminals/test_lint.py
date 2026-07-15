"""Tests for modules.web_terminals validation (osprey.deployment.web_terminals.lint)."""

from __future__ import annotations

import copy

from osprey.deployment.web_terminals.lint import Finding, lint_web_terminals

_CLEAN_CONFIG = {
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


def test_lint_missing_port_family_is_an_error() -> None:
    """A user list that can't fully resolve allocate_ports() is a consistency error."""
    # Arrange
    config = copy.deepcopy(_CLEAN_CONFIG)
    del config["modules"]["web_terminals"]["lattice_base_port"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    errors = _errors(findings)
    assert any(f.code == "web_terminals.incomplete_port_families" for f in errors)


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

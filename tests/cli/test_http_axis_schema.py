"""Tests for the ``http:`` axis on the build-profile surface.

A deployment's own service is one the framework has never heard of, so nothing
in it can say whether the port that service publishes answers HTTP.
``services.<name>.config.http`` is the service saying so, and the only thing it
changes is whether the deploy summary prints an address an operator can click.

Both authoring surfaces are exercised, as for the ``network:`` axis: the
``services:`` block and the dotted ``config:`` overrides reach the same key in
the rendered ``config.yml``, so a rule catching only one would be trivially
bypassable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile import BuildProfile, ServiceDef, _parse_profile
from osprey.cli.build_profile_schema import DEFAULT_SPEAKS_HTTP, http_errors
from osprey.errors import BuildProfileError


@pytest.fixture(autouse=True)
def _facility_data_tree(tmp_path: Path) -> None:
    """The tree every profile's ``data:`` key names, beside the profile."""
    (tmp_path / "data").mkdir(exist_ok=True)


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _http_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Return only the failures that mention an ``http`` key."""
    return [e for e in _errors(profile, profile_dir) if ".http" in e]


def _profile(**raw: Any) -> BuildProfile:
    """Parse a profile from the YAML surface, filling in the required name."""
    return _parse_profile({"name": "httpaxis", "data": "data", **raw})


# --- the accessor ---------------------------------------------------------


def test_the_default_is_off() -> None:
    """A binary-protocol service shown as a link is worse than no link."""
    assert DEFAULT_SPEAKS_HTTP is False
    assert ServiceDef(template="services/facility-mcp").speaks_http() is False
    assert ServiceDef(template="services/facility-mcp", config={}).speaks_http() is False


def test_a_declared_service_says_so() -> None:
    """``speaks_http()`` is the single place the axis default is applied."""
    service = ServiceDef(template="services/facility-mcp", config={"http": True})

    assert service.speaks_http() is True


def test_an_explicit_false_reads_as_the_default() -> None:
    assert (
        ServiceDef(template="services/facility-mcp", config={"http": False}).speaks_http() is False
    )


def test_a_non_boolean_reads_as_the_default_rather_than_being_coerced() -> None:
    """A profile that never validated must not produce a link that cannot open."""
    assert (
        ServiceDef(template="services/facility-mcp", config={"http": "yes"}).speaks_http() is False
    )


# --- the shared checker ---------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_both_booleans_pass_the_shared_checker(value: bool) -> None:
    assert http_errors(value, "services.facility-mcp.http") == []


def test_the_checker_names_the_key_and_what_the_value_means() -> None:
    (message,) = http_errors("yes", "services.facility-mcp.http")

    assert "services.facility-mcp.http" in message
    assert "'yes'" in message
    assert "true or false" in message


# --- validation, both authoring surfaces ----------------------------------


def test_a_non_boolean_in_the_services_block_is_refused(tmp_path: Path) -> None:
    profile = _profile(
        services={"facility-mcp": {"template": "services/facility-mcp", "config": {"http": "yes"}}}
    )

    assert _http_errors(profile, tmp_path) == [
        "services.facility-mcp.http must be true or false — whether this service "
        "answers HTTP on the port it publishes (got 'yes')"
    ]


def test_a_non_boolean_in_a_dotted_override_is_refused(tmp_path: Path) -> None:
    profile = _profile(
        services={"facility-mcp": {"template": "services/facility-mcp"}},
        config={"services.facility-mcp.http": 1},
    )

    assert _http_errors(profile, tmp_path) == [
        "services.facility-mcp.http must be true or false — whether this service "
        "answers HTTP on the port it publishes (got 1)"
    ]


@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_validates_on_either_surface(tmp_path: Path, value: bool) -> None:
    """A valid axis contributes no failure, on either surface."""
    (tmp_path / "services" / "facility-mcp").mkdir(parents=True)
    (tmp_path / "services" / "facility-mcp" / "docker-compose.yml.j2").write_text(
        "services: {}\n", encoding="utf-8"
    )
    profile = _profile(
        services={
            "facility-mcp": {"template": "services/facility-mcp", "config": {"http": value}},
        },
        config={"services.facility-gateway.http": value},
    )

    profile.validate(tmp_path)

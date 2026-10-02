"""Tests for the ``listens:`` / ``bind_env:`` axis on the build-profile surface.

A service on the host network publishes nothing through compose, so the readers
that decide whether a deployment is reachable from other machines learn what it
binds from the service itself: ``listens: false`` (it opens no socket) or
``bind_env: NAME`` (the variable its compose file renders the bind address
into).

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
from osprey.cli.build_profile_schema import bind_env_errors, listens_errors
from osprey.deployment.host_binding import HostBinding
from osprey.errors import BuildProfileError

_FACILITY = "site_poller"


@pytest.fixture(autouse=True)
def _facility_tree(tmp_path: Path) -> None:
    """The data tree and the facility service template every profile names."""
    (tmp_path / "data").mkdir(exist_ok=True)
    service_dir = tmp_path / "services" / _FACILITY
    service_dir.mkdir(parents=True, exist_ok=True)
    (service_dir / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _bind_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Return only the failures that mention a ``listens`` or ``bind_env`` key."""
    return [e for e in _errors(profile, profile_dir) if "listens" in e or "bind_env" in e]


def _profile(**raw: Any) -> BuildProfile:
    """Parse a profile from the YAML surface, filling in the required name."""
    return _parse_profile({"name": "bindaxis", "data": "data", **raw})


def _facility(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """A ``services:`` entry for the facility service, with the given config."""
    return {_FACILITY: {"template": f"services/{_FACILITY}", "config": config or {}}}


# --- the accessor ---------------------------------------------------------


def test_the_default_is_undeclared() -> None:
    binding = ServiceDef(template=f"services/{_FACILITY}").host_binding()

    assert binding == HostBinding()
    assert binding.declared is False


def test_listens_false_is_read() -> None:
    service = ServiceDef(template=f"services/{_FACILITY}", config={"listens": False})

    assert service.host_binding().listens is False


def test_bind_env_is_read() -> None:
    service = ServiceDef(template=f"services/{_FACILITY}", config={"bind_env": "SITE_BIND"})

    assert service.host_binding().bind_env == "SITE_BIND"


def test_malformed_values_read_as_the_default() -> None:
    service = ServiceDef(template=f"services/{_FACILITY}", config={"listens": "no", "bind_env": ""})

    assert service.host_binding() == HostBinding()


# --- the shared checkers --------------------------------------------------


def test_the_listens_checker_names_the_key_and_what_false_means() -> None:
    assert listens_errors(False, "services.site_poller.listens") == []
    (message,) = listens_errors("no", "services.site_poller.listens")

    assert message == (
        "services.site_poller.listens must be true or false — false says this "
        "service opens no listening socket (got 'no')"
    )


def test_the_bind_env_checker_wants_a_variable_name() -> None:
    assert bind_env_errors("SITE_BIND", "services.site_poller.bind_env") == []
    (message,) = bind_env_errors("0.0.0.0", "services.site_poller.bind_env")

    assert message == (
        "services.site_poller.bind_env must name the environment variable its "
        "compose template renders the bind address into (got '0.0.0.0')"
    )


def test_the_bind_env_checker_refuses_a_trailing_newline() -> None:
    (message,) = bind_env_errors("SITE_BIND\n", "services.site_poller.bind_env")

    assert message.startswith("services.site_poller.bind_env must name the environment variable")


# --- validation, both authoring surfaces ----------------------------------


@pytest.mark.parametrize("surface", ["services", "dotted"])
def test_a_non_boolean_listens_is_refused_on_either_surface(tmp_path: Path, surface: str) -> None:
    if surface == "services":
        profile = _profile(services=_facility({"listens": "no"}))
    else:
        profile = _profile(services=_facility(), config={f"services.{_FACILITY}.listens": "no"})

    assert _bind_errors(profile, tmp_path) == [
        f"services.{_FACILITY}.listens must be true or false — false says this "
        "service opens no listening socket (got 'no')"
    ]


@pytest.mark.parametrize("surface", ["services", "dotted"])
def test_a_bind_env_that_is_not_a_variable_name_is_refused(tmp_path: Path, surface: str) -> None:
    if surface == "services":
        profile = _profile(services=_facility({"bind_env": "0.0.0.0"}))
    else:
        profile = _profile(
            services=_facility(), config={f"services.{_FACILITY}.bind_env": "0.0.0.0"}
        )

    assert _bind_errors(profile, tmp_path) == [
        f"services.{_FACILITY}.bind_env must name the environment variable its "
        "compose template renders the bind address into (got '0.0.0.0')"
    ]


@pytest.mark.parametrize("split", [False, True], ids=["same-surface", "split-surfaces"])
def test_listens_false_with_a_bind_env_is_refused(tmp_path: Path, split: bool) -> None:
    if split:
        profile = _profile(
            services=_facility({"listens": False}),
            config={f"services.{_FACILITY}.bind_env": "SITE_BIND"},
        )
    else:
        profile = _profile(services=_facility({"listens": False, "bind_env": "SITE_BIND"}))

    assert _bind_errors(profile, tmp_path) == [
        f"services.{_FACILITY} declares both `listens: false` and `bind_env:`; a "
        "service that opens no socket has no bind address. Remove one."
    ]


def test_listens_true_validates_and_reads_as_undeclared(tmp_path: Path) -> None:
    profile = _profile(services=_facility({"network": "host", "listens": True}))

    profile.validate(tmp_path)

    assert profile.services[_FACILITY].host_binding().declared is False


def test_a_declaration_on_a_bridge_mode_service_validates(tmp_path: Path) -> None:
    """Readers consult it only under host; flipping the network must not force a delete."""
    profile = _profile(services=_facility({"network": "bridge", "bind_env": "SITE_BIND"}))

    profile.validate(tmp_path)


# --- OSPREY's own services ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        (
            {"services": {"qmd": {"template": "osprey.qmd", "config": {"listens": False}}}},
            "services.qmd.listens",
        ),
        ({"config": {"services.teams_bridge.listens": False}}, "services.teams_bridge.listens"),
        (
            {"config": {"services.event_dispatcher.bind_env": "SITE_BIND"}},
            "services.event_dispatcher.bind_env",
        ),
    ],
    ids=["services-entry", "dotted", "dispatch-half"],
)
def test_a_declaration_on_a_bundled_service_is_refused(
    tmp_path: Path, raw: dict[str, Any], key: str
) -> None:
    name = key.split(".")[1]
    profile = _profile(**raw)

    assert _bind_errors(profile, tmp_path) == [
        f"`{key}` is declared by OSPREY for its bundled {name} service. Remove it. "
        f"To declare your own, claim the service: `osprey scaffold claim services/{name}`."
    ]


def test_a_declaration_on_a_claimed_bundled_service_validates(tmp_path: Path) -> None:
    claimed = tmp_path / "services" / "archive"
    claimed.mkdir(parents=True)
    (claimed / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    profile = _profile(
        services={"archive": {"template": "osprey.archive", "config": {"listens": False}}}
    )

    profile.validate(tmp_path)

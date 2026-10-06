"""The one reader of a service's host-binding declaration.

``host_binding_of`` is where the defaults apply, so every consumer — the
host-port preflight, the off-host bind check and the reach contracts — reads a
``services.<name>`` block the same way. A value of the wrong type reads as the
default rather than being coerced: a coerced ``listens: false`` would hide a
real socket from the exposure check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import osprey
from osprey.deployment.host_binding import (
    BIND_ENV_KEY,
    BUNDLED_HOST_BINDINGS,
    LISTENS_KEY,
    HostBinding,
    host_binding_of,
    osprey_owns_binding,
)

_TEMPLATES = Path(osprey.__file__).parent / "templates" / "services"


def _template_text(name: str) -> str:
    return (_TEMPLATES / name / "docker-compose.yml.j2").read_text(encoding="utf-8")


def test_the_key_names_are_the_ones_a_profile_spells() -> None:
    assert LISTENS_KEY == "listens"
    assert BIND_ENV_KEY == "bind_env"


@pytest.mark.parametrize("block", [None, {}, "host", ["listens"], 0])
def test_an_empty_or_non_mapping_block_is_undeclared(block: Any) -> None:
    binding = host_binding_of(block)

    assert binding == HostBinding()
    assert binding.listens is True
    assert binding.bind_env is None
    assert binding.declared is False


def test_listens_false_is_read_and_declared() -> None:
    binding = host_binding_of({"network": "host", "listens": False})

    assert binding.listens is False
    assert binding.declared is True


def test_listens_true_alone_is_undeclared() -> None:
    assert host_binding_of({"listens": True}).declared is False


def test_bind_env_is_read_and_declared() -> None:
    binding = host_binding_of({"network": "host", "bind_env": "SITE_BIND"})

    assert binding.bind_env == "SITE_BIND"
    assert binding.listens is True
    assert binding.declared is True


@pytest.mark.parametrize(
    "block",
    [
        {"listens": "no"},
        {"listens": 0},
        {"listens": None},
        {"bind_env": ""},
        {"bind_env": 1},
        {"bind_env": ["SITE_BIND"]},
    ],
)
def test_junk_values_read_as_the_default(block: dict[str, Any]) -> None:
    assert host_binding_of(block) == HostBinding()


# --- OSPREY's own declarations ---------------------------------------------


def test_every_host_capable_template_has_a_bundled_declaration() -> None:
    """A template that can render host mode has exactly one declaration entry."""
    host_capable = {
        path.parent.name
        for path in _TEMPLATES.glob("*/docker-compose.yml.j2")
        if "_network_axis.j2" in path.read_text(encoding="utf-8")
    }

    assert host_capable == set(BUNDLED_HOST_BINDINGS)


def test_every_bundled_bind_variable_is_rendered_by_its_template() -> None:
    for name, entry in BUNDLED_HOST_BINDINGS.items():
        if entry.binding.bind_env is not None:
            assert f"{entry.binding.bind_env}:" in _template_text(name), name


def test_every_outbound_only_template_publishes_nothing() -> None:
    for name, entry in BUNDLED_HOST_BINDINGS.items():
        if not entry.binding.listens:
            assert "net.ports(" not in _template_text(name), name


def test_every_bundled_declaration_says_why() -> None:
    for name, entry in BUNDLED_HOST_BINDINGS.items():
        assert entry.why, name


@pytest.mark.parametrize(
    ("name", "template", "claimed", "owned"),
    [
        ("archive", None, False, True),
        ("archive", "osprey.archive", False, True),
        ("archive", "osprey.archive", True, False),
        ("archive", "services/archive", False, False),
        ("site_poller", None, False, False),
    ],
)
def test_osprey_owns_an_unclaimed_bundled_service(
    name: str, template: str | None, claimed: bool, owned: bool
) -> None:
    assert osprey_owns_binding(name, template=template, claimed=claimed) is owned

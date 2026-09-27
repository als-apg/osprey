"""The one reader of a service's host-binding declaration.

``host_binding_of`` is where the defaults apply, so every consumer — the
host-port preflight, the off-host bind check and the reach contracts — reads a
``services.<name>`` block the same way. A value of the wrong type reads as the
default rather than being coerced: a coerced ``listens: false`` would hide a
real socket from the exposure check.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.deployment.host_binding import (
    BIND_ENV_KEY,
    LISTENS_KEY,
    HostBinding,
    host_binding_of,
)


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

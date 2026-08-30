"""Tests for the top-level ``port_base`` profile shorthand.

``port_base: 42000`` is the short spelling of
``config: {deployment.port_base: 42000}`` — the convenient way to move a whole
deployment off the default 10000 block (``osprey init --set port_base=42000``),
so a dev or CI stack can never collide with a real deployment running on the
defaults. Like ``connector``, the shorthand is folded into the literal dotted
config key on every path a profile can arrive by, and its value is range-checked
by the same resolver every runtime consumer uses, so a base whose thousand-port
block cannot exist fails the parse instead of the deploy.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile import _KNOWN_PROFILE_KEYS, _parse_profile
from osprey.cli.build_profile_load import PORT_BASE_PROFILE_KEY
from osprey.cli.build_profile_resolve import (
    SHORTHAND_OVERRIDE_KEYS,
    merge_cli_overrides,
)
from osprey.errors import BuildProfileError
from osprey.port_layout import PORT_BASE_CONFIG_KEY

pytestmark = pytest.mark.unit


def _minimal(**extra) -> dict:
    return {"name": "demo", **extra}


# ── the shorthand is part of the schema ──────────────────────────────────────


def test_port_base_is_a_known_profile_key() -> None:
    """A profile spelling ``port_base:`` is not rejected as an unknown key."""
    assert PORT_BASE_PROFILE_KEY in _KNOWN_PROFILE_KEYS


def test_port_base_joins_the_shorthand_override_keys() -> None:
    assert PORT_BASE_PROFILE_KEY in SHORTHAND_OVERRIDE_KEYS


# ── folding into the literal dotted config key ───────────────────────────────


def test_parse_folds_shorthand_into_dotted_config_key() -> None:
    profile = _parse_profile(_minimal(port_base=42000))

    assert profile.config[PORT_BASE_CONFIG_KEY] == 42000


def test_parse_consumes_the_shorthand_key() -> None:
    raw = _minimal(port_base=42000)
    _parse_profile(raw)

    assert PORT_BASE_PROFILE_KEY not in raw


def test_shorthand_overrides_an_existing_literal_key() -> None:
    profile = _parse_profile(_minimal(port_base=42000, config={PORT_BASE_CONFIG_KEY: 11000}))

    assert profile.config[PORT_BASE_CONFIG_KEY] == 42000


def test_merge_cli_overrides_folds_set_shorthand() -> None:
    raw = merge_cli_overrides(_minimal(), (), ("port_base=42000",))

    assert raw["config"][PORT_BASE_CONFIG_KEY] == 42000
    assert PORT_BASE_PROFILE_KEY not in raw


# ── validation: the resolver's rules apply at parse time ─────────────────────


@pytest.mark.parametrize("bad", ["a-string", True, 12.5, None])
def test_a_non_integer_base_is_refused(bad) -> None:
    with pytest.raises(BuildProfileError, match="port_base"):
        _parse_profile(_minimal(port_base=bad))


@pytest.mark.parametrize("bad", [500, 65000])
def test_an_impossible_block_is_refused(bad) -> None:
    """Below 1024, or a block running past 65535 — the same rule
    ``resolve_port_base`` enforces for every runtime consumer."""
    with pytest.raises(BuildProfileError, match="port_base"):
        _parse_profile(_minimal(port_base=bad))


def test_the_resolved_base_reaches_the_deployment_ports() -> None:
    """The folded key is the one every derived port is computed from."""
    profile = _parse_profile(_minimal(port_base=42000))

    assert profile._resolved_port_base() == 42000

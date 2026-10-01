"""The MCP health probe address a web-terminal render states (pure functions)."""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_health import (
    health_config_overrides,
    health_url_key_errors,
    serves_web_terminals,
)

ENABLED = {"modules.web_terminals.enabled": True}


@pytest.mark.parametrize(
    "config",
    [
        {"modules.web_terminals.enabled": True},
        {"modules": {"web_terminals": {"enabled": True}}},
        {"modules.web_terminals": {"enabled": True}},
    ],
    ids=["dotted", "nested", "prefix"],
)
def test_a_deployment_that_enables_web_terminals_serves_them(config: dict[str, Any]) -> None:
    assert serves_web_terminals(config) is True


def test_a_persona_render_serves_web_terminals_with_the_module_switched_off() -> None:
    config = {
        "modules.web_terminals": {"enabled": True, "personas": {"ro": {"project": "x-ro"}}},
        "modules.web_terminals.enabled": False,
    }
    assert serves_web_terminals(config) is True


@pytest.mark.parametrize(
    "config",
    [{}, {"modules.web_terminals.enabled": False}, None, []],
    ids=["empty", "disabled", "none", "list"],
)
def test_a_profile_without_web_terminals_serves_none(config: Any) -> None:
    assert serves_web_terminals(config) is False


@pytest.mark.parametrize("enabled", [True, False])
def test_an_unrelated_config_conflict_does_not_decide_the_verdict(enabled: bool) -> None:
    config = {"env": "not-a-list", "env.required": ["x"], "modules.web_terminals.enabled": enabled}
    assert serves_web_terminals(config) is enabled


def test_the_override_pins_host_url_only_where_terminals_are_served() -> None:
    assert health_config_overrides(ENABLED) == {"health.auto.mcp.url_key": "host_url"}
    assert health_config_overrides({}) == {}


@pytest.mark.parametrize(
    ("spelling", "health"),
    [
        ("health.auto.mcp.url_key", {"health.auto.mcp.url_key": "docker_url"}),
        ("health.auto.mcp: url_key", {"health.auto.mcp": {"url_key": "docker_url"}}),
        ("health: auto: mcp: url_key", {"health": {"auto": {"mcp": {"url_key": "docker_url"}}}}),
    ],
    ids=["dotted", "prefix", "nested"],
)
def test_a_contradicting_url_key_is_refused_in_every_spelling(
    spelling: str, health: dict[str, Any]
) -> None:
    errors = health_url_key_errors({**ENABLED, **health})
    assert len(errors) == 1
    assert spelling in errors[0]
    assert "'docker_url'" in errors[0]
    assert errors[0].endswith("Remove the line from profile.yml.")


def test_an_agreeing_url_key_is_accepted() -> None:
    assert health_url_key_errors({**ENABLED, "health.auto.mcp.url_key": "host_url"}) == []


def test_a_render_without_terminals_keeps_the_operators_url_key() -> None:
    config = {"health.auto.mcp.url_key": "docker_url"}
    assert health_url_key_errors(config) == []
    assert health_config_overrides(config) == {}

"""The login throttle's settings cross the compose overlay to the auth sidecar.

``modules.web_terminals.auth.throttle`` reaches the sidecar over the seam every
other sidecar setting uses — the rendered ``environment:`` block — and is read
back here by the sidecar's own parser, so a rename on either side fails. An
unset key emits no line and the sidecar's own default applies. A value the
throttle cannot be built with is refused by the render and reported by lint,
through the one predicate the throttle itself uses.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from osprey.deployment.web_terminals.lint import _check_auth_throttle
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar import app as app_mod
from osprey.services.auth_sidecar.app import AuthSettings
from osprey.services.auth_sidecar.throttle import (
    DEFAULT_FORGET_AFTER,
    DEFAULT_INITIAL_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MULTIPLIER,
)

_THROTTLE_PREFIX = "OSPREY_AUTH_THROTTLE_"

DEFAULTS = {
    "initial_delay": DEFAULT_INITIAL_DELAY,
    "multiplier": DEFAULT_MULTIPLIER,
    "max_delay": DEFAULT_MAX_DELAY,
    "forget_after": DEFAULT_FORGET_AFTER,
}

UNUSABLE: list[Any] = [
    {"max_delay_s": 0.5},
    {"initial_delay_s": 60},
    {"initial_delay_s": 0},
    {"multiplier": 0.5},
    {"forget_after_s": -1},
    {"initial_delay_s": True},
    {"max_delay_s": "30s"},
    {"max_delay_s": float("inf")},
    {"initial_delay_s": float("nan")},
    {"max_delay": 60},
    5,
]
UNUSABLE_IDS = [
    "cap-below-initial",
    "initial-above-default-cap",
    "zero-initial",
    "shrinking-multiplier",
    "negative-forget-after",
    "bool-initial",
    "string-cap",
    "infinite-cap",
    "nan-initial",
    "unknown-key",
    "not-a-mapping",
]
EXPECTED_KEY = [
    "max_delay_s",
    "max_delay_s",
    "initial_delay_s",
    "multiplier",
    "forget_after_s",
    "initial_delay_s",
    "max_delay_s",
    "max_delay_s",
    "initial_delay_s",
    "max_delay",
    None,
]


def _web_terminals(throttle: Any = None, *, authored: bool = True) -> dict:
    auth: dict[str, Any] = {"method": "password", "allow_insecure_http": True}
    if authored:
        auth["throttle"] = throttle
    return {"enabled": True, "users": ["alice", "bob"], "auth": auth}


def _config(throttle: Any = None, *, authored: bool = True) -> dict:
    """A sidecar-bearing render config, optionally carrying an ``auth.throttle`` block."""
    return {
        "facility": {"prefix": "dls", "name": "Demo Light Source"},
        "system": {"timezone": "America/Los_Angeles"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": _web_terminals(throttle, authored=authored)},
    }


def _sidecar_env(config: dict) -> dict[str, str]:
    """The sidecar's rendered environment, as the mapping its parser reads."""
    overlay = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    lines = overlay["services"]["auth"]["environment"]
    return dict(line.split("=", 1) for line in lines)


def test_an_unset_throttle_emits_no_throttle_variables() -> None:
    env = _sidecar_env(_config(authored=False))

    assert not [name for name in env if name.startswith(_THROTTLE_PREFIX)]
    assert AuthSettings.from_env(env).throttle_parameters == DEFAULTS


def test_each_throttle_setting_reaches_the_sidecar() -> None:
    env = _sidecar_env(
        _config({"initial_delay_s": 2, "multiplier": 1.5, "max_delay_s": 90, "forget_after_s": 600})
    )

    assert AuthSettings.from_env(env).throttle_parameters == {
        "initial_delay": 2.0,
        "multiplier": 1.5,
        "max_delay": 90.0,
        "forget_after": 600.0,
    }


def test_one_authored_setting_leaves_the_others_on_their_defaults() -> None:
    env = _sidecar_env(_config({"max_delay_s": 60}))

    assert [name for name in env if name.startswith(_THROTTLE_PREFIX)] == [
        f"{_THROTTLE_PREFIX}MAX_DELAY"
    ]
    assert AuthSettings.from_env(env).throttle_parameters == dict(DEFAULTS, max_delay=60.0)


@pytest.mark.parametrize(
    ("throttle", "key"), list(zip(UNUSABLE, EXPECTED_KEY, strict=True)), ids=UNUSABLE_IDS
)
def test_render_refuses_an_unusable_throttle(throttle: Any, key: str | None) -> None:
    dotted = "modules.web_terminals.auth.throttle" + (f".{key}" if key else "")
    with pytest.raises(ValueError, match=dotted.replace(".", r"\.")):
        render_web_terminals(_config(throttle))


@pytest.mark.parametrize(
    "throttle",
    [*UNUSABLE, {"max_delay_s": 60}, None],
    ids=[
        *UNUSABLE_IDS,
        "usable",
        "null",
    ],
)
def test_lint_and_render_refuse_the_same_throttle(throttle: Any) -> None:
    try:
        render_web_terminals(_config(throttle))
    except ValueError:
        refused = True
    else:
        refused = False

    errors = [f for f in _check_auth_throttle(_web_terminals(throttle)) if f.severity == "error"]
    assert bool(errors) is refused


def test_the_env_names_render_writes_are_the_ones_the_sidecar_reads() -> None:
    env = _sidecar_env(
        _config({"initial_delay_s": 2, "multiplier": 3, "max_delay_s": 90, "forget_after_s": 600})
    )

    rendered = {name for name in env if name.startswith(_THROTTLE_PREFIX)}
    assert rendered == set(app_mod._THROTTLE_ENV.values())

"""``.env.users`` drift: the file the terminals RUN with versus the chain.

``.env.users`` is rendered from the env chain once and handed to every
per-user web-terminal container. When the chain moves afterwards -- a provider
key rotated in ``.env`` -- the terminals keep running on the value the render
still holds, and every one of them fails authentication on its first prompt
while ``.env`` looks fine. Pinned here:

- a file OSPREY rendered (banner, nothing added by hand) is OSPREY's to
  re-render, and ``osprey up`` does so whenever the chain has moved;
- a file an operator authored (no banner, or lines under the banner the render
  would not write) is never rewritten -- but a provider secret in it that
  disagrees with the chain refuses the deploy, naming the variable and the
  remedy and never a value;
- the collect-all preflight asks the same question the gate raises on.
"""

from __future__ import annotations

import os

import pytest

from osprey.deployment.web_terminals import env_production
from osprey.utils.dotenv import ENV_USERS_BANNER, parse_dotenv_file

_CC_CONFIG = {
    "facility": {},
    "system": {"timezone": "UTC"},
    "claude_code": {"provider": "cborg"},
    "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
}

# Distinctive values, so "never a value in a message" can be asserted without
# colliding with ordinary words ("old" is a substring of "would").
_STALE = "sk-stale-0001"
_FRESH = "sk-fresh-0002"


def _write_dotenv(path, values: dict) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def _generate(tmp_path, env: dict) -> str:
    _write_dotenv(tmp_path / ".env", env)
    path = env_production.ensure_env_production(_CC_CONFIG, tmp_path)
    return path.read_text(encoding="utf-8")


def test_drift_is_none_without_the_file(tmp_path):
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})
    assert env_production.users_env_drift(_CC_CONFIG, tmp_path) is None


def test_drift_is_none_without_a_chain(tmp_path):
    (tmp_path / ".env.users").write_text(f"CBORG_API_KEY={_STALE}\n", encoding="utf-8")
    assert env_production.users_env_drift(_CC_CONFIG, tmp_path) is None


def test_drift_is_none_when_the_render_matches(tmp_path):
    _generate(tmp_path, {"CBORG_API_KEY": _FRESH})
    assert env_production.users_env_drift(_CC_CONFIG, tmp_path) is None


def test_generated_file_is_rerendered_when_the_chain_moves(tmp_path, caplog):
    """The key in .env was rotated; the terminals kept the value .env.users
    still carried. A file OSPREY rendered is OSPREY's to re-render."""
    before = _generate(tmp_path, {"CBORG_API_KEY": _STALE})
    assert before.startswith(ENV_USERS_BANNER)

    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})
    with caplog.at_level("INFO"):
        path = env_production.ensure_env_production(_CC_CONFIG, tmp_path)

    assert parse_dotenv_file(path)["CBORG_API_KEY"] == _FRESH
    assert path.read_text(encoding="utf-8").startswith(ENV_USERS_BANNER)
    assert (path.stat().st_mode & 0o777) == 0o600
    assert "CBORG_API_KEY" in caplog.text
    assert _STALE not in caplog.text and _FRESH not in caplog.text  # values never logged


def test_generated_file_that_matches_is_left_untouched(tmp_path):
    _generate(tmp_path, {"CBORG_API_KEY": _FRESH})
    path = tmp_path / ".env.users"
    os.utime(path, (1_000_000, 1_000_000))

    env_production.ensure_env_production(_CC_CONFIG, tmp_path)

    assert path.stat().st_mtime == 1_000_000


def test_authored_file_with_a_stale_secret_refuses_naming_the_var_not_the_value(tmp_path):
    """No banner: the file is the operator's and is not rewritten -- but a
    provider secret that disagrees with the chain is the stale-key 401 waiting
    on every terminal, so the deploy refuses with the remedy."""
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})
    authored = f"CBORG_API_KEY={_STALE}\nTZ=UTC\n"
    (tmp_path / ".env.users").write_text(authored, encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        env_production.ensure_env_production(_CC_CONFIG, tmp_path)

    message = str(excinfo.value)
    assert "CBORG_API_KEY" in message
    assert "osprey users env --output .env.users" in message
    assert _STALE not in message and _FRESH not in message
    assert (tmp_path / ".env.users").read_text(encoding="utf-8") == authored


def test_authored_file_that_agrees_on_the_secret_is_returned_as_is(tmp_path):
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})
    authored = f"CBORG_API_KEY={_FRESH}\nMY_OWN=thing\n"
    (tmp_path / ".env.users").write_text(authored, encoding="utf-8")

    path = env_production.ensure_env_production(_CC_CONFIG, tmp_path)

    assert path.read_text(encoding="utf-8") == authored
    assert env_production.users_env_drift(_CC_CONFIG, tmp_path) is None


def test_generated_file_with_operator_additions_counts_as_authored(tmp_path):
    """A line under the banner the render would not write is a hand edit;
    re-rendering would drop it. The file is the operator's from then on."""
    rendered = _generate(tmp_path, {"CBORG_API_KEY": _STALE})
    (tmp_path / ".env.users").write_text(rendered + "MY_OWN=thing\n", encoding="utf-8")
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})

    drift = env_production.users_env_drift(_CC_CONFIG, tmp_path)

    assert drift is not None
    assert drift.generated is False
    assert drift.stale_vars == ("CBORG_API_KEY",)
    with pytest.raises(RuntimeError, match="CBORG_API_KEY"):
        env_production.ensure_env_production(_CC_CONFIG, tmp_path)
    assert "MY_OWN=thing" in (tmp_path / ".env.users").read_text(encoding="utf-8")


def test_drift_problem_is_asked_only_for_an_authored_file(tmp_path):
    """The collect-all preflight asks what the gate raises on: a generated
    file is re-rendered, so it is never a problem to report."""
    _generate(tmp_path, {"CBORG_API_KEY": _STALE})
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})
    assert env_production.users_env_drift_problem(_CC_CONFIG, tmp_path) is None

    (tmp_path / ".env.users").write_text(f"CBORG_API_KEY={_STALE}\n", encoding="utf-8")
    problem = env_production.users_env_drift_problem(_CC_CONFIG, tmp_path)
    assert problem is not None and "CBORG_API_KEY" in problem


def test_keyless_provider_drift_is_not_a_secret_drift(tmp_path):
    """OLLAMA_API_KEY differing cannot produce an authentication failure, so
    an authored file is not refused over it."""
    config = {
        "facility": {},
        "system": {"timezone": "UTC"},
        "api": {"providers": {"ollama": {"base_url": "http://localhost:11434"}}},
        "claude_code": {"provider": "ollama"},
        "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
    }
    _write_dotenv(tmp_path / ".env", {"OLLAMA_API_KEY": _FRESH})
    (tmp_path / ".env.users").write_text(f"OLLAMA_API_KEY={_STALE}\n", encoding="utf-8")

    assert env_production.users_env_drift(config, tmp_path) is None
    env_production.ensure_env_production(config, tmp_path)


def test_the_preflight_report_carries_the_drift_refusal(tmp_path):
    """Same position as the generation question: the collect-all pass reports
    the stale secret instead of costing a deploy attempt to meet it."""
    from osprey.deployment.web_terminals.provision import web_terminal_preflight_report

    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _FRESH})
    (tmp_path / ".env.users").write_text(f"CBORG_API_KEY={_STALE}\n", encoding="utf-8")

    blocking, _advisories = web_terminal_preflight_report(_CC_CONFIG, repo_root=tmp_path)

    assert any("CBORG_API_KEY" in problem for problem, _remedy in blocking)


# A provider switch renames the secret: the render now writes a variable the
# file never carried, and stops writing one it did.
_SWITCHED_CONFIG = {**_CC_CONFIG, "claude_code": {"provider": "anthropic"}}


def test_provider_switch_rerenders_an_unedited_render(tmp_path):
    """The file OSPREY wrote for the old provider is still OSPREY's: the old
    provider's key is one its own render put there, not a hand edit, so the
    switch re-renders the file instead of turning it into an authored one."""
    _generate(tmp_path, {"CBORG_API_KEY": _STALE})
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": _STALE, "ANTHROPIC_API_KEY": _FRESH})

    drift = env_production.users_env_drift(_SWITCHED_CONFIG, tmp_path)
    assert drift is not None and drift.generated is True

    path = env_production.ensure_env_production(_SWITCHED_CONFIG, tmp_path)
    present = parse_dotenv_file(path)
    assert present.get("ANTHROPIC_API_KEY") == _FRESH
    assert "CBORG_API_KEY" not in present


def test_authored_file_missing_the_new_providers_secret_refuses(tmp_path):
    """A file without the credential the terminals authenticate with is every
    terminal opening on a login prompt, whoever wrote it. An authored file is
    not rewritten, so the deploy refuses, naming the variable and the remedy."""
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": _FRESH})
    authored = f"CBORG_API_KEY={_STALE}\nTZ=UTC\n"
    (tmp_path / ".env.users").write_text(authored, encoding="utf-8")

    drift = env_production.users_env_drift(_SWITCHED_CONFIG, tmp_path)
    assert drift is not None
    assert drift.generated is False
    assert drift.missing_vars == ("ANTHROPIC_API_KEY",)

    with pytest.raises(RuntimeError) as excinfo:
        env_production.ensure_env_production(_SWITCHED_CONFIG, tmp_path)
    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "osprey users env --output .env.users" in message
    assert _STALE not in message and _FRESH not in message
    assert (tmp_path / ".env.users").read_text(encoding="utf-8") == authored


def test_edited_render_missing_the_new_providers_secret_refuses(tmp_path):
    """Once the operator has edited the render, the old key could be theirs:
    the file is authored, and the missing credential refuses the deploy."""
    rendered = _generate(tmp_path, {"CBORG_API_KEY": _STALE})
    (tmp_path / ".env.users").write_text(rendered + "MY_OWN=thing\n", encoding="utf-8")
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": _FRESH})

    drift = env_production.users_env_drift(_SWITCHED_CONFIG, tmp_path)
    assert drift is not None and drift.generated is False
    assert drift.missing_vars == ("ANTHROPIC_API_KEY",)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        env_production.ensure_env_production(_SWITCHED_CONFIG, tmp_path)


def test_extra_line_beside_the_old_providers_key_keeps_the_file_authored(tmp_path):
    """Only a provider secret is an earlier render's line. Any other variable the
    render would not write is the operator's, so the file stays theirs and the
    missing credential refuses the deploy instead of a re-render dropping it."""
    legacy = ENV_USERS_BANNER + f"CBORG_API_KEY={_STALE}\nHTTPS_PROXY=http://proxy:3128\n"
    (tmp_path / ".env.users").write_text(legacy, encoding="utf-8")
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": _FRESH})

    assert env_production.users_env_drift_problem(_SWITCHED_CONFIG, tmp_path) is not None
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        env_production.ensure_env_production(_SWITCHED_CONFIG, tmp_path)
    assert (tmp_path / ".env.users").read_text(encoding="utf-8") == legacy


def test_leftover_key_of_an_unknown_provider_keeps_the_file_authored(tmp_path):
    """A custom provider removed from every config is no longer known, so its
    leftover key cannot be told from an operator's line: the safe answer is
    to refuse with the remedy rather than re-render it away."""
    legacy = ENV_USERS_BANNER + f"GONE_GATEWAY_TOKEN={_STALE}\n"
    (tmp_path / ".env.users").write_text(legacy, encoding="utf-8")
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": _FRESH})

    drift = env_production.users_env_drift(_SWITCHED_CONFIG, tmp_path)
    assert drift is not None and drift.generated is False
    assert drift.missing_vars == ("ANTHROPIC_API_KEY",)


def test_custom_provider_key_from_a_config_counts_as_a_render_line(tmp_path):
    """A provider the config's ``api.providers`` declares is known: switching
    away from it to a built-in leaves a key the earlier render wrote."""
    config = {
        **_SWITCHED_CONFIG,
        "api": {"providers": {"site-gw": {"base_url": "https://gw.example"}}},
    }
    legacy = ENV_USERS_BANNER + f"SITE_GW_API_KEY={_STALE}\n"
    (tmp_path / ".env.users").write_text(legacy, encoding="utf-8")
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": _FRESH})

    path = env_production.ensure_env_production(config, tmp_path)

    present = parse_dotenv_file(path)
    assert present.get("ANTHROPIC_API_KEY") == _FRESH
    assert "SITE_GW_API_KEY" not in present


def test_health_reports_the_missing_credential(tmp_path):
    """``osprey health`` asks the same function, so it no longer reports a file
    without the terminals' credential as agreeing with .env."""
    from osprey.health.core.file_system import _check_users_env

    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": _FRESH})
    (tmp_path / ".env.users").write_text(f"CBORG_API_KEY={_STALE}\n", encoding="utf-8")

    [row] = _check_users_env(_SWITCHED_CONFIG, tmp_path)

    assert row.status.name == "ERROR"
    assert "ANTHROPIC_API_KEY" in row.message
    assert _STALE not in row.message and _FRESH not in row.message

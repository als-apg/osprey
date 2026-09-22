"""The ``file_system`` category's ``users_env`` row.

``.env.users`` is what the web terminals run with; ``.env`` is what the rest of
the deployment (and every other health row) reads. The row says whether the
two still agree on the provider secrets -- the one drift a healthy-looking
``.env`` cannot reveal, and the one that 401s every terminal on its first
prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.health.core.file_system import file_system
from osprey.health.models import CheckResult, Status

_CONFIG = {
    "facility": {"timezone": "UTC"},
    "claude_code": {"provider": "cborg"},
    "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
}


def _run(config: dict[str, Any], cwd: Path) -> dict[str, CheckResult]:
    return {r.name: r for r in file_system(config, None, cwd=cwd)()}


def _dotenv(path: Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def test_no_users_env_file_emits_no_row(tmp_path: Path) -> None:
    _dotenv(tmp_path / ".env", {"CBORG_API_KEY": "k"})
    assert "users_env" not in _run(_CONFIG, tmp_path)


def test_agreeing_file_is_ok(tmp_path: Path) -> None:
    _dotenv(tmp_path / ".env", {"CBORG_API_KEY": "k"})
    _dotenv(tmp_path / ".env.users", {"CBORG_API_KEY": "k"})
    assert _run(_CONFIG, tmp_path)["users_env"].status is Status.OK


def test_stale_secret_is_an_error_naming_the_var_and_the_remedy(tmp_path: Path) -> None:
    """.env was re-keyed, .env.users was not: every terminal 401s while .env
    looks fine. The row names the variable and the fix, never a value."""
    _dotenv(tmp_path / ".env", {"CBORG_API_KEY": "sk-fresh-0002"})
    _dotenv(tmp_path / ".env.users", {"CBORG_API_KEY": "sk-stale-0001"})
    row = _run(_CONFIG, tmp_path)["users_env"]
    assert row.status is Status.ERROR
    assert "CBORG_API_KEY" in row.message
    assert "osprey users env --output .env.users" in row.message
    assert "sk-stale-0001" not in row.message and "sk-fresh-0002" not in row.message


def test_generated_file_behind_on_a_non_secret_is_a_warning(tmp_path: Path) -> None:
    from osprey.deployment.web_terminals.env_production import ensure_env_production

    _dotenv(tmp_path / ".env", {"CBORG_API_KEY": "k"})
    ensure_env_production(_CONFIG, tmp_path)
    moved = {**_CONFIG, "facility": {"timezone": "Europe/Berlin"}}
    row = _run(moved, tmp_path)["users_env"]
    assert row.status is Status.WARNING
    assert "osprey up" in row.message


def test_row_never_raises_on_an_unresolvable_config(tmp_path: Path) -> None:
    """A config the drift check cannot interpret degrades to a warning row;
    the category must still produce its other rows."""
    _dotenv(tmp_path / ".env", {"CBORG_API_KEY": "k"})
    _dotenv(tmp_path / ".env.users", {"CBORG_API_KEY": "k"})
    broken = {**_CONFIG, "modules": {"web_terminals": {"personas": "not-a-mapping"}}}
    rows = _run(broken, tmp_path)
    assert "env_file" in rows
    assert rows.get("users_env") is None or rows["users_env"].status in (
        Status.OK,
        Status.WARNING,
    )


# A built-in gateway provider that ships no endpoint. Synthetic: no shipped
# provider is in that shape, and the row has to stay right for the next one.
_GATEWAY = "gateway-without-endpoint"
_GATEWAY_SECRET_VAR = "GATEWAY_WITHOUT_ENDPOINT_API_KEY"
_GATEWAY_ENDPOINT_VAR = "GATEWAY_WITHOUT_ENDPOINT_BASE_URL"
_GATEWAY_TABLE_ENTRY = {
    "auth_env_var": "ANTHROPIC_AUTH_TOKEN",
    "auth_secret_env": _GATEWAY_SECRET_VAR,
    "base_url": None,
    "requires_base_url": True,
    "base_url_env_var": _GATEWAY_ENDPOINT_VAR,
    "default_model_tier": "haiku",
    "models": {"haiku": "h", "sonnet": "s", "opus": "o"},
}
_GATEWAY_CONFIG = {
    "facility": {"timezone": "UTC"},
    "claude_code": {"provider": _GATEWAY},
    "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
}


def test_existing_file_missing_a_required_endpoint_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment rendered before its provider required a gateway endpoint:
    .env and .env.users agree, both lack it, and every terminal restarts
    forever. Agreement with the chain is not health here -- the row names the
    variable and the remedy."""
    from osprey.build.claude_code_resolver import CLAUDE_CODE_PROVIDERS

    monkeypatch.setitem(CLAUDE_CODE_PROVIDERS, _GATEWAY, _GATEWAY_TABLE_ENTRY)
    _dotenv(tmp_path / ".env", {_GATEWAY_SECRET_VAR: "k"})
    _dotenv(tmp_path / ".env.users", {_GATEWAY_SECRET_VAR: "k"})

    row = _run(_GATEWAY_CONFIG, tmp_path)["users_env"]

    assert row.status is Status.ERROR
    assert _GATEWAY_ENDPOINT_VAR in row.message
    assert _GATEWAY in row.message


def test_existing_file_with_the_required_endpoint_is_ok(tmp_path: Path) -> None:
    both = {"ALS_APG_API_KEY": "k", "ALS_APG_BASE_URL": "https://gw.test/v1"}
    _dotenv(tmp_path / ".env", both)
    _dotenv(tmp_path / ".env.users", both)

    assert _run(_GATEWAY_CONFIG, tmp_path)["users_env"].status is Status.OK

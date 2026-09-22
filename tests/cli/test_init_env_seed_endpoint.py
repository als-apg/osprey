"""``osprey init``'s ``.env`` seed harvests every variable the provider needs.

``osprey init`` writes the repo's ``.env`` first, out of the shell, and
``osprey up`` never seeds a file that already exists. So whatever ``init``
leaves out of the harvest stays out: a gateway that ships no default endpoint
needs the endpoint beside its key, and a ``.env`` seeded with the key alone is
refused by the ``.env.users`` gate over the endpoint -- with the value that
would have settled it exported in the same shell the seed just read.

The two seeds share one rule for which variable that is
(:func:`~osprey.deployment.web_terminals.env_production.required_provider_endpoint_var`),
so they cannot resolve different spellings of it.
"""

from __future__ import annotations

import pytest

from osprey.build.claude_code_resolver import CLAUDE_CODE_PROVIDERS
from osprey.cli.profile_cmd import _exported_provider_keys

_GATEWAY = "gateway-without-endpoint"
_SECRET_VAR = "GATEWAY_WITHOUT_ENDPOINT_API_KEY"
_ENDPOINT_VAR = "GATEWAY_WITHOUT_ENDPOINT_BASE_URL"
_CATALOG = {
    _GATEWAY: {
        "api_key": f"${{{_SECRET_VAR}}}",
        "base_url": f"${{{_ENDPOINT_VAR}}}",
        "models": {"haiku": "h", "sonnet": "s", "opus": "o"},
    }
}


@pytest.fixture(autouse=True)
def _a_builtin_gateway_that_ships_no_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A built-in provider in the shape that needs an endpoint from the environment.

    Synthetic, so the test does not depend on any shipped provider being in
    that shape -- none is, and the seed has to stay correct for the next one.
    """
    monkeypatch.setitem(
        CLAUDE_CODE_PROVIDERS,
        _GATEWAY,
        {
            "auth_env_var": "ANTHROPIC_AUTH_TOKEN",
            "auth_secret_env": _SECRET_VAR,
            "base_url": None,
            "requires_base_url": True,
            "base_url_env_var": _ENDPOINT_VAR,
            "default_model_tier": "haiku",
            "models": {"haiku": "h", "sonnet": "s", "opus": "o"},
        },
    )
    for var in (_SECRET_VAR, _ENDPOINT_VAR):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own exported keys out of the harvest under test.

    The harvest reads ``os.environ`` by design, and a shipped provider's key
    exported on the machine running the tests would land in ``skipped``.
    """
    from osprey.cli.templates.scaffolding import provider_api_key_entries

    for entry in provider_api_key_entries():
        monkeypatch.delenv(entry["var"], raising=False)


def test_the_seed_writes_the_endpoint_beside_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_SECRET_VAR, "sk-from-the-shell")
    monkeypatch.setenv(_ENDPOINT_VAR, "https://gw.test/v1")

    harvest = _exported_provider_keys({_GATEWAY}, _CATALOG)

    assert harvest.seeded == {
        _SECRET_VAR: "sk-from-the-shell",
        _ENDPOINT_VAR: "https://gw.test/v1",
    }
    assert harvest.skipped == ()


def test_an_endpoint_the_shell_does_not_export_is_left_out_not_written_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SECRET_VAR, "sk-from-the-shell")

    harvest = _exported_provider_keys({_GATEWAY}, _CATALOG)

    assert harvest.seeded == {_SECRET_VAR: "sk-from-the-shell"}


def test_the_endpoint_of_a_provider_the_profile_never_names_is_not_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SECRET_VAR, "sk-from-the-shell")
    monkeypatch.setenv(_ENDPOINT_VAR, "https://gw.test/v1")

    harvest = _exported_provider_keys({"anthropic"}, _CATALOG)

    assert _ENDPOINT_VAR not in harvest.seeded
    assert harvest.seeded == {}

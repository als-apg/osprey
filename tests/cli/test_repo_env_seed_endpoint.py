"""``osprey up``'s ``.env`` seed harvests every variable the provider needs.

The seed exists so a first ``up`` on a machine whose shell already exports the
provider's credentials does not stop to ask for a file. A provider that fronts
a gateway with no default host needs two variables, not one: the secret and
the endpoint. Seeding only the secret hands the next step a chain that refuses
the deploy over the endpoint -- one prompt answered, one refusal earned.

The provider here is synthetic: no shipped provider is in that shape, and the
seed has to stay correct for the next one that is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.build.claude_code_resolver import CLAUDE_CODE_PROVIDERS
from osprey.cli import deploy_cmd
from osprey.utils.dotenv import parse_dotenv_file

_GATEWAY_PROVIDER = "gateway-without-endpoint"
_SECRET_VAR = "GATEWAY_WITHOUT_ENDPOINT_API_KEY"
_ENDPOINT_VAR = "GATEWAY_WITHOUT_ENDPOINT_BASE_URL"
_GATEWAY_CONFIG = {"claude_code": {"provider": _GATEWAY_PROVIDER}}
_SECRET = "sk-from-the-shell"
_ENDPOINT = "https://gw.test/v1"


@pytest.fixture(autouse=True)
def _a_builtin_gateway_that_ships_no_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A built-in provider that needs its endpoint from the environment."""
    monkeypatch.setitem(
        CLAUDE_CODE_PROVIDERS,
        _GATEWAY_PROVIDER,
        {
            "auth_env_var": "ANTHROPIC_AUTH_TOKEN",
            "auth_secret_env": _SECRET_VAR,
            "base_url": None,
            "requires_base_url": True,
            "base_url_env_var": _ENDPOINT_VAR,
            "default_model": "h",
            "models": ["h", "s", "o"],
        },
    )


def _shell_exports(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for var in (_SECRET_VAR, _ENDPOINT_VAR):
        monkeypatch.delenv(var, raising=False)
    for var, value in values.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: True)


def test_the_seed_writes_the_endpoint_beside_the_secret(monkeypatch, tmp_path: Path) -> None:
    _shell_exports(monkeypatch, {_SECRET_VAR: _SECRET, _ENDPOINT_VAR: _ENDPOINT})
    monkeypatch.setattr(deploy_cmd.click, "confirm", lambda *a, **k: True)
    repo = tmp_path / "repo"
    repo.mkdir()

    deploy_cmd.ensure_repo_env(repo, _GATEWAY_CONFIG)

    seeded = parse_dotenv_file(repo / ".env")
    assert seeded[_SECRET_VAR] == _SECRET
    assert seeded[_ENDPOINT_VAR] == _ENDPOINT


def test_the_offer_names_both_variables_and_no_value(monkeypatch, tmp_path: Path) -> None:
    _shell_exports(monkeypatch, {_SECRET_VAR: _SECRET, _ENDPOINT_VAR: _ENDPOINT})
    prompts: list[str] = []

    def _confirm(text, *a, **k):
        prompts.append(text)
        return False

    monkeypatch.setattr(deploy_cmd.click, "confirm", _confirm)
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(deploy_cmd.click.Abort):
        deploy_cmd.ensure_repo_env(repo, _GATEWAY_CONFIG)

    assert prompts, "the seed was never offered"
    assert _SECRET_VAR in prompts[0] and _ENDPOINT_VAR in prompts[0]
    assert _SECRET not in prompts[0] and _ENDPOINT not in prompts[0]


def test_a_secret_alone_still_seeds_what_the_shell_has(monkeypatch, tmp_path: Path) -> None:
    """Only the secret is exported: the seed runs for what it has. The endpoint
    is then the users-env gate's refusal to name, not this seed's to invent."""
    _shell_exports(monkeypatch, {_SECRET_VAR: _SECRET})
    monkeypatch.setattr(deploy_cmd.click, "confirm", lambda *a, **k: True)
    repo = tmp_path / "repo"
    repo.mkdir()

    deploy_cmd.ensure_repo_env(repo, _GATEWAY_CONFIG)

    seeded = parse_dotenv_file(repo / ".env")
    assert seeded[_SECRET_VAR] == _SECRET
    assert _ENDPOINT_VAR not in seeded

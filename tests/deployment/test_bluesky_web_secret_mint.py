"""Unit tests for the bluesky-web sidecar's operator secret in the token-mint map.

Mirrors ``test_bluesky_token_mint.py``, but for the ``bluesky_web``
deployed-service entry in ``_SERVICE_TOKEN_VARS`` (container_lifecycle.py).
The sidecar is gated by WebAuthMiddleware and its container declares
``OSPREY_TERMINAL_BIND_HOST``, so web_auth refuses startup on an empty secret
rather than minting one only the container log would ever see — this mint is
what makes the deployed panel reachable at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'bluesky_web' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["bluesky_web"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(container_lifecycle, "run_captured", _fake_run)
    return captured


@pytest.fixture
def _clean_secret_env(monkeypatch):
    monkeypatch.delenv("OSPREY_TERMINAL_SECRET", raising=False)
    monkeypatch.delenv("ARIEL_DSN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    path = tmp_path / ".env"
    return parse_dotenv_file(path) if path.is_file() else {}


def test_bluesky_web_deploy_mints_the_operator_secret(captured_argv, _clean_secret_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    secret = env.get("OSPREY_TERMINAL_SECRET")
    assert secret, "deploy_up did not mint OSPREY_TERMINAL_SECRET for a bluesky_web deploy"
    assert len(secret) >= 32, f"minted secret is implausibly short: {len(secret)} chars"


def test_bluesky_web_existing_secret_is_preserved(captured_argv, _clean_secret_env, tmp_path):
    (tmp_path / ".env").write_text("OSPREY_TERMINAL_SECRET=operator-chose-this\n", encoding="utf-8")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("OSPREY_TERMINAL_SECRET") == "operator-chose-this", (
        "an operator-supplied secret must never be overwritten by the mint"
    )


def test_service_token_vars_map_includes_bluesky_web():
    assert container_lifecycle._SERVICE_TOKEN_VARS.get("bluesky_web") == ("OSPREY_TERMINAL_SECRET",)


def test_sidecar_compose_template_matches_the_mint():
    """The template names exactly the minted var, fail-closed, container-marked.

    Pins the template half of the contract the mint map is the other half of:
    ``${OSPREY_TERMINAL_SECRET}`` must carry NO ``:-`` default (an unset secret
    must stay empty so web_auth refuses startup instead of booting on a
    guessable value), and ``OSPREY_TERMINAL_BIND_HOST`` must be declared so
    that refusal actually fires in the container shape.
    """
    template = (
        Path(container_lifecycle.__file__).parent.parent
        / "templates"
        / "services"
        / "bluesky_web"
        / "docker-compose.yml.j2"
    ).read_text(encoding="utf-8")

    assert "OSPREY_TERMINAL_SECRET: ${OSPREY_TERMINAL_SECRET}" in template, (
        "the sidecar compose template must pass the minted secret through verbatim, "
        "with no :- default"
    )
    assert "${OSPREY_TERMINAL_SECRET:-" not in template, (
        "the secret must not grow a :- default -- empty must mean refuse, not a shared known value"
    )
    assert "OSPREY_TERMINAL_BIND_HOST" in template, (
        "the container must declare the bind-host marker so web_auth refuses an "
        "empty secret at startup instead of minting an unreachable one"
    )

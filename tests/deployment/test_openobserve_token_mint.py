"""Unit tests for the OpenObserve root-password entry in the token-mint map.

control_assistant ships telemetry LIVE against a co-deployed OpenObserve store,
so ``osprey deploy up`` must self-provision a ``ZO_ROOT_USER_PASSWORD`` into the
project ``.env`` (``_SERVICE_TOKEN_VARS["openobserve"]``). Two properties make
this entry different from every other minted token and are pinned here:

* OpenObserve refuses to start unless the root password has all four character
  classes (lower/upper/digit/special) — so the mint uses a dedicated recipe
  (``_generate_openobserve_password``), not the ``token_urlsafe`` default whose
  ``[A-Za-z0-9_-]`` alphabet would crash-loop the container non-deterministically
  (the same failure shape as ``BLUESKY_TILED_API_KEY``/Tiled).
* Its compose template carries an insecure ``:-Complexpass#123`` default, so the
  motivation is not fail-closed arming (as with dispatch/bluesky) but replacing a
  shared, transcript-guarding default with a per-deploy secret.
"""

from __future__ import annotations

import base64

import pytest

from osprey.deployment import container_lifecycle

_MINT_SAMPLES = 200


def _parse_dotenv(path):
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(path) if path.is_file() else {}


def _satisfies_openobserve_policy(pw: str) -> bool:
    return (
        8 <= len(pw) <= 128
        and any(c.islower() for c in pw)
        and any(c.isupper() for c in pw)
        and any(c.isdigit() for c in pw)
        and any(not c.isalnum() for c in pw)
    )


# ---------------------------------------------------------------------------
# Generator — must satisfy OpenObserve's policy on EVERY mint. A single sample
# proves nothing: a broken recipe could still pass by luck. Assert over many.
# ---------------------------------------------------------------------------


def test_openobserve_password_generator_always_satisfies_policy():
    pws = [container_lifecycle._generate_openobserve_password() for _ in range(_MINT_SAMPLES)]

    offenders = [p for p in pws if not _satisfies_openobserve_policy(p)]
    assert not offenders, (
        f"{len(offenders)}/{_MINT_SAMPLES} fail OpenObserve policy: {offenders[:3]}"
    )
    # Its own validator must agree with the policy check on every mint.
    assert all(container_lifecycle._validate_openobserve_password(p) for p in pws)
    # Strong + random: >=256 bits of entropy means unique values across the sample.
    assert len(set(pws)) == _MINT_SAMPLES, "generator is not random"


def test_openobserve_password_uses_only_dotenv_safe_characters():
    """A minted password must survive ``.env`` write/parse verbatim.

    ``# $ " ' `` backslash ``= space`` and control chars would break dotenv
    parsing or shell reuse; the special alphabet deliberately excludes them.
    """
    unsafe = set("#$\"'`\\= \t\n\r")
    for _ in range(_MINT_SAMPLES):
        pw = container_lifecycle._generate_openobserve_password()
        assert not (set(pw) & unsafe), f"minted a dotenv-hostile password: {pw!r}"


def test_openobserve_password_routes_through_its_own_generator():
    """``_generate_token`` consults the registry rather than the default recipe."""
    a = container_lifecycle._generate_token("ZO_ROOT_USER_PASSWORD")
    b = container_lifecycle._generate_token("ZO_ROOT_USER_PASSWORD")
    assert _satisfies_openobserve_policy(a)
    assert a != b  # random, not a constant


# ---------------------------------------------------------------------------
# Mint path (_ensure_service_tokens) — the value that reaches .env is what the
# OpenObserve container parses, so assert the property end-to-end, and prove the
# minted value round-trips through parse_dotenv_file intact.
# ---------------------------------------------------------------------------


def test_deploying_openobserve_mints_a_policy_valid_password_every_time(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}

    for i in range(50):
        env_path = tmp_path / f"{i}.env"
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
        pw = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
        assert _satisfies_openobserve_policy(pw), f"mint {i} would crash-loop OpenObserve: {pw!r}"


def test_openobserve_mint_does_not_mint_the_email(tmp_path, monkeypatch):
    """Only the password is a secret; the email is a username with a non-secret
    default, so it is never fabricated into .env."""
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    monkeypatch.delenv("ZO_ROOT_USER_EMAIL", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    env = _parse_dotenv(env_path)
    assert env.get("ZO_ROOT_USER_PASSWORD")
    assert "ZO_ROOT_USER_EMAIL" not in env


def test_openobserve_mint_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    first = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]
    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
    second = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]

    assert first == second
    assert env_path.read_text().count("ZO_ROOT_USER_PASSWORD=") == 1


def test_minted_password_yields_a_valid_openobserve_auth_header(tmp_path, monkeypatch):
    """Cross-check the two halves: a minted password flows through the agent's
    telemetry resolver into a clean base64 Basic-auth header (no ${VAR} leak)."""
    from osprey.cli.claude_code_telemetry import _openobserve_auth_header

    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    env_path = tmp_path / ".env"
    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["openobserve"]}, expose_network=False, env_path=env_path
    )
    pw = _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"]

    key, value = _openobserve_auth_header(
        {"openobserve": {"user": "root@example.com", "password": pw}}
    )
    assert key == "Authorization"
    decoded = base64.b64decode(value.removeprefix("Basic ")).decode()
    assert decoded == f"root@example.com:{pw}"


# ---------------------------------------------------------------------------
# Deploy-boundary validation of an OPERATOR-SUPPLIED password (never minted).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weak",
    [
        "alllowercase1!",  # no uppercase
        "ALLUPPERCASE1!",  # no lowercase
        "NoSpecialChars123",  # no special
        "NoDigits!ABCdef",  # no digit
        "aB3!",  # too short (<8)
    ],
)
def test_ensure_service_tokens_rejects_weak_operator_password_from_dotenv(
    tmp_path, monkeypatch, weak
):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    assert not _satisfies_openobserve_policy(weak), "sample must be genuinely weak"
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"
    env_path.write_text(f"ZO_ROOT_USER_PASSWORD={weak}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    # The offending value is never echoed in the error.
    assert weak not in str(exc_info.value)


def test_ensure_service_tokens_rejects_weak_password_from_process_env(tmp_path, monkeypatch):
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"
    monkeypatch.setenv("ZO_ROOT_USER_PASSWORD", "nospecialchars123AB")

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "nospecialchars123AB" not in str(exc_info.value)


def test_ensure_service_tokens_accepts_a_freshly_minted_password(tmp_path, monkeypatch):
    """Happy path: an unset password is minted, then validated against the same
    policy — the mint always satisfies its own validator."""
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert _satisfies_openobserve_policy(_parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"])


def test_ensure_service_tokens_accepts_a_strong_operator_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ZO_ROOT_USER_PASSWORD", raising=False)
    config = {"deployed_services": ["openobserve"]}
    env_path = tmp_path / ".env"
    strong = "MyStr0ng#Facility@Pass"
    env_path.write_text(f"ZO_ROOT_USER_PASSWORD={strong}\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert _parse_dotenv(env_path)["ZO_ROOT_USER_PASSWORD"] == strong  # untouched


def test_service_token_vars_map_includes_openobserve():
    """Lock in the map shape: only the password, not the email."""
    assert container_lifecycle._SERVICE_TOKEN_VARS["openobserve"] == ("ZO_ROOT_USER_PASSWORD",)

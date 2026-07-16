"""Unit tests for the bluesky-bridge auth token in the generalized token-mint map.

Mirrors ``test_container_lifecycle.py``'s dispatch-token coverage, but for the
``bluesky`` deployed-service entry added to ``_SERVICE_TOKEN_VARS``
(container_lifecycle.py). Without this mint, the fail-closed bridge
(``osprey.services.bluesky_bridge.security.require_armed``) 503s on every
promote attempt after a fresh deploy.
"""

from __future__ import annotations

import secrets

import pytest

from osprey.deployment import container_lifecycle

# Every var any service declares, so the scoping tests below cannot silently
# stop covering a var that gets added later.
_NON_TILED_VARS = ("BLUESKY_PROMOTE_TOKEN", "EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN")


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'bluesky' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["bluesky"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


@pytest.fixture
def _clean_token_env(monkeypatch):
    """Ensure the bluesky tokens (and dispatch tokens) are unset in the process env."""
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    monkeypatch.delenv("BLUESKY_TILED_API_KEY", raising=False)
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    # ARIEL_DSN is validate-only (see _VALIDATE_ONLY_VARS below), but process
    # env still wins over .env in that check -- a stray exported ARIEL_DSN
    # would otherwise break the clean/absent tests.
    monkeypatch.delenv("ARIEL_DSN", raising=False)


def _parse_dotenv(path):
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(path) if path.is_file() else {}


def _parse_env(tmp_path):
    return _parse_dotenv(tmp_path / ".env")


def test_bluesky_deploy_generates_promote_token(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    # token_urlsafe(32) → ~43 url-safe chars
    assert len(env["BLUESKY_PROMOTE_TOKEN"]) >= 40
    # A deploy with only 'bluesky' deployed must not mint unrelated dispatch tokens.
    assert "EVENT_DISPATCHER_TOKEN" not in env
    assert "DISPATCH_WORKER_TOKEN" not in env


def test_bluesky_deploy_generates_tiled_api_key(captured_argv, _clean_token_env, tmp_path):
    """The Tiled catalog key hangs off the deployed 'bluesky' service.

    Keying it off a 'tiled' service would never mint: 'tiled' is never in
    deployed_services, so the membership guard skips it before any minting.
    """
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_TILED_API_KEY")
    assert len(env["BLUESKY_TILED_API_KEY"]) >= 40
    assert env["BLUESKY_TILED_API_KEY"] != env["BLUESKY_PROMOTE_TOKEN"]


def test_bluesky_token_generation_is_idempotent(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["BLUESKY_PROMOTE_TOKEN"] == second["BLUESKY_PROMOTE_TOKEN"]
    assert first["BLUESKY_TILED_API_KEY"] == second["BLUESKY_TILED_API_KEY"]
    # No duplicate keys appended on the second run.
    text = (tmp_path / ".env").read_text()
    assert text.count("BLUESKY_PROMOTE_TOKEN=") == 1
    assert text.count("BLUESKY_TILED_API_KEY=") == 1


def test_bluesky_existing_env_token_is_preserved(captured_argv, _clean_token_env, tmp_path):
    (tmp_path / ".env").write_text("BLUESKY_PROMOTE_TOKEN=my-real-token\n", encoding="utf-8")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_PROMOTE_TOKEN"] == "my-real-token"  # untouched


def test_bluesky_process_env_token_not_written_to_dotenv(captured_argv, monkeypatch, tmp_path):
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "from-shell")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    # A token resolvable from the process env is not duplicated into .env.
    assert "BLUESKY_PROMOTE_TOKEN" not in env


def test_bluesky_expose_refuses_empty_token(captured_argv, monkeypatch, tmp_path):
    # A token explicitly set empty must not be auto-overwritten, and --expose must
    # refuse rather than bind a fail-open server to 0.0.0.0.
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "")

    with pytest.raises(RuntimeError, match="refusing to --expose"):
        container_lifecycle.deploy_up(
            str(tmp_path / "config.yml"), detached=True, expose_network=True
        )


def test_bluesky_alongside_dispatch_mints_both_independently(
    monkeypatch, _clean_token_env, tmp_path
):
    """Per-service-instance behavior: deploying bluesky AND dispatch mints all
    three vars (no cross-service leakage or accidental sharing)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"deployed_services": ["event_dispatcher", "dispatch_worker", "bluesky"]},
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("EVENT_DISPATCHER_TOKEN")
    assert env.get("DISPATCH_WORKER_TOKEN")
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    # All three distinct values — no accidental sharing across services.
    assert (
        len(
            {
                env["EVENT_DISPATCHER_TOKEN"],
                env["DISPATCH_WORKER_TOKEN"],
                env["BLUESKY_PROMOTE_TOKEN"],
            }
        )
        == 3
    )


def test_service_token_vars_map_includes_bluesky():
    """Locks in the generalized map shape."""
    assert container_lifecycle._SERVICE_TOKEN_VARS["bluesky"] == (
        "BLUESKY_PROMOTE_TOKEN",
        "BLUESKY_TILED_API_KEY",
    )
    assert "event_dispatcher" in container_lifecycle._SERVICE_TOKEN_VARS
    assert "dispatch_worker" in container_lifecycle._SERVICE_TOKEN_VARS


# ---------------------------------------------------------------------------
# Per-variable secret alphabets.
#
# Tiled validates its --api-key at server startup and exits with
# ValueError("The API key must only contain alphanumeric characters") on
# anything else, so a key drawn from token_urlsafe's alphabet (which includes
# '-' and '_') crash-loops the container. Roughly 71% of token_urlsafe(32)
# values contain at least one such character, which makes the failure a
# non-deterministic property of the deploy rather than of the code.
# ---------------------------------------------------------------------------

_MINT_SAMPLES = 200


def test_tiled_api_key_generator_is_always_alphanumeric():
    """Assert over many mints: a single sample proves nothing here.

    A token_urlsafe(32) value happens to be all-alphanumeric about 29% of the
    time, so a one-shot ``.isalnum()`` assertion would pass by luck on roughly
    three runs in ten even against the unfixed generator — a flaky test, not a
    guard. Over ``_MINT_SAMPLES`` draws the regression's survival probability is
    0.29**200, i.e. zero for any practical purpose.
    """
    keys = [
        container_lifecycle._generate_token("BLUESKY_TILED_API_KEY") for _ in range(_MINT_SAMPLES)
    ]

    assert all(k for k in keys), "minted an empty API key"
    offenders = [k for k in keys if not k.isalnum()]
    assert not offenders, (
        f"{len(offenders)}/{_MINT_SAMPLES} keys are not alphanumeric: {offenders[:3]}"
    )
    # token_hex(32): 64 hex chars, 256 bits — no weaker than token_urlsafe(32).
    assert {len(k) for k in keys} == {64}
    assert len(set(keys)) == _MINT_SAMPLES, "generator is not random"


def test_ensure_service_tokens_writes_an_alphanumeric_tiled_key_every_time(
    tmp_path, _clean_token_env
):
    """The alphanumeric alphabet survives the real mint path, not just the generator.

    Same reasoning as above on the sample count: the value that reaches ``.env``
    is what Tiled parses, so the property is asserted end-to-end over many
    independent mints rather than once.
    """
    config = {"deployed_services": ["bluesky"]}

    for i in range(50):
        env_path = tmp_path / f"{i}.env"
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
        key = _parse_dotenv(env_path)["BLUESKY_TILED_API_KEY"]
        assert key.isalnum(), f"mint {i} produced a Tiled-rejecting key: {key!r}"


def test_var_generators_registry_is_pinned():
    """Pin the blast radius: only vars with a downstream alphabet/policy
    constraint override the default token recipe (Tiled's alphanumeric
    --api-key, OpenObserve's four-class root password)."""
    assert set(container_lifecycle._VAR_GENERATORS) == {
        "BLUESKY_TILED_API_KEY",
        "ZO_ROOT_USER_PASSWORD",
    }
    declared = {
        var for token_vars in container_lifecycle._SERVICE_TOKEN_VARS.values() for var in token_vars
    }
    # Every overridden var is one some service actually declares — no dead entries.
    assert set(container_lifecycle._VAR_GENERATORS) <= declared


@pytest.mark.parametrize("var", _NON_TILED_VARS)
def test_non_tiled_vars_still_use_token_urlsafe(monkeypatch, var):
    """Deterministic proof of routing: patch both recipes and see which is called."""
    monkeypatch.setattr(secrets, "token_urlsafe", lambda n: f"urlsafe-{n}")
    monkeypatch.setattr(secrets, "token_hex", lambda n: f"hex{n}")

    assert container_lifecycle._generate_token(var) == "urlsafe-32"
    assert container_lifecycle._generate_token("BLUESKY_TILED_API_KEY") == "hex32"


@pytest.mark.parametrize("var", _NON_TILED_VARS)
def test_non_tiled_vars_are_not_forced_alphanumeric(var):
    """The other tokens keep the url-safe alphabet; they were never the problem.

    Behavioral counterpart to the routing test above: a '-' or '_' must still
    appear across many mints. (P(false failure) = 0.29**200.)
    """
    minted = [container_lifecycle._generate_token(var) for _ in range(_MINT_SAMPLES)]

    assert any(not t.isalnum() for t in minted), (
        f"{var} looks forced-alphanumeric — the fix should have been scoped to "
        "BLUESKY_TILED_API_KEY alone"
    )
    assert all(len(t) >= 40 for t in minted)


def test_deploy_up_routes_each_var_through_its_own_generator(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """The mint site consults the registry — it does not hardcode one recipe."""
    monkeypatch.setattr(secrets, "token_urlsafe", lambda n: "sentinel-urlsafe_value")
    monkeypatch.setattr(secrets, "token_hex", lambda n: "sentinelhexvalue")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_TILED_API_KEY"] == "sentinelhexvalue"
    assert env["BLUESKY_PROMOTE_TOKEN"] == "sentinel-urlsafe_value"


# ---------------------------------------------------------------------------
# _VAR_VALIDATORS — deploy-boundary validation of the effective value (F1/F3)
#
# _ensure_service_tokens(config, expose_network=False, env_path=...) is the
# DEFAULT loopback deploy path (deploy_up's default is --expose off). These
# tests call it directly, mirroring test_ensure_service_tokens_writes_an_
# alphanumeric_tiled_key_every_time above, to prove the boundary check fires
# on that path and not only under --expose.
# ---------------------------------------------------------------------------


def test_ensure_service_tokens_rejects_non_alphanumeric_tiled_key_from_dotenv(
    tmp_path, _clean_token_env
):
    """An operator-supplied .env value that fails BLUESKY_TILED_API_KEY's
    validator is rejected on the default (non-expose) deploy."""
    config = {"deployed_services": ["bluesky"]}
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BLUESKY_PROMOTE_TOKEN=some-real-looking-token\nBLUESKY_TILED_API_KEY=has-a-dash-in-it\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="BLUESKY_TILED_API_KEY") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "has-a-dash-in-it" not in str(exc_info.value)


def test_ensure_service_tokens_rejects_non_alphanumeric_tiled_key_from_process_env(
    tmp_path, monkeypatch, _clean_token_env
):
    """Same, but the malformed value comes from a shell/process-env override."""
    config = {"deployed_services": ["bluesky"]}
    env_path = tmp_path / ".env"
    monkeypatch.setenv("BLUESKY_TILED_API_KEY", "not-alphanumeric!")

    with pytest.raises(RuntimeError, match="BLUESKY_TILED_API_KEY") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "not-alphanumeric!" not in str(exc_info.value)


def test_ensure_service_tokens_passes_for_a_freshly_minted_tiled_key(tmp_path, _clean_token_env):
    """The happy path: an unset key is minted by _generate_token and the mint
    always satisfies the validator it is then checked against."""
    config = {"deployed_services": ["bluesky"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    env = _parse_dotenv(env_path)
    assert env["BLUESKY_TILED_API_KEY"].isalnum()


def test_ariel_dsn_validator_rejects_unescaped_reserved_char_in_password():
    """@ : / ? # left unescaped inside a DSN password corrupts URI parsing (F3)."""
    assert not container_lifecycle._validate_var(
        "ARIEL_DSN", "postgresql://ariel:p@ss@ariel-postgres:5432/ariel"
    )


def test_ariel_dsn_validator_accepts_a_clean_dsn():
    assert container_lifecycle._validate_var(
        "ARIEL_DSN", "postgresql://ariel:ariel@ariel-postgres:5432/ariel"
    )


# ---------------------------------------------------------------------------
# _VALIDATE_ONLY_VARS (ARIEL_DSN) — checked at the boundary when present, but
# never minted and never a _SERVICE_TOKEN_VARS member. ARIEL_DSN has no
# osprey-native service consumer in this deploy system: it belongs to the
# separate osprey-build-deploy facility-scaffolding pipeline (its own
# generated docker-compose.yml/.env.template, brought up by the facility's
# own scripts/deploy.sh via a raw `docker compose`/`podman compose` call,
# never through `osprey deploy up`). These tests call the real
# _ensure_service_tokens with NO _SERVICE_TOKEN_VARS monkeypatch — proving
# the check fires unconditionally, independent of deployed_services/service
# membership, as defense-in-depth for the case where an operator or other
# tooling places ARIEL_DSN into *this* project's effective env anyway.
# ---------------------------------------------------------------------------


def test_ensure_service_tokens_rejects_reserved_char_in_ariel_dsn(tmp_path):
    """A malformed ARIEL_DSN in .env is rejected even with no deployed
    service requiring it — the validate-only check does not depend on
    _SERVICE_TOKEN_VARS membership."""
    config = {"deployed_services": []}
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ARIEL_DSN=postgresql://ariel:p@ss@ariel-postgres:5432/ariel\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="ARIEL_DSN") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "p@ss" not in str(exc_info.value)


def test_ensure_service_tokens_rejects_reserved_char_in_ariel_dsn_alongside_a_real_service(
    captured_argv, _clean_token_env, tmp_path
):
    """Same check, but on a deploy that also mints an unrelated service token —
    proves the two loops (required_vars and _VALIDATE_ONLY_VARS) coexist."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ARIEL_DSN=postgresql://ariel:p@ss@ariel-postgres:5432/ariel\n", encoding="utf-8"
    )
    config = {"deployed_services": ["bluesky"]}

    with pytest.raises(RuntimeError, match="ARIEL_DSN") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "p@ss" not in str(exc_info.value)


def test_ensure_service_tokens_passes_a_clean_ariel_dsn(tmp_path):
    config = {"deployed_services": []}
    env_path = tmp_path / ".env"
    dsn = "postgresql://ariel:ariel@ariel-postgres:5432/ariel"
    env_path.write_text(f"ARIEL_DSN={dsn}\n", encoding="utf-8")

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    # No RuntimeError raised, and the clean DSN was left untouched.
    assert _parse_dotenv(env_path)["ARIEL_DSN"] == dsn


def test_ensure_service_tokens_never_mints_or_fabricates_ariel_dsn(tmp_path, _clean_token_env):
    """Absent ARIEL_DSN is skipped, not fabricated — the "validate-only,
    never mint" half of the mechanism. Deploying bluesky (which does mint
    its own tokens) alongside proves ARIEL_DSN's absence doesn't get treated
    like an unset _SERVICE_TOKEN_VARS entry."""
    config = {"deployed_services": ["bluesky"]}
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    env = _parse_dotenv(env_path)
    assert "ARIEL_DSN" not in env
    # The real service tokens still mint normally alongside the no-op.
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    assert env.get("BLUESKY_TILED_API_KEY")


def test_ensure_service_tokens_rejects_reserved_char_in_ariel_dsn_from_process_env(
    tmp_path, monkeypatch
):
    """Same rejection, but the malformed value comes from a shell/process-env
    override rather than .env — mirrors the BLUESKY_TILED_API_KEY process-env
    coverage above."""
    config = {"deployed_services": []}
    env_path = tmp_path / ".env"
    monkeypatch.setenv("ARIEL_DSN", "postgresql://ariel:p@ss@ariel-postgres:5432/ariel")

    with pytest.raises(RuntimeError, match="ARIEL_DSN") as exc_info:
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)

    assert "p@ss" not in str(exc_info.value)


def test_validate_only_vars_set_is_pinned():
    """Blast-radius pin for _VALIDATE_ONLY_VARS, mirroring the _VAR_GENERATORS
    and _VAR_VALIDATORS pins."""
    assert container_lifecycle._VALIDATE_ONLY_VARS == {"ARIEL_DSN"}


def test_validator_registry_keyset_is_pinned():
    """Blast-radius pin for _VAR_VALIDATORS, mirroring the _VAR_GENERATORS pin
    (test_only_the_tiled_key_overrides_the_default_generator above). Closes
    "someone silently widens the registry."
    """
    assert set(container_lifecycle._VAR_VALIDATORS) == {
        "BLUESKY_TILED_API_KEY",
        "ARIEL_DSN",
        "ZO_ROOT_USER_PASSWORD",
    }

    declared = {
        var for token_vars in container_lifecycle._SERVICE_TOKEN_VARS.values() for var in token_vars
    }
    # Every registered validator must be EITHER a declared service-token var
    # (BLUESKY_TILED_API_KEY, minted by a real deployed service) OR a member
    # of _VALIDATE_ONLY_VARS (ARIEL_DSN, which has no osprey-native service
    # consumer — see _VALIDATE_ONLY_VARS' docstring). No entry may be in
    # neither, the same "no dead entries" invariant _VAR_GENERATORS pins.
    live_validators = set(container_lifecycle._VAR_VALIDATORS)
    assert live_validators <= declared | container_lifecycle._VALIDATE_ONLY_VARS
    # And the two sets are actually disjoint from each other today — ARIEL_DSN
    # reaches _ensure_service_tokens only through the validate-only path.
    assert container_lifecycle._VALIDATE_ONLY_VARS.isdisjoint(declared)

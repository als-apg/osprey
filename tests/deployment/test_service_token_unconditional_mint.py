"""Service-token minting is unconditional — it reads no execution/writes posture.

``_SERVICE_TOKEN_VARS`` entries are *network-caller authentication*: a token
proves the caller of a service's own HTTP boundary is the co-deployed agent
rather than anything else that can reach a shared loopback port. That is an
authentication concern, not a hardware-safety one. Whether a scan or a write is
permitted is decided at the connector — ``writes_enabled`` plus the per-put
channel limits — which every write path must clear: the agent's own
``write_channel`` calls and the bluesky bridge's plan execution alike end at the
same gate.

An earlier design withheld ``BLUESKY_LAUNCH_TOKEN`` whenever
``control_system.writes_enabled`` and ``execution.execution_method: local`` were
both set, on the theory that agent-authored Python could read the token out of
``.env`` and POST ``/runs/{id}/launch`` directly. Withholding it never closed
that path — it only made the bridge 503 while leaving the *shorter* route (the
agent writing channels through the connector) exactly as open, gated by the
connector as designed. So the withholding bought no safety and cost a deploy
that silently came up half-armed. It is gone, along with its allowlist:
``_ensure_service_tokens`` now mints every var a deployed service declares,
every time, and no code path consults ``execution_method`` for safety semantics.

These tests pin that. The property under test is *absence of coupling*: the
minted key set must be identical no matter what the execution and control-system
sections say — including when they are absent entirely.
"""

from __future__ import annotations

import logging

import pytest

from osprey.deployment import container_lifecycle

# The legacy spelling ("local") and the honest one ("subprocess") name the same
# backend (see ``osprey.utils.config.resolve_execution_method``). "local" is the
# value the deleted guard keyed on, so it is the one a regression would most
# plausibly reintroduce — cover both wherever the posture is varied.
_SUBPROCESS_METHODS = ("local", "subprocess")


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'bluesky' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


def _patch_prepare_compose_files(monkeypatch, config: dict) -> None:
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )


@pytest.fixture
def _clean_token_env(monkeypatch):
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    monkeypatch.delenv("BLUESKY_TILED_API_KEY", raising=False)
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    # ARIEL_DSN is validate-only, but the process env still wins over .env in
    # that check — a stray exported value would fail these deploys.
    monkeypatch.delenv("ARIEL_DSN", raising=False)


def _parse_dotenv(path):
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(path) if path.is_file() else {}


def _parse_env(tmp_path):
    return _parse_dotenv(tmp_path / ".env")


def _config(**overrides) -> dict:
    base: dict = {"deployed_services": ["bluesky"]}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The posture does not change what is minted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_method", _SUBPROCESS_METHODS)
def test_both_bluesky_tokens_mint_when_writes_enabled_and_subprocess_execution(
    captured_argv, _clean_token_env, monkeypatch, tmp_path, execution_method
):
    """The exact configuration the deleted guard withheld under. Both vars mint.

    A deploy that came up with an unarmed bridge was not a safer deploy — the
    connector gate is what stands between the agent and the hardware, and it is
    unaffected by whether this token exists.
    """
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": execution_method},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_LAUNCH_TOKEN")
    assert len(env["BLUESKY_LAUNCH_TOKEN"]) >= 40
    assert env.get("BLUESKY_TILED_API_KEY")
    assert env["BLUESKY_TILED_API_KEY"] != env["BLUESKY_LAUNCH_TOKEN"]


def test_both_bluesky_tokens_mint_under_read_only_posture(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """``writes_enabled: False`` is a connector-level posture, not a mint input.

    The bridge still needs to authenticate its callers for read-only browsing
    (plan listing, catalog access), so a read-only deploy is armed for
    authentication exactly like any other.
    """
    config = _config(
        control_system={"writes_enabled": False},
        execution={"execution_method": "subprocess"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_LAUNCH_TOKEN")
    assert env.get("BLUESKY_TILED_API_KEY")


def test_both_bluesky_tokens_mint_with_no_execution_or_control_system_block(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """A config carrying neither section mints the same set as one carrying both.

    The old guard read defaults out of the missing sections; nothing reads them
    now, so their absence must be uneventful rather than a third behavior.
    """
    config = _config()
    assert "execution" not in config and "control_system" not in config
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_LAUNCH_TOKEN")
    assert env.get("BLUESKY_TILED_API_KEY")


def test_minted_key_set_is_identical_across_every_posture(_clean_token_env, tmp_path):
    """The property itself: posture is not an input to minting.

    The per-posture tests above each pin one case and would all still pass if a
    future edit withheld some *other* var under some *other* combination. This
    one asserts the whole minted key set is invariant across the cross product
    of write postures and execution methods — including the deprecated
    ``container`` spelling and both sections omitted — so any newly introduced
    coupling shows up here no matter which var it touches.
    """
    postures: list[dict] = [{}]
    for writes_enabled in (True, False):
        for method in (*_SUBPROCESS_METHODS, "container"):
            postures.append(
                {
                    "control_system": {"writes_enabled": writes_enabled},
                    "execution": {"execution_method": method},
                }
            )

    minted: list[tuple[str, ...]] = []
    for i, posture in enumerate(postures):
        env_path = tmp_path / f"{i}.env"
        container_lifecycle._ensure_service_tokens(
            _config(**posture), expose_network=False, env_path=env_path
        )
        minted.append(tuple(sorted(_parse_dotenv(env_path))))

    assert set(minted) == {("BLUESKY_LAUNCH_TOKEN", "BLUESKY_TILED_API_KEY")}, (
        f"minting varies with posture: {list(zip(postures, minted, strict=True))}"
    )


def test_mint_logs_no_withholding_warning_under_any_posture(
    captured_argv, _clean_token_env, monkeypatch, tmp_path, caplog
):
    """The operator-facing half must be silent too.

    A per-var warning that a token had been withheld would tell operators
    something was held back when a deploy mints everything — text that sends
    them looking for a safety property that is not there.
    """
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    with caplog.at_level(logging.WARNING, logger="deployment.lifecycle"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    # caplog's handler already calls record.getMessage(), so r.message is the
    # final rendered string. Scoped to the logger this test opened, because the
    # claim is about what the MINT says: an unscoped sweep also reads warnings
    # that echo a filesystem path, and pytest names ``tmp_path`` after the test
    # — so this test's own name would match its own assertion.
    warnings = " ".join(
        r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "deployment.lifecycle"
    ).lower()
    assert "withheld" not in warnings
    assert "withholding" not in warnings
    assert "bluesky_launch_token" not in warnings


# ---------------------------------------------------------------------------
# Scope: every declared var, for every deployed service.
# ---------------------------------------------------------------------------


def test_dispatch_tokens_mint_alongside_bluesky_under_writes_enabled_local(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """A multi-service deploy mints the full union of its services' declared vars."""
    config = _config(
        deployed_services=["bluesky", "event_dispatcher", "dispatch_worker"],
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_LAUNCH_TOKEN")
    assert env.get("BLUESKY_TILED_API_KEY")
    assert env.get("EVENT_DISPATCHER_TOKEN")
    assert env.get("DISPATCH_WORKER_TOKEN")


def test_a_new_services_token_mints_without_being_triaged_first(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """Declaring a var is sufficient — there is no allowlist to be added to.

    Under the old fail-closed allowlist an untriaged var was withheld by
    omission, so a service added without an accompanying allowlist edit
    deployed unarmed. Declaring the var in ``_SERVICE_TOKEN_VARS`` is the
    single step now.
    """
    monkeypatch.setattr(
        container_lifecycle,
        "_SERVICE_TOKEN_VARS",
        {**container_lifecycle._SERVICE_TOKEN_VARS, "scan_engine": ("SCAN_ENGINE_ARM_TOKEN",)},
    )
    monkeypatch.delenv("SCAN_ENGINE_ARM_TOKEN", raising=False)
    config = _config(
        deployed_services=["bluesky", "scan_engine"],
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("SCAN_ENGINE_ARM_TOKEN")
    assert env.get("BLUESKY_LAUNCH_TOKEN")
    assert env.get("BLUESKY_TILED_API_KEY")


def test_undeployed_service_vars_are_still_not_minted(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """ "Unconditional" is scoped to *deployed* services, not to the whole map.

    Without this, a mint path that ignored ``deployed_services`` entirely would
    satisfy every other test in this file.
    """
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert "EVENT_DISPATCHER_TOKEN" not in env
    assert "DISPATCH_WORKER_TOKEN" not in env
    assert "ZO_ROOT_USER_PASSWORD" not in env
    assert "ARIEL_DB_PASSWORD" not in env


def test_bluesky_declared_vars_are_pinned():
    """``bluesky`` declares exactly the launch token and the Tiled key.

    The authoring MCP tools (``write_plan``, ``validate_plan``) reach no
    hardware and gate on ``_APPROVAL`` in the registry (see
    ``tests/registry/test_bluesky_server_definition.py``), so they need no
    token of their own. Pinning the pair keeps this file's "every declared var
    mints" claim covering the service's whole token surface.
    """
    assert container_lifecycle._SERVICE_TOKEN_VARS["bluesky"] == (
        "BLUESKY_LAUNCH_TOKEN",
        "BLUESKY_TILED_API_KEY",
    )


# ---------------------------------------------------------------------------
# Operator-supplied values still win — minting is additive, never corrective.
# ---------------------------------------------------------------------------


def test_existing_env_value_is_never_overwritten(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """A user-set token is a deliberate override; the mint neither reads nor clobbers it."""
    (tmp_path / ".env").write_text("BLUESKY_LAUNCH_TOKEN=manually-set\n", encoding="utf-8")
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_LAUNCH_TOKEN"] == "manually-set"
    # The missing var alongside it still mints — unconditional means per var.
    assert env.get("BLUESKY_TILED_API_KEY")


def test_explicitly_empty_value_is_left_empty_not_minted(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """``TOKEN=`` is a deliberate value: the operator is choosing a fail-closed bridge.

    Minting over it would override that choice. On a loopback deploy the
    service simply fails closed on its own; only a deployment the build
    rendered as reachable off-host refuses (covered in
    ``test_bluesky_token_mint.py``).
    """
    (tmp_path / ".env").write_text("BLUESKY_LAUNCH_TOKEN=\n", encoding="utf-8")
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_LAUNCH_TOKEN"] == ""
    assert env.get("BLUESKY_TILED_API_KEY")

"""Tests for the local-exec arming guard.

The local python-executor path runs agent-authored code with cwd=project_root
and no filesystem/network sandboxing, so it can read BLUESKY_PROMOTE_TOKEN
straight out of .env/config.yml and POST the bridge's /runs/{id}/promote
directly — bypassing launch_scan's in-tool writes_enabled re-check entirely.
The container execution_method is fs/network isolated and doesn't have this
exposure. Per the user ruling (2026-07-06, "guard + document"),
container_lifecycle._ensure_service_tokens must refuse to mint
BLUESKY_PROMOTE_TOKEN — leaving the bridge unarmed (its own require_armed()
then 503s) — whenever control_system.writes_enabled and
execution.execution_method=local are both true.

The guard is *per token variable*, not per service, and is a fail-closed
allowlist: under the unsafe config a declared var mints only if it appears in
_LOCAL_EXEC_SAFE_VARS. BLUESKY_TILED_API_KEY hangs off the same deployed
'bluesky' service but gates catalog access, not an arming capability, so it is
allowlisted and must still mint. A var nobody triaged — e.g. an arming token a
future service declares — is withheld by omission rather than minted silently.

Every test that asserts the promote token is withheld therefore also asserts the
Tiled key is present: a one-sided assertion cannot distinguish the correct gate
from an inverted one.
"""

from __future__ import annotations

import logging

import pytest

from osprey.deployment import container_lifecycle


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
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    monkeypatch.delenv("BLUESKY_TILED_API_KEY", raising=False)
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)


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
# _local_exec_arming_unsafe: pure function, easiest to pin down directly.
# ---------------------------------------------------------------------------


def test_arming_unsafe_true_when_writes_enabled_and_local():
    config = {
        "control_system": {"writes_enabled": True},
        "execution": {"execution_method": "local"},
    }
    assert container_lifecycle._local_exec_arming_unsafe(config) is True


def test_arming_safe_when_writes_enabled_and_container():
    config = {
        "control_system": {"writes_enabled": True},
        "execution": {"execution_method": "container"},
    }
    assert container_lifecycle._local_exec_arming_unsafe(config) is False


def test_arming_safe_when_writes_disabled_even_if_local():
    config = {
        "control_system": {"writes_enabled": False},
        "execution": {"execution_method": "local"},
    }
    assert container_lifecycle._local_exec_arming_unsafe(config) is False


def test_arming_safe_by_default_when_sections_absent():
    # writes_enabled defaults False, execution_method defaults "container".
    assert container_lifecycle._local_exec_arming_unsafe({}) is False


# ---------------------------------------------------------------------------
# End-to-end through deploy_up: the guard must actually suppress the mint.
# ---------------------------------------------------------------------------


def test_promote_token_withheld_but_tiled_key_minted_when_writes_enabled_and_local(
    captured_argv, _clean_token_env, monkeypatch, tmp_path, caplog
):
    """The gate is variable-scoped: it withholds exactly the arming token.

    Both directions are asserted in one run. An inverted membership check —
    gating BLUESKY_TILED_API_KEY instead of BLUESKY_PROMOTE_TOKEN — silently
    arms the promote path under local execution, and a test that only checked
    for the Tiled key would still pass.
    """
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    with caplog.at_level(logging.WARNING, logger="deployment.lifecycle"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    # The arming token is withheld: the bridge's require_armed() keeps 503ing.
    assert "BLUESKY_PROMOTE_TOKEN" not in env
    # Tiled persistence is not an arming capability — its key still mints.
    assert env.get("BLUESKY_TILED_API_KEY")
    assert len(env["BLUESKY_TILED_API_KEY"]) >= 40
    # Smoke assertion only — a single sample is ~29% likely to pass even against
    # a `token_urlsafe` recipe. The real guard on this invariant is
    # `test_allowlisted_tiled_key_is_alphanumeric_on_every_unsafe_local_mint`
    # below, which asserts it across 50 independent mints. Do not delete that
    # one on the strength of this one.
    assert env["BLUESKY_TILED_API_KEY"].isalnum()

    # caplog's handler already calls record.getMessage() (formatting in %args),
    # so r.message here is the final rendered string — no further % needed.
    warnings = " ".join(r.message for r in caplog.records)
    assert "BLUESKY_PROMOTE_TOKEN" in warnings
    assert "BLUESKY_TILED_API_KEY" not in warnings
    assert "bluesky" in warnings
    assert "writes_enabled" in warnings
    assert "local" in warnings


def test_allowlisted_tiled_key_is_alphanumeric_on_every_unsafe_local_mint(
    _clean_token_env, tmp_path
):
    """The allowlist path mints a key Tiled will actually accept.

    ``.isalnum()`` on a single mint is not a guard: a token_urlsafe(32) value is
    all-alphanumeric ~29% of the time, so one sample would pass by luck against
    the recipe that crash-loops the Tiled container. Asserting the invariant
    across 50 independent mints drops that to 0.29**50.

    This is the guard for the *withheld-promote* configuration specifically: the
    two behaviors are coupled — the promote token stays absent while this key
    must still mint, and mint usably.
    """
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )

    for i in range(50):
        env_path = tmp_path / f"{i}.env"
        container_lifecycle._ensure_service_tokens(config, expose_network=False, env_path=env_path)
        env = _parse_dotenv(env_path)
        assert "BLUESKY_PROMOTE_TOKEN" not in env
        assert env["BLUESKY_TILED_API_KEY"].isalnum(), f"mint {i}: {env['BLUESKY_TILED_API_KEY']!r}"


def test_both_bluesky_tokens_minted_when_writes_enabled_and_container(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "container"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    assert len(env["BLUESKY_PROMOTE_TOKEN"]) >= 40
    assert env.get("BLUESKY_TILED_API_KEY")
    assert env["BLUESKY_TILED_API_KEY"] != env["BLUESKY_PROMOTE_TOKEN"]


def test_allowlist_membership_is_pinned():
    """Pin the allowlist so an inverted edit fails loudly, not silently."""
    assert container_lifecycle._LOCAL_EXEC_SAFE_VARS == {
        "BLUESKY_TILED_API_KEY",
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
        "ZO_ROOT_USER_PASSWORD",
    }
    assert "BLUESKY_PROMOTE_TOKEN" not in container_lifecycle._LOCAL_EXEC_SAFE_VARS


def test_every_declared_var_is_classified():
    """No declared token var may be un-triaged.

    Under the allowlist an unclassified var fails *closed*, so this test never
    guards a security hole — it guards against a var silently ceasing to mint
    because nobody reviewed it. If this fails, triage the new var: add it to
    ``_LOCAL_EXEC_SAFE_VARS`` only if it grants no write-capable route the agent
    can walk. Never edit the allowlist merely to make this pass.
    """
    declared = {
        var for token_vars in container_lifecycle._SERVICE_TOKEN_VARS.values() for var in token_vars
    }
    unclassified = declared - container_lifecycle._LOCAL_EXEC_SAFE_VARS
    assert unclassified == {"BLUESKY_PROMOTE_TOKEN"}
    # Every allowlist entry is actually declared by some service — no dead entries.
    assert container_lifecycle._LOCAL_EXEC_SAFE_VARS <= declared


def test_unlisted_arming_token_fails_closed_under_unsafe_local_exec(
    captured_argv, _clean_token_env, monkeypatch, tmp_path, caplog
):
    """A future service's arming token is withheld by *omission* from the allowlist.

    This is the whole reason the gate is an allowlist rather than a blocklist. A
    blocklist mints this token, warns about nothing, and leaves every other test
    in the suite green.
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

    with caplog.at_level(logging.WARNING, logger="deployment.lifecycle"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert "SCAN_ENGINE_ARM_TOKEN" not in env
    assert "BLUESKY_PROMOTE_TOKEN" not in env
    # The allowlisted key still mints — the gate withholds, it does not disable.
    assert env.get("BLUESKY_TILED_API_KEY")

    warnings = " ".join(r.message for r in caplog.records)
    assert "SCAN_ENGINE_ARM_TOKEN" in warnings
    assert "scan_engine" in warnings


def test_unlisted_token_still_mints_under_safe_config(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """Fail-closed applies only under unsafe local exec, not to every unlisted var."""
    monkeypatch.setattr(
        container_lifecycle,
        "_SERVICE_TOKEN_VARS",
        {**container_lifecycle._SERVICE_TOKEN_VARS, "scan_engine": ("SCAN_ENGINE_ARM_TOKEN",)},
    )
    monkeypatch.delenv("SCAN_ENGINE_ARM_TOKEN", raising=False)
    config = _config(
        deployed_services=["scan_engine"],
        control_system={"writes_enabled": True},
        execution={"execution_method": "container"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("SCAN_ENGINE_ARM_TOKEN")


def test_bluesky_token_minted_when_writes_enabled_and_execution_section_absent(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """Default execution_method (no `execution:` section at all) is "container" — safe."""
    config = _config(control_system={"writes_enabled": True})
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")


def test_bluesky_token_behavior_unchanged_when_writes_disabled(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """Even with local exec, writes_enabled=False means no bypass risk — mint as usual."""
    config = _config(
        control_system={"writes_enabled": False},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")


def test_guard_does_not_affect_dispatch_tokens(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """The dispatch tokens are allowlisted, so they mint normally under unsafe local exec.

    They gate an inbound webhook / worker-routing boundary, not a write-capable
    route the agent itself walks.
    """
    config = _config(
        deployed_services=["bluesky", "event_dispatcher", "dispatch_worker"],
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert "BLUESKY_PROMOTE_TOKEN" not in env
    assert env.get("BLUESKY_TILED_API_KEY")
    assert env.get("EVENT_DISPATCHER_TOKEN")
    assert env.get("DISPATCH_WORKER_TOKEN")


def test_existing_manual_token_left_untouched_under_unsafe_config(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """A user-set token is a deliberate override; the guard neither reads nor clobbers it."""
    (tmp_path / ".env").write_text("BLUESKY_PROMOTE_TOKEN=manually-set\n", encoding="utf-8")
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_PROMOTE_TOKEN"] == "manually-set"
    # The Tiled key still mints alongside the operator's token.
    assert env.get("BLUESKY_TILED_API_KEY")

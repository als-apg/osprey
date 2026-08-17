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
import os
import subprocess

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

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # run_captured hangs its spool path off the result, so the stand-in has
        # to be an object, and it passes redirection kwargs this ignores.
        return subprocess.CompletedProcess(list(cmd), 0)

    # Stubbed at `run_captured`, the one seam every deploy child goes through:
    # a watched capture (the image builds pass `on_line=`) reads its child
    # through a pipe instead of `subprocess.run`, so a `subprocess.run` stub
    # would be walked straight past and this test would run a real build.
    monkeypatch.setattr(container_lifecycle, "run_captured", _fake_run)
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


def test_an_explicitly_empty_dotenv_value_is_minted_over(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """``TOKEN=`` in ``.env`` is a blank, not a decision — the mint fills it.

    This replaces the opposite expectation. A minted service token has no
    meaningful empty state: an empty ``BLUESKY_LAUNCH_TOKEN`` does not choose a
    fail-closed bridge, it starts one with no authentication and says nothing
    about it, on the default loopback deploy where no other guard is watching.
    The narrow carve-out — an empty value exported in a shell whose ``.env``
    never mentions the var — is pinned separately below.
    """
    (tmp_path / ".env").write_text("BLUESKY_LAUNCH_TOKEN=\n", encoding="utf-8")
    # The deploy loads that .env over the process environment before it mints,
    # so this is the state the predicate actually sees. Set it through
    # monkeypatch so the minted value does not outlive the test.
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "")
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    minted = env["BLUESKY_LAUNCH_TOKEN"]
    assert minted, "an empty launch token was left empty on a loopback deploy"
    assert env.get("BLUESKY_TILED_API_KEY")
    assert minted != env["BLUESKY_TILED_API_KEY"]
    # The appended line is what the deploy resolves: parse order is last-wins,
    # the same rule compose applies to a repeated key in an --env-file.
    assert (tmp_path / ".env").read_text().index(f"BLUESKY_LAUNCH_TOKEN={minted}") > 0
    # And the stale empty copy the .env load left in this process is replaced,
    # so compose resolves the minted value rather than being shadowed by it.
    assert os.environ["BLUESKY_LAUNCH_TOKEN"] == minted


def test_an_exported_empty_value_is_left_alone_because_a_mint_cannot_reach_it(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """The bound on the exception above: an export the ``.env`` cannot outrank.

    The ``.env`` here does not carry the var at all, so the empty value is the
    shell's own rather than that file's mirrored back. An export wins over
    ``--env-file`` for compose's interpolation, so a mint appended to ``.env``
    would be a token reported and never delivered. Nothing is written, and a
    deployment the build rendered as reachable off-host refuses instead (covered
    in ``test_bluesky_token_mint.py``).
    """
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "")
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert "BLUESKY_LAUNCH_TOKEN" not in env
    # The var beside it still mints — the carve-out is per var, not per deploy.
    assert env.get("BLUESKY_TILED_API_KEY")


# ---------------------------------------------------------------------------
# The shared half of the chain: what this function's own read can and cannot see
# ---------------------------------------------------------------------------
# The deployment's environment is a two-file chain at the repo root —
# ``.env.shared`` (committed defaults) below ``.env`` (host-local secrets). This
# function reads exactly one of those files: ``parse_dotenv_file(env_path)``
# names the LOCAL one, and ``_effective_value`` consults ``os.environ`` ahead of
# it. So the shared half reaches the mint by a single indirect route — the CLI
# entry point loads the whole chain over ``os.environ`` before any deploy code
# runs, and the mint reads that mirror rather than the file.
#
# Both halves of that are pinned below, as observed rather than as wished for:
# what the function makes of a shared-only value on its own, and what it makes
# of the same value once the entry-point load has happened. The pair is the
# measurement — the second test is what says the mint's answer depends on a step
# some other component took.

#: Obviously-fake stand-ins. Every value in this section is a fixture, never a
#: credential recipe, and none of them is ever asserted against a minted value
#: except to show the two differ.
SHARED_HALF_TOKEN = "shared-half-fixture-token-not-a-secret"


def _write_shared_half(tmp_path, text: str):
    """Lay down the chain's committed-defaults file beside the local one."""
    from osprey.utils.dotenv import ENV_SHARED_FILENAME

    path = tmp_path / ENV_SHARED_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def _load_the_entry_point_chain(monkeypatch, repo):
    """Run the load ``osprey <verb>`` performs before it reaches any deploy code.

    ``osprey.cli.main`` calls this first thing, cwd-rooted, with override
    semantics — so by the time the mint runs, ``os.environ`` carries the merged
    chain. Reproduced here verbatim rather than simulated with ``setenv``,
    because the thing being measured IS that this step is what makes the shared
    half visible.
    """
    import osprey.utils.config as config

    monkeypatch.setattr(config, "_dotenv_shell_overrides", {})
    monkeypatch.chdir(repo)
    config.load_project_dotenv()


def test_a_token_only_the_shared_half_carries_is_minted_over(_clean_token_env, tmp_path):
    """``.env.shared`` alone is invisible to the mint, so the var reads as absent.

    Pinned as it behaves, not as it ought to: the function's file read names the
    local ``.env``, and a value the operator committed to the shared half is not
    one this predicate can see. It therefore mints — and the minted line lands
    in ``.env``, which outranks ``.env.shared`` everywhere in the chain, so the
    operator's committed value stops being the one the deployment resolves.

    The shared file itself is left untouched, which is the only part of this a
    reader would have guessed.
    """
    _write_shared_half(tmp_path, f"BLUESKY_LAUNCH_TOKEN={SHARED_HALF_TOKEN}\n")
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(_config(), expose_network=False, env_path=env_path)

    local = _parse_dotenv(env_path)
    assert local["BLUESKY_LAUNCH_TOKEN"], "the shared-only value did not even reach the predicate"
    assert local["BLUESKY_LAUNCH_TOKEN"] != SHARED_HALF_TOKEN
    from osprey.utils.dotenv import ENV_SHARED_FILENAME

    assert _parse_dotenv(tmp_path / ENV_SHARED_FILENAME)["BLUESKY_LAUNCH_TOKEN"] == (
        SHARED_HALF_TOKEN
    )


def test_the_shared_half_is_honoured_once_the_entry_point_chain_load_has_run(
    _clean_token_env, monkeypatch, tmp_path
):
    """The route that makes the shared half visible: the process-env mirror.

    Same fixture as the test above, with the one step a real ``osprey up``
    takes first. The chain load puts the shared value in ``os.environ``,
    ``_effective_value`` reads that ahead of the ``.env`` mapping, and the var
    is left alone. Nothing else changed — so if this ever disagrees with the
    test above again, the delivery of the environment moved, not the mint.
    """
    _write_shared_half(tmp_path, f"BLUESKY_LAUNCH_TOKEN={SHARED_HALF_TOKEN}\n")
    env_path = tmp_path / ".env"

    _load_the_entry_point_chain(monkeypatch, tmp_path)
    container_lifecycle._ensure_service_tokens(_config(), expose_network=False, env_path=env_path)

    local = _parse_dotenv(env_path)
    assert "BLUESKY_LAUNCH_TOKEN" not in local
    assert os.environ["BLUESKY_LAUNCH_TOKEN"] == SHARED_HALF_TOKEN
    # Per var, as everywhere else here: the one the shared half does not carry
    # still mints.
    assert local.get("BLUESKY_TILED_API_KEY")


class TestEffectiveValueWithAllThreeSourcesDisagreeing:
    """The conflict configuration: shell, ``.env`` and ``.env.shared`` differ.

    ``_effective_value`` is a two-argument function — process env, then the
    mapping the caller passes — and its docstring makes a claim about a THIRD
    thing, the deploy path it sits on: that the ``.env`` has already been loaded
    over the process env, so a shell export is only visible when the file does
    not carry the key. The first test below pins the function's own mechanics;
    the second pins the composed behaviour the docstring describes, with all
    three sources set to different values so no two of them can be confused.
    """

    VAR = "BLUESKY_LAUNCH_TOKEN"

    def _effective_value(self):
        from osprey.deployment.service_tokens import _effective_value

        return _effective_value

    def test_the_process_env_outranks_the_mapping_the_caller_passes(
        self, _clean_token_env, monkeypatch
    ):
        """The mechanic, in isolation: ``os.environ`` first, mapping second, ``""``.

        Called with a mapping the process env contradicts, which on the deploy
        path never happens by accident — the loader put the file's value there.
        """
        effective_value = self._effective_value()
        monkeypatch.setenv(self.VAR, "value-from-the-shell")

        assert effective_value(self.VAR, {self.VAR: "value-from-the-local-file"}) == (
            "value-from-the-shell"
        )
        monkeypatch.delenv(self.VAR)
        assert effective_value(self.VAR, {self.VAR: "value-from-the-local-file"}) == (
            "value-from-the-local-file"
        )
        assert effective_value(self.VAR, {}) == ""

    def test_on_the_deploy_path_the_local_file_wins_and_the_shell_loses(
        self, _clean_token_env, monkeypatch, tmp_path
    ):
        """The composed answer, with every source of the chain disagreeing.

        Order observed: ``.env`` beats ``.env.shared`` beats the shell export.
        The shell losing is not ``_effective_value``'s doing — it reads the
        process env first — but the entry-point load overwrote that export with
        the chain's value, which is exactly what the docstring says: the derived
        file, not the ambient shell, decides a present key's value.
        """
        effective_value = self._effective_value()
        _write_shared_half(tmp_path, f"{self.VAR}=value-from-the-shared-file\n")
        (tmp_path / ".env").write_text(f"{self.VAR}=value-from-the-local-file\n", encoding="utf-8")
        monkeypatch.setenv(self.VAR, "value-from-the-shell")

        _load_the_entry_point_chain(monkeypatch, tmp_path)
        on_disk = _parse_dotenv(tmp_path / ".env")

        assert effective_value(self.VAR, on_disk) == "value-from-the-local-file"

    def test_a_key_only_the_shared_file_sets_outranks_the_shell_too(
        self, _clean_token_env, monkeypatch, tmp_path
    ):
        """The middle rank, which the two-file case cannot show on its own.

        With no local line for the key, the winner is the committed default
        rather than the operator's export — the chain load overrides both files
        over the process env, and the shared half is a member of that chain.
        """
        effective_value = self._effective_value()
        _write_shared_half(tmp_path, f"{self.VAR}=value-from-the-shared-file\n")
        monkeypatch.setenv(self.VAR, "value-from-the-shell")

        _load_the_entry_point_chain(monkeypatch, tmp_path)

        assert effective_value(self.VAR, {}) == "value-from-the-shared-file"

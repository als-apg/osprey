"""Tests for the web-terminal ``decommission`` lifecycle verb
(osprey.deployment.web_terminals.lifecycle).

The container runtime is entirely mocked: ``lifecycle.subprocess.run`` is patched
to record every emitted argv instead of touching a real docker/podman daemon, and
``lifecycle.get_runtime_command`` is pinned to a fixed ``["docker", "compose"]``.
Typed confirmation is exercised via ``builtins.input`` monkeypatching. No real
container or volume is ever created or removed by these tests.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml
from ruamel.yaml import YAML

from osprey.deployment.compose_generator import resolve_user_volume_names
from osprey.deployment.web_terminals import lifecycle
from osprey.utils import config_writer

_FORBIDDEN_ARGV_TOKENS = {"prune", "-a", "--all", "system", "network"}
_GLOB_METACHARACTERS = set("*?[")


def _config(users, *, project_name="demo-project", facility_prefix="dls"):
    """Minimal-but-complete facility config exercising every field decommission_user reads."""
    return {
        "project_name": project_name,
        "facility": {"name": "Demo Light Source", "prefix": facility_prefix, "timezone": "UTC"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 8080,
                "web_base_port": 9000,
                "artifact_base_port": 9100,
                "ariel_base_port": 9200,
                "lattice_base_port": 9300,
                "users": users,
            }
        },
    }


def _write_config(tmp_path, config):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _reload_users(config_path):
    """Reload config.yml and return modules.web_terminals.users as written."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["modules"]["web_terminals"]["users"]


@pytest.fixture
def fake_runtime(monkeypatch):
    """Patch subprocess.run + get_runtime_command; return the list of captured argvs."""
    calls: list[list[str]] = []

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    return calls


@pytest.fixture
def fake_runtime_prune(monkeypatch):
    """Like ``fake_runtime``, but also stubs ``ps -a``/``volume ls`` discovery output.

    Returns ``(calls, listing)``: ``calls`` records every emitted argv (same
    contract as ``fake_runtime``); ``listing`` is a mutable
    ``{"containers": [...], "volumes": [...]}`` dict tests populate *before*
    calling ``prune_users`` to control what the (mocked) runtime reports as
    existing containers/volumes.
    """
    calls: list[list[str]] = []
    listing: dict[str, list[str]] = {"containers": [], "volumes": []}

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False):
        calls.append(list(argv))
        if argv[1:4] == ["ps", "-a", "--format"]:
            stdout = "\n".join(listing["containers"])
        elif argv[1:3] == ["volume", "ls"]:
            stdout = "\n".join(listing["volumes"])
        else:
            stdout = ""
        return subprocess.CompletedProcess(argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    return calls, listing


@pytest.fixture
def fake_runtime_nuke(monkeypatch):
    """Like ``fake_runtime_prune``, but also lets tests control ``compose down``'s
    exit code/stderr — needed to exercise nuke's abort-before-volume-removal path
    on a failed teardown.

    Returns ``(calls, listing, down_result)``: ``calls``/``listing`` are the same
    contract as ``fake_runtime_prune`` (``listing["volumes"]`` drives
    ``_discover_orphan_volumes``, the same off-roster sweep ``prune_users`` uses);
    ``down_result`` is a mutable ``{"returncode": 0, "stderr": ""}`` dict tests
    can set *before* calling ``nuke_stack`` to simulate a failed ``compose down``.
    """
    calls: list[list[str]] = []
    listing: dict[str, list[str]] = {"containers": [], "volumes": []}
    down_result = {"returncode": 0, "stderr": ""}

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False):
        calls.append(list(argv))
        if argv[1:4] == ["ps", "-a", "--format"]:
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout="\n".join(listing["containers"]), stderr=""
            )
        if argv[1:3] == ["volume", "ls"]:
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout="\n".join(listing["volumes"]), stderr=""
            )
        if "down" in argv:
            return subprocess.CompletedProcess(
                argv, returncode=down_result["returncode"], stdout="", stderr=down_result["stderr"]
            )
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    return calls, listing, down_result


def _assert_no_input_prompt(monkeypatch):
    """Fail the test if decommission_user ever falls through to an interactive prompt."""

    def _unexpected_input(prompt=""):
        raise AssertionError(f"input() must not be called with assume_yes=True: {prompt!r}")

    monkeypatch.setattr("builtins.input", _unexpected_input)


# =============================================================================
# retain (default): no volume destruction
# =============================================================================


def test_decommission_retain_by_default_removes_container_not_volumes(
    tmp_path, monkeypatch, fake_runtime
):
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    volume_calls = [c for c in fake_runtime if c[1:2] == ["volume"]]
    assert volume_calls == []

    container_calls = [c for c in fake_runtime if c[1] == "rm"]
    assert container_calls == [["docker", "rm", "-f", "dls-web-alice"]]

    # Roster entry gone; the survivor remains.
    users = _reload_users(config_path)
    assert users == [{"name": "bob", "index": 1}]

    # Artifacts re-rendered from the updated config.
    compose = (tmp_path / "docker-compose.web.yml").read_text(encoding="utf-8")
    assert "web-alice" not in compose
    assert "web-bob" in compose


# =============================================================================
# --purge
# =============================================================================


def test_decommission_purge_removes_both_exact_volumes(tmp_path, monkeypatch, fake_runtime):
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)
    claude_vol, agent_vol = resolve_user_volume_names(config, "alice")

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    volume_rm_calls = [c for c in fake_runtime if c[1:3] == ["volume", "rm"]]
    assert volume_rm_calls == [
        ["docker", "volume", "rm", claude_vol],
        ["docker", "volume", "rm", agent_vol],
    ]
    assert claude_vol == "demo-project_alice-claude-config"
    assert agent_vol == "demo-project_alice-agent-data"

    # No archive ("run") call when purging.
    run_calls = [c for c in fake_runtime if c[1] == "run"]
    assert run_calls == []


# =============================================================================
# --archive
# =============================================================================


def test_decommission_archive_tars_then_removes_each_volume_in_order(
    tmp_path, monkeypatch, fake_runtime
):
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    claude_vol, agent_vol = resolve_user_volume_names(config, "alice")

    lifecycle.decommission_user(str(config_path), "alice", archive=True, assume_yes=True)

    for volume in (claude_vol, agent_vol):
        mount_arg = f"type=volume,source={volume},destination=/from,readonly"
        archive_index = next(
            i for i, c in enumerate(fake_runtime) if c[1] == "run" and mount_arg in c
        )
        rm_index = next(
            i for i, c in enumerate(fake_runtime) if c[1:3] == ["volume", "rm"] and c[3] == volume
        )
        assert archive_index < rm_index, f"{volume} must be archived before it is removed"


# =============================================================================
# confirmation gate
# =============================================================================


def test_decommission_purge_without_confirmation_leaves_everything_untouched(
    tmp_path, monkeypatch, fake_runtime
):
    """Confirmation is gated up-front for --archive/--purge, before any mutation:
    a decline must be a true no-op, not a partial removal."""
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    original_text = config_path.read_text(encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # blank, non-matching response

    with pytest.raises(RuntimeError):
        lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=False)

    # Zero runtime calls at all — not container removal, not volume ops.
    assert fake_runtime == []
    # config.yml (roster) is untouched.
    assert config_path.read_text(encoding="utf-8") == original_text
    # Artifacts were never re-rendered.
    assert not (tmp_path / "docker-compose.web.yml").exists()


def test_decommission_purge_assume_yes_skips_prompt_and_proceeds(
    tmp_path, monkeypatch, fake_runtime
):
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    _assert_no_input_prompt(monkeypatch)

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    volume_rm_calls = [c for c in fake_runtime if c[1:3] == ["volume", "rm"]]
    assert len(volume_rm_calls) == 2


def test_decommission_purge_typed_username_confirms(tmp_path, monkeypatch, fake_runtime):
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "alice")

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=False)

    volume_rm_calls = [c for c in fake_runtime if c[1:3] == ["volume", "rm"]]
    assert len(volume_rm_calls) == 2


def test_decommission_purge_generic_yes_no_longer_confirms(tmp_path, monkeypatch, fake_runtime):
    """A literal "yes" is deliberately NOT accepted — only the exact username is.
    Muscle-memory "yes" on an irreversible two-volume destroy defeats the point
    of a typed confirmation."""
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

    with pytest.raises(RuntimeError):
        lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=False)

    assert fake_runtime == []


# =============================================================================
# roster migration
# =============================================================================


def test_decommission_migrates_legacy_roster_and_freezes_survivor_index(
    tmp_path, monkeypatch, fake_runtime
):
    """Decommissioning a mid-list user in a legacy bare roster must freeze indices:
    the later survivor keeps its ORIGINAL positional index, not a renumbered one."""
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob", "carol"])  # bare roster, positions 0/1/2
    config_path = _write_config(tmp_path, config)

    lifecycle.decommission_user(str(config_path), "bob", assume_yes=True)

    users = _reload_users(config_path)
    assert users == [
        {"name": "alice", "index": 0},
        {"name": "carol", "index": 2},  # frozen at its original position, not renumbered to 1
    ]


# =============================================================================
# user not found
# =============================================================================


def test_decommission_unknown_user_raises_and_destroys_nothing(tmp_path, monkeypatch, fake_runtime):
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    original_text = config_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        lifecycle.decommission_user(str(config_path), "eve", assume_yes=True)

    assert fake_runtime == []
    assert config_path.read_text(encoding="utf-8") == original_text


# =============================================================================
# argv safety
# =============================================================================


def test_decommission_argv_safety_no_dangerous_flags_or_bare_volume_rm(
    tmp_path, monkeypatch, fake_runtime
):
    """Scan every argv emitted across the most destructive path (--archive) for
    blanket/dangerous runtime operations. Every emitted argv must operate on a
    single exact-named resource."""
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)

    lifecycle.decommission_user(str(config_path), "alice", archive=True, assume_yes=True)

    assert fake_runtime, "expected at least one runtime call to inspect"
    for argv in fake_runtime:
        joined = " ".join(argv)
        for token in _FORBIDDEN_ARGV_TOKENS:
            assert token not in argv, f"forbidden token {token!r} in argv: {joined}"
        if argv[1:3] == ["volume", "rm"]:
            name = argv[3] if len(argv) == 4 else ""
            assert name, f"volume rm without exact name: {joined}"
            assert not any(ch in name for ch in _GLOB_METACHARACTERS), (
                f"glob metacharacter in volume name: {joined}"
            )
        if argv[1] == "rm":
            name = argv[3] if len(argv) == 4 else ""
            assert name, f"container rm without exact name: {joined}"
            assert not any(ch in name for ch in _GLOB_METACHARACTERS), (
                f"glob metacharacter in container name: {joined}"
            )


# =============================================================================
# prune
# =============================================================================


def test_prune_no_orphans_is_a_noop(tmp_path, monkeypatch, fake_runtime_prune, capsys):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    alice_claude, alice_agent = resolve_user_volume_names(config, "alice")
    listing["containers"] = ["dls-web-alice"]
    listing["volumes"] = [alice_claude, alice_agent]

    lifecycle.prune_users(str(config_path), assume_yes=True)

    removal_calls = [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]]
    assert removal_calls == []
    assert "no off-roster" in capsys.readouterr().out


def test_prune_dry_run_prints_plan_and_removes_nothing(
    tmp_path, monkeypatch, fake_runtime_prune, capsys
):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-alice", "dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), dry_run=True)

    removal_calls = [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]]
    assert removal_calls == []
    out = capsys.readouterr().out
    assert "eve" in out
    assert "dry-run" in out.lower()


def test_prune_removes_only_off_roster_resources(tmp_path, monkeypatch, fake_runtime_prune):
    """On-roster alice's container/volumes must be left untouched; only orphaned
    eve's exact-named resources are removed."""
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    alice_claude, alice_agent = resolve_user_volume_names(config, "alice")
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-alice", "dls-web-eve"]
    listing["volumes"] = [alice_claude, alice_agent, eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), purge=True, assume_yes=True)

    container_calls = [c for c in calls if c[1] == "rm"]
    assert container_calls == [["docker", "rm", "-f", "dls-web-eve"]]

    volume_rm_calls = [c for c in calls if c[1:3] == ["volume", "rm"]]
    assert volume_rm_calls == [
        ["docker", "volume", "rm", eve_claude],
        ["docker", "volume", "rm", eve_agent],
    ]


def test_prune_retain_default_removes_container_not_volumes(
    tmp_path, monkeypatch, fake_runtime_prune
):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config([])
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), assume_yes=True)

    assert [c for c in calls if c[1] == "rm"] == [["docker", "rm", "-f", "dls-web-eve"]]
    assert [c for c in calls if c[1:3] == ["volume", "rm"]] == []


def test_prune_purge_removes_volumes_after_confirmation(tmp_path, monkeypatch, fake_runtime_prune):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config([])
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]
    monkeypatch.setattr("builtins.input", lambda prompt="": "prune")

    lifecycle.prune_users(str(config_path), purge=True, assume_yes=False)

    volume_rm_calls = [c for c in calls if c[1:3] == ["volume", "rm"]]
    assert volume_rm_calls == [
        ["docker", "volume", "rm", eve_claude],
        ["docker", "volume", "rm", eve_agent],
    ]


def test_prune_archive_tars_then_removes_each_volume_in_order(
    tmp_path, monkeypatch, fake_runtime_prune
):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config([])
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), archive=True, assume_yes=True)

    for volume in (eve_claude, eve_agent):
        mount_arg = f"type=volume,source={volume},destination=/from,readonly"
        archive_index = next(i for i, c in enumerate(calls) if c[1] == "run" and mount_arg in c)
        rm_index = next(
            i for i, c in enumerate(calls) if c[1:3] == ["volume", "rm"] and c[3] == volume
        )
        assert archive_index < rm_index, f"{volume} must be archived before it is removed"


def test_prune_without_confirmation_leaves_everything_untouched(
    tmp_path, monkeypatch, fake_runtime_prune
):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config([])
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # blank, non-matching response

    with pytest.raises(RuntimeError):
        lifecycle.prune_users(str(config_path), assume_yes=False)

    removal_calls = [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]]
    assert removal_calls == []


def test_prune_assume_yes_skips_prompt_and_proceeds(tmp_path, monkeypatch, fake_runtime_prune):
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config([])
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]
    _assert_no_input_prompt(monkeypatch)

    lifecycle.prune_users(str(config_path), purge=True, assume_yes=True)

    assert [c for c in calls if c[1:3] == ["volume", "rm"]]


def test_prune_argv_safety_removal_commands_are_exact_named(
    tmp_path, monkeypatch, fake_runtime_prune
):
    """Discovery (``ps -a`` / ``volume ls``) is read-only and legitimately contains
    ``-a``; the guardrail applies only to REMOVAL argv, which must never contain a
    forbidden token and must name exactly one resource."""
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    alice_claude, alice_agent = resolve_user_volume_names(config, "alice")
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-alice", "dls-web-eve"]
    listing["volumes"] = [alice_claude, alice_agent, eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), purge=True, assume_yes=True)

    removal_calls = [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"]]
    assert removal_calls, "expected at least one removal call"
    for argv in removal_calls:
        joined = " ".join(argv)
        for token in _FORBIDDEN_ARGV_TOKENS:
            assert token not in argv, f"forbidden token {token!r} in argv: {joined}"
        assert len(argv) == 4, f"removal argv must name exactly one resource: {joined}"
        name = argv[3]
        assert name, f"removal argv without exact name: {joined}"
        assert not any(ch in name for ch in _GLOB_METACHARACTERS), (
            f"glob metacharacter in resource name: {joined}"
        )

    # Discovery calls are distinct from removal calls and are the only place
    # `-a` legitimately appears.
    discovery_calls = [c for c in calls if c not in removal_calls]
    assert any(c[1:3] == ["ps", "-a"] for c in discovery_calls)
    assert any(c[1:3] == ["volume", "ls"] for c in discovery_calls)


# =============================================================================
# nuke
# =============================================================================


def _removal_calls(calls):
    """Filter emitted argv down to actual removal/teardown calls, excluding the
    read-only ``ps -a``/``volume ls`` discovery ``nuke_stack`` also emits (same
    read-vs-removal distinction the prune tests draw)."""
    return [c for c in calls if c[1] == "rm" or c[1:3] == ["volume", "rm"] or "down" in c]


def test_nuke_without_confirmation_is_a_true_noop(tmp_path, monkeypatch, fake_runtime_nuke):
    """A decline must leave everything untouched — same ordering lesson as
    decommission/prune: confirm BEFORE any removal. Read-only orphan-volume
    discovery is allowed to run (it feeds the pre-confirmation plan), but zero
    removal/teardown argv may be emitted."""
    calls, listing, _down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # blank, non-matching response

    with pytest.raises(RuntimeError):
        lifecycle.nuke_stack(str(config_path), assume_yes=False)

    assert _removal_calls(calls) == []


def test_nuke_generic_yes_does_not_confirm(tmp_path, monkeypatch, fake_runtime_nuke):
    """Only the literal 'nuke' confirms — a muscle-memory "yes" must not."""
    calls, listing, _down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

    with pytest.raises(RuntimeError):
        lifecycle.nuke_stack(str(config_path), assume_yes=False)

    assert _removal_calls(calls) == []


def test_nuke_typed_confirmation_proceeds(tmp_path, monkeypatch, fake_runtime_nuke):
    calls, listing, _down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "nuke")

    lifecycle.nuke_stack(str(config_path), assume_yes=False)

    down_calls = [c for c in calls if "down" in c]
    assert len(down_calls) == 1

    volume_rm_calls = [c for c in calls if c[1:3] == ["volume", "rm"]]
    assert len(volume_rm_calls) == 4  # 2 users * 2 volumes each


def test_nuke_assume_yes_skips_prompt_removes_project_scoped_containers_and_all_volumes(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    calls, listing, _down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    alice_claude, alice_agent = resolve_user_volume_names(config, "alice")
    bob_claude, bob_agent = resolve_user_volume_names(config, "bob")
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    down_calls = [c for c in calls if "down" in c]
    assert down_calls == [["docker", "compose", "-p", "demo-project", "down"]]

    volume_rm_calls = [c for c in calls if c[1:3] == ["volume", "rm"]]
    assert volume_rm_calls == [
        ["docker", "volume", "rm", alice_claude],
        ["docker", "volume", "rm", alice_agent],
        ["docker", "volume", "rm", bob_claude],
        ["docker", "volume", "rm", bob_agent],
    ]

    # `down` never carries --volumes/-v; volume destruction is exact-named only.
    assert down_calls[0] == ["docker", "compose", "-p", "demo-project", "down"]
    assert "--volumes" not in down_calls[0]
    assert "-v" not in down_calls[0]


def test_nuke_sweeps_off_roster_orphan_volumes_too(
    tmp_path, monkeypatch, capsys, fake_runtime_nuke
):
    """A user who was decommissioned with the default retain policy (or hand-
    edited out of config.yml) still has volumes sitting in the runtime, off the
    current roster. `compose down` tears down their container regardless of
    roster membership, so nuke's volume sweep must reach their volumes too —
    otherwise "tear down everything this project owns" is false. The orphan
    must also show up in the printed plan before confirmation."""
    calls, listing, _down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    alice_claude, alice_agent = resolve_user_volume_names(config, "alice")
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")  # off-roster orphan
    listing["volumes"] = [alice_claude, alice_agent, eve_claude, eve_agent]
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    volume_rm_calls = [c for c in calls if c[1:3] == ["volume", "rm"]]
    removed_names = {c[3] for c in volume_rm_calls}
    assert removed_names == {alice_claude, alice_agent, eve_claude, eve_agent}

    plan_output = capsys.readouterr().out
    assert eve_claude in plan_output
    assert eve_agent in plan_output


def test_nuke_aborts_before_removing_any_volume_when_compose_down_fails(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """A failed `compose down` must abort the whole teardown before any volume
    is touched — proceeding would remove volumes out from under containers
    `down` failed to stop (failing again, "in use", while masking the real
    error) and the CLI would otherwise report success on a failed nuke."""
    calls, listing, down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    down_result["returncode"] = 1
    down_result["stderr"] = "Error: containers still running"
    _assert_no_input_prompt(monkeypatch)

    with pytest.raises(RuntimeError):
        lifecycle.nuke_stack(str(config_path), assume_yes=True)

    down_calls = [c for c in calls if "down" in c]
    assert len(down_calls) == 1  # attempted exactly once, never retried

    volume_rm_calls = [c for c in calls if c[1:3] == ["volume", "rm"]]
    assert volume_rm_calls == []


def test_nuke_argv_safety_no_dangerous_flags_project_scoped_down_exact_named_volumes(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """Scan every argv emitted by the most destructive verb for blanket/dangerous
    runtime operations. The container teardown must be an explicit
    ``compose -p <project> down`` (never a bare/global down); every volume
    removal must be exact-named."""
    calls, listing, _down_result = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    assert calls, "expected at least one runtime call to inspect"
    for argv in calls:
        joined = " ".join(argv)
        for token in _FORBIDDEN_ARGV_TOKENS:
            assert token not in argv, f"forbidden token {token!r} in argv: {joined}"
        if argv[1:4] == ["ps", "-a", "--format"] or argv[1:3] == ["volume", "ls"]:
            continue  # read-only discovery legitimately contains `-a`
        for glob_arg in argv:
            assert not any(ch in glob_arg for ch in _GLOB_METACHARACTERS), (
                f"glob metacharacter in argv: {joined}"
            )
        if "down" in argv:
            assert "-p" in argv, f"compose down must be project-scoped with -p: {joined}"
            assert argv[argv.index("-p") + 1] == "demo-project", (
                f"compose down must be scoped to THIS project: {joined}"
            )
            assert "--volumes" not in argv and "-v" not in argv, (
                f"nuke's compose down must never remove volumes itself: {joined}"
            )
        if argv[1:3] == ["volume", "rm"]:
            name = argv[3] if len(argv) == 4 else ""
            assert name, f"volume rm without exact name: {joined}"


# =============================================================================
# config_writer helper
# =============================================================================


def test_decommission_config_writer_round_trips_object_list_preserving_comments(tmp_path):
    original = """\
# top-level comment
project_name: demo-project  # inline comment
modules:
  web_terminals:
    users:
      - alice
      - bob
other_section:
  keep: true  # keep this comment too
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(original, encoding="utf-8")

    config_writer.config_replace_list(
        config_path,
        ["modules", "web_terminals", "users"],
        [{"name": "alice", "index": 0}, {"name": "carol", "index": 2}],
    )

    updated_text = config_path.read_text(encoding="utf-8")
    assert "# top-level comment" in updated_text
    assert "# inline comment" in updated_text
    assert "# keep this comment too" in updated_text

    yaml_rt = YAML(typ="rt")
    with open(config_path, encoding="utf-8") as f:
        data = yaml_rt.load(f)
    assert data["other_section"]["keep"] is True
    assert data["modules"]["web_terminals"]["users"] == [
        {"name": "alice", "index": 0},
        {"name": "carol", "index": 2},
    ]

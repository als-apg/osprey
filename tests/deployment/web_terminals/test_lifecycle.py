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
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME, PW_HASH_VAR_PREFIX
from osprey.services.auth_sidecar.passwords import verify_password
from osprey.utils import config_writer
from osprey.utils.dotenv import parse_dotenv_file

_FORBIDDEN_ARGV_TOKENS = {"prune", "-a", "--all", "system", "network"}
_GLOB_METACHARACTERS = set("*?[")


def _config(
    users,
    *,
    project_name="demo-project",
    facility_prefix="dls",
    personas=None,
    default_persona=None,
    image_source=None,
):
    """Minimal-but-complete facility config exercising every field decommission_user reads.

    ``personas``/``default_persona``/``image_source`` are only exercised by the
    nuke persona-image tests; omitted, ``resolve_personas`` resolves every
    entry to the zero-migration (non-persona) path, exactly as it does for a
    config predating persona catalogs.
    """
    web_terminals = {
        "enabled": True,
        "nginx_port": 8080,
        "web_base_port": 9000,
        "artifact_base_port": 9100,
        "ariel_base_port": 9200,
        "lattice_base_port": 9300,
        "users": users,
    }
    if personas is not None:
        web_terminals["personas"] = personas
    if default_persona is not None:
        web_terminals["default_persona"] = default_persona
    if image_source is not None:
        web_terminals["image_source"] = image_source
    return {
        "project_name": project_name,
        "facility": {"name": "Demo Light Source", "prefix": facility_prefix, "timezone": "UTC"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {"web_terminals": web_terminals},
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
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (True, ""))
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
        if argv[1:3] == ["ps", "-a"]:
            stdout = "\n".join(listing["containers"])
        elif argv[1:3] == ["volume", "ls"]:
            stdout = "\n".join(listing["volumes"])
        else:
            stdout = ""
        return subprocess.CompletedProcess(argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (True, ""))
    return calls, listing


@pytest.fixture
def fake_runtime_nuke(monkeypatch):
    """Like ``fake_runtime_prune``, but also lets tests control ``compose down``'s
    exit code/stderr — needed to exercise nuke's abort-before-volume-removal path
    on a failed teardown — and each candidate persona image's simulated
    ``image inspect`` result.

    Returns ``(calls, listing, down_result, image_labels)``: ``calls``/
    ``listing`` are the same contract as ``fake_runtime_prune``
    (``listing["volumes"]`` drives ``_discover_orphan_volumes``, the same
    off-roster sweep ``prune_users`` uses); ``down_result`` is a mutable
    ``{"returncode": 0, "stderr": ""}`` dict tests can set *before* calling
    ``nuke_stack`` to simulate a failed ``compose down``; ``image_labels`` is a
    mutable ``{tag: com.osprey.project value or None}`` dict — a tag absent
    from this mapping simulates ``image inspect`` failing (tag doesn't exist on
    this host); a tag present with value ``None`` simulates an image that
    exists but carries no ``com.osprey.project`` label at all.
    """
    calls: list[list[str]] = []
    listing: dict[str, list[str]] = {"containers": [], "volumes": []}
    down_result = {"returncode": 0, "stderr": ""}
    image_labels: dict[str, str | None] = {}

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False):
        calls.append(list(argv))
        if argv[1:3] == ["ps", "-a"]:
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout="\n".join(listing["containers"]), stderr=""
            )
        if argv[1:3] == ["volume", "ls"]:
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout="\n".join(listing["volumes"]), stderr=""
            )
        if argv[1:3] == ["image", "inspect"]:
            tag = argv[3]
            if tag not in image_labels:
                return subprocess.CompletedProcess(
                    argv, returncode=1, stdout="", stderr=f"Error: no such image: {tag}"
                )
            label_value = image_labels[tag]
            labels_json = (
                "null" if label_value is None else f'{{"com.osprey.project":"{label_value}"}}'
            )
            return subprocess.CompletedProcess(argv, returncode=0, stdout=labels_json, stderr="")
        if "down" in argv:
            return subprocess.CompletedProcess(
                argv, returncode=down_result["returncode"], stdout="", stderr=down_result["stderr"]
            )
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (True, ""))
    return calls, listing, down_result, image_labels


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


@pytest.mark.parametrize("verb", ["decommission_user", "prune_users", "nuke_stack"])
def test_downed_runtime_raises_instead_of_touching_anything(tmp_path, monkeypatch, verb):
    """A downed daemon must surface as a RuntimeError, never as a silent no-op.

    Without the preflight, prune's discovery (``ps -a``/``volume ls``) against a
    downed daemon returns empty output and reports "nothing to do".
    """
    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(
        lifecycle, "verify_runtime_is_running", lambda config=None: (False, "daemon is down")
    )
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _config(["alice"]))

    args = ("alice",) if verb == "decommission_user" else ()
    with pytest.raises(RuntimeError, match="daemon is down"):
        getattr(lifecycle, verb)(str(config_path), *args, assume_yes=True)
    assert calls == []


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


def test_prune_discovery_filters_by_compose_project_label(
    tmp_path, monkeypatch, fake_runtime_prune
):
    """Orphan discovery must scope by the compose-assigned
    ``com.docker.compose.project`` label, not a name prefix, so a sibling
    OSPREY deployment on the same host — even one whose project name shares a
    prefix with this one — can never contribute a false match."""
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), dry_run=True)

    ps_calls = [c for c in calls if c[1:3] == ["ps", "-a"]]
    assert ps_calls == [
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=com.docker.compose.project=demo-project",
            "--format",
            "{{.Names}}",
        ]
    ]

    volume_ls_calls = [c for c in calls if c[1:3] == ["volume", "ls"]]
    assert volume_ls_calls == [
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            "label=com.docker.compose.project=demo-project",
            "--format",
            "{{.Name}}",
        ]
    ]


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
    read-only ``ps -a``/``volume ls``/``image inspect`` discovery ``nuke_stack``
    also emits (same read-vs-removal distinction the prune tests draw)."""
    return [
        c
        for c in calls
        if c[1] == "rm" or c[1:3] in (["volume", "rm"], ["image", "rm"]) or "down" in c
    ]


def test_nuke_without_confirmation_is_a_true_noop(tmp_path, monkeypatch, fake_runtime_nuke):
    """A decline must leave everything untouched — same ordering lesson as
    decommission/prune: confirm BEFORE any removal. Read-only orphan-volume
    discovery is allowed to run (it feeds the pre-confirmation plan), but zero
    removal/teardown argv may be emitted."""
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # blank, non-matching response

    with pytest.raises(RuntimeError):
        lifecycle.nuke_stack(str(config_path), assume_yes=False)

    assert _removal_calls(calls) == []


def test_nuke_generic_yes_does_not_confirm(tmp_path, monkeypatch, fake_runtime_nuke):
    """Only the literal 'nuke' confirms — a muscle-memory "yes" must not."""
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice"])
    config_path = _write_config(tmp_path, config)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

    with pytest.raises(RuntimeError):
        lifecycle.nuke_stack(str(config_path), assume_yes=False)

    assert _removal_calls(calls) == []


def test_nuke_typed_confirmation_proceeds(tmp_path, monkeypatch, fake_runtime_nuke):
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
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
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
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
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
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

    volume_ls_calls = [c for c in calls if c[1:3] == ["volume", "ls"]]
    assert volume_ls_calls == [
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            "label=com.docker.compose.project=demo-project",
            "--format",
            "{{.Name}}",
        ]
    ]


def test_nuke_aborts_before_removing_any_volume_when_compose_down_fails(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """A failed `compose down` must abort the whole teardown before any volume
    is touched — proceeding would remove volumes out from under containers
    `down` failed to stop (failing again, "in use", while masking the real
    error) and the CLI would otherwise report success on a failed nuke."""
    calls, listing, down_result, _image_labels = fake_runtime_nuke
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
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    assert calls, "expected at least one runtime call to inspect"
    for argv in calls:
        joined = " ".join(argv)
        for token in _FORBIDDEN_ARGV_TOKENS:
            assert token not in argv, f"forbidden token {token!r} in argv: {joined}"
        if argv[1:3] == ["ps", "-a"] or argv[1:3] == ["volume", "ls"]:
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
# nuke: persona-local image teardown
# =============================================================================


def _persona_config(users, **kwargs):
    """A ``_config`` with one local-mode persona catalog entry, ``control-room``,
    set as ``default_persona`` — so every bare-string roster entry (no explicit
    ``persona:`` key) resolves to it, and ``image_source: local`` means every
    such entry's image is the local-build tag under test."""
    return _config(
        users,
        personas={"control-room": {"project": "acc-control", "project_path": "/x"}},
        default_persona="control-room",
        image_source="local",
        **kwargs,
    )


def test_nuke_removes_label_verified_persona_local_image(tmp_path, monkeypatch, fake_runtime_nuke):
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "demo-project"  # matches THIS deployment
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    image_rm_calls = [c for c in calls if c[1:3] == ["image", "rm"]]
    assert image_rm_calls == [["docker", "image", "rm", "acc-control-control-room:local"]]


def test_nuke_skips_image_with_mismatched_project_label_and_warns(
    tmp_path, monkeypatch, capsys, fake_runtime_nuke
):
    """Image tags are host-global: a tag that resolves for THIS roster but whose
    com.osprey.project label names a different deployment must survive — it may
    belong to a sibling deployment that happens to use the same persona/project
    naming."""
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "some-other-project"
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    image_rm_calls = [c for c in calls if c[1:3] == ["image", "rm"]]
    assert image_rm_calls == []

    out = capsys.readouterr().out
    assert "acc-control-control-room:local" in out
    assert "SKIPPED" in out
    assert "some-other-project" in out


def test_nuke_skips_image_with_missing_label_and_warns(
    tmp_path, monkeypatch, capsys, fake_runtime_nuke
):
    """An image that exists but carries no com.osprey.project label at all is
    treated the same as a mismatch: skipped with a warning, never removed."""
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = None  # exists, but unlabeled
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    assert [c for c in calls if c[1:3] == ["image", "rm"]] == []
    assert "SKIPPED" in capsys.readouterr().out


def test_nuke_tolerates_absent_image_silently(tmp_path, monkeypatch, capsys, fake_runtime_nuke):
    """A candidate tag that doesn't exist on this host at all (image inspect
    fails) is tolerated — never an error, and not called out as a warning
    (there's nothing wrong, it just was never built/already gone)."""
    calls, listing, _down_result, _image_labels = fake_runtime_nuke  # no entry -> inspect fails
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)  # must not raise

    assert [c for c in calls if c[1:3] == ["image", "rm"]] == []
    assert "SKIPPED" not in capsys.readouterr().out


def test_nuke_zero_migration_config_performs_no_image_operations(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """A config with no persona catalog at all (today's zero-migration roster)
    must never touch images: every entry resolves off the non-":local" default
    image, so there are no candidates to inspect or remove — pinned explicitly
    since this is the common case for every facility that hasn't adopted
    personas yet."""
    calls, listing, _down_result, _image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _config(["alice", "bob"], project_name="demo-project")  # no personas configured
    config_path = _write_config(tmp_path, config)
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    assert [c for c in calls if c[1:3] in (["image", "inspect"], ["image", "rm"])] == []


def test_nuke_dedupes_one_image_removal_per_shared_persona(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """Two roster users on the same persona share one image tag — nuke must
    inspect and remove it once, not once per user."""
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "demo-project"
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    inspect_calls = [c for c in calls if c[1:3] == ["image", "inspect"]]
    assert inspect_calls == [
        [
            "docker",
            "image",
            "inspect",
            "acc-control-control-room:local",
            "--format",
            "{{json .Config.Labels}}",
        ]
    ]
    image_rm_calls = [c for c in calls if c[1:3] == ["image", "rm"]]
    assert image_rm_calls == [["docker", "image", "rm", "acc-control-control-room:local"]]


def test_nuke_image_removal_happens_after_compose_down_and_volume_removal(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "demo-project"
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    down_index = next(i for i, c in enumerate(calls) if "down" in c)
    volume_rm_index = next(i for i, c in enumerate(calls) if c[1:3] == ["volume", "rm"])
    image_rm_index = next(i for i, c in enumerate(calls) if c[1:3] == ["image", "rm"])
    assert down_index < volume_rm_index < image_rm_index


def test_nuke_prints_image_plan_before_confirmation(
    tmp_path, monkeypatch, capsys, fake_runtime_nuke
):
    """Both the removed and skipped persona images must appear in the printed
    plan before the typed-confirmation prompt is read — the operator must see
    the exact image outcome, same as the container/volume plan."""
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "demo-project"

    prompts: list[str] = []

    def _record_prompt(prompt=""):
        prompts.append(prompt)
        return "nuke"

    monkeypatch.setattr("builtins.input", _record_prompt)

    lifecycle.nuke_stack(str(config_path), assume_yes=False)

    out = capsys.readouterr().out
    assert "acc-control-control-room:local" in out
    # "image(s)", not "persona image(s)": the set now also carries the auth
    # sidecar's own local tag when authentication is on in local mode.
    assert "image(s)" in out
    # The plan (containing the image line) was printed strictly before input() was read.
    assert prompts, "expected the confirmation prompt to have been read"
    assert "1 image(s)" in prompts[0]


def test_nuke_without_confirmation_never_removes_or_inspects_images_when_declined(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """Declining nuke must still be a true no-op for images: the plan may read
    labels (read-only), but zero image rm argv may ever be emitted."""
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "demo-project"
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    with pytest.raises(RuntimeError):
        lifecycle.nuke_stack(str(config_path), assume_yes=False)

    assert [c for c in calls if c[1:3] == ["image", "rm"]] == []


def test_nuke_argv_safety_image_rm_is_single_exact_named_tag(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """Extends the argv-safety guardrail to image removal: exact one tag per
    argv, no glob metacharacters, no forbidden tokens."""
    calls, listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    image_labels["acc-control-control-room:local"] = "demo-project"

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    image_rm_calls = [c for c in calls if c[1:3] == ["image", "rm"]]
    assert image_rm_calls, "expected at least one image rm call"
    for argv in image_rm_calls:
        joined = " ".join(argv)
        for token in _FORBIDDEN_ARGV_TOKENS:
            assert token not in argv, f"forbidden token {token!r} in argv: {joined}"
        assert len(argv) == 4, f"image rm must name exactly one tag: {joined}"
        assert not any(ch in argv[3] for ch in _GLOB_METACHARACTERS), (
            f"glob metacharacter in image tag: {joined}"
        )


def test_decommission_never_touches_images_even_with_local_personas_configured(
    tmp_path, monkeypatch, fake_runtime
):
    """decommission stays containers+volumes only — image teardown is nuke's
    responsibility alone, even when the facility config has local-mode personas
    configured."""
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice", "bob"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    assert [c for c in fake_runtime if c[1] == "image" or "image" in c] == []


def test_prune_never_touches_images_even_with_local_personas_configured(
    tmp_path, monkeypatch, fake_runtime_prune
):
    """prune stays containers+volumes only — same boundary as decommission."""
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    config = _persona_config(["alice"], project_name="demo-project")
    config_path = _write_config(tmp_path, config)
    eve_claude, eve_agent = resolve_user_volume_names(config, "eve")
    listing["containers"] = ["dls-web-eve"]
    listing["volumes"] = [eve_claude, eve_agent]

    lifecycle.prune_users(str(config_path), purge=True, assume_yes=True)

    assert [c for c in calls if c[1] == "image" or "image" in c] == []


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


# =============================================================================
# passwd: rotate one user's login password
# =============================================================================


def _auth_config(users, method="password"):
    """`_config` plus an `auth` stanza, the only thing the passwd verb adds."""
    config = _config(users)
    config["modules"]["web_terminals"]["auth"] = {"method": method}
    return config


def _read_hash(project_root, user_suffix):
    return parse_dotenv_file(project_root / AUTH_ENV_FILENAME).get(
        f"{PW_HASH_VAR_PREFIX}{user_suffix}", ""
    )


@pytest.fixture
def recreate_calls(monkeypatch):
    """Record calls to the SHARED sidecar-recreate primitive.

    That primitive lives in provision.py, which owns the web stack's compose
    argv; the argv it emits is pinned by provision's own tests. What belongs
    here is that this verb delegates to it — and with which arguments — rather
    than assembling a second copy of the invocation.
    """
    from osprey.deployment.web_terminals import provision

    calls: list[tuple] = []
    monkeypatch.setattr(
        provision,
        "force_recreate_auth_sidecar",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_passwd_stores_the_new_hash_and_recreates_the_sidecar(
    tmp_path, monkeypatch, fake_runtime, recreate_calls
):
    """The hash is replaced AND the sidecar is recreated — a stored hash the
    running sidecar never re-reads would leave the operator with a password
    that does not work."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice", "bob"]))

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert verify_password("alices-new-password", _read_hash(tmp_path, "ALICE"))
    # Delegation contract: the shared primitive is called exactly once, with
    # this deploy's config and nothing else — resolving the --env-file fragment
    # is the primitive's job, not this verb's (one definition, shared with
    # `up`/`down`).
    assert len(recreate_calls) == 1
    (config_arg, *rest), kwargs = recreate_calls[0]
    assert config_arg["modules"]["web_terminals"]["users"] == ["alice", "bob"]
    assert rest == [] and kwargs == {}


def test_passwd_emits_no_runtime_argv_of_its_own(
    tmp_path, monkeypatch, fake_runtime, recreate_calls
):
    """Changing a password touches no container directly: no per-user terminal
    is bounced, nothing is removed, and the only runtime action is the delegated
    single-service recreate."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice", "bob"]))

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert fake_runtime == []


def _capture_recreate_argv(monkeypatch) -> list[list[str]]:
    """Record the argv the REAL recreate primitive emits.

    The fragment used to be resolved at this call site and handed to the
    primitive, so it could be asserted on the delegated call's arguments. It is
    now resolved inside the primitive's own argv builder (one definition shared
    with `up`/`down`), so the honest assertion is on the argv that reaches the
    runtime — a stricter check than the old one, and one that cannot pass while
    the fragment is dropped on the floor.

    CAUTION for anyone extending a test that uses this: `lifecycle`, `provision`
    and `postup_hooks` all do a plain `import subprocess`, so they share ONE
    module object. Patching `provision.subprocess.run` here REPLACES the patch
    `fake_runtime` installed, which means `fake_runtime`'s own list stays EMPTY
    in these tests. Assert on the list this returns; an assertion against
    `fake_runtime` here would be vacuously true.
    """
    from osprey.deployment.web_terminals import provision

    calls: list[list[str]] = []
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **kwargs: calls.append(list(cmd)))
    return calls


def test_passwd_carries_the_env_file_fragment_into_the_recreate(
    tmp_path, monkeypatch, fake_runtime
):
    """The recreate resolves the same variables as every other web-stack
    invocation — omitting the fragment would make this a differently-configured
    `up`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SOMETHING=1\n")
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n")
    config_path = _write_config(tmp_path, _auth_config(["alice"]))
    argv = _capture_recreate_argv(monkeypatch)

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert argv[0][:8] == [
        "docker",
        "compose",
        # Global flags precede the -f list; docker only (see
        # runtime_helper.with_plain_progress).
        "--progress",
        "plain",
        "-f",
        "docker-compose.web.yml",
        "--env-file",
        ".env",
    ]


def test_passwd_before_the_stack_is_deployed_warns_and_does_not_raise(
    tmp_path, monkeypatch, fake_runtime, caplog
):
    """A password can be set before the stack has ever been brought up: with no
    rendered docker-compose.web.yml there is no container to recreate, and
    failing here would turn "not deployed yet" into a rotation error. The hash
    is still stored, and the next `osprey deploy up` puts it in force."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))
    argv = _capture_recreate_argv(monkeypatch)  # no compose file written

    with caplog.at_level("WARNING"):
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert argv == []
    assert "docker-compose.web.yml" in caplog.text
    assert verify_password("alices-new-password", _read_hash(tmp_path, "ALICE"))


def test_passwd_recreate_carries_no_env_file_argument_without_a_dotenv(
    tmp_path, monkeypatch, fake_runtime
):
    """No `.env` at the project root means no `--env-file` argument at all —
    compose rejects one pointing at a file that does not exist."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n")
    config_path = _write_config(tmp_path, _auth_config(["alice"]))
    argv = _capture_recreate_argv(monkeypatch)

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert "--env-file" not in argv[0]


def test_passwd_refuses_when_authentication_is_off(tmp_path, monkeypatch, fake_runtime):
    """`auth.method: none` has no password to change — writing a hash would
    produce a credential nothing ever checks, over a success message."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"], method="none"))

    with pytest.raises(ValueError, match="none"):
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()
    assert fake_runtime == []


def test_passwd_refuses_when_credentials_are_held_by_the_idp(tmp_path, monkeypatch, fake_runtime):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"], method="oidc"))

    with pytest.raises(ValueError, match="identity provider"):
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()
    assert fake_runtime == []


def test_passwd_refuses_an_off_roster_user(tmp_path, monkeypatch, fake_runtime):
    """An off-roster name keys a variable no rendered sidecar reads, so its
    "new password" could never be used to log in anywhere."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))

    with pytest.raises(ValueError, match="not present in modules.web_terminals.users"):
        lifecycle.rotate_user_password(str(config_path), "mallory", "some-password")

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()
    assert fake_runtime == []


def test_passwd_refuses_before_writing_when_the_runtime_is_down(tmp_path, monkeypatch):
    """Gate on the runtime BEFORE the write: otherwise the operator is handed a
    password the deployment was never told about."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (False, "down"))
    config_path = _write_config(tmp_path, _auth_config(["alice"]))

    with pytest.raises(RuntimeError, match="down"):
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_passwd_does_not_recreate_when_the_credential_write_failed(
    tmp_path, monkeypatch, fake_runtime, recreate_calls
):
    """A failed write must abort loudly, not recreate a sidecar around an
    unchanged file and report success."""
    monkeypatch.chdir(tmp_path)

    def _unwritable(*args, **kwargs):
        raise RuntimeError("Could not write .env.auth; the password was NOT changed.")

    monkeypatch.setattr(lifecycle, "set_auth_password", _unwritable)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))

    with pytest.raises(RuntimeError, match="NOT changed"):
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert recreate_calls == []


def test_passwd_reports_a_failed_recreate_as_a_password_that_DID_change(
    tmp_path, monkeypatch, fake_runtime
):
    """The mirror of the write-failure case, and the harder half to get right.

    By the time the recreate runs the hash is already stored, so a bare failure
    would reach the CLI's generic handler as "Deployment failed" — and the
    operator would conclude their old password still works. It does not. The
    error has to say the password changed and name what puts it in force.
    """
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))

    from osprey.deployment.web_terminals import provision

    def _failed_recreate(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["docker", "compose", "up"])

    monkeypatch.setattr(provision, "force_recreate_auth_sidecar", _failed_recreate)

    with pytest.raises(RuntimeError) as exc_info:
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    message = str(exc_info.value)
    assert "WAS changed" in message
    assert "osprey deploy up" in message
    # The stored hash really is the new one — the message is not a consolation.
    assert verify_password("alices-new-password", _read_hash(tmp_path, "ALICE"))


def test_passwd_never_logs_or_prints_the_password(
    tmp_path, monkeypatch, fake_runtime, capsys, caplog
):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))

    with caplog.at_level("DEBUG"):
        lifecycle.rotate_user_password(str(config_path), "alice", "unmistakable-secret")

    assert "unmistakable-secret" not in capsys.readouterr().out
    assert "unmistakable-secret" not in caplog.text


# =============================================================================
# Roster removal with authentication on: credential purge + perimeter reconcile
# =============================================================================


@pytest.fixture
def auth_reconcile_runtime(monkeypatch):
    """Make the delegated web-stack calls (nginx reload, sidecar recreate) safe
    to run, and let the existing runtime fixtures record them.

    Deliberately patches NO subprocess: ``lifecycle``, ``provision`` and
    ``postup_hooks`` all do a plain ``import subprocess``, so they share one
    module object — ``fake_runtime``/``fake_runtime_prune`` already intercept
    every one of these calls, and a second patch here would silently replace
    theirs (including the ``ps -a`` discovery output prune depends on). What
    does need patching is ``provision``'s own ``get_runtime_command`` binding,
    which those fixtures do not reach.
    """
    from osprey.deployment.web_terminals import provision

    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])


def _web_stack_cmds(calls) -> list[list[str]]:
    """The recorded argv that address the web compose project."""
    return [cmd for cmd in calls if "docker-compose.web.yml" in cmd]


def _renderable_auth_config(users, method="password"):
    """`_auth_config` that also passes the render gate: authentication over
    cleartext HTTP is refused at render time unless explicitly allowed, and
    decommission re-renders the artifacts as part of the verb."""
    config = _auth_config(users, method=method)
    config["modules"]["web_terminals"]["auth"]["allow_insecure_http"] = True
    return config


def _seed_env_auth(tmp_path, **hashes) -> None:
    """Write a .env.auth holding one stored hash per named user suffix."""
    lines = [f"{PW_HASH_VAR_PREFIX}{suffix}={value}\n" for suffix, value in hashes.items()]
    (tmp_path / AUTH_ENV_FILENAME).write_text("".join(lines), encoding="utf-8")


def test_decommission_purges_the_departed_users_auth_credentials(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """A re-added same-name user must be minted a FRESH password, never inherit
    the departed user's. Leaving the hash behind would silently hand the next
    holder of that name the previous holder's credential."""
    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice", BOB="scrypt.bob")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    stored = parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)
    assert f"{PW_HASH_VAR_PREFIX}ALICE" not in stored
    assert stored[f"{PW_HASH_VAR_PREFIX}BOB"] == "scrypt.bob"  # untouched


def test_decommission_reloads_nginx_and_recreates_the_auth_sidecar(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """Re-rendering the artifacts is not enough: nginx still serves the previous
    config, and the sidecar holds the old roster and hashes baked in at its
    creation. The reload drops the user's routes; the recreate re-reads both and
    ends their live session now rather than at its natural expiry."""
    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    web_cmds = _web_stack_cmds(fake_runtime)
    assert any(cmd[-4:] == ["nginx", "nginx", "-s", "reload"] for cmd in web_cmds)
    recreates = [cmd for cmd in web_cmds if "--force-recreate" in cmd]
    assert len(recreates) == 1
    # Service-scoped: a bare --force-recreate would bounce every live terminal.
    assert recreates[0][-3:] == ["-d", "--force-recreate", "auth"]


def test_decommission_with_auth_off_touches_no_auth_state(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """`auth.method: none` has no sidecar, no .env.auth and no perimeter to
    reconcile — the verb must behave exactly as it did before auth existed."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"], method="none"))

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert _web_stack_cmds(fake_runtime) == []
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_prune_purges_off_roster_auth_credentials_and_recreates(
    tmp_path, monkeypatch, fake_runtime_prune, auth_reconcile_runtime
):
    """An orphan was hand-edited off the roster, so nothing re-renders — but
    their hash is still in .env.auth and would be inherited by the next user of
    that name. Purge it, then recreate so the sidecar stops honoring it."""
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    # prune re-renders nothing, so the deployed compose file is the one a
    # previous `deploy up` left at the project root.
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")
    _seed_env_auth(tmp_path, ALICE="scrypt.alice", CAROL="scrypt$carol")
    listing["containers"] = ["dls-web-carol"]
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice"]))

    lifecycle.prune_users(str(config_path), assume_yes=True)

    stored = parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)
    assert f"{PW_HASH_VAR_PREFIX}CAROL" not in stored
    assert stored[f"{PW_HASH_VAR_PREFIX}ALICE"] == "scrypt.alice"  # on-roster, untouched
    assert [cmd for cmd in _web_stack_cmds(calls) if "--force-recreate" in cmd]


def test_prune_with_no_auth_entries_to_purge_recreates_nothing(
    tmp_path, monkeypatch, fake_runtime_prune, auth_reconcile_runtime
):
    """Nothing changed in .env.auth and nothing was re-rendered, so there is
    nothing to put into force — bouncing the sidecar would drop every live
    session for no reason."""
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    # The deployed compose file MUST be present: without it warn-and-no-op
    # suppresses the argv on its own and this test would pass even with the
    # nothing-to-put-into-force guard deleted.
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    listing["containers"] = ["dls-web-carol"]  # orphan that never had a credential
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice"]))

    lifecycle.prune_users(str(config_path), assume_yes=True)

    assert _web_stack_cmds(calls) == []


def test_decommission_warns_when_a_departed_users_plaintext_auth_password_survives(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime, caplog
):
    """Purging .env.auth is only half the story: `ensure_auth_credentials` also
    hashes `OSPREY_AUTH_PW_<SUFFIX>` out of the project .env whenever no hash is
    established, so the purge re-opens that fallback and the next `deploy up`
    would re-establish the DEPARTED holder's password for a re-added name. .env
    is operator-owned, so this warns (naming the exact variable) rather than
    silently editing it."""
    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    (tmp_path / ".env").write_text("SOMETHING=1\nOSPREY_AUTH_PW_ALICE=hunter2\n", encoding="utf-8")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    with caplog.at_level("WARNING"):
        lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert "OSPREY_AUTH_PW_ALICE" in caplog.text
    assert ".env" in caplog.text
    assert "hunter2" not in caplog.text  # names the variable, never its value


def test_an_unreadable_env_does_not_become_a_recreate_failure(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime, caplog
):
    """The plaintext-survival check only READS the project `.env`, so a failure
    to read it says nothing about the sidecar. It must not be reported as a
    recreate failure, and above all it must not skip the recreate — the step that
    actually puts the purge into force."""
    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    (tmp_path / ".env").write_text("SOMETHING=1\n", encoding="utf-8")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    def _unreadable(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(lifecycle, "parse_dotenv_file", _unreadable)

    with caplog.at_level("WARNING"):
        lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert [cmd for cmd in _web_stack_cmds(fake_runtime) if "--force-recreate" in cmd]
    assert "Permission denied" in caplog.text
    assert "could not be recreated" not in caplog.text


def test_decommission_auth_recreate_failure_is_fatal_after_the_volume_policy_runs(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """A failed recreate is fatal — the removed user's session may still be live,
    which is the whole point of the verb — but it is raised only AFTER the
    remaining local cleanup completes: every local change already succeeded, and
    abandoning the volume policy would leave a state more inconsistent than the
    error reports."""
    from osprey.deployment.web_terminals import provision

    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    def _failed_recreate(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["docker", "compose", "up"])

    monkeypatch.setattr(provision, "force_recreate_auth_sidecar", _failed_recreate)

    with pytest.raises(RuntimeError) as exc_info:
        lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    message = str(exc_info.value)
    assert "purged" in message
    # Narrow and true: auth_request lives only in the per-user location blocks
    # and the (advisory, non-raising) reload already dropped alice's. What
    # survives is an unreconciled sidecar, NOT a live session — an overstated
    # error an operator later finds untrue costs the next one its credibility.
    assert "still holds their roster entry and password hash" in message
    assert "may still be live" not in message
    assert "osprey deploy up" in message
    # The claim in that message is a verified fact, not a consolation.
    assert f"{PW_HASH_VAR_PREFIX}ALICE" not in parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)
    # ...and the deferred raise let the volume policy finish first.
    assert [cmd for cmd in fake_runtime if cmd[1:3] == ["volume", "rm"]]


def test_prune_auth_recreate_failure_is_fatal_too(
    tmp_path, monkeypatch, fake_runtime_prune, auth_reconcile_runtime
):
    """The prune path reaches the same guard — a decommission-only test would
    miss half of it."""
    from osprey.deployment.web_terminals import provision

    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")
    _seed_env_auth(tmp_path, CAROL="scrypt$carol")
    listing["containers"] = ["dls-web-carol"]
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice"]))

    def _failed_recreate(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["docker", "compose", "up"])

    monkeypatch.setattr(provision, "force_recreate_auth_sidecar", _failed_recreate)

    with pytest.raises(RuntimeError, match="still holds their roster entry"):
        lifecycle.prune_users(str(config_path), assume_yes=True)

    assert f"{PW_HASH_VAR_PREFIX}CAROL" not in parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)


def test_decommission_purge_failure_names_the_user_whose_credential_survives(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """An unwritable .env.auth surfaced as a bare PermissionError after the
    container was already removed, with nothing saying the credential survived
    and that the perimeter was never reconciled."""
    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    def _unwritable(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(lifecycle, "purge_auth_credentials", _unwritable)

    with pytest.raises(RuntimeError) as exc_info:
        lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    message = str(exc_info.value)
    assert "alice" in message
    assert AUTH_ENV_FILENAME in message
    assert "still opens a terminal" in message
    # Decommission targets ONE user, so this purge cleared nobody before it
    # failed: there is no partial change to put into force, and bouncing every
    # live session would be gratuitous. The hash is STILL THERE and no recreate
    # ran. Contrast the multi-user case below, where the recreate MUST run.
    assert f"{PW_HASH_VAR_PREFIX}ALICE" in parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)
    assert not [cmd for cmd in fake_runtime if "--force-recreate" in cmd]
    # The nginx reload is NOT skipped with it: the re-rendered config — already
    # missing alice's route to her removed container — is on disk either way,
    # and only the reload applies it.
    assert any(cmd[-4:] == ["nginx", "nginx", "-s", "reload"] for cmd in fake_runtime)


def test_prune_partial_purge_failure_still_forces_the_earlier_removals_into_effect(
    tmp_path, monkeypatch, fake_runtime_prune, auth_reconcile_runtime
):
    """The failure this guard exists to prevent, and the reason the purge loop
    has a guard of its own.

    With several orphans, a purge that succeeds for carol and then raises for
    dave has ALREADY taken carol's hash out of `.env.auth`. Skipping the
    recreate there would leave the deployment lying in the operator's favour:
    the file they would read no longer lists carol, while the running sidecar
    still holds her hash and still lets her log in — until the next deploy, with
    nothing saying so. So the recreate runs anyway, and only then is the error
    raised, naming dave (whose credential really did survive) and NOT carol.
    """
    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")
    _seed_env_auth(tmp_path, CAROL="scrypt$carol", DAVE="scrypt$dave")
    listing["containers"] = ["dls-web-carol", "dls-web-dave"]
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice"]))

    real_purge = lifecycle.purge_auth_credentials

    def _fails_for_dave(user, project_root):
        if user == "dave":
            raise PermissionError(13, "Permission denied")
        return real_purge(user, project_root)

    monkeypatch.setattr(lifecycle, "purge_auth_credentials", _fails_for_dave)

    with pytest.raises(RuntimeError) as exc_info:
        lifecycle.prune_users(str(config_path), assume_yes=True)

    message = str(exc_info.value)
    # Named as removed-and-still-credentialed: only dave.
    assert "'dave'" in message
    assert "still opens a terminal" in message
    assert "were removed and their auth credentials could not be removed" in message
    # carol appears only as the reassurance that her removal IS in force — never
    # as one of the users whose credential survived.
    assert "'carol' WERE removed" in message
    # The disk state the message describes is the real one.
    stored = parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)
    assert f"{PW_HASH_VAR_PREFIX}CAROL" not in stored
    assert f"{PW_HASH_VAR_PREFIX}DAVE" in stored
    # ...and the partial removal was put into force before the raise.
    assert [cmd for cmd in calls if "--force-recreate" in cmd]


def _auth_persona_config(users, *, method="password", auth_image=None, image_source="local"):
    """A local-mode persona config with authentication on — the shape that makes
    the sidecar's own :local tag a nuke candidate."""
    config = _persona_config(users, project_name="demo-project")
    config["modules"]["web_terminals"]["image_source"] = image_source
    auth: dict = {"method": method}
    if auth_image is not None:
        auth["image"] = auth_image
    config["modules"]["web_terminals"]["auth"] = auth
    return config


def test_nuke_auth_sidecar_image_is_removed_when_label_verified(
    tmp_path, monkeypatch, fake_runtime_nuke
):
    """The sidecar's local tag is this deployment's to tear down, on the same
    label-verified terms as a persona tag — nothing else ever removes it."""
    calls, _listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_persona_config(["alice"]))
    image_labels["dls-assistant-auth:local"] = "demo-project"
    image_labels["acc-control-control-room:local"] = "demo-project"
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    removed = [c[3] for c in calls if c[1:3] == ["image", "rm"]]
    assert "dls-assistant-auth:local" in removed


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"method": "none"}, "auth off renders no sidecar at all"),
        ({"image_source": "registry"}, "registry mode never built a local tag"),
        ({"auth_image": "reg/auth:1"}, "auth.image pins an externally owned image"),
    ],
)
def test_nuke_auth_sidecar_image_is_not_a_candidate(
    tmp_path, monkeypatch, fake_runtime_nuke, kwargs, why
):
    """Only the exact conditions that PRODUCE the local tag make it a teardown
    candidate; removing an image this deployment never built is not ours to do."""
    calls, _listing, _down_result, image_labels = fake_runtime_nuke
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_persona_config(["alice"], **kwargs))
    image_labels["dls-assistant-auth:local"] = "demo-project"  # exists and is ours
    image_labels["acc-control-control-room:local"] = "demo-project"
    _assert_no_input_prompt(monkeypatch)

    lifecycle.nuke_stack(str(config_path), assume_yes=True)

    inspected = [c[3] for c in calls if c[1:3] == ["image", "inspect"]]
    assert "dls-assistant-auth:local" not in inspected, why

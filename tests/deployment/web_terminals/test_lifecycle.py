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
from osprey.deployment.web_terminals.artifacts import BashLaunchTokenConflictError
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME, PW_HASH_VAR_PREFIX
from osprey.deployment.web_terminals.personas import resolve_personas
from osprey.deployment.web_terminals.provision import AUTH_SERVICE_NAME
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


def _stub_rendered_web_stack(tmp_path, body: str = "services: {}\n"):
    """Put a rendered web compose file where the verbs address it: build/.

    The roster verbs probe and re-render ``<repo>/build/docker-compose.web.yml``
    (the pinned invocation contract), so a stub at the repo root would leave
    them believing nothing is deployed — which is exactly the defect that made
    a password rotation report success while recreating nothing.
    """
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    path = build / "docker-compose.web.yml"
    path.write_text(body, encoding="utf-8")
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

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False, **kwargs):
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

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False, **kwargs):
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

    def _fake_run(argv, capture_output=True, text=True, env=None, check=False, **kwargs):
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
    compose = (tmp_path / "build" / "docker-compose.web.yml").read_text(encoding="utf-8")
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
    assert not (tmp_path / "build" / "docker-compose.web.yml").exists()


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


def test_decommission_purge_generic_yes_does_not_confirm(tmp_path, monkeypatch, fake_runtime):
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
    assert "dry run" in out.lower()


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
    # The repo this verb resolved from its config_path is handed over rather
    # than left for the primitive to re-derive: that is what stops the recreate
    # addressing a different repo than the .env.auth this verb just wrote.
    assert rest == [] and set(kwargs) == {"repo_root"}
    assert kwargs["repo_root"] == tmp_path


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

    The fragment is resolved inside the primitive's own argv builder (one
    definition shared with `up`/`down`), not at this call site, so the honest
    assertion is on the argv that reaches the runtime rather than on a
    delegated call's arguments — a stricter check, and one that cannot pass
    while the fragment is dropped on the floor.

    CAUTION for anyone extending a test that uses this: the recreate is a
    captured run, so this patches `provision.run_captured` — the seam the
    primitive actually calls — while `fake_runtime` patches the shared
    `subprocess` module. The recreate therefore never reaches `fake_runtime`'s
    list, which stays EMPTY in these tests. Assert on the list this returns; an
    assertion against `fake_runtime` here would be vacuously true.
    """
    from osprey.deployment.web_terminals import provision

    calls: list[list[str]] = []
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(
        provision,
        "run_captured",
        lambda cmd, **kwargs: calls.append(list(cmd)) or subprocess.CompletedProcess(list(cmd), 0),
    )
    return calls


def test_passwd_carries_the_env_file_fragment_into_the_recreate(
    tmp_path, monkeypatch, fake_runtime
):
    """The recreate resolves the same variables as every other web-stack
    invocation — omitting the fragment would make this a differently-configured
    `up`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SOMETHING=1\n")
    _stub_rendered_web_stack(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))
    argv = _capture_recreate_argv(monkeypatch)

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert argv[0][:10] == [
        "docker",
        "compose",
        # Global flags precede the -f list; docker only (see
        # runtime_helper.with_plain_progress).
        "--progress",
        "plain",
        # The pinned base: the repo IS the compose project directory, the
        # rendered file is addressed under build/, and the env file is the
        # repo's own — none of it inferred from the working directory.
        "--project-directory",
        str(tmp_path),
        "-f",
        str(tmp_path / "build" / "docker-compose.web.yml"),
        "--env-file",
        str(tmp_path / ".env"),
    ]


def test_passwd_before_the_stack_is_deployed_warns_and_does_not_raise(
    tmp_path, monkeypatch, fake_runtime, caplog
):
    """A password can be set before the stack has ever been brought up: with no
    rendered docker-compose.web.yml there is no container to recreate, and
    failing here would turn "not deployed yet" into a rotation error. The hash
    is still stored, and the next `osprey up` puts it in force."""
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
    _stub_rendered_web_stack(tmp_path)
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
    assert "osprey up" in message
    # The stored hash really is the new one — the message is not a consolation.
    assert verify_password("alices-new-password", _read_hash(tmp_path, "ALICE"))


def test_passwd_reports_a_spooled_recreate_failure_the_same_way(
    tmp_path, monkeypatch, fake_runtime
):
    """The recreate's compose output is spooled, so this is the type it raises.

    ``CalledProcessError`` above is the shape the guard was written for;
    ``CapturedProcessError`` is the one it now actually meets. Caught, the
    operator gets the same two facts plus the spool to read. Uncaught, the
    generic CLI handler prints "Deployment failed" over a changed password.
    """
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _auth_config(["alice"]))

    from osprey.deployment.errors import CapturedProcessError
    from osprey.deployment.web_terminals import provision

    spool = tmp_path / "var" / "logs" / "compose-auth-recreate.log"

    def _failed_recreate(*args, **kwargs):
        raise CapturedProcessError(["docker", "compose", "up"], 1, spool)

    monkeypatch.setattr(provision, "force_recreate_auth_sidecar", _failed_recreate)

    with pytest.raises(RuntimeError) as exc_info:
        lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    message = str(exc_info.value)
    assert "'alice'" in message
    assert "WAS changed" in message
    assert str(spool) in message, "the spooled output is unreadable if nothing names it"


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
    """The recorded argv that address the web compose project.

    Matched by substring: the pinned invocation contract spells the ``-f``
    argument as an absolute path into the repo's ``build/`` zone, so exact
    membership of the bare filename no longer holds.
    """
    return [cmd for cmd in calls if any("docker-compose.web.yml" in arg for arg in cmd)]


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
    # previous `osprey up` left at the project root.
    _stub_rendered_web_stack(tmp_path)
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
    _stub_rendered_web_stack(tmp_path)
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
    established, so the purge re-opens that fallback and the next `osprey up`
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
    assert "osprey up" in message
    # The claim in that message is a verified fact, not a consolation.
    assert f"{PW_HASH_VAR_PREFIX}ALICE" not in parse_dotenv_file(tmp_path / AUTH_ENV_FILENAME)
    # ...and the deferred raise let the volume policy finish first.
    assert [cmd for cmd in fake_runtime if cmd[1:3] == ["volume", "rm"]]


def test_decommission_reports_a_spooled_recreate_failure_the_same_way(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """The reconcile guard meets the spooled type too, and must still name both.

    Which user was removed, and where the compose output that explains the
    failure went — an unreconciled sidecar is only actionable with both.
    """
    from osprey.deployment.errors import CapturedProcessError
    from osprey.deployment.web_terminals import provision

    monkeypatch.chdir(tmp_path)
    _seed_env_auth(tmp_path, ALICE="scrypt.alice")
    config_path = _write_config(tmp_path, _renderable_auth_config(["alice", "bob"]))

    spool = tmp_path / "var" / "logs" / "compose-auth-recreate.log"

    def _failed_recreate(*args, **kwargs):
        raise CapturedProcessError(["docker", "compose", "up"], 1, spool)

    monkeypatch.setattr(provision, "force_recreate_auth_sidecar", _failed_recreate)

    with pytest.raises(RuntimeError) as exc_info:
        lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    message = str(exc_info.value)
    assert "'alice'" in message
    assert "still holds their roster entry and password hash" in message
    assert str(spool) in message, "the spooled output is unreadable if nothing names it"


def test_prune_auth_recreate_failure_is_fatal_too(
    tmp_path, monkeypatch, fake_runtime_prune, auth_reconcile_runtime
):
    """The prune path reaches the same guard — a decommission-only test would
    miss half of it."""
    from osprey.deployment.web_terminals import provision

    calls, listing = fake_runtime_prune
    monkeypatch.chdir(tmp_path)
    _stub_rendered_web_stack(tmp_path)
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
    _stub_rendered_web_stack(tmp_path)
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


# =============================================================================
# roster rewrite: what a removal must NOT change about the survivors
# =============================================================================


def _persona_roster_config(users):
    """A config whose catalog makes the two personas distinguishable by image.

    ``readwrite`` is the default, so a survivor whose ``persona: readonly`` is
    lost re-resolves onto it — the privilege change this section exists to
    catch. Only the non-default persona carries a suffixed image, so the two are
    told apart by what ``resolve_personas`` returns, not by the roster text.
    """
    return _config(
        users,
        personas={
            "readonly": {"project": "ctl-readonly"},
            "readwrite": {"project": "ctl-readwrite"},
        },
        default_persona="readwrite",
    )


def _resolved_by_name(config_path):
    """resolve_personas() over the config as it now stands ON DISK."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    web_terminals = data["modules"]["web_terminals"]
    resolved = resolve_personas(web_terminals, data.get("registry", {}), "dls", strict=True)
    return {entry["name"]: entry for entry in resolved}


def test_decommission_keeps_each_survivors_persona_in_the_rewritten_roster(
    tmp_path, monkeypatch, fake_runtime
):
    """The rewritten roster carries every survivor's ``persona:`` key.

    It is written back from the AUTHORED entries, not from the render-facing
    projection: that projection drops ``persona``, and a roster with no
    ``persona:`` keys re-resolves every survivor onto ``default_persona``.
    """
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        _persona_roster_config(
            [
                {"name": "alice", "index": 0, "persona": "readonly"},
                {"name": "bob", "index": 1, "persona": "readwrite"},
                {"name": "carol", "index": 2, "persona": "readonly"},
            ]
        ),
    )

    lifecycle.decommission_user(str(config_path), "carol", assume_yes=True)

    assert _reload_users(config_path) == [
        {"name": "alice", "index": 0, "persona": "readonly"},
        {"name": "bob", "index": 1, "persona": "readwrite"},
    ]


def test_decommission_leaves_survivors_resolving_to_their_own_persona_image(
    tmp_path, monkeypatch, fake_runtime
):
    """The consequence that roster text stands for: after the removal each
    survivor still resolves to the image their own persona names — the
    non-default ``readonly`` user is NOT moved onto the default persona."""
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        _persona_roster_config(
            [
                {"name": "alice", "index": 0, "persona": "readonly"},
                {"name": "bob", "index": 1, "persona": "readwrite"},
                {"name": "carol", "index": 2, "persona": "readonly"},
            ]
        ),
    )

    lifecycle.decommission_user(str(config_path), "carol", assume_yes=True)

    resolved = _resolved_by_name(config_path)
    assert resolved["alice"]["persona"] == "readonly"
    assert resolved["alice"]["image"] == "registry.example.org/web-terminal-readonly:latest"
    # bob was already on the default persona: unsuffixed image, unchanged.
    assert resolved["bob"]["persona"] == "readwrite"
    assert resolved["bob"]["image"] == "registry.example.org/web-terminal:latest"


def test_decommission_preserves_survivor_keys_the_normalizer_does_not_read(
    tmp_path, monkeypatch, fake_runtime
):
    """Removing one user rewrites the whole roster, so it must write back what
    the operator authored about the OTHERS rather than a projection of it.
    ``persona`` is the key that made this a safety problem; the rule is general,
    which is what ``shift`` — a key nothing in this module reads — stands for.
    """
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        _config(
            [
                {
                    "name": "alice",
                    "index": 0,
                    "display_name": "Alice",
                    "theme": "desy-light",
                    "shift": "swing",
                },
                {"name": "bob", "index": 1},
            ]
        ),
    )

    lifecycle.decommission_user(str(config_path), "bob", assume_yes=True)

    assert _reload_users(config_path) == [
        {
            "name": "alice",
            "index": 0,
            "display_name": "Alice",
            "theme": "desy-light",
            "shift": "swing",
        },
    ]


# =============================================================================
# repo-root contract: paths follow config_path, not the working directory
# =============================================================================


def _repo_with_config(tmp_path, config):
    """A deployment repo at ``<tmp>/repo`` whose rendered config sits in build/.

    The layout the verbs actually meet: ``config.yml`` is build output one zone
    below the repo root, while ``.env.auth`` and ``var/`` hang off the root.
    Tests using it deliberately run from a DIFFERENT working directory, so a
    cwd-derived path resolves somewhere the assertions can see.
    """
    repo = tmp_path / "repo"
    (repo / "build").mkdir(parents=True)
    path = repo / "build" / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return repo, path


def test_archive_tarballs_land_under_the_repo_not_the_working_directory(
    tmp_path, monkeypatch, fake_runtime
):
    monkeypatch.chdir(tmp_path)  # NOT the repo root
    repo, config_path = _repo_with_config(tmp_path, _config(["alice"]))

    lifecycle.decommission_user(str(config_path), "alice", archive=True, assume_yes=True)

    assert (repo / "var" / "web_terminal_archives").is_dir()
    assert not (tmp_path / "var").exists()


def test_credential_purge_reads_the_repos_env_auth_not_the_working_directorys(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """A purge resolving ``.env.auth`` against the cwd would report success
    having cleared nothing — the departed user's hash would stay in the file the
    sidecar actually reads."""
    monkeypatch.chdir(tmp_path)  # NOT the repo root
    repo, config_path = _repo_with_config(tmp_path, _renderable_auth_config(["alice", "bob"]))
    _seed_env_auth(repo, ALICE="scrypt.alice", BOB="scrypt.bob")
    _stub_rendered_web_stack(repo)

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    surviving = parse_dotenv_file(repo / AUTH_ENV_FILENAME)
    assert f"{PW_HASH_VAR_PREFIX}ALICE" not in surviving
    assert f"{PW_HASH_VAR_PREFIX}BOB" in surviving
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_artifacts_are_rerendered_into_the_repos_build_zone(tmp_path, monkeypatch, fake_runtime):
    """The re-render follows the resolved repo, not the cwd.

    Rendered anywhere else, the compose file and nginx.conf the running stack
    reads are never updated: the removed user's route and service survive, which
    is the deployment change the verb exists to make.
    """
    monkeypatch.chdir(tmp_path)  # NOT the repo root
    repo, config_path = _repo_with_config(tmp_path, _config(["alice", "bob"]))

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert (repo / "build" / "docker-compose.web.yml").is_file()
    assert (repo / "build" / "nginx" / "nginx.conf").is_file()
    assert not (tmp_path / "build" / "docker-compose.web.yml").exists()


def test_nginx_reload_argv_addresses_the_repos_compose_file(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """The reload has to name the compose file that was just re-rendered.

    Pointed at a repo with no rendered stack, the reload is a no-op against a
    project compose does not know about, and nginx keeps serving the departed
    user's route until the next deploy.
    """
    monkeypatch.chdir(tmp_path)  # NOT the repo root
    repo, config_path = _repo_with_config(tmp_path, _renderable_auth_config(["alice", "bob"]))
    _seed_env_auth(repo, ALICE="scrypt.alice")
    _stub_rendered_web_stack(repo)

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    reloads = [
        cmd
        for cmd in _web_stack_cmds(fake_runtime)
        if cmd[-4:] == ["nginx", "nginx", "-s", "reload"]
    ]
    assert len(reloads) == 1
    compose_file = next(arg for arg in reloads[0] if "docker-compose.web.yml" in arg)
    assert compose_file == str(repo / "build" / "docker-compose.web.yml")


def test_passwd_writes_the_hash_into_the_repos_env_auth(
    tmp_path, monkeypatch, fake_runtime, recreate_calls
):
    """The hash lands in the repo's ``.env.auth`` — the file the sidecar reads.

    That file is what the rendered stack mounts, so a hash stored beside the cwd
    instead is one no deployment ever loads: the operator is told the password
    changed and the old one keeps working indefinitely.

    Scoped to the write. That the recreate then addresses the same repo — which
    is what makes the rotation immediate rather than deferred to the next
    ``osprey up`` — is pinned separately, by
    ``test_passwd_recreate_argv_addresses_the_repos_web_stack_from_any_directory``."""
    monkeypatch.chdir(tmp_path)  # NOT the repo root
    repo, config_path = _repo_with_config(tmp_path, _auth_config(["alice"]))

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    assert verify_password("alices-new-password", _read_hash(repo, "ALICE"))
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_passwd_recreate_argv_addresses_the_repos_web_stack_from_any_directory(
    tmp_path, monkeypatch, fake_runtime, auth_reconcile_runtime
):
    """The rotation is only in force once the sidecar that serves it is recreated.

    The recreate probes for a rendered stack and skips with a warning when it
    finds none, so a recreate resolving its own root from the working directory
    reported "not deployed" for a repo whose stack was rendered and running: the
    hash was rotated in the file, the sidecar kept serving the previous password,
    and ``osprey users passwd`` said the rotation had succeeded. Every part of
    the invocation is asserted, because each one independently decides which
    deployment is acted on: ``-f`` names the compose file, ``--project-directory``
    is what every relative path inside it (including the sidecar's
    ``env_file: .env.auth``) resolves against, and ``--env-file`` supplies the
    values compose substitutes.
    """
    monkeypatch.chdir(tmp_path)  # NOT the repo root
    repo, config_path = _repo_with_config(tmp_path, _renderable_auth_config(["alice"]))
    _seed_env_auth(repo, ALICE="scrypt.alice")
    _stub_rendered_web_stack(repo)
    (repo / ".env").write_text("", encoding="utf-8")

    lifecycle.rotate_user_password(str(config_path), "alice", "alices-new-password")

    recreates = [
        cmd
        for cmd in _web_stack_cmds(fake_runtime)
        if cmd[-4:] == ["up", "-d", "--force-recreate", AUTH_SERVICE_NAME]
    ]
    assert len(recreates) == 1
    argv = recreates[0]
    assert argv[argv.index("-f") + 1] == str(repo / "build" / "docker-compose.web.yml")
    assert argv[argv.index("--project-directory") + 1] == str(repo)
    assert argv[argv.index("--env-file") + 1] == str(repo / ".env")
    # The pre-recreate re-render follows the same repo: it exists to refresh the
    # digest label on the compose file this argv names, so a render landing in
    # the cwd would leave that file carrying the label of the pre-rotation
    # `.env.auth` — and a byte-exact revert would then be skipped as unchanged.
    assert (repo / "build" / "nginx" / "nginx.conf").is_file()
    assert not (tmp_path / "build").exists()


# =============================================================================
# decommission_user + the Bash/launch-token conflict guard
#
# The guard also runs at the render seam this verb calls, so a conflicted
# deployment was always refused. What these pin is WHERE: refusing at the
# re-render would leave config.yml already rewritten, the artifacts stale, and
# the container/volume removal below never run. The verb must refuse before it
# touches the roster — and must NOT lose the escape hatch that lets an operator
# decommission the offending persona's own user.
# =============================================================================


def _conflicted_persona_config(tmp_path, users, *, personas):
    """A roster whose persona projects are really on disk under *tmp_path*.

    `personas` maps a persona name to `(writes_enabled, denies_bash)`, so a case
    can pair the launch-token entitlement with either shipped permission state.
    """
    import json as _json

    catalog = {}
    for name, (writes, denies_bash) in personas.items():
        project_dir = tmp_path / "profiles" / name
        (project_dir / ".claude").mkdir(parents=True)
        (project_dir / "config.yml").write_text(
            yaml.safe_dump({"project_name": name, "control_system": {"writes_enabled": writes}}),
            encoding="utf-8",
        )
        (project_dir / ".claude" / "settings.json").write_text(
            _json.dumps({"permissions": {"deny": ["Bash"] if denies_bash else []}}),
            encoding="utf-8",
        )
        catalog[name] = {"project": name, "project_path": f"profiles/{name}"}
    return _config(users, personas=catalog)


_TIERED_PERSONAS = {"armed": (True, False), "safe": (False, True)}
_TIERED_USERS = [
    {"name": "alice", "index": 0, "persona": "armed"},
    {"name": "bob", "index": 1, "persona": "safe"},
]


def test_decommission_refuses_before_touching_the_roster_when_a_survivor_is_conflicted(
    tmp_path, monkeypatch, fake_runtime
):
    """Removing bob while alice's persona is conflicted must abort cleanly. The
    roster edit is precisely what makes a failed decommission half-applied, so
    the refusal has to land before it — not at the re-render that follows."""
    monkeypatch.chdir(tmp_path)
    config = _conflicted_persona_config(tmp_path, _TIERED_USERS, personas=_TIERED_PERSONAS)
    config_path = _write_config(tmp_path, config)

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        lifecycle.decommission_user(str(config_path), "bob", assume_yes=True)

    assert "armed" in str(excinfo.value)
    # config.yml untouched: both users still on the roster.
    assert [entry["name"] for entry in _reload_users(config_path)] == ["alice", "bob"]
    # And nothing downstream ran.
    assert [c for c in fake_runtime if c[1] == "rm"] == []


def test_decommission_of_the_conflicted_personas_own_user_still_succeeds(
    tmp_path, monkeypatch, fake_runtime
):
    """THE ESCAPE HATCH. Moving the check earlier must not cost the operator the
    one remediation that needs no image rebuild: removing the conflicted
    persona's last user. The probed roster is the POST-removal one, so that
    persona stops being referenced and the conflict dissolves."""
    monkeypatch.chdir(tmp_path)
    config = _conflicted_persona_config(tmp_path, _TIERED_USERS, personas=_TIERED_PERSONAS)
    config_path = _write_config(tmp_path, config)

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert [entry["name"] for entry in _reload_users(config_path)] == ["bob"]
    assert [c for c in fake_runtime if c[1] == "rm"] == [["docker", "rm", "-f", "dls-web-alice"]]

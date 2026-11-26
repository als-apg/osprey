"""Every command that reads or writes a deployment's source zone refuses a retired root layout.

A deployment repo still holding an entry at a retired root path is stopped at
the command's entry: before any profile is read or validated and before
anything is written. Each command prints the refusal block once, exits 1 and
leaves the tree byte-identical, the state zone and the env files included.

The ``rules`` row lives at ``agent/rules`` and has moved from ``rules``; each
test moves an initialised deployment's ``agent/rules`` back to ``rules``, the
tree an unmigrated deployment holds. The dissolved ``data/`` tree is planted
the five ways an unmigrated deployment still holds it: a source at its old
``data/...`` path, a ``data:`` line, a ``data:`` naming a directory outside the
repo, a file of the operator's own left in ``data/``, and a lane's
``data/<lane>_curve`` directory. The profile also gains a key no profile
accepts, so a command that read or validated the profile before refusing would
fail on that key instead.

An inventory walks the whole click tree: every command is either proved to
refuse here or named in an exempt table with its reason, so a command added
later cannot skip the refusal unnoticed.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey import repo_layout
from osprey.cli.main import cli
from osprey.repo_retired import RetiredRootRefused, refuse_retired_layout
from tests._builds import init_project

pytestmark = [pytest.mark.slow]

#: The project name, and the repo's directory name.
PROJECT = "demo"

#: The declared move: the row's key, its current path and its retired path.
MOVED_KEY = "rules"
MOVED_TO = "agent/rules"
MOVED_FROM = "rules"

#: A top-level key no profile accepts: validation would stop on it.
UNVALIDATABLE_KEY = "not_a_profile_key"


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo deployment as `osprey init` writes it."""
    return init_project(tmp_path_factory.mktemp("retired-root"), "demo", PROJECT)


@pytest.fixture
def planted() -> None:
    """The live layout declares the ``rules`` move these tests rely on."""
    row = repo_layout.entry(MOVED_KEY)
    assert (row.path, row.moved_from) == (MOVED_TO, (MOVED_FROM,))


@pytest.fixture
def repo(initialised: Path, tmp_path: Path, planted: None) -> Path:  # noqa: ARG001
    """A fresh copy of the deployment, with a marking ``pre_build`` step and an unknown key."""
    copy = tmp_path / PROJECT
    shutil.copytree(initialised, copy, symlinks=True)
    (copy / MOVED_TO).rename(copy / MOVED_FROM)
    group = (copy / MOVED_TO).parent
    if not any(group.iterdir()):
        group.rmdir()
    _mark(copy)
    return copy


def _plant_old_data_path(repo: Path) -> None:
    (repo / "data").mkdir()
    (repo / "facility").rename(repo / "data" / "facility")


def _plant_data_line(repo: Path) -> None:
    _append_profile_line(repo, "data: data")


def _plant_out_of_repo_data_line(repo: Path) -> None:
    outside = repo.parent / "shared-data"
    outside.mkdir()
    (outside / "README.md").write_text("shared\n", encoding="utf-8")
    _append_profile_line(repo, "data: ../shared-data")


def _plant_leftover_file(repo: Path) -> None:
    (repo / "data").mkdir()
    (repo / "data" / "notes.md").write_text("mine\n", encoding="utf-8")


def _plant_lane_curve_dir(repo: Path) -> None:
    curve = repo / "data" / "bluesky_curve" / "bridge"
    curve.mkdir(parents=True)
    (curve / "proxy.key_secret").write_text("secret\n", encoding="utf-8")


#: Each way a deployment still holds the dissolved ``data/`` tree, with what
#: its refusal names.
DATA_PLANTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "old-data-path": (_plant_old_data_path, "data/facility"),
    "data-line": (_plant_data_line, "data: data"),
    "out-of-repo-data-line": (_plant_out_of_repo_data_line, "data: ../shared-data"),
    "leftover-file": (_plant_leftover_file, "data/notes.md"),
    "lane-curve-dir": (_plant_lane_curve_dir, "data/bluesky_curve"),
}


def _append_profile_line(repo: Path, line: str) -> None:
    profile = repo / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8") + f"{line}\n", encoding="utf-8")


@pytest.fixture(params=list(DATA_PLANTS))
def data_plant(request: pytest.FixtureRequest) -> str:
    """The name of one way the dissolved ``data/`` tree is still held."""
    return request.param


@pytest.fixture
def data_repo(initialised: Path, tmp_path: Path, data_plant: str) -> Path:
    """A fresh copy of the deployment holding *data_plant*, marked like :func:`repo`."""
    copy = tmp_path / PROJECT
    shutil.copytree(initialised, copy, symlinks=True)
    assert not (copy / "data").exists()
    _mark(copy)
    plant, _named = DATA_PLANTS[data_plant]
    plant(copy)
    return copy


def _mark(copy: Path) -> None:
    """Add a ``pre_build`` step that leaves a marker, and a key no profile accepts."""
    profile = copy / "profile.yml"
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    steps = data.setdefault("lifecycle", {}).setdefault("pre_build", [])
    steps.insert(0, {"name": "mark", "run": f"touch {_marker(copy)}"})
    data[UNVALIDATABLE_KEY] = True
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _marker(repo: Path) -> Path:
    return repo.parent / "pre_build_ran"


def _snapshot(repo: Path) -> dict[str, bytes | None]:
    """Every path in the deployment, the state zone and dot-files included; files with bytes."""
    return {
        path.relative_to(repo).as_posix(): (
            path.read_bytes() if path.is_file() and not path.is_symlink() else None
        )
        for path in sorted(repo.rglob("*"))
    }


def _expected_message(repo: Path) -> str:
    with pytest.raises(RetiredRootRefused) as refusal:
        refuse_retired_layout(repo)
    return refusal.value.format_message()


def _assert_refused(repo: Path, args: list[str]) -> None:
    expected = _expected_message(repo)
    before = _snapshot(repo)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert result.output.count(expected) == 1, result.output
    assert "Unexpected error" not in result.output
    assert UNVALIDATABLE_KEY not in result.output
    assert _snapshot(repo) == before
    assert not _marker(repo).exists()


def _tier_file(repo: Path) -> Path:
    return sorted((repo / "tiers").glob("*.yml"))[0]


#: Each refusing command, by its click path, with the arguments that run it.
#: ``{repo}`` stands for the deployment, ``{tier}`` for one of its tier files,
#: ``{export}`` for an input file and ``{out}`` for an output directory outside it.
REFUSING: dict[str, list[list[str]]] = {
    "build": [["build", "--repo", "{repo}"]],
    "up": [
        ["up", "-d", "--repo", "{repo}"],
        ["up", "-d", "--as-built", "--repo", "{repo}"],
        ["up", "-d", "--build", "--repo", "{repo}"],
    ],
    "restart": [
        ["restart", "-d", "--repo", "{repo}"],
        ["restart", "-d", "--as-built", "--repo", "{repo}"],
    ],
    "init": [["init", "{repo}", "--preset", "demo", "--force"]],
    "validate": [
        ["validate", "--repo", "{repo}"],
        ["validate", "{repo}"],
        ["validate", "{repo}/profile.yml"],
        ["validate", "{tier}"],
    ],
    "profile validate": [["profile", "validate", "{repo}/profile.yml"]],
    "profile card": [["profile", "card", "--repo", "{repo}"]],
    "profile panels": [["profile", "panels", "--repo", "{repo}"]],
    "facility validate": [["facility", "validate", "--repo", "{repo}"]],
    "facility show": [["facility", "show", "--repo", "{repo}"]],
    "facility import mml": [["facility", "import", "mml", "{export}", "--repo", "{repo}"]],
    "facility import list": [["facility", "import", "list", "{export}", "--repo", "{repo}"]],
    "facility import relations": [
        ["facility", "import", "relations", "{export}", "--layer", "cables", "--repo", "{repo}"]
    ],
    "scaffold ci": [["scaffold", "ci", "--force", "--repo", "{repo}"]],
    "scaffold systemd": [["scaffold", "systemd", "--force", "--repo", "{repo}"]],
    "scaffold tiers": [["scaffold", "tiers", "--force", "--repo", "{repo}"]],
    "scaffold pull": [["scaffold", "pull", "demo:data/facility/knowledge", "--repo", "{repo}"]],
    "scaffold list": [["scaffold", "list", "--repo", "{repo}"]],
    "scaffold claim": [["scaffold", "claim", "rules", "--repo", "{repo}"]],
    "scaffold diff": [["scaffold", "diff", "rules", "--repo", "{repo}"]],
    "scaffold unclaim": [["scaffold", "unclaim", "rules", "--repo", "{repo}"]],
    "scaffold web-terminals lint": [["scaffold", "web-terminals", "lint", "--repo", "{repo}"]],
    "scaffold web-terminals render": [
        ["scaffold", "web-terminals", "render", "-o", "{out}", "--repo", "{repo}"]
    ],
    "app new": [["app", "new", "beam-tool", "--repo", "{repo}"]],
    "app check": [["app", "check", "--repo", "{repo}"]],
    "users list": [["users", "list", "--repo", "{repo}"]],
    "users remove": [["users", "remove", "alice", "-y", "--repo", "{repo}"]],
    "users prune": [["users", "prune", "-y", "--repo", "{repo}"]],
    "users seed": [["users", "seed", "--repo", "{repo}"]],
    "users login-url": [["users", "login-url", "alice", "--repo", "{repo}"]],
    "users passwd": [["users", "passwd", "alice", "--repo", "{repo}"]],
    "users env": [["users", "env", "--repo", "{repo}"]],
    "set": [["set", "model=claude-opus-5", "--repo", "{repo}"]],
}

#: Every other command that runs, with why it does not refuse.
EXEMPT: dict[str, str] = {
    # Usable on an unmigrated tree by design: they read the running stack.
    "down": "stops the running stack; usable on an unmigrated tree",
    "status": "reports the running stack; usable on an unmigrated tree",
    "logs": "follows the running stack's logs; usable on an unmigrated tree",
    "reset": "tears the deployment down and deletes runtime state, like down",
    # Read the running deployment or its build zone, never the source zone.
    "chat": "talks to the running agent",
    "query": "queries the running agent",
    "config": "prints the profile as it stands on disk, reading no path in it",
    "agent-record export": "reads the running agent record",
    "agent-record health": "reads the running agent record",
    "sim apply": "drives the running simulator",
    "sim list": "lists the running simulator's scenarios",
    "sim status": "reports the running simulator",
    "web sessions clear": "clears the running web terminal's sessions",
    "web stop": "stops a running web terminal",
    "feedback export": "reads the running deployment's feedback store",
    "feedback list": "reads the running deployment's feedback store",
    "write-authorization approve": "acts on the running deployment's grants",
    "write-authorization list": "reads the running deployment's grants",
    "write-authorization revoke": "acts on the running deployment's grants",
    "knowledge seed-from-ttl": "reads the build's graph and writes the bundle it is given",
    # Resolve no deployment repo.
    "web": "launches a web terminal over the directory it is given",
    "health": "checks the host and the services it finds",
    "theme-lab": "runs without a deployment",
    "archive": "reads an archiver it is pointed at",
    "history": "reads an archiver it is pointed at",
    "audit": "reads an audit log it is pointed at",
    "profile artifacts": "lists bundled artifacts",
    "profile presets": "lists bundled presets",
    "agent-record serve": "serves the agent record it is configured with",
    "artifacts web": "serves the artifact gallery",
    "watchdog replay": "replays a recording it is given",
    "scaffold apps": "a retired verb that only says what replaced it",
    "vendor fetch": "fills the vendor cache",
    "vendor verify": "checks the vendor cache",
    "knowledge regen-index": "rewrites the bundle it is given",
    "knowledge validate": "checks the bundle it is given",
    # Groups that only print their help when run bare.
    "scaffold": "prints help",
    "scaffold web-terminals": "prints help",
}

#: Command groups whose own name runs no command: each subcommand is listed.
_GROUP_PREFIXES = ("channel-finder", "ariel")


def _commands() -> list[str]:
    """Every command path click runs: leaves, and groups that run without a subcommand."""
    found: list[str] = []

    def walk(group: click.Group, prefix: tuple[str, ...]) -> None:
        ctx = click.Context(group, info_name=prefix[-1] if prefix else "osprey")
        for name in group.list_commands(ctx):
            command = group.get_command(ctx, name)
            if command is None:
                continue
            path = (*prefix, name)
            if isinstance(command, click.Group):
                if command.invoke_without_command:
                    found.append(" ".join(path))
                walk(command, path)
            else:
                found.append(" ".join(path))

    walk(cli, ())
    return found


def test_every_command_refuses_or_is_exempt_with_a_reason() -> None:
    commands = {command for command in _commands() if not command.startswith(_GROUP_PREFIXES)}
    classified = set(REFUSING) | set(EXEMPT)
    assert not set(REFUSING) & set(EXEMPT)
    assert commands - classified == set(), "classify each new command: refuse, or exempt why"
    assert classified - commands == set(), "a classified command no longer exists"
    assert all(EXEMPT.values())


def test_the_plant_is_detected(repo: Path) -> None:
    assert (repo / MOVED_FROM).is_dir()
    message = _expected_message(repo)
    assert f"{MOVED_FROM} " in message and MOVED_TO in message


_CASES = [
    pytest.param(args, id=" ".join(args[:4]))
    for invocations in REFUSING.values()
    for args in invocations
]


def _invocation(repo: Path, tmp_path: Path, args: list[str]) -> list[str]:
    export = tmp_path / "export.csv"
    export.write_text("address\nSR:NEW:CHANNEL\n", encoding="utf-8")
    values = {
        "repo": str(repo),
        "tier": str(_tier_file(repo)),
        "export": str(export),
        "out": str(tmp_path / "render"),
    }
    return [arg.format(**values) for arg in args]


@pytest.mark.parametrize("args", _CASES)
def test_command_refuses_before_reading_or_writing(
    repo: Path, tmp_path: Path, args: list[str]
) -> None:
    _assert_refused(repo, _invocation(repo, tmp_path, args))


def test_each_data_plant_is_detected(data_repo: Path, data_plant: str) -> None:
    _plant, named = DATA_PLANTS[data_plant]
    assert named in _expected_message(data_repo)


@pytest.mark.parametrize("args", _CASES)
def test_command_refuses_a_held_data_tree_before_reading_or_writing(
    data_repo: Path, tmp_path: Path, args: list[str]
) -> None:
    _assert_refused(data_repo, _invocation(data_repo, tmp_path, args))


def test_build_refuses_before_the_state_zone(repo: Path) -> None:
    shutil.rmtree(repo / "var")
    _assert_refused(repo, ["build", "--repo", str(repo)])
    assert not (repo / "var").exists()
    assert not (repo / "build").exists()


def test_init_without_force_is_not_refused_for_the_layout(repo: Path) -> None:
    result = CliRunner().invoke(cli, ["init", str(repo), "--preset", "demo"])
    assert result.exit_code != 0
    assert _expected_message(repo) not in result.output


def test_scaffold_pull_list_does_not_refuse(repo: Path) -> None:
    result = CliRunner().invoke(cli, ["scaffold", "pull", "demo", "--list", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "retired paths" not in result.output


@pytest.mark.usefixtures("planted")
def test_a_file_outside_any_deployment_is_validated_without_the_root_check(
    tmp_path: Path,
) -> None:
    # A directory holding the retired path but no profile.yml is no deployment.
    (tmp_path / MOVED_FROM).mkdir()
    login = tmp_path / "standalone.yml"
    login.write_text("name: standalone\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["validate", str(login)])
    assert "retired paths" not in result.output


#: A lane's certificate set as a deployment minted it under the data tree.
_UNMIGRATED_CERTS = (
    "data/.runtime/bluesky_curve/bridge/proxy.key_secret",
    "data/.runtime/bluesky_curve/bridge/clients/publisher.key",
    "data/.runtime/bluesky_curve/queueserver/publisher.key_secret",
    "data/.runtime/bluesky_curve/queueserver/proxy.key",
)


@pytest.mark.parametrize(
    "args",
    [["up", "-d"], ["up", "-d", "--as-built"], ["restart", "-d"]],
    ids=" ".join,
)
def test_a_start_on_unmigrated_certificates_mints_nothing(
    initialised: Path, tmp_path: Path, args: list[str]
) -> None:
    """Certificates still under ``data/.runtime`` stop a start before it mints a second set."""
    from osprey.utils.workspace import RUNTIME_DIR_RELPATH

    copy = tmp_path / PROJECT
    shutil.copytree(initialised, copy, symlinks=True)
    for rel in _UNMIGRATED_CERTS:
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        (copy / rel).write_bytes(f"minted {rel}\n".encode())
    assert [(move.old, move.new) for move in _refusal_moves(copy)] == [
        ("data/.runtime", "var/runtime")
    ]

    _assert_refused(copy, [*args, "--repo", str(copy)])
    assert not (copy / RUNTIME_DIR_RELPATH).exists()


def _refusal_moves(repo: Path) -> tuple:
    with pytest.raises(RetiredRootRefused) as refusal:
        refuse_retired_layout(repo)
    return refusal.value.layout.moves

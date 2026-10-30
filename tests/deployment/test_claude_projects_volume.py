"""Claude Code transcripts on a named volume of their own.

Each web terminal mounts ``<user>-claude-projects`` at
``/data/claude-config/projects``, nested over its claude-config volume; each
dispatch worker mounts ``dispatch_claude_projects[_<i>]`` at
``<agent-data dir>/claude-config/projects``, nested in its workspace. What is
load-bearing:

**The nesting order.** A nested mount is made inside its parent, so the
transcripts volume is listed after the parent in every render.

**The archive mounts no transcripts.** Transcripts reach the agent record,
never the archive service: it mounts neither a transcripts volume nor a
claude-config volume.

**The volume is part of the user.** Removing a user, the orphan sweep and reset
all see it, because it carries the labels compose stamps.

**Existing transcripts move once.** ``osprey up`` copies a parent's transcripts
into a transcripts volume that does not exist yet, before any compose
invocation could create it empty, and never again after that.
"""

from __future__ import annotations

import copy
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_runner.claude_state import (
    CLAUDE_CONFIG_VOLUME_SUFFIX,
    CLAUDE_PROJECTS_VOLUME_SUFFIX,
    dispatch_claude_projects_volume,
)
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    _inject_project_metadata,
    claude_projects_volumes,
    resolve_user_projects_volume_name,
)
from osprey.deployment.reset import RuntimeProbe
from osprey.deployment.runtime_helper import (
    TRANSCRIPTS_COPY_IMAGE,
    TranscriptsCopyError,
    copy_transcripts_once,
    transcripts_copy_image,
)
from osprey.deployment.web_terminals import lifecycle
from osprey.deployment.web_terminals.render import render_web_terminals
from tests.deployment.test_compose_generator import _packaged_compose_template
from tests.deployment.test_container_lifecycle import _fake_popen
from tests.deployment.test_up_as_built import (  # noqa: F401 - fixtures, taken by name
    lifecycle_repo,
    render_build,
    run_up,
    started,
)
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG
from tests.deployment.web_terminals.test_lifecycle import _config as _lifecycle_config
from tests.deployment.web_terminals.test_lifecycle import (  # noqa: F401 - fixtures, taken by name
    _write_config,
    fake_runtime,
    fake_runtime_prune,
)

PROJECT = "proj"


# ── Renders ─────────────────────────────────────────────────────────────────


def _context(tmp_path: Path, *, mode: str = "isolated", count: int = 2) -> tuple[dict, dict]:
    config = copy.deepcopy(EXAMPLE_CONFIG)
    config.update(
        {
            "project_name": "dls",
            "project_root": str(tmp_path),
            "system": {"timezone": "UTC"},
            "deployment": {},
            "services": {
                "archive": {},
                "dispatch_worker": {"worker_count": count, "workspace_mode": mode},
            },
            "deployed_services": ["archive", "dispatch_worker"],
            "control_system": {},
        }
    )
    return config, _inject_project_metadata(config)


def _render(context: dict[str, Any], rel_path: str) -> dict[str, Any]:
    return yaml.safe_load(_packaged_compose_template(f"services/{rel_path}").render(**context))


def test_each_terminal_mounts_its_transcripts_volume_after_its_claude_config(tmp_path):
    config, _context_unused = _context(tmp_path)
    web = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    users = [key.removeprefix("web-") for key in web["services"] if key.startswith("web-")]
    assert users
    for user in users:
        mounts = web["services"][f"web-{user}"]["volumes"]
        parent = mounts.index(f"{user}{CLAUDE_CONFIG_VOLUME_SUFFIX}:/data/claude-config")
        nested = mounts.index(f"{user}{CLAUDE_PROJECTS_VOLUME_SUFFIX}:/data/claude-config/projects")
        assert nested == parent + 1
        assert f"{user}{CLAUDE_PROJECTS_VOLUME_SUFFIX}" in web["volumes"]


@pytest.mark.parametrize(
    ("mode", "count", "expected"),
    [
        ("isolated", 2, [("dispatch_workspace_1", 1), ("dispatch_workspace_2", 2)]),
        ("shared", 2, [("dispatch_workspace", None), ("dispatch_workspace", None)]),
    ],
)
def test_each_dispatch_worker_mounts_its_transcripts_volume_after_its_workspace(
    tmp_path, mode, count, expected
):
    _config_unused, context = _context(tmp_path, mode=mode, count=count)
    doc = _render(context, "dispatch_worker/docker-compose.yml.j2")
    agent_data_dir = context["osprey_container_agent_data_dir"]

    workers = [key for key in doc["services"] if key.startswith("dispatch-worker-")]
    assert len(workers) == count
    for service, (workspace, index) in zip(workers, expected, strict=True):
        mounts = doc["services"][service]["volumes"]
        projects = dispatch_claude_projects_volume(index)
        parent = mounts.index(f"{workspace}:{agent_data_dir}")
        assert mounts[parent + 1] == f"{projects}:{agent_data_dir}/claude-config/projects"
        assert projects in doc["volumes"]


def test_the_dispatch_mount_is_where_the_runner_points_claude_code(tmp_path):
    from osprey.agent_runner.artifact_resolve import DISPATCH_CLAUDE_CONFIG_DIRNAME

    _config_unused, context = _context(tmp_path, count=1)
    doc = _render(context, "dispatch_worker/docker-compose.yml.j2")
    (mount,) = [
        m
        for m in doc["services"]["dispatch-worker-1"]["volumes"]
        if isinstance(m, str) and m.startswith("dispatch_claude_projects_1:")
    ]
    target = mount.split(":", 1)[1]

    assert (
        target
        == f"{context['osprey_container_agent_data_dir']}/{DISPATCH_CLAUDE_CONFIG_DIRNAME}/projects"
    )


def test_the_archive_mounts_no_transcripts_and_no_claude_config(tmp_path):
    _config_unused, context = _context(tmp_path)
    doc = _render(context, "archive/docker-compose.yml.j2")
    mounts = doc["services"]["archive"]["volumes"]
    sources = [m.split(":", 1)[0] for m in mounts]

    assert not [v for v in sources if "claude-config" in v]
    assert not [v for v in sources if "claude-projects" in v or "claude_projects" in v]
    # The workspace is still mounted, for the artifacts.
    assert "dispatch_workspace_1:/archive/sources/dispatch/dispatch-worker-1:ro" in mounts


# ── User removal and the orphan sweep ───────────────────────────────────────


def test_purging_a_user_removes_their_transcripts_volume_with_its_parent(
    tmp_path,
    monkeypatch,
    fake_runtime,  # noqa: F811 - the imported fixture, by name
):
    monkeypatch.chdir(tmp_path)
    config = _lifecycle_config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    removed = [c[3] for c in fake_runtime if c[1:3] == ["volume", "rm"]]
    assert removed == [
        "demo-project_alice-claude-config",
        "demo-project_alice-claude-projects",
        "demo-project_alice-agent-data",
    ]


def test_the_orphan_sweep_includes_the_transcripts_volume(
    tmp_path,
    monkeypatch,
    fake_runtime_prune,  # noqa: F811 - the imported fixture, by name
):
    monkeypatch.chdir(tmp_path)
    calls, listing = fake_runtime_prune
    listing["volumes"] = [
        "demo-project_eve-claude-config",
        "demo-project_eve-claude-projects",
        "demo-project_eve-agent-data",
        "demo-project_alice-claude-projects",
    ]

    orphans = lifecycle._discover_orphan_volumes("docker", "demo-project", {"alice"})

    assert orphans == {
        "eve": [
            "demo-project_eve-claude-config",
            "demo-project_eve-claude-projects",
            "demo-project_eve-agent-data",
        ]
    }


def test_the_user_volume_name_is_namespaced_like_its_siblings():
    assert resolve_user_projects_volume_name({"project_name": "p"}, "alice") == (
        "p_alice-claude-projects"
    )


# ── The one-time copy ───────────────────────────────────────────────────────


class FakeRuntime:
    """A runtime that holds volumes with labels and containers that run.

    Answers the runtime calls the copy step, reset's probe and the orphan sweep
    make, so a volume the copy creates is what the readers then list.
    """

    def __init__(self, runtime: str = "docker") -> None:
        self.runtime = runtime
        self.volumes: dict[str, dict[str, str]] = {}
        self.running: set[str] = set()
        self.images: set[str] = set()
        self.calls: list[list[str]] = []
        self.fail_copy = False
        self.fail_pull = False

    def __call__(self, argv, **_kwargs) -> subprocess.CompletedProcess:
        argv = list(argv)
        self.calls.append(argv)
        rc, out = self._answer(argv[1:])
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="" if rc == 0 else "boom")

    def _answer(self, args: list[str]) -> tuple[int, str]:
        match args:
            case ["volume", "inspect", name, "--format", _]:
                return (0, json.dumps(self.volumes[name])) if name in self.volumes else (1, "")
            case ["volume", "inspect", name]:
                return (0, "[]") if name in self.volumes else (1, "")
            case ["volume", "create", *rest]:
                labels = dict(
                    rest[i + 1].split("=", 1) for i, a in enumerate(rest) if a == "--label"
                )
                self.volumes[rest[-1]] = labels
                return 0, rest[-1]
            case ["volume", "rm", *rest]:
                self.volumes.pop(rest[-1], None)
                return 0, ""
            case ["volume", "ls", "--filter", selector, "--format", _]:
                key, value = selector.removeprefix("label=").split("=", 1)
                return 0, "\n".join(
                    n for n, labels in self.volumes.items() if labels.get(key) == value
                )
            case ["container", "inspect", "-f", _, name]:
                return 0, "true\n" if name in self.running else "false\n"
            case ["stop", name]:
                self.running.discard(name)
                return 0, ""
            case ["image", "inspect", image]:
                return (0, "[]") if image in self.images else (1, "")
            case ["pull", image]:
                if self.fail_pull:
                    return 1, ""
                self.images.add(image)
                return 0, ""
            case ["run", *_]:
                return (1, "") if self.fail_copy else (0, "")
        return 0, ""

    def copies(self) -> list[list[str]]:
        return [c for c in self.calls if c[1] == "run"]


def _deploy_config(*, users=("alice",), workers: int = 0, mode: str = "isolated", **extra) -> dict:
    config: dict[str, Any] = {"project_name": PROJECT, "deployed_services": [], **extra}
    if users:
        config["modules"] = {"web_terminals": {"enabled": True, "users": list(users)}}
    if workers:
        config["deployed_services"] = ["dispatch_worker"]
        config["services"] = {"dispatch_worker": {"worker_count": workers, "workspace_mode": mode}}
    return config


def _copy(config: dict, fake: FakeRuntime) -> list[str]:
    return copy_transcripts_once(config, runtime=fake.runtime, run=fake)


def test_the_pairs_follow_the_roster_and_the_dispatch_workspaces():
    config = _deploy_config(users=("alice",), workers=2)

    assert [
        (p["volume"], p["parent"], p["parent_dir"], p["containers"])
        for p in claude_projects_volumes(config)
    ] == [
        ("alice-claude-projects", "alice-claude-config", "projects", ["proj-web-alice"]),
        (
            "dispatch_claude_projects_1",
            "dispatch_workspace_1",
            "claude-config/projects",
            ["proj-dispatch-worker-1"],
        ),
        (
            "dispatch_claude_projects_2",
            "dispatch_workspace_2",
            "claude-config/projects",
            ["proj-dispatch-worker-2"],
        ),
    ]
    shared = claude_projects_volumes(_deploy_config(users=(), workers=2, mode="shared"))
    assert [(p["volume"], p["containers"]) for p in shared] == [
        ("dispatch_claude_projects", ["proj-dispatch-worker-1", "proj-dispatch-worker-2"])
    ]


def test_it_copies_only_when_the_target_is_absent_and_a_second_up_copies_nothing():
    fake = FakeRuntime()
    fake.volumes["proj_alice-claude-config"] = {}
    fake.volumes["proj_dispatch_workspace_1"] = {}
    fake.volumes["proj_dispatch_claude_projects_2"] = {"already": "there"}
    fake.volumes["proj_dispatch_workspace_2"] = {}
    config = _deploy_config(workers=2)

    created = _copy(config, fake)

    assert created == ["proj_alice-claude-projects", "proj_dispatch_claude_projects_1"]
    sources = [c[c.index("--mount") + 1] for c in fake.copies()]
    assert sources == [
        "type=volume,source=proj_alice-claude-config,destination=/from,readonly",
        "type=volume,source=proj_dispatch_workspace_1,destination=/from,readonly",
    ]
    assert [c[-1] for c in fake.copies()] == ["projects", "claude-config/projects"]

    fake.calls.clear()
    assert _copy(config, fake) == []
    assert fake.copies() == []
    assert not [c for c in fake.calls if c[1:3] == ["volume", "create"]]


def test_a_missing_parent_is_left_alone():
    fake = FakeRuntime()

    assert _copy(_deploy_config(), fake) == []
    assert fake.volumes == {}


def test_a_running_owner_is_stopped_before_the_copy():
    fake = FakeRuntime()
    fake.volumes["proj_alice-claude-config"] = {}
    fake.running.add("proj-web-alice")

    _copy(_deploy_config(), fake)

    stop = fake.calls.index(["docker", "stop", "proj-web-alice"])
    assert stop < fake.calls.index(fake.copies()[0])
    assert "proj-web-alice" not in fake.running


def test_a_failed_copy_removes_the_target_and_names_it():
    fake = FakeRuntime()
    fake.volumes["proj_alice-claude-config"] = {}
    fake.fail_copy = True

    with pytest.raises(TranscriptsCopyError, match="proj_alice-claude-projects"):
        _copy(_deploy_config(), fake)

    assert "proj_alice-claude-projects" not in fake.volumes
    # The next `up` therefore copies again.
    fake.fail_copy = False
    assert _copy(_deploy_config(), fake) == ["proj_alice-claude-projects"]


def test_a_failed_pull_refuses_naming_the_image():
    fake = FakeRuntime()
    fake.volumes["proj_alice-claude-config"] = {}
    fake.fail_pull = True

    with pytest.raises(TranscriptsCopyError, match=TRANSCRIPTS_COPY_IMAGE):
        _copy(_deploy_config(), fake)
    assert "proj_alice-claude-projects" not in fake.volumes


def test_a_missing_source_directory_makes_an_empty_labeled_volume():
    """The copy is guarded inside the container: a parent without the directory
    copies nothing and succeeds, and the volume exists with compose's labels."""
    fake = FakeRuntime()
    fake.volumes["proj_alice-claude-config"] = {}

    _copy(_deploy_config(), fake)

    (copy_argv,) = fake.copies()
    script = copy_argv[copy_argv.index("-c") + 1]
    assert script.startswith('if [ -d "/from/$1" ]; then cp -a ')
    assert fake.volumes["proj_alice-claude-projects"] == {
        "com.docker.compose.project": PROJECT,
        "com.docker.compose.volume": "alice-claude-projects",
    }


def test_the_copy_runs_the_pinned_image_as_root_without_a_network():
    fake = FakeRuntime()
    fake.volumes["proj_alice-claude-config"] = {}

    _copy(_deploy_config(), fake)

    (copy_argv,) = fake.copies()
    assert TRANSCRIPTS_COPY_IMAGE.startswith("docker.io/library/alpine:")
    assert TRANSCRIPTS_COPY_IMAGE.rsplit(":", 1)[1] not in {"", "latest"}
    assert TRANSCRIPTS_COPY_IMAGE in copy_argv
    assert copy_argv[copy_argv.index("--user") + 1] == "0:0"
    assert copy_argv[copy_argv.index("--network") + 1] == "none"
    assert ["docker", "pull", TRANSCRIPTS_COPY_IMAGE] in fake.calls


def test_the_image_has_its_own_override_and_ignores_the_osprey_registry():
    mirror = "mirror.example.org/hub/library/alpine:3.20"

    assert transcripts_copy_image({"OSPREY_TRANSCRIPTS_COPY_IMAGE": mirror}) == mirror
    assert transcripts_copy_image({"OSPREY_TRANSCRIPTS_COPY_IMAGE": "  "}) == (
        TRANSCRIPTS_COPY_IMAGE
    )
    assert transcripts_copy_image({}) == TRANSCRIPTS_COPY_IMAGE
    # The registry OSPREY's own images live in is not a mirror of library images.
    assert (
        transcripts_copy_image({"OSPREY_IMAGE_REGISTRY": "ghcr.io/als-apg"})
        == TRANSCRIPTS_COPY_IMAGE
    )


def test_the_override_reaches_the_pull_and_the_copy():
    mirror = "mirror.example.org/hub/library/alpine:3.20"
    fake = FakeRuntime("docker")
    fake.volumes["proj_alice-claude-config"] = {}

    copy_transcripts_once(
        _deploy_config(images={"registry": "ghcr.io/als-apg"}),
        runtime=fake.runtime,
        run=fake,
        env={"OSPREY_TRANSCRIPTS_COPY_IMAGE": mirror},
    )

    assert ["docker", "pull", mirror] in fake.calls
    assert any(argv[:2] == ["docker", "run"] and mirror in argv for argv in fake.calls)


def test_the_podman_form_stamps_the_podman_compose_project():
    fake = FakeRuntime("podman")
    fake.volumes["proj_alice-claude-config"] = {}

    _copy(_deploy_config(), fake)

    assert fake.volumes["proj_alice-claude-projects"]["io.podman.compose.project"] == PROJECT
    assert fake.volumes["proj_alice-claude-projects"]["com.docker.compose.project"] == PROJECT
    assert fake.copies()[0][0] == "podman"


def test_reset_and_the_orphan_sweep_list_a_volume_the_copy_created(monkeypatch):
    fake = FakeRuntime()
    fake.volumes["proj_eve-claude-config"] = {"com.docker.compose.project": PROJECT}

    _copy(_deploy_config(users=("eve",)), fake)

    listed = {v.name for v in RuntimeProbe("docker", run=fake).volumes_for_project(PROJECT)}
    assert "proj_eve-claude-projects" in listed

    monkeypatch.setattr(lifecycle.subprocess, "run", fake)
    orphans = lifecycle._discover_orphan_volumes("docker", PROJECT, set())
    assert orphans == {"eve": ["proj_eve-claude-config", "proj_eve-claude-projects"]}


# ── Where the copy runs in `osprey up` ──────────────────────────────────────


def _is_compose(argv: list[str]) -> bool:
    return "compose" in argv or any(a.endswith("podman-compose") for a in argv)


def _assert_copies_precede_compose(calls: list[list[str]]) -> None:
    copies = [i for i, c in enumerate(calls) if TRANSCRIPTS_COPY_IMAGE in c]
    composes = [i for i, c in enumerate(calls) if _is_compose(c)]
    assert copies, calls
    assert composes, calls
    assert max(copies) < min(composes), calls


#: The start's copy step, bound before the suite's stand-in replaces it.
_REAL_COPY_EXISTING_TRANSCRIPTS = container_lifecycle._copy_existing_transcripts


def _route_runtime_through(monkeypatch, fake: FakeRuntime, record: list[list[str]]) -> None:
    """Send every subprocess call to *fake*, keeping one ordered record.

    The start's copy step is put back in place of the suite's stand-in
    (``tests/deployment/conftest.py``), so the copy these tests order is the real one.
    """
    monkeypatch.setattr(
        container_lifecycle, "_copy_existing_transcripts", _REAL_COPY_EXISTING_TRANSCRIPTS
    )

    def _run(cmd, env=None, check=False, **kwargs):  # noqa: ARG001 - subprocess.run's keywords
        cmd = list(cmd)
        if cmd[0] == fake.runtime and not _is_compose(cmd):
            result = fake(cmd)
            record.append(cmd)
            return result
        record.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _run)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: record.append(list(cmd))),
    )


_AS_BUILT_CONFIG = """\
project_name: als-exemplar
build_dir: ./build
deployed_services:
  - event_dispatcher
  - dispatch_worker
services:
  event_dispatcher:
    path: services/event_dispatcher
  dispatch_worker:
    path: services/dispatch_worker
    worker_count: 1
claude_code:
  provider: anthropic
"""


@pytest.mark.usefixtures("started")
def test_the_as_built_start_copies_before_its_first_compose_call(
    lifecycle_repo,  # noqa: F811 - the imported fixture, by name
    monkeypatch,
):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, config=_AS_BUILT_CONFIG)
    fake = FakeRuntime()
    fake.volumes["als-exemplar_dispatch_workspace_1"] = {}
    record: list[list[str]] = []
    _route_runtime_through(monkeypatch, fake, record)

    result = run_up(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    assert "als-exemplar_dispatch_claude_projects_1" in fake.volumes
    _assert_copies_precede_compose(record)


def test_deploy_up_copies_before_its_first_compose_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "project_name": "legacy",
        "deployed_services": ["event_dispatcher", "dispatch_worker"],
        "services": {"dispatch_worker": {"worker_count": 1}},
    }
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    fake = FakeRuntime()
    fake.volumes["legacy_dispatch_workspace_1"] = {}
    record: list[list[str]] = []
    _route_runtime_through(monkeypatch, fake, record)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert "legacy_dispatch_claude_projects_1" in fake.volumes
    _assert_copies_precede_compose(record)


# ── The entrypoint hands a fresh transcripts mount to osprey ───────────────


_ENTRYPOINT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "project"
    / "entrypoint.sh"
)


def _hand_back_function() -> str:
    text = _ENTRYPOINT.read_text()
    start = text.index("hand_back_transcript_mounts() {")
    return text[start : text.index("\n}\n", start) + 3]


def _stub(bindir: Path, name: str, body: str) -> None:
    path = bindir / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _hand_back(tmp_path: Path, mountinfo: str | None, owners: dict[str, str]) -> list[str]:
    """Run the entrypoint's transcripts hand-back against a fake mount table.

    ``find`` answers from *owners* (path -> user) and keeps the entrypoint's
    predicate honest: it hands the path to ``-exec chown`` only for the
    ``-maxdepth 0 ! -user osprey`` form. ``chown`` records the path it was given.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    owners_file = tmp_path / "owners"
    owners_file.write_text("".join(f"{p} {u}\n" for p, u in owners.items()))
    chowned = tmp_path / "chowned"
    chowned.write_text("")
    _stub(
        bindir,
        "find",
        'path="$1"; shift\n'
        'case "$*" in\n'
        '  "-maxdepth 0 ! -user osprey -exec chown osprey:osprey {} +") ;;\n'
        '  *) echo "unexpected predicate: $*" >&2; exit 2 ;;\n'
        "esac\n"
        'owner=$(awk -v p="$path" \'$1 == p { print $2 }\' "$OWNERS")\n'
        '[ "$owner" = osprey ] || chown osprey:osprey "$path"\n',
    )
    _stub(bindir, "chown", 'printf "%s %s\\n" "$1" "$2" >> "$CHOWNED"\n')
    table = tmp_path / "mountinfo"
    if mountinfo is not None:
        table.write_text(mountinfo)
    script = (
        'log() { echo "LOG $*" >&2; }\n'
        + _hand_back_function()
        + f'hand_back_transcript_mounts "{table}"\n'
    )
    result = subprocess.run(
        ["/bin/sh", "-c", script],
        env={
            "PATH": f"{bindir}:/usr/bin:/bin",
            "OWNERS": str(owners_file),
            "CHOWNED": str(chowned),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "unexpected predicate" not in result.stderr, result.stderr
    assert "LOG" not in result.stderr, result.stderr
    return chowned.read_text().splitlines()


def _mount_line(n: int, mount_point: str) -> str:
    return f"{n} 1 0:{n} / {mount_point} rw,relatime - ext4 /dev/vda1 rw\n"


def test_the_entrypoint_hands_a_root_owned_transcripts_mount_to_osprey(tmp_path):
    """A fresh transcripts volume's root, and the mount-point directory the
    runtime made in its parent, are root-owned; both become osprey's."""
    terminal = "/data/claude-config/projects"
    mountinfo = (
        _mount_line(1, "/")
        + _mount_line(2, "/data/claude-config")
        + _mount_line(3, terminal)
        + _mount_line(4, "/workspace")
    )

    chowned = _hand_back(
        tmp_path,
        mountinfo,
        {"/data/claude-config": "root", terminal: "root", "/workspace": "root", "/": "root"},
    )

    assert chowned == [
        "osprey:osprey /data/claude-config",
        f"osprey:osprey {terminal}",
    ]


def test_the_entrypoint_leaves_an_osprey_owned_transcripts_mount_alone(tmp_path):
    """Only a wrong owner is fixed: a dispatch worker's mount that osprey
    already owns is not touched, while a root-owned one beside it is."""
    owned = "/home/osprey/agent-data/claude-config/projects"
    fresh = "/home/osprey/agent-data-2/claude-config/projects"
    mountinfo = _mount_line(1, "/") + _mount_line(2, owned) + _mount_line(3, fresh)

    chowned = _hand_back(
        tmp_path,
        mountinfo,
        {
            owned: "osprey",
            "/home/osprey/agent-data/claude-config": "osprey",
            fresh: "osprey",
            "/home/osprey/agent-data-2/claude-config": "root",
        },
    )

    assert chowned == ["osprey:osprey /home/osprey/agent-data-2/claude-config"]


def test_the_entrypoint_hand_back_is_a_no_op_without_a_mount_table(tmp_path):
    assert _hand_back(tmp_path, None, {}) == []


# ── A real copy ─────────────────────────────────────────────────────────────


_OSPREY_UID = 1000


@pytest.mark.xdist_group("docker")
def test_a_real_copy_leaves_the_transcript_readable_at_its_new_mount():
    """A transcript in an old claude-config volume is read at
    ``/data/claude-config/projects`` after the copy, owned by ``osprey``."""
    from tests._container_support import is_docker_available, start_or_skip, stop_quietly

    if not is_docker_available():
        pytest.skip("no Docker daemon on this host")
    from testcontainers.core.container import DockerContainer

    project = f"osprey-test-{uuid.uuid4().hex[:8]}"
    parent = f"{project}_alice{CLAUDE_CONFIG_VOLUME_SUFFIX}"
    target = f"{project}_alice{CLAUDE_PROJECTS_VOLUME_SUFFIX}"
    image = TRANSCRIPTS_COPY_IMAGE

    def _docker(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)

    try:
        assert _docker("volume", "create", parent).returncode == 0
        seeded = _docker(
            "run", "--rm", "--mount", f"type=volume,source={parent},destination=/v",
            image, "sh", "-c",
            "mkdir -p /v/projects/-app && echo '{\"t\":1}' > /v/projects/-app/s.jsonl "
            f"&& chown -R {_OSPREY_UID}:{_OSPREY_UID} /v/projects",
        )  # fmt: skip
        if seeded.returncode != 0:
            pytest.skip(f"cannot run {image} here: {seeded.stderr.strip()}")

        config = _deploy_config(users=("alice",))
        config["project_name"] = project
        assert copy_transcripts_once(config, runtime="docker") == [target]
        assert copy_transcripts_once(config, runtime="docker") == []

        labels = json.loads(
            _docker("volume", "inspect", target, "--format", "{{json .Labels}}").stdout
        )
        assert labels["com.docker.compose.project"] == project

        def _reader() -> Any:
            return (
                DockerContainer(image)
                .with_volume_mapping(parent, "/data/claude-config", "ro")
                .with_volume_mapping(target, "/data/claude-config/projects", "ro")
                .with_command("sleep 120")
            )

        reader = start_or_skip(_reader, label="transcripts reader")
        try:
            code, out = reader.exec("stat -c '%u %n' /data/claude-config/projects/-app/s.jsonl")
            assert code == 0, out
            assert out.decode().split()[0] == str(_OSPREY_UID)
            code, out = reader.exec("cat /data/claude-config/projects/-app/s.jsonl")
            assert code == 0 and out.decode().strip() == '{"t":1}'
        finally:
            stop_quietly(reader)
    finally:
        _docker("volume", "rm", "-f", target)
        _docker("volume", "rm", "-f", parent)

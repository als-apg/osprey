"""The one helper that builds every push-review container argv, pinned by a golden per verb.

``fixtures/push_argv/<verb>.json`` holds the argv byte for byte, the fixed script
included. A change to the hardening flags, the user, the mounts or a script
changes the golden and is reviewed as such. Nothing here needs CANOPY or a
container runtime: the runtime probe is faked in process.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from osprey.deployment import runtime_helper
from osprey.integrations.canopy import push
from osprey.integrations.canopy.push import container, refusals
from osprey.integrations.canopy.push import deployment as deployment_module

GOLDENS = Path(__file__).parent / "fixtures" / "push_argv"
HOST_IDS = (501, 20)
NONCE = "0badf00d"
IMAGE = "proj-readonly:local"
PAYLOAD = "/app/proj-readonly/canopy_payload"


def _golden(name: str, argv: list[str]) -> None:
    path = GOLDENS / f"{name}.json"
    if not path.exists():  # pragma: no cover - first run only
        path.write_text(json.dumps(argv, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == argv, f"update {path} deliberately"


def _list_argv() -> list[str]:
    return push.container_argv(
        "docker",
        "list",
        nonce=NONCE,
        image=IMAGE,
        script=container.LIST_SCRIPT,
        args=["/v/canopy/push", "256", "65536"],
        volume="proj_bob-agent-data",
        host_ids=HOST_IDS,
    )


def _stream_argv() -> list[str]:
    return push.container_argv(
        "docker",
        "show-stream",
        nonce=NONCE,
        image=IMAGE,
        script=container.STREAM_SCRIPT,
        args=["/v/canopy/push/0123456789abcdef", "256", "33554432", "5242880"],
        volume="proj_bob-agent-data",
        host_ids=HOST_IDS,
    )


def _check_argv() -> list[str]:
    return push.container_argv(
        "docker",
        "show-check",
        nonce=NONCE,
        image=IMAGE,
        script=container.CHECK_SCRIPT,
        args=["/s", "sr/pushed", "sr/pushed/panel"],
        stage="/srv/mca/var/canopy-push/bob/0123456789abcdef.1a2b3c4d",
        payload_dir=PAYLOAD,
        host_ids=HOST_IDS,
    )


# --- the goldens -----------------------------------------------------------------------------


def test_list_argv_golden() -> None:
    _golden("list", _list_argv())


def test_show_stream_argv_golden() -> None:
    _golden("show-stream", _stream_argv())


def test_show_check_argv_golden() -> None:
    _golden("show-check", _check_argv())


# --- what the goldens pin, spelled out ------------------------------------------------------------

HARDENING = [
    ("--network", "none"),
    ("--tmpfs", "/tmp"),
    ("-e", "HOME=/tmp"),
    ("--entrypoint", "python3"),
    ("--cap-drop", "ALL"),
    ("--security-opt", "no-new-privileges"),
    ("--memory", "1g"),
    ("--pids-limit", "256"),
]


@pytest.mark.parametrize("build", [_list_argv, _stream_argv, _check_argv])
def test_every_container_is_hardened_the_same_way(build) -> None:
    argv = build()
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--label") + 1] == "osprey.canopy.push=1"
    assert "--read-only" in argv
    for flag, value in HARDENING:
        assert argv[argv.index(flag) + 1] == value
    assert argv.count("--mount") == 1


def test_a_container_that_mounts_a_user_volume_runs_as_the_runtime_uid() -> None:
    for argv in (_list_argv(), _stream_argv()):
        assert argv[argv.index("--user") + 1] == "1000:1000"
        assert (
            argv[argv.index("--mount") + 1] == "type=volume,src=proj_bob-agent-data,dst=/v,readonly"
        )
        assert "CANOPY_WIDGETS_DIR" not in " ".join(argv)


def test_a_container_that_mounts_only_the_stage_runs_as_the_invoking_user() -> None:
    argv = _check_argv()
    assert argv[argv.index("--user") + 1] == "501:20"
    mount = argv[argv.index("--mount") + 1]
    assert (
        mount
        == "type=bind,src=/srv/mca/var/canopy-push/bob/0123456789abcdef.1a2b3c4d,dst=/s,readonly"
    )
    env = [argv[at + 1] for at, flag in enumerate(argv) if flag == "-e"]
    assert env == [
        "HOME=/tmp",
        "CANOPY_LIBRARY_IGNORED_NAMES=claude.local.md dockerfile prefetched-models var",
        f"CANOPY_VIEWER_DIST={PAYLOAD}/viewer",
    ]


@pytest.mark.parametrize("build", [_list_argv, _stream_argv, _check_argv])
def test_the_image_is_followed_by_the_fixed_script_and_its_arguments(build) -> None:
    argv = build()
    at = argv.index(IMAGE)
    assert argv[at + 1 : at + 3] == ["-I", "-c"]
    assert argv[at + 3] in (container.LIST_SCRIPT, container.STREAM_SCRIPT, container.CHECK_SCRIPT)
    assert all(
        not arg.startswith("/Users") and not arg.startswith("/home") for arg in argv[at + 4 :]
    )


def test_the_name_carries_the_verb_and_the_nonce() -> None:
    argv = _check_argv()
    assert argv[argv.index("--name") + 1] == f"osprey-canopy-push-show-check-{NONCE}"


def test_a_container_mounts_exactly_one_of_volume_and_stage() -> None:
    with pytest.raises(ValueError):
        push.container_argv(
            "docker", "x", nonce=NONCE, image=IMAGE, script="", args=[], host_ids=HOST_IDS
        )
    with pytest.raises(ValueError):
        push.container_argv(
            "docker",
            "x",
            nonce=NONCE,
            image=IMAGE,
            script="",
            args=[],
            volume="v",
            stage="/s",
            host_ids=HOST_IDS,
        )


def test_the_helper_refuses_any_runtime_but_docker() -> None:
    with pytest.raises(ValueError, match="requires docker"):
        push.container_argv(
            "podman",
            "x",
            nonce=NONCE,
            image=IMAGE,
            script="",
            args=[],
            volume="v",
            host_ids=HOST_IDS,
        )


def test_the_fixed_scripts_hold_no_host_path_and_no_git() -> None:
    for script in (container.LIST_SCRIPT, container.STREAM_SCRIPT, container.CHECK_SCRIPT):
        assert "/Users" not in script and "/home" not in script
        assert "git" not in script.replace("digest", "")
    assert "canopy" not in container.LIST_SCRIPT and "canopy" not in container.STREAM_SCRIPT


# --- the podman refusal through main --------------------------------------------------------------


def _deployment(tmp_path: Path) -> Path:
    root = tmp_path / "mca"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: proj\n", encoding="utf-8")
    (root / "build" / "config.yml").write_text(
        "project_name: proj\nmodules:\n  web_terminals:\n    enabled: true\n    users:\n"
        "      - name: bob\n        index: 0\n",
        encoding="utf-8",
    )
    library = root / "data" / "canopy" / "library" / "screens"
    library.mkdir(parents=True)
    return root


def _fake_probe(monkeypatch: pytest.MonkeyPatch, healthy: str) -> None:
    def _run(cmd, **_kwargs):
        rc = 0 if cmd[0] == healthy else 1
        return subprocess.CompletedProcess(cmd, rc, stdout=b"", stderr=b"")

    monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime_helper.subprocess, "run", _run)
    runtime_helper.reset_runtime_cache()


@pytest.fixture(autouse=True)
def _clean_runtime(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    runtime_helper.reset_runtime_cache()
    yield
    runtime_helper.reset_runtime_cache()


def test_a_resolved_podman_runtime_is_refused_with_its_own_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``CONTAINER_RUNTIME=podman`` resolves podman, which this release refuses."""
    root = _deployment(tmp_path)
    monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
    _fake_probe(monkeypatch, "podman")
    assert push.main(["--repo", str(root), "list"]) == refusals.EXIT_RUNTIME_PODMAN
    assert capsys.readouterr().err == "push: push review requires docker in this release\n"


def test_no_usable_runtime_is_its_own_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _deployment(tmp_path)
    _fake_probe(monkeypatch, "none")
    assert push.main(["--repo", str(root), "list"]) == refusals.EXIT_RUNTIME_MISSING


def test_uid_zero_is_refused_before_anything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(deployment_module.os, "geteuid", lambda: 0)
    assert push.main(["--repo", str(tmp_path / "absent"), "list"]) == refusals.EXIT_ROOT_UID
    assert "add that user to the docker group instead of using sudo" in capsys.readouterr().err


def test_every_exit_code_is_distinct() -> None:
    codes = {name: value for name, value in vars(refusals).items() if name.startswith("EXIT_")}
    assert len(set(codes.values())) == len(codes), codes
    assert set(refusals.GRAMMAR_EXIT.values()) <= set(codes.values())
    assert set(refusals.CHECK_EXIT.values()) <= set(codes.values())
    assert os.getuid() != 0

"""``guarded_run_dir`` resolves one deployment-wide directory per control target.

Every process of a deployment — on the host or in any of its containers — has
to land on the same ``<repo root>/var/guarded_run/<target>/`` for a target, so
the directory is keyed by the target NAME and anchored on the repo root. A
host layout that has no such directory yet gets it created.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    GUARDED_RUN_DIR,
    GUARDED_RUN_DIR_MODE,
    JOURNAL_FILE_NAME,
    LOCK_FILE_NAME,
    GuardedRunDirError,
    guarded_run_dir,
    guarded_run_target,
)

#: A deployment whose own control system is the Virtual Accelerator.
VA_BASELINE = {"type": "virtual_accelerator", "connector": {"virtual_accelerator": {"port": 1}}}

#: A deployment on a real control system.
LIVE_BASELINE = {"type": "epics", "connector": {"epics": {"gateways": {}}}}


def _host_repo(root: Path, control_system: dict, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo with a render and no ``var/`` zone, entered."""
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "config.yml").write_text(
        yaml.safe_dump({"control_system": control_system}), encoding="utf-8"
    )
    monkeypatch.chdir(root)
    monkeypatch.delenv(ENV_CONTROL_TARGET, raising=False)
    return root


def test_the_exported_names() -> None:
    assert GUARDED_RUN_DIR == "guarded_run"
    assert LOCK_FILE_NAME != JOURNAL_FILE_NAME
    assert "/" not in LOCK_FILE_NAME and "/" not in JOURNAL_FILE_NAME


def test_a_host_layout_without_var_gets_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)
    assert not (repo / "var").exists()

    directory = guarded_run_dir("va")

    assert directory == repo.resolve() / "var" / GUARDED_RUN_DIR / "va"
    assert directory.is_dir()
    for level in (directory, directory.parent):
        mode = stat.S_IMODE(level.stat().st_mode)
        assert mode & 0o777 == GUARDED_RUN_DIR_MODE & 0o777
        if level.stat().st_gid in os.getgroups():
            assert mode & stat.S_ISGID


def test_an_existing_directory_keeps_its_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)
    provisioned = repo / "var" / GUARDED_RUN_DIR / "live"
    provisioned.mkdir(parents=True)
    os.chmod(provisioned, 0o770)

    assert guarded_run_dir("live") == provisioned.resolve()
    assert stat.S_IMODE(provisioned.stat().st_mode) == 0o770


@pytest.mark.parametrize("stamp", [None, "baseline"], ids=["unstamped", "baseline-literal"])
def test_no_target_is_the_baseline_target_name(
    stamp: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _host_repo(tmp_path, VA_BASELINE, monkeypatch)
    if stamp is not None:
        monkeypatch.setenv(ENV_CONTROL_TARGET, stamp)

    assert guarded_run_target(None) == "va"
    assert guarded_run_dir(None) == repo.resolve() / "var" / GUARDED_RUN_DIR / "va"
    assert not (repo / "var" / GUARDED_RUN_DIR / "baseline").exists()


def test_the_executor_baseline_literal_maps_to_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)

    assert guarded_run_target("baseline") == "live"


def test_the_stamp_names_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "standin")

    assert guarded_run_dir(None) == repo.resolve() / "var" / GUARDED_RUN_DIR / "standin"


def test_an_unstamped_and_a_live_stamped_run_share_one_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)

    unstamped = guarded_run_dir(None)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")

    assert guarded_run_dir(None) == unstamped


def test_a_nested_working_directory_resolves_the_same_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)
    nested = repo / "data" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert guarded_run_dir("live") == repo.resolve() / "var" / GUARDED_RUN_DIR / "live"


def test_a_container_layout_resolves_under_the_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container's project directory holds the render at its root."""
    project = tmp_path / "app" / "proj"
    project.mkdir(parents=True)
    (project / "config.yml").write_text(
        yaml.safe_dump({"project_root": "/no/such/host/path", "control_system": LIVE_BASELINE}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(project / "config.yml"))
    monkeypatch.chdir(tmp_path)

    assert guarded_run_dir("live") == project / "var" / GUARDED_RUN_DIR / "live"


def test_an_unknown_target_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)

    with pytest.raises(GuardedRunDirError, match="Unknown control target 'nowhere'"):
        guarded_run_dir("nowhere")
    assert not (repo / "var").exists()


def test_an_unresolvable_repo_root_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ENV_CONTROL_TARGET, raising=False)

    with pytest.raises(GuardedRunDirError, match="no deployment repo root"):
        guarded_run_dir("live")
    assert not (tmp_path / "var").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode")
def test_an_unwritable_path_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)
    locked = repo / "var" / GUARDED_RUN_DIR / "live"
    locked.mkdir(parents=True)
    os.chmod(locked, 0o555)
    try:
        with pytest.raises(GuardedRunDirError, match="is not writable"):
            guarded_run_dir("live")
    finally:
        os.chmod(locked, 0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode")
def test_an_uncreatable_path_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _host_repo(tmp_path, LIVE_BASELINE, monkeypatch)
    state_zone = repo / "var"
    state_zone.mkdir()
    os.chmod(state_zone, 0o555)
    try:
        with pytest.raises(GuardedRunDirError, match="cannot create"):
            guarded_run_dir("live")
    finally:
        os.chmod(state_zone, 0o755)

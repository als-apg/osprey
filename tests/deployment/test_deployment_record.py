"""The deployment record: what the last start drove, and how it is read back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.deployment_record import (
    COMPOSE_PROJECT_KEY,
    DEPLOYMENT_RECORD_RELPATH,
    NOTHING_DEPLOYED,
    RUNTIME_KEY,
    DeploymentRecord,
    deployed_project,
    deployed_runtime,
    read_deployment_record,
    write_deployment_record,
)
from osprey.deployment.errors import DeploymentRecordError


def _project() -> str:
    return resolve_project_name({"project_name": "Demo Facility"})


def test_a_written_record_reads_back_equal(tmp_path: Path) -> None:
    record = DeploymentRecord("podman", _project())

    path = write_deployment_record(tmp_path, record)

    assert path == tmp_path / DEPLOYMENT_RECORD_RELPATH
    assert read_deployment_record(tmp_path) == record


def test_no_record_reads_as_none(tmp_path: Path) -> None:
    assert read_deployment_record(tmp_path) is None


def test_the_written_bytes_are_deterministic(tmp_path: Path) -> None:
    record = DeploymentRecord("docker", _project())
    first = write_deployment_record(tmp_path / "a", record).read_bytes()
    second = write_deployment_record(tmp_path / "b", record).read_bytes()

    assert first == second
    assert (
        first
        == (
            json.dumps({COMPOSE_PROJECT_KEY: _project(), RUNTIME_KEY: "docker"}, sort_keys=True)
            + "\n"
        ).encode()
    )


def test_the_record_is_readable_by_every_account(tmp_path: Path) -> None:
    path = write_deployment_record(tmp_path, DeploymentRecord("docker", _project()))

    assert path.stat().st_mode & 0o777 == 0o644


def test_a_rewrite_replaces_the_previous_record(tmp_path: Path) -> None:
    write_deployment_record(tmp_path, DeploymentRecord("docker", _project()))
    write_deployment_record(tmp_path, DeploymentRecord("podman", _project()))

    assert read_deployment_record(tmp_path) == DeploymentRecord("podman", _project())


def _raw(tmp_path: Path, text: str) -> None:
    path = tmp_path / DEPLOYMENT_RECORD_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    "document",
    [
        "{not json",
        "[]",
        json.dumps({RUNTIME_KEY: "docker"}),
        json.dumps({RUNTIME_KEY: "docker", COMPOSE_PROJECT_KEY: "demo", "extra": 1}),
        json.dumps({RUNTIME_KEY: "nerdctl", COMPOSE_PROJECT_KEY: "demo"}),
        json.dumps({RUNTIME_KEY: "docker", COMPOSE_PROJECT_KEY: "Demo Facility"}),
        json.dumps({RUNTIME_KEY: "docker", COMPOSE_PROJECT_KEY: ""}),
    ],
    ids=[
        "corrupt",
        "not-an-object",
        "missing-key",
        "extra-key",
        "nerdctl",
        "unnormalised",
        "empty",
    ],
)
def test_a_record_that_is_not_one_is_refused_naming_the_path(tmp_path: Path, document: str) -> None:
    _raw(tmp_path, document)

    with pytest.raises(DeploymentRecordError, match=str(DEPLOYMENT_RECORD_RELPATH)):
        read_deployment_record(tmp_path)


def test_an_unreadable_record_is_refused_naming_the_path(tmp_path: Path) -> None:
    (tmp_path / DEPLOYMENT_RECORD_RELPATH).mkdir(parents=True)

    with pytest.raises(DeploymentRecordError, match=str(DEPLOYMENT_RECORD_RELPATH)):
        read_deployment_record(tmp_path)


@pytest.mark.parametrize(
    ("runtime", "project"),
    [("nerdctl", "demo"), ("Docker", "demo"), ("docker", ""), ("docker", "Demo Facility")],
)
def test_a_record_refuses_construction_outside_its_invariant(runtime: str, project: str) -> None:
    with pytest.raises(ValueError):
        DeploymentRecord(runtime, project)


def test_observers_read_the_record_over_a_renamed_build_and_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deployment_record(tmp_path, DeploymentRecord("podman", _project()))
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("project_name: Renamed Facility\n", encoding="utf-8")
    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")

    assert deployed_project(tmp_path) == _project()
    assert deployed_runtime(tmp_path) == "podman"


def test_observers_answer_none_with_no_record(tmp_path: Path) -> None:
    assert deployed_project(tmp_path) is None
    assert deployed_runtime(tmp_path) is None


def test_observers_raise_on_a_corrupt_record(tmp_path: Path) -> None:
    _raw(tmp_path, "{not json")

    with pytest.raises(DeploymentRecordError, match=str(DEPLOYMENT_RECORD_RELPATH)):
        deployed_project(tmp_path)
    with pytest.raises(DeploymentRecordError, match=str(DEPLOYMENT_RECORD_RELPATH)):
        deployed_runtime(tmp_path)


def test_the_health_script_says_what_the_observers_say() -> None:
    import osprey

    template = Path(osprey.__file__).parent / "templates" / "deploy" / "verify.sh.j2"

    assert NOTHING_DEPLOYED in template.read_text(encoding="utf-8")

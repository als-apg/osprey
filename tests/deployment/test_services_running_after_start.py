"""A start that returns has a running container for every service it started.

``compose up -d`` exiting 0 is not that guarantee: a compose provider can exit
0 having created nothing. The check reads this checkout's containers through
the same listing the closing card reads, which claims containers by the
deployment's ownership rule, matches them to services by compose's own service
label, and fails the start naming what it found.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import repo_identity
from osprey.deployment.container_lifecycle import check_started_services
from osprey.deployment.deployment_record import DeploymentRecord, write_deployment_record
from osprey.deployment.errors import ServicesNotRunningError
from osprey.deployment.web_terminals.render import render_web_terminals
from tests._renders import tier_render
from tests._scoped_stdlib import scoped

# The real check, taken before the conftest seam replaces it for every test.
_REAL_REFUSE_SERVICES_NOT_RUNNING = container_lifecycle._refuse_services_not_running

SERVICE_LABEL = "com.docker.compose.service"
PROJECT_LABEL = "com.docker.compose.project"
REPO_ID_LABEL = "com.osprey.repo-id"


class _FakePs:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="")


class _RuntimePs:
    """A ``ps`` that, like a real runtime, lists only rows matching every label filter."""

    def __init__(self, *rows: dict) -> None:
        self.rows = rows
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        wanted = [
            cmd[i + 1].removeprefix("label=").split("=", 1)
            for i, arg in enumerate(cmd)
            if arg == "--filter" and cmd[i + 1].startswith("label=")
        ]
        kept = [row for row in self.rows if all(row["Labels"].get(k) == v for k, v in wanted)]
        return subprocess.CompletedProcess(cmd, 0, stdout=_docker(*kept), stderr="")


def _missing(repo: Path, services: dict) -> dict[str, str]:
    """The services the check names as neither running nor done."""
    return check_started_services(repo, services).missing


def _docker(*rows: dict) -> str:
    """Docker's NDJSON listing, its labels one comma-joined string."""
    return "\n".join(
        json.dumps({**row, "Labels": ",".join(f"{k}={v}" for k, v in row["Labels"].items())})
        for row in rows
    )


def _podman(*rows: dict) -> str:
    """Podman's array listing, its labels an object."""
    return json.dumps(list(rows))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo whose last start drove docker."""
    write_deployment_record(tmp_path, DeploymentRecord("docker", "demo"))
    return tmp_path


@pytest.fixture
def real_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        container_lifecycle, "_refuse_services_not_running", _REAL_REFUSE_SERVICES_NOT_RUNNING
    )


def test_docker_rows_satisfy_their_service_by_compose_label(repo, monkeypatch):
    listing = _docker(
        {"Names": "demo-api-1", "State": "running", "Status": "Up 3 seconds",
         "Labels": {SERVICE_LABEL: "api", "com.docker.compose.project": "demo"}},
        {"Names": "demo-worker-1", "State": "running", "Status": "Up 3 seconds",
         "Labels": {SERVICE_LABEL: "worker"}},
    )  # fmt: skip
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(listing))

    assert _missing(repo, {"api": {}, "worker": {}}) == {}


def test_podman_rows_satisfy_their_service_by_compose_label(repo, monkeypatch):
    listing = _podman(
        {"Names": ["demo-api-1"], "State": "running", "Labels": {SERVICE_LABEL: "api"}},
        {"Names": ["demo-worker-1"], "State": "running", "Labels": {SERVICE_LABEL: "worker"}},
    )
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(listing))

    assert _missing(repo, {"api": {}, "worker": {}}) == {}


def test_a_service_with_no_row_is_named_no_container(repo, monkeypatch):
    """A container whose name merely contains the service's is not its container."""
    listing = _podman(
        {"Names": ["demo-api-recorder-1"], "State": "running",
         "Labels": {SERVICE_LABEL: "api-recorder"}},
    )  # fmt: skip
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(listing))

    assert _missing(repo, {"api": {}}) == {"api": "no container"}


def test_an_exited_container_is_named_with_its_exit_code(repo, monkeypatch):
    listing = _docker(
        {"Names": "demo-api-1", "State": "exited", "Status": "Exited (1) 2 seconds ago",
         "Labels": {SERVICE_LABEL: "api"}},
        {"Names": "demo-worker-1", "State": "restarting",
         "Status": "Restarting (2) 1 second ago", "Labels": {SERVICE_LABEL: "worker"}},
    )  # fmt: skip
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(listing))

    assert _missing(repo, {"api": {}, "worker": {}}) == {
        "api": "exited (1)",
        "worker": "restarting",
    }


def test_a_run_once_service_that_exited_zero_is_satisfied(repo, monkeypatch):
    listing = _podman(
        {"Names": ["demo-init-1"], "State": "exited", "ExitCode": 0,
         "Labels": {SERVICE_LABEL: "init"}},
        {"Names": ["demo-api-1"], "State": "running", "Labels": {SERVICE_LABEL: "api"}},
    )  # fmt: skip
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(listing))
    services = {
        "init": {"image": "init"},
        "api": {"depends_on": {"init": {"condition": "service_completed_successfully"}}},
    }

    assert _missing(repo, services) == {}


def test_a_daemon_that_exited_zero_is_not_satisfied(repo, monkeypatch):
    listing = _podman(
        {"Names": ["demo-api-1"], "State": "exited", "ExitCode": 0,
         "Labels": {SERVICE_LABEL: "api"}},
    )  # fmt: skip
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(listing))
    services = {"api": {"depends_on": {"db": {"condition": "service_healthy"}}}}

    assert _missing(repo, services) == {"api": "exited (0)"}


def _running(labels: dict) -> dict:
    return {"Names": "demo-qmd-okf-1", "State": "running", "Status": "Up 1 second",
            "Labels": labels}  # fmt: skip


def test_a_running_container_without_the_checkout_label_satisfies_its_service(repo, monkeypatch):
    ps = _RuntimePs(_running({PROJECT_LABEL: "demo", SERVICE_LABEL: "qmd-okf"}))
    scoped(monkeypatch, container_lifecycle, "subprocess", run=ps)

    assert _missing(repo, {"qmd-okf": {}}) == {}


def test_another_checkouts_container_does_not_satisfy_the_service(repo, monkeypatch):
    ps = _RuntimePs(
        _running({PROJECT_LABEL: "demo", SERVICE_LABEL: "qmd-okf", REPO_ID_LABEL: "someone-else"})
    )
    scoped(monkeypatch, container_lifecycle, "subprocess", run=ps)

    assert _missing(repo, {"qmd-okf": {}}) == {"qmd-okf": "no container"}


def test_this_checkouts_labelled_container_satisfies_its_service(repo, monkeypatch):
    ps = _RuntimePs(
        _running(
            {PROJECT_LABEL: "demo", SERVICE_LABEL: "qmd-okf", REPO_ID_LABEL: repo_identity(repo)}
        )
    )
    scoped(monkeypatch, container_lifecycle, "subprocess", run=ps)

    assert _missing(repo, {"qmd-okf": {}}) == {}


def test_another_projects_container_does_not_satisfy_the_service(repo, monkeypatch):
    ps = _RuntimePs(_running({PROJECT_LABEL: "other", SERVICE_LABEL: "qmd-okf"}))
    scoped(monkeypatch, container_lifecycle, "subprocess", run=ps)

    assert _missing(repo, {"qmd-okf": {}}) == {"qmd-okf": "no container"}


def _compose(path: Path, services: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"services:\n{services}", encoding="utf-8")


@pytest.mark.usefixtures("real_check")
def test_a_listing_failure_fails_the_start(repo, monkeypatch):
    _compose(repo / "build" / "services" / "docker-compose.yml", "  api:\n    image: a\n")
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(returncode=125))

    with pytest.raises(RuntimeError, match=r"could not verify the start: `docker ps` exited 125"):
        container_lifecycle._refuse_services_not_running(
            {}, ["build/services/docker-compose.yml"], repo
        )


@pytest.mark.usefixtures("real_check")
def test_a_missing_service_fails_the_start_naming_it_and_its_logs(repo, monkeypatch):
    _compose(repo / "build" / "services" / "docker-compose.yml", "  api:\n    image: a\n")
    scoped(monkeypatch, container_lifecycle, "subprocess", run=_FakePs(""))

    with pytest.raises(ServicesNotRunningError) as raised:
        container_lifecycle._refuse_services_not_running(
            {}, ["build/services/docker-compose.yml"], repo
        )

    assert str(raised.value) == (
        "compose reported success, but 1 service(s) have no running container: "
        "api (no container). Run `osprey logs api` to see why."
    )


def test_the_expected_set_is_every_rendered_service_plus_the_web_stack(tmp_path):
    """The web stack's services are the ones its renderer renders, known
    before the start writes its file."""
    _compose(tmp_path / "build" / "services" / "docker-compose.yml", "  api:\n    image: a\n")
    absolute = tmp_path / "build" / "services" / "worker" / "docker-compose.yml"
    _compose(absolute, "  worker:\n    image: w\n")
    files = ["build/services/docker-compose.yml", str(absolute)]
    web_config = {
        "deploy": {"fqdn": "example.invalid"},
        "registry": {"url": "registry.example.org/demo"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice"],
                "default_tier": "assistant",
                "renders": {"assistant": tier_render("assistant", "ex-assistant")},
            }
        },
    }
    rendered = yaml.safe_load(render_web_terminals(web_config)["docker-compose.web.yml"])

    without_web = container_lifecycle._started_services({}, files, tmp_path)
    with_web = container_lifecycle._started_services(web_config, files, tmp_path)

    assert list(without_web) == ["api", "worker"]
    assert list(with_web) == ["api", "worker", *rendered["services"]]


def test_a_service_two_files_define_keeps_what_each_says(tmp_path):
    """The labels override names every service again; its labels add to the
    service's own definition rather than replacing it."""
    _compose(tmp_path / "base.yml", "  api:\n    image: a\n    depends_on:\n      init: {}\n")
    _compose(tmp_path / "labels.yml", "  api:\n    labels:\n      x: y\n")

    started = container_lifecycle._started_services({}, ["base.yml", "labels.yml"], tmp_path)

    assert started == {"api": {"image": "a", "depends_on": {"init": {}}, "labels": {"x": "y"}}}


@pytest.mark.usefixtures("real_check")
def test_a_start_that_returns_is_checked_once(repo, monkeypatch):
    (repo / "build").mkdir(exist_ok=True)
    (repo / "build" / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    _compose(repo / "build" / "services" / "docker-compose.yml", "  api:\n    image: a\n")
    monkeypatch.setattr(container_lifecycle, "_start_stack", lambda *args, **kwargs: None)
    checked: list[list[str]] = []

    def _spy(_repo_root, services):
        checked.append(list(services))
        return container_lifecycle.StartCheck({}, {})

    monkeypatch.setattr(container_lifecycle, "check_started_services", _spy)

    container_lifecycle._start_as_built(
        repo,
        {"project_name": "demo"},
        ["build/services/docker-compose.yml"],
        False,
        detached=True,
        dev_mode=False,
    )

    assert checked == [["api"]]

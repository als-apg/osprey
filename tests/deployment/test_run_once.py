"""Services that run once: the declaration, the start check's reading of it, and the status card.

A service declares itself run-once with the compose label ``osprey.run-once``
(``must-succeed`` or ``may-fail``). Nothing else makes a service run-once: a
service that exited 0 with no restart policy and no declaration is a service
that is not running.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import click
import pytest
import yaml

from osprey.cli.build_cmd import _render_compose_files, _render_zones
from osprey.deployment import container_lifecycle, run_once, status_display
from osprey.deployment.container_lifecycle import check_started_services
from osprey.deployment.deployment_record import DeploymentRecord, write_deployment_record
from osprey.deployment.errors import ServicesNotRunningError

# The real check, taken before the conftest seam replaces it for every test.
_REAL_REFUSE_SERVICES_NOT_RUNNING = container_lifecycle._refuse_services_not_running

SERVICE_LABEL = "com.docker.compose.service"


class _FakePs:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout

    def __call__(self, cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr="")


def _podman(*rows: dict) -> str:
    return json.dumps(list(rows))


def _exited(service: str, code: int) -> dict:
    return {"Names": [f"demo-{service}-1"], "State": "exited", "ExitCode": code,
            "Labels": {SERVICE_LABEL: service}}  # fmt: skip


def _running(service: str) -> dict:
    return {"Names": [f"demo-{service}-1"], "State": "running",
            "Labels": {SERVICE_LABEL: service}}  # fmt: skip


def _declared(value: str, **extra: Any) -> dict:
    return {"image": "x", "labels": {run_once.RUN_ONCE_LABEL: value}, **extra}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_deployment_record(tmp_path, DeploymentRecord("podman", "demo"))
    return tmp_path


def _check(repo: Path, monkeypatch: pytest.MonkeyPatch, services: dict, *rows: dict):
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _FakePs(_podman(*rows)))
    return check_started_services(repo, services)


# -- the start check ---------------------------------------------------------------


def test_a_must_succeed_service_nothing_waits_on_is_done_when_it_exited_zero(repo, monkeypatch):
    services = {"seed": _declared(run_once.MUST_SUCCEED), "api": {"image": "a"}}
    found = _check(repo, monkeypatch, services, _exited("seed", 0), _running("api"))
    assert found == container_lifecycle.StartCheck({}, {})


def test_a_must_succeed_service_that_exited_non_zero_fails_by_name_and_code(repo, monkeypatch):
    services = {"seed": _declared(run_once.MUST_SUCCEED)}
    assert _check(repo, monkeypatch, services, _exited("seed", 3)).missing == {"seed": "exited (3)"}


def test_a_may_fail_service_that_exited_non_zero_is_reported_and_allowed(repo, monkeypatch):
    services = {"seed": _declared(run_once.MAY_FAIL), "api": {"image": "a"}}
    found = _check(repo, monkeypatch, services, _exited("seed", 3), _running("api"))
    assert found.missing == {}
    assert found.allowed == {"seed": 3}


@pytest.mark.parametrize(
    "rows",
    [(), ({"Names": ["demo-seed-1"], "State": "created", "Labels": {SERVICE_LABEL: "seed"}},)],
    ids=["no container", "created"],
)
def test_a_run_once_service_that_never_ran_fails_whatever_it_declares(repo, monkeypatch, rows):
    for value in run_once.VALUES:
        found = _check(repo, monkeypatch, {"seed": _declared(value)}, *rows)
        assert list(found.missing) == ["seed"]
        assert found.allowed == {}


def test_an_undeclared_service_that_exited_zero_without_a_restart_policy_is_not_running(
    repo, monkeypatch
):
    """The rule is a declaration, never a guess: a long-running service that
    stopped cleanly must not pass for one that ran once."""
    services = {"worker": {"image": "w"}}
    assert _check(repo, monkeypatch, services, _exited("worker", 0)).missing == {
        "worker": "exited (0)"
    }


def test_labels_in_list_form_declare_the_same(repo, monkeypatch):
    services = {"seed": {"image": "x", "labels": [f"{run_once.RUN_ONCE_LABEL}=may-fail"]}}
    assert _check(repo, monkeypatch, services, _exited("seed", 1)).allowed == {"seed": 1}


def test_a_service_waited_on_to_complete_is_must_succeed_by_compose_s_own_declaration(
    repo, monkeypatch
):
    services = {
        "init": {"image": "i"},
        "api": {"depends_on": {"init": {"condition": "service_completed_successfully"}}},
    }
    found = _check(repo, monkeypatch, services, _exited("init", 2), _running("api"))
    assert found.missing == {"init": "exited (2)"}


def test_the_labels_the_build_adds_keep_the_label_the_template_set(tmp_path):
    base = tmp_path / "base.yml"
    base.write_text(
        yaml.safe_dump({"services": {"seed": _declared(run_once.MAY_FAIL)}}), encoding="utf-8"
    )
    stamped = tmp_path / "labels.yml"
    stamped.write_text(
        yaml.safe_dump({"services": {"seed": {"labels": {"osprey.config.digest": "d"}}}}),
        encoding="utf-8",
    )
    merged = container_lifecycle.merged_compose_services([base, stamped])
    assert run_once.declarations(merged) == {"seed": run_once.MAY_FAIL}


def _compose(path: Path, services: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"services": services}), encoding="utf-8")


def test_a_start_with_an_allowed_failure_succeeds_and_names_it(repo, monkeypatch):
    _compose(
        repo / "build" / "services" / "docker-compose.yml",
        {"seed": _declared(run_once.MAY_FAIL), "api": {"image": "a"}},
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        _FakePs(_podman(_exited("seed", 4), _running("api"))),
    )
    warned: list[tuple] = []
    monkeypatch.setattr(container_lifecycle, "_warn_fact", lambda *words: warned.append(words))

    _REAL_REFUSE_SERVICES_NOT_RUNNING({}, ["build/services/docker-compose.yml"], repo)

    assert [words[0] for words in warned] == ["run-once service seed exited 4"]


def test_a_start_with_a_failed_must_succeed_service_fails(repo, monkeypatch):
    _compose(
        repo / "build" / "services" / "docker-compose.yml", {"seed": _declared("must-succeed")}
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _FakePs(_podman(_exited("seed", 4))))

    with pytest.raises(ServicesNotRunningError, match=r"seed \(exited \(4\)\)"):
        _REAL_REFUSE_SERVICES_NOT_RUNNING({}, ["build/services/docker-compose.yml"], repo)


# -- the declaration ---------------------------------------------------------------


def test_a_value_the_label_does_not_take_is_refused_by_name():
    (error,) = run_once.declaration_errors({"seed": _declared("maybe")})
    assert "'seed'" in error and "must-succeed or may-fail" in error


@pytest.mark.parametrize("restart", ["always", "unless-stopped"])
def test_a_run_once_service_the_runtime_restarts_is_refused(restart):
    (error,) = run_once.declaration_errors({"seed": _declared("must-succeed", restart=restart)})
    assert f"restart: {restart}" in error


@pytest.mark.parametrize("restart", [None, "no", "on-failure"])
def test_a_run_once_service_may_stay_down_or_retry_on_failure(restart):
    extra = {"restart": restart} if restart else {}
    assert run_once.declaration_errors({"seed": _declared("may-fail", **extra)}) == []


def test_may_fail_on_a_service_another_waits_on_is_refused():
    services = {
        "seed": _declared(run_once.MAY_FAIL),
        "api": {"depends_on": {"seed": {"condition": "service_completed_successfully"}}},
    }
    (error,) = run_once.declaration_errors(services)
    assert "service_completed_successfully" in error


def test_the_build_refuses_a_render_whose_declarations_are_wrong(tmp_path, monkeypatch):
    zones = _render_zones(tmp_path, tmp_path.name)
    zones.stage.mkdir(parents=True)
    (zones.stage / "config.yml").write_text("{}\n", encoding="utf-8")
    rendered = zones.stage / "build" / "services" / "seed" / "docker-compose.yml"
    _compose(rendered, {"seed": _declared("must-succeed", restart="always")})
    config = {"project_name": "demo", "services": {}, "deployed_services": []}
    monkeypatch.setattr(
        "osprey.deployment.compose_generator.prepare_compose_files",
        lambda *args, **kwargs: (config, [str(rendered)]),
    )
    before = os.getcwd()

    with pytest.raises(click.UsageError, match="Run-once declarations are refused"):
        _render_compose_files(zones)
    assert os.getcwd() == before


# -- the status card ---------------------------------------------------------------


def _row(state: str, status: str, value: str | None, code: int | None = None) -> dict:
    labels = {run_once.RUN_ONCE_LABEL: value} if value else {}
    row: dict[str, Any] = {"State": state, "Status": status, "Labels": labels}
    if code is not None:
        row["ExitCode"] = code
    return row


def test_the_status_card_shows_a_run_once_service_that_exited_zero_as_done():
    label = status_display._run_once_state(
        _row("exited", "Exited (0) 1 minute ago", "must-succeed")
    )
    assert label is not None and label.plain == "● Done"


def test_the_status_card_shows_an_allowed_failure_as_done_with_its_code():
    label = status_display._run_once_state(_row("exited", "Exited (5) 1 minute ago", "may-fail", 5))
    assert label is not None and label.plain == "● Done (exit 5)"


@pytest.mark.parametrize(
    "row",
    [
        _row("exited", "Exited (5) 1 minute ago", "must-succeed"),
        _row("exited", "Exited (0) 1 minute ago", None),
        _row("running", "Up 2 minutes", "may-fail"),
    ],
    ids=["must-succeed failed", "undeclared", "still running"],
)
def test_the_status_card_leaves_every_other_end_to_the_plain_state(row):
    assert status_display._run_once_state(row) is None

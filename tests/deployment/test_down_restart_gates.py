"""``osprey down`` and ``osprey restart`` — stopping a deployment repo, and cycling it.

Four properties, and everything here serves one of them.

**A stop never renders.** ``down`` acts on the compose files a build left and
re-derives none of them. The legacy ``osprey down`` re-rendered when it could not
find them, which produces a ``down`` aimed at the stack the *current* profile
would start rather than the one that is running. The tests assert that by the
absence of ``prepare_compose_files`` on every path.

**A stop that cannot find its compose files still stops the stack.** When
``build/`` is gone, ``down`` selects the containers of the services the last
start recorded, in the recorded compose project, minus another checkout's; when
the record names no services, or there is no record, it selects this checkout's
containers by the ``com.osprey.repo-id`` label the render baked into them. The
tests pin what each selection removes, what it leaves (volumes, always; the
unlabelled network; another checkout's containers), and — the part that is easy
to get wrong — that the label sweep says so honestly when it finds nothing,
because a stack created before OSPREY labelled its containers is invisible to it
and still running.

**A restart is a stop followed by a start, and the start is gated first.**
``compose restart`` restarts containers against the definition they were created
with, so a rebuilt image or a new token reaches nothing; these recreate them. And
every refusal a start can raise — drift, no build, a ``--dev`` that cannot be
honored — is raised while the running stack is still up. The tests assert both
halves: the ordering of the two compose invocations, and that a refused restart
stopped nothing.

**Volumes survive both verbs.** On every path, compose and label alike. Losing
data is ``osprey reset``'s job, and it asks first.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.deploy_cmd import down_verb, restart_verb
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity, resolve_project_name
from osprey.deployment.deployment_record import DeploymentRecord, write_deployment_record
from tests.deployment.test_up_as_built import (
    _RENDERED_CONFIG,
    bake_labels_override,
    render_build,
)
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

# A rendered config that also turns the web-terminal stack on. `down` has to
# stop that stack first, so the ordering test needs a config that has one.
_WEB_CONFIG = _RENDERED_CONFIG + "modules:\n  web_terminals:\n    enabled: true\n"


@pytest.fixture
def lifecycle_repo(tmp_path: Path) -> Path:
    """The gold-standard three-zone deployment repo, with no ``.env`` in it.

    It holds the record an earlier ``osprey up`` of the rendered config wrote,
    which is what ``down`` acts on.
    """
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)
    project = resolve_project_name(yaml.safe_load(_RENDERED_CONFIG))
    write_deployment_record(repo, DeploymentRecord("docker", project))
    return repo


@pytest.fixture
def runtime(monkeypatch):
    """Stub the container runtime; record every argv, and let tests script it.

    Returns the record dict. Three keys are writable by a test before it acts:
    ``ps_stdout`` (what the label listing finds), ``rows`` (containers on the
    host, which ``ps`` then filters by label as a runtime does, and which win
    over ``ps_stdout`` when set) and ``fail`` (a mapping from a runtime verb to
    the exit code it should report).

    Unlike ``up``'s equivalent this returns real ``CompletedProcess`` objects,
    because the label sweep reads ``returncode`` and ``stdout`` off them.
    """
    record: dict = {"cmds": [], "ps_stdout": "", "rows": None, "fail": {}}

    monkeypatch.setattr(
        container_lifecycle, "verify_runtime_is_running", lambda config, **_: (True, "")
    )
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda *a, **k: record.setdefault("built", True),
    )

    def _fake_run(
        cmd,
        env=None,  # noqa: ARG001 - subprocess.run's keywords
        check=False,  # noqa: ARG001 - subprocess.run's keywords
        capture_output=False,  # noqa: ARG001 - subprocess.run's keywords
        text=False,  # noqa: ARG001 - subprocess.run's keywords
        **kwargs,
    ):
        # ``**kwargs`` swallows the redirection keywords ``run_captured`` passes
        # (``cwd``/``stdout``/``stderr``): a captured child's output goes to a
        # spool file, and nothing here writes any, so they are ignored.
        argv = list(cmd)
        record["cmds"].append(argv)
        # Matched on presence rather than position: a compose argv carries its
        # subcommand after the pinned `--project-directory`/`-f`/`--env-file`
        # block, while the label sweep's is the second token. Rendered paths end
        # in `docker-compose.yml`, so no path token collides with a verb name.
        failed = next((code for verb, code in record["fail"].items() if verb in argv), 0)
        stdout = ""
        if "ps" in argv:
            stdout = record["ps_stdout"] if record["rows"] is None else _ps_rows(argv, record)
        return subprocess.CompletedProcess(argv, failed, stdout, "boom")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    def _forbidden(*args, **kwargs):
        raise AssertionError("a stop must not render compose files")

    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", _forbidden)
    return record


def _ps_rows(argv: list[str], record: dict) -> str:
    """What a runtime's ``ps`` prints for *argv* over the rows on the host.

    Only the rows whose labels hold every ``--filter label=<k>=<v>`` are listed:
    as Docker's NDJSON, labels one comma-joined string, under ``--format json``,
    and one ``ID`` per line otherwise.
    """
    wanted = [
        argv[i + 1].removeprefix("label=").split("=", 1)
        for i, arg in enumerate(argv)
        if arg == "--filter" and argv[i + 1].startswith("label=")
    ]
    kept = [row for row in record["rows"] if all(row["Labels"].get(k) == v for k, v in wanted)]
    if "--format" in argv and argv[argv.index("--format") + 1] == "json":
        return "\n".join(
            json.dumps({**row, "Labels": ",".join(f"{k}={v}" for k, v in row["Labels"].items())})
            for row in kept
        )
    return "".join(f"{row['ID']}\n" for row in kept)


@pytest.fixture
def no_web(monkeypatch):
    """Record web-stack teardowns instead of running them."""
    calls: list = []
    monkeypatch.setattr(
        container_lifecycle,
        "deploy_down_web_terminals",
        lambda config, env, env_file_args, **kwargs: calls.append(config),
    )
    return calls


def run_down(repo: Path, *args: str) -> object:
    """Invoke ``osprey down`` against *repo* from outside it."""
    return CliRunner().invoke(down_verb, ["--repo", str(repo), *args])


def run_restart(repo: Path, *args: str) -> object:
    """Invoke ``osprey restart`` against *repo* from outside it."""
    return CliRunner().invoke(restart_verb, ["--repo", str(repo), *args])


def argv_for(record: dict, verb: str) -> list[str]:
    """The first recorded argv whose compose/runtime subcommand is *verb*."""
    for cmd in record["cmds"]:
        if verb in cmd:
            return cmd
    raise AssertionError(f"no `{verb}` in {record['cmds']}")


def verbs_in_order(record: dict) -> list[str]:
    """The compose/runtime subcommands that ran, in the order they ran."""
    known = {"up", "down", "ps", "stop", "rm", "build", "restart"}
    seen = []
    for cmd in record["cmds"]:
        for token in cmd:
            if token in known:
                seen.append(token)
                break
    return seen


# ---------------------------------------------------------------------------
# down: it stops what the build rendered, and renders nothing
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_web")
def test_down_stops_the_compose_files_the_build_left(lifecycle_repo, runtime):
    render_build(lifecycle_repo)

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    argv = argv_for(runtime, "down")
    build = lifecycle_repo / "build" / "services"
    assert str(build / "docker-compose.yml") in argv
    assert str(build / "event_dispatcher" / "docker-compose.yml") in argv


@pytest.mark.usefixtures("no_web")
def test_down_pins_the_project_directory_and_env_file_to_the_repo(lifecycle_repo, runtime):
    """The invocation contract, on the stop path too.

    A ``down`` that resolved its project directory anywhere else would target a
    different compose project than the ``up`` that started the stack, and stop
    nothing.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    run_down(lifecycle_repo)

    argv = argv_for(runtime, "down")
    assert argv[argv.index("--project-directory") + 1] == str(lifecycle_repo)
    env_files = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--env-file"]
    assert env_files[-1] == str(lifecycle_repo / ".env")


@pytest.mark.usefixtures("no_web")
def test_down_never_removes_volumes(lifecycle_repo, runtime):
    """Stopping a deployment is not a way to lose its data."""
    render_build(lifecycle_repo)

    run_down(lifecycle_repo)

    argv = argv_for(runtime, "down")
    assert "-v" not in argv
    assert "--volumes" not in argv
    assert "--rmi" not in argv


@pytest.mark.usefixtures("runtime")
def test_down_stops_the_web_stack_before_the_services(lifecycle_repo, monkeypatch):
    """Web first, or its containers keep host-global names the next deploy wants.

    The two stacks are separate compose invocations against one project, and the
    services ``down`` does not carry ``docker-compose.web.yml`` in its ``-f``
    list — so nothing else would ever stop them.
    """
    order: list[str] = []
    monkeypatch.setattr(
        container_lifecycle,
        "deploy_down_web_terminals",
        lambda config, env, env_file_args, **kwargs: order.append("web"),
    )
    render_build(lifecycle_repo, config=_WEB_CONFIG)

    original_run = container_lifecycle.subprocess.run

    def _tracking_run(cmd, **kwargs):
        if "down" in cmd:
            order.append("services")
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _tracking_run)

    run_down(lifecycle_repo)

    assert order == ["web", "services"]


@pytest.mark.usefixtures("runtime")
def test_down_leaves_the_web_stack_alone_when_it_is_not_enabled(lifecycle_repo, no_web):
    render_build(lifecycle_repo)

    run_down(lifecycle_repo)

    assert no_web == []


@pytest.mark.usefixtures("no_web")
def test_down_is_correct_from_a_subdirectory(lifecycle_repo, runtime, monkeypatch):
    render_build(lifecycle_repo)
    inside = lifecycle_repo / "profiles"
    inside.mkdir(exist_ok=True)
    monkeypatch.chdir(inside)

    result = CliRunner().invoke(down_verb, [])

    assert result.exit_code == 0, result.output
    argv = argv_for(runtime, "down")
    assert argv[argv.index("--project-directory") + 1] == str(lifecycle_repo)


@pytest.mark.usefixtures("runtime", "no_web")
def test_the_working_directory_survives_a_down(lifecycle_repo):
    """``down`` chdirs into the repo and does not execvpe away; it must restore."""
    render_build(lifecycle_repo)
    before = Path.cwd()

    run_down(lifecycle_repo)

    assert Path.cwd() == before


@pytest.mark.usefixtures("no_web")
def test_a_failed_compose_down_says_the_containers_may_still_be_running(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["fail"]["down"] = 1

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    assert "✗ Could not stop this deployment" in result.stderr
    assert "may still be running" in result.stderr
    assert "may still be running" not in result.stdout


# ---------------------------------------------------------------------------
# down: the label fallback, for a build/ that is gone
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_web")
def test_a_wiped_build_falls_back_to_stopping_by_label(lifecycle_repo, runtime):
    """The CC-3 stranding recovery: no compose files, but containers are running.

    Without this the containers are unreachable by any OSPREY verb — ``compose
    down`` needs the files that declared them, and they are exactly what is gone.
    """
    render_build(lifecycle_repo)
    runtime["ps_stdout"] = "abc123\ndef456\n"
    for path in (lifecycle_repo / "build").rglob("docker-compose.yml"):
        path.unlink()
    (lifecycle_repo / "build" / "config.yml").unlink()

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert verbs_in_order(runtime) == ["ps", "stop", "rm"]
    assert argv_for(runtime, "stop")[2:] == ["abc123", "def456"]
    assert argv_for(runtime, "rm")[2:] == ["abc123", "def456"]


_PROJECT = "als-exemplar"
_PROJECT_LABEL = "com.docker.compose.project"
_SERVICE_LABEL = "com.docker.compose.service"


def _row(name: str, service: str, *, project: str = _PROJECT, repo_id: str | None = None) -> dict:
    """One container on the host, as compose labels it."""
    labels = {_PROJECT_LABEL: project, _SERVICE_LABEL: service}
    if repo_id is not None:
        labels[REPO_ID_LABEL] = repo_id
    return {"ID": f"id-{name}", "Names": name, "Labels": labels}


def _record_services_and_wipe_the_build(repo: Path) -> None:
    """The record of a start that drove two services, and a build that is gone."""
    write_deployment_record(
        repo,
        DeploymentRecord("docker", _PROJECT, compose_services=("event-dispatcher", "qmd")),
    )
    render_build(repo)
    for path in (repo / "build").rglob("docker-compose.yml"):
        path.unlink()
    (repo / "build" / "config.yml").unlink()


def _stopped(record: dict) -> list[list[str]]:
    """The targets of every ``stop`` and ``rm`` that ran, in order."""
    return [cmd[2:] for cmd in record["cmds"] if cmd[1:2] in (["stop"], ["rm"])]


@pytest.mark.usefixtures("no_web")
def test_a_wiped_build_stops_a_recorded_services_unlabelled_container(lifecycle_repo, runtime):
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [
        _row("qmd-1", "qmd"),
        _row("dispatcher-1", "event-dispatcher", repo_id=repo_identity(lifecycle_repo)),
    ]

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert verbs_in_order(runtime) == ["ps", "stop", "rm"]
    assert sorted(argv_for(runtime, "stop")[2:]) == ["dispatcher-1", "qmd-1"]
    assert sorted(argv_for(runtime, "rm")[2:]) == ["dispatcher-1", "qmd-1"]


@pytest.mark.usefixtures("no_web")
def test_an_unlabelled_container_of_an_unrecorded_service_is_left(lifecycle_repo, runtime):
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [_row("qmd-1", "qmd"), _row("postgres-1", "postgres")]

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert all("postgres-1" not in targets for targets in _stopped(runtime))
    assert ["qmd-1"] in _stopped(runtime)
    assert (
        f"left 1 container(s) of compose project {_PROJECT} that the last start did not record"
        in result.output
    )


@pytest.mark.usefixtures("no_web")
def test_a_recorded_services_container_of_another_checkout_is_left(lifecycle_repo, runtime):
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [_row("qmd-1", "qmd"), _row("qmd-2", "qmd", repo_id="someone-else")]

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert all("qmd-2" not in targets for targets in _stopped(runtime))
    assert (
        f"left 1 container(s) of compose project {_PROJECT} that another checkout created"
        in result.output
    )


# ---------------------------------------------------------------------------
# down: another copy of this repo holds the project name, so nothing is stopped
# ---------------------------------------------------------------------------


_OTHER_ID = "0123456789ab"
_WORKING_DIR_LABEL = "com.docker.compose.project.working_dir"


def _host_lists(monkeypatch, containers=(), *, fails: bool = False) -> list[tuple]:
    """Install a runtime probe that lists *containers* (name, labels) for the project.

    Returns the ``(runtime, project)`` pairs the probe was built for, so a test
    can tell which runtime and project ``down`` asked about.
    """
    from osprey.deployment.container_ownership import Resource

    built: list[tuple] = []

    class _Probe:
        runtime = "docker"

        def containers_for_project(self, project, *, include_stopped=True):  # noqa: ARG002 - RuntimeProbe's signature
            if fails:
                raise RuntimeError("Cannot connect to the Docker daemon")
            return [Resource("container", name, labels) for name, labels in containers]

        def volumes_for_project(self, project):
            return [Resource("volume", f"{project}_data", {_PROJECT_LABEL: project})]

    def _probe(config, *, runtime=None, project=None):  # noqa: ARG001 - _ownership_probe's signature
        built.append((runtime, project))
        return _Probe()

    monkeypatch.setattr(container_lifecycle, "_ownership_probe", _probe)
    return built


def _other_copy(path: Path | str) -> dict[str, str]:
    """The labels of a container another checkout, at *path*, created under this project."""
    return {_PROJECT_LABEL: _PROJECT, REPO_ID_LABEL: _OTHER_ID, _WORKING_DIR_LABEL: str(path)}


def _stopped_nothing(record: dict) -> bool:
    """No compose ``down`` and no runtime ``stop`` or ``rm`` ran."""
    return not {"down", "stop", "rm"} & set(verbs_in_order(record))


def test_down_with_compose_files_refuses_another_copys_containers(
    lifecycle_repo, runtime, no_web, monkeypatch, tmp_path
):
    other = tmp_path / "other-copy"
    other.mkdir()
    render_build(lifecycle_repo)
    asked = _host_lists(monkeypatch, [("als-exemplar-qmd", _other_copy(other))])

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    flowed = " ".join(result.output.split())
    assert "osprey down will not stop containers from another copy of this repo" in flowed
    assert f"`osprey down --repo {other}`" in flowed
    assert "Nothing was stopped." in flowed
    assert _stopped_nothing(runtime)
    assert no_web == []
    assert asked == [("docker", _PROJECT)]


def test_down_without_compose_files_refuses_another_copys_containers(
    lifecycle_repo, runtime, no_web, monkeypatch, tmp_path
):
    other = tmp_path / "other-copy"
    other.mkdir()
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [_row("qmd-1", "qmd")]
    _host_lists(monkeypatch, [("als-exemplar-qmd", _other_copy(other))])

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    flowed = " ".join(result.output.split())
    assert "osprey down will not stop containers from another copy of this repo" in flowed
    assert f"`osprey down --repo {other}`" in flowed
    assert "Nothing was stopped." in flowed
    assert runtime["cmds"] == []
    assert no_web == []


@pytest.mark.usefixtures("no_web")
def test_down_proceeds_over_its_own_and_unlabelled_containers(lifecycle_repo, runtime, monkeypatch):
    render_build(lifecycle_repo)
    own = {_PROJECT_LABEL: _PROJECT, REPO_ID_LABEL: repo_identity(lifecycle_repo)}
    _host_lists(monkeypatch, [("own-1", own), ("bare-1", {_PROJECT_LABEL: _PROJECT})])

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert verbs_in_order(runtime) == ["down"]


@pytest.mark.usefixtures("no_web")
def test_down_with_the_other_copy_gone_offers_the_rm_command(
    lifecycle_repo, runtime, monkeypatch, tmp_path
):
    render_build(lifecycle_repo)
    _host_lists(monkeypatch, [("als-exemplar-qmd", _other_copy(tmp_path / "gone"))])

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    flowed = " ".join(result.output.split())
    assert "`docker rm -f als-exemplar-qmd`" in flowed
    assert "The project's volumes are untouched and keep their data." in flowed
    assert _stopped_nothing(runtime)


def test_down_refuses_when_the_runtime_cannot_list_the_project(
    lifecycle_repo, runtime, no_web, monkeypatch
):
    render_build(lifecycle_repo)
    _host_lists(monkeypatch, fails=True)

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    flowed = " ".join(result.output.split())
    assert f"Could not check which copy of this repo holds project {_PROJECT}" in flowed
    assert "Cannot connect to the Docker daemon" in flowed
    assert _stopped_nothing(runtime)
    assert no_web == []


@pytest.mark.usefixtures("no_web")
def test_down_without_a_record_runs_no_ownership_check(tmp_path, runtime, monkeypatch):
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)

    def _unasked(*_args, **_kwargs):
        raise AssertionError("a stop with no record names no project to check")

    monkeypatch.setattr(container_lifecycle, "_ownership_probe", _unasked)

    result = run_down(repo)

    assert result.exit_code == 0, result.output
    assert _repo_label_ps(repo) in runtime["cmds"]


@pytest.mark.usefixtures("no_web")
def test_a_stop_given_its_project_by_the_caller_refuses_another_copys_containers(
    lifecycle_repo, runtime, monkeypatch, tmp_path
):
    from osprey.deployment.container_ownership import ForeignCheckoutError

    other = tmp_path / "other-copy"
    other.mkdir()
    _host_lists(monkeypatch, [("als-exemplar-qmd", _other_copy(other))])

    with pytest.raises(ForeignCheckoutError) as caught:
        container_lifecycle.down_deployment(lifecycle_repo, runtime="docker", project=_PROJECT)

    assert caught.value.summary("osprey down") == (
        "osprey down will not stop containers from another copy of this repo"
    )
    assert runtime["cmds"] == []


def _repo_label_ps(repo: Path) -> list[str]:
    return ["docker", "ps", "-aq", "--filter", f"label={REPO_ID_LABEL}={repo_identity(repo)}"]


@pytest.mark.usefixtures("no_web")
def test_a_record_that_names_no_services_sweeps_by_repo_label_and_says_so(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    (lifecycle_repo / "build" / "config.yml").unlink()
    runtime["rows"] = [
        _row("qmd-1", "qmd"),
        _row("qmd-2", "qmd", repo_id=repo_identity(lifecycle_repo)),
    ]

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert argv_for(runtime, "ps") == _repo_label_ps(lifecycle_repo)
    assert _stopped(runtime) == [["id-qmd-2"], ["id-qmd-2"]]
    assert (
        "no compose files in build/; stopping the 1 container(s) labelled for this repo instead"
        in result.output
    )


@pytest.mark.usefixtures("no_web")
def test_with_no_record_the_sweep_is_by_repo_label_and_says_so(tmp_path, runtime):
    repo = build_exemplar_repo(tmp_path / "one")
    render_build(repo)
    (repo / "build" / "config.yml").unlink()
    runtime["rows"] = [_row("qmd-1", "qmd"), _row("qmd-2", "qmd", repo_id=repo_identity(repo))]

    result = run_down(repo)

    assert result.exit_code == 0, result.output
    assert argv_for(runtime, "ps") == _repo_label_ps(repo)
    assert _stopped(runtime) == [["id-qmd-2"], ["id-qmd-2"]]
    assert (
        "no deployment record; stopping the 1 container(s) labelled for this repo instead"
        in result.output
    )

    runtime["rows"] = []
    runtime["cmds"].clear()

    result = run_down(repo)

    assert result.exit_code == 0, result.output
    assert (
        "nothing to stop: no deployment record and no containers labelled for this repo"
        in result.output
    )


@pytest.mark.usefixtures("no_web")
def test_a_stop_given_its_project_by_the_caller_sweeps_by_repo_label_only(lifecycle_repo, runtime):
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [_row("qmd-1", "qmd")]

    container_lifecycle.down_deployment(lifecycle_repo, runtime="docker", project=_PROJECT)

    assert argv_for(runtime, "ps") == _repo_label_ps(lifecycle_repo)
    assert _stopped(runtime) == []


@pytest.mark.usefixtures("no_web")
def test_another_projects_container_of_a_recorded_service_is_never_stopped(lifecycle_repo, runtime):
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [_row("qmd-1", "qmd", project="other")]

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert _stopped(runtime) == []


@pytest.mark.usefixtures("no_web")
def test_nothing_of_the_recorded_services_to_stop_says_so(lifecycle_repo, runtime, caplog):
    import logging

    caplog.set_level(logging.INFO)
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = []

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert (
        "nothing to stop: no compose files in build/ and no container of a service "
        "the last start recorded" in result.output
    )
    assert "may still be running" not in result.output
    assert "may still be running" not in caplog.text


@pytest.mark.usefixtures("no_web")
def test_a_failed_listing_of_the_recorded_project_refuses(lifecycle_repo, runtime):
    _record_services_and_wipe_the_build(lifecycle_repo)
    runtime["rows"] = [_row("qmd-1", "qmd")]
    runtime["fail"] = {"ps": 1}

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    assert "Could not list" in result.output
    assert _stopped(runtime) == []


@pytest.mark.usefixtures("no_web")
def test_the_label_filter_is_the_identity_the_render_baked_in(lifecycle_repo, runtime):
    """One spelling of the repo identity, shared with the render.

    If the sweep derived the identity any other way, it would select nothing on a
    correctly labelled stack and report success — the worst available failure.
    """
    render_build(lifecycle_repo)
    (lifecycle_repo / "build" / "config.yml").unlink()

    run_down(lifecycle_repo)

    argv = argv_for(runtime, "ps")
    expected = f"label={REPO_ID_LABEL}={repo_identity(lifecycle_repo)}"
    assert argv[argv.index("--filter") + 1] == expected
    assert "-aq" in argv


@pytest.mark.usefixtures("no_web")
def test_the_label_sweep_is_scoped_to_this_checkout_only(tmp_path, runtime):
    """Two checkouts of one deployment must not stop each other's containers."""
    one = build_exemplar_repo(tmp_path / "one")
    two = build_exemplar_repo(tmp_path / "two")
    render_build(one)
    (one / "build" / "config.yml").unlink()

    run_down(one)

    argv = argv_for(runtime, "ps")
    assert f"label={REPO_ID_LABEL}={repo_identity(one)}" in argv
    assert repo_identity(two) not in " ".join(argv)


@pytest.mark.usefixtures("no_web")
def test_the_label_sweep_removes_no_volumes(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["ps_stdout"] = "abc123\n"
    (lifecycle_repo / "build" / "config.yml").unlink()

    run_down(lifecycle_repo)

    flat = [token for cmd in runtime["cmds"] for token in cmd]
    assert "volume" not in flat
    assert "-v" not in flat
    assert "--volumes" not in flat


@pytest.mark.usefixtures("no_web")
def test_finding_no_labelled_container_does_not_claim_the_stack_is_down(
    lifecycle_repo,
    runtime,
    caplog,
):
    """The honest limit, said out loud rather than implied by silence.

    Labels are applied at container CREATE time, so a stack started before this
    version of OSPREY carries none and is invisible here. Reporting a bare
    "done" would tell an operator their stack is stopped when it is running.
    """
    import logging

    caplog.set_level(logging.INFO)
    render_build(lifecycle_repo)
    runtime["ps_stdout"] = ""
    (lifecycle_repo / "build" / "config.yml").unlink()

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert verbs_in_order(runtime) == ["ps"]
    # All five elements of the message, because each carries a different part of
    # the answer: WHERE it looked, WHAT it looked for, WHY that can miss, what
    # may therefore still be true, and what to do about it.
    #
    # Log seam, not the renderer: this path's emitters live in
    # container_lifecycle and web_terminals/provision, which stayed on the
    # logger. The refusal assertions elsewhere in this module read
    # result.stderr because their emitter moved to the renderer; both are
    # deliberate, and which one applies is decided by who emits.
    text = caplog.text
    assert str(lifecycle_repo / "build") in text
    assert REPO_ID_LABEL in text
    assert "label when they are created" in text
    assert "may still be running" in text
    assert "osprey build" in text


@pytest.mark.usefixtures("no_web")
def test_a_failed_label_listing_refuses_rather_than_reporting_success(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["fail"]["ps"] = 1
    (lifecycle_repo / "build" / "config.yml").unlink()

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    assert "Could not list" in result.stderr
    assert "Could not list" not in result.stdout
    assert verbs_in_order(runtime) == ["ps"]


@pytest.mark.usefixtures("no_web")
def test_a_failed_label_stop_says_the_containers_may_still_be_running(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["ps_stdout"] = "abc123\n"
    runtime["fail"]["stop"] = 1
    (lifecycle_repo / "build" / "config.yml").unlink()

    result = run_down(lifecycle_repo)

    assert result.exit_code != 0
    assert "may still be running" in result.stderr
    assert "may still be running" not in result.stdout


@pytest.mark.usefixtures("no_web")
def test_a_build_with_a_config_but_no_compose_files_still_stops_by_label(lifecycle_repo, runtime):
    """The config survived; the rendered compose files did not. Same recovery."""
    render_build(lifecycle_repo, with_compose=False)
    runtime["ps_stdout"] = "abc123\n"

    result = run_down(lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert verbs_in_order(runtime) == ["ps", "stop", "rm"]


@pytest.mark.usefixtures("no_web")
def test_the_label_sweep_stops_before_it_removes(lifecycle_repo, runtime):
    """``stop`` then ``rm``, as ``compose down`` does it.

    A bare ``rm -f`` would SIGKILL the containers instead of giving them the
    shutdown grace period the compose path gives them.
    """
    render_build(lifecycle_repo)
    runtime["ps_stdout"] = "abc123\n"
    (lifecycle_repo / "build" / "config.yml").unlink()

    run_down(lifecycle_repo)

    assert verbs_in_order(runtime).index("stop") < verbs_in_order(runtime).index("rm")
    assert "-f" not in argv_for(runtime, "rm")


# ---------------------------------------------------------------------------
# restart: down then up, never `compose restart`
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_web")
def test_restart_stops_and_then_starts(lifecycle_repo, runtime):
    """Recreate, do not resume.

    ``compose restart`` restarts each container against the definition it was
    created with, so a rebuilt image, an edited compose file or a newly minted
    token reaches nothing. That is the whole reason this verb changed.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    result = run_restart(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output
    order = verbs_in_order(runtime)
    assert "restart" not in order
    assert order.index("down") < order.index("up")


@pytest.mark.usefixtures("runtime", "no_web")
def test_restart_never_renders_the_services_stack(lifecycle_repo):
    """The ``prepare_compose_files`` stub in the fixture raises if it is reached."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    result = run_restart(lifecycle_repo, "-d")

    assert result.exit_code == 0, result.output


@pytest.mark.usefixtures("no_web")
def test_restart_starts_the_compose_files_the_build_left(lifecycle_repo, runtime):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    run_restart(lifecycle_repo, "-d")

    build = lifecycle_repo / "build" / "services"
    assert str(build / "event_dispatcher" / "docker-compose.yml") in argv_for(runtime, "up")


# ---------------------------------------------------------------------------
# restart: the drift gate, and refusing before anything is stopped
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_web")
def test_drift_refuses_the_restart_and_stops_nothing(lifecycle_repo, runtime):
    """The property that makes the gate worth having on this verb.

    A restart that stopped the stack and *then* discovered it could not start it
    would leave the deployment down, which is strictly worse than the ungated
    re-render it replaces.
    """
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash="stale")

    result = run_restart(lifecycle_repo, "-d")

    assert result.exit_code != 0
    assert runtime["cmds"] == []
    assert "osprey restart --build" in result.stderr
    assert "osprey restart --as-built" in result.stderr
    assert "osprey restart --build" not in result.stdout


@pytest.mark.usefixtures("runtime", "no_web")
def test_the_drift_refusal_names_this_verb_not_up(lifecycle_repo):
    """The remedies have to be commands the operator can actually type."""
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash="stale")

    result = run_restart(lifecycle_repo, "-d")

    assert "osprey up --build" not in result.output


@pytest.mark.usefixtures("no_web")
def test_as_built_restarts_the_drifted_build_and_says_so(lifecycle_repo, runtime):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash="stale")

    result = run_restart(lifecycle_repo, "-d", "--as-built")

    assert result.exit_code == 0, result.output
    assert verbs_in_order(runtime).index("down") < verbs_in_order(runtime).index("up")
    assert "⚠ Starting build/ as it was rendered (--as-built)" in result.stderr
    assert "Starting build/ as it was rendered (--as-built)" not in result.stdout


@pytest.mark.usefixtures("no_web")
def test_build_chains_the_render_then_stops_and_starts(lifecycle_repo, runtime, monkeypatch):
    chained: list = []

    def _fake_chain(_ctx, repo_root, *, dev=False):  # noqa: ARG001 - _chain_build's dev keyword
        chained.append(repo_root)

    import osprey.cli.deploy_cmd as deploy_cmd

    monkeypatch.setattr(deploy_cmd, "_chain_build", _fake_chain)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo, stamped_hash="stale")

    result = run_restart(lifecycle_repo, "-d", "--build")

    assert result.exit_code == 0, result.output
    assert chained == [lifecycle_repo]
    assert verbs_in_order(runtime).index("down") < verbs_in_order(runtime).index("up")


@pytest.mark.usefixtures("no_web")
def test_build_and_as_built_are_refused_together(lifecycle_repo, runtime):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    result = run_restart(lifecycle_repo, "-d", "--build", "--as-built")

    assert result.exit_code != 0
    assert "opposites" in result.output
    assert runtime["cmds"] == []


@pytest.mark.usefixtures("no_web")
def test_a_missing_build_refuses_and_stops_nothing(lifecycle_repo, runtime):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")

    result = run_restart(lifecycle_repo, "-d")

    assert result.exit_code != 0
    assert runtime["cmds"] == []
    assert "→ run `osprey build` first" in result.stderr
    assert "run `osprey build` first" not in result.stdout


@pytest.mark.usefixtures("no_web")
def test_as_built_is_not_an_escape_from_having_no_build(lifecycle_repo, runtime):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")

    result = run_restart(lifecycle_repo, "-d", "--as-built")

    assert result.exit_code != 0
    assert runtime["cmds"] == []
    assert "--as-built starts a build that already exists" in result.stderr
    assert "--as-built starts a build that already exists" not in result.stdout


@pytest.mark.usefixtures("no_web")
def test_a_build_rendered_elsewhere_refuses_the_restart_and_stops_nothing(
    lifecycle_repo, runtime, tmp_path
):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bake_labels_override(lifecycle_repo, elsewhere)

    result = run_restart(lifecycle_repo, "-d")

    assert result.exit_code != 0
    assert runtime["cmds"] == []
    assert f"this build was rendered in {elsewhere}" in result.stderr
    assert "→ run `osprey build` here" in result.stderr


@pytest.mark.usefixtures("no_web")
def test_dev_mode_refuses_before_anything_is_stopped(lifecycle_repo, runtime, monkeypatch):
    """A ``--dev`` that cannot stage a wheel must not take the stack down first."""
    from osprey.deployment.errors import DevModeUnavailableError

    def _refuse():
        raise DevModeUnavailableError(reason="no wheel", remedy="build one")

    monkeypatch.setattr("osprey.deployment.wheel_build.preflight_dev_mode", _refuse)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)

    result = run_restart(lifecycle_repo, "-d", "--dev")

    assert result.exit_code != 0
    assert runtime["cmds"] == []
    assert "Nothing was stopped" in result.stderr
    assert "Nothing was stopped" not in result.stdout


@pytest.mark.usefixtures("no_web")
def test_a_missing_env_refuses_before_anything_is_stopped(lifecycle_repo, runtime):
    render_build(lifecycle_repo)

    result = run_restart(lifecycle_repo, "-d")

    assert result.exit_code != 0
    assert runtime["cmds"] == []
    # Markless: the refusal is raised inside the open Preflight phase, whose own
    # ✗ on the way out is this run's one failure marker.
    assert "No .env in" in result.stderr
    assert "No .env in" not in result.stdout
    assert result.stdout.count("✗") == 1, result.stdout
    assert result.stderr.count("✗") == 0, result.stderr


# ---------------------------------------------------------------------------
# Both verbs are reachable, and resolve to the new commands rather than the old
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["down", "restart"])
def test_the_top_level_verb_resolves_to_the_new_command_not_the_legacy_one(name):
    """The registration hazard, which fails silently if it is wrong.

    ``deploy_cmd`` holds two commands per name: the top-level verb
    (``down_verb``) and the legacy ``deploy`` subcommand (``down``). The lazy
    group's default lookup is ``getattr(mod, cmd_name)``, which returns the
    LEGACY one — a working command with a different option surface, so nothing
    raises and ``osprey down`` quietly means the old thing.
    """
    from osprey.cli.deploy_cmd import down_verb, restart_verb
    from osprey.cli.main import cli

    expected = {"down": down_verb, "restart": restart_verb}[name]
    resolved = cli.get_command(None, name)

    assert resolved is expected
    assert name in cli.list_commands(None)


@pytest.mark.parametrize("name", ["down", "restart"])
def test_the_top_level_verb_takes_no_project_flag(name):
    """Zero-arg via the resolver, not via the legacy ``--project``/``--config``.

    The clearest signal that the wrong command object was registered: the legacy
    subcommands are project-scoped, these are repo-scoped.
    """
    from osprey.cli.main import cli

    params = {param.name for param in cli.get_command(None, name).params}

    assert "repo" in params
    assert "project" not in params
    assert "config" not in params


# ---------------------------------------------------------------------------
# restart: the token mint follows the repo, never the working directory
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_web")
def test_restart_mints_into_the_repo_env_from_any_directory(lifecycle_repo, runtime, monkeypatch):
    """Carry-forward from ``up``: the mint path is passed explicitly.

    ``_ensure_service_tokens`` falls back to a cwd-relative ``.env``. If restart
    inherited that fallback, the tokens would land in whatever directory the
    process was standing in while ``--env-file`` still pointed at the repo's
    ``.env``, and the stack would come up with its fail-closed tokens unset —
    secure-looking, and silent.
    """
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    elsewhere = lifecycle_repo.parent / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    container_lifecycle.restart_deployment(lifecycle_repo, detached=True)

    from osprey_connectors.dotenv import parse_dotenv_file

    assert parse_dotenv_file(lifecycle_repo / ".env")["EVENT_DISPATCHER_TOKEN"]
    assert not (elsewhere / ".env").exists()
    argv = argv_for(runtime, "up")
    env_files = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--env-file"]
    assert env_files[-1] == str(lifecycle_repo / ".env")


@pytest.mark.usefixtures("runtime", "no_web")
def test_the_working_directory_survives_a_restart(lifecycle_repo):
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    render_build(lifecycle_repo)
    before = Path.cwd()

    run_restart(lifecycle_repo, "-d")

    assert Path.cwd() == before


# ---------------------------------------------------------------------------
# The help text promises exactly what the code does
# ---------------------------------------------------------------------------


def test_restart_help_does_not_claim_it_renders_nothing_at_all():
    """The web-terminal stack IS re-rendered at every start, ``restart`` included.

    An unqualified "renders nothing" would be the run's recurring defect —
    prose promising more than the code delivers — and here it would tell an
    operator a roster edit needs a rebuild when it does not.
    """
    help_text = restart_verb.__doc__ or ""
    assert "re-renders nothing from profile.yml" in help_text
    assert "roster" in help_text
    assert "Web terminals are the same documented exception" in help_text


def test_down_help_does_not_promise_the_fallback_finds_everything():
    """It cannot find a container created before OSPREY labelled them."""
    help_text = down_verb.__doc__ or ""
    assert "CREATED" in help_text
    assert "osprey build" in help_text


def test_down_help_does_not_offer_flags_the_verb_does_not_have():
    """``osprey down`` took ``--dev`` to re-render compose files. This never renders."""
    assert "--dev" not in (down_verb.__doc__ or "")
    assert {param.name for param in down_verb.params} == {"repo"}


def test_restart_help_states_the_build_flag_orphan_limit():
    """``down`` never passes ``--remove-orphans``, so a deleted service keeps running.

    Both stacks share one ``COMPOSE_PROJECT_NAME``, which is why orphan removal
    is deliberately absent everywhere on this path — so the limit is real and
    saying so is cheaper than an operator finding a stale container by accident.
    """
    help_text = restart_verb.__doc__ or ""
    assert "no longer named in them and" in help_text
    assert "osprey down` before `osprey build" in help_text

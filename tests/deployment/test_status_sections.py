"""``osprey status`` and ``osprey logs`` — the two read-only lifecycle verbs.

Four properties, and everything here serves one of them.

**Neither verb changes anything.** No render, no start, no stop, no write into
the repo. The tests assert that positively — the tree is compared file by file
across a status run, and the only runtime commands either verb is allowed to
issue are listings and ``logs``.

**Status answers about THIS checkout.** Two clones of one deployment on a single
host share a project name and a volume namespace, so containers are sorted by
the ``com.osprey.repo-id`` label a build bakes in. The tests pin all four
outcomes of that sort, including the two that are easy to get quietly wrong: a
container from the *other* checkout must never be reported as this
deployment's, and a container carrying no label at all must be reported with the
reason it cannot be found by label.

**Status never reports a failure as an absence.** A runtime that cannot be
queried is not an empty deployment, and an artifact check that cannot run is not
an artifact check that passed. Both say so.

**Logs is a wrapper, not a reimplementation.** It builds its invocation through
the pinned compose contract, carries both stacks' compose files, and pins
``COMPOSE_PROJECT_NAME`` — the three things that decide whether compose looks at
this deployment's containers or at nothing at all.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

import osprey.cli.deploy_cmd as deploy_cmd
from osprey.cli import styles
from osprey.cli.deploy_cmd import logs_verb, status_verb
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.styles import osprey_theme
from osprey.deployment import status_display
from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity
from tests.deployment.test_up_as_built import _RENDERED_CONFIG, render_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

PROJECT = "als-exemplar"


@pytest.fixture
def lifecycle_repo(tmp_path: Path) -> Path:
    """The gold-standard three-zone deployment repo, with no ``.env`` in it."""
    return build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)


def container(
    name: str,
    *,
    state: str = "running",
    project: str | None = PROJECT,
    repo_id: str | None = None,
    ports: list[dict] | None = None,
) -> dict:
    """One ``ps --format json`` record, in podman's object-labels shape."""
    labels: dict[str, str] = {}
    if project is not None:
        labels[status_display.PROJECT_LABEL] = project
    if repo_id is not None:
        labels[REPO_ID_LABEL] = repo_id
    return {
        "Names": [name],
        "Labels": labels,
        "State": state,
        "Ports": ports or [],
        "Image": "example:local",
    }


@pytest.fixture
def runtime(monkeypatch):
    """Stub the container runtime; record every argv, and let tests script ``ps``.

    Returns the record dict. ``containers`` is what ``ps`` answers with (written
    by a test before it acts); ``volumes`` is what ``volume ls`` answers with;
    ``ps_returncode`` scripts a failed query.
    """
    record: dict = {"cmds": [], "containers": [], "volumes": [], "ps_returncode": 0}

    monkeypatch.setattr(
        status_display,
        "get_ps_command",
        lambda config, all_containers=False: ["docker", "ps", "-a", "--format", "json"],
    )
    monkeypatch.setattr(status_display, "get_runtime_command", lambda config=None: ["docker"])

    def _fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        argv = list(cmd)
        record["cmds"].append(argv)
        if argv[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(
                argv, record["ps_returncode"], json.dumps(record["containers"]), ""
            )
        if argv[:2] == ["docker", "volume"]:
            return subprocess.CompletedProcess(argv, 0, "\n".join(record["volumes"]), "")
        raise AssertionError(f"a read-only verb ran: {argv}")

    monkeypatch.setattr(status_display.subprocess, "run", _fake_run)
    return record


#: Where :func:`report` reads the run it just made. The renderer resolves its
#: own console per call, so a test cannot hand one in — the fixture below
#: installs a reporter that answers with a recording console and leaves it here.
_CAPTURE: dict[str, Console] = {}


@pytest.fixture(autouse=True)
def renderer_capture(monkeypatch):
    """Capture BOTH renderer streams into one wide recording console.

    Status prints through :mod:`osprey.cli.output`, which sends report/note/
    section lines to the installed reporter's console and warnings/failures to
    ``styles.err_console``. Both are pointed at one console here so an assertion
    reads what an operator sees, whichever stream a line went out on.

    Wide deliberately: the console wraps at the terminal width, and a wrapped
    sentence defeats every substring assertion below. What is under test is what
    status *says*, not how Rich folds it.
    """
    console = Console(record=True, width=200, theme=osprey_theme)

    class _Recording(PhaseReporter):
        # Untyped on purpose: rich's ``Console`` is Any to this repo's mypy
        # settings, so an annotated override trips ``no-any-return``.
        def out(self):
            return console

    previous = install_reporter(_Recording(color=False))
    monkeypatch.setattr(styles, "err_console", console)
    _CAPTURE["console"] = console
    yield console
    _CAPTURE.clear()
    install_reporter(previous)


def report(repo: Path, **kwargs) -> str:
    """Render ``osprey status`` for *repo* and return everything it printed."""
    status_display.show_repo_status(repo, **kwargs)
    return _CAPTURE["console"].export_text()


# ---------------------------------------------------------------------------
# Build: the drift verdict and the version line
# ---------------------------------------------------------------------------


def test_a_clean_build_is_reported_as_in_sync(lifecycle_repo, runtime):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "in sync with profile.yml" in text


def test_drift_names_what_moved_and_that_the_start_verbs_refuse(lifecycle_repo, runtime):
    """The same verdict ``up`` refuses on, so a refusal is never a surprise here."""
    render_build(lifecycle_repo)
    profile = lifecycle_repo / "profile.yml"
    profile.write_text(profile.read_text(encoding="utf-8") + "\nmodel: opus\n", encoding="utf-8")

    text = report(lifecycle_repo)

    assert "OUT OF DATE" in text
    assert "osprey up` and `osprey restart` refuse" in text


def test_a_repo_with_no_build_says_so_and_still_shows_its_containers(lifecycle_repo, runtime):
    """The case an operator most needs status for: build/ deleted, stack still up."""
    runtime["containers"] = [container("dispatch", repo_id=repo_identity(lifecycle_repo))]

    text = report(lifecycle_repo)

    assert "none — run `osprey build`" in text
    assert "dispatch" in text
    # The two sections that are read out of build/config.yml say they cannot be
    # computed rather than not appearing: two silently absent sections read as a
    # deployment with no endpoints and no agent.
    assert "read from build/config.yml" in text
    # And the project name in the header is labelled as the derivation it is.
    assert "from the directory name" in text


def test_the_version_line_is_printed_when_there_is_no_skew(lifecycle_repo, runtime):
    """A build rendered by the installed version still reports which one that was.

    ``version_skew`` is populated only when the two versions DIFFER, so a report
    without one cannot distinguish "they matched" from "the build recorded no
    version" — and the line has to.
    """
    from osprey.cli.templates.manifest import get_framework_release_version

    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert f"rendered by osprey {get_framework_release_version()} (installed)" in text


def test_version_skew_is_reported_as_skew(lifecycle_repo, runtime):
    render_build(lifecycle_repo, version="1900.1.0")

    text = report(lifecycle_repo)

    assert "rendered by osprey 1900.1.0" in text
    assert "Re-run `osprey build`" in text


def test_a_build_that_records_no_version_says_it_recorded_none(lifecycle_repo, runtime):
    """Not the same as "matches", and not something to leave unsaid."""
    build = render_build(lifecycle_repo)
    manifest_path = build / ".osprey-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["creation"].pop("osprey_version", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    text = report(lifecycle_repo)

    assert "records no version that rendered it" in text


# ---------------------------------------------------------------------------
# Containers: which of them are this checkout's
# ---------------------------------------------------------------------------


def test_a_container_labelled_for_this_checkout_is_this_deployments(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["containers"] = [container("dispatch", repo_id=repo_identity(lifecycle_repo))]

    text = report(lifecycle_repo)

    assert "dispatch" in text
    assert "Nothing is running for this deployment" not in text


def test_another_checkouts_containers_are_not_reported_as_this_ones(lifecycle_repo, runtime):
    """The defect this label exists to prevent, and it cannot be seen by name.

    Two clones of one deployment share a project name, so a project-name match
    would fold the other checkout's containers into this one's table — and an
    operator would read "running" off a stack this repo never started.
    """
    render_build(lifecycle_repo)
    runtime["containers"] = [container("dispatch", repo_id="ffffffffffff")]

    text = report(lifecycle_repo)

    assert "DIFFERENT copy" in text
    assert "ffffffffffff" in text
    assert repo_identity(lifecycle_repo) in text
    assert "Nothing is running for this deployment" in text


def test_the_other_checkout_is_not_described_as_safe_from_this_ones_verbs(lifecycle_repo, runtime):
    """Naming them is not enough — the operator's next move is usually ``down``.

    Both checkouts render the same project_name, so both sets of containers
    carry the same ``com.docker.compose.project``. The compose ``down`` this
    repo runs selects on that shared name and stops the other checkout's
    containers too. Saying "this repo's verbs act on its own containers only"
    would be reassuring and wrong in the direction that costs someone their
    running stack.
    """
    render_build(lifecycle_repo)
    runtime["containers"] = [container("dispatch", repo_id="ffffffffffff")]

    text = report(lifecycle_repo)

    # Collapse the console's line wrapping: the phrase may break mid-sentence.
    assert "stops them too" in " ".join(text.split())
    assert "act on its own containers only" not in text


def test_an_unlabelled_container_is_shown_and_the_label_limit_is_stated(lifecycle_repo, runtime):
    """These are exactly the containers ``osprey down``'s label fallback misses.

    Labels are applied at container CREATE time, so a stack started before this
    version carries none. Showing it without saying that would leave an operator
    who deletes ``build/`` and runs ``down`` reading "nothing found" while it
    keeps running.
    """
    render_build(lifecycle_repo)
    runtime["containers"] = [container("legacy-dispatch", repo_id=None)]

    text = report(lifecycle_repo)

    assert "legacy-dispatch" in text
    assert REPO_ID_LABEL in text
    assert "label fallback" in text


def test_another_projects_containers_are_listed_separately(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["containers"] = [container("someone-else", project="other-project")]

    text = report(lifecycle_repo)

    assert "Other OSPREY containers on this host" in text
    assert "someone-else" in text


def test_a_container_that_is_not_ospreys_appears_nowhere(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["containers"] = [container("some-postgres", project=None)]

    text = report(lifecycle_repo)

    assert "some-postgres" not in text


def test_a_failed_runtime_query_is_not_reported_as_an_empty_deployment(lifecycle_repo, runtime):
    """The one wrong answer that looks right: a stopped daemon rendering as "nothing deployed"."""
    render_build(lifecycle_repo)
    runtime["ps_returncode"] = 1

    text = report(lifecycle_repo)

    assert "Could not query container status" in text
    assert "Nothing is running for this deployment" not in text


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_endpoints_come_from_the_rendered_compose_files(lifecycle_repo, runtime):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "Service endpoints" in text
    assert "127.0.0.1:18020" in text


def test_the_endpoint_list_does_not_claim_anything_is_answering(lifecycle_repo, runtime):
    """It is read out of build/, not probed. Saying otherwise would be the run's defect."""
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "nothing was contacted to check" in text


# ---------------------------------------------------------------------------
# Agent: provider, credential, artifacts
# ---------------------------------------------------------------------------


def test_a_credential_that_lives_only_in_the_repo_env_counts_as_found(
    lifecycle_repo, runtime, monkeypatch
):
    """``.env`` is the deployment's secret store; a check against os.environ alone lies.

    Every container reads its credentials from this file. A status that only
    consulted the exported shell would report a correctly configured deployment
    as unauthenticated.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n", encoding="utf-8")

    text = report(lifecycle_repo)

    assert "ANTHROPIC_API_KEY" in text
    assert "set in .env" in text
    assert "sk-secret" not in text


def test_a_credential_that_is_nowhere_is_reported_missing(lifecycle_repo, runtime, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "not found in this shell" in text


def test_the_artifact_check_mirrors_the_arguments_the_build_renders_with(
    lifecycle_repo, runtime, monkeypatch
):
    """A dry run that does not reproduce the build's own render arguments reports
    drift that does not exist.

    ``osprey build`` renders the agent artifacts with ``project_root`` set to the
    REPO root and the runtime venv at ``build/.venv``. Omit either and the
    comparison renders a different interpreter path and project root into
    ``.mcp.json``, then reports the difference as drift — sending an operator to
    rebuild a build that was already current.

    The rendered config carries ``project_root`` EXPLICITLY, which the shared
    ``_RENDERED_CONFIG`` does not. Without it this test passed through
    ``_artifact_drift``'s ``or repo_root`` fallback and proved only that the
    fallback happens to produce the same answer — the branch a real build
    exercises, where the value is read back off the config the build recorded,
    went untested while the assertion read as though it pinned exactly that.
    The fallback has its own test below.
    """
    render_build(lifecycle_repo, config=_RENDERED_CONFIG + f"project_root: {lifecycle_repo}\n")
    calls: list[dict] = []

    class _Manager:
        def regenerate_claude_code(self, project_dir, **kwargs):
            calls.append({"project_dir": project_dir, **kwargs})
            return {"changed": [], "unchanged": ["x"]}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    report(lifecycle_repo)

    assert calls == [
        {
            "project_dir": lifecycle_repo / "build",
            "dry_run": True,
            "project_root_override": str(lifecycle_repo),
            "runtime_venv_dir": lifecycle_repo / "build",
        }
    ]


def test_a_config_without_project_root_falls_back_to_the_repo_root(
    lifecycle_repo, runtime, monkeypatch
):
    """The fallback, tested as a fallback rather than as an accident.

    A render that recorded no ``project_root`` — an older build, or one written
    by a path that did not set it — still has to produce a dry run anchored
    somewhere real. The repo root is the right answer, and it is a DIFFERENT
    code path from reading the key, so it gets its own test instead of being
    the thing the test above silently exercised.
    """
    render_build(lifecycle_repo)
    assert "project_root" not in _RENDERED_CONFIG, (
        "this test's subject is a config with no project_root; the shared "
        "fixture now sets one, so it is no longer testing the fallback"
    )
    calls: list[dict] = []

    class _Manager:
        def regenerate_claude_code(self, project_dir, **kwargs):
            calls.append({"project_dir": project_dir, **kwargs})
            return {"changed": [], "unchanged": ["x"]}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    report(lifecycle_repo)

    assert calls[0]["project_root_override"] == str(lifecycle_repo)


def test_a_container_targeted_build_is_not_told_to_probe_a_host_venv(
    lifecycle_repo, runtime, monkeypatch
):
    """``--runtime-root`` records a path inside a container, which has no venv here."""
    render_build(
        lifecycle_repo,
        config=_RENDERED_CONFIG + "project_root: /app/als-exemplar\n",
    )
    calls: list[dict] = []

    class _Manager:
        def regenerate_claude_code(self, project_dir, **kwargs):
            calls.append(kwargs)
            return {"changed": [], "unchanged": []}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    report(lifecycle_repo)

    assert calls[0]["project_root_override"] == "/app/als-exemplar"
    assert calls[0]["runtime_venv_dir"] is None


def test_out_of_sync_artifacts_name_the_rebuild(lifecycle_repo, runtime, monkeypatch):
    render_build(lifecycle_repo)

    class _Manager:
        def regenerate_claude_code(self, project_dir, **kwargs):
            return {"changed": [".claude/settings.json"], "unchanged": []}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    text = report(lifecycle_repo)

    assert "no longer match" in text
    assert "osprey build" in text
    assert ".claude/settings.json" in text


def test_an_artifact_check_that_cannot_run_is_not_a_passing_one(
    lifecycle_repo, runtime, monkeypatch
):
    """A template that will not render is a fact about the build, not a clean bill."""
    render_build(lifecycle_repo)

    class _Manager:
        def regenerate_claude_code(self, project_dir, **kwargs):
            raise RuntimeError("template exploded")

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    text = report(lifecycle_repo)

    assert "could not be checked" in text
    # Scoped to the artifact verdict: "in sync" alone also matches the BUILD
    # section's own clean verdict two sections above, and would pass here
    # whatever the artifact check reported.
    assert "in sync (" not in text


def test_the_per_agent_model_table_is_opt_in(lifecycle_repo, runtime):
    """A dozen model rows would bury the two lines an operator came for."""
    render_build(lifecycle_repo)

    default = report(lifecycle_repo)
    with_agents = report(lifecycle_repo, show_agents=True)

    assert "agent models" not in default
    assert "agent models" in with_agents
    assert "model tiers" in default


# ---------------------------------------------------------------------------
# Status changes nothing
# ---------------------------------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_status_writes_nothing_into_the_repo(lifecycle_repo, runtime):
    """Including the artifact check, which renders into a temporary directory."""
    render_build(lifecycle_repo)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    before = _tree(lifecycle_repo)

    report(lifecycle_repo)

    assert _tree(lifecycle_repo) == before


def test_status_never_announces_creating_the_files_it_did_not_create(
    lifecycle_repo, runtime, capsys
):
    """The artifact renderer must not narrate itself inside a read-only report.

    Rendering into a temporary directory still runs the writer. Any "Created N
    …" sentence it puts on stdout would tell an operator that this read-only
    command just rewrote their repo. The renderer now keeps that detail in the
    log rather than on the terminal, and this test is the guard that it stays
    there: it runs the real renderer, not a stub.

    Pinned on BOTH halves of what a terminal shows: the process's own stdout,
    where the artifact writer would print, and the text status itself renders.
    Either one alone leaves a hole, since status prints through a console rather
    than at stdout directly.
    """
    render_build(lifecycle_repo)
    capsys.readouterr()

    text = report(lifecycle_repo)

    assert "Created" not in capsys.readouterr().out
    assert "Created" not in text


def test_status_only_ever_asks_the_runtime_to_list(lifecycle_repo, runtime):
    """``ps`` and ``volume ls``. Any other verb reaching the runtime is a bug.

    The stub raises on anything else, so this asserts the shape of what DID run
    rather than trusting that nothing else could have.
    """
    render_build(lifecycle_repo)
    runtime["containers"] = [container("dispatch", repo_id=repo_identity(lifecycle_repo))]

    report(lifecycle_repo)

    assert runtime["cmds"]
    for argv in runtime["cmds"]:
        assert argv[1] in ("ps", "volume"), argv


# ---------------------------------------------------------------------------
# logs: the invocation it hands to the runtime
# ---------------------------------------------------------------------------


def logs_argv(repo: Path, **kwargs) -> list[str]:
    cmd, _env = status_display.logs_command(repo, **kwargs)
    return cmd


def test_logs_is_built_through_the_pinned_compose_contract(lifecycle_repo, runtime):
    """One base for every invocation: project directory, repo-anchored -f, env file."""
    render_build(lifecycle_repo)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")

    argv = logs_argv(lifecycle_repo)

    assert argv[argv.index("--project-directory") + 1] == str(lifecycle_repo)
    env_files = [argv[i + 1] for i, flag in enumerate(argv) if flag == "--env-file"]
    assert env_files[-1] == str(lifecycle_repo / ".env")
    compose_files = [argv[i + 1] for i, token in enumerate(argv) if token == "-f"]
    assert compose_files
    assert all(path.startswith(str(lifecycle_repo / "build")) for path in compose_files)
    assert argv[-1] == "logs"


def test_follow_is_passed_through(lifecycle_repo, runtime):
    render_build(lifecycle_repo)

    assert "--follow" in logs_argv(lifecycle_repo, follow=True)
    assert "--follow" not in logs_argv(lifecycle_repo)


def test_tail_is_passed_through_and_otherwise_left_to_the_runtime(lifecycle_repo, runtime):
    """No invented default: an unset ``--tail`` means compose's own behaviour."""
    render_build(lifecycle_repo)

    argv = logs_argv(lifecycle_repo, tail=50)

    assert argv[argv.index("--tail") + 1] == "50"
    assert "--tail" not in logs_argv(lifecycle_repo)


def test_a_named_service_is_the_last_argument(lifecycle_repo, runtime):
    render_build(lifecycle_repo)

    argv = logs_argv(lifecycle_repo, service="event-dispatcher", follow=True)

    assert argv[-1] == "event-dispatcher"


def test_the_web_stack_is_carried_when_the_deployment_has_one(lifecycle_repo, runtime):
    """Two compose invocations start the deployment; one reads its logs.

    They are a single compose project, so a ``logs`` that carried only the
    services file would silently omit every web-terminal container.
    """
    from osprey.deployment.web_terminals.artifacts import web_compose_file

    render_build(lifecycle_repo)
    assert str(web_compose_file(lifecycle_repo)) not in logs_argv(lifecycle_repo)

    web_compose_file(lifecycle_repo).write_text("services: {}\n", encoding="utf-8")

    assert str(web_compose_file(lifecycle_repo)) in logs_argv(lifecycle_repo)


def test_logs_does_not_warn_about_services_starting(lifecycle_repo, runtime, caplog):
    """The shared ``--env-file`` resolver warns about a missing ``.env`` in terms
    of what the stack will come up with. Nothing comes up here.

    True for every other caller of it, false for this one, and a wrong sentence
    about credentials is the kind an operator acts on.
    """
    render_build(lifecycle_repo)
    assert not (lifecycle_repo / ".env").exists()

    argv = logs_argv(lifecycle_repo)

    assert "--env-file" not in argv
    assert "will start" not in caplog.text


def test_logs_pins_the_compose_project_name(lifecycle_repo, runtime):
    """Left unset, compose derives a project from the directory and finds nothing —
    which prints an empty log and looks like a deployment that logs nothing."""
    render_build(lifecycle_repo)

    _cmd, env = status_display.logs_command(lifecycle_repo)

    assert env["COMPOSE_PROJECT_NAME"] == PROJECT


def test_logs_without_a_build_refuses_and_names_the_remedy(lifecycle_repo):
    with pytest.raises(status_display.NoComposeFilesError) as excinfo:
        status_display.logs_command(lifecycle_repo)

    # The remedy is a field, not prose spliced into the message — that is what
    # lets one CLI handler render every precondition refusal the same way.
    assert "osprey build" in excinfo.value.remedy


def test_logs_renders_nothing(lifecycle_repo, runtime, monkeypatch):
    """A read verb that re-rendered would answer about a stack that was never started."""
    from osprey.deployment import container_lifecycle

    def _forbidden(*args, **kwargs):
        raise AssertionError("reading logs must not render compose files")

    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", _forbidden)
    render_build(lifecycle_repo)

    logs_argv(lifecycle_repo)


def test_the_cli_reports_a_missing_build_rather_than_raising(lifecycle_repo):
    result = CliRunner().invoke(logs_verb, ["--repo", str(lifecycle_repo)])

    assert result.exit_code != 0
    # The shared unmet-precondition handler renders the reason and the one
    # remedy on the console, the same way every other deploy refusal does.
    assert "No build found" in result.output
    assert "osprey build" in result.output


# ---------------------------------------------------------------------------
# Both verbs are reachable, and resolve to the new commands rather than the old
# ---------------------------------------------------------------------------


def test_status_resolves_to_the_new_verb_not_the_legacy_subcommand():
    """The registration hazard, which fails silently if it is wrong.

    ``deploy_cmd`` holds two commands named ``status``: the top-level verb
    (``status_verb``) and the legacy ``osprey status`` subcommand. The lazy
    group's default lookup is ``getattr(mod, cmd_name)``, which returns the
    LEGACY one — a working, project-scoped command — so nothing raises and
    ``osprey status`` quietly means the old thing.

    Object identity is the only honest check: the two objects have identical
    reprs (``<Command status>``), so an equality or name assertion would pass on
    the wrong one.
    """
    from osprey.cli.main import cli

    assert cli.get_command(None, "status") is status_verb
    assert "status" in cli.list_commands(None)


def test_logs_has_no_legacy_namesake_at_all():
    """``logs`` is registered for the OPPOSITE reason to its three siblings.

    ``down``, ``restart`` and ``status`` each need resolving by their ``_verb``
    name because a legacy ``deploy`` subcommand of the same bare name would be
    returned instead. ``logs`` has no namesake — which is precisely why the
    generic ``getattr(mod, "logs")`` cannot resolve it either: the command
    object is ``logs_verb``, and with nothing named ``logs`` in the module the
    lookup raises and the command is unreachable.

    Pinned because it is the assumption the registration rests on. If a legacy
    ``osprey logs`` is ever added, this fails and the comment in
    :class:`~osprey.cli.main.LazyGroup` needs rewriting — while the branch
    itself keeps resolving correctly.
    """
    assert not hasattr(deploy_cmd, "logs")


def test_logs_resolves_to_the_new_verb():
    from osprey.cli.main import cli

    assert cli.get_command(None, "logs") is logs_verb
    assert "logs" in cli.list_commands(None)


@pytest.mark.parametrize("name", ["status", "logs"])
def test_the_top_level_verbs_take_no_project_flag(name):
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
# The help text promises exactly what the code does
# ---------------------------------------------------------------------------


def test_status_help_states_the_label_limit():
    """An unqualified "matched by label" would hide the containers down cannot find."""
    help_text = status_verb.__doc__ or ""

    assert "created before this labelling existed" in help_text
    assert "project name" in help_text


def test_status_help_does_not_promise_a_reachability_check():
    """It reports declared endpoints, not answered ones."""
    help_text = status_verb.__doc__ or ""

    assert "declared to answer" in help_text


def test_logs_help_states_that_both_stacks_are_covered():
    help_text = logs_verb.__doc__ or ""

    assert "web-terminal stack" in help_text

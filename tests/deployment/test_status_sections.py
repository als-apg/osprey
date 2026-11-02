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

import base64
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

import osprey.cli.deploy_cmd as deploy_cmd
from osprey.cli import styles
from osprey.cli.deploy_cmd import logs_verb, status_verb
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.styles import osprey_theme
from osprey.deployment import status_display
from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity
from osprey.deployment.deployment_record import (
    DEPLOYMENT_RECORD_RELPATH,
    NOTHING_DEPLOYED,
    DeploymentRecord,
    write_deployment_record,
)
from tests.deployment.test_up_as_built import _RENDERED_CONFIG, render_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

PROJECT = "als-exemplar"


@pytest.fixture
def lifecycle_repo(tmp_path: Path) -> Path:
    """The gold-standard three-zone deployment repo, with no ``.env`` in it.

    It holds the record a start of :data:`PROJECT` on docker wrote, which is
    what both verbs report on.
    """
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)
    write_deployment_record(repo, DeploymentRecord("docker", PROJECT))
    return repo


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
        lambda config, all_containers=False, runtime=None: [
            runtime,
            "ps",
            "-a",
            "--format",
            "json",
        ],
    )

    def _fake_run(
        cmd,
        capture_output=True,  # noqa: ARG001 - subprocess.run's keywords
        text=True,  # noqa: ARG001 - subprocess.run's keywords
        timeout=None,  # noqa: ARG001 - subprocess.run's keywords
        **kwargs,
    ):
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


@pytest.mark.usefixtures("runtime")
def test_a_clean_build_is_reported_as_in_sync(lifecycle_repo):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "in sync with profile.yml" in text


@pytest.mark.usefixtures("runtime")
def test_drift_names_what_moved_and_that_the_start_verbs_refuse(lifecycle_repo):
    """The same verdict ``up`` refuses on, so a refusal is never a surprise here."""
    render_build(lifecycle_repo)
    profile = lifecycle_repo / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8") + "\nmodel: claude-opus-5\n", encoding="utf-8"
    )

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
    # The project in the header is the one the last start recorded.
    assert PROJECT in text.split("Containers", 1)[1]


@pytest.mark.usefixtures("runtime")
def test_the_version_line_is_printed_when_there_is_no_skew(lifecycle_repo):
    """A build rendered by the installed version still reports which one that was.

    ``version_skew`` is populated only when the two versions DIFFER, so a report
    without one cannot distinguish "they matched" from "the build recorded no
    version" — and the line has to.
    """
    from osprey.cli.templates.manifest import get_framework_release_version

    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert f"rendered by osprey {get_framework_release_version()} (installed)" in text


@pytest.mark.usefixtures("runtime")
def test_version_skew_is_reported_as_skew(lifecycle_repo):
    render_build(lifecycle_repo, version="1900.1.0")

    text = report(lifecycle_repo)

    assert "rendered by osprey 1900.1.0" in text
    assert "Re-run `osprey build`" in text


@pytest.mark.usefixtures("runtime")
def test_a_build_that_records_no_version_says_it_recorded_none(lifecycle_repo):
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


def test_a_container_with_only_the_compose_project_label_is_this_deployments(
    lifecycle_repo, runtime
):
    render_build(lifecycle_repo)
    row = container("als-exemplar-event-dispatcher", project=None)
    row["Labels"] = {"com.docker.compose.project": PROJECT}
    runtime["containers"] = [row]

    text = report(lifecycle_repo)

    assert "als-exemplar-event-dispatcher" in text
    assert "Nothing is running for this deployment" not in text
    assert f"carry no {REPO_ID_LABEL} label" in text


def test_an_unlabelled_container_is_claimed_by_a_deployed_service_name(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    runtime["containers"] = [container("als-exemplar-event-dispatcher", project=None)]

    text = report(lifecycle_repo)

    assert "als-exemplar-event-dispatcher" in text
    assert "matched by container name only" in text
    assert "Nothing is running for this deployment" not in text


def test_a_container_of_another_compose_project_appears_nowhere(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    row = container("stack-event-dispatcher", project=None)
    row["Labels"] = {"com.docker.compose.project": "stack"}
    runtime["containers"] = [row]

    text = report(lifecycle_repo)

    assert "stack-event-dispatcher" not in text


def test_a_repo_with_no_build_claims_nothing_by_name(lifecycle_repo, runtime):
    runtime["containers"] = [container("event-dispatcher", project=None)]

    text = report(lifecycle_repo)

    assert "event-dispatcher" not in text
    assert "Nothing is running for this deployment" in text


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


@pytest.mark.usefixtures("runtime")
def test_endpoints_come_from_the_rendered_compose_files(lifecycle_repo):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "Service endpoints" in text
    assert "127.0.0.1:18020" in text


@pytest.mark.usefixtures("runtime")
def test_the_endpoint_list_does_not_claim_anything_is_answering(lifecycle_repo):
    """It is read out of build/, not probed. Saying otherwise would be the run's defect."""
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "nothing was contacted to check" in text


# ---------------------------------------------------------------------------
# Agent: provider, credential, artifacts
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("runtime")
def test_a_credential_that_lives_only_in_the_repo_env_counts_as_found(lifecycle_repo, monkeypatch):
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


@pytest.mark.usefixtures("runtime")
def test_a_credential_that_is_nowhere_is_reported_missing(lifecycle_repo, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "not found in this shell" in text
    # Both members of the env chain, because both are read.
    assert ".env.shared" in text
    assert ".env" in text


@pytest.mark.usefixtures("runtime")
def test_a_credential_in_the_shared_file_is_found_and_flagged(lifecycle_repo, monkeypatch):
    """The chain is `.env.shared` then `.env`, and the spec beside this row
    expands `${VAR}` through the whole chain. Reporting only `.env` calls a
    resolvable credential missing. It is still the wrong file for a secret —
    `.env.shared` is committed — so the row says so instead of blessing it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo)
    (lifecycle_repo / ".env.shared").write_text("ANTHROPIC_API_KEY=sk-shared\n", encoding="utf-8")

    text = report(lifecycle_repo)

    assert "set in .env.shared" in text
    assert "committed file" in text
    assert "sk-shared" not in text


@pytest.mark.usefixtures("runtime")
def test_the_local_env_wins_over_the_shared_one(lifecycle_repo, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo)
    (lifecycle_repo / ".env.shared").write_text("ANTHROPIC_API_KEY=sk-shared\n", encoding="utf-8")
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n", encoding="utf-8")

    text = report(lifecycle_repo)

    assert "set in .env" in text
    assert "committed file" not in text


@pytest.mark.usefixtures("runtime")
def test_an_unset_provider_names_the_key_the_operator_sets(lifecycle_repo, monkeypatch):
    """`claude_code.provider` is a rendered key the build derives; writing it
    by hand is refused. The remedy names the profile field that sets it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    render_build(lifecycle_repo, config="project_name: als-exemplar\n")

    text = report(lifecycle_repo)

    assert "not configured" in text
    assert "in profile.yml" in text
    assert "claude_code.provider" not in text


#: The record's address the launch is handed.
_RECORD_URL = "http://127.0.0.1:10061"

#: The record login the exporter authenticates with, as ``<login>:<token>``.
_WORKING_LOGIN = "observability-user:not-a-real-password"


@pytest.mark.usefixtures("runtime")
def test_a_missing_record_login_is_not_a_provider_failure(lifecycle_repo, monkeypatch):
    """Without a record login the launch exports nothing; the provider still reads."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OSPREY_AGENT_RECORD_LOGIN", raising=False)
    render_build(lifecycle_repo, config=_RENDERED_CONFIG)

    text = report(lifecycle_repo)

    assert "could not be read from the build" not in text
    assert "credentials are missing or unresolved" not in text


def _telemetry_report(repo: Path, monkeypatch) -> str:
    """``osprey status`` for a repo whose launch exports to the agent record."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
    monkeypatch.delenv("OSPREY_AGENT_RECORD_LOGIN", raising=False)
    monkeypatch.delenv("OSPREY_AGENT_RECORD_URL", raising=False)
    monkeypatch.delenv("OSPREY_AGENT_RECORD_OFF", raising=False)
    env_file = repo / ".env"
    existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    env_file.write_text(
        existing
        + f"OSPREY_AGENT_RECORD_URL={_RECORD_URL}\n"
        + f"OSPREY_AGENT_RECORD_LOGIN={_WORKING_LOGIN}\n"
    )
    render_build(repo, config=_RENDERED_CONFIG)
    return report(repo)


@pytest.mark.usefixtures("runtime")
def test_the_observability_authorization_header_is_never_printed(lifecycle_repo, monkeypatch):
    """The env block is printed key AND value, and one of those values is a credential.

    ``OTEL_EXPORTER_OTLP_HEADERS`` carries HTTP Basic auth for the agent
    record — base64 of ``login:token``, decodable by anyone who reads the
    terminal. Neither the credentials nor the encoding of them may appear.
    """
    text = _telemetry_report(lifecycle_repo, monkeypatch)

    token = base64.b64encode(_WORKING_LOGIN.encode()).decode()
    assert token not in text
    assert "not-a-real-password" not in text
    assert "Basic " not in text


@pytest.mark.usefixtures("runtime")
def test_the_headers_variable_is_still_reported_as_configured(lifecycle_repo, monkeypatch):
    """Masking the value must not delete the row.

    Whether the exporter is authenticated at all is a fact an operator came to
    this section for; only the credential itself is withheld.
    """
    text = _telemetry_report(lifecycle_repo, monkeypatch)

    assert "OTEL_EXPORTER_OTLP_HEADERS" in text
    assert "value not shown" in text


@pytest.mark.usefixtures("runtime")
def test_the_rest_of_the_telemetry_block_still_shows_its_values(lifecycle_repo, monkeypatch):
    """Only the credential-bearing variable is masked, not the block around it.

    The other telemetry variables are ordinary configuration, and a mask broad
    enough to cover them would leave the section unable to answer where the
    agent ships its events — which is what the section is for.
    """
    text = _telemetry_report(lifecycle_repo, monkeypatch)

    endpoint_row = text.split("OTEL_EXPORTER_OTLP_ENDPOINT")[1].splitlines()[0]
    assert _RECORD_URL in endpoint_row
    assert "value not shown" not in endpoint_row


@pytest.mark.parametrize(
    ("name", "masked"),
    [
        ("OTEL_EXPORTER_OTLP_HEADERS", True),
        ("SOME_SERVICE_TOKEN", True),
        ("WEBHOOK_SIGNING_SECRET", True),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", False),
        ("ANTHROPIC_MODEL", False),
    ],
)
def test_the_mask_is_a_rule_and_not_one_remembered_variable(name, masked):
    """A variable named after a credential is withheld even if nobody listed it.

    The leak this closes was one variable, but a mask spelled as one literal
    comparison would let the next secret added to the env block out unnoticed
    while looking like a finished fix.
    """
    shown = status_display._display_env_value(name, "the-value")

    assert (shown != "the-value") is masked


@pytest.mark.usefixtures("runtime")
def test_the_artifact_check_mirrors_the_arguments_the_build_renders_with(
    lifecycle_repo,
    monkeypatch,
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


@pytest.mark.usefixtures("runtime")
def test_a_config_without_project_root_falls_back_to_the_repo_root(lifecycle_repo, monkeypatch):
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


@pytest.mark.usefixtures("runtime")
def test_a_container_targeted_build_is_not_told_to_probe_a_host_venv(lifecycle_repo, monkeypatch):
    """``--runtime-root`` records a path inside a container, which has no venv here."""
    render_build(
        lifecycle_repo,
        config=_RENDERED_CONFIG + "project_root: /app/als-exemplar\n",
    )
    calls: list[dict] = []

    class _Manager:
        def regenerate_claude_code(
            self,
            project_dir,  # noqa: ARG002 - the template manager's signature, the rest in **kwargs
            **kwargs,
        ):
            calls.append(kwargs)
            return {"changed": [], "unchanged": []}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    report(lifecycle_repo)

    assert calls[0]["project_root_override"] == "/app/als-exemplar"
    assert calls[0]["runtime_venv_dir"] is None


@pytest.mark.usefixtures("runtime")
def test_out_of_sync_artifacts_name_the_rebuild(lifecycle_repo, monkeypatch):
    render_build(lifecycle_repo)

    class _Manager:
        def regenerate_claude_code(
            self,
            project_dir,  # noqa: ARG002 - the template manager's signature, the rest in **kwargs
            **kwargs,
        ):
            return {"changed": [".claude/settings.json"], "unchanged": []}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    text = report(lifecycle_repo)

    assert "no longer match" in text
    assert "osprey build" in text
    assert ".claude/settings.json" in text


@pytest.mark.usefixtures("runtime")
def test_an_artifact_check_that_cannot_run_is_not_a_passing_one(lifecycle_repo, monkeypatch):
    """A template that will not render is a fact about the build, not a clean bill."""
    render_build(lifecycle_repo)

    class _Manager:
        def regenerate_claude_code(
            self,
            project_dir,  # noqa: ARG002 - the template manager's signature, the rest in **kwargs
            **kwargs,
        ):
            raise RuntimeError("template exploded")

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", lambda: _Manager())

    text = report(lifecycle_repo)

    assert "could not be checked" in text
    # Scoped to the artifact verdict: "in sync" alone also matches the BUILD
    # section's own clean verdict two sections above, and would pass here
    # whatever the artifact check reported.
    assert "in sync (" not in text


@pytest.mark.usefixtures("runtime")
def test_the_per_agent_model_table_is_opt_in(lifecycle_repo):
    """A dozen model rows would bury the two lines an operator came for."""
    render_build(lifecycle_repo)

    default = report(lifecycle_repo)
    with_agents = report(lifecycle_repo, show_agents=True)

    assert "agent models" not in default
    assert "agent models" in with_agents
    assert "Claude Code aliases" in default
    assert "model tiers" not in default


@pytest.mark.usefixtures("runtime")
def test_each_alias_names_its_model_and_where_it_came_from(lifecycle_repo):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "claude-sonnet-5 (derived)" in text
    assert "claude-opus-5-5 (derived)" in text


def _agent_file(build: Path, name: str, model: str | None) -> None:
    """Write ``build/.claude/agents/<name>.md``, with a ``model:`` line when *model* is given."""
    agents = build / ".claude" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    model_line = f"model: {model}\n" if model is not None else ""
    (agents / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Test agent.\n{model_line}---\n", encoding="utf-8"
    )


@pytest.mark.usefixtures("runtime")
def test_the_agent_table_lists_every_agent_the_build_ships(lifecycle_repo):
    """The table is the agent files the build ships, each with the model its file names."""
    build = render_build(lifecycle_repo)
    _agent_file(build, "channel-finder", "claude-sonnet-5")
    _agent_file(build, "logbook-search", "claude-sonnet-5")

    text = report(lifecycle_repo, show_agents=True)

    assert "channel-finder" in text
    assert "logbook-search" in text
    assert "claude-sonnet-5  (main model)" in text
    assert "pyat-specialist" not in text


@pytest.mark.usefixtures("runtime")
def test_a_pinned_agent_names_its_pin(lifecycle_repo):
    build = render_build(
        lifecycle_repo,
        config=_RENDERED_CONFIG + "  agent_models:\n    logbook-search: claude-haiku-4-5\n",
    )
    _agent_file(build, "logbook-search", "claude-haiku-4-5")

    text = report(lifecycle_repo, show_agents=True)

    assert "claude-haiku-4-5  (claude_code.agent_models)" in text


@pytest.mark.usefixtures("runtime")
def test_an_agent_file_of_the_deployment_names_its_own_model(lifecycle_repo):
    build = render_build(lifecycle_repo)
    _agent_file(build, "site-helper", "claude-opus-5-5")
    _agent_file(build, "site-notes", None)

    text = report(lifecycle_repo, show_agents=True)

    assert "claude-opus-5-5  (agents/site-helper.md)" in text
    assert "no model: line (Claude Code chooses)" in text


@pytest.mark.usefixtures("runtime")
def test_a_build_with_no_agent_files_says_so(lifecycle_repo):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo, show_agents=True)

    assert "none in build/.claude/agents" in text


# ---------------------------------------------------------------------------
# Status changes nothing
# ---------------------------------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.usefixtures("runtime")
def test_status_writes_nothing_into_the_repo(lifecycle_repo):
    """Including the artifact check, which renders into a temporary directory."""
    render_build(lifecycle_repo)
    (lifecycle_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    before = _tree(lifecycle_repo)

    report(lifecycle_repo)

    assert _tree(lifecycle_repo) == before


@pytest.mark.usefixtures("runtime")
def test_status_never_announces_creating_the_files_it_did_not_create(lifecycle_repo, capsys):
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


def _corrupt_the_record(repo: Path) -> None:
    (repo / DEPLOYMENT_RECORD_RELPATH).write_text("{not json", encoding="utf-8")


def test_status_owns_the_containers_of_the_recorded_project(lifecycle_repo, runtime):
    """A rebuild renamed the project; the running stack is still the recorded one's."""
    render_build(lifecycle_repo)
    write_deployment_record(lifecycle_repo, DeploymentRecord("docker", "earlier-name"))
    runtime["containers"] = [container("dispatch", project="earlier-name")]

    text = report(lifecycle_repo)

    containers = text.split("Containers", 1)[1]
    assert "earlier-name" in containers
    assert "dispatch" in containers
    assert "Other OSPREY containers" not in text


def test_status_asks_the_recorded_runtime(lifecycle_repo, monkeypatch):
    """ps, volume ls and inspect all go to the runtime the last start drove."""
    from osprey.cli.web_cmd import PREFLIGHT_REFUSED_MARKER
    from osprey.deployment.compose_generator import state_identity_dir
    from osprey.deployment.web_terminals.naming import web_container_name

    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
    render_build(
        lifecycle_repo,
        config=_RENDERED_CONFIG + "modules:\n  web_terminals:\n    enabled: true\n"
        "    users:\n      - alice\n",
    )
    write_deployment_record(lifecycle_repo, DeploymentRecord("podman", PROJECT))
    marker = state_identity_dir(lifecycle_repo, "alice") / PREFLIGHT_REFUSED_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("2099-01-01T00:00:00+00:00\n- refused\n", encoding="utf-8")
    rows = [container(web_container_name(PROJECT, "alice"), repo_id=repo_identity(lifecycle_repo))]
    asked: list[list[str]] = []

    def _run(cmd, **kwargs):
        argv = list(cmd)
        asked.append(argv)
        stdout = json.dumps(rows) if argv[1] == "ps" else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(status_display.subprocess, "run", _run)

    report(lifecycle_repo)

    assert {argv[1] for argv in asked} == {"ps", "volume", "inspect"}
    assert {argv[0] for argv in asked} == {"podman"}


def test_status_with_no_record_says_nothing_is_deployed_and_asks_nothing(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    (lifecycle_repo / DEPLOYMENT_RECORD_RELPATH).unlink()

    text = report(lifecycle_repo)

    assert NOTHING_DEPLOYED in text
    assert runtime["cmds"] == []


def test_status_names_an_unreadable_record_and_asks_nothing(lifecycle_repo, runtime):
    render_build(lifecycle_repo)
    _corrupt_the_record(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "unreadable" in text
    assert str(DEPLOYMENT_RECORD_RELPATH) in text
    assert runtime["cmds"] == []


# ---------------------------------------------------------------------------
# logs: the invocation it hands to the runtime
# ---------------------------------------------------------------------------


def logs_argv(repo: Path, **kwargs) -> list[str]:
    cmd, _env = status_display.logs_command(repo, **kwargs)
    return cmd


@pytest.mark.usefixtures("runtime")
def test_logs_is_built_through_the_pinned_compose_contract(lifecycle_repo):
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


@pytest.mark.usefixtures("runtime")
def test_follow_is_passed_through(lifecycle_repo):
    render_build(lifecycle_repo)

    assert "--follow" in logs_argv(lifecycle_repo, follow=True)
    assert "--follow" not in logs_argv(lifecycle_repo)


@pytest.mark.usefixtures("runtime")
def test_tail_is_passed_through_and_otherwise_left_to_the_runtime(lifecycle_repo):
    """No invented default: an unset ``--tail`` means compose's own behaviour."""
    render_build(lifecycle_repo)

    argv = logs_argv(lifecycle_repo, tail=50)

    assert argv[argv.index("--tail") + 1] == "50"
    assert "--tail" not in logs_argv(lifecycle_repo)


@pytest.mark.usefixtures("runtime")
def test_a_named_service_is_the_last_argument(lifecycle_repo):
    render_build(lifecycle_repo)

    argv = logs_argv(lifecycle_repo, service="event-dispatcher", follow=True)

    assert argv[-1] == "event-dispatcher"


@pytest.mark.usefixtures("runtime")
def test_the_web_stack_is_carried_when_the_deployment_has_one(lifecycle_repo):
    """Two compose invocations start the deployment; one reads its logs.

    They are a single compose project, so a ``logs`` that carried only the
    services file would silently omit every web-terminal container.
    """
    from osprey.deployment.web_terminals.artifacts import web_compose_file

    render_build(lifecycle_repo)
    assert str(web_compose_file(lifecycle_repo)) not in logs_argv(lifecycle_repo)

    web_compose_file(lifecycle_repo).write_text("services: {}\n", encoding="utf-8")

    assert str(web_compose_file(lifecycle_repo)) in logs_argv(lifecycle_repo)


@pytest.mark.usefixtures("runtime")
def test_logs_does_not_warn_about_services_starting(lifecycle_repo, caplog):
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


@pytest.mark.usefixtures("runtime")
def test_logs_pins_the_compose_project_name(lifecycle_repo):
    """Left unset, compose derives a project from the directory and finds nothing —
    which prints an empty log and looks like a deployment that logs nothing."""
    render_build(lifecycle_repo)

    _cmd, env = status_display.logs_command(lifecycle_repo)

    assert env["COMPOSE_PROJECT_NAME"] == PROJECT


@pytest.mark.usefixtures("runtime")
def test_logs_follows_the_recorded_project_with_the_recorded_runtime(lifecycle_repo, monkeypatch):
    from osprey.deployment.compose_generator import resolve_project_name

    monkeypatch.setenv("CONTAINER_RUNTIME", "docker")
    render_build(lifecycle_repo)
    recorded = DeploymentRecord("podman", "earlier-name")
    assert resolve_project_name({"project_name": PROJECT}) != recorded.compose_project
    write_deployment_record(lifecycle_repo, recorded)

    cmd, env = status_display.logs_command(lifecycle_repo)

    assert cmd[0] == recorded.runtime
    assert env["COMPOSE_PROJECT_NAME"] == recorded.compose_project


def test_logs_with_no_record_says_nothing_is_deployed(lifecycle_repo):
    render_build(lifecycle_repo)
    (lifecycle_repo / DEPLOYMENT_RECORD_RELPATH).unlink()

    result = CliRunner().invoke(logs_verb, ["--repo", str(lifecycle_repo)])

    assert result.exit_code != 0
    assert NOTHING_DEPLOYED in result.output


def test_logs_names_an_unreadable_record(lifecycle_repo):
    from osprey.deployment.errors import DeploymentRecordError

    render_build(lifecycle_repo)
    _corrupt_the_record(lifecycle_repo)

    with pytest.raises(DeploymentRecordError, match=str(DEPLOYMENT_RECORD_RELPATH)):
        status_display.logs_command(lifecycle_repo)


def test_logs_without_a_build_refuses_and_names_the_remedy(lifecycle_repo):
    with pytest.raises(status_display.NoComposeFilesError) as excinfo:
        status_display.logs_command(lifecycle_repo)

    # The remedy is a field, not prose spliced into the message — that is what
    # lets one CLI handler render every precondition refusal the same way.
    assert "osprey build" in excinfo.value.remedy


@pytest.mark.usefixtures("runtime")
def test_logs_renders_nothing(lifecycle_repo, monkeypatch):
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


# ---------------------------------------------------------------------------
# Writes: per login its tier, its record's target, and each armed machine's position
# ---------------------------------------------------------------------------


def _write_record(repo: Path, identity: str, *, target: str, switch: dict | None = None) -> None:
    """One identity's control-context record, where the deploy binds the tree."""
    from osprey.deployment.compose_generator import control_target_identity_dir
    from osprey_connectors.control_context import RECORD_FILENAME

    directory = control_target_identity_dir({}, repo, identity)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "owner": None,
        "target": target,
        "generation": 1,
        "posture": {},
        "last_switch": None,
        "switch": switch or {},
        "switch_epoch": None,
    }
    (directory / RECORD_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


#: The map every multi-user case stages unless it says otherwise.
_MAP = {"alice": "allowed", "bob": "never"}


def _stage_map(repo: Path, login_tiers, *, service: str = "bluesky") -> Path:
    """Stage a shared container's ``config.yml`` carrying *login_tiers*, as a build does."""
    path = repo / "build" / "services" / service / "config.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"control_system": {"writes": "allowed", "login_tiers": login_tiers}}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _render(repo: Path, name: str, writes: str) -> dict:
    """A rendered tier project whose ``control_system.writes`` is *writes*.

    Status must not read it: the staged map is the one answer (CC-3). The
    cases below keep renders that disagree with the map to prove it.
    """
    project = repo / "build" / f"demo-{name}"
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yml").write_text(f"control_system:\n  writes: {writes}\n", encoding="utf-8")
    return {"kind": "tier", "project": f"demo-{name}", "project_path": f"build/demo-{name}"}


def _multi_user_config(repo: Path, *, login_tiers=_MAP, **web_terminals) -> dict:
    """A two-person roster; *login_tiers* is staged unless it is ``None``."""
    block = {
        "enabled": True,
        "users": [{"name": "alice", "tier": "operator"}, {"name": "bob", "tier": "viewer"}],
        "renders": {
            # Inverted against the map: a reader of the renders gets every tier wrong.
            "operator": _render(repo, "operator", "never"),
            "viewer": _render(repo, "viewer", "allowed"),
        },
    }
    block.update(web_terminals)
    if login_tiers is not None:
        _stage_map(repo, login_tiers)
    return {"modules": {"web_terminals": block}}


def _rows_by_login(config: dict, repo: Path) -> dict[str, list[str]]:
    return {login: row for login, *row in status_display.writes_rows(config, repo)}


def _writes_text(config: dict, repo: Path) -> str:
    status_display._print_writes_section(config, repo)
    return _CAPTURE["console"].export_text()


def test_single_user_lists_its_record_at_the_deployments_own_tier(tmp_path):
    _write_record(tmp_path, "thorsten", target="standin", switch={"standin": "on"})
    config = {"control_system": {"writes": "allowed"}}

    rows = status_display.writes_rows(config, tmp_path)

    assert rows == [("thorsten", "allowed", "standin", "standin:asked")]
    text = _writes_text(config, tmp_path)
    assert "Writes:" in text
    assert "not in the roster" not in text


def test_single_user_with_no_record_says_every_machine_is_off(tmp_path):
    text = _writes_text({"control_system": {"writes": "allowed"}}, tmp_path)

    assert "Writes:" in text
    assert "every machine's writes are off" in text


def test_multi_user_lists_each_login_with_its_tier_target_and_machines(tmp_path):
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "alice", target="live", switch={"live": "asked", "standin": "allowed"})

    rows = status_display.writes_rows(config, tmp_path)

    assert rows == [
        ("alice", "allowed", "live", "live:asked, standin:allowed"),
        # No record is every switch off; nothing is created to say so.
        ("bob", "never", "—", "off"),
    ]
    from osprey.deployment.compose_generator import control_target_identity_dir

    assert not control_target_identity_dir({}, tmp_path, "bob").exists()


def test_a_position_under_a_never_tier_is_marked_inert(tmp_path):
    """O-3: a demoted login's positions stay recorded and inert, in neither case capped."""
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "bob", target="standin", switch={"live": "allowed", "standin": "asked"})

    rows = _rows_by_login(config, tmp_path)

    expected = "live:allowed, standin:asked — tier never, inert"
    assert rows["bob"] == ["never", "standin", expected]
    assert expected in _writes_text(config, tmp_path)


def test_a_staged_asked_entry_is_the_tier_cell(tmp_path):
    config = _multi_user_config(tmp_path, login_tiers={"alice": "asked", "bob": "never"})

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["asked", "—", "off"]


def test_a_legacy_on_record_reads_asked(tmp_path):
    """The stored ``on`` from before the three positions is the word ``asked`` everywhere."""
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "alice", target="standin", switch={"standin": "on"})

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["allowed", "standin", "standin:asked"]


def test_an_allowed_under_an_asked_ceiling_is_marked_capped(tmp_path):
    """Only the entry above the ceiling is marked; ``asked`` is in force as stored."""
    config = _multi_user_config(tmp_path, login_tiers={"alice": "asked", "bob": "never"})
    _write_record(tmp_path, "alice", target="live", switch={"live": "allowed", "standin": "asked"})

    rows = _rows_by_login(config, tmp_path)

    expected = "live:allowed (capped: asked), standin:asked"
    assert rows["alice"] == ["asked", "live", expected]
    assert expected in _writes_text(config, tmp_path)


@pytest.mark.parametrize("tier", ["allowed", "unknown"])
def test_an_allowed_is_not_capped_without_an_asked_ceiling(tmp_path, tier):
    """``unknown`` states no ceiling, so nothing here claims one."""
    login_tiers = {"bob": "never"} if tier == "unknown" else {"alice": tier, "bob": "never"}
    config = _multi_user_config(tmp_path, login_tiers=login_tiers)
    _write_record(tmp_path, "alice", target="live", switch={"live": "allowed"})

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == [tier, "live", "live:allowed"]


def test_an_all_off_record_reads_off(tmp_path):
    """A switch map with nothing armed states the record's target and no machine."""
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "alice", target="live", switch={"standin": "maybe"})

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["allowed", "live", "off"]


def test_the_writes_table_names_the_three_words_not_the_retired_one(tmp_path):
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "alice", target="standin", switch={"standin": "asked"})

    text = _writes_text(config, tmp_path)

    assert "Writes (off, asked, allowed)" in text
    assert "Writes on" not in text


def test_an_identity_not_in_the_roster_is_named_as_such(tmp_path):
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "mallory", target="live", switch={"live": "on"})

    rows = status_display.writes_rows(config, tmp_path)

    assert rows[-1] == ("mallory", "not in the roster — cannot write", "live", "live:asked")
    assert [row[0] for row in rows] == ["alice", "bob", "mallory"]


def test_an_unreadable_record_reads_refused_never_off(tmp_path):
    """Carried R3 CF-6: fail closed and say so."""
    from osprey.deployment.compose_generator import control_target_identity_dir
    from osprey_connectors.control_context import RECORD_FILENAME

    config = _multi_user_config(tmp_path)
    directory = control_target_identity_dir({}, tmp_path, "alice")
    directory.mkdir(parents=True)
    (directory / RECORD_FILENAME).write_text("{not json", encoding="utf-8")

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["allowed", "?", "record unreadable — writes refused"]
    text = _writes_text(config, tmp_path)
    assert "record unreadable — writes refused" in text


def test_an_unreadable_record_under_an_asked_tier_reads_refused(tmp_path):
    from osprey.deployment.compose_generator import control_target_identity_dir
    from osprey_connectors.control_context import RECORD_FILENAME

    config = _multi_user_config(tmp_path, login_tiers={"alice": "asked", "bob": "never"})
    directory = control_target_identity_dir({}, tmp_path, "alice")
    directory.mkdir(parents=True)
    (directory / RECORD_FILENAME).write_text("{not json", encoding="utf-8")

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["asked", "?", "record unreadable — writes refused"]


def test_an_sso_login_without_a_role_has_its_tier_decided_at_sign_in(tmp_path):
    """Carried R2 CF-4: the claim picks the role, so no file here names the tier."""
    config = _multi_user_config(
        tmp_path,
        users=[{"name": "alice", "role": "operator"}, {"name": "carol"}],
        default_tier="viewer",
        auth={"method": "oidc"},
        authorization={
            "roles": {"operator": {"tier": "operator"}, "expert": {"tier": "viewer"}},
            "claims": {"claim": "groups", "map": {"ops": "operator", "experts": "expert"}},
        },
    )
    _write_record(tmp_path, "carol", target="standin", switch={"standin": "on"})

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["allowed", "—", "off"]
    assert rows["carol"] == ["by role at sign-in", "standin", "standin:asked"]


def test_no_staged_map_answers_unknown_for_every_card_not_a_guess(tmp_path):
    """A build that staged no shared container has no map to show; renders are not read."""
    config = _multi_user_config(tmp_path, login_tiers=None)

    rows = _rows_by_login(config, tmp_path)

    assert rows == {"alice": ["unknown", "—", "off"], "bob": ["unknown", "—", "off"]}


def test_the_map_is_read_from_the_dispatch_worker_copy_too(tmp_path):
    config = _multi_user_config(tmp_path, login_tiers=None)
    _stage_map(tmp_path, _MAP, service="dispatch_worker")

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"][0] == "allowed"
    assert rows["bob"][0] == "never"


def test_the_rendered_map_wins_over_the_live_roster(tmp_path):
    """CC-3: a roster edit shows after the next build, and status flags no drift."""
    config = _multi_user_config(tmp_path)
    # alice was moved to the viewer tier in the profile, not yet rebuilt.
    config["modules"]["web_terminals"]["users"][0]["tier"] = "viewer"

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"] == ["allowed", "—", "off"]
    text = _writes_text(config, tmp_path)
    assert "drift" not in text.lower()


def test_a_card_the_map_does_not_name_answers_unknown(tmp_path):
    config = _multi_user_config(tmp_path, login_tiers={"alice": "allowed"})

    rows = _rows_by_login(config, tmp_path)

    assert rows["bob"] == ["unknown", "—", "off"]


def test_a_mapped_login_outside_the_roster_lists_only_with_a_record_at_its_mapped_tier(tmp_path):
    """The build account is mapped for its owned jobs; it is not "not in the roster"."""
    config = _multi_user_config(
        tmp_path, login_tiers={**_MAP, "builder": "allowed", "ghost": "never"}
    )
    _write_record(tmp_path, "builder", target="standin", switch={"standin": "on"})

    rows = status_display.writes_rows(config, tmp_path)

    assert [row[0] for row in rows] == ["alice", "bob", "builder"]
    assert rows[-1] == ("builder", "allowed", "standin", "standin:asked")


@pytest.mark.parametrize(
    "login_tiers",
    [["alice"], {"alice": "allowed", "bob": "maybe"}, {"alice": True}, "allowed"],
    ids=["list", "bad-value", "non-string-value", "scalar"],
)
def test_a_malformed_map_answers_unknown_for_every_card(tmp_path, login_tiers):
    """One render's output: unreadable as a whole, as the write verdict reads it."""
    config = _multi_user_config(tmp_path, login_tiers=login_tiers)

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"][0] == "unknown"
    assert rows["bob"][0] == "unknown"


def test_an_unparseable_staged_config_answers_unknown(tmp_path):
    config = _multi_user_config(tmp_path)
    (tmp_path / "build" / "services" / "bluesky" / "config.yml").write_text(
        "control_system: [unclosed\n", encoding="utf-8"
    )

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"][0] == "unknown"


def test_the_map_is_read_with_no_tree_bound_in_the_reading_process(tmp_path, monkeypatch):
    """Status's own environment never turns a readable map into unknown."""
    monkeypatch.setenv("OSPREY_CONTROL_CONTEXT_TREE", str(tmp_path / "elsewhere"))
    config = _multi_user_config(tmp_path)

    rows = _rows_by_login(config, tmp_path)

    assert rows["alice"][0] == "allowed"


def test_an_unrostered_identity_reads_cannot_write_in_the_printed_section(tmp_path):
    config = _multi_user_config(tmp_path)
    _write_record(tmp_path, "mallory", target="live", switch={"live": "on"})

    assert "not in the roster — cannot write" in _writes_text(config, tmp_path)


@pytest.mark.usefixtures("runtime")
def test_status_prints_the_writes_section_after_the_endpoints(lifecycle_repo):
    render_build(lifecycle_repo)

    text = report(lifecycle_repo)

    assert "Writes:" in text
    assert text.index("Writes:") < text.index("Agent:")

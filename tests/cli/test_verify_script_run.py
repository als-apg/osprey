"""The emitted health check, run.

``test_emitted_artifacts_clean`` holds ``scripts/verify.sh`` to its
hand-authored specification byte for byte, which pins what the script says. A
byte comparison cannot say what the script does when it runs: that it lists
the compose project the deployment record names, with the runtime the record
names, which containers it flags, and how it exits. This module runs the
rendered exemplar script under ``bash`` against stub runtimes on a private
``PATH``, so it needs no container runtime, no network and no daemon. Each
record is written through the product's own writer, so the format is never
restated here.

Each stub logs its arguments to ``$STUB_LOG``. ``docker`` and ``podman`` answer
the runtime probe (``compose version`` and a bare ``ps``) and print the file
named by ``$STUB_PS`` for any other call; ``curl`` exits ``$STUB_CURL_RC``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.deploy_scaffold_templates import (
    VERIFY_PATH,
    VERIFY_TEMPLATE,
    build_verify_context,
    render,
)
from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.deployment_record import (
    DEPLOYMENT_RECORD_RELPATH,
    DeploymentRecord,
    write_deployment_record,
)
from tests.fixtures.lifecycle_repo import build_exemplar_repo

_RUNTIME_STUB = """\
#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  "compose version"|ps) exit 0 ;;
esac
cat "$STUB_PS"
"""

_CURL_STUB = """\
#!/bin/sh
printf 'curl %s\\n' "$*" >> "$STUB_LOG"
exit "${STUB_CURL_RC:-0}"
"""

#: A runtime that is installed but does not answer. It shadows any real
#: ``docker`` or ``podman`` in ``/usr/bin``, which a CI runner may well have.
_DEAD_RUNTIME_STUB = """\
#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
exit 1
"""

_BASH = shutil.which("bash")

#: The four states a listing can hold, one row each, as docker prints them.
_FOUR_ROWS = [
    {"Name": "demo-graphdb-1", "State": "exited", "ExitCode": 137, "Health": ""},
    {"Name": "demo-mongo-1", "State": "running", "ExitCode": 0, "Health": "unhealthy"},
    {"Name": "demo-postgres-1", "State": "running", "ExitCode": 0, "Health": "healthy"},
    {"Name": "demo-qmd-1", "State": "running", "ExitCode": 0, "Health": ""},
]


@pytest.fixture(scope="module")
def script_text(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The health check rendered for the exemplar deployment."""
    repo = build_exemplar_repo(tmp_path_factory.mktemp("verify") / "als-exemplar", with_ci=True)
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    return render(VERIFY_TEMPLATE, build_verify_context(profile, "@OSPREY_VERSION@"))


def _install(tmp_path: Path, script_text: str) -> Path:
    """Write the script into an ``als-exemplar`` repo directory; return the script."""
    script = tmp_path / "als-exemplar" / VERIFY_PATH
    script.parent.mkdir(parents=True)
    script.write_text(script_text, encoding="utf-8")
    return script


def _record(script: Path, runtime: str = "docker", project: str | None = None) -> str:
    """Record, in the script's repo, that ``osprey up`` drove ``project`` with ``runtime``.

    ``project`` defaults to the name the repo's directory resolves to. Returns
    the recorded project.
    """
    repo = script.parents[1]
    project = project or resolve_project_name({"project_root": str(repo)})
    write_deployment_record(repo, DeploymentRecord(runtime, project))
    return project


@pytest.fixture
def script(tmp_path: Path, script_text: str) -> Path:
    """The rendered script, at its place in an ``als-exemplar`` repo."""
    return _install(tmp_path, script_text)


@pytest.fixture
def deployed(script: Path) -> Path:
    """The rendered script, in a repo whose record names a docker deployment."""
    _record(script)
    return script


def _stubs(tmp_path: Path, *, runtimes: bool = True) -> Path:
    """A directory holding ``python3``, ``curl``, and runtimes that answer unless told not to."""
    stubs = tmp_path / ("stubs" if runtimes else "stubs-no-runtime")
    stubs.mkdir()
    (stubs / "python3").symlink_to(sys.executable)
    runtime = _RUNTIME_STUB if runtimes else _DEAD_RUNTIME_STUB
    names = {"curl": _CURL_STUB, "docker": runtime, "podman": runtime}
    for name, body in names.items():
        path = stubs / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    return stubs


def _listing(tmp_path: Path, rows: Any, *, ndjson: bool = True) -> Path:
    """The file the runtime stubs print for the listing call."""
    path = tmp_path / "ps.json"
    if ndjson:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _run(
    script: Path,
    *args: str,
    stubs: Path,
    listing: Path,
    **env: str,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the script with only the stubs and the system tools on ``PATH``.

    Returns the finished process and the stub log, one call per line. The
    timeout only guards against a hang: a run spawns a dozen short processes,
    and on a loaded host each spawn can take a second.
    """
    assert _BASH, "bash must be on PATH"
    log = script.parents[1].parent / "stub.log"
    log.write_text("", encoding="utf-8")
    result = subprocess.run(
        [_BASH, str(script), *args],
        env={
            "PATH": f"{stubs}:/usr/bin:/bin",
            "STUB_LOG": str(log),
            "STUB_PS": str(listing),
            **env,
        },
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result, log.read_text(encoding="utf-8").splitlines()


def _runtime_calls(log: list[str]) -> list[str]:
    return [line for line in log if not line.startswith("curl ")]


def test_the_rendered_script_parses(script: Path) -> None:
    """``bash -n`` accepts the script."""
    assert _BASH
    assert (
        subprocess.run([_BASH, "-n", str(script)], capture_output=True, timeout=60).returncode == 0
    )


def test_every_container_is_listed_and_the_failing_ones_flagged(
    tmp_path: Path, deployed: Path
) -> None:
    """A stopped container and an unhealthy one are flagged; the rest pass."""
    result, _ = _run(
        deployed,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert result.returncode == 0
    assert "✗\033[0m demo-graphdb-1 — exited (137)" in result.stdout
    assert "✗\033[0m demo-mongo-1 — running but unhealthy" in result.stdout
    assert "✓\033[0m demo-postgres-1\n" in result.stdout
    assert "✓\033[0m demo-qmd-1\n" in result.stdout
    assert "2 flagged." in result.stdout


def test_a_starting_healthcheck_is_shown_not_flagged(tmp_path: Path, deployed: Path) -> None:
    """A healthcheck still starting is reported, and does not count."""
    row = {"Name": "demo-graphdb-1", "State": "running", "Health": "starting"}
    result, _ = _run(
        deployed, "containers", stubs=_stubs(tmp_path), listing=_listing(tmp_path, [row])
    )

    assert result.returncode == 0
    assert "demo-graphdb-1 — healthcheck still starting" in result.stdout
    assert "Nothing flagged." in result.stdout


def test_docker_lists_the_recorded_project_through_compose(tmp_path: Path, script: Path) -> None:
    """The project listed is the one the record names, not the repo directory's."""
    project = _record(script, "docker", resolve_project_name({"project_name": "Another Name"}))

    _, log = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert _runtime_calls(log) == [f"compose -p {project} ps -a --format json"]


def test_podman_lists_by_the_compose_project_label(tmp_path: Path, script: Path) -> None:
    """A podman record lists by label, and its health comes from the status text."""
    project = _record(script, "podman")
    rows = [{"Names": ["demo-graphdb-1"], "State": "running", "Status": "Up 3 minutes (unhealthy)"}]
    result, log = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, rows, ndjson=False),
    )

    assert _runtime_calls(log) == [
        f"ps -a --filter label=com.docker.compose.project={project} --format json"
    ]
    assert "demo-graphdb-1 — running but unhealthy" in result.stdout
    assert "1 flagged." in result.stdout


def test_the_recorded_runtime_is_used_even_when_another_is_pinned_in_the_environment(
    tmp_path: Path, script: Path
) -> None:
    """The environment's runtime and project pins are not read: the record decides."""
    _record(script, "podman", "demo")

    _, log = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
        CONTAINER_RUNTIME="docker",
        COMPOSE_PROJECT_NAME="other",
    )

    assert _runtime_calls(log) == [
        "ps -a --filter label=com.docker.compose.project=demo --format json"
    ]


def test_a_recorded_runtime_that_does_not_answer_is_flagged_as_unlistable(
    tmp_path: Path, script: Path
) -> None:
    """A record naming a runtime that does not answer is flagged, naming both facts."""
    project = _record(script, "docker")

    result, _ = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path, runtimes=False),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert result.returncode == 0
    assert f"could not list project {project} with docker" in result.stdout
    assert "1 flagged." in result.stdout


def test_a_project_with_no_containers_is_flagged(tmp_path: Path, script: Path) -> None:
    """An empty listing means nothing was verified, and that is flagged."""
    project = _record(script)

    result, _ = _run(script, "containers", stubs=_stubs(tmp_path), listing=_listing(tmp_path, []))

    assert result.returncode == 0
    assert f"project {project} has no containers" in result.stdout


# ── No record, or one that cannot be read ────────────────────────────────────


def test_no_record_says_nothing_has_been_deployed_and_probes_nothing(
    tmp_path: Path, script: Path
) -> None:
    """Before the first ``osprey up`` there is nothing to check, and nothing is asked."""
    result, log = _run(script, stubs=_stubs(tmp_path), listing=_listing(tmp_path, _FOUR_ROWS))

    assert result.returncode == 0
    assert "nothing has been deployed from this repo yet — run osprey up" in result.stdout
    assert "1 flagged." in result.stdout
    assert log == []


def test_no_record_exits_1_under_strict(tmp_path: Path, script: Path) -> None:
    """``--strict`` counts the missing deployment as flagged."""
    result, log = _run(
        script, "--strict", stubs=_stubs(tmp_path), listing=_listing(tmp_path, _FOUR_ROWS)
    )

    assert result.returncode == 1
    assert "nothing has been deployed from this repo yet" in result.stdout
    assert log == []


@pytest.mark.parametrize(
    "document",
    ["{not json", "[]", '{"runtime": "docker"}', '{"runtime": "nerdctl", "compose_project": "x"}'],
    ids=["corrupt", "not-an-object", "missing-key", "unknown-runtime"],
)
def test_an_unreadable_record_is_named_and_is_not_called_undeployed(
    tmp_path: Path, script: Path, document: str
) -> None:
    """A record that is there but cannot be read is its own finding, and nothing is probed."""
    path = script.parents[1] / DEPLOYMENT_RECORD_RELPATH
    path.parent.mkdir(parents=True)
    path.write_text(document, encoding="utf-8")

    result, log = _run(script, stubs=_stubs(tmp_path), listing=_listing(tmp_path, _FOUR_ROWS))

    assert result.returncode == 0
    assert (
        f"{DEPLOYMENT_RECORD_RELPATH.as_posix()} is unreadable — run osprey up to rewrite it"
        in result.stdout
    )
    assert "nothing has been deployed" not in result.stdout
    assert log == []


# ── --strict and the argument line ───────────────────────────────────────────


def test_strict_exits_1_when_a_container_is_flagged(tmp_path: Path, deployed: Path) -> None:
    """``--strict`` turns anything flagged into exit 1."""
    result, _ = _run(
        deployed,
        "--strict",
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert result.returncode == 1
    assert "2 flagged. Exit 1: --strict." in result.stdout


def test_strict_exits_0_when_nothing_is_flagged(tmp_path: Path, deployed: Path) -> None:
    """``--strict`` with nothing flagged still exits 0."""
    rows = [row for row in _FOUR_ROWS if row["Name"] in {"demo-postgres-1", "demo-qmd-1"}]
    result, _ = _run(
        deployed,
        "--strict",
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, rows),
    )

    assert result.returncode == 0
    assert "Nothing flagged." in result.stdout


def test_strict_exits_1_on_a_failed_probe_alone(tmp_path: Path, deployed: Path) -> None:
    """An endpoint that gets no answer counts under ``--strict`` too."""
    result, log = _run(
        deployed,
        "--strict",
        "dispatch",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
        STUB_CURL_RC="7",
    )

    assert result.returncode == 1
    assert "1 flagged. Exit 1: --strict." in result.stdout
    assert _runtime_calls(log) == []


def test_group_arguments_still_select_groups(tmp_path: Path, deployed: Path) -> None:
    """A group argument runs that group alone, wherever ``--strict`` sits."""
    stubs = _stubs(tmp_path)
    listing = _listing(tmp_path, _FOUR_ROWS)

    _, log = _run(deployed, "containers", stubs=stubs, listing=listing)
    assert _runtime_calls(log) and not [line for line in log if line.startswith("curl ")]

    _, log = _run(deployed, "web", "dispatch", stubs=stubs, listing=listing)
    assert log and _runtime_calls(log) == []

    before, before_log = _run(
        deployed, "dispatch", "--strict", stubs=stubs, listing=listing, STUB_CURL_RC="7"
    )
    after, after_log = _run(
        deployed, "--strict", "dispatch", stubs=stubs, listing=listing, STUB_CURL_RC="7"
    )
    assert (before.returncode, before.stdout, before_log) == (
        after.returncode,
        after.stdout,
        after_log,
    )


@pytest.mark.parametrize("argument", ["--strcit", "servics"])
def test_an_unknown_argument_exits_2_and_runs_nothing(
    tmp_path: Path, deployed: Path, argument: str
) -> None:
    """A mistyped option or group is refused before anything runs."""
    result, log = _run(
        deployed, argument, stubs=_stubs(tmp_path), listing=_listing(tmp_path, _FOUR_ROWS)
    )

    assert result.returncode == 2
    assert argument in result.stderr
    assert log == []

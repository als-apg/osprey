"""The emitted health check, run.

``test_emitted_artifacts_clean`` holds ``scripts/verify.sh`` to its
hand-authored specification byte for byte, which pins what the script says. A
byte comparison cannot say what the script does when it runs: which runtime it
asks, which compose project it names, which containers it flags, and how it
exits. This module runs the rendered exemplar script under ``bash`` against
stub runtimes on a private ``PATH``, so it needs no container runtime, no
network and no daemon.

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


def _install(tmp_path: Path, script_text: str, repo_name: str = "als-exemplar") -> Path:
    """Write the script into a repo directory named ``repo_name``; return the script."""
    script = tmp_path / repo_name / VERIFY_PATH
    script.parent.mkdir(parents=True)
    script.write_text(script_text, encoding="utf-8")
    return script


@pytest.fixture
def script(tmp_path: Path, script_text: str) -> Path:
    """The rendered script, at its place in an ``als-exemplar`` repo."""
    return _install(tmp_path, script_text)


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
    assert subprocess.run([_BASH, "-n", str(script)], capture_output=True).returncode == 0


def test_every_container_is_listed_and_the_failing_ones_flagged(
    tmp_path: Path, script: Path
) -> None:
    """A stopped container and an unhealthy one are flagged; the rest pass."""
    result, _ = _run(
        script,
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


def test_a_starting_healthcheck_is_shown_not_flagged(tmp_path: Path, script: Path) -> None:
    """A healthcheck still starting is reported, and does not count."""
    row = {"Name": "demo-graphdb-1", "State": "running", "Health": "starting"}
    result, _ = _run(
        script, "containers", stubs=_stubs(tmp_path), listing=_listing(tmp_path, [row])
    )

    assert result.returncode == 0
    assert "demo-graphdb-1 — healthcheck still starting" in result.stdout
    assert "Nothing flagged." in result.stdout


@pytest.mark.parametrize("repo_name", ["Demo Facility_", "als-exemplar", "---"])
def test_docker_is_asked_through_compose_for_this_repo_project(
    tmp_path: Path, script_text: str, repo_name: str
) -> None:
    """The project is the repo directory's name, normalized as ``osprey up`` does."""
    script = _install(tmp_path, script_text, repo_name)
    _, log = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    project = resolve_project_name({"project_name": repo_name})
    assert f"compose -p {project} ps -a --format json" in log


def test_a_pinned_compose_project_name_wins(tmp_path: Path, script: Path) -> None:
    """``COMPOSE_PROJECT_NAME`` names the project when the caller sets it."""
    _, log = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
        COMPOSE_PROJECT_NAME="pinned",
    )

    assert "compose -p pinned ps -a --format json" in log


def test_podman_lists_by_the_compose_project_label(tmp_path: Path, script: Path) -> None:
    """podman lists by label, and its health comes from the status text."""
    rows = [{"Names": ["demo-graphdb-1"], "State": "running", "Status": "Up 3 minutes (unhealthy)"}]
    result, log = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, rows, ndjson=False),
        CONTAINER_RUNTIME="podman",
    )

    project = resolve_project_name({"project_name": "als-exemplar"})
    assert _runtime_calls(log) == [
        f"ps -a --filter label=com.docker.compose.project={project} --format json"
    ]
    assert "demo-graphdb-1 — running but unhealthy" in result.stdout
    assert "1 flagged." in result.stdout


def test_the_built_config_names_the_runtime(tmp_path: Path, script: Path) -> None:
    """``container_runtime`` in the built config picks the runtime; the environment wins."""
    config = script.parents[1] / "build" / "config.yml"
    config.parent.mkdir()
    config.write_text("project_name: als-exemplar\ncontainer_runtime: podman\n", encoding="utf-8")
    stubs = _stubs(tmp_path)
    listing = _listing(tmp_path, _FOUR_ROWS)

    _, log = _run(script, "containers", stubs=stubs, listing=listing)
    assert _runtime_calls(log)[0].startswith("ps -a --filter label=")

    _, log = _run(script, "containers", stubs=stubs, listing=listing, CONTAINER_RUNTIME="docker")
    assert _runtime_calls(log)[0].startswith("compose -p ")


def test_no_answering_runtime_is_flagged(tmp_path: Path, script: Path) -> None:
    """No runtime to ask means nothing was verified, and that is flagged."""
    result, _ = _run(
        script,
        "containers",
        stubs=_stubs(tmp_path, runtimes=False),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert result.returncode == 0
    assert "no container runtime answers" in result.stdout
    assert "1 flagged." in result.stdout


def test_a_project_with_no_containers_is_flagged(tmp_path: Path, script: Path) -> None:
    """An empty listing means nothing was verified, and that is flagged."""
    result, _ = _run(script, "containers", stubs=_stubs(tmp_path), listing=_listing(tmp_path, []))

    project = resolve_project_name({"project_name": "als-exemplar"})
    assert result.returncode == 0
    assert f"project {project} has no containers" in result.stdout


# ── --strict and the argument line ───────────────────────────────────────────


def test_strict_exits_1_when_a_container_is_flagged(tmp_path: Path, script: Path) -> None:
    """``--strict`` turns anything flagged into exit 1."""
    result, _ = _run(
        script,
        "--strict",
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert result.returncode == 1
    assert "2 flagged. Exit 1: --strict." in result.stdout


def test_strict_exits_0_when_nothing_is_flagged(tmp_path: Path, script: Path) -> None:
    """``--strict`` with nothing flagged still exits 0."""
    rows = [row for row in _FOUR_ROWS if row["Name"] in {"demo-postgres-1", "demo-qmd-1"}]
    result, _ = _run(
        script,
        "--strict",
        "containers",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, rows),
    )

    assert result.returncode == 0
    assert "Nothing flagged." in result.stdout


def test_strict_exits_1_on_a_failed_probe_alone(tmp_path: Path, script: Path) -> None:
    """An endpoint that gets no answer counts under ``--strict`` too."""
    result, log = _run(
        script,
        "--strict",
        "dispatch",
        stubs=_stubs(tmp_path),
        listing=_listing(tmp_path, _FOUR_ROWS),
        STUB_CURL_RC="7",
    )

    assert result.returncode == 1
    assert "1 flagged. Exit 1: --strict." in result.stdout
    assert _runtime_calls(log) == []


def test_group_arguments_still_select_groups(tmp_path: Path, script: Path) -> None:
    """A group argument runs that group alone, wherever ``--strict`` sits."""
    stubs = _stubs(tmp_path)
    listing = _listing(tmp_path, _FOUR_ROWS)

    _, log = _run(script, "containers", stubs=stubs, listing=listing)
    assert _runtime_calls(log) and not [line for line in log if line.startswith("curl ")]

    _, log = _run(script, "web", "dispatch", stubs=stubs, listing=listing)
    assert log and _runtime_calls(log) == []

    before, before_log = _run(
        script, "dispatch", "--strict", stubs=stubs, listing=listing, STUB_CURL_RC="7"
    )
    after, after_log = _run(
        script, "--strict", "dispatch", stubs=stubs, listing=listing, STUB_CURL_RC="7"
    )
    assert (before.returncode, before.stdout, before_log) == (
        after.returncode,
        after.stdout,
        after_log,
    )


@pytest.mark.parametrize("argument", ["--strcit", "servics"])
def test_an_unknown_argument_exits_2_and_runs_nothing(
    tmp_path: Path, script: Path, argument: str
) -> None:
    """A mistyped option or group is refused before anything runs."""
    result, log = _run(
        script, argument, stubs=_stubs(tmp_path), listing=_listing(tmp_path, _FOUR_ROWS)
    )

    assert result.returncode == 2
    assert argument in result.stderr
    assert log == []


def test_no_answering_runtime_fails_under_strict(tmp_path: Path, script: Path) -> None:
    """Nothing verified is flagged, so ``--strict`` exits 1 on it."""
    result, _ = _run(
        script,
        "--strict",
        "containers",
        stubs=_stubs(tmp_path, runtimes=False),
        listing=_listing(tmp_path, _FOUR_ROWS),
    )

    assert result.returncode == 1
    assert "no container runtime answers" in result.stdout

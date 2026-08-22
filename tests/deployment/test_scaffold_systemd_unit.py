"""The systemd unit template, pinned to the bytes it emits.

A unit file is read by systemd on a host nobody is watching, at boot, and every
line of it is a directive rather than prose: a stray blank, a section header
that lost its brackets, a directive that drifted out of its section, and the
unit either fails to load or loads meaning something else. So this template is
held to its output byte for byte, against ``goldens/osprey.service``.

The byte specification lives here rather than in the three-zone exemplar that
pins the CI pair (``tests/fixtures/lifecycle_repo.py``, held byte-exact by
``tests/cli/test_emitted_artifacts_clean.py``), because the unit is not part of
what a deployment repo carries: it is emitted for a host, from two absolute
paths the repo does not know, and installed outside the repo entirely. One
specification, in one place — the thing the retired hand-built goldens got
wrong was having a second.

**Update discipline.** The golden is renderer output, never hand-edited. When a
template change is deliberate, regenerate it in the same reviewed change::

    PYTHONPATH=src .venv/bin/python -c "
    from pathlib import Path
    from tests.deployment.test_scaffold_systemd_unit import GOLDEN_PATH, render_unit
    GOLDEN_PATH.write_text(render_unit(), encoding='utf-8')"

so the diff a reviewer reads is the template edit and its consequence, side by
side.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey
from osprey.cli.deploy_scaffold_templates import (
    DEPLOY_DOCS_URL,
    SYSTEMD_MARKER,
    SYSTEMD_START_TIMEOUT_SEC,
    SYSTEMD_TEMPLATE,
    SYSTEMD_UNIT_NAME,
    build_systemd_context,
    render,
)

GOLDENS = Path(__file__).parent / "goldens"
EXEMPLAR_DIR = GOLDENS / "exemplar-profile"
GOLDEN_PATH = GOLDENS / SYSTEMD_UNIT_NAME

#: Passed in place of the installed version so the golden does not change with
#: the release the test happens to run under.
FROZEN_VERSION = "OSPREY_VERSION"

#: The deploy host's coordinates, frozen. Both are properties of a machine
#: rather than of the profile, so a render that read them off the host running
#: the tests would produce a different unit on every checkout.
FROZEN_REPO_ROOT = Path("/srv/osprey/demo-facility")
FROZEN_OSPREY_BIN = "/usr/local/bin/osprey"


def render_unit(profile: dict[str, Any] | None = None) -> str:
    """Render the unit for a profile, with the host's half held fixed."""
    if profile is None:
        profile = yaml.safe_load((EXEMPLAR_DIR / "profile.yml").read_text(encoding="utf-8"))
    context = build_systemd_context(profile, FROZEN_REPO_ROOT, FROZEN_OSPREY_BIN, FROZEN_VERSION)
    return render(SYSTEMD_TEMPLATE, context)


@pytest.fixture(scope="module")
def rendered() -> str:
    """The unit the exemplar renders."""
    return render_unit()


@pytest.fixture(scope="module")
def directives(rendered: str) -> dict[str, list[str]]:
    """The unit's directives, keyed by ``[Section].Name``.

    Parsed rather than pattern-matched so a directive that drifted into the
    wrong section reads as a missing one here, which is how systemd reads it
    too.
    """
    parsed: dict[str, list[str]] = {}
    section = ""
    for line in rendered.splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif line and not line.startswith("#"):
            name, _, value = line.partition("=")
            parsed.setdefault(f"{section}.{name}", []).append(value)
    return parsed


def test_render_matches_the_golden_byte_for_byte(rendered: str) -> None:
    """The template emits exactly what is committed under ``goldens/``."""
    assert GOLDEN_PATH.is_file(), f"missing golden fixture: {GOLDEN_PATH}"
    assert rendered == GOLDEN_PATH.read_text(encoding="utf-8")


def test_installed_version_is_the_only_value_that_moves_with_a_release(
    rendered: str,
) -> None:
    """Two renders of one profile differ in exactly the ``osprey-version:`` line.

    The scaffolder normalizes that line away before deciding whether a re-emit
    would change anything, so an upgrade rewrites no unit. A second
    release-dependent value would be masked by the same normalization and would
    silently stop re-emission from noticing a real change.
    """
    profile = yaml.safe_load((EXEMPLAR_DIR / "profile.yml").read_text(encoding="utf-8"))
    context = build_systemd_context(
        profile, FROZEN_REPO_ROOT, FROZEN_OSPREY_BIN, osprey.__version__
    )
    current = render(SYSTEMD_TEMPLATE, context)

    differing = [
        (left, right)
        for left, right in zip(rendered.splitlines(), current.splitlines(), strict=True)
        if left != right
    ]
    assert differing == [
        (f"# osprey-version: {FROZEN_VERSION}", f"# osprey-version: {osprey.__version__}")
    ]


def test_the_unit_carries_the_provenance_marker_in_its_header(rendered: str) -> None:
    """Re-emission is only safe for a file the scaffolder can recognize.

    The marker is looked for in the header alone, so it has to be there — a
    unit that lost it would be treated as hand-written and never updated again.
    """
    header = rendered.splitlines()[:20]
    assert f"# osprey-scaffold: {SYSTEMD_MARKER}" in header


def test_the_deployment_survives_the_start_command_returning(
    directives: dict[str, list[str]],
) -> None:
    """``osprey up -d`` returns while the containers keep running.

    Without ``RemainAfterExit``, systemd would call the unit dead the moment
    the start command exits, and a later ``systemctl --user stop`` would find
    nothing to stop — leaving ``ExecStop`` unreached and the stack up.
    """
    assert directives["Service.Type"] == ["oneshot"]
    assert directives["Service.RemainAfterExit"] == ["yes"]


def test_both_lifecycle_commands_are_absolute_and_run_in_the_repo(
    directives: dict[str, list[str]],
) -> None:
    """A unit inherits a stripped PATH and no working directory.

    Every osprey verb resolves the deployment by walking up from where it runs,
    so a unit that did not pin the directory would act on whatever repo happens
    to be above systemd's cwd — or on none at all.
    """
    assert directives["Service.WorkingDirectory"] == [str(FROZEN_REPO_ROOT)]
    assert directives["Service.ExecStart"] == [f"{FROZEN_OSPREY_BIN} up -d"]
    assert directives["Service.ExecStop"] == [f"{FROZEN_OSPREY_BIN} down"]


def test_the_start_timeout_outlasts_an_image_pull(directives: dict[str, list[str]]) -> None:
    """systemd's 90-second default would kill a cold start halfway through."""
    assert directives["Service.TimeoutStartSec"] == [str(SYSTEMD_START_TIMEOUT_SEC)]
    assert SYSTEMD_START_TIMEOUT_SEC > 90


def test_the_stop_timeout_outlasts_a_full_teardown(directives: dict[str, list[str]]) -> None:
    """A stop that runs long is SIGKILLed, not waited for.

    ``TimeoutStopSec`` falls back to the same ~90-second default as the start,
    and a stack whose containers each take their SIGTERM grace period can pass
    it. systemd then kills the unit's cgroup, cutting ``osprey down`` off partway
    and leaving half a stack for the next start to trip over — so the unit has to
    name its own bound rather than inherit that one.
    """
    stop = directives["Service.TimeoutStopSec"]
    assert stop, "the unit stopped bounding its own teardown"
    assert int(stop[0]) > 90


def test_the_unit_installs_into_the_user_manager(
    rendered: str, directives: dict[str, list[str]]
) -> None:
    """``default.target`` is what a ``systemd --user`` instance reaches at login.

    Paired with ``loginctl enable-linger``, which the header spells out: without
    it the user manager never starts at boot and an enabled unit simply does not
    run.
    """
    assert directives["Install.WantedBy"] == ["default.target"]
    assert "loginctl enable-linger" in rendered[: rendered.index("[Unit]")]


def test_the_header_names_only_commands_that_exist(rendered: str) -> None:
    """A header naming a verb the CLI does not have costs the reader a try."""
    invocations = re.findall(r"`osprey ([a-z][a-z -]*)`", rendered)
    assert invocations, "the header stopped explaining how the unit is emitted"
    assert set(invocations) <= {"scaffold systemd", "up", "up -d", "down"}


def test_the_documentation_link_is_the_deployment_how_to(
    directives: dict[str, list[str]],
) -> None:
    """``systemctl status`` shows it to whoever is debugging a failed boot."""
    assert directives["Unit.Documentation"] == [DEPLOY_DOCS_URL]
    assert DEPLOY_DOCS_URL.startswith("https://")


def test_the_description_names_the_facility(directives: dict[str, list[str]]) -> None:
    """One host can run more than one deployment; ``systemctl`` lists them all."""
    assert directives["Unit.Description"] == ["Demo Facility OSPREY deployment"]

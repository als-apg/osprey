"""Assertions against the parsed `.github/workflows/ci.yml`, proving the
secret-free Docker-stack e2e lanes (Tiled roundtrip, VA substrate
equivalence, control-assistant demo) are correctly wired — and that every
``dockerbuild``-marked e2e file stays out of the shared e2e-tests lane.

Everything here loads ci.yml with ``yaml.safe_load`` and asserts against the
parsed structure — never a text/regex match, so re-flowing YAML style can't
fool a check. YAML 1.1 parses the bare `on:` workflow-trigger key to the
Python boolean ``True``; this module never indexes ``workflow["on"]``.

Two secret-free jobs exist because two e2e files needed one each:

* ``test_tiled_roundtrip.py`` (Task 4.1) is new in this phase and never had
  a lane at all.
* ``test_va_substrate_equivalence.py`` (Phase 1) had a CI bug: it carried no
  dedicated job and was only ever swept up by ``e2e-tests``' glob over
  ``tests/e2e/`` — contradicting its own module docstring ("never collected
  by the fast lane"). Fixing that means giving it a real lane, not just
  removing its only lane via ``--ignore``.

Every assertion is paired with a mutation test: a fresh, in-memory-mutated
copy of the parsed workflow reintroduces exactly the bug the assertion
exists to catch, and the same assertion must then fail. ci.yml itself is
never edited by this module.

One check reads a source file rather than the parsed workflow: the unit
lane's ``--dist loadgroup`` flag is only safe because ``tests/conftest.py``
overrides ``pytest_xdist_make_scheduler``, and that override lives in
Python, not in YAML. Its *behaviour* is covered by
``tests/infrastructure/test_xdist_scheduler.py``; what is pinned here is the
pairing — the flag the unit lane runs with must never outlive the scheduler
behind it.

The steps of the caller jobs (``test``, ``lint``, ``docs``, ``package``,
``frontend-js``) run ``scripts/check.py run <id>`` and spell no check. The
workflow every pin reads has each such call expanded into the commands it
resolves to, through the script's own ``resolve``, so a pin on what a step
runs reads what CI runs; the section "One check set" reads the workflow as
written and holds that its callers spell no check themselves.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import importlib.util
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from osprey_connectors_tests._paths import TESTS_ROOT as CONNECTORS_TESTS_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CHECK_SCRIPT = "scripts/check.py"


def _load_by_path(relative: str, name: str) -> ModuleType:
    # import-time required because scripts/ is not a package: the module is
    # loaded by path, under a name no import uses, and registered in
    # sys.modules before exec so @dataclass can resolve annotations through
    # cls.__module__.
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The one declaration of the repository's checks, which the caller jobs run:
#: what every checks module declares, collected.
check_set = _load_by_path(CHECK_SCRIPT, "_check_set_by_path")
#: The framework's checks module: the unit sweep's selection and the package build.
framework_checks = check_set.FRAMEWORK

TILED_JOB = "tiled-roundtrip-e2e"
VA_JOB = "va-substrate-equivalence-e2e"
UNIT_TEST_JOB = "test"
E2E_TESTS_JOB = "e2e-tests"
GATE_JOB = "all-checks-passed"
SECRET_TOKEN = "secrets.ALS_APG_API_KEY"
TILED_TEST_FILE = "tests/e2e/test_tiled_roundtrip.py"
VA_TEST_FILE = "tests/e2e/test_va_substrate_equivalence.py"
LIFECYCLE_TEST_FILE = "tests/e2e/test_deploy_lifecycle.py"
ORM_JOB = "orm-roundtrip-e2e"
ORM_TEST_FILE = "tests/e2e/test_orm_roundtrip.py"
GRID_TEST_FILE = "tests/e2e/test_grid_scan_roundtrip.py"
QUEUE_JOB = "bluesky-queue-e2e"
QUEUE_TEST_FILE = "tests/e2e/test_bluesky_queue_e2e.py"
SCAN_AGENTIC_JOB = "scan-agentic-e2e"
SCAN_AGENTIC_TEST_FILE = "tests/e2e/test_plan_stack_agentic.py"
SCAN_AGENTIC_SKIP_GATE_STEP = "Fail the lane on any skipped test"
ARCHIVER_JOB = "archiver-world-e2e"
ARCHIVER_TEST_FILE = "tests/e2e/test_archiver_world_e2e.py"
TARGET_SWITCH_JOB = "target-switch-e2e"
#: The agent-driven half of the same feature. A separate lane rather than a
#: second step in the one above because it needs the LLM secret and the CLI,
#: and lane-level `if:` gating is the only granularity GitHub offers.
TARGET_SWITCH_AGENTIC_JOB = "target-switch-agentic-e2e"
VA_LIVE_JOB = "va-live-e2e"
VA_BLUESKY_JOB = "va-bluesky-deploy-e2e"
#: The run step of each lane that executes modules from tests/va/e2e/. The
#: partition below is over this mapping, so a further live lane is registered
#: here and nowhere else.
VA_LIVE_LANE_RUN_STEPS = {
    TARGET_SWITCH_JOB: "Run the control-target switch E2E",
    VA_LIVE_JOB: "Run the live virtual accelerator E2E",
    VA_BLUESKY_JOB: "Run the two-lane plan deployment E2E",
}
VA_E2E_ENABLE_FLAG = "OSPREY_VA_E2E_ENABLE"
VA_LIVE_SKIP_GATE_STEP = "Fail the lane on any skipped test"
TWO_SHAPE_JOB = "two-shape-boot-e2e"
TWO_SHAPE_TEST_FILE = "tests/e2e/test_two_shape_boot.py"
TWO_SHAPE_BOOT_STEP = "Run two-shape boot E2E (podman)"
TWO_SHAPE_PROVIDER_STEP = "Step 0 — assert the compose provider is podman-compose 1.0.6"
#: The parse-only counterpart runs in a Tier 0 job, not a lane of its own —
#: it needs no container runtime, only a renderer and a YAML parser.
PARSE_ONLY_JOB = "static-checks"
PARSE_ONLY_STEP = "Parse the merged podman topology (podman-compose config, no containers)"
#: Re-proves the recorded CLI tool inventory against the real pinned build.
INVENTORY_JOB = "static-checks"
INVENTORY_STEP = "Check the recorded CLI tool inventory against the pinned build"
PODMAN_COMPOSE_PIN = "podman-compose==1.0.6"
#: The containers.conf key that forces podman's choice of external provider.
COMPOSE_PROVIDER_FORCE_KEY = "compose_providers"
#: The env-var expression of the same choice, which sidesteps the config chain.
COMPOSE_PROVIDER_FORCE_ENV = "PODMAN_COMPOSE_PROVIDER"
PODMAN_PIN_STEP = "Pin podman-compose 1.0.6 as podman's compose provider"
#: The runner-image presence guard. Matched with a negative lookahead rather
#: than as a substring: ``command -v podman-compose`` contains the shorter
#: string and appears in BOTH pin steps, so a plain ``in`` check reports the
#: lane as guarded when it is not. Anchoring on the redirection that happens to
#: follow it today would work but would break the moment someone writes
#: ``command -v podman || ...``; the lookahead pins the command name itself.
_PODMAN_PRESENCE_GUARD_RE = re.compile(r"command -v podman(?![-\w])")
OVERLAY_JOB = "dispatch-overlay-e2e"
OVERLAY_TEST_FILE = "tests/e2e/test_dispatch_overlay_visibility.py"
CATALOG_JOB = "bluesky-catalog-e2e"
SANDBOX_JOB = "bluesky-sandbox-escape-e2e"
BENCHMARKS_JOB = "channel-finder-benchmarks"
BENCHMARKS_TEST_FILE = "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py"
TIER2_JOB = "tier2-model-benchmark"
TIER2_DIR = "tests/e2e/tier2"
FLOOR_JOB = "dependency-floor"
NEXTCLOUD_JOB = "nextcloud-talk-bridge-e2e"
NEXTCLOUD_TEST_FILE = "tests/e2e/test_nextcloud_talk_bridge_e2e.py"
NEXTCLOUD_DOCKERFILE = "tests/e2e/fixtures/Dockerfile.nextcloud_talk"
GCHAT_JOB = "gchat-bridge-e2e"
GCHAT_TEST_FILE = "tests/e2e/test_gchat_bridge_e2e.py"
GCHAT_SMOKE_FILE = "tests/e2e/fixtures/test_gchat_fixture_smoke.py"
GCHAT_EXTRA = "gchat"
GCHAT_SKIP_GATE_STEP = "Fail the lane on any skipped test"
PUBSUB_FIXTURE_NAME = "pubsub_emulator"
TEAMS_JOB = "teams-bridge-e2e"
TEAMS_TEST_FILE = "tests/e2e/test_teams_bridge_e2e.py"
TEAMS_SMOKE_FILE = "tests/e2e/fixtures/test_teams_fixture_smoke.py"
TEAMS_EXTRA = "teams"
TEAMS_SKIP_GATE_STEP = "Fail the lane on any skipped test"

NO_MODEL_JOB = "e2e-no-model"
NO_MODEL_RUN_STEP = "Run the model-free E2E files"
NO_MODEL_SKIP_GATE_STEP = "Fail the lane on any skipped test"
#: The files the model-free lane names, in the order its run step names them.
NO_MODEL_TEST_FILES = (
    "tests/e2e/test_scripted_model.py",
    "tests/e2e/claude_code/test_safety_gates_scripted.py",
    "tests/e2e/test_dispatch_deploy_render.py",
    "tests/e2e/web_terminals/test_scaffold_render_roundtrip.py",
    "tests/e2e/web_terminals/test_skills_overlay_discovery.py",
    "tests/e2e/web_terminals/test_loopback_chokepoint.py",
    "tests/e2e/web_terminals/test_session_restart_e2e.py",
    "tests/e2e/web_terminals/test_terminal_auth_e2e.py",
    "tests/e2e/web_terminals/test_render_permissions_scripted.py",
    "tests/e2e/claude_code/test_chat_approval_gates_scripted.py",
    "tests/e2e/claude_code/test_chat_plan_mode_scripted.py",
    "tests/e2e/claude_code/test_chat_attachment_read_scripted.py",
)

#: The agent record's replay: model-free, but its own lane, because it deploys
#: two stacks in turn and waits out a stopped source twice.
RECORD_E2E_JOB = "agent-record-e2e"
RECORD_E2E_RUN_STEP = "Run the agent record replay"
RECORD_E2E_SKIP_GATE_STEP = "Fail the lane on any skipped test"
RECORD_E2E_FILE = "tests/e2e/test_agent_record_e2e.py"
#: The two stopped-source waits of five minutes each, plus two deploys: the
#: run step's floor. Below it the lane reds on its own waits, not on a fault.
RECORD_E2E_MIN_STEP_MINUTES = 40
#: Every model-free file, whichever of the two model-free lanes runs it.
MODEL_FREE_TEST_FILES = (*NO_MODEL_TEST_FILES, RECORD_E2E_FILE)
SAME_REPO_CLAUSE = "github.event.pull_request.head.repo.full_name == github.repository"

ALS_APG_BASE_URL_ENV = "ALS_APG_BASE_URL"
E2E_PROVIDER_ENV = "OSPREY_E2E_PROVIDER"
PROBE_BASE_VAR = "ALS_APG_PROBE_BASE"
PROBE_KEY_ENV = "ALS_APG_API_KEY"
PROBE_TARGET_PATH = "/v1/messages"
# `ALS_APG_PROBE_BASE="${ALS_APG_BASE_URL:-https://…}"` — the `:-default` is optional.
PROBE_BASE_ASSIGNMENT = re.compile(rf'{PROBE_BASE_VAR}="\$\{{{ALS_APG_BASE_URL_ENV}(:-[^}}]*)?\}}"')
# Guards against silent under-discovery: if probe steps are renamed or reshaped
# so the finder stops matching them, the count check fails loudly instead of
# passing over an empty list. Raise this when probe lanes are added.
EXPECTED_MIN_ALS_APG_PROBES = 10

#: Where the scheduler override is defined. Both conftests import it by name
#: from here (see ``test_both_conftests_import_the_scheduler_override``).
SCHEDULER_SOURCE = CONNECTORS_TESTS_ROOT / "osprey_connectors_tests" / "_harness" / "_guards.py"
#: The module both conftests import the override from.
SCHEDULER_MODULE = "osprey_connectors_tests._harness._guards"
#: The conftests that must carry the override for pytest to call it.
SCHEDULER_CONFTESTS = (
    Path(__file__).resolve().parents[1] / "conftest.py",
    CONNECTORS_TESTS_ROOT / "conftest.py",
)
PARALLEL_FLAGS = ("-n 4", "--dist loadgroup")

#: The per-test hang-breaker on the unit lane. Both halves are load-bearing and
#: pinned as literal substrings: the value has to clear a cold container image
#: pull (the ``xdist_group("docker")`` files legitimately sit in one for
#: minutes) while still ending a hang far inside the step cap, and the method
#: has to be ``signal`` — ``thread`` ends a timed-out test with ``os._exit`` of
#: the xdist worker, which is the "node down: Not properly terminated" wedge
#: osprey_connectors_tests._harness.ci_diagnostics documents.
PER_TEST_TIMEOUT_FLAGS = ("--timeout=600", "--timeout-method=signal")
SCHEDULER_HOOK = "pytest_xdist_make_scheduler"
SCHEDULER_CLASS = "FileOrGroupScheduling"

#: An xdist worker count on a pytest invocation. Used by two jobs that must
#: stay serial for different reasons, so it lives here rather than beside
#: either. The lookbehind keeps it from matching inside a longer flag.
_XDIST_WORKERS_RE = re.compile(r"(?<!\S)-n\s+\d")


@dataclass(frozen=True)
class SerialHost:
    """The one CI step that runs a lane-leaving marker, alone."""

    #: ci.yml job id.
    job: str
    #: The step's name inside that job.
    step: str
    #: What the step collects. A marked file must live under it, because the
    #: parallel lane deselects the marker everywhere and nothing else runs it.
    path: str
    #: The literal on the step's pytest line that picks the marker back up.
    #: A step may hold other pytest lines beside that one; each of them
    #: deselects the marker.
    select: str


@dataclass(frozen=True)
class LaneLeavingMarker:
    """A marker whose tests cannot share a machine with xdist workers.

    Each row holds the pair: every invocation that can reach a marked file is
    one of the row's hosts or deselects the marker, and the local check
    scripts mirror both halves. A row with no marked file left is dead wiring
    and fails the module, so deleting a marker's last test deletes its row and
    its host step with it.
    """

    marker: str
    hosts: tuple[SerialHost, ...]

    def deselect_term(self) -> str:
        return f"not {self.marker}"

    def select(self) -> str:
        return f"-m {self.marker}"

    def mark_re(self) -> re.Pattern[str]:
        return re.compile(rf"pytest\.mark\.{re.escape(self.marker)}\b")


#: Every marker that leaves the parallel lane, one row each. A new one is a row
#: here plus its serial step in ci.yml; every case below is parametrised over
#: this table.
LANE_LEAVING_MARKERS = (
    # The real-terminal suite: scenarios that assert on repaint cadence and on a
    # spinner advancing between frames.
    LaneLeavingMarker(
        "pty",
        (SerialHost("test", "Run real-terminal pty suite serially", "tests/pty/", "-m pty"),),
    ),
    # Absolute wall-clock budgets: a budget timed beside loaded workers
    # measures the workers.
    LaneLeavingMarker(
        "wallclock",
        (
            SerialHost(
                "qmd-sidecar-e2e",
                "Run the wall-clock budgets serially",
                "tests/integration/",
                "-m wallclock",
            ),
            # The image-free budgets, on every event: the step selects the
            # marker over the tree, minus the directories with hosts of their own.
            SerialHost("test", "Run the wall-clock budgets serially", "tests/", "-m wallclock"),
        ),
    ),
)


def _row(marker: str) -> LaneLeavingMarker:
    return next(row for row in LANE_LEAVING_MARKERS if row.marker == marker)


def _lane_deselect(rows: tuple[LaneLeavingMarker, ...]) -> str:
    """The parallel lane's marker expression for *rows*."""
    return '-m "' + " and ".join(row.deselect_term() for row in rows) + '"'


#: The parallel lane's marker expression, derived from the table. Matched as a
#: literal substring: single spaces and the quoting are part of the pin.
UNIT_LANE_DESELECT = _lane_deselect(LANE_LEAVING_MARKERS)
assert UNIT_LANE_DESELECT == '-m "not pty and not wallclock"', UNIT_LANE_DESELECT

#: The row the terminal-only cases read (Linux gate, files that open a pty).
TERMINAL_ROW = _row("pty")
TERMINAL_HOST = TERMINAL_ROW.hosts[0]
#: One call of the check command in a step's script, up to the shell operator
#: that ends it. ``2>&1`` is a redirection of the call, not an operator.
_CHECK_CALL_RE = re.compile(
    r"(?:uv run python|python3) " + re.escape(CHECK_SCRIPT) + r"(?P<args>(?:2>&1|[^|&;\n#])*)"
)


def _uncommented_lines(run: object) -> list[str]:
    """The lines of a ``run:`` script, each without its shell comment.

    Everything from a ``#`` onward is dropped, so a flag parked in a trailing
    comment, as in ``... $COV_ARGS  # restore -n 4``, is not read as part
    of the command. Truncating at a ``#`` inside a quoted argument would only
    ever make a pin stricter, so the naive split is the safe direction to be
    wrong in.
    """
    return [line.split("#", 1)[0].rstrip() for line in str(run).splitlines()]


def _check_calls(step: dict[str, Any]) -> list[list[str]]:
    """The arguments of each ``scripts/check.py`` call in *step*, as shell words."""
    return [
        [word for word in shlex.split(match["args"]) if word != "2>&1"]
        for line in _uncommented_lines(step.get("run", ""))
        for match in _CHECK_CALL_RE.finditer(line)
    ]


def _resolved(call: list[str]) -> list[list[str]]:
    """The command lines *call* resolves to on a GitHub runner, without running them.

    A shard named by a matrix expression is dropped, so the unit sweep resolves
    to the selection all three shards add up to; ``$COV_ARGS`` is the empty
    string every cell but one sets.
    """
    words: list[str] = []
    skip = False
    for index, word in enumerate(call):
        if skip:
            skip = False
            continue
        if word == "--shard" and "${{" in call[index + 1]:
            skip = True
            continue
        if word != "$COV_ARGS":
            words.append(word)
    with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
        return check_set.resolve(words)


def _render(argv: list[str]) -> str:
    """*argv* as one shell line; a word with a space and no quote is double-quoted,
    as ci.yml spells such a word."""
    words = []
    for word in argv:
        if shlex.quote(word) == word:
            words.append(word)
        elif not set(word) & set('"`\\\n'):
            words.append(f'"{word}"')
        else:
            words.append(shlex.quote(word))
    return " ".join(words)


def _expand_run(run: str) -> str:
    """*run* with each check.py call replaced by the commands it resolves to."""
    lines = []
    for line in run.splitlines():
        code = line.split("#", 1)[0]
        match = _CHECK_CALL_RE.search(code)
        if match is None:
            lines.append(line)
            continue
        call = [word for word in shlex.split(match["args"]) if word != "2>&1"]
        rendered = "\n".join(_render(argv) for argv in _resolved(call))
        tail = code[match.end() :]
        if "2>&1" in match["args"]:
            tail = " 2>&1" + tail
        lines.append(code[: match.start()] + rendered + tail)
    return "\n".join(lines) + ("\n" if run.endswith("\n") else "")


@functools.cache
def _raw_workflow() -> dict[str, Any]:
    """ci.yml as written, parsed once per process; read only through ``_load_raw_workflow``."""
    with CI_YML.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded is not None, f"{CI_YML} parsed to None"
    return loaded


def _load_raw_workflow() -> dict[str, Any]:
    """A private deep copy of ci.yml as written: its caller steps call check.py."""
    return copy.deepcopy(_raw_workflow())


@functools.cache
def _parsed_workflow() -> dict[str, Any]:
    """ci.yml with every check.py call expanded; read only through ``_load_workflow``."""
    expanded = copy.deepcopy(_raw_workflow())
    for body in expanded["jobs"].values():
        for step in body.get("steps", []):
            if isinstance(step.get("run"), str) and CHECK_SCRIPT in step["run"]:
                step["run"] = _expand_run(step["run"])
    return expanded


def _load_workflow() -> dict[str, Any]:
    """A private deep copy of the parsed ci.yml, so a mutation test's edits
    never reach any other test."""
    return copy.deepcopy(_parsed_workflow())


@pytest.fixture()
def workflow() -> dict[str, Any]:
    return _load_workflow()


@pytest.fixture()
def pyproject() -> dict[str, Any]:
    return _load_pyproject()


def _jobs(wf: dict[str, Any]) -> dict[str, Any]:
    return wf["jobs"]


def _job_declares_secret(wf: dict[str, Any], job_name: str, token: str) -> bool:
    """Search a job's fully serialized form for a secret-expression token.

    Serializing the whole job (not just top-level keys) so the search also
    covers step-level ``env:``/``run:`` blocks, matching how
    ``ALS_APG_API_KEY`` actually appears in the secret-gated jobs (e.g.
    ``e2e-tests``'s pre-flight probe step).
    """
    return token in json.dumps(_jobs(wf)[job_name])


def _find_named_step(wf: dict[str, Any], job_name: str, step_name: str) -> dict[str, Any]:
    for step in _jobs(wf)[job_name]["steps"]:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"job '{job_name}' has no step named '{step_name}'")


# ---------------------------------------------------------------------------
# (c) e2e-tests' run step --ignores both new-lane files
# ---------------------------------------------------------------------------


def test_e2e_tests_ignores_both_new_lane_files(workflow: dict[str, Any]) -> None:
    step = _find_named_step(workflow, E2E_TESTS_JOB, "Run E2E tests")
    run_text = step["run"]
    assert f"--ignore={TILED_TEST_FILE}" in run_text
    assert f"--ignore={VA_TEST_FILE}" in run_text


def _drop_ignore_line(run_text: str, target_file: str) -> str:
    """Remove the `--ignore=<target_file>` line, whichever way it happens to
    be terminated (line-continuation backslash mid-list, or bare newline on
    the last entry)."""
    lines = run_text.splitlines(keepends=True)
    kept = [line for line in lines if f"--ignore={target_file}" not in line]
    assert len(kept) == len(lines) - 1, f"expected exactly one line dropped for {target_file}"
    return "".join(kept)


def test_e2e_tests_ignores_both_new_lane_files__mutation_drops_tiled_ignore() -> None:
    """Removing only the tiled-roundtrip ignore line must fail — proves the
    two ignore assertions are independent, not satisfied by either alone."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], TILED_TEST_FILE)
    assert f"--ignore={VA_TEST_FILE}" in step["run"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert f"--ignore={TILED_TEST_FILE}" in step["run"]


def test_e2e_tests_ignores_both_new_lane_files__mutation_drops_va_ignore() -> None:
    """Removing only the VA-substrate ignore line must fail the same way,
    confirming the two files are checked independently in the real assertion."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], VA_TEST_FILE)
    assert f"--ignore={TILED_TEST_FILE}" in step["run"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert f"--ignore={VA_TEST_FILE}" in step["run"]


# ---------------------------------------------------------------------------
# (d) all-checks-passed depends on the new job(s)
# ---------------------------------------------------------------------------


def _needs_contains_both_new_jobs(wf: dict[str, Any]) -> bool:
    """Deliberately `all(...)`, not `any(...)`. An `any`-shaped check would
    pass with only one of the two new jobs wired into the gate — exactly the
    silent-partial-fix shape this phase has repeatedly caught elsewhere."""
    needs = _jobs(wf)[GATE_JOB]["needs"]
    return all(job in needs for job in (TILED_JOB, VA_JOB))


def test_all_checks_passed_needs_both_new_jobs(workflow: dict[str, Any]) -> None:
    """(g) A single job nobody depends on can go red forever without
    blocking a merge; both new e2e lanes must actually gate the merge."""
    assert _needs_contains_both_new_jobs(workflow)


def test_all_checks_passed_needs_both_new_jobs__mutation_drops_only_tiled() -> None:
    """Dropping just tiled-roundtrip-e2e (VA stays) must still fail the
    'both' check — proves it isn't secretly an `any`."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(TILED_JOB)
    assert VA_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert _needs_contains_both_new_jobs(mutated)


def test_all_checks_passed_needs_both_new_jobs__mutation_drops_only_va() -> None:
    """Dropping just va-substrate-equivalence-e2e (tiled stays) must also
    fail the 'both' check, confirming both entries are independently load-bearing."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(VA_JOB)
    assert TILED_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert _needs_contains_both_new_jobs(mutated)


def _gate_stands_down_only_when_cancelled(wf: dict[str, Any]) -> bool:
    """The gate runs on every run that was not cancelled. `always()` would also
    run it in a superseded run, where every need reads `cancelled` and the gate
    reports a failure nobody can act on; `success()` (the default) would skip
    it when a need fails, and a skipped required check counts as passing."""
    condition = str(_jobs(wf)[GATE_JOB].get("if", ""))
    return "".join(condition.split()) == "${{!cancelled()}}"


def test_gate_job_stands_down_only_when_the_run_is_cancelled(workflow: dict[str, Any]) -> None:
    assert _gate_stands_down_only_when_cancelled(workflow)


def test_gate_job_stands_down_only_when_the_run_is_cancelled__mutation_reverts_to_always() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["if"] = "always()"
    with pytest.raises(AssertionError):
        assert _gate_stands_down_only_when_cancelled(mutated)


def test_gate_job_stands_down_only_when_the_run_is_cancelled__mutation_uses_success() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["if"] = "${{ success() }}"
    with pytest.raises(AssertionError):
        assert _gate_stands_down_only_when_cancelled(mutated)


# ---------------------------------------------------------------------------
# The unit-test job must install the extras its tests import
# ---------------------------------------------------------------------------


def _unit_test_install_cmd(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, UNIT_TEST_JOB, "Install dependencies")["run"]


def test_unit_test_job_installs_required_extras(workflow: dict[str, Any]) -> None:
    """The `virtual-accelerator` extra carries the serving stack the live
    tests/va suites need — `lume-pva-apg[ca,pva]`, which brings pcaspy and p4p
    with it. Those suites guard their imports, so without the extra they SKIP
    rather than error and the lane reports green having never touched Channel
    Access; `dev` carries pytest itself."""
    cmd = _unit_test_install_cmd(workflow)
    for extra in ("dev", "virtual-accelerator"):
        assert f"--extra {extra}" in cmd, (
            f"unit-test job must `uv sync --extra {extra}`; got: {cmd}"
        )


def test_unit_test_job_installs_required_extras__mutation_drops_extra() -> None:
    """Dropping a pinned extra must fail — otherwise the guard is decorative."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Install dependencies")
    step["run"] = step["run"].replace(" --extra virtual-accelerator", "")
    with pytest.raises(AssertionError):
        test_unit_test_job_installs_required_extras(mutated)


def test_bluesky_stack_is_a_core_dependency() -> None:
    """The bridge's unit tests guard their imports with `pytest.importorskip`,
    so a missing bluesky stack does not error — it SKIPS, and the scanner,
    RunEngine-integration and Tiled fault-isolation guards vanish from CI
    silently, inside a green check. Bluesky is a core part of OSPREY: pin the
    stack in [project] dependencies so plain `uv sync` always installs it and
    those tests can never resume skipping unnoticed."""
    pyproject = tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())
    core_deps = pyproject["project"]["dependencies"]
    for stack_dep in ("bluesky", "ophyd-async", "tiled"):
        assert any(dep.startswith(stack_dep) for dep in core_deps), (
            f"{stack_dep} must be a core dependency"
        )


def test_lume_pyat_is_a_core_dependency() -> None:
    """`osprey.simulation.engines.pyat_model` imports lume_pyat at module level
    — PyATLatticeModel subclasses LUMEPyATModel — so the VA cannot boot
    without it. Placement, not just
    presence, is what this guards: the dev-wheel channel builds the VA image's
    dependency manifest from the wheel's base `Requires-Dist` only
    (`wheel_build._wheel_base_requirements` drops every entry gated behind an
    extra). A pin parked in an optional extra would therefore vanish from
    `osprey-local-requirements.txt` silently, and the image would build green
    and fail at runtime on the first model import."""
    pyproject = tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())
    core_deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("lume-pyat") for dep in core_deps), (
        "lume-pyat must be a core dependency, not an extra"
    )


# ---------------------------------------------------------------------------
# The serving package is pinned in the `virtual-accelerator` extra, with both
# transports and the platform marker its Channel Access half needs
# ---------------------------------------------------------------------------
#
# The placement is the opposite of lume-pyat's above, and deliberately so.
# lume-pyat has to be core because the VA cannot boot without it on any host;
# the serving stack can only be installed where pcaspy loads, so it belongs in
# the extra that already carries that constraint. Both Dockerfiles install
# `[virtual-accelerator]` rather than naming the package, so this extra is the
# only place either image can learn the pin — which is what these guards
# protect.


#: The two platforms with a loadable Channel Access server — linux x86_64 on
#: PyPI's wheels, linux aarch64 on the one the VA service image builds — and
#: therefore the marker the serving entry must carry. Spelled with DOUBLE
#: quotes because this is compared against ``str(Requirement(...).marker)``,
#: which is packaging's normalised rendering — pyproject.toml itself writes the
#: same marker with single quotes, and both parse to this.
LIVE_CA_MARKER = (
    'sys_platform == "linux" and (platform_machine == "x86_64" or platform_machine == "aarch64")'
)


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())


def _va_extra(pyproject: dict[str, Any]) -> list[str]:
    return pyproject["project"]["optional-dependencies"]["virtual-accelerator"]


def _requirement(deps: list[str], name: str) -> Requirement | None:
    """The single requirement in ``deps`` naming ``name``, or None.

    Parsed rather than string-matched: every property asserted below (extras,
    specifier, marker) lives in a different part of the grammar, and a
    ``startswith`` check would be satisfied by a line that carries none of
    them.
    """
    matches = [req for dep in deps if (req := Requirement(dep)).name == name]
    assert len(matches) <= 1, f"{name} declared more than once: {matches}"
    return matches[0] if matches else None


def test_lume_pva_is_pinned_in_the_va_extra(pyproject: dict[str, Any]) -> None:
    """The extra is what puts the serving stack in both VA images: neither
    Dockerfile names lume-pva-apg, so an entry that is not here reaches no
    image at all. Core placement would be worse than absent — the `ca` extra
    requires pcaspy unconditionally, so a core pin would drag the Channel
    Access server onto every plain `uv sync`, including the macOS hosts the
    marker it carries exists to keep clear of a source build."""
    assert _requirement(_va_extra(pyproject), "lume-pva-apg") is not None, (
        "lume-pva-apg must be pinned in the `virtual-accelerator` extra"
    )
    assert _requirement(pyproject["project"]["dependencies"], "lume-pva-apg") is None, (
        "lume-pva-apg must not be a core dependency — see the docstring"
    )


def test_lume_pva_is_pinned_in_the_va_extra__mutation_drops_the_pin() -> None:
    """Dropping the entry must fail — otherwise the guard is decorative."""
    mutated = _load_pyproject()
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep for dep in _va_extra(mutated) if Requirement(dep).name != "lume-pva-apg"
    ]
    with pytest.raises(AssertionError):
        test_lume_pva_is_pinned_in_the_va_extra(mutated)


def test_lume_pva_is_pinned_in_the_va_extra__mutation_also_declares_it_in_core() -> None:
    """A core declaration must fail even with the extra entry left intact.

    The extra is deliberately NOT disturbed here. Removing it as well would
    trip the presence half of the assertion and the mutation would pass for a
    reason that has nothing to do with placement — leaving the core-exclusion
    half never exercised. Copying instead of moving is what aims this mutation
    at the assertion it is named for."""
    mutated = _load_pyproject()
    copied = [dep for dep in _va_extra(mutated) if Requirement(dep).name == "lume-pva-apg"]
    assert copied, "no lume-pva-apg in the extra — mutation is stale"
    mutated["project"]["dependencies"] = [*mutated["project"]["dependencies"], *copied]
    with pytest.raises(AssertionError, match="must not be a core dependency"):
        test_lume_pva_is_pinned_in_the_va_extra(mutated)


def test_lume_pva_pin_carries_both_transports(pyproject: dict[str, Any]) -> None:
    """`serving/runner.py` imports pcaspy, lume_pva_apg AND p4p at module
    scope, and the base package pulls neither transport — pcaspy arrives only
    through the `ca` extra and p4p only through `pva`. Either one missing is
    not a degraded serve, it is an ImportError at runner import, long after a
    green image build."""
    req = _requirement(_va_extra(pyproject), "lume-pva-apg")
    assert req is not None and req.extras == {"ca", "pva"}, (
        f"lume-pva-apg must carry both the `ca` and `pva` extras; got {req}"
    )


@pytest.mark.parametrize("dropped", ["ca", "pva"])
def test_lume_pva_pin_carries_both_transports__mutation_drops_one(dropped: str) -> None:
    """Dropping either transport must fail — a guard that only noticed the
    loss of both would let the p4p-typed value layer go missing silently."""
    kept = "pva" if dropped == "ca" else "ca"
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep.replace("[ca,pva]", f"[{kept}]") if Requirement(dep).name == "lume-pva-apg" else dep
        for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the extras spelling moved"
    )
    with pytest.raises(AssertionError):
        test_lume_pva_pin_carries_both_transports(mutated)


def test_lume_pva_pin_carries_the_live_ca_marker(pyproject: dict[str, Any]) -> None:
    """`lume-pva-apg[ca]` requires pcaspy with no marker of its own, so an
    unmarked serving pin drags the Channel Access server onto hosts with no
    loadable wheel — verified, not assumed (resolving `lume-pva-apg[ca,pva]`
    for macOS at 3.13 selects pcaspy 0.8.1, whose only artifact there is the
    sdist, i.e. the EPICS source build). The extra names pcaspy nowhere: the
    serving pin is its only route in, and that pin's `ca` extra carries the
    floor."""
    extra = _va_extra(pyproject)
    lume_pva = _requirement(extra, "lume-pva-apg")
    # Anchored absolutely: a marker retargeted to some other platform is still
    # a marker, and only the comparison against the platforms with a loadable
    # Channel Access server rejects it.
    assert lume_pva is not None and str(lume_pva.marker) == LIVE_CA_MARKER, (
        f"lume-pva-apg must stay marked {LIVE_CA_MARKER!r} — the platforms with a "
        f"loadable Channel Access server; got {lume_pva and lume_pva.marker}"
    )
    assert _requirement(extra, "pcaspy") is None, (
        "the `virtual-accelerator` extra must not declare pcaspy directly — it arrives "
        "through lume-pva-apg[ca], which carries its floor"
    )


def test_lume_pva_pin_carries_the_live_ca_marker__mutation_unmarks_it() -> None:
    """An unmarked serving pin must fail."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep.split(";", 1)[0].strip() if Requirement(dep).name == "lume-pva-apg" else dep
        for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the marker is already absent"
    )
    with pytest.raises(AssertionError):
        test_lume_pva_pin_carries_the_live_ca_marker(mutated)


def test_lume_pva_pin_carries_the_live_ca_marker__mutation_moves_to_another_platform() -> None:
    """Retargeting the serving pin to another platform must fail. Darwin is the
    pointed choice — it is where pcaspy's wheels exist but do not load, so a
    plausible edit could land here."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        f"{dep.split(';', 1)[0].strip()}; sys_platform == 'darwin'"
        if Requirement(dep).name == "lume-pva-apg"
        else dep
        for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the serving pin is gone"
    )
    with pytest.raises(AssertionError, match="lume-pva-apg must stay marked"):
        test_lume_pva_pin_carries_the_live_ca_marker(mutated)


def test_lume_pva_pin_carries_the_live_ca_marker__mutation_drops_aarch64() -> None:
    """Narrowing the serving pin back to x86_64 must fail: the arm64 VA image
    builds its own pcaspy, and without aarch64 in the marker its serving stack
    installs nothing there."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        f"{dep.split(';', 1)[0].strip()}; sys_platform == 'linux' and platform_machine == 'x86_64'"
        if Requirement(dep).name == "lume-pva-apg"
        else dep
        for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the marker is already x86_64-only"
    )
    with pytest.raises(AssertionError, match="lume-pva-apg must stay marked"):
        test_lume_pva_pin_carries_the_live_ca_marker(mutated)


def test_lume_pva_pin_carries_the_live_ca_marker__mutation_redeclares_pcaspy() -> None:
    """A direct pcaspy entry beside the serving pin must fail, marked or not."""
    mutated = _load_pyproject()
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        *_va_extra(mutated),
        "pcaspy>=0.8.1; sys_platform == 'linux' and platform_machine == 'x86_64'",
    ]
    with pytest.raises(AssertionError, match="must not declare pcaspy directly"):
        test_lume_pva_pin_carries_the_live_ca_marker(mutated)


def test_lume_pva_pin_is_exact(pyproject: dict[str, Any]) -> None:
    """Exact-pinned for the same reason as lume-base and lume-pyat: the runner
    subclass is written against one contract of a package whose extension
    seams are not yet a stable API, and a floor would let a resolve pick up a
    surface it was never written against. The version itself is not asserted —
    only that a bump is a deliberate edit rather than a resolver's choice."""
    req = _requirement(_va_extra(pyproject), "lume-pva-apg")
    assert req is not None
    assert [spec.operator for spec in req.specifier] == ["=="], (
        f"lume-pva-apg must be exact-pinned; got {req.specifier}"
    )


def test_lume_pva_pin_is_exact__mutation_relaxes_to_a_floor() -> None:
    """A `>=` floor must fail."""
    mutated = _load_pyproject()
    extra = _va_extra(mutated)
    before = list(extra)
    mutated["project"]["optional-dependencies"]["virtual-accelerator"] = [
        dep.replace("==", ">=") if Requirement(dep).name == "lume-pva-apg" else dep for dep in extra
    ]
    assert mutated["project"]["optional-dependencies"]["virtual-accelerator"] != before, (
        "mutation matched nothing — the pin is already not exact"
    )
    with pytest.raises(AssertionError):
        test_lume_pva_pin_is_exact(mutated)


# ---------------------------------------------------------------------------
# The unit-test job runs in parallel, and the scheduler that makes it safe
# still exists
# ---------------------------------------------------------------------------


def _unit_test_pytest_line(wf: dict[str, Any]) -> str:
    """The single ``pytest tests/`` invocation from the unit-test job's run
    block, isolated from the surrounding ``COV_ARGS`` shell conditional.

    Everything from an unquoted ``#`` onward is dropped first, so a flag named
    in a shell comment — whole-line or *trailing*, as in
    ``... $COV_ARGS  # TODO restore -n 4 --dist loadgroup`` — cannot satisfy
    the pin while the lane actually runs serial. Truncating at a ``#`` inside a
    quoted argument would only ever make the pin stricter, so the naive split
    is the safe direction to be wrong in.
    """
    run_text = _find_named_step(wf, UNIT_TEST_JOB, "Run unit tests")["run"]
    commands = [line.split("#", 1)[0] for line in run_text.splitlines()]
    lines = [cmd for cmd in commands if "pytest tests/" in cmd]
    assert len(lines) == 1, f"expected exactly one pytest invocation; got: {lines}"
    return lines[0]


def _missing_parallel_flags(cmd: str) -> list[str]:
    """Which of the parallel flags are absent (empty = fully wired). Both are
    load-bearing and checked independently: `-n 4` without `--dist loadgroup`
    falls back to xdist's default per-test `load` scheduling, which scatters a
    file's tests across workers and reinstates the module-state bleed."""
    return [flag for flag in PARALLEL_FLAGS if flag not in cmd]


def test_unit_test_job_runs_xdist_parallel(workflow: dict[str, Any]) -> None:
    """The unit lane was serial for as long as the suite had cross-test
    global-state bleed. That debt is paid, so the lane must actually collect
    the win — a silent revert to serial would cost ~3x wall clock on every
    push without failing anything."""
    line = _unit_test_pytest_line(workflow)
    missing = _missing_parallel_flags(line)
    assert missing == [], (
        f"the '{UNIT_TEST_JOB}' job in .github/workflows/ci.yml must invoke pytest with "
        f"{' '.join(PARALLEL_FLAGS)} — missing {missing} in: {line.strip()!r}. "
        f"Flags are matched as literal substrings (exact spelling, single spaces) on the "
        f"one `pytest tests/` line, so reflowing the command across backslash-continued "
        f"lines (the e2e lane's style) means updating _unit_test_pytest_line too."
    )


def test_unit_test_job_runs_xdist_parallel__mutation_flags_only_in_trailing_comment() -> None:
    """The hole a reviewer probe found: flags parked in a TRAILING shell
    comment while the lane runs serial. `# TODO restore ...` is exactly how a
    temporary revert gets written, so it must fail the pin, not satisfy it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace(" -n 4 --dist loadgroup", "").replace(
        "| tee ci-diag/pytest.log", "| tee ci-diag/pytest.log  # TODO restore -n 4 --dist loadgroup"
    )
    assert step["run"] != original, "parallel invocation not found — mutation is stale"
    assert _missing_parallel_flags(_unit_test_pytest_line(mutated)) == list(PARALLEL_FLAGS)


def test_unit_test_job_runs_xdist_parallel__mutation_drops_worker_count() -> None:
    """Dropping `-n 4` (leaving `--dist loadgroup`) must be reported: without
    a worker count xdist does not parallelize at all and the dist mode is
    inert."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    step["run"] = step["run"].replace(" -n 4", "")
    assert _missing_parallel_flags(_unit_test_pytest_line(mutated)) == ["-n 4"]


def test_unit_test_job_runs_xdist_parallel__mutation_drops_dist_mode() -> None:
    """Dropping `--dist loadgroup` (leaving `-n 4`) must be reported too — the
    dangerous half of the revert, since the lane still looks parallelized
    while the scheduler override goes inert."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    step["run"] = step["run"].replace(" --dist loadgroup", "")
    assert _missing_parallel_flags(_unit_test_pytest_line(mutated)) == ["--dist loadgroup"]


# ---------------------------------------------------------------------------
# The boot smoke selects a single preset per cell, so the matrix and the module's
# own list are two halves of one statement
# ---------------------------------------------------------------------------

BOOT_SMOKE_JOB = "build-boot-smoke"
BOOT_SMOKE_STEP = "Run Tier 1 boot smoke for preset"
BOOT_SMOKE_TEST_FILE = "tests/integration/test_build_boot.py"
#: The module-level list every test in the boot module is parametrised over, and
#: the source the lane's matrix has to reproduce one value at a time.
BOOT_SMOKE_PRESETS_CONSTANT = "PRESETS"
#: The parametrize argument whose ids the lane's `-k` selector matches on.
BOOT_SMOKE_PRESET_ARG = "preset"
#: The shell variable the step interpolates into `-k`, and the step-level env
#: entry that has to bind it to the matrix value.
BOOT_SMOKE_SELECTOR = '-k "$PRESET"'
BOOT_SMOKE_PRESET_ENV = "PRESET"


def _boot_module_source() -> str:
    """``tests/integration/test_build_boot.py`` as text.

    Read and parsed with ``ast`` rather than imported, as the two-shape constants above
    are: the module pulls in the MCP handshake helpers and exists to shell out to a real
    build, none of which a wiring check has any business loading.
    """
    return (CI_YML.parents[2] / BOOT_SMOKE_TEST_FILE).read_text(encoding="utf-8")


def _boot_module_presets(source: str) -> list[str]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == BOOT_SMOKE_PRESETS_CONSTANT for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError(
        f"{BOOT_SMOKE_TEST_FILE} defines no module-level {BOOT_SMOKE_PRESETS_CONSTANT}"
    )


def _parametrises_over_presets(decorator: ast.expr) -> bool:
    """Whether *decorator* is the ``preset``/``PRESETS`` parametrize.

    Both arguments are read. The argument name decides what the test id looks like and
    so what ``-k`` can match; the list decides which ids exist at all, and a parametrize
    over some other list would produce ids no cell of the matrix names.
    """
    if not isinstance(decorator, ast.Call) or len(decorator.args) < 2:
        return False
    if not ast.unparse(decorator.func).endswith("parametrize"):
        return False
    first, second = decorator.args[0], decorator.args[1]
    return (
        isinstance(first, ast.Constant)
        and first.value == BOOT_SMOKE_PRESET_ARG
        and isinstance(second, ast.Name)
        and second.id == BOOT_SMOKE_PRESETS_CONSTANT
    )


def _boot_module_tests(source: str) -> tuple[list[str], list[str]]:
    """Every test in the boot module, and those a ``-k`` cell cannot reach.

    Class bodies are descended into as well as the module's own. A case written as a
    method is a case, and a ``parametrize`` on the class covers every method under it,
    so a scan reading only module-level ``def``s would report a clean sweep over a
    file whose cases it never looked at.
    """
    names: list[str] = []
    stray: list[str] = []

    def collect(body: list[ast.stmt], parametrised: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                inherited = parametrised or any(
                    _parametrises_over_presets(d) for d in node.decorator_list
                )
                collect(node.body, inherited)
                continue
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            names.append(node.name)
            own = any(_parametrises_over_presets(d) for d in node.decorator_list)
            if not parametrised and not own:
                stray.append(node.name)

    collect(ast.parse(source).body, False)
    return names, stray


def test_boot_smoke_matrix_runs_every_preset_the_module_parametrises(
    workflow: dict[str, Any],
) -> None:
    """The matrix and the module's list have to be equal in both directions, and
    each direction fails differently. A preset the module parametrises and the
    matrix omits is a set of tests selected by no cell, which passes silently; a
    preset the matrix names and the module does not selects nothing, which reds
    the lane with an empty run."""
    matrix = _jobs(workflow)[BOOT_SMOKE_JOB]["strategy"]["matrix"]["preset"]
    module = _boot_module_presets(_boot_module_source())
    assert sorted(matrix) == sorted(module), (
        f"the '{BOOT_SMOKE_JOB}' matrix in .github/workflows/ci.yml names "
        f"{sorted(matrix)} while {BOOT_SMOKE_TEST_FILE} parametrises {sorted(module)} — "
        f"a cell for every preset, and no cell without one"
    )


def test_boot_smoke_matrix_runs_every_preset__mutation_drops_a_matrix_preset() -> None:
    """A preset the module still parametrises but the matrix no longer names is
    the silent half: those tests are selected by no cell and nothing reds."""
    mutated = copy.deepcopy(_load_workflow())
    presets = _jobs(mutated)[BOOT_SMOKE_JOB]["strategy"]["matrix"]["preset"]
    presets.remove("control-assistant")
    with pytest.raises(AssertionError):
        test_boot_smoke_matrix_runs_every_preset_the_module_parametrises(mutated)


def test_boot_smoke_matrix_runs_every_preset__mutation_adds_a_preset_no_test_names() -> None:
    """And the other direction: a cell whose ``-k`` matches nothing at all."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[BOOT_SMOKE_JOB]["strategy"]["matrix"]["preset"].append("not-a-preset")
    with pytest.raises(AssertionError):
        test_boot_smoke_matrix_runs_every_preset_the_module_parametrises(mutated)


def test_boot_smoke_step_selects_the_matrix_preset(workflow: dict[str, Any]) -> None:
    """The chain from the matrix value to the selector is what makes the two
    lists comparable at all, and it is three links long — the matrix entry, the
    step ``env`` binding it, and the ``-k`` reading it. Each one is silently
    droppable, so each one is pinned."""
    step = _find_named_step(workflow, BOOT_SMOKE_JOB, BOOT_SMOKE_STEP)
    assert BOOT_SMOKE_TEST_FILE in step["run"], (
        f"the '{BOOT_SMOKE_STEP}' step must name {BOOT_SMOKE_TEST_FILE} by path; got: "
        f"{step['run'].strip()!r}"
    )
    assert BOOT_SMOKE_SELECTOR in step["run"], (
        f"the '{BOOT_SMOKE_STEP}' step must select a single preset with {BOOT_SMOKE_SELECTOR} — "
        f"without it every cell runs the whole file and the matrix buys nothing; got: "
        f"{step['run'].strip()!r}"
    )
    assert step.get("env", {}).get(BOOT_SMOKE_PRESET_ENV) == "${{ matrix.preset }}", (
        f"the '{BOOT_SMOKE_STEP}' step must bind {BOOT_SMOKE_PRESET_ENV} to "
        f"${{{{ matrix.preset }}}}; got "
        f"{step.get('env', {}).get(BOOT_SMOKE_PRESET_ENV)!r}"
    )


def test_boot_smoke_step_selects_the_matrix_preset__mutation_drops_the_selector() -> None:
    """The dangerous half: with no ``-k`` the lane runs the whole file in every
    cell, pays the build twice over and stays green while doing it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BOOT_SMOKE_JOB, BOOT_SMOKE_STEP)
    original = step["run"]
    step["run"] = original.replace(BOOT_SMOKE_SELECTOR, "")
    assert step["run"] != original, "no selector in the step — mutation is stale"
    with pytest.raises(AssertionError):
        test_boot_smoke_step_selects_the_matrix_preset(mutated)


def test_boot_smoke_step_selects_the_matrix_preset__mutation_unbinds_the_preset_env() -> None:
    """A selector reading a shell variable the step never bound to the matrix
    value selects the same thing in all cells."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BOOT_SMOKE_JOB, BOOT_SMOKE_STEP)
    step["env"][BOOT_SMOKE_PRESET_ENV] = "hello-world"
    with pytest.raises(AssertionError):
        test_boot_smoke_step_selects_the_matrix_preset(mutated)


def test_every_boot_smoke_test_runs_in_exactly_one_matrix_cell() -> None:
    """The selector matches on test id, so a test that is not parametrised over
    the preset list carries no preset in its id and is selected by no cell of
    the matrix."""
    names, stray = _boot_module_tests(_boot_module_source())
    assert names, (
        f"{BOOT_SMOKE_TEST_FILE} defines no test functions any more; either the module "
        f"left or this scan broke"
    )
    assert stray == [], (
        f"test(s) in {BOOT_SMOKE_TEST_FILE} are not parametrised over "
        f"{BOOT_SMOKE_PRESETS_CONSTANT}: {stray}. The lane selects a single preset per cell "
        f"with {BOOT_SMOKE_SELECTOR}, so a test with no preset in its id runs in no cell."
    )


def test_every_boot_smoke_test_runs_in_one_cell__mutation_forgets_the_parametrize() -> None:
    """The concrete case: a new case lands in the module, passes locally, and is
    selected by nothing in CI because nobody noticed the missing decorator."""
    source = _boot_module_source()
    names, _ = _boot_module_tests(source)
    mutated = source + "\n\ndef test_a_preset_free_case() -> None:\n    pass\n"
    mutated_names, mutated_stray = _boot_module_tests(mutated)
    assert mutated_stray == ["test_a_preset_free_case"]
    assert len(mutated_names) == len(names) + 1


def test_every_boot_smoke_test_runs_in_one_cell__mutation_hides_the_case_in_a_class() -> None:
    """The same case one indent deeper. A method carries no preset in its id for
    exactly the same reason a function does, so the scan has to see it."""
    source = _boot_module_source()
    names, _ = _boot_module_tests(source)
    mutated = source + (
        "\n\nclass TestPresetFree:\n"
        "    def test_a_preset_free_method(self) -> None:\n"
        "        pass\n"
    )
    mutated_names, mutated_stray = _boot_module_tests(mutated)
    assert mutated_stray == ["test_a_preset_free_method"]
    assert len(mutated_names) == len(names) + 1


# ---------------------------------------------------------------------------
# A file the unit lane ignores by name runs somewhere that is not gated
# ---------------------------------------------------------------------------

#: Files the unit lane drops from its sweep that are NOT subject to the rule
#: below, each with the reason. A file named here runs somewhere that does not
#: run on every event the unit lane does, which is the deliberate trade in each
#: case:
#:
#:  * every live module scripts/va/live_ca/gate.py runs — its Channel Access
#:    suites, and in --pva mode its PVAccess suites too — runs one pytest
#:    process per module through that gate, in a step of the unit job itself
#:    that is gated on a single matrix cell. libca is process-global and not
#:    thread-safe, so a shared xdist worker is precisely what they cannot have,
#:    and the gate — not a path in a run step — is what proves they ran.
UNIT_LANE_IGNORE_EXEMPTIONS = frozenset(
    {
        "tests/va/test_apply_fault.py",
        "tests/va/test_lume_pva_seam.py",
    }
)

#: An `--ignore=` argument on a pytest invocation.
_IGNORE_ARG_RE = re.compile(r"--ignore=(\S+)")


def _unit_lane_ignored_files(wf: dict[str, Any]) -> list[str]:
    """Files the unit lane's own pytest line drops by name, minus the exemptions.

    Only paths ending in ``.py`` are subject to the rule. A directory ignore is a
    different statement — ``tests/e2e`` names a suite with a whole set of lanes of its
    own and a conftest that refuses to run outside them — and it is not a file that
    could be hosted by one job.
    """
    return sorted(
        path
        for path in _IGNORE_ARG_RE.findall(_unit_test_pytest_line(wf))
        if path.endswith(".py") and path not in UNIT_LANE_IGNORE_EXEMPTIONS
    )


def _unconditional_hosts(wf: dict[str, Any], path: str) -> list[str]:
    """Jobs that run *path* by name with nothing gating the job or the step.

    Both gates are read, because either alone reinstates the hole the ignore would
    otherwise open: a job with an ``if:`` runs on a subset of the events the unit lane
    runs on, and so does an unconditional job whose step is gated on a matrix cell.

    ``--ignore=`` arguments are stripped from a step's text before the search. A lane
    that declines to run a file names it exactly as literally as one that runs it, and
    reading the first as coverage would let any file be hosted by whichever lane
    dropped it.
    """
    hosts = []
    for name, job in _jobs(wf).items():
        if name == UNIT_TEST_JOB or "if" in job:
            continue
        for step in job.get("steps", []):
            if "if" in step:
                continue
            if path in _IGNORE_ARG_RE.sub("", str(step.get("run", ""))):
                hosts.append(name)
                break
    return sorted(hosts)


def test_every_file_the_unit_lane_ignores_runs_in_an_unconditional_job(
    workflow: dict[str, Any],
) -> None:
    """An ignore is the one way to stop running a test that reports nothing and
    reds nothing: no skip, no summary line, every check still green. So each
    one has to name a file that some unconditional job runs by name. The
    exemptions above are the entries for which that is deliberately untrue, and
    they are listed with their reasons rather than skipped."""
    ignored = _unit_lane_ignored_files(workflow)
    assert ignored, (
        f"the '{UNIT_TEST_JOB}' lane's pytest line names no file outside "
        f"{sorted(UNIT_LANE_IGNORE_EXEMPTIONS)} — either this scan or the shape of that "
        f"line has gone stale, and the rule below is holding nothing"
    )
    unhosted = [path for path in ignored if not _unconditional_hosts(workflow, path)]
    assert unhosted == [], (
        f"the '{UNIT_TEST_JOB}' lane ignores {unhosted} and no unconditional job runs "
        f"them by name, so they run on fewer events than the lane does — or on none. "
        f"Give each one a job that carries no `if:` and a step that carries none "
        f"either, or add it to UNIT_LANE_IGNORE_EXEMPTIONS beside its reason."
    )


def test_every_ignored_file_has_a_host__mutation_drops_the_host_step() -> None:
    """A file whose only unconditional host step is deleted is ignored by the
    lane and run by nothing at all."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[BOOT_SMOKE_JOB]["steps"]
    _jobs(mutated)[BOOT_SMOKE_JOB]["steps"] = [
        step for step in steps if step.get("name") != BOOT_SMOKE_STEP
    ]
    assert len(_jobs(mutated)[BOOT_SMOKE_JOB]["steps"]) == len(steps) - 1, (
        f"no step named {BOOT_SMOKE_STEP!r} in {BOOT_SMOKE_JOB} — mutation is stale"
    )
    assert _unconditional_hosts(mutated, "tests/integration/test_preset_static.py"), (
        "the other ignored module must keep its host, or the mutation proves nothing "
        "about the one it removed"
    )
    with pytest.raises(AssertionError):
        test_every_file_the_unit_lane_ignores_runs_in_an_unconditional_job(mutated)


def test_every_ignored_file_has_a_host__mutation_gates_the_host_job() -> None:
    """The dangerous half: the step still names the file, so the file still
    appears to have a home — it just stopped being read on most events."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[PARSE_ONLY_JOB]["if"] = "github.event_name == 'pull_request'"
    hosting = [
        step
        for step in _jobs(mutated)[PARSE_ONLY_JOB]["steps"]
        if "tests/integration/test_preset_static.py" in str(step.get("run", ""))
    ]
    assert len(hosting) == 1 and "if" not in hosting[0], (
        f"expected one ungated step naming the file in {PARSE_ONLY_JOB}; got {hosting}"
    )
    with pytest.raises(AssertionError):
        test_every_file_the_unit_lane_ignores_runs_in_an_unconditional_job(mutated)


def test_every_ignored_file_has_a_host__mutation_gates_the_host_step() -> None:
    """A gate on the step is the same hole as a gate on the job: the file is
    read on a subset of the events the unit lane would have read it on."""
    mutated = copy.deepcopy(_load_workflow())
    assert "if" not in _jobs(mutated)[BOOT_SMOKE_JOB], (
        f"{BOOT_SMOKE_JOB} already carries an `if:` — mutation is stale"
    )
    _find_named_step(mutated, BOOT_SMOKE_JOB, BOOT_SMOKE_STEP)["if"] = (
        "matrix.preset == 'hello-world'"
    )
    with pytest.raises(AssertionError):
        test_every_file_the_unit_lane_ignores_runs_in_an_unconditional_job(mutated)


def test_every_ignored_file_has_a_host__mutation_a_host_that_only_ignores_the_file() -> None:
    """The shared e2e lane spells one of its files in an ``--ignore=``, so a
    substring read of that step would report the file as covered by the very
    lane that specifically declines to run it.

    That lane's own job gate is lifted first, and the step naming the file is
    checked to carry none either, so what keeps the lane out of the hosts is the
    stripping under test and not a gate that would have excluded it anyway.
    """
    mutated = copy.deepcopy(_load_workflow())
    assert TEAMS_TEST_FILE in json.dumps(_jobs(mutated)[E2E_TESTS_JOB]), (
        f"{E2E_TESTS_JOB} no longer names {TEAMS_TEST_FILE} at all; this probe is stale"
    )
    del _jobs(mutated)[E2E_TESTS_JOB]["if"]
    naming = [
        step
        for step in _jobs(mutated)[E2E_TESTS_JOB]["steps"]
        if TEAMS_TEST_FILE in str(step.get("run", ""))
    ]
    assert naming and all("if" not in step for step in naming), (
        f"expected an ungated step naming {TEAMS_TEST_FILE} in {E2E_TESTS_JOB}, or the "
        f"probe passes on the gate rather than on the stripping; got {naming}"
    )
    assert E2E_TESTS_JOB not in _unconditional_hosts(mutated, TEAMS_TEST_FILE)


def test_every_ignored_file_has_a_host__mutation_an_unhosted_ignore_is_reported() -> None:
    """A new ignore with no host anywhere must come back named, not swallowed
    by the entries that do have one."""
    mutated = copy.deepcopy(_load_workflow())
    orphan = "tests/integration/test_nothing_else_runs_this.py"
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    line = _unit_test_pytest_line(mutated)
    step["run"] = step["run"].replace(line, f"{line} --ignore={orphan}")
    ignored = _unit_lane_ignored_files(mutated)
    assert orphan in ignored, "the appended ignore did not reach the scan"
    unhosted = [path for path in ignored if not _unconditional_hosts(mutated, path)]
    assert unhosted == [orphan]


#: The module that builds every shipped preset.
PRESET_BUILD_TEST_FILE = "tests/integration/test_preset_build.py"


def test_the_preset_build_runs_in_the_static_job_and_not_in_the_unit_lane(
    workflow: dict[str, Any],
) -> None:
    """Every preset is built once per event, by the Tier 0 static job, rather
    than once per cell of the unit lane's matrix."""
    assert PRESET_BUILD_TEST_FILE in _unit_lane_ignored_files(workflow)
    assert _unconditional_hosts(workflow, PRESET_BUILD_TEST_FILE) == [PARSE_ONLY_JOB]


# ---------------------------------------------------------------------------
# Coverage is measured on a cell where PEP 669 tracing is actually available
# ---------------------------------------------------------------------------


#: Below this, ``coverage.py`` cannot use ``sys.monitoring`` and silently falls
#: back to the C trace function. See ``coverage/core.py``: ``pep669`` is a bare
#: ``hasattr(sys, "monitoring")`` check, so the fallback is a warning, not an
#: error, and the lane just pays ~2x for identical numbers.
SYSMON_MIN_PYTHON = (3, 12)

#: The env var that opts into PEP 669 tracing. Not self-enabling until 3.14
#: (``coverage/env.py``: ``SYSMON_DEFAULT``), so on 3.12/3.13 it must be set
#: explicitly or the slow core is chosen by default.
SYSMON_ENV = ("COVERAGE_CORE", "sysmon")

CODECOV_STEP = "Upload coverage reports to Codecov"


def _coverage_cell_version(wf: dict[str, Any]) -> str:
    """The Python version whose ubuntu cell builds a non-empty ``COV_ARGS``."""
    run_text = _find_named_step(wf, UNIT_TEST_JOB, "Run unit tests")["run"]
    found = re.findall(
        r'\[\s+"\$\{\{\s*matrix\.python-version\s*\}\}"\s*=\s*"(\d+\.\d+)"', run_text
    )
    assert len(found) == 1, f"expected one coverage-cell guard in the run block; got {found}"
    return found[0]


def _codecov_upload_version(wf: dict[str, Any]) -> str:
    """The Python version the Codecov upload step is gated on."""
    condition = _find_named_step(wf, UNIT_TEST_JOB, CODECOV_STEP)["if"]
    found = re.findall(r"matrix\.python-version\s*==\s*'(\d+\.\d+)'", condition)
    assert len(found) == 1, f"expected one version gate on {CODECOV_STEP!r}; got {found}"
    return found[0]


def test_coverage_cell_and_codecov_upload_name_the_same_version(
    workflow: dict[str, Any],
) -> None:
    """One cell writes ``coverage.xml``; one cell uploads it. If they drift
    apart the upload runs on a cell that produced no file, and
    ``fail_ci_if_error: false`` turns that into a silent coverage blackout --
    green lane, no report, nobody notified. The two conditions are written out
    separately in YAML and cannot reference each other, so this is the only
    thing holding them together."""
    measured = _coverage_cell_version(workflow)
    uploaded = _codecov_upload_version(workflow)
    assert measured == uploaded, (
        f"the coverage cell measures on Python {measured} but the {CODECOV_STEP!r} step "
        f"is gated on {uploaded} -- coverage.xml would be written by one cell and looked "
        f"for by another. Move both, in .github/workflows/ci.yml."
    )


def test_coverage_cell_can_use_pep669_tracing(workflow: dict[str, Any]) -> None:
    """The coverage cell must sit on an interpreter where ``sys.monitoring``
    exists. Measuring on the floor version looks tidy and costs ~2x the lane's
    wall clock: coverage refuses sysmon below 3.12 with a warning and uses the
    C tracer instead, which is how this lane came to run 36 minutes next to a
    14-minute sibling running the same tests."""
    measured = _coverage_cell_version(workflow)
    parsed = tuple(int(part) for part in measured.split("."))
    assert parsed >= SYSMON_MIN_PYTHON, (
        f"coverage is measured on Python {measured}, below the "
        f"{'.'.join(map(str, SYSMON_MIN_PYTHON))} that `sys.monitoring` requires -- the "
        f"lane is paying the C tracer's overhead for identical numbers."
    )


def test_coverage_cell_arms_the_sysmon_core(workflow: dict[str, Any]) -> None:
    """Being eligible for PEP 669 is not the same as using it: coverage only
    defaults to sysmon at 3.14+, so on 3.12/3.13 the env var is what actually
    selects it. Dropping it re-buys the 2x without changing a single flag."""
    env = _find_named_step(workflow, UNIT_TEST_JOB, "Run unit tests").get("env", {})
    name, value = SYSMON_ENV
    assert env.get(name) == value, (
        f"the 'Run unit tests' step must set {name}: {value} -- without it coverage.py "
        f"picks the C tracer by default below Python 3.14. Found {env.get(name)!r}."
    )


def test_coverage_cell_is_a_version_the_matrix_actually_runs(workflow: dict[str, Any]) -> None:
    """A coverage cell naming a version absent from the matrix measures
    nothing at all, and would fail none of the pins above."""
    versions = workflow["jobs"][UNIT_TEST_JOB]["strategy"]["matrix"]["python-version"]
    measured = _coverage_cell_version(workflow)
    assert measured in versions, (
        f"coverage is measured on Python {measured}, which the matrix does not run "
        f"({versions}) -- no cell would produce coverage.xml."
    )


def test_coverage_cell_pins__mutation_versions_drift_apart() -> None:
    """The failure this set exists for: someone moves one condition and not
    the other."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, CODECOV_STEP)
    original = step["if"]
    step["if"] = original.replace("'3.12'", "'3.11'")
    assert step["if"] != original, "Codecov version gate not found -- mutation is stale"
    assert _coverage_cell_version(mutated) != _codecov_upload_version(mutated)


def test_coverage_cell_pins__mutation_back_to_the_floor_version() -> None:
    """Reverting the cell to 3.11 must fail the PEP 669 pin rather than pass
    quietly -- the whole point is that the slow path is invisible at runtime."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace('}}" = "3.12" ]', '}}" = "3.11" ]')
    assert step["run"] != original, "coverage-cell guard not found -- mutation is stale"
    assert _coverage_cell_version(mutated) == "3.11"


def _conftest_source() -> str:
    return SCHEDULER_SOURCE.read_text(encoding="utf-8")


def _hook_source(source: str) -> str:
    """Source of the top-level ``pytest_xdist_make_scheduler`` definition, or
    ``""`` if it is not defined.

    Parsed with ``ast`` rather than grepped because the explanatory comment
    above the hook also spells "loadgroup": a whole-file substring search
    would stay green after the guard itself was deleted.
    """
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == SCHEDULER_HOOK:
            return ast.get_source_segment(source, node) or ""
    return ""


def _missing_scheduler_parts(source: str) -> list[str]:
    """Which parts of the scheduler override are absent (empty = all present)."""
    hook = _hook_source(source)
    present = {
        f"def {SCHEDULER_HOOK}": bool(hook),
        f"class {SCHEDULER_CLASS}": f"class {SCHEDULER_CLASS}(" in source,
        "loadgroup guard": 'getoption("dist"' in hook and "loadgroup" in hook,
        f"returns {SCHEDULER_CLASS}": f"{SCHEDULER_CLASS}(config, log)" in hook,
    }
    return [label for label, ok in present.items() if not ok]


def test_conftest_defines_the_loadgroup_scheduler_override() -> None:
    """Source-level counterpart to the ci.yml pin above. `--dist loadgroup` is
    only safe for this suite because the conftests replace stock
    ``LoadGroupScheduling`` — which gives every unmarked nodeid its own scope
    — with one that falls back to file scope. Deleting the override while
    leaving the CI flag in place would parallelize the lane per-test again,
    silently, and the failures would land as intermittent reds in unrelated
    PRs."""
    assert _missing_scheduler_parts(_conftest_source()) == []


def test_both_conftests_import_the_scheduler_override() -> None:
    """The override is defined in the harness, and pytest calls a hook only from
    a conftest or a registered plugin. A conftest that stopped importing it
    would hand its tree back to stock scheduling while the definition above
    still passed."""
    for conftest in SCHEDULER_CONFTESTS:
        imported = {
            alias.name
            for node in ast.parse(conftest.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ImportFrom) and node.module == SCHEDULER_MODULE
            for alias in node.names
        }
        assert SCHEDULER_HOOK in imported, f"{conftest} does not import {SCHEDULER_HOOK}"


def test_conftest_scheduler_override__mutation_renames_hook() -> None:
    """A hook pytest never calls is the same as no hook at all: renaming the
    definition must be reported, not papered over by the class still being
    defined below it."""
    source = _conftest_source()
    mutated = source.replace(f"def {SCHEDULER_HOOK}(", f"def _disabled_{SCHEDULER_HOOK}(")
    assert mutated != source, (
        f"no `def {SCHEDULER_HOOK}(` in {SCHEDULER_SOURCE} — mutation is stale"
    )
    assert _missing_scheduler_parts(mutated) == [
        f"def {SCHEDULER_HOOK}",
        "loadgroup guard",
        f"returns {SCHEDULER_CLASS}",
    ]


def test_conftest_scheduler_override__mutation_drops_loadgroup_guard() -> None:
    """The guard returns None for every dist mode but ``loadgroup``, which is
    what keeps the e2e lane's ``--dist loadfile`` (and ad-hoc ``load`` /
    ``worksteal`` runs) on stock xdist. Deleting it would silently widen the
    override's reach beyond the lane it was written for."""
    source = _conftest_source()
    mutated = "\n".join(line for line in source.splitlines() if 'getoption("dist"' not in line)
    assert mutated != source, f"no dist-option guard in {SCHEDULER_SOURCE} — mutation is stale"
    assert _missing_scheduler_parts(mutated) == ["loadgroup guard"]


# ---------------------------------------------------------------------------
# Lane-leaving markers: out of the parallel lane, into a serial step
# ---------------------------------------------------------------------------

# Some tests read the machine as much as the code. ``tests/pty/`` runs osprey's
# own verbs on a real terminal and asserts that a spinner ADVANCED between two
# frames; a ``wallclock`` test asserts an absolute time budget. Beside three
# loaded xdist workers both measure the workers, so each marker is handed out
# of the four-worker lane and given a serial step of its own.
#
# Both halves of that handover are silently droppable, in opposite directions,
# which is why each is pinned here for every row of LANE_LEAVING_MARKERS. Lose
# the deselection and the tests go back to running beside loaded workers, where
# they do not fail now but flake in someone else's PR weeks later. Lose the step
# and they run nowhere at all, with every check still green: the same "an
# unnamed suite never runs" shape the rest of this module exists to prevent.

_ROW_IDS = [row.marker for row in LANE_LEAVING_MARKERS]
_HOSTED = [(row, host) for row in LANE_LEAVING_MARKERS for host in row.hosts]
_HOSTED_IDS = [f"{row.marker}-{host.job}" for row, host in _HOSTED]
_HOSTS = [host for _, host in _HOSTED]

#: A pytest invocation in a ci.yml run block, in both spellings the file uses.
_CI_PYTEST_RE = re.compile(r"uv run (?:--no-sync )?pytest\b")
#: Where one shell command ends and the next (or a pipe) begins.
_SHELL_SEPARATOR_RE = re.compile(r"\|\||&&|[|;]")


def _ci_pytest_invocations(wf: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every pytest invocation in ci.yml as ``(job, step, command)``.

    Everything from a ``#`` onward is dropped first, so a flag parked in a shell
    comment never counts (the same safe-direction split
    ``_unit_test_pytest_line`` makes). Backslash continuations are joined, so an
    invocation whose paths sit on the following lines is read whole. A step may
    hold more than one invocation; each is returned, cut at the next shell
    separator.
    """
    found = []
    for job, spec in _jobs(wf).items():
        for step in spec.get("steps") or []:
            run = step.get("run")
            if not run:
                continue
            text = "\n".join(line.split("#", 1)[0] for line in str(run).splitlines())
            text = re.sub(r"\\[ \t]*\n", " ", text)
            for line in text.splitlines():
                for match in _CI_PYTEST_RE.finditer(line):
                    command = _SHELL_SEPARATOR_RE.split(line[match.start() :], maxsplit=1)[0]
                    found.append((job, str(step.get("name", "")), command.strip()))
    return found


def _invocation_reaches(command: str, test_file: str) -> bool:
    """Whether *command* collects *test_file*.

    A path argument covers the file when the file is it or lies under it; a
    ``--ignore=`` argument takes it back out. An invocation whose only path is
    a bare variable (``$E2E_FILTER``) is an operator-chosen debug path on a
    manual dispatch, not a lane, and reaches nothing here; an invocation with
    no path at all collects ``testpaths``, which is ``tests`` and the
    packages' own suites (:data:`PACKAGE_TEST_ROOTS`).
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    paths = [t for t in tokens if t == "tests" or t.startswith(("tests/", "packages/"))]
    ignores = [t.removeprefix("--ignore=") for t in tokens if t.startswith("--ignore=")]
    if not paths:
        if any(t.startswith("$") for t in tokens):
            return False
        paths = ["tests", *PACKAGE_TEST_ROOTS]

    def _covers(path: str) -> bool:
        path = path.split("::", 1)[0].rstrip("/")
        return test_file == path or test_file.startswith(f"{path}/")

    return any(_covers(p) for p in paths) and not any(_covers(i) for i in ignores)


def _carriers(row: LaneLeavingMarker, sources: dict[str, str]) -> list[str]:
    """Test files whose source carries the row's marker. Text scan, not a
    collection run: this module runs in the parallel lane, and importing the
    marked modules would need their optional dependencies."""
    pattern = row.mark_re()
    return sorted(name for name, source in sources.items() if pattern.search(source))


def _unhosted_carriers(row: LaneLeavingMarker, sources: dict[str, str]) -> list[str]:
    """Marked files no host step's path reaches (empty = all reachable)."""
    return [
        name
        for name in _carriers(row, sources)
        if not any(name.startswith(host.path) for host in row.hosts)
    ]


def _sole_marker(command: str) -> str | None:
    """The one marker an invocation's ``-m`` expression names, when it names
    nothing else; such an invocation collects only that marker's tests."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for flag, expression in itertools.pairwise(tokens):
        if flag == "-m":
            return expression if re.fullmatch(r"\w+", expression) else None
    return None


def _undeselected_reaches(
    wf: dict[str, Any], row: LaneLeavingMarker, sources: dict[str, str]
) -> list[str]:
    """Non-host invocations that reach a marked file without deselecting the
    marker (empty = the row's tests run in its hosts and nowhere else).

    An invocation that selects one other marker alone reaches a marked file
    only when the file carries that marker too."""
    hosts = {(host.job, host.step) for host in row.hosts}

    def _selects(command: str, name: str) -> bool:
        sole = _sole_marker(command)
        if sole is None or sole == row.marker:
            return True
        return re.search(rf"pytest\.mark\.{re.escape(sole)}\b", sources[name]) is not None

    carriers = _carriers(row, sources)
    return [
        f"{job} · {step}: {command}"
        for job, step, command in _ci_pytest_invocations(wf)
        if (job, step) not in hosts
        and any(_invocation_reaches(command, f) and _selects(command, f) for f in carriers)
        and row.deselect_term() not in command
    ]


def _host_invocations(wf: dict[str, Any], host: SerialHost) -> list[str]:
    _find_named_step(wf, host.job, host.step)
    return [
        command
        for job, step, command in _ci_pytest_invocations(wf)
        if (job, step) == (host.job, host.step)
    ]


def _per_pr_hosts(wf: dict[str, Any], row: LaneLeavingMarker) -> tuple[SerialHost, ...]:
    """The row's hosts whose job gates the merge (``all-checks-passed.needs``).

    A host outside the merge gate runs on demand, so nothing a pull request
    triggers times its files; a local mirror owes exactly the files under the
    hosts this returns."""
    needs = set(_jobs(wf)[GATE_JOB]["needs"])
    return tuple(host for host in row.hosts if host.job in needs)


def _registered_markers(pyproject: dict[str, Any]) -> list[str]:
    """The marker NAMES declared in ``[tool.pytest.ini_options]``, without the
    ``name: description`` prose after the first colon."""
    declared = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    return [entry.split(":", 1)[0].strip() for entry in declared]


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_lane_leaving_marker_is_registered(
    row: LaneLeavingMarker, pyproject: dict[str, Any]
) -> None:
    """ci.yml and the scripts name the marker as a bare string, and pyproject
    is the only place that says what it means. Unregistered, the marked files
    collect with an unknown-mark warning under the lane's own invocation and
    error outright under ``--strict-markers``, while the deselection and the
    selection go on matching nothing but each other."""
    assert row.marker in _registered_markers(pyproject), (
        f"register the '{row.marker}' marker in [tool.pytest.ini_options].markers; "
        f"ci.yml selects and deselects it by name"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_lane_leaving_marker_is_registered__mutation_drops_it(row: LaneLeavingMarker) -> None:
    mutated = _load_pyproject()
    ini = mutated["tool"]["pytest"]["ini_options"]
    original = ini["markers"]
    ini["markers"] = [e for e in original if e.split(":", 1)[0].strip() != row.marker]
    assert ini["markers"] != original, "marker not registered; mutation is stale"
    with pytest.raises(AssertionError):
        test_lane_leaving_marker_is_registered(row, mutated)


@functools.cache
def _test_tree_paths() -> tuple[Path, ...]:
    """Every ``tests/**/test_*.py`` path, walked once per process."""
    return tuple((CI_YML.parents[2] / "tests").rglob("test_*.py"))


@functools.cache
def _test_tree_sources() -> dict[Path, str]:
    """The source of every path in ``_test_tree_paths``, read once per process.

    Callers must not mutate the returned dict; ``_test_module_sources`` hands
    out a fresh one.
    """
    return {path: path.read_text(encoding="utf-8") for path in _test_tree_paths()}


def _e2e_test_sources() -> dict[Path, str]:
    """The ``tests/e2e/**/test_*.py`` subset of ``_test_tree_sources``."""
    e2e_dir = CI_YML.parents[2] / "tests" / "e2e"
    return {path: src for path, src in _test_tree_sources().items() if path.is_relative_to(e2e_dir)}


def _test_module_sources() -> dict[str, str]:
    """Every test module under ``tests/``, keyed by repo-relative posix path.

    This module excludes itself: it is the scanner, and its own source spells
    the markers it searches for: in the table above, in these docstrings, and
    in the synthetic files the mutations below invent. A scanner that found
    itself would report a permanent false positive.
    """
    root = CI_YML.parents[2]
    me = Path(__file__).resolve()
    return {
        path.relative_to(root).as_posix(): source
        for path, source in _test_tree_sources().items()
        if path.resolve() != me
    }


def _check_carried(row: LaneLeavingMarker, sources: dict[str, str]) -> None:
    assert _carriers(row, sources), (
        f"no test file carries {row.marker!r} any more. Either its last test was "
        f"removed (drop the row and its serial step with it) or the marker scan broke"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_lane_leaving_marker_is_carried(row: LaneLeavingMarker) -> None:
    """A row whose marker no test carries is wiring for nothing: a serial step
    that selects zero tests and a deselection that guards no file."""
    _check_carried(row, _test_module_sources())


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_lane_leaving_marker_is_carried__mutation_drops_the_carriers(
    row: LaneLeavingMarker,
) -> None:
    sources = _test_module_sources()
    carriers = set(_carriers(row, sources))
    assert carriers, f"no file carries {row.marker!r}; mutation is stale"
    mutated = {name: src for name, src in sources.items() if name not in carriers}
    with pytest.raises(AssertionError):
        _check_carried(row, mutated)


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_every_marked_file_lives_where_a_serial_step_looks(row: LaneLeavingMarker) -> None:
    """The deselection is repo-wide but each host selects one directory, so a
    marked file written anywhere else is dropped by the lane and picked up by
    nothing. It would not fail, it would not skip, and it would not appear in
    any summary: it would simply stop being run, which is the one outcome this
    module exists to make impossible."""
    stray = _unhosted_carriers(row, _test_module_sources())
    paths = [host.path for host in row.hosts]
    assert stray == [], (
        f"{row.marker}-marked file(s) outside {paths}: {stray}. The parallel lane "
        f"deselects the marker everywhere, so these run nowhere. Move them under a "
        f"host path or add a serial step to the row in LANE_LEAVING_MARKERS."
    )


#: The rows whose every host path is narrower than the tree. A row hosted over
#: the whole tree has no stray under tests/ by construction.
_NARROW_ROWS = [row for row in LANE_LEAVING_MARKERS if all(h.path != "tests/" for h in row.hosts)]


@pytest.mark.parametrize("row", _NARROW_ROWS, ids=[row.marker for row in _NARROW_ROWS])
def test_every_marked_file_lives_where_a_serial_step_looks__mutation_strays(
    row: LaneLeavingMarker,
) -> None:
    """A future marked file one directory over must be reported by name, not
    tolerated because the tests it was modelled on are still in place. A row
    hosted over the whole tree has no stray under tests/ by construction."""
    sources = _test_module_sources()
    phantom = f"tests/cli/test_future_{row.marker}_scenarios.py"
    sources[phantom] = f"pytestmark = [pytest.mark.{row.marker}]\n"
    assert _unhosted_carriers(row, sources) == [phantom]


def _check_host(wf: dict[str, Any], host: SerialHost) -> None:
    commands = _host_invocations(wf, host)
    assert commands, f"'{host.step}' must invoke pytest"
    host_lines = [run for run in commands if host.select in run]
    assert host_lines, (
        f"'{host.step}' must select the marker with {host.select!r}, so unmarked "
        f"files under {host.path} are not run a second time here; got: {commands!r}"
    )
    deselect = f"not {host.select.removeprefix('-m ')}"
    for run in commands:
        if run not in host_lines:
            assert deselect in run, (
                f"'{host.step}' holds a pytest line beside its host line that does not "
                f"carry {deselect!r}, so it times the marked tests among the others; "
                f"got: {run!r}"
            )
    for run in host_lines:
        assert host.path.rstrip("/") in run, f"'{host.step}' must select {host.path}; got: {run!r}"
        assert not _XDIST_WORKERS_RE.search(run), (
            f"'{host.step}' runs pytest with xdist workers; the marked tests read the "
            f"machine and must not compete with each other"
        )
        assert "--dist" not in run, f"'{host.step}' sets an xdist dist mode; it must stay serial"


@pytest.mark.parametrize(("row", "host"), _HOSTED, ids=_HOSTED_IDS)
def test_marker_runs_in_its_own_serial_step(
    row: LaneLeavingMarker, host: SerialHost, workflow: dict[str, Any]
) -> None:
    """The host step must pick the marker back up, serially. No worker count
    and no dist mode: a second marked test running alongside is the load the
    deselection was for, and running it inside this step would be no better
    than leaving it in the lane."""
    assert host.select == row.select()
    _check_host(workflow, host)


@pytest.mark.parametrize("host", _HOSTS, ids=_HOSTED_IDS)
def test_marker_runs_in_its_own_serial_step__mutation_drops_the_step(host: SerialHost) -> None:
    """Deleting the step is the failure that costs nothing at the time: the
    deselection keeps the tests out of the lane, so CI goes green over tests
    nobody runs."""
    mutated = _load_workflow()
    steps = _jobs(mutated)[host.job]["steps"]
    steps.remove(_find_named_step(mutated, host.job, host.step))
    with pytest.raises(AssertionError):
        _check_host(mutated, host)


@pytest.mark.parametrize("host", _HOSTS, ids=_HOSTED_IDS)
def test_marker_runs_in_its_own_serial_step__mutation_adds_xdist_workers(host: SerialHost) -> None:
    """Speeding the step up with ``-n 2`` reinstates precisely the contention
    the tests were moved here to escape."""
    mutated = _load_workflow()
    step = _find_named_step(mutated, host.job, host.step)
    original = step["run"]
    step["run"] = original.replace(f" {host.select}", f" {host.select} -n 2", 1)
    assert step["run"] != original, "no marker select on the host step; mutation is stale"
    with pytest.raises(AssertionError, match="xdist workers"):
        _check_host(mutated, host)


@pytest.mark.parametrize("host", _HOSTS, ids=_HOSTED_IDS)
def test_marker_runs_in_its_own_serial_step__mutation_drops_the_select(host: SerialHost) -> None:
    """Selecting the bare directory instead would re-run the unmarked files
    under it, which already ran elsewhere."""
    mutated = _load_workflow()
    step = _find_named_step(mutated, host.job, host.step)
    original = step["run"]
    step["run"] = original.replace(f" {host.select}", "")
    assert step["run"] != original, "marker selection not found; mutation is stale"
    with pytest.raises(AssertionError):
        _check_host(mutated, host)


def test_marker_runs_in_its_own_serial_step__mutation_neighbour_line_drops_the_term() -> None:
    """A host step may run other tests on a line of its own; that line without
    the term times the budgets a second time, among the others."""
    row = _row("wallclock")
    host = next(h for h in row.hosts if h.job == UNIT_TEST_JOB)
    mutated = _load_workflow()
    step = _find_named_step(mutated, host.job, host.step)
    assert row.deselect_term() not in step["run"], "neighbour line already present; stale"
    step["run"] = f"{step['run'].rstrip()}\nuv run pytest tests/ --ignore=tests/e2e -v\n"
    _check_host(_load_workflow(), host)
    with pytest.raises(AssertionError, match="beside its host line"):
        _check_host(mutated, host)


def _check_nothing_else_times_it(
    wf: dict[str, Any], row: LaneLeavingMarker, sources: dict[str, str] | None = None
) -> None:
    leaks = _undeselected_reaches(wf, row, _test_module_sources() if sources is None else sources)
    assert leaks == [], (
        f"pytest invocation(s) reach a {row.marker}-marked file without "
        f"{row.deselect_term()!r}: {leaks}. Only the row's serial step may run it; "
        f"every other invocation that collects the file must deselect the marker"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_nothing_else_times_it(row: LaneLeavingMarker, workflow: dict[str, Any]) -> None:
    """Every pytest invocation in ci.yml that can collect a marked file is the
    row's host or deselects the marker. This is "every parallel lane hands it
    away", and also "the step beside the host does not run it a second time".
    An invocation that collects no marked file owes nothing."""
    _check_nothing_else_times_it(workflow, row)


def _unit_lane_with(wf: dict[str, Any], replacement: str) -> dict[str, Any]:
    step = _find_named_step(wf, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace(f" {UNIT_LANE_DESELECT}", replacement)
    assert step["run"] != original, "unit-lane deselection not found; mutation is stale"
    return wf


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_nothing_else_times_it__mutation_unit_lane_drops_the_term(row: LaneLeavingMarker) -> None:
    others = tuple(r for r in LANE_LEAVING_MARKERS if r is not row)
    replacement = f" {_lane_deselect(others)}" if others else ""
    mutated = _unit_lane_with(_load_workflow(), replacement)
    with pytest.raises(AssertionError):
        _check_nothing_else_times_it(mutated, row)


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_nothing_else_times_it__mutation_parks_it_in_a_comment(row: LaneLeavingMarker) -> None:
    """A deselection that survives only in a trailing shell comment, which is
    exactly how a temporary revert gets written, must not count."""
    mutated = _unit_lane_with(_load_workflow(), f"  # TODO restore {UNIT_LANE_DESELECT}")
    with pytest.raises(AssertionError):
        _check_nothing_else_times_it(mutated, row)


def test_nothing_else_times_it__mutation_sidecar_step_drops_the_term() -> None:
    """The step beside the budget's host collects the same directory by
    keyword; without the term it would time the budget a second time, among
    the module's other tests."""
    row = _row("wallclock")
    mutated = _load_workflow()
    step = _find_named_step(mutated, "qmd-sidecar-e2e", "Run qmd sidecar E2E")
    original = step["run"]
    step["run"] = original.replace(f" and {row.deselect_term()}", "")
    assert step["run"] != original, "sidecar-step deselection not found; mutation is stale"
    with pytest.raises(AssertionError):
        _check_nothing_else_times_it(mutated, row)


def test_nothing_else_times_it__mutation_a_file_carries_both_markers() -> None:
    """A host that selects its own marker alone still reaches a file that
    carries another row's marker as well, and runs those tests a second time."""
    sources = _test_module_sources()
    phantom = "tests/cli/test_future_pty_wallclock_scenarios.py"
    sources[phantom] = "pytestmark = [pytest.mark.pty, pytest.mark.wallclock]\n"
    _check_nothing_else_times_it(_load_workflow(), TERMINAL_ROW)
    with pytest.raises(AssertionError, match="Run the wall-clock budgets serially"):
        _check_nothing_else_times_it(_load_workflow(), TERMINAL_ROW, sources)


def test_pty_step_runs_only_where_a_terminal_is_real(workflow: dict[str, Any]) -> None:
    """Pinned to the Linux cells. The scenarios self-skip where a pty is not a
    thing, and a step whose whole suite skipped reports green over nothing, so
    the venue is chosen rather than discovered, by the same rule the live Channel
    Access step in this job already follows."""
    condition = _find_named_step(workflow, TERMINAL_HOST.job, TERMINAL_HOST.step).get("if", "")
    assert "matrix.os" in condition and "ubuntu-latest" in condition, (
        f"'{TERMINAL_HOST.step}' must be gated on the Linux matrix cells; got if: {condition!r}"
    )


def test_pty_step_runs_only_where_a_terminal_is_real__mutation_drops_the_gate() -> None:
    mutated = _load_workflow()
    del _find_named_step(mutated, TERMINAL_HOST.job, TERMINAL_HOST.step)["if"]
    with pytest.raises(AssertionError):
        test_pty_step_runs_only_where_a_terminal_is_real(mutated)


def _imports_stdlib_pty(source: str) -> bool:
    """Whether *source* imports the stdlib ``pty`` module, at any nesting.

    Parsed rather than grepped, for both directions. The file that does it today
    imports inside an ``if os.name == "posix":`` guard (an unguarded import
    raises at COLLECTION, which is before any ``skipif`` can apply), so a scan
    that only looked at top-level statements would miss it. And the letters
    "pty" are in every path, docstring and fixture name in that directory, so a
    substring search would match everything.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(alias.name == "pty" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "pty":
            return True
    return False


def _unmarked_terminal_files(sources: dict[str, str]) -> list[str]:
    """Files under the pty directory that open a terminal without the marker."""
    pattern = TERMINAL_ROW.mark_re()
    return sorted(
        name
        for name, source in sources.items()
        if name.startswith(TERMINAL_HOST.path)
        and _imports_stdlib_pty(source)
        and not pattern.search(source)
    )


def test_every_file_that_opens_a_terminal_carries_the_marker() -> None:
    """The other direction, and the one that fails quietly. A new file under
    ``tests/pty/`` that allocates a terminal but forgets the marker is not
    deselected, so it runs in the four-worker lane against three loaded
    siblings, and what it reports there is the runner rather than the code.

    Scoped to files that import the stdlib module rather than to the directory,
    because the exception is deliberate and must stay clean:
    ``test_stub_coverage.py`` lives here, needs no terminal, imports no ``pty``,
    and belongs in the parallel lane. ``tests/interfaces/web_terminal/`` also
    opens ptys and is correctly unmarked, which is why the scope is this
    directory and not the whole tree."""
    sources = _test_module_sources()
    terminal_files = [
        name
        for name, source in sources.items()
        if name.startswith(TERMINAL_HOST.path) and _imports_stdlib_pty(source)
    ]
    assert terminal_files, (
        f"no file under {TERMINAL_HOST.path} imports the stdlib pty module any more; "
        f"either the suite left or the import scan broke"
    )
    unmarked = _unmarked_terminal_files(sources)
    assert unmarked == [], (
        f"file(s) under {TERMINAL_HOST.path} open a terminal without the "
        f"{TERMINAL_ROW.marker!r} marker: {unmarked}. They would run in the parallel "
        f"lane, where a scenario that times its own repaints measures the runner. Add "
        f"`pytest.mark.{TERMINAL_ROW.marker}`."
    )


def test_every_file_that_opens_a_terminal_carries_the_marker__mutation_forgets_it() -> None:
    """The concrete case this is here for: a new scenario file lands under
    ``tests/pty/``, opens a pty inside the usual POSIX guard, and nobody
    notices the missing marker because it passes on an idle machine."""
    sources = _test_module_sources()
    phantom = f"{TERMINAL_HOST.path}test_future_scenarios.py"
    sources[phantom] = 'import os\n\nif os.name == "posix":\n    import pty\n'
    assert _unmarked_terminal_files(sources) == [phantom]


def test_every_file_that_opens_a_terminal_carries_the_marker__mutation_marked_is_accepted() -> None:
    """And the same file WITH the marker must pass, so the guard is reporting
    the missing marker rather than the import."""
    sources = _test_module_sources()
    sources[f"{TERMINAL_HOST.path}test_future_scenarios.py"] = (
        'import os\n\nimport pytest\n\nif os.name == "posix":\n    import pty\n'
        "\npytestmark = [pytest.mark.pty]\n"
    )
    assert _unmarked_terminal_files(sources) == []


# The same premise applied to the local check command. Its tiers run the
# identical wholesale sweep, so without the deselection a developer's
# pre-commit check is where these tests flake, and the report is unreadable
# there: no artifacts, no diagnostics, just a red that does not reproduce.

#: The tiers of the check command; each runs a parallel sweep over the suite
#: the way the CI lane does, and therefore inherits the same problem.
LOCAL_TIERS = check_set.TIERS
#: The tier that is the merge gate, and so must also run what the sweep drops.
SERIAL_MIRROR_TIERS = ("full",)


def _script_pytest_lines(source: str) -> list[str]:
    return [line for line in source.splitlines() if "uv run pytest " in line]


def _repo_source(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _tier_source(tier: str, base: str | None = None) -> str:
    """The commands *tier* runs over the whole tree, one shell line each; with
    *base*, the checks that judge a range are armed against it."""
    argv = [tier, "--base", base] if base else [tier]
    return "\n".join(_render(line).replace("\n", " ") for line in check_set.resolve(argv))


def _lane_invocations(source: str) -> list[str]:
    """The tier's wholesale ``pytest tests/`` sweeps, ignoring narrower runs
    and each row's serial run, which may name the same tree."""
    selects = {row.select() for row in LANE_LEAVING_MARKERS}
    return [
        line
        for line in _script_pytest_lines(source)
        if "--ignore=tests/e2e" in line and not any(select in line for select in selects)
    ]


def _sweeps_missing_the_deselection(source: str, row: LaneLeavingMarker) -> list[str]:
    """Wholesale sweeps in *source* that do not hand the row's marker away
    (empty = all of them do)."""
    return [line.strip() for line in _lane_invocations(source) if row.deselect_term() not in line]


def _drop_the_term(source: str, row: LaneLeavingMarker) -> str:
    """*source* with the row's term removed from every ``-m "..."`` expression,
    whatever it is combined with (the quick tier adds ``not slow``); an
    expression left empty is removed whole."""

    def _without(match: re.Match[str]) -> str:
        terms = [t for t in match.group(1).split(" and ") if t != row.deselect_term()]
        return f' -m "{" and ".join(terms)}"' if terms else ""

    return re.sub(r' -m "([^"]*)"', _without, source)


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", LOCAL_TIERS)
def test_every_check_tier_deselects_every_lane_leaving_marker(
    tier: str, row: LaneLeavingMarker
) -> None:
    """Every tier of the check command sweeps ``tests/`` and must hand the marker away too.
    The premise of the handover is that these tests cannot share a machine
    with four workers, and that is as true on a laptop as on a runner. It is
    worse on a laptop, in fact: there are no artifacts and no per-worker
    diagnostics there, so the flake arrives as a red that does not reproduce."""
    source = _tier_source(tier)
    assert _lane_invocations(source), (
        f"{tier} no longer runs a wholesale `pytest tests/` sweep; this pin is stale"
    )
    assert _sweeps_missing_the_deselection(source, row) == [], (
        f"{tier} sweeps tests/ without deselecting the {row.marker!r} marker: "
        f"{_sweeps_missing_the_deselection(source, row)}"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", LOCAL_TIERS)
def test_every_check_tier_deselects_every_lane_leaving_marker__mutation_drops_it(
    tier: str, row: LaneLeavingMarker
) -> None:
    source = _tier_source(tier)
    mutated = _drop_the_term(source, row)
    assert mutated != source, f"no {row.marker!r} deselection in {tier}; mutation is stale"
    assert _sweeps_missing_the_deselection(mutated, row) != []


#: Any xdist worker count on a pytest line, a digit or ``auto``. A serial run
#: carries none.
_ANY_XDIST_WORKERS_RE = re.compile(r"(?<!\S)-n\s+(auto|\d)")


def _missing_serial_run(source: str, row: LaneLeavingMarker) -> list[str]:
    """What the full tier's serial follow-up for *row* is missing (empty = wired)."""
    serial = [line for line in _script_pytest_lines(source) if row.select() in line]
    if not serial:
        return [f"a serial {row.marker} invocation"]
    return [
        f"xdist flags on {line.strip()!r}"
        for line in serial
        if _ANY_XDIST_WORKERS_RE.search(line) or "--dist" in line
    ]


def _unreached_by_serial_runs(source: str, row: LaneLeavingMarker, files: list[str]) -> list[str]:
    """The *files* that none of the tier's serial runs for *row* collects
    (empty = every one is reached).

    Each line carrying the row's selection is read from ``uv run pytest`` up to
    the next shell separator, the way ``_ci_pytest_invocations`` cuts a ci.yml
    command, so a wrapper before it (``run_bounded.py … --``) and a ``; then``
    after it are not read as paths. A line whose only path is ``"$@"`` reaches
    nothing by ``_invocation_reaches``' shell-variable rule: a run over the
    caller's paths owes no fixed file."""
    runs = [
        _SHELL_SEPARATOR_RE.split(line[line.index("uv run pytest ") :], maxsplit=1)[0]
        for line in _script_pytest_lines(source)
        if row.select() in line
    ]
    return [name for name in files if not any(_invocation_reaches(run, name) for run in runs)]


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", SERIAL_MIRROR_TIERS)
def test_the_full_tier_runs_what_the_sweep_deselects(
    tier: str, row: LaneLeavingMarker, workflow: dict[str, Any]
) -> None:
    """The full tier is the merge gate, so it owes the same serial follow-up the
    lane has. The quick tier is a documented fast sweep and legitimately stops
    at the deselection; the full tier would otherwise quietly cover less than
    CI does."""
    missing = _missing_serial_run(_tier_source(tier), row)
    assert missing == [], (
        f"{tier} deselects the {row.marker!r} marker in its sweep but does not "
        f"run it serially afterwards ({missing}): add `uv run pytest "
        f"{' '.join(host.path for host in _per_pr_hosts(workflow, row))} {row.select()}`"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", SERIAL_MIRROR_TIERS)
def test_the_full_tier_runs_what_the_sweep_deselects__mutation_drops_the_serial_run(
    tier: str, row: LaneLeavingMarker
) -> None:
    """Dropping the follow-up from the full tier leaves a
    gate that covers strictly less than it used to while reporting the same
    green."""
    source = _tier_source(tier)
    mutated = "\n".join(line for line in source.splitlines() if row.select() not in line)
    assert mutated != source, f"no serial {row.marker} run in {tier}; mutation is stale"
    assert _missing_serial_run(mutated, row) == [f"a serial {row.marker} invocation"]


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", SERIAL_MIRROR_TIERS)
def test_the_full_tier_runs_what_the_sweep_deselects__mutation_parallelizes_it(
    tier: str, row: LaneLeavingMarker
) -> None:
    """And speeding the follow-up in the full tier up with
    workers must be reported: it would put the tests back on a contended
    machine, which is what the sweep dropped them to avoid."""
    source = _tier_source(tier)
    mutated = re.sub(rf"({re.escape(row.select())})(?=\s)", r"\1 -n 4", source)
    assert mutated != source, f"serial {row.marker} run not found; mutation is stale"
    assert _missing_serial_run(mutated, row) != []


@pytest.mark.parametrize(
    ("marker", "jobs"), [("pty", ["test"]), ("wallclock", ["qmd-sidecar-e2e", "test"])]
)
def test_per_pr_hosts_are_the_hosts_the_merge_gate_needs(
    marker: str, jobs: list[str], workflow: dict[str, Any]
) -> None:
    """Every host gates the merge, and ``_per_pr_hosts`` is the filter that
    still holds the day an on-demand host is added."""
    assert [host.job for host in _per_pr_hosts(workflow, _row(marker))] == jobs


def test_unreached_by_serial_runs_reads_each_serial_line() -> None:
    """Each serial line is read from ``uv run pytest`` to the next separator;
    a file under neither line's path is the one reported."""
    row = _row("wallclock")
    source = (
        "if ! uv run python scripts/run_bounded.py 900 -- uv run pytest tests/integration "
        "-m wallclock -v; then\n"
        "uv run pytest tests/cli/test_a.py -m wallclock && echo tests/elsewhere\n"
    )
    files = ["tests/integration/test_x.py", "tests/cli/test_a.py", "tests/elsewhere/test_y.py"]
    assert _unreached_by_serial_runs(source, row, files) == ["tests/elsewhere/test_y.py"]


def _per_pr_carriers(
    wf: dict[str, Any], row: LaneLeavingMarker, sources: dict[str, str]
) -> list[str]:
    """The row's marked files under a host that runs on every pull request."""
    paths = [host.path for host in _per_pr_hosts(wf, row)]
    return [name for name in _carriers(row, sources) if any(name.startswith(p) for p in paths)]


def _check_serial_reach(
    tier: str,
    source: str,
    row: LaneLeavingMarker,
    wf: dict[str, Any],
    sources: dict[str, str],
) -> None:
    paths = [host.path for host in _per_pr_hosts(wf, row)]
    files = _per_pr_carriers(wf, row, sources)
    if paths:
        assert files, (
            f"no {row.marker}-marked file under the per-PR host paths {paths}; "
            f"the reach check is stale"
        )
    unreached = _unreached_by_serial_runs(source, row, files)
    assert unreached == [], (
        f"{tier}'s serial {row.marker} runs reach none of {unreached}, which CI "
        f"times on every pull request under {paths}: widen the check's serial path "
        f"or move the file"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", SERIAL_MIRROR_TIERS)
def test_serial_runs_reach_every_per_pr_marked_file(
    tier: str, row: LaneLeavingMarker, workflow: dict[str, Any]
) -> None:
    """A serial run that exists but collects a narrower path than CI's host
    drops the files in between: the sweep deselected them and the follow-up
    never looks there, so the tier reports green over tests it never ran."""
    _check_serial_reach(tier, _tier_source(tier), row, workflow, _test_module_sources())


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", SERIAL_MIRROR_TIERS)
def test_serial_runs_reach_every_per_pr_marked_file__mutation_narrows_the_path(
    tier: str, row: LaneLeavingMarker, workflow: dict[str, Any]
) -> None:
    source = _tier_source(tier)
    hosts = [host.path.rstrip("/") for host in _per_pr_hosts(workflow, row)]
    host_re = re.compile(rf"(?<!\S)(?:{'|'.join(map(re.escape, hosts))})/?(?=\s)")
    mutated = "\n".join(
        host_re.sub("tests/_nowhere/test_x.py", line) if row.select() in line else line
        for line in source.splitlines()
    )
    assert mutated != source, f"no serial {row.marker} run over {hosts} in {tier}; stale"
    files = _per_pr_carriers(workflow, row, _test_module_sources())
    assert _unreached_by_serial_runs(mutated, row, files) == files


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("tier", SERIAL_MIRROR_TIERS)
def test_serial_runs_reach_every_per_pr_marked_file__mutation_adds_a_per_pr_directory(
    tier: str, row: LaneLeavingMarker, workflow: dict[str, Any]
) -> None:
    """A second per-PR host directory is owed too, not only the first."""
    widened = replace(
        row, hosts=(*row.hosts, SerialHost(UNIT_TEST_JOB, "x", "tests/e2e/", row.select()))
    )
    sources = _test_module_sources()
    phantom = f"tests/e2e/test_future_{row.marker}.py"
    assert phantom not in sources, f"{phantom} exists; mutation is stale"
    sources[phantom] = f"pytestmark = [pytest.mark.{row.marker}]\n"
    assert phantom in _per_pr_carriers(workflow, widened, sources)
    assert _unreached_by_serial_runs(
        _tier_source(tier), widened, _per_pr_carriers(workflow, widened, sources)
    ) == [phantom]


# The full tier is held tighter than the quick one: every pytest line it runs
# is the lane's sweep, with exactly the lane's marker expression, one table
# row's serial run, with no workers, or the invariant tests, named file by file
# with no marker and no workers. A line that is none of these (a hand-added
# term, a dropped expression, a parallelized serial run) is drift.

#: A ``-m`` expression on a command line, double-quoted or a bare word.
_MARKER_EXPR_RE = re.compile(r'(?<!\S)-m\s+("[^"]*"|\S+)')


def _gate_lines_off_the_table(source: str) -> list[str]:
    """The tier's pytest lines that are neither the lane's sweep nor one
    row's serial run (empty = every line is on the table)."""
    selects = {row.select() for row in LANE_LEAVING_MARKERS}
    off = []
    for line in _script_pytest_lines(source):
        exprs = [f"-m {expr}" for expr in _MARKER_EXPR_RE.findall(line)]
        has_workers = bool(_ANY_XDIST_WORKERS_RE.search(line))
        sweep = len(exprs) == 1 and exprs[0] == UNIT_LANE_DESELECT and has_workers
        serial = (
            len(exprs) == 1 and exprs[0] in selects and not has_workers and "--dist" not in line
        )
        invariant = (
            not exprs
            and not has_workers
            and all(f" {path} " in f"{line} " for path in framework_checks.INVARIANT_TESTS)
        )
        if not (sweep or serial or invariant):
            off.append(line.strip())
    return off


def _gate_sweeps(source: str) -> list[str]:
    return [line for line in _script_pytest_lines(source) if UNIT_LANE_DESELECT in line]


def _gate_runs_missing(source: str, row: LaneLeavingMarker) -> list[str]:
    """What the tier lacks: a serial run for *row* over a tests directory, and
    the sweep over the whole suite (empty = both are wired)."""
    missing = []
    serial = [line for line in _script_pytest_lines(source) if row.select() in line]
    if not any(re.search(r"(?<!\S)tests/", line) for line in serial):
        missing.append("serial run")
    if not any("--ignore=tests/e2e" in line for line in _gate_sweeps(source)):
        missing.append("sweep")
    return missing


def test_the_gate_runs_exactly_the_lane_expression() -> None:
    """The full tier is the one local spelling of the merge gate, so it runs
    exactly what the lane runs: a term the lane does not carry is a test the
    gate skips and CI runs."""
    source = _tier_source("full")
    assert _script_pytest_lines(source), "the full tier runs no pytest line; this pin is stale"
    assert _gate_lines_off_the_table(source) == [], (
        f"the full tier runs pytest lines that are neither the lane's sweep "
        f"({UNIT_LANE_DESELECT}, with workers) nor one row's serial run (no workers): "
        f"{_gate_lines_off_the_table(source)}"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_the_gate_runs_every_row_serially(row: LaneLeavingMarker) -> None:
    """The sweep deselects every row, so the full tier runs every row serially."""
    missing = _gate_runs_missing(_tier_source("full"), row)
    assert missing == [], f"the full tier lacks, for {row.marker!r}: {missing}"


def test_the_gate_runs_exactly_the_lane_expression__mutation_adds_a_term() -> None:
    source = _tier_source("full")
    expr = UNIT_LANE_DESELECT[:-1]
    mutated = source.replace(UNIT_LANE_DESELECT, f'{expr} and not browser"', 1)
    assert mutated != source, "no lane sweep in the full tier; mutation is stale"
    assert _gate_lines_off_the_table(mutated) != []


def test_the_gate_runs_exactly_the_lane_expression__mutation_drops_the_expression() -> None:
    source = _tier_source("full")
    mutated = source.replace(f" {UNIT_LANE_DESELECT}", "", 1)
    assert mutated != source, "no lane sweep in the full tier; mutation is stale"
    assert _gate_lines_off_the_table(mutated) != []


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_the_gate_runs_every_row_serially__mutation_drops_the_serial_run(
    row: LaneLeavingMarker,
) -> None:
    source = _tier_source("full")
    mutated = "\n".join(line for line in source.splitlines() if row.select() not in line)
    assert mutated != source, f"no serial {row.marker} run in the full tier; mutation is stale"
    assert "serial run" in _gate_runs_missing(mutated, row)


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
def test_the_gate_runs_exactly_the_lane_expression__mutation_parallelizes_a_serial_run(
    row: LaneLeavingMarker,
) -> None:
    source = _tier_source("full")
    mutated = re.sub(rf"({re.escape(row.select())})(?=\s)", r"\1 -n 4", source, count=1)
    assert mutated != source, f"no serial {row.marker} run in the full tier; mutation is stale"
    assert _gate_lines_off_the_table(mutated) != []


#: The written-down copies of the parallel sweep a contributor or a skill copies
#: by hand. Each must carry the unit lane's marker expression, or the copy runs
#: the lane-leaving markers under workers that distort them.
DOCUMENTED_SWEEPS = (
    "tests/README.md",
    "CONTRIBUTING.md",
    "plugins/osprey/skills/release/SKILL.md",
    "plugins/osprey/skills/test-audit/SKILL.md",
)
#: A pytest invocation with xdist workers: a fixed count or ``auto``.
_DOCUMENTED_XDIST_RE = re.compile(r"(?<!\S)-n\s+(auto|\d+)")


def _documented_sweep_lines(source: str) -> list[str]:
    """Lines that run pytest with workers over ``tests/`` wholesale, or over one
    area when the line already carries a marker expression."""
    return [
        line
        for line in source.splitlines()
        if "pytest " in line
        and _DOCUMENTED_XDIST_RE.search(line)
        and ("tests/ --ignore=tests/e2e" in line or ("tests/<area>" in line and "-m " in line))
    ]


def _documented_sweeps_missing_terms(source: str, row: LaneLeavingMarker) -> list[str]:
    """Documented sweeps in *source* that do not hand the row's marker away
    (empty = all of them do)."""
    return [
        line.strip() for line in _documented_sweep_lines(source) if row.deselect_term() not in line
    ]


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("path", DOCUMENTED_SWEEPS, ids=DOCUMENTED_SWEEPS)
def test_documented_sweeps_carry_the_lane_expression(path: str, row: LaneLeavingMarker) -> None:
    """A documented parallel sweep is the command a reader copies, so it hands
    away every marker the CI lane hands away."""
    source = _repo_source(path)
    assert _documented_sweep_lines(source), (
        f"{path} no longer documents a parallel sweep; this pin is stale"
    )
    assert _documented_sweeps_missing_terms(source, row) == [], (
        f"{path} documents a parallel sweep without deselecting the {row.marker!r} marker "
        f"({row.deselect_term()!r}): {_documented_sweeps_missing_terms(source, row)}"
    )


@pytest.mark.parametrize("row", LANE_LEAVING_MARKERS, ids=_ROW_IDS)
@pytest.mark.parametrize("path", DOCUMENTED_SWEEPS, ids=DOCUMENTED_SWEEPS)
def test_documented_sweeps_carry_the_lane_expression__mutation_drops_a_term(
    path: str, row: LaneLeavingMarker
) -> None:
    source = _repo_source(path)
    mutated = _drop_the_term(source, row)
    assert mutated != source, f"no {row.marker!r} deselection in {path}; mutation is stale"
    assert _documented_sweeps_missing_terms(mutated, row) != []


# ---------------------------------------------------------------------------
# (e) the dockerbuild --ignore guard
# ---------------------------------------------------------------------------


def _dockerbuild_marked_e2e_files() -> list[str]:
    """Every ``tests/e2e/`` file whose source carries the ``dockerbuild``
    marker. Text scan, not collection: importing the files would need their
    (heavy, optional) e2e dependencies, and the marker is always spelled
    literally at module or test level."""
    e2e_dir = CI_YML.parents[2] / "tests" / "e2e"
    return sorted(
        (p.relative_to(e2e_dir.parents[1])).as_posix()
        for p, source in _e2e_test_sources().items()
        if "pytest.mark.dockerbuild" in source
    )


def _run_step_ignores_all(wf: dict[str, Any], files: list[str]) -> list[str]:
    """Return the subset of ``files`` MISSING from the e2e-tests run step's
    ``--ignore`` list (empty = fully guarded)."""
    run_text = _find_named_step(wf, E2E_TESTS_JOB, "Run E2E tests")["run"]
    return [f for f in files if f"--ignore={f}" not in run_text]


def test_every_dockerbuild_marked_file_is_ignored_in_e2e_lane(workflow: dict[str, Any]) -> None:
    """A ``dockerbuild``-marked e2e file runs a real image build + full-stack
    deploy; swept into the shared e2e-tests lane it either double-executes
    (if it has its own job) or leaves host-global residue — fixed
    ``<prefix>-web-<user>``/``<prefix>-nginx`` container names and named
    store volumes whose credentials are pinned on first init —
    that breaks later tests on the same runner. There is no marker-expression
    equivalent (``-m "not dockerbuild"`` would also drop legit in-lane
    ``slow``-marked tests sharing files), so the curated ``--ignore`` list IS
    the mechanism; this guard makes it total: every marked file, present or
    future, must be ignored here and given its own job."""
    files = _dockerbuild_marked_e2e_files()
    assert files, "expected at least one dockerbuild-marked e2e file (marker scan broke?)"
    missing = _run_step_ignores_all(workflow, files)
    assert missing == [], (
        f"dockerbuild-marked e2e file(s) not --ignored in the '{E2E_TESTS_JOB}' lane: "
        f"{missing} — add an --ignore AND a dedicated job for each"
    )


def test_every_dockerbuild_marked_file_is_ignored__mutation_drops_one_ignore() -> None:
    """Removing a single marked file's ignore line must fail the guard."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], LIFECYCLE_TEST_FILE)
    assert _run_step_ignores_all(mutated, [ORM_TEST_FILE]) == []  # others survive
    assert _run_step_ignores_all(mutated, _dockerbuild_marked_e2e_files()) == [LIFECYCLE_TEST_FILE]


def test_every_dockerbuild_marked_file_is_ignored__mutation_new_marked_file() -> None:
    """A future dockerbuild-marked file with no ignore entry must be reported
    missing — the exact 'leaked into the shared lane' shape this guard exists
    to catch before it costs a 50-minute red run."""
    workflow = _load_workflow()
    phantom = "tests/e2e/test_future_dockerbuild_stack.py"
    assert _run_step_ignores_all(workflow, [*_dockerbuild_marked_e2e_files(), phantom]) == [phantom]


# ---------------------------------------------------------------------------
# (f) e2e-lane slimming: orm-roundtrip-e2e + dispatch-overlay-e2e extractions,
# the on-demand channel-finder benchmarks, and the no-advisory-tier gate
# ---------------------------------------------------------------------------


def test_orm_roundtrip_job_runs_the_grid_scan_module(workflow: dict[str, Any]) -> None:
    """The grid-scan roundtrip's ``dockerbuild`` marker takes it OUT of the
    shared lane; this job is where it went. The blanket marker->--ignore guard
    proves it left, and nothing else proves it arrived — delete the ``Run grid
    scan roundtrip`` step and every other wiring test still passes while the
    module runs nowhere at all."""
    assert GRID_TEST_FILE in json.dumps(_jobs(workflow)[ORM_JOB]["steps"])


def test_orm_roundtrip_job_runs_the_grid_scan_module__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[ORM_JOB]["steps"]
    kept = [step for step in steps if GRID_TEST_FILE not in json.dumps(step)]
    assert len(kept) == len(steps) - 1, "expected exactly one step to name the grid module"
    _jobs(mutated)[ORM_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_orm_roundtrip_job_runs_the_grid_scan_module(mutated)


def test_dispatch_overlay_job_declares_llm_secret(workflow: dict[str, Any]) -> None:
    """Inverse of the secret-free checks: the overlay test runs a REAL agent
    turn, and its fixture skips outright without ALS_APG_API_KEY — a job
    missing the secret would green-wash the lane by skipping its only test."""
    assert _job_declares_secret(workflow, OVERLAY_JOB, SECRET_TOKEN)


def test_dispatch_overlay_job_declares_llm_secret__mutation_strips_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][OVERLAY_JOB] = json.loads(
        json.dumps(mutated["jobs"][OVERLAY_JOB]).replace("secrets.ALS_APG_API_KEY", "")
    )
    with pytest.raises(AssertionError):
        assert _job_declares_secret(mutated, OVERLAY_JOB, SECRET_TOKEN)


def test_benchmarks_job_is_dispatch_gated_and_lane_ignores_it(workflow: dict[str, Any]) -> None:
    """The channel-finder benchmarks are a statistical quality score, not a
    per-PR correctness gate: they must be ignored by the shared e2e-tests
    lane AND still exist as a manually-dispatched job behind the
    ``run_benchmarks`` input (otherwise the --ignore silently deletes the
    only benchmark signal). The e2e-tests lane must honor the same input in
    the opposite direction, so the benchmark button doesn't also burn a full
    ~19-min LLM lane run."""
    assert _run_step_ignores_all(workflow, [BENCHMARKS_TEST_FILE]) == []
    job_if = _jobs(workflow)[BENCHMARKS_JOB]["if"]
    assert "workflow_dispatch" in job_if and "run_benchmarks" in job_if, (
        f"'{BENCHMARKS_JOB}' must gate on the run_benchmarks dispatch input; got: {job_if!r}"
    )
    lane_if = _jobs(workflow)[E2E_TESTS_JOB]["if"]
    assert "run_benchmarks" in lane_if, (
        f"'{E2E_TESTS_JOB}' must exclude run_benchmarks dispatches; got: {lane_if!r}"
    )


def test_benchmarks_job_is_dispatch_gated__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], BENCHMARKS_TEST_FILE)
    assert _run_step_ignores_all(mutated, [BENCHMARKS_TEST_FILE]) == [BENCHMARKS_TEST_FILE]


def test_tier2_job_is_dispatch_gated_and_lane_ignores_it(workflow: dict[str, Any]) -> None:
    """Tier 2 tests score a model, not a change: a fair task a capable model may
    still miss. The shared e2e-tests lane must ignore their directory, or a
    model's miss reds a PR, AND the on-demand job must exist behind the
    ``run_benchmarks`` input and run that directory, or the --ignore silently
    deletes the only place Tier 2 runs."""
    assert _run_step_ignores_all(workflow, [TIER2_DIR]) == []
    job = _jobs(workflow)[TIER2_JOB]
    assert "workflow_dispatch" in job["if"] and "run_benchmarks" in job["if"], (
        f"'{TIER2_JOB}' must gate on the run_benchmarks dispatch input; got: {job['if']!r}"
    )
    run_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert f"pytest {TIER2_DIR}" in run_text, f"'{TIER2_JOB}' must run {TIER2_DIR}"


def test_tier2_job_is_dispatch_gated__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], TIER2_DIR)
    with pytest.raises(AssertionError):
        test_tier2_job_is_dispatch_gated_and_lane_ignores_it(mutated)


def test_tier2_job_is_dispatch_gated__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _jobs(mutated)[TIER2_JOB]
    with pytest.raises((AssertionError, KeyError)):
        test_tier2_job_is_dispatch_gated_and_lane_ignores_it(mutated)


def test_tier2_job_is_dispatch_gated__mutation_runs_on_every_pr() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[TIER2_JOB]["if"] = "always()"
    with pytest.raises(AssertionError):
        test_tier2_job_is_dispatch_gated_and_lane_ignores_it(mutated)


def _gate_run_text(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, GATE_JOB, "Check all jobs status")["run"]


def test_gate_checks_every_needed_job(workflow: dict[str, Any]) -> None:
    """Completeness: every job listed in the gate's ``needs`` must have its
    ``needs.<job>.result`` examined by the gate script. A needs entry the
    script never reads is decorative — the job could go red forever inside a
    green check (the exact shape the old advisory tier had)."""
    run_text = _gate_run_text(workflow)
    unchecked = [
        job for job in _jobs(workflow)[GATE_JOB]["needs"] if f"needs.{job}.result" not in run_text
    ]
    assert unchecked == [], f"gate never examines: {unchecked}"


def test_gate_checks_every_needed_job__mutation_adds_unchecked_need() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].append("phantom-lane")
    run_text = _gate_run_text(mutated)
    unchecked = [
        job for job in _jobs(mutated)[GATE_JOB]["needs"] if f"needs.{job}.result" not in run_text
    ]
    assert unchecked == ["phantom-lane"]


def test_gate_has_no_advisory_tier(workflow: dict[str, Any]) -> None:
    """No lane result may be waved through as 'non-blocking': every checked
    lane either passes, is legitimately skipped (its ``if:`` didn't match the
    event), or fails the gate. The literal advisory marker phrase from the
    old gate must never reappear."""
    run_text = _gate_run_text(workflow)
    assert "non-blocking" not in run_text
    assert "exit 1" in run_text


def _gating_e2e_jobs(wf: dict[str, Any]) -> list[str]:
    needs = _jobs(wf)[GATE_JOB]["needs"]
    return [j for j in (ORM_JOB, OVERLAY_JOB, CATALOG_JOB, SANDBOX_JOB) if j in needs]


def test_all_checks_passed_needs_promoted_and_new_lanes(workflow: dict[str, Any]) -> None:
    """The two extracted lanes AND the two bluesky lanes must all gate the
    merge. Deliberately `all`, not `any` — the same
    silent-partial-fix guard shape as ``_needs_contains_both_new_jobs``."""
    assert _gating_e2e_jobs(workflow) == [ORM_JOB, OVERLAY_JOB, CATALOG_JOB, SANDBOX_JOB]


def test_all_checks_passed_needs_promoted_lanes__mutation_drops_sandbox() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(SANDBOX_JOB)
    assert CATALOG_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the other survives untouched
    with pytest.raises(AssertionError):
        assert _gating_e2e_jobs(mutated) == [ORM_JOB, OVERLAY_JOB, CATALOG_JOB, SANDBOX_JOB]


# ---------------------------------------------------------------------------
# dependency-floor lane: the declared pandas/numpy minimums are actually run
# ---------------------------------------------------------------------------


def test_dependency_floor_job_pins_the_versions_pyproject_declares() -> None:
    """The lane certifies nothing if its pin drifts from the declared floor."""
    pyproject = tomllib.loads((CI_YML.parents[2] / "pyproject.toml").read_text())
    declared = {
        dep.split(">=")[0]: dep.split(">=")[1]
        for dep in pyproject["project"]["dependencies"]
        if dep.startswith(("pandas>=", "numpy>="))
    }
    assert declared, "pyproject no longer declares pandas/numpy floors"

    job = json.dumps(_jobs(_load_workflow())[FLOOR_JOB])
    for package, floor in declared.items():
        assert f"{package}=={floor}" in job, (
            f"{FLOOR_JOB} does not pin {package}=={floor}; pyproject declares >={floor}"
        )


def test_dependency_floor_job_keeps_the_pin_when_running_tests() -> None:
    """A bare ``uv run`` re-syncs and silently restores the newest pandas,
    so every ``uv run`` in the job must carry ``--no-sync``.
    """
    steps = _jobs(_load_workflow())[FLOOR_JOB]["steps"]
    uv_runs = [s["run"] for s in steps if "uv run" in s.get("run", "")]
    assert uv_runs, f"{FLOOR_JOB} has no `uv run` step"
    for run in uv_runs:
        assert "uv run --no-sync" in run, f"`uv run` without --no-sync would undo the pin: {run!r}"


def test_dependency_floor_gates_the_merge(workflow: dict[str, Any]) -> None:
    assert FLOOR_JOB in _jobs(workflow)[GATE_JOB]["needs"]


def test_dependency_floor_gates_the_merge__mutation_drops_needs_entry() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(FLOOR_JOB)
    with pytest.raises(AssertionError):
        assert FLOOR_JOB in _jobs(mutated)[GATE_JOB]["needs"]


# ---------------------------------------------------------------------------
# (h) nextcloud-talk-bridge-e2e lane
# ---------------------------------------------------------------------------


def test_e2e_lane_ignores_nextcloud_talk_bridge(workflow: dict[str, Any]) -> None:
    """The lane runs a real Nextcloud container plus a dispatcher/worker pair;
    swept into the shared e2e-tests lane it would double-execute against its
    own fixed host port and exact-named container."""
    assert _run_step_ignores_all(workflow, [NEXTCLOUD_TEST_FILE]) == []


def test_e2e_lane_ignores_nextcloud_talk_bridge__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], NEXTCLOUD_TEST_FILE)
    assert _run_step_ignores_all(mutated, [BENCHMARKS_TEST_FILE]) == []  # others survive
    assert _run_step_ignores_all(mutated, [NEXTCLOUD_TEST_FILE]) == [NEXTCLOUD_TEST_FILE]


def test_gate_checks_nextcloud_talk_bridge_result(workflow: dict[str, Any]) -> None:
    """A ``needs`` entry the gate script never reads is decorative — the
    generic completeness guard above catches that, but this lane gets its own
    named check so a partial removal names the right lane in the failure."""
    assert f"needs.{NEXTCLOUD_JOB}.result" in _gate_run_text(workflow)


def test_gate_checks_nextcloud_talk_bridge_result__mutation_drops_check_line() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if NEXTCLOUD_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    with pytest.raises(AssertionError):
        assert f"needs.{NEXTCLOUD_JOB}.result" in _gate_run_text(mutated)


# ---------------------------------------------------------------------------
# (i) Nextcloud fixture image-pin drift guard
# ---------------------------------------------------------------------------
#
# .github/workflows/build-nextcloud-fixture.yml already cross-checks the four
# values that live INSIDE the Dockerfile and refuses to publish on any
# disagreement. What no workflow can see is the fifth value: the ghcr tag the
# e2e module asks for at run time. That constant is what binds the published
# image to the pins it claims to carry, so a bump that moves `FROM`/`ARG`
# without moving the tag (or the reverse) yields an image whose name lies about
# its contents. The comparison below is therefore always against the
# authoritative `FROM`/`ARG`, never against the `# ..._PIN=` comments — a guard
# that validated a comment against a comment would constrain nothing.

_FROM_RE = re.compile(r"^FROM\s+nextcloud:(\S+)-apache\s*$", re.MULTILINE)
_ARG_RE = re.compile(r"^ARG\s+SPREED_VERSION=(\S+)\s*$", re.MULTILINE)
_NEXTCLOUD_COMMENT_RE = re.compile(r"^#\s*NEXTCLOUD_PIN=(\S+)\s*$", re.MULTILINE)
_SPREED_COMMENT_RE = re.compile(r"^#\s*SPREED_PIN=(\S+)\s*$", re.MULTILINE)


def _dockerfile_text() -> str:
    return (CI_YML.parents[2] / NEXTCLOUD_DOCKERFILE).read_text(encoding="utf-8")


def _sole_match(pattern: re.Pattern[str], text: str, label: str) -> str:
    found = pattern.findall(text)
    assert len(found) == 1, f"expected exactly one {label} in {NEXTCLOUD_DOCKERFILE}; got {found}"
    return found[0]


def _e2e_fixture_image() -> str:
    """The image reference constant from the e2e module, imported not regexed.

    The module imports cleanly with no container runtime present: its only
    module-level runtime touch is ``shutil.which`` inside a ``skipif`` marker,
    and the daemon/image checks are ``pytest.skip`` calls inside a fixture
    body. So the real object is available, and a source parse would only be a
    second, weaker spelling of the same fact. Loaded under a private module
    name so it can never collide with pytest's own collection of that file.
    """
    spec = importlib.util.spec_from_file_location(
        "_nextcloud_e2e_pin_probe", CI_YML.parents[2] / NEXTCLOUD_TEST_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module: @dataclass resolves annotations through
    # sys.modules[cls.__module__] and raises AttributeError without it.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return str(module.NEXTCLOUD_FIXTURE_IMAGE)
    finally:
        sys.modules.pop(spec.name, None)


def _assert_fixture_pins_agree(dockerfile_text: str, image_reference: str) -> None:
    """The whole pin contract in one place, parameterized on its two inputs so
    a mutation can feed it a drifted copy of either side."""
    nextcloud = _sole_match(_FROM_RE, dockerfile_text, "FROM nextcloud:<ver>-apache")
    spreed = _sole_match(_ARG_RE, dockerfile_text, "ARG SPREED_VERSION=<ver>")
    nextcloud_comment = _sole_match(_NEXTCLOUD_COMMENT_RE, dockerfile_text, "# NEXTCLOUD_PIN=")
    spreed_comment = _sole_match(_SPREED_COMMENT_RE, dockerfile_text, "# SPREED_PIN=")

    assert nextcloud == nextcloud_comment, (
        f"Dockerfile FROM pins nextcloud {nextcloud} but the NEXTCLOUD_PIN comment "
        f"says {nextcloud_comment}"
    )
    assert spreed == spreed_comment, (
        f"Dockerfile ARG pins spreed {spreed} but the SPREED_PIN comment says {spreed_comment}"
    )

    expected_tag = f"{nextcloud}-spreed{spreed}"
    actual_tag = image_reference.rsplit(":", 1)[-1]
    assert actual_tag == expected_tag, (
        f"{NEXTCLOUD_TEST_FILE}'s NEXTCLOUD_FIXTURE_IMAGE is tagged {actual_tag!r}, but the "
        f"authoritative FROM/ARG in {NEXTCLOUD_DOCKERFILE} build {expected_tag!r} — the "
        f"published image would advertise contents it does not have"
    )


def test_nextcloud_fixture_pins_agree_across_dockerfile_and_e2e_module() -> None:
    _assert_fixture_pins_agree(_dockerfile_text(), _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_bumps_from_without_the_tag() -> None:
    """The real drift shape: someone bumps the base image (and dutifully the
    NEXTCLOUD_PIN comment with it) but leaves the module's ghcr tag behind."""
    text = _dockerfile_text()
    mutated = text.replace("FROM nextcloud:33.0.7-apache", "FROM nextcloud:33.0.8-apache").replace(
        "NEXTCLOUD_PIN=33.0.7", "NEXTCLOUD_PIN=33.0.8"
    )
    assert mutated != text, "mutation matched nothing — the pin spelling moved"
    with pytest.raises(AssertionError, match="advertise contents it does not have"):
        _assert_fixture_pins_agree(mutated, _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_bumps_spreed_arg_without_the_tag() -> None:
    """Same failure from the other authoritative value, proving both halves of
    the derived tag are independently load-bearing."""
    text = _dockerfile_text()
    mutated = text.replace("ARG SPREED_VERSION=23.0.9", "ARG SPREED_VERSION=23.1.0").replace(
        "SPREED_PIN=23.0.9", "SPREED_PIN=23.1.0"
    )
    assert mutated != text, "mutation matched nothing — the pin spelling moved"
    with pytest.raises(AssertionError, match="advertise contents it does not have"):
        _assert_fixture_pins_agree(mutated, _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_comment_drifts_from_authoritative_from() -> None:
    """Bumping only the comment must fail on the comment/authoritative check —
    and must NOT be waved through by the tag check, which is deliberately
    derived from FROM/ARG and would still agree here."""
    text = _dockerfile_text()
    mutated = text.replace("NEXTCLOUD_PIN=33.0.7", "NEXTCLOUD_PIN=33.0.8")
    assert mutated != text, "mutation matched nothing — the pin spelling moved"
    with pytest.raises(AssertionError, match="NEXTCLOUD_PIN comment"):
        _assert_fixture_pins_agree(mutated, _e2e_fixture_image())


def test_nextcloud_fixture_pins__mutation_tag_drifts_from_dockerfile() -> None:
    """And the inverse: the module's tag moves while the Dockerfile stands
    still. Feeding a drifted reference through the same helper proves the tag
    comparison, not just the comment comparison, is doing work."""
    with pytest.raises(AssertionError, match="advertise contents it does not have"):
        _assert_fixture_pins_agree(
            _dockerfile_text(), "ghcr.io/als-apg/nextcloud-talk-fixture:33.0.7-spreed23.1.0"
        )


# ---------------------------------------------------------------------------
# (j) gchat-bridge-e2e lane
# ---------------------------------------------------------------------------


def _emulator_container_e2e_files() -> list[str]:
    """Every ``tests/e2e/`` file that reaches the fixed-name Pub/Sub emulator
    container, found by the fixture's own name.

    The scan token is the FIXTURE name, not the module name, and deliberately
    so: ``tests/e2e/fixtures/conftest.py`` registers ``pubsub_emulator``
    directory-wide, so a file there can reach the container by taking it as a
    test argument without importing anything. ``pubsub_emulator`` is a
    substring of ``gchat_pubsub_emulator``, so this one token covers the
    import spelling and the fixture-argument spelling at once.

    Text scan rather than collection, for the same reason as the
    ``dockerbuild`` scan above: importing these files needs their heavy
    optional e2e dependencies, and both spellings are always literal.
    """
    e2e_dir = CI_YML.parents[2] / "tests" / "e2e"
    return sorted(
        p.relative_to(e2e_dir.parents[1]).as_posix()
        for p, source in _e2e_test_sources().items()
        if PUBSUB_FIXTURE_NAME in source
    )


def test_shared_lane_ignores_every_emulator_container_file(workflow: dict[str, Any]) -> None:
    """The collision this guard exists for: ``PubSubEmulator.start()`` pre-cleans
    by ``rm -f``-ing the exact container name it is about to use, and every file
    below reaches that same name. The shared e2e lane runs ``-n 4 --dist
    loadfile``, which keeps a file's tests on one worker but runs different
    FILES concurrently — so two of these landing in that lane means one file's
    pre-clean kills the other's live emulator mid-test. Ignoring only the
    bridge module would fix today's pair and leave the hazard armed for the
    next file to use the fixture, so the guard is over the whole scan."""
    files = _emulator_container_e2e_files()
    assert set(files) >= {GCHAT_SMOKE_FILE, GCHAT_TEST_FILE}, (
        f"emulator-fixture scan found {files} — expected at least the two Google Chat "
        f"files; the fixture module was renamed or the scan broke"
    )
    missing = _run_step_ignores_all(workflow, files)
    assert missing == [], (
        f"file(s) using the fixed-name Pub/Sub emulator container not --ignored in the "
        f"'{E2E_TESTS_JOB}' lane: {missing} — they would run concurrently there and "
        f"tear down each other's emulator"
    )


def test_shared_lane_ignores_emulator_files__mutation_drops_bridge_ignore() -> None:
    """Dropping only the bridge module's ignore must fail — the smoke file
    surviving is exactly the half-fix this guard rejects."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], GCHAT_TEST_FILE)
    assert _run_step_ignores_all(mutated, [GCHAT_SMOKE_FILE]) == []  # the other survives
    with pytest.raises(AssertionError):
        test_shared_lane_ignores_every_emulator_container_file(mutated)


def test_shared_lane_ignores_emulator_files__mutation_drops_smoke_ignore() -> None:
    """And the mirror image: leaving the fixture smoke test in the parallel
    lane is just as fatal, since it starts the same container by the same name."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], GCHAT_SMOKE_FILE)
    assert _run_step_ignores_all(mutated, [GCHAT_TEST_FILE]) == []  # the other survives
    with pytest.raises(AssertionError):
        test_shared_lane_ignores_every_emulator_container_file(mutated)


def _gchat_pytest_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """The gchat job's pytest invocations, in the order Actions will run them.

    Matched on the full ``uv run pytest`` invocation rather than a bare
    ``pytest`` substring, so a step that only MENTIONS pytest — in a comment,
    or in a message naming the command a developer should run locally — is not
    mistaken for one that runs it. A false positive there would be read as an
    unguarded emulator user and fail the sequencing check for no reason.
    """
    return [s for s in _jobs(wf)[GCHAT_JOB]["steps"] if "uv run pytest " in s.get("run", "")]


def _emulator_files_per_step(wf: dict[str, Any]) -> list[list[str]]:
    """For each pytest step, which emulator-using files it selects."""
    files = _emulator_container_e2e_files()
    return [sorted(f for f in files if f in step["run"]) for step in _gchat_pytest_steps(wf)]


def test_emulator_files_run_one_per_sequential_step(workflow: dict[str, Any]) -> None:
    """Having moved both files out of the parallel lane, the job they moved
    INTO must not reintroduce the same overlap. Two properties make that
    provable rather than incidental: one file per pytest step (Actions runs
    steps strictly in order, in separate processes, so two steps can never
    overlap), and no xdist flags (which would parallelize within a step). A
    single pytest invocation naming both files would satisfy neither — it puts
    two module-scoped emulator fixtures in one process, where teardown order
    decides whether they overlap."""
    per_step = _emulator_files_per_step(workflow)
    assert [len(selected) for selected in per_step] == [1] * len(per_step), (
        f"each pytest step in '{GCHAT_JOB}' must select exactly one emulator-using file; "
        f"got {per_step}"
    )
    assert sorted(f for selected in per_step for f in selected) == (
        _emulator_container_e2e_files()
    ), f"'{GCHAT_JOB}' must run every emulator-using file exactly once; got {per_step}"
    for step in _gchat_pytest_steps(workflow):
        assert not _XDIST_WORKERS_RE.search(step["run"]), (
            f"step {step.get('name')!r} runs pytest with xdist workers; the emulator's "
            f"fixed container name cannot survive parallel workers"
        )
        assert "--dist" not in step["run"], (
            f"step {step.get('name')!r} sets an xdist dist mode; this job must stay serial"
        )


def test_emulator_files_run_one_per_step__mutation_merges_both_into_one_step() -> None:
    """The tempting simplification — one pytest call listing both files — must
    fail, because it makes non-overlap depend on fixture teardown order rather
    than on the job's structure."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _gchat_pytest_steps(mutated)
    assert len(steps) == 2, f"expected two pytest steps to merge; got {len(steps)}"
    steps[0]["run"] = steps[0]["run"].replace(
        GCHAT_SMOKE_FILE, f"{GCHAT_SMOKE_FILE} {GCHAT_TEST_FILE}"
    )
    _jobs(mutated)[GCHAT_JOB]["steps"].remove(steps[1])
    with pytest.raises(AssertionError):
        test_emulator_files_run_one_per_sequential_step(mutated)


def test_emulator_files_run_one_per_step__mutation_adds_xdist_workers() -> None:
    """Speeding the job up with ``-n 2`` reinstates the collision inside a
    single step, so the pin must reject it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _gchat_pytest_steps(mutated)[0]
    step["run"] = step["run"].replace("-v --tb=short", "-v --tb=short -n 2")
    with pytest.raises(AssertionError, match="xdist workers"):
        test_emulator_files_run_one_per_sequential_step(mutated)


def _gchat_install_cmd(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, GCHAT_JOB, "Install osprey")["run"]


def test_gchat_job_installs_the_gchat_extra(workflow: dict[str, Any]) -> None:
    """The e2e module drives a REAL ``google-cloud-pubsub`` client against the
    emulator and guards that with a module-level ``skipif``, so a job that
    syncs only ``dev`` skips the entire file and reports success — a green
    check over zero executed tests, confirmed live before this lane existed."""
    cmd = _gchat_install_cmd(workflow)
    assert f"--extra {GCHAT_EXTRA}" in cmd, (
        f"'{GCHAT_JOB}' must `uv sync --extra {GCHAT_EXTRA}`; got: {cmd}"
    )


def test_gchat_job_installs_the_gchat_extra__mutation_drops_extra() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GCHAT_JOB, "Install osprey")
    step["run"] = step["run"].replace(f" --extra {GCHAT_EXTRA}", "")
    with pytest.raises(AssertionError):
        test_gchat_job_installs_the_gchat_extra(mutated)


_JUNIT_RE = re.compile(r"--junitxml=(\S+)")


def _junit_reports_written(wf: dict[str, Any]) -> list[str]:
    return [r for step in _gchat_pytest_steps(wf) for r in _JUNIT_RE.findall(step["run"])]


def test_gchat_job_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """The extra above is the fix; this is the backstop that makes the whole
    class of failure loud. Every skip condition this lane has — absent
    container runtime, absent emulator image, absent ``gchat`` extra, absent
    provider key — is a CI misconfiguration that pytest reports as success, so
    the job reads the junit reports and fails on any skip (and on an empty
    collection, which would otherwise pass a zero-skip check trivially). Every
    pytest step must write a report the gate actually reads: a run whose report
    nothing inspects is back to skipping its way to green."""
    reports = _junit_reports_written(workflow)
    assert len(reports) == len(_gchat_pytest_steps(workflow)), (
        f"every pytest step in '{GCHAT_JOB}' must write a --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, GCHAT_JOB, GCHAT_SKIP_GATE_STEP)["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{GCHAT_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_gchat_job_fails_on_any_skipped_test__mutation_drops_a_junit_report() -> None:
    """A pytest step that writes no report is invisible to the gate — the
    silent-partial-fix shape, one lane guarded and one not."""
    mutated = copy.deepcopy(_load_workflow())
    step = _gchat_pytest_steps(mutated)[0]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_gchat_job_fails_on_any_skipped_test(mutated)


def test_gchat_job_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is
    decorative: the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GCHAT_JOB, GCHAT_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_gchat_job_fails_on_any_skipped_test(mutated)


def test_gchat_bridge_job_declares_llm_secret(workflow: dict[str, Any]) -> None:
    """Same shape as the dispatch-overlay check, for the same reason: the
    agentic tier in this module runs a REAL agent turn and is marked
    ``requires_als_apg``, which the root conftest turns into a skip when the
    key is absent. The zero-skip gate would catch that, but only as "something
    skipped"; this assertion names the cause, and fails at the one place a
    reviewer would look for it."""
    assert _job_declares_secret(workflow, GCHAT_JOB, SECRET_TOKEN)


def test_gchat_bridge_job_declares_llm_secret__mutation_strips_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][GCHAT_JOB] = json.loads(
        json.dumps(mutated["jobs"][GCHAT_JOB]).replace("secrets.ALS_APG_API_KEY", "")
    )
    with pytest.raises(AssertionError):
        assert _job_declares_secret(mutated, GCHAT_JOB, SECRET_TOKEN)


# ---------------------------------------------------------------------------
# (k) teams-bridge-e2e lane
# ---------------------------------------------------------------------------


def test_shared_lane_ignores_both_teams_files(workflow: dict[str, Any]) -> None:
    """Neither Teams file may be left in the shared ``e2e-tests`` lane, and the
    failure it prevents is not a slow test — it is a COLLECTION error for the
    whole lane. Both files import ``azure.servicebus`` at module scope, and
    that lane syncs ``--extra dev`` only, so pytest errors there before running
    anything. Unlike the Google Chat pair the reason is the dependency, not a
    shared container, so the two files are named rather than discovered."""
    missing = _run_step_ignores_all(workflow, [TEAMS_SMOKE_FILE, TEAMS_TEST_FILE])
    assert missing == [], (
        f"Teams e2e file(s) not --ignored in the '{E2E_TESTS_JOB}' lane: {missing} — "
        f"that lane installs no '{TEAMS_EXTRA}' extra, so collecting them errors"
    )


def test_shared_lane_ignores_both_teams_files__mutation_drops_module_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], TEAMS_TEST_FILE)
    assert _run_step_ignores_all(mutated, [TEAMS_SMOKE_FILE]) == []  # the other survives
    with pytest.raises(AssertionError):
        test_shared_lane_ignores_both_teams_files(mutated)


def test_shared_lane_ignores_both_teams_files__mutation_drops_smoke_ignore() -> None:
    """The mirror image, and the half-fix this guard exists for: the smoke file
    imports the same stack, so leaving it behind breaks the lane just as hard."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], TEAMS_SMOKE_FILE)
    assert _run_step_ignores_all(mutated, [TEAMS_TEST_FILE]) == []  # the other survives
    with pytest.raises(AssertionError):
        test_shared_lane_ignores_both_teams_files(mutated)


#: Matches both ``uv run pytest`` and ``uv run --no-sync pytest`` so the
#: ``--no-sync`` pin below still FINDS the step it is asserting about after the
#: flag is mutated away — a finder keyed on the flag would silently find no
#: steps and pass.
_UV_PYTEST_RE = re.compile(r"uv run [^\n]*\bpytest ")


def _teams_pytest_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """The Teams job's pytest invocations, in the order Actions will run them."""
    return [s for s in _jobs(wf)[TEAMS_JOB]["steps"] if _UV_PYTEST_RE.search(s.get("run", ""))]


def test_teams_job_runs_both_teams_files(workflow: dict[str, Any]) -> None:
    """Having moved both files out of the shared lane, this is the lane they
    moved INTO — so it must actually name both. Dropping either leaves a file
    that runs nowhere while every check stays green, which is the same
    silent-coverage-loss shape the ``--ignore`` guard above defends from the
    other side. How many steps they are split across is deliberately not
    pinned: this lane starts no container and binds no host port, so nothing
    makes one arrangement safer than another."""
    selected = " ".join(step["run"] for step in _teams_pytest_steps(workflow))
    missing = [f for f in (TEAMS_SMOKE_FILE, TEAMS_TEST_FILE) if f not in selected]
    assert missing == [], f"'{TEAMS_JOB}' runs no pytest step naming: {missing}"


def test_teams_job_runs_both_teams_files__mutation_drops_the_module() -> None:
    """The tempting half-lane: fixtures prove the fakes answer, and the bridge
    cells — the thing the lane exists for — quietly stop running."""
    mutated = copy.deepcopy(_load_workflow())
    step = _teams_pytest_steps(mutated)[0]
    kept = [line for line in step["run"].splitlines(keepends=True) if TEAMS_TEST_FILE not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert TEAMS_SMOKE_FILE in step["run"]  # the fixture half survives
    with pytest.raises(AssertionError):
        test_teams_job_runs_both_teams_files(mutated)


def _teams_install_cmd(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, TEAMS_JOB, "Install osprey")["run"]


def test_teams_job_installs_the_teams_extra(workflow: dict[str, Any]) -> None:
    """Load-bearing at COLLECTION time, which makes it stricter than the Google
    Chat case: that module skips itself without its extra, this one imports
    ``azure.servicebus`` at module scope and errors. A job that syncs only
    ``dev`` therefore reds — but it reds with an import traceback that reads
    like a code bug, so the requirement is pinned where the cause is named."""
    cmd = _teams_install_cmd(workflow)
    assert f"--extra {TEAMS_EXTRA}" in cmd, (
        f"'{TEAMS_JOB}' must `uv sync --extra {TEAMS_EXTRA}`; got: {cmd}"
    )


def test_teams_job_installs_the_teams_extra__mutation_drops_extra() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TEAMS_JOB, "Install osprey")
    step["run"] = step["run"].replace(f" --extra {TEAMS_EXTRA}", "")
    with pytest.raises(AssertionError):
        test_teams_job_installs_the_teams_extra(mutated)


def test_teams_job_runs_pytest_without_resyncing(workflow: dict[str, Any]) -> None:
    """The other half of the extra, and the half that looks like nothing: a bare
    ``uv run pytest`` re-syncs the environment to the project's default extras
    first, which strips ``azure-servicebus`` straight back out of the venv the
    step above just built. The lane then fails at collection with the extra
    plainly installed in its own log. ``--no-sync`` is what makes the install
    step mean something."""
    steps = _teams_pytest_steps(workflow)
    assert steps, f"'{TEAMS_JOB}' runs no pytest step at all"
    unsynced = [s.get("name") for s in steps if "uv run --no-sync pytest" not in s["run"]]
    assert unsynced == [], (
        f"pytest step(s) in '{TEAMS_JOB}' missing `--no-sync`: {unsynced} — a bare "
        f"`uv run` re-syncs and drops the '{TEAMS_EXTRA}' extra"
    )


def test_teams_job_runs_pytest_without_resyncing__mutation_drops_the_flag() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _teams_pytest_steps(mutated)[0]
    step["run"] = step["run"].replace("uv run --no-sync pytest", "uv run pytest")
    assert _teams_pytest_steps(mutated), "the finder must still see the step it asserts about"
    with pytest.raises(AssertionError, match="missing `--no-sync`"):
        test_teams_job_runs_pytest_without_resyncing(mutated)


def _teams_junit_reports(wf: dict[str, Any]) -> list[str]:
    return [r for step in _teams_pytest_steps(wf) for r in _JUNIT_RE.findall(step["run"])]


def test_teams_job_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """This module declares no ``skipif`` and no runtime gate, so a skip in this
    lane is a bug rather than an environment gap — and pytest's exit code
    cannot express the difference between "everything passed" and "everything
    was skipped". So the junit report is read: zero skips, and a non-zero test
    count so an empty selection cannot pass either. Every pytest step must
    write a report the gate actually reads; a run whose report nothing inspects
    is back to skipping its way to green."""
    reports = _teams_junit_reports(workflow)
    assert len(reports) == len(_teams_pytest_steps(workflow)), (
        f"every pytest step in '{TEAMS_JOB}' must write a --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, TEAMS_JOB, TEAMS_SKIP_GATE_STEP)["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{TEAMS_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_teams_job_fails_on_any_skipped_test__mutation_drops_the_junit_report() -> None:
    """A pytest step that writes no report is invisible to the gate."""
    mutated = copy.deepcopy(_load_workflow())
    step = _teams_pytest_steps(mutated)[0]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_teams_job_fails_on_any_skipped_test(mutated)


def test_teams_job_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is
    decorative: the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TEAMS_JOB, TEAMS_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_teams_job_fails_on_any_skipped_test(mutated)


def test_teams_bridge_job_runs_on_every_pull_request(workflow: dict[str, Any]) -> None:
    """The whole point of a secret-free, container-free bridge lane: no ``if:``
    gate, so it also runs on fork and Dependabot pull requests, where every
    label-gated lane skips. An ``if:`` added here would not fail anything —
    the lane would simply stop running on most PRs while reporting success as a
    skip, which is the failure mode the gated lanes accept in exchange for the
    money they save and this one has no reason to."""
    assert "if" not in _jobs(workflow)[TEAMS_JOB], (
        f"'{TEAMS_JOB}' declares an `if:` gate: {_jobs(workflow)[TEAMS_JOB].get('if')!r} — "
        f"it spends no model tokens and pulls no image, so it runs unconditionally"
    )


def test_teams_bridge_job_runs_on_every_pull_request__mutation_adds_label_gate() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[TEAMS_JOB]["if"] = FULL_CI_LABEL_CLAUSE
    with pytest.raises(AssertionError, match="declares an `if:` gate"):
        test_teams_bridge_job_runs_on_every_pull_request(mutated)


# ---------------------------------------------------------------------------
# the model-free end-to-end lane: the tests/e2e files that reach no gateway
# ---------------------------------------------------------------------------


def _model_free_selected_paths(wf: dict[str, Any]) -> list[str]:
    """The paths the model-free lane's run step hands pytest to collect."""
    run_text = _find_named_step(wf, NO_MODEL_JOB, NO_MODEL_RUN_STEP)["run"]
    return [token for token in run_text.split() if token.startswith("tests/")]


def _record_lane_paths(wf: dict[str, Any]) -> list[str]:
    """The paths the agent record lane's run step hands pytest to collect."""
    run_text = _find_named_step(wf, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)["run"]
    return [token for token in run_text.split() if token.startswith("tests/")]


def _model_free_lanes_paths(wf: dict[str, Any]) -> list[str]:
    """Every path either model-free lane names."""
    return [*_model_free_selected_paths(wf), *_record_lane_paths(wf)]


def _collected_ids(args: list[str]) -> set[str]:
    """The node ids pytest collects for ``args``, from the repo root.

    ``--collect-only`` is exempt from the e2e provider refusal, so no provider
    has to be named.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *args,
        ],
        cwd=CI_YML.parents[2],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {line.strip() for line in result.stdout.splitlines() if "::" in line}


def test_the_model_free_marker_equals_the_model_free_lane_selection(
    workflow: dict[str, Any],
) -> None:
    """A run that names no provider may select only ``model_free`` tests, and
    the lane is where those tests run. A test the lane runs without the marker
    would be refused in a run with no provider; a marked test the lane leaves
    out claims a lane that never runs it."""
    paths = _model_free_lanes_paths(workflow)
    marked = _collected_ids([*paths, "-m", "model_free"])
    selected = _collected_ids(paths)
    lanes = f"'{NO_MODEL_JOB}'/'{RECORD_E2E_JOB}'"
    assert marked == selected, (
        f"marked model_free but not run by {lanes}: {sorted(marked - selected)}; "
        f"run by {lanes} but not marked model_free: {sorted(selected - marked)}"
    )


def test_only_the_model_free_lane_files_carry_the_marker(workflow: dict[str, Any]) -> None:
    """The equality above collects only the lane's own files; a marked module
    anywhere else under ``tests/e2e/`` is read from source instead of from a
    whole-directory collection."""
    root = CI_YML.parents[2]
    lane = set(_model_free_lanes_paths(workflow))
    stray = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "tests" / "e2e").rglob("test_*.py")
        if "mark.model_free" in path.read_text(encoding="utf-8")
        and path.relative_to(root).as_posix() not in lane
    )
    assert stray == [], (
        f"marked model_free but not named in '{NO_MODEL_RUN_STEP}' or "
        f"'{RECORD_E2E_RUN_STEP}': {stray}"
    )


def test_model_free_lane_exists(workflow: dict[str, Any]) -> None:
    assert NO_MODEL_JOB in _jobs(workflow)


def test_model_free_lane_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][NO_MODEL_JOB]
    with pytest.raises(AssertionError):
        assert NO_MODEL_JOB in _jobs(mutated)


def test_model_free_lane_runs_on_fork_pull_requests(workflow: dict[str, Any]) -> None:
    """Every image the lane pulls is public and it declares no gateway key, so
    nothing in it needs a token a fork run does not hold: a same-repo clause
    would take the proof off contributors' pull requests for no reason, and a
    label clause would put a lane that can honestly run on every pull request
    back on the few that are labelled."""
    condition = _jobs(workflow)[NO_MODEL_JOB]["if"]
    assert "github.event_name == 'pull_request'" in condition, (
        f"'{NO_MODEL_JOB}' must run on pull requests"
    )
    assert SAME_REPO_CLAUSE not in condition, (
        f"'{NO_MODEL_JOB}' is limited to same-repo pull requests but pulls only "
        f"public images, so it must run on fork pull requests too"
    )
    assert FULL_CI_LABEL_CLAUSE not in condition, (
        f"'{NO_MODEL_JOB}' is label-gated but spends no model tokens"
    )


def test_model_free_lane_runs_on_fork_pull_requests__mutation_adds_the_label_gate() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[NO_MODEL_JOB]
    job["if"] = job["if"] + f" && {FULL_CI_LABEL_CLAUSE}"
    assert SAME_REPO_CLAUSE not in job["if"]  # the other half survives untouched
    with pytest.raises(AssertionError, match="spends no model tokens"):
        test_model_free_lane_runs_on_fork_pull_requests(mutated)


def test_model_free_lane_runs_on_fork_pull_requests__mutation_re_adds_the_same_repo_clause() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[NO_MODEL_JOB]
    job["if"] = job["if"] + f" && {SAME_REPO_CLAUSE}"
    assert FULL_CI_LABEL_CLAUSE not in job["if"]  # the other half survives untouched
    with pytest.raises(AssertionError, match="fork pull requests too"):
        test_model_free_lane_runs_on_fork_pull_requests(mutated)


def test_model_free_lane_names_every_file_it_owns(workflow: dict[str, Any]) -> None:
    """Named by path rather than by marker because a path that stops existing
    exits pytest non-zero and reds the step, while a marker that stops matching
    selects nothing and greens it — and the on-disk half is what makes the
    first half mean something, since a renamed file would otherwise leave the
    lane asserting about a string."""
    missing = [f for f in NO_MODEL_TEST_FILES if f not in _model_free_selected_paths(workflow)]
    assert missing == [], (
        f"'{NO_MODEL_RUN_STEP}' does not name: {missing} — a file the lane owns and nothing runs"
    )
    absent = [f for f in NO_MODEL_TEST_FILES if not (CI_YML.parents[2] / f).is_file()]
    assert absent == [], (
        f"the lane names file(s) no longer on disk: {absent} — pytest exits non-zero on "
        f"a path it cannot collect, so the step reds until the list is corrected"
    )


def test_model_free_lane_names_every_file_it_owns__mutation_drops_a_file_from_the_run_step() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, NO_MODEL_JOB, NO_MODEL_RUN_STEP)
    lines = step["run"].splitlines(keepends=True)
    kept = [line for line in lines if "tests/e2e/test_scripted_model.py" not in line]
    assert len(kept) == len(lines) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    # the neighbour survives, so the mutation is one file wide
    assert "tests/e2e/claude_code/test_safety_gates_scripted.py" in _model_free_selected_paths(
        mutated
    )
    with pytest.raises(AssertionError, match="does not name"):
        test_model_free_lane_names_every_file_it_owns(mutated)


def test_model_free_lane_carries_no_dependabot_guard(workflow: dict[str, Any]) -> None:
    """The actor guard is a secret guard: the lanes that carry it resolve
    ``ALS_APG_API_KEY``, and a Dependabot run resolves secrets against the
    Dependabot store rather than the Actions one. This lane resolves none,
    and every image it pulls is public. A neighbouring lane's condition copied here
    would drop the model-free proof from the pull requests that move the
    dependency set, and every other pin on this lane would stay green while
    it did. Absence is asserted only while the premise holds: a lane that
    grows an Actions secret fails the first half and wants the guard."""
    assert not _job_declares_secret(workflow, NO_MODEL_JOB, SECRET_TOKEN), (
        f"'{NO_MODEL_JOB}' now declares an Actions secret — the guard this case "
        f"asserts is absent belongs here"
    )
    condition = _jobs(workflow)[NO_MODEL_JOB]["if"]
    assert _DEPENDABOT_GUARD not in condition, (
        f"'{NO_MODEL_JOB}' excludes Dependabot but needs no Actions secret"
    )


def test_model_free_lane_carries_no_dependabot_guard__mutation_adds_the_actor_guard() -> None:
    """A neighbouring lane's actor guard appended to this condition. The
    existing pin on the same condition still passes — neither the same-repo
    clause nor the label clause is there — which is why the absence needs a pin
    of its own."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[NO_MODEL_JOB]
    job["if"] = job["if"] + f" && {_DEPENDABOT_GUARD}"
    test_model_free_lane_runs_on_fork_pull_requests(mutated)  # still green
    with pytest.raises(AssertionError, match="needs no Actions secret"):
        test_model_free_lane_carries_no_dependabot_guard(mutated)


def test_model_free_lane_carries_no_dependabot_guard__mutation_adds_the_gateway_key() -> None:
    """A lane that grows a secret wants the guard, so the case must stop
    asserting its absence rather than pin the lane into a red it cannot
    escape."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][NO_MODEL_JOB]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    assert _DEPENDABOT_GUARD not in _jobs(mutated)[NO_MODEL_JOB]["if"]  # guard half untouched
    with pytest.raises(AssertionError, match="now declares an Actions secret"):
        test_model_free_lane_carries_no_dependabot_guard(mutated)


def test_model_free_lane_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """Every file here is named by path and every test in them runs without a
    credential, so a skip is the lane being wrong about its own runner rather
    than an environment gap — and pytest's exit code cannot tell "everything
    passed" from "everything was skipped"."""
    reports = _JUNIT_RE.findall(_find_named_step(workflow, NO_MODEL_JOB, NO_MODEL_RUN_STEP)["run"])
    assert len(reports) == 1, (
        f"'{NO_MODEL_RUN_STEP}' must write exactly one --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, NO_MODEL_JOB, NO_MODEL_SKIP_GATE_STEP)["run"]
    assert reports[0] in gate, f"'{NO_MODEL_SKIP_GATE_STEP}' never reads: {reports[0]}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_model_free_lane_fails_on_any_skipped_test__mutation_drops_the_junit_report() -> None:
    """A run step that writes no report is invisible to the gate."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, NO_MODEL_JOB, NO_MODEL_RUN_STEP)
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="exactly one --junitxml report"):
        test_model_free_lane_fails_on_any_skipped_test(mutated)


def test_model_free_lane_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is
    decorative: the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, NO_MODEL_JOB, NO_MODEL_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_model_free_lane_fails_on_any_skipped_test(mutated)


# ---------------------------------------------------------------------------
# The agent record replay: its own model-free lane
# ---------------------------------------------------------------------------


def test_record_lane_runs_exactly_the_replay(workflow: dict[str, Any]) -> None:
    """The replay deploys two stacks one after the other on one port block, so
    it runs alone, serially, and by path — a path that stops existing exits
    pytest 4 instead of selecting nothing."""
    paths = _record_lane_paths(workflow)
    assert paths == [RECORD_E2E_FILE], f"'{RECORD_E2E_RUN_STEP}' runs {paths}, not the replay alone"
    assert (REPO_ROOT / RECORD_E2E_FILE).is_file(), f"{RECORD_E2E_FILE} is not on disk"
    run = _find_named_step(workflow, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)["run"]
    assert not re.search(r"(^|\s)(-n|--numprocesses)\b", run), (
        f"'{RECORD_E2E_RUN_STEP}' runs under xdist; its cases share one port block"
    )


def test_record_lane_runs_exactly_the_replay__mutation_adds_xdist() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)
    step["run"] = step["run"].replace("--durations=15", "--durations=15 -n 2")
    with pytest.raises(AssertionError, match="xdist"):
        test_record_lane_runs_exactly_the_replay(mutated)


def test_record_lane_runs_exactly_the_replay__mutation_adds_a_file() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)
    step["run"] = step["run"].replace(
        RECORD_E2E_FILE, f"{RECORD_E2E_FILE} tests/e2e/test_scripted_model.py"
    )
    with pytest.raises(AssertionError, match="replay alone"):
        test_record_lane_runs_exactly_the_replay(mutated)


def test_record_lane_is_not_in_the_model_free_lane(workflow: dict[str, Any]) -> None:
    """One file, one lane: in both it is deployed twice per pull request."""
    assert RECORD_E2E_FILE not in _model_free_selected_paths(workflow), (
        f"'{NO_MODEL_JOB}' still runs {RECORD_E2E_FILE}, which has its own lane"
    )


def test_record_lane_is_not_in_the_model_free_lane__mutation_names_it_there() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, NO_MODEL_JOB, NO_MODEL_RUN_STEP)
    step["run"] = step["run"].replace(
        "tests/e2e/test_scripted_model.py", f"{RECORD_E2E_FILE} tests/e2e/test_scripted_model.py"
    )
    with pytest.raises(AssertionError, match="has its own lane"):
        test_record_lane_is_not_in_the_model_free_lane(mutated)


def test_record_lane_timeout_covers_its_waits(workflow: dict[str, Any]) -> None:
    """The run step's cap holds the two five-minute stopped-source waits and
    both deploys; the job's cap sits above it so a hang fails the run step and
    the diagnostics capture still runs."""
    job = _jobs(workflow)[RECORD_E2E_JOB]
    step = _find_named_step(workflow, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)
    step_cap = int(step.get("timeout-minutes", 0))
    assert step_cap >= RECORD_E2E_MIN_STEP_MINUTES, (
        f"'{RECORD_E2E_RUN_STEP}' caps at {step_cap} min, under the "
        f"{RECORD_E2E_MIN_STEP_MINUTES} min its waits need"
    )
    assert int(job.get("timeout-minutes", 0)) > step_cap, (
        f"'{RECORD_E2E_JOB}' job cap must sit above its run step's {step_cap} min"
    )


def test_record_lane_timeout_covers_its_waits__mutation_shrinks_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)["timeout-minutes"] = 25
    with pytest.raises(AssertionError, match="its waits need"):
        test_record_lane_timeout_covers_its_waits(mutated)


def test_record_lane_timeout_covers_its_waits__mutation_job_cap_under_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)
    _jobs(mutated)[RECORD_E2E_JOB]["timeout-minutes"] = step["timeout-minutes"]
    with pytest.raises(AssertionError, match="must sit above"):
        test_record_lane_timeout_covers_its_waits(mutated)


def test_record_lane_runs_on_every_pull_request_without_a_key(workflow: dict[str, Any]) -> None:
    """Model-free: no label, no actor guard, no gateway key."""
    job = _jobs(workflow)[RECORD_E2E_JOB]
    assert job.get("if") == "github.event_name == 'pull_request'", job.get("if")
    assert not _job_declares_secret(workflow, RECORD_E2E_JOB, SECRET_TOKEN), (
        f"'{RECORD_E2E_JOB}' declares the gateway key; it reaches no model"
    )


def test_record_lane_runs_on_every_pull_request_without_a_key__mutation_adds_a_label() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[RECORD_E2E_JOB]
    job["if"] = f"{job['if']} && {FULL_CI_LABEL_CLAUSE}"
    with pytest.raises(AssertionError):
        test_record_lane_runs_on_every_pull_request_without_a_key(mutated)


def test_record_lane_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    reports = _JUNIT_RE.findall(
        _find_named_step(workflow, RECORD_E2E_JOB, RECORD_E2E_RUN_STEP)["run"]
    )
    assert len(reports) == 1, (
        f"'{RECORD_E2E_RUN_STEP}' must write exactly one --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, RECORD_E2E_JOB, RECORD_E2E_SKIP_GATE_STEP)["run"]
    assert reports[0] in gate, f"'{RECORD_E2E_SKIP_GATE_STEP}' never reads: {reports[0]}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_record_lane_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, RECORD_E2E_JOB, RECORD_E2E_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_record_lane_fails_on_any_skipped_test(mutated)


def _e2e_files_the_shared_lane_sweeps(wf: dict[str, Any]) -> set[str]:
    """The ``tests/e2e/**/test_*.py`` paths the shared lane's directory sweep
    still collects, repo-relative and POSIX.

    Derived from disk rather than from a list, so a module added to the tree is
    in this set the moment it exists — the sweep names a directory, and what
    leaves it leaves by an ``--ignore``.
    """
    run_text = _find_named_step(wf, E2E_TESTS_JOB, "Run E2E tests")["run"]
    ignored = set(re.findall(r"--ignore=(\S+)", run_text))
    repo_root = CI_YML.parents[2]
    swept = {path.relative_to(repo_root).as_posix() for path in _e2e_test_sources()}
    return swept - ignored


def test_shared_lane_hands_over_every_model_free_file(workflow: dict[str, Any]) -> None:
    """The handover is one ``--ignore`` per file: the shared lane names a
    directory, so a file only stops being swept when it is named."""
    missing = _run_step_ignores_all(workflow, list(MODEL_FREE_TEST_FILES))
    assert missing == [], (
        f"model-free e2e file(s) still swept by the '{E2E_TESTS_JOB}' lane: {missing} — "
        f"they would run in both lanes, paying the label for work that needs no label"
    )


def test_shared_lane_hands_over_every_model_free_file__mutation_drops_the_record_smoke_ignore() -> (
    None
):
    smoke = RECORD_E2E_FILE
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], smoke)
    others = [f for f in MODEL_FREE_TEST_FILES if f != smoke]
    assert _run_step_ignores_all(mutated, others) == []  # every other file survives
    with pytest.raises(AssertionError, match="still swept"):
        test_shared_lane_hands_over_every_model_free_file(mutated)


def test_shared_lane_hands_over_every_model_free_file__mutation_drops_the_terminal_auth_ignore() -> (
    None
):
    """The neighbouring ``--ignore`` names ``test_terminal_auth_multiuser_e2e.py``,
    which does not contain this file's path as a substring — so exactly one
    line goes and the mutation stays one file wide."""
    terminal_auth = "tests/e2e/web_terminals/test_terminal_auth_e2e.py"
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], terminal_auth)
    others = [f for f in MODEL_FREE_TEST_FILES if f != terminal_auth]
    assert _run_step_ignores_all(mutated, others) == []  # the other nine survive
    with pytest.raises(AssertionError, match="still swept"):
        test_shared_lane_hands_over_every_model_free_file(mutated)


def test_no_end_to_end_file_runs_in_both_lanes(workflow: dict[str, Any]) -> None:
    """The two lanes divide one directory, and the division is the claim — a
    file in both is paid for twice, and the set is computed from what is on
    disk rather than from the lists, so it also sees a file that returns to the
    sweep under a path neither list spells the same way."""
    swept = _e2e_files_the_shared_lane_sweeps(workflow)
    assert swept, (
        f"the '{E2E_TESTS_JOB}' sweep was computed as empty — the discovery broke, and "
        f"an empty set makes the disjointness below vacuous"
    )
    both = sorted(set(MODEL_FREE_TEST_FILES) & swept)
    assert both == [], (
        f"file(s) in both end-to-end lanes: {both} — each is collected twice, once "
        f"behind the label and once without it"
    )


def test_no_end_to_end_file_runs_in_both_lanes__mutation_returns_a_file_to_the_sweep() -> None:
    render = "tests/e2e/web_terminals/test_scaffold_render_roundtrip.py"
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], render)
    assert set(MODEL_FREE_TEST_FILES) & _e2e_files_the_shared_lane_sweeps(mutated) == {render}
    with pytest.raises(AssertionError, match="in both end-to-end lanes"):
        test_no_end_to_end_file_runs_in_both_lanes(mutated)


# ---------------------------------------------------------------------------
# als-apg endpoint-override drift guard
# ---------------------------------------------------------------------------


def _als_apg_probe_steps(wf: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every step that preflight-probes the als-apg gateway, as
    ``(job, step name, run text)``.

    Discovery keys on what a probe *does* — POST an authenticated request to
    ``/v1/messages`` — not on whether it happens to define
    ``ALS_APG_PROBE_BASE``. That distinction is the whole point: a newly added
    probe that hardcodes its URL defines no such variable, so keying on the
    variable would skip exactly the step the guard exists to catch.
    """
    found: list[tuple[str, str, str]] = []
    for job_name, job in _jobs(wf).items():
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str) and PROBE_TARGET_PATH in run and PROBE_KEY_ENV in run:
                found.append((job_name, step.get("name", "<unnamed>"), run))
    return found


def test_workflow_exports_the_als_apg_base_url_override(workflow: dict[str, Any]) -> None:
    """The workflow-level ``env:`` block is what lets a single repository
    variable retarget every LLM-touching job at once. Without it the probes
    below would each need their own endpoint and the override would silently
    do nothing — green locally, wrong in CI.

    The exported value is pinned exactly, not merely searched for the
    variable: the endpoint is repository data, and a ``|| 'https://…'`` tail
    added beside the variable would point every fork that has not set it at
    somebody else's gateway while still reading ``vars.``. ci.yml is outside
    the roots the shipped-literal guard walks, so this assertion is the only
    thing standing between that tail and a merge."""
    env = workflow.get("env") or {}
    exported = env.get(ALS_APG_BASE_URL_ENV)
    assert exported == "${{ vars." + ALS_APG_BASE_URL_ENV + " }}", (
        f"ci.yml must export {ALS_APG_BASE_URL_ENV} from the repository variable "
        f"at workflow level, with no fallback behind it; found {exported!r}"
    )


def test_workflow_exports_the_als_apg_base_url_override__mutation_drops_export() -> None:
    mutated = copy.deepcopy(_load_workflow())
    (mutated.get("env") or {}).pop(ALS_APG_BASE_URL_ENV, None)
    with pytest.raises(AssertionError, match="must export"):
        test_workflow_exports_the_als_apg_base_url_override(mutated)


def test_workflow_exports_the_als_apg_base_url_override__mutation_adds_a_fallback() -> None:
    """A fallback behind the variable is the regression that matters: it reads
    ``vars.`` and still bakes an endpoint into every fork of the workflow."""
    mutated = copy.deepcopy(_load_workflow())
    (mutated.get("env") or {})[ALS_APG_BASE_URL_ENV] = (
        "${{ vars." + ALS_APG_BASE_URL_ENV + " || 'https://gateway.example.org' }}"
    )
    with pytest.raises(AssertionError, match="no fallback"):
        test_workflow_exports_the_als_apg_base_url_override(mutated)


def _e2e_provider_overrides(wf: dict[str, Any]) -> list[str]:
    """Every job or step that redefines the end-to-end provider, as labels."""
    found: list[str] = []
    for job_name, job in _jobs(wf).items():
        if E2E_PROVIDER_ENV in (job.get("env") or {}):
            found.append(f"job {job_name}")
        for step in job.get("steps") or []:
            if E2E_PROVIDER_ENV in (step.get("env") or {}):
                found.append(f"step {job_name} / {step.get('name', '<unnamed>')}")
    return found


def test_the_workflow_names_the_end_to_end_provider(workflow: dict[str, Any]) -> None:
    """Every lane that collects ``tests/e2e/`` states what it builds with.

    There is no constant behind the variable any more — a run that names no
    provider is refused when it selects a test not marked model_free — so the
    name has to be stated
    somewhere, and workflow level is the one place that covers every lane at
    once. A fork driving its own gateway changes this line and nothing else,
    which is also why the value is a plain name rather than an expression: a
    ``vars.`` lookup a fork has not set would resolve to empty and refuse every
    e2e lane in the fork.

    The second half is the one that rots quietly: a job or step that sets the
    variable itself would build against a different gateway than the one whose
    key the lane holds, and the failure would read as a credential problem."""
    named = (workflow.get("env") or {}).get(E2E_PROVIDER_ENV)
    assert isinstance(named, str) and named.strip(), (
        f"ci.yml must name {E2E_PROVIDER_ENV} in its workflow-level env block; found {named!r}"
    )
    assert "${{" not in named, (
        f"{E2E_PROVIDER_ENV} must be a plain provider name, not an expression; found {named!r}"
    )
    overrides = _e2e_provider_overrides(workflow)
    assert not overrides, (
        f"no job or step may override {E2E_PROVIDER_ENV} — one provider for every e2e "
        f"lane; found: {', '.join(overrides)}"
    )


def test_the_workflow_names_the_end_to_end_provider__mutation_drops_the_name() -> None:
    mutated = copy.deepcopy(_load_workflow())
    (mutated.get("env") or {}).pop(E2E_PROVIDER_ENV, None)
    with pytest.raises(AssertionError, match="must name"):
        test_the_workflow_names_the_end_to_end_provider(mutated)


def test_the_workflow_names_the_end_to_end_provider__mutation_overrides_it_in_a_job() -> None:
    """The dangerous half: the workflow-level name survives, so the block still
    reads right, while one lane quietly builds somewhere else."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[E2E_TESTS_JOB]
    job["env"] = {**(job.get("env") or {}), E2E_PROVIDER_ENV: "somewhere-else"}
    with pytest.raises(AssertionError, match="may override"):
        test_the_workflow_names_the_end_to_end_provider(mutated)


def test_every_als_apg_probe_honors_the_base_url_override(workflow: dict[str, Any]) -> None:
    """Each probe must derive its base from ``$ALS_APG_BASE_URL`` (a baked-in
    default after ``:-`` is fine) and must aim curl at that derived variable.

    Both halves are load-bearing and fail independently: a probe can read the
    override into ``ALS_APG_PROBE_BASE`` and then still curl a literal URL,
    which is how a hardcoded endpoint survived review once already. Such a
    lane fails against a perfectly healthy gateway, and reads as an outage."""
    probes = _als_apg_probe_steps(workflow)
    assert len(probes) >= EXPECTED_MIN_ALS_APG_PROBES, (
        f"expected at least {EXPECTED_MIN_ALS_APG_PROBES} als-apg probe steps in ci.yml, "
        f"found {len(probes)} — discovery has drifted and this guard is now vacuous"
    )
    for job_name, step_name, run in probes:
        where = f"{job_name} / {step_name}"
        assert PROBE_BASE_ASSIGNMENT.search(run), (
            f"{where}: als-apg probe must derive its base from "
            f'${{{ALS_APG_BASE_URL_ENV}}} (e.g. ALS_APG_PROBE_BASE="${{{ALS_APG_BASE_URL_ENV}'
            ':-https://default}"), so the repository variable can retarget it'
        )
        target_lines = [line for line in run.splitlines() if PROBE_TARGET_PATH in line]
        assert target_lines, f"{where}: probe target line vanished"
        for line in target_lines:
            assert f"${PROBE_BASE_VAR}" in line, (
                f"{where}: probe must curl $"
                + PROBE_BASE_VAR
                + f", not a literal URL: {line.strip()}"
            )


def test_every_als_apg_probe_honors_the_base_url_override__mutation_hardcodes_base() -> None:
    """The original bug's exact shape: the assignment stops reading the
    override and pins a URL instead."""
    mutated = copy.deepcopy(_load_workflow())
    job_name, step_name, _ = _als_apg_probe_steps(mutated)[0]
    step = _find_named_step(mutated, job_name, step_name)
    step["run"] = PROBE_BASE_ASSIGNMENT.sub(
        f'{PROBE_BASE_VAR}="https://hardcoded.example"', step["run"], count=1
    )
    with pytest.raises(AssertionError, match="must derive its base"):
        test_every_als_apg_probe_honors_the_base_url_override(mutated)


def test_every_als_apg_probe_honors_the_base_url_override__mutation_curls_literal_url() -> None:
    """The subtler half: the override is still read into the variable — so a
    reader skimming the assignment sees the right thing — while curl ignores
    it and hits a literal endpoint."""
    mutated = copy.deepcopy(_load_workflow())
    job_name, step_name, _ = _als_apg_probe_steps(mutated)[0]
    step = _find_named_step(mutated, job_name, step_name)
    step["run"] = step["run"].replace(
        f'"${PROBE_BASE_VAR}{PROBE_TARGET_PATH}"', f'"https://hardcoded.example{PROBE_TARGET_PATH}"'
    )
    assert PROBE_BASE_ASSIGNMENT.search(step["run"]), "the assignment must survive this mutation"
    with pytest.raises(AssertionError, match="not a literal URL"):
        test_every_als_apg_probe_honors_the_base_url_override(mutated)


# ---------------------------------------------------------------------------
# YAML 1.1 `on:` gotcha regression guard
# ---------------------------------------------------------------------------


def test_workflow_on_key_parses_to_bool_true_not_string(workflow: dict[str, Any]) -> None:
    """Documents the YAML 1.1 footgun this module deliberately avoids: the
    bare `on:` trigger key parses to the Python bool True, not the string
    'on'. Indexing workflow['on'] would KeyError; every fixture/helper above
    only ever indexes workflow['jobs']."""
    assert True in workflow
    assert "on" not in workflow


# ---------------------------------------------------------------------------
# (h) multi-user login-flow browser test: named in the lane's file list, which
# is the only way it is ever collected
# ---------------------------------------------------------------------------

BROWSER_JOB = "visual-theming"
BROWSER_RUN_STEP = "Run behavioral theming suite"
AUTH_BROWSER_TEST_FILE = "tests/interfaces/web_terminal/test_auth_login_browser.py"


#: What makes a step of the browser job one that runs suites. The lane runs
#: its suites in more than one step — the visual-regression step stands apart
#: so its baseline upload and commit steps can follow it — so a suite named by
#: any of them is run by the lane, and matching the invocation rather than
#: listing step names keeps a further step from being named by the lane and
#: invisible to every guard below.
_BROWSER_PYTEST_INVOCATION = "uv run pytest "


def _browser_lane_files(wf: dict[str, Any]) -> str:
    return "\n".join(
        step["run"]
        for step in _jobs(wf)[BROWSER_JOB]["steps"]
        if _BROWSER_PYTEST_INVOCATION in step.get("run", "")
    )


# ---------------------------------------------------------------------------
# (h0) the browser lane carries no condition: a skipped step reports as a step
# success and a job whose steps all skipped reports as a job success, and this
# lane sits in the merge gate's always-on tier
# ---------------------------------------------------------------------------

#: Any `if:` that reads another step's output. A lane whose steps can be
#: switched off by an earlier step is a lane that can report success having
#: proven nothing.
_STEP_OUTPUT_CONDITION = re.compile(r"steps\.[A-Za-z0-9_-]+\.outputs")

#: The one step a step output may switch off: saving the Playwright cache. It
#: proves nothing and installs nothing, and skipping it on an exact cache hit
#: only spares re-uploading what the cache already holds.
_CACHE_SAVE_ACTION = "actions/cache/save@"


def _browser_steps_gated_on_outputs(wf: dict[str, Any]) -> list[str | None]:
    return [
        step.get("name")
        for step in _jobs(wf)[BROWSER_JOB]["steps"]
        if _STEP_OUTPUT_CONDITION.search(str(step.get("if", "")))
        and not str(step.get("uses", "")).startswith(_CACHE_SAVE_ACTION)
    ]


def test_the_browser_lane_runs_unconditionally(workflow: dict[str, Any]) -> None:
    """Three ways this lane could stop running, all of them silent.

    A job-level ``if:`` skips the job; a paths-filter step lets a later step
    decide it has nothing to do; an ``if:`` on a step reading that step's
    output does the same one step at a time. Each reports as success, and
    ``all-checks-passed`` requires this lane to be ``success``, so any of the
    three turns the browser proof off at exactly the moment a reviewer is
    reading a green check."""
    job = _jobs(workflow)[BROWSER_JOB]
    assert "if" not in job, f"the '{BROWSER_JOB}' lane carries a job-level condition"
    filters = [
        step.get("name")
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("dorny/paths-filter")
    ]
    assert filters == [], f"the '{BROWSER_JOB}' lane computes a paths filter: {filters}"
    gated = _browser_steps_gated_on_outputs(workflow)
    assert gated == [], (
        f"steps of the '{BROWSER_JOB}' lane gated on another step's output, which would "
        f"report the lane green having run nothing: {gated}"
    )


def test_the_browser_lane_runs_unconditionally__mutation_adds_a_filter_step() -> None:
    """Reintroducing the filter step must be reported."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[BROWSER_JOB]
    job["steps"].insert(
        1,
        {
            "name": "Detect design-system changes",
            "id": "changes",
            "uses": "dorny/paths-filter@v4",
            "with": {"filters": "theming:\n  - 'src/osprey/interfaces/**'\n"},
        },
    )
    filters = [
        step.get("name")
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("dorny/paths-filter")
    ]
    assert filters == ["Detect design-system changes"]


def test_the_browser_lane_runs_unconditionally__mutation_gates_the_run_step() -> None:
    """Gating the run step on a step output must be reported."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, BROWSER_JOB, BROWSER_RUN_STEP)["if"] = (
        "steps.changes.outputs.theming == 'true'"
    )
    assert _browser_steps_gated_on_outputs(mutated) == [BROWSER_RUN_STEP]


def test_the_browser_lane_runs_unconditionally__mutation_gates_an_install_on_the_cache_hit() -> (
    None
):
    """The cache-save exemption is by action, not by condition: an install
    skipped on a cache hit must still be reported."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, BROWSER_JOB, "Install Playwright webkit")["if"] = (
        "steps.playwright-cache.outputs.cache-hit != 'true'"
    )
    assert _browser_steps_gated_on_outputs(mutated) == ["Install Playwright webkit"]


COMBOBOX_BROWSER_TEST_FILE = "tests/interfaces/bluesky_web/test_channel_combobox_browser.py"
HISTORY_BROWSER_TEST_FILE = "tests/interfaces/history/test_history_page_browser.py"
LATTICE_BROWSER_TEST_FILE = "tests/interfaces/test_lattice_dashboard_browser.py"

#: The browser suites (h) and its siblings each pin by name: the login flow, the
#: channel combobox, the history page and the lattice dashboard.
_NAMED_BROWSER_TEST_FILES = [
    AUTH_BROWSER_TEST_FILE,
    COMBOBOX_BROWSER_TEST_FILE,
    HISTORY_BROWSER_TEST_FILE,
    LATTICE_BROWSER_TEST_FILE,
]


@pytest.mark.parametrize("test_file", _NAMED_BROWSER_TEST_FILES)
def test_browser_test_runs_in_the_browser_lane(workflow: dict[str, Any], test_file: str) -> None:
    """The suite has to be NAMED in the lane's pytest invocation.

    The lane runs an explicit file list, not a directory glob, so a new
    browser-marked file that nobody adds here is never collected anywhere: the
    unit lane skips it (no chromium), and this lane never hears of it. It
    reports green by running nothing at all."""
    assert test_file in _browser_lane_files(workflow)


@pytest.mark.parametrize("test_file", _NAMED_BROWSER_TEST_FILES)
def test_browser_test_runs_in_the_browser_lane__mutation_drops_the_file(test_file: str) -> None:
    """Removing the file from the invocation must fail the guard — the exact
    vacuous-green shape above."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BROWSER_JOB, BROWSER_RUN_STEP)
    step["run"] = step["run"].replace(f"{test_file} \\\n", "")
    assert test_file not in _browser_lane_files(mutated)


# ---------------------------------------------------------------------------
# (h3) every browser suite on disk: (h) and (h2) each pin one file by name,
# which is exactly how the next one gets forgotten. The lane's explicit file
# list is checked against the tree instead — a module that marks itself
# ``browser`` is skipped by the unit lane for want of a chromium binary, so
# one this lane does not name is collected nowhere at all.
# ---------------------------------------------------------------------------

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
#: Workspace packages' own suites, which the root ``testpaths`` lists beside
#: ``tests``. Each is one entry of the unit shards' partition, like a top-level
#: ``tests/<dir>``.
PACKAGE_TEST_ROOTS = ("packages/osprey-connectors/tests",)


#: The marker a module applies to itself to say it drives a browser. It is
#: what makes the unit lane skip the module — there is no chromium binary
#: there — so it is also what decides which lane has to name it.
BROWSER_MARKER = "browser"


def _module_marks(source: str) -> set[str]:
    """The pytest marker names a module applies to every test it defines.

    Read with ``ast`` rather than by importing: a browser suite pulls in
    playwright and starts the interface server it drives, neither of which a
    workflow-wiring check has any business doing. Both ``pytestmark = [...]``
    and the annotated spelling are read, and a bare ``pytestmark =
    pytest.mark.browser`` is one mark rather than none.
    """
    marks: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if value is None:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
            continue
        for sub in ast.walk(value):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "mark"
                and isinstance(sub.value.value, ast.Name)
                and sub.value.value.id == "pytest"
            ):
                marks.add(sub.attr)
    return marks


def _browser_suites_on_disk() -> list[str]:
    """A fresh list of ``_browser_suites_on_disk_once``, safe for the caller to edit."""
    return list(_browser_suites_on_disk_once())


@functools.cache
def _browser_suites_on_disk_once() -> tuple[str, ...]:
    """Every test module that marks itself ``browser``, repo-relative, POSIX-spelled.

    Discovery follows the marker, never the filename: the marker is the
    module's own declaration that it needs a browser, and a suite is free to
    be called anything. A filename convention accounts only for the modules
    that follow it and passes over every other in silence.
    """
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path, source in _test_tree_sources().items()
            if "__pycache__" not in path.parts and BROWSER_MARKER in _module_marks(source)
        )
    )


def _browser_lane_named_files(wf: dict[str, Any]) -> set[str]:
    """The pytest file arguments of the lane's run step, one token at a time —
    not a substring search, so ``test_panel_browser.py`` is never satisfied by
    ``test_panels_browser.py``."""
    return {tok for tok in _browser_lane_files(wf).split() if tok.endswith(".py")}


#: The number of browser-marked modules in the tree. A discovery that reads
#: the marker wrongly still returns a plausible-looking list, and every guard
#: below it would then pass over the difference, so the floor is worth what
#: the distance between it and the count is worth: a suite added raises this
#: and a suite deliberately deleted lowers it, both by hand.
BROWSER_SUITE_FLOOR = 40


def test_browser_suite_discovery_has_a_floor() -> None:
    """A discovery that finds nothing, or finds a fraction, would make the
    guards below vacuous rather than red."""
    found = _browser_suites_on_disk()
    assert AUTH_BROWSER_TEST_FILE in found
    assert COMBOBOX_BROWSER_TEST_FILE in found
    assert len(found) >= BROWSER_SUITE_FLOOR, found


def test_browser_suite_discovery_covers_every_suite_named_like_one() -> None:
    """A module called ``test_*_browser.py`` says the same thing its marker
    says, so the marker has to find all of them. This is the floor that
    maintains itself: the naming convention is a subset of the marker, and a
    walk that silently stops reading one of the two spellings of
    ``pytestmark`` fails here without anyone remembering to raise a count."""
    found = set(_browser_suites_on_disk())
    named_like_one = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in _test_tree_paths()
        if path.match("test_*_browser.py") and "__pycache__" not in path.parts
    )
    missing = [suite for suite in named_like_one if suite not in found]
    assert missing == [], f"browser suites the marker discovery does not find: {missing}"


def test_the_marker_discovery_reads_the_marker_not_the_name() -> None:
    """The helper the guards rest on, exercised directly: a module is a
    browser suite because of what it declares, and a module that declares
    nothing is not one however it is named."""
    assert BROWSER_MARKER in _module_marks(
        "import pytest\n\npytestmark = [pytest.mark.browser, pytest.mark.slow]\n"
    )
    assert BROWSER_MARKER in _module_marks("import pytest\n\npytestmark = pytest.mark.browser\n")
    assert BROWSER_MARKER not in _module_marks("import pytest\n\npytestmark = [pytest.mark.slow]\n")
    assert _module_marks("x = 1\n") == set()


def test_every_browser_suite_on_disk_is_named_in_the_browser_lane(
    workflow: dict[str, Any],
) -> None:
    named = _browser_lane_named_files(workflow)
    missing = [suite for suite in _browser_suites_on_disk() if suite not in named]
    assert missing == [], (
        f"browser suites on disk that the '{BROWSER_JOB}' lane never runs (the unit lane "
        f"skips them without chromium, so nothing collects them anywhere): {missing}"
    )


def test_every_browser_suite_on_disk_is_named__mutation_drops_one_suite() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BROWSER_JOB, BROWSER_RUN_STEP)
    step["run"] = step["run"].replace(f"{AUTH_BROWSER_TEST_FILE} \\\n", "")
    named = _browser_lane_named_files(mutated)
    assert [s for s in _browser_suites_on_disk() if s not in named] == [AUTH_BROWSER_TEST_FILE]


# ---------------------------------------------------------------------------
# (h4) every browser engine a fixture launches is installed by the lane: a
# fixture whose engine binary is absent skips every test that asks for it, and
# a lane whose tests all skipped reports green having proved nothing.
# ---------------------------------------------------------------------------

_INTERFACES_CONFTEST = TESTS_ROOT / "interfaces" / "conftest.py"


def _browser_engines() -> tuple[str, ...]:
    """``BROWSER_ENGINES`` from the interfaces conftest, read with ``ast``.

    Not imported, for the reason :func:`_module_marks` gives: the conftest pulls
    in playwright and the interface servers, which a workflow-wiring check has
    no business loading.
    """
    for node in ast.parse(_INTERFACES_CONFTEST.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "BROWSER_ENGINES" for t in targets):
            continue
        assert isinstance(value, ast.Tuple), "BROWSER_ENGINES is not a tuple literal"
        return tuple(
            elt.value
            for elt in value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        )
    raise AssertionError(f"no BROWSER_ENGINES in {_INTERFACES_CONFTEST}")


def _engines_the_lane_installs(wf: dict[str, Any]) -> set[str]:
    """Every engine named after ``playwright install`` in the browser job's steps.

    Flags (``--with-deps``) are dropped, so an option is never read as an engine.
    """
    engines: set[str] = set()
    for step in _jobs(wf)[BROWSER_JOB].get("steps", []):
        run = step.get("run")
        if not isinstance(run, str):
            continue
        for match in re.finditer(r"playwright install([^\n;&|]*)", run):
            engines.update(tok for tok in match.group(1).split() if not tok.startswith("-"))
    return engines


def test_browser_engine_discovery_finds_both_engines() -> None:
    """The floor: a mis-parse that found nothing would make the next guard vacuous."""
    assert _browser_engines() == ("chromium", "webkit")


def test_every_engine_a_fixture_launches_is_installed_by_the_browser_lane(
    workflow: dict[str, Any],
) -> None:
    missing = set(_browser_engines()) - _engines_the_lane_installs(workflow)
    assert missing == set(), (
        f"browser engines the '{BROWSER_JOB}' lane never installs: {sorted(missing)}. "
        "An engine the lane does not install skips every test that asks for it, "
        "and the lane still reports green."
    )


def test_every_engine_is_installed__mutation_drops_the_webkit_install() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[BROWSER_JOB]
    job["steps"] = [s for s in job["steps"] if s.get("name") != "Install Playwright webkit"]
    assert set(_browser_engines()) - _engines_the_lane_installs(mutated) == {"webkit"}


# ---------------------------------------------------------------------------
# (h4) the browser lane's installs are cached and capped: their `--with-deps`
# debs come from an Ubuntu mirror that some days crawls, and an uncapped,
# uncached install then eats the whole job timeout before a test runs
# ---------------------------------------------------------------------------

_PLAYWRIGHT_INSTALL_STEPS = ("Install Playwright chromium", "Install Playwright webkit")
_PLAYWRIGHT_INSTALL_CAP_MINUTES = 10
_APT_ARCHIVES_SETTING = re.compile(r'Dir::Cache::Archives\s+"\$HOME/([^"]+)"')


def _browser_cache_problems(wf: dict[str, Any]) -> list[str]:
    """Every way the Playwright cache fails to cover the installs.

    The restore has to come before both installs and the save after them but
    before any suite, so a failing suite still leaves the next run a warm
    cache; both have to name the same key and paths; and the paths have to
    hold the browsers and the directory apt is told to keep its debs in.
    Between the installs and the save, that directory has to be handed back
    to the runner user: apt leaves it root-owned, and a save that cannot read
    it only warns, so the cache would silently never be written.
    """
    steps = _jobs(wf)[BROWSER_JOB]["steps"]

    def where(pred: Callable[[dict[str, Any]], bool]) -> list[int]:
        return [i for i, s in enumerate(steps) if pred(s)]

    restores = where(lambda s: str(s.get("uses", "")).startswith("actions/cache/restore@"))
    saves = where(lambda s: str(s.get("uses", "")).startswith(_CACHE_SAVE_ACTION))
    installs = where(lambda s: s.get("name") in _PLAYWRIGHT_INSTALL_STEPS)
    suites = where(lambda s: _BROWSER_PYTEST_INVOCATION in s.get("run", ""))
    archives = [
        m.group(1) for s in steps for m in _APT_ARCHIVES_SETTING.finditer(str(s.get("run", "")))
    ]
    if len(restores) != 1 or len(saves) != 1 or len(archives) != 1:
        return [f"restores={restores} saves={saves} apt archive dirs={archives}"]
    restore, save = steps[restores[0]], steps[saves[0]]
    problems = []
    if not restores[0] < min(installs) or not max(installs) < saves[0] < min(suites):
        problems.append("order")
    if restore["with"]["key"] != save["with"]["key"]:
        problems.append("key")
    paths = set(restore["with"]["path"].split())
    if paths != set(save["with"]["path"].split()):
        problems.append("paths")
    if not {"~/.cache/ms-playwright", f"~/{archives[0]}"} <= paths:
        problems.append("coverage")
    handbacks = where(
        lambda s: "chown" in s.get("run", "") and f"~/{archives[0]}" in s.get("run", "")
    )
    if not any(max(installs) < i < saves[0] for i in handbacks):
        problems.append("ownership")
    return problems


def test_the_playwright_cache_brackets_the_installs(workflow: dict[str, Any]) -> None:
    assert _browser_cache_problems(workflow) == []


def test_the_playwright_cache_brackets_the_installs__mutation_saves_after_the_suites() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[BROWSER_JOB]["steps"]
    save = next(s for s in steps if str(s.get("uses", "")).startswith(_CACHE_SAVE_ACTION))
    steps.remove(save)
    steps.append(save)
    assert _browser_cache_problems(mutated) == ["order"]


def test_the_playwright_cache_brackets_the_installs__mutation_drops_the_apt_archive() -> None:
    mutated = copy.deepcopy(_load_workflow())
    for step in _jobs(mutated)[BROWSER_JOB]["steps"]:
        if str(step.get("uses", "")).startswith("actions/cache/"):
            step["with"]["path"] = "~/.cache/ms-playwright\n"
    assert _browser_cache_problems(mutated) == ["coverage"]


def test_the_playwright_cache_brackets_the_installs__mutation_drops_the_handback() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[BROWSER_JOB]
    job["steps"] = [s for s in job["steps"] if "chown" not in s.get("run", "")]
    assert _browser_cache_problems(mutated) == ["ownership"]


def test_the_playwright_installs_carry_their_own_cap(workflow: dict[str, Any]) -> None:
    """A stalled mirror has to fail the install in minutes, not after the job's
    whole timeout, so the log points at the install rather than the job."""
    job = _jobs(workflow)[BROWSER_JOB]
    caps = {
        name: _find_named_step(workflow, BROWSER_JOB, name).get("timeout-minutes")
        for name in _PLAYWRIGHT_INSTALL_STEPS
    }
    assert all(
        isinstance(cap, int) and cap <= _PLAYWRIGHT_INSTALL_CAP_MINUTES for cap in caps.values()
    ), caps
    assert sum(c for c in caps.values() if isinstance(c, int)) < job["timeout-minutes"], (
        caps,
        job["timeout-minutes"],
    )


def test_the_playwright_installs_carry_their_own_cap__mutation_drops_one() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, BROWSER_JOB, "Install Playwright webkit")["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_the_playwright_installs_carry_their_own_cap(mutated)


def test_every_browser_engine_has_its_fixture() -> None:
    functions = {
        node.name
        for node in ast.parse(_INTERFACES_CONFTEST.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    missing = [e for e in _browser_engines() if f"{e}_browser" not in functions]
    assert missing == [], f"engines in BROWSER_ENGINES with no <engine>_browser fixture: {missing}"


# ---------------------------------------------------------------------------
# (i) bluesky-queue-e2e: the queue stack's own lane, secret-free
# ---------------------------------------------------------------------------


def test_bluesky_queue_job_exists(workflow: dict[str, Any]) -> None:
    """``test_bluesky_queue_e2e.py`` is ``--ignore``'d by the shared e2e-tests
    lane (it deploys the whole queue stack twice), so this dedicated job is the
    only place it runs at all — without it the file is collected nowhere. The
    job existing is not enough: one of its steps has to actually name the
    module, or the lane reports green having run nothing."""
    assert QUEUE_JOB in _jobs(workflow)
    assert QUEUE_TEST_FILE in json.dumps(_jobs(workflow)[QUEUE_JOB])


def test_bluesky_queue_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][QUEUE_JOB]
    with pytest.raises(AssertionError):
        test_bluesky_queue_job_exists(mutated)


def test_bluesky_queue_job_exists__mutation_drops_the_run_step() -> None:
    """The vacuous-green half: the job survives, but nothing in it names the
    e2e module any more."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[QUEUE_JOB]["steps"]
    kept = [step for step in steps if QUEUE_TEST_FILE not in json.dumps(step)]
    assert len(kept) == len(steps) - 1, "expected exactly one step to name the e2e module"
    _jobs(mutated)[QUEUE_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_bluesky_queue_job_exists(mutated)


def test_e2e_lane_ignores_bluesky_queue(workflow: dict[str, Any]) -> None:
    """The guards above catch "runs nowhere"; this one catches "runs twice".

    The module carries no ``dockerbuild`` marker, so the blanket
    marker->--ignore guard does not reach it and this bespoke check is the
    only thing holding the ``--ignore`` in place. Drop that line and the whole
    queueserver stack — bridge, Redis, Tiled, VA, bluesky-web sidecar, on fixed
    host ports — boots a second time inside the shared lane's ``-n 4`` run,
    concurrently with its own job. Nothing would fail; it would just get slow
    and flaky in a way nobody could attribute."""
    assert _run_step_ignores_all(workflow, [QUEUE_TEST_FILE]) == []


def test_e2e_lane_ignores_bluesky_queue__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], QUEUE_TEST_FILE)
    assert _run_step_ignores_all(mutated, [ORM_TEST_FILE]) == []  # others survive
    assert _run_step_ignores_all(mutated, [QUEUE_TEST_FILE]) == [QUEUE_TEST_FILE]


# ---------------------------------------------------------------------------
# (k) scan-agentic-e2e: the agent-driven plan lane, secret-gated
# ---------------------------------------------------------------------------


def test_scan_agentic_job_exists(workflow: dict[str, Any]) -> None:
    """``test_plan_stack_agentic.py`` is ``--ignore``'d by the shared e2e-tests
    lane (its two live tests deploy the VA/bridge/queue stack on fixed host
    ports), so this dedicated job is the only place any of it runs — the
    offline floor and judge checks included, since they live in the same
    module. The job existing is not enough: one of its steps has to name the
    module, or the lane reports green having run nothing.

    Unlike the queue lane above, nothing bespoke holds that ``--ignore`` in
    place: the module's live tests carry the ``dockerbuild`` marker, so the
    blanket guard in section (e)
    (``test_every_dockerbuild_marked_file_is_ignored_in_e2e_lane``) is what
    pins it. This check is the other half of that pair — (e) proves the module
    left the shared lane, and only this one proves it arrived somewhere."""
    assert SCAN_AGENTIC_JOB in _jobs(workflow)
    assert SCAN_AGENTIC_TEST_FILE in json.dumps(_jobs(workflow)[SCAN_AGENTIC_JOB])


def test_scan_agentic_job_exists__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][SCAN_AGENTIC_JOB]
    with pytest.raises(AssertionError):
        test_scan_agentic_job_exists(mutated)


def test_scan_agentic_job_exists__mutation_drops_the_run_step() -> None:
    """The vacuous-green half: the job survives, but nothing in it names the
    e2e module any more."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[SCAN_AGENTIC_JOB]["steps"]
    kept = [step for step in steps if SCAN_AGENTIC_TEST_FILE not in json.dumps(step)]
    assert len(kept) == len(steps) - 1, "expected exactly one step to name the e2e module"
    _jobs(mutated)[SCAN_AGENTIC_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_scan_agentic_job_exists(mutated)


def test_scan_agentic_job_declares_llm_secret(workflow: dict[str, Any]) -> None:
    """Both live tests drive a real agent and then a real judge, and both are
    ``requires_als_apg``-marked — without the secret they skip, so a job
    missing it would green-wash the lane down to its offline checks."""
    assert _job_declares_secret(workflow, SCAN_AGENTIC_JOB, SECRET_TOKEN)


def test_scan_agentic_job_declares_llm_secret__mutation_strips_secret() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][SCAN_AGENTIC_JOB] = json.loads(
        json.dumps(mutated["jobs"][SCAN_AGENTIC_JOB]).replace("secrets.ALS_APG_API_KEY", "")
    )
    with pytest.raises(AssertionError):
        assert _job_declares_secret(mutated, SCAN_AGENTIC_JOB, SECRET_TOKEN)


def _junit_reports_written_by(steps: list[dict[str, Any]]) -> list[str]:
    """Report paths the given steps write, in step order."""
    return [r for step in steps for r in _JUNIT_RE.findall(step["run"])]


def _scan_agentic_pytest_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """This job's pytest invocations. Matched on the full ``uv run pytest``
    invocation for the same reason as the gchat helper: the zero-skip gate step
    below runs ``uv run python`` and merely names pytest in its comment, and
    counting that as a pytest step would demand a junit report it never
    writes."""
    return [s for s in _jobs(wf)[SCAN_AGENTIC_JOB]["steps"] if "uv run pytest " in s.get("run", "")]


def test_scan_agentic_job_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """The secret check above only proves the key is present; this proves the
    lane cannot green while its expensive half quietly does nothing.

    Every skip this module can take is a CI misconfiguration rather than a
    legitimate environment gap — absent docker, absent SDK, absent ``claude``
    CLI, absent provider key — and pytest exits 0 for all of them. The CLI one
    is not hypothetical: a fresh runner has no ``claude`` on PATH until the lane
    installs the pinned one, so ``is_claude_code_available()`` can skip both
    live tests, leaving the offline floor and judge checks to carry a green
    check on their own. So the job
    reads the junit report and fails on any skip, and on an empty collection,
    which would otherwise satisfy a zero-skip check trivially. Every pytest
    step must write a report the gate actually reads: a run whose report
    nothing inspects is back to skipping its way to green."""
    reports = _junit_reports_written_by(_scan_agentic_pytest_steps(workflow))
    assert len(reports) == len(_scan_agentic_pytest_steps(workflow)), (
        f"every pytest step in '{SCAN_AGENTIC_JOB}' must write a --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, SCAN_AGENTIC_JOB, SCAN_AGENTIC_SKIP_GATE_STEP)["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{SCAN_AGENTIC_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_scan_agentic_job_fails_on_any_skipped_test__mutation_drops_the_junit_report() -> None:
    """A pytest step that writes no report is invisible to the gate — the
    lane keeps running, and the gate keeps passing on a file it never reads."""
    mutated = copy.deepcopy(_load_workflow())
    step = _scan_agentic_pytest_steps(mutated)[0]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_scan_agentic_job_fails_on_any_skipped_test(mutated)


def test_scan_agentic_job_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is
    decorative: the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, SCAN_AGENTIC_JOB, SCAN_AGENTIC_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_scan_agentic_job_fails_on_any_skipped_test(mutated)


# ---------------------------------------------------------------------------
# (l) every CLI-dependent lane installs the pinned Claude Code CLI
# ---------------------------------------------------------------------------

#: Lanes that run Claude Code on the runner itself: agent tests gated on
#: ``is_claude_code_available()``, which shells ``claude --version`` off PATH,
#: and the unit lane's real-binary checks. OSPREY points every agent run at an
#: installed ``claude`` that reports the pinned version and refuses to start
#: otherwise, so each lane installs that version before running pytest — on a
#: bare runner the gated tests would skip and the lane green having executed
#: nothing, and the ungated ones would fail. The chat-bridge lanes belong here
#: too: each starts a real dispatch worker on the runner, and the worker refuses
#: to start without that install.
CLI_DEPENDENT_JOBS = (
    "test",
    "plugin-smoke",
    "agentic-per-preset",
    "e2e-tests",
    "channel-finder-benchmarks",
    NO_MODEL_JOB,
    RECORD_E2E_JOB,
    SCAN_AGENTIC_JOB,
    TARGET_SWITCH_AGENTIC_JOB,
    NEXTCLOUD_JOB,
    GCHAT_JOB,
    TEAMS_JOB,
)

PINNED_CLI_STEP = "Install the pinned Claude Code CLI"
PINNED_CLI_ACTION = "./.github/actions/install-pinned-claude-code"
PINNED_CLI_ACTION_FILE = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "actions"
    / "install-pinned-claude-code"
    / "action.yml"
)


def _pinned_cli_action() -> dict[str, Any]:
    return yaml.safe_load(PINNED_CLI_ACTION_FILE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("job_name", CLI_DEPENDENT_JOBS)
def test_cli_dependent_lane_installs_the_pinned_claude(
    job_name: str, workflow: dict[str, Any] | None = None
) -> None:
    """A lane running Claude Code must install the pinned version first.

    Without it the lane is vacuously green or needlessly red: pytest skips
    every test gated on a ``claude`` on PATH, and fails every one that is not.
    Pinned by content, not just by step name: the step must use the one action
    that installs and verifies the pin, after the project venv is synced (the
    action reads the pin out of it) and ahead of the lane's tests.
    """
    wf = workflow if workflow is not None else _load_workflow()
    step = _find_named_step(wf, job_name, PINNED_CLI_STEP)
    assert step.get("uses") == PINNED_CLI_ACTION, (
        f"{job_name}: the step must use {PINNED_CLI_ACTION}, got {step.get('uses')!r}"
    )
    steps = _jobs(wf)[job_name]["steps"]
    at = steps.index(step)
    assert any("uv sync" in (s.get("run") or "") for s in steps[:at]), (
        f"{job_name}: the venv must be synced before the pinned CLI is installed"
    )
    uses_claude = "claude plugin" if job_name == "plugin-smoke" else "pytest"
    assert any(uses_claude in (s.get("run") or "") for s in steps[at:]), (
        f"{job_name}: nothing after the install step runs {uses_claude}"
    )


@pytest.mark.parametrize("job_name", CLI_DEPENDENT_JOBS)
def test_cli_dependent_lane_installs_the_pinned_claude__mutation_drops_step(job_name: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[job_name]["steps"]
    kept = [s for s in steps if s.get("name") != PINNED_CLI_STEP]
    assert len(kept) == len(steps) - 1, "expected exactly one step dropped"
    _jobs(mutated)[job_name]["steps"] = kept
    with pytest.raises(AssertionError):
        test_cli_dependent_lane_installs_the_pinned_claude(job_name, mutated)


@pytest.mark.parametrize("job_name", CLI_DEPENDENT_JOBS)
def test_cli_dependent_lane_installs_the_pinned_claude__mutation_other_action(
    job_name: str,
) -> None:
    """The quiet half: the step keeps its name but no longer installs the pin."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, job_name, PINNED_CLI_STEP)
    step["uses"] = "actions/setup-node@v7"
    with pytest.raises(AssertionError):
        test_cli_dependent_lane_installs_the_pinned_claude(job_name, mutated)


def test_the_pinned_cli_action_installs_and_verifies_the_one_pinned_version() -> None:
    """The action installs exactly the framework's pin and fails unless PATH yields it."""
    action = _pinned_cli_action()
    assert action["runs"]["using"] == "composite"
    steps = action["runs"]["steps"]
    assert any(str(s.get("uses", "")).startswith("actions/setup-node@") for s in steps)
    (install,) = [s for s in steps if "npm install" in (s.get("run") or "")]
    run = install["run"]
    assert "set -euo pipefail" in run
    # The pin is read from the one constant that names it, never spelled here.
    assert "_DEFAULT_CLAUDE_CLI_VERSION" in run
    assert not re.search(r"claude-code@[0-9]", run), "the action spells a version of its own"
    assert 'npm install -g "@anthropic-ai/claude-code@$pin"' in run
    # Verified, so a registry or PATH surprise fails here and not in a test.
    assert "claude --version" in run
    assert "exit 1" in run


def test_no_lane_reaches_for_the_agent_sdks_bundled_cli() -> None:
    """One Claude Code per lane: nothing puts the SDK's own copy on PATH."""
    text = CI_YML.read_text(encoding="utf-8")
    assert "_bundled" not in text
    assert "_bundled" not in PINNED_CLI_ACTION_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (g) the archiver-world lane: its own job, secret-free, and gated on both halves
# ---------------------------------------------------------------------------


def test_archiver_world_job_installs_osprey(workflow: dict[str, Any]) -> None:
    """The staged bring-up preflights pymongo before any image build, so a lane
    that never installed osprey would abort in seconds with an install hint
    instead of running — a green-looking job that tested nothing. pymongo is a
    core dependency, so a plain sync is enough; there is no extra to select."""
    installed = json.dumps(_jobs(workflow)[ARCHIVER_JOB])
    assert "uv sync" in installed, f"the '{ARCHIVER_JOB}' lane must install osprey"


def test_archiver_world_job_installs_osprey__mutation_drops_sync() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][ARCHIVER_JOB] = json.loads(
        json.dumps(mutated["jobs"][ARCHIVER_JOB]).replace("uv sync", "true")
    )
    with pytest.raises(AssertionError):
        test_archiver_world_job_installs_osprey(mutated)


def test_archiver_world_job_runs_pytest_unbuffered(workflow: dict[str, Any]) -> None:
    """``-s`` is what puts this lane's measured budgets in the log on a GREEN run.

    Pytest captures stdout and replays it only for failures, so without this the
    seed duration, store size and write->read latency appear exactly when nobody
    can act on them any more. The budgets are the lane's product — a run that
    hides them still passes, and the trend that precedes a breach is invisible.
    """
    steps = json.dumps(_jobs(workflow)[ARCHIVER_JOB]["steps"])
    assert ARCHIVER_TEST_FILE in steps
    assert " -s " in steps or steps.rstrip().endswith(" -s"), (
        f"the '{ARCHIVER_JOB}' lane must run pytest with -s so the MEASURED "
        "budget lines reach the log on a passing run"
    )


def test_archiver_world_job_runs_pytest_unbuffered__mutation_drops_the_flag() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][ARCHIVER_JOB] = json.loads(
        json.dumps(mutated["jobs"][ARCHIVER_JOB]).replace(" -v -s ", " -v ")
    )
    with pytest.raises(AssertionError):
        test_archiver_world_job_runs_pytest_unbuffered(mutated)


# ---------------------------------------------------------------------------
# CI diagnostics: the evidence a killed lane leaves behind
# ---------------------------------------------------------------------------
#
# A lane that FAILS explains itself in the job log. A lane that is KILLED — job
# timeout, runner OOM, a wedged xdist worker held to the cap — explains nothing,
# because the signal lands while pytest's output is still buffered. Everything
# pinned below exists to make that second case readable, and every piece of it
# is silently droppable: removing a step, raising a timeout, or restoring an ini
# option costs no test failure of its own. Hence these pins.

CAPTURE_ACTION = "./.github/actions/capture-ci-diagnostics"
CAPTURE_ACTION_YML = REPO_ROOT / ".github" / "actions" / "capture-ci-diagnostics" / "action.yml"
DIAG_DIR_ENV = "OSPREY_CI_DIAG_DIR"
#: A step that runs pytest. Plain ``\bpytest\b`` would also match the shell
#: comments in the skip-gate steps.
_PYTEST_RUN = re.compile(r"\b(?:uv run(?: --no-sync)?|python3? -m) pytest\b")
CAPTURE_STEP = "Capture failure diagnostics"
WRITEBACK_JOB = "deploy-writeback-e2e"
PODMAN_LIFECYCLE_JOB = "multi-user-deploy-lifecycle-e2e-podman"
#: Jobs that name a container engine without running one — the gate lists the
#: Docker lanes in its human-readable status strings.
NON_CONTAINER_JOBS = {GATE_JOB}


def _container_jobs(wf: dict[str, Any]) -> set[str]:
    """Jobs that touch a container engine, derived rather than listed.

    Derived on purpose: a hand-maintained list is exactly what a newly added
    Docker lane forgets to join. YAML comments cannot pollute the match because
    ``yaml.safe_load`` has already dropped them — but a ``#`` line *inside* a
    ``run:`` block is part of the shell script, so it is data and it does match.
    Prose about the engine belongs above the ``run:`` key for that reason.
    """
    pattern = re.compile(r"\b(docker|podman)\b")
    return {
        name
        for name, job in _jobs(wf).items()
        if name not in NON_CONTAINER_JOBS and pattern.search(json.dumps(job))
    }


def _capture_step_index(job: dict[str, Any]) -> int | None:
    for index, step in enumerate(job.get("steps") or []):
        if step.get("uses") == CAPTURE_ACTION:
            return index
    return None


def test_every_container_job_captures_diagnostics(workflow: dict[str, Any]) -> None:
    """No Docker lane may fail without leaving its container logs behind.

    Before this action existed, the only ``if: always()`` steps on these lanes
    were teardown blocks — so a red lane deleted the containers holding the
    explanation and uploaded nothing at all.
    """
    missing = sorted(
        name
        for name in _container_jobs(workflow)
        if _capture_step_index(_jobs(workflow)[name]) is None
    )
    assert missing == [], (
        f"these jobs run containers but have no '{CAPTURE_STEP}' step: {missing}. "
        f"Add `uses: {CAPTURE_ACTION}` with `if: failure()`, positioned before any "
        f"teardown step — or add the job to NON_CONTAINER_JOBS if it only names an "
        f"engine in a status string."
    )


def test_every_container_job_captures_diagnostics__mutation_drops_a_lane() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    del job["steps"][_capture_step_index(job)]
    with pytest.raises(AssertionError):
        test_every_container_job_captures_diagnostics(mutated)


def test_capture_runs_before_teardown(workflow: dict[str, Any]) -> None:
    """Ordering is the whole point: the teardown blocks ``docker rm -f`` the very
    containers whose logs are the diagnosis. Capture after cleanup captures
    nothing."""
    late = []
    for name in sorted(_container_jobs(workflow)):
        steps = _jobs(workflow)[name]["steps"]
        capture = _capture_step_index(_jobs(workflow)[name])
        teardown = [i for i, step in enumerate(steps) if step.get("if") == "always()"]
        if capture is not None and teardown and capture > min(teardown):
            late.append(name)
    assert late == [], (
        f"'{CAPTURE_STEP}' runs after an `if: always()` teardown step in {late} — "
        f"by then the containers it reads have been removed."
    )


def test_capture_runs_before_teardown__mutation_moves_it_after_cleanup() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[WRITEBACK_JOB]["steps"]
    steps.append(steps.pop(_capture_step_index(_jobs(mutated)[WRITEBACK_JOB])))
    with pytest.raises(AssertionError):
        test_capture_runs_before_teardown(mutated)


WRITEBACK_E2E_MODULE = REPO_ROOT / "tests" / "e2e" / "test_deploy_writeback_e2e.py"
WRITEBACK_CLEANUP_STEP = "Clean up any stranded write-back-e2e resources"
WRITEBACK_PROJECT_PREFIX = "osprey-e2e-wb-"


def _writeback_project_cases() -> set[str]:
    """The suffix of every ``PROJECT_*`` compose project the write-back e2e creates.

    Read with ``ast``, never by import: the module needs Docker fixtures.
    """
    tree = ast.parse(WRITEBACK_E2E_MODULE.read_text(encoding="utf-8"))
    return {
        node.value.value.removeprefix(WRITEBACK_PROJECT_PREFIX)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("PROJECT_") for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and node.value.value.startswith(WRITEBACK_PROJECT_PREFIX)
    }


def test_writeback_cleanup_sweeps_every_project_the_test_creates(
    workflow: dict[str, Any],
) -> None:
    """The cleanup loop's cases are exactly the projects the e2e module names, so
    renaming a project cannot leave its containers and volumes behind."""
    run = _find_named_step(workflow, WRITEBACK_JOB, WRITEBACK_CLEANUP_STEP)["run"]
    loop = re.search(r"for case in ([^;\n]+); do", run)
    assert loop is not None, f"no `for case in …; do` loop in step {WRITEBACK_CLEANUP_STEP!r}"
    assert f'project="{WRITEBACK_PROJECT_PREFIX}${{case}}"' in run
    cases = _writeback_project_cases()
    assert cases, f"no PROJECT_* constants found in {WRITEBACK_E2E_MODULE}"
    assert set(loop.group(1).split()) == cases


def test_writeback_cleanup_sweeps_every_project_the_test_creates__mutation_stale_case() -> None:
    mutated = _load_workflow()
    step = _find_named_step(mutated, WRITEBACK_JOB, WRITEBACK_CLEANUP_STEP)
    step["run"] = re.sub(
        r"for case in [^;\n]+; do", "for case in rebuild divergent orphan; do", step["run"]
    )
    with pytest.raises(AssertionError):
        test_writeback_cleanup_sweeps_every_project_the_test_creates(mutated)


def test_capture_steps_run_on_failure(workflow: dict[str, Any]) -> None:
    """``if: failure()`` and not the default: a step with no condition is skipped
    the moment anything upstream fails, which is the only time it matters."""
    wrong = {}
    for name in sorted(_container_jobs(workflow)):
        index = _capture_step_index(_jobs(workflow)[name])
        if index is None:
            continue
        condition = _jobs(workflow)[name]["steps"][index].get("if")
        if condition != "failure()":
            wrong[name] = condition
    assert wrong == {}, f"capture steps must carry `if: failure()`; got {wrong}"


def test_capture_steps_run_on_failure__mutation_drops_the_condition() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    del job["steps"][_capture_step_index(job)]["if"]
    with pytest.raises(AssertionError):
        test_capture_steps_run_on_failure(mutated)


def test_capture_artifact_names_are_unique(workflow: dict[str, Any]) -> None:
    """Artifact names collide silently within a run — two lanes sharing one name
    means one lane's evidence overwrites the other's."""
    names = []
    for name in sorted(_container_jobs(workflow)):
        index = _capture_step_index(_jobs(workflow)[name])
        if index is not None:
            names.append(_jobs(workflow)[name]["steps"][index]["with"]["name"])
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert duplicates == [], f"duplicate capture artifact names: {duplicates}"


def test_capture_artifact_names_are_unique__mutation_collides_two_lanes() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    job["steps"][_capture_step_index(job)]["with"]["name"] = VA_JOB
    with pytest.raises(AssertionError):
        test_capture_artifact_names_are_unique(mutated)


def test_podman_lane_captures_with_podman(workflow: dict[str, Any]) -> None:
    """The podman lane has no ``docker`` binary; asking for one captures nothing."""
    job = _jobs(workflow)[PODMAN_LIFECYCLE_JOB]
    step = job["steps"][_capture_step_index(job)]
    assert step["with"]["engine"] == "podman"


def test_podman_lane_captures_with_podman__mutation_asks_for_docker() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[PODMAN_LIFECYCLE_JOB]
    job["steps"][_capture_step_index(job)]["with"]["engine"] = "docker"
    with pytest.raises(AssertionError):
        test_podman_lane_captures_with_podman(mutated)


def _capture_upload_dir() -> str:
    """The directory the capture action uploads, read from the action itself."""
    for step in _load_action_yml(CAPTURE_ACTION_YML)["runs"]["steps"]:
        if str(step.get("uses", "")).startswith("actions/upload-artifact"):
            return str(step["with"]["path"]).rstrip("/")
    raise AssertionError(f"{CAPTURE_ACTION_YML} has no upload-artifact step")


def _resolved_env(
    wf: dict[str, Any], job: dict[str, Any], step: dict[str, Any], key: str
) -> str | None:
    """``key`` as Actions resolves it for ``step``: step, then job, then workflow."""
    for scope in (step, job, wf):
        env = scope.get("env") or {}
        if key in env:
            return None if env[key] is None else str(env[key])
    return None


def test_container_lanes_write_diagnostics_where_the_capture_uploads(
    workflow: dict[str, Any],
) -> None:
    """Every pytest step in a container lane arms the failure-time snapshot.

    That snapshot is the only record of a failing test's containers: the
    module's fixtures remove them inside the pytest step, before any step after
    it runs, so the capture step can only upload what pytest wrote.
    """
    expected = _capture_upload_dir()
    offenders = [
        (name, step.get("name"), _resolved_env(workflow, job, step, DIAG_DIR_ENV))
        for name, job in _jobs(workflow).items()
        if name in _container_jobs(workflow)
        for step in job.get("steps") or []
        if _PYTEST_RUN.search(str(step.get("run", "")))
        and _resolved_env(workflow, job, step, DIAG_DIR_ENV) != expected
    ]
    assert not offenders, (
        f"pytest steps whose {DIAG_DIR_ENV} is not {expected!r}: {offenders}. "
        "Such a step leaves no failure-time container snapshot for the capture "
        "step to upload."
    )


def test_container_lanes_write_diagnostics_where_the_capture_uploads__mutation_drops_the_workflow_default() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    mutated["env"].pop(DIAG_DIR_ENV, None)
    with pytest.raises(AssertionError):
        test_container_lanes_write_diagnostics_where_the_capture_uploads(mutated)


def test_container_lanes_write_diagnostics_where_the_capture_uploads__mutation_points_a_lane_elsewhere() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, TILED_JOB, "Run Tiled roundtrip E2E").setdefault("env", {})[
        DIAG_DIR_ENV
    ] = "elsewhere"
    with pytest.raises(AssertionError):
        test_container_lanes_write_diagnostics_where_the_capture_uploads(mutated)


# ---------------------------------------------------------------------------
# The unit lane's own survivability
# ---------------------------------------------------------------------------


def test_unit_test_step_timeout_is_below_the_job_cap(workflow: dict[str, Any]) -> None:
    """A JOB timeout is a hard cancel — the collection steps never dependably
    run, which is how a frozen lane ends up recorded as ``cancelled`` with
    nothing to read. A STEP timeout fails cleanly and leaves the rest of the
    job's time for the summary and upload steps."""
    job = _jobs(workflow)[UNIT_TEST_JOB]
    step = _find_named_step(workflow, UNIT_TEST_JOB, "Run unit tests")
    step_cap = step.get("timeout-minutes")
    assert step_cap is not None, (
        "the 'Run unit tests' step needs its own `timeout-minutes`; without one a "
        "wedged worker is killed by the job cap and takes the evidence with it"
    )
    assert step_cap < job["timeout-minutes"], (
        f"step cap ({step_cap}m) must leave headroom under the job cap "
        f"({job['timeout-minutes']}m) for the diagnostics steps to run"
    )


def test_unit_test_step_timeout_is_below_the_job_cap__mutation_drops_it() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_unit_test_step_timeout_is_below_the_job_cap(mutated)


def test_unit_test_step_timeout_is_below_the_job_cap__mutation_raises_it_to_the_cap() -> None:
    """Equal is as bad as absent: no headroom means no collection phase."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[UNIT_TEST_JOB]
    _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")["timeout-minutes"] = job[
        "timeout-minutes"
    ]
    with pytest.raises(AssertionError):
        test_unit_test_step_timeout_is_below_the_job_cap(mutated)


def test_unit_lane_arms_the_diagnostics_recorder(workflow: dict[str, Any]) -> None:
    """``osprey_connectors_tests._harness.ci_diagnostics`` installs nothing unless
    this variable is set, so without it the lane runs exactly as blind as before."""
    step = _find_named_step(workflow, UNIT_TEST_JOB, "Run unit tests")
    assert step.get("env", {}).get("OSPREY_CI_DIAG_DIR"), (
        "the unit lane must set OSPREY_CI_DIAG_DIR on the pytest step to enable the "
        "per-worker event log and stack dumps"
    )


def test_unit_lane_arms_the_diagnostics_recorder__mutation_drops_the_env() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")["env"]["OSPREY_CI_DIAG_DIR"]
    with pytest.raises(AssertionError):
        test_unit_lane_arms_the_diagnostics_recorder(mutated)


#: The runner-disk scratch directory, as the step-level env has to spell it.
#: ``runner.temp`` is on the same filesystem as uv's cache, so the project
#: venvs the real-build tests install there are hardlinked rather than copied.
RUNNER_DISK_TMPDIR = "${{ runner.temp }}"
#: Every step that runs real ``osprey build``s under pytest's basetemp: each
#: installs a ~2 GB project venv there, and on the Ubuntu 26.04 image /tmp is a
#: RAM-backed tmpfs with a ~6.3 GB per-user quota (actions/runner-images#14777).
REAL_BUILD_STEPS = (
    (UNIT_TEST_JOB, "Run unit tests"),
    (BOOT_SMOKE_JOB, BOOT_SMOKE_STEP),
)


@pytest.mark.parametrize(("job", "step_name"), REAL_BUILD_STEPS)
def test_real_build_lanes_keep_scratch_on_the_runner_disk(
    workflow: dict[str, Any], job: str, step_name: str
) -> None:
    """A lane that builds real project venvs under ``tmp_path`` must not do
    so under ``/tmp``: on ubuntu-latest that is a quota-limited tmpfs, and
    the venvs overran it with ``[Errno 122] Disk quota exceeded``."""
    env = _find_named_step(workflow, job, step_name).get("env", {})
    assert env.get("TMPDIR") == RUNNER_DISK_TMPDIR, (
        f"step '{step_name}' of job '{job}' must set TMPDIR to {RUNNER_DISK_TMPDIR}: its "
        "real builds install gigabytes of project venvs under pytest's basetemp, which "
        "lands on the runner's tmpfs /tmp otherwise"
    )


@pytest.mark.parametrize(("job", "step_name"), REAL_BUILD_STEPS)
def test_real_build_lanes_keep_scratch_on_the_runner_disk__mutation_drops_tmpdir(
    job: str, step_name: str
) -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, job, step_name)["env"]["TMPDIR"]
    with pytest.raises(AssertionError):
        test_real_build_lanes_keep_scratch_on_the_runner_disk(mutated, job, step_name)


@pytest.mark.parametrize(("job", "step_name"), REAL_BUILD_STEPS)
def test_real_build_lanes_keep_scratch_on_the_runner_disk__mutation_points_back_at_tmp(
    job: str, step_name: str
) -> None:
    """Set but to the tmpfs itself is the regression, not a variant of the fix."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, job, step_name)["env"]["TMPDIR"] = "/tmp"
    with pytest.raises(AssertionError):
        test_real_build_lanes_keep_scratch_on_the_runner_disk(mutated, job, step_name)


def test_unit_lane_uploads_its_diagnostics(workflow: dict[str, Any]) -> None:
    """``always()``, not ``failure()``: a lane killed by the job cap is recorded
    as ``cancelled``, and a ``failure()`` step would not run for it."""
    upload = _find_named_step(workflow, UNIT_TEST_JOB, "Upload lane diagnostics")
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == "ci-diag"
    summary = _find_named_step(workflow, UNIT_TEST_JOB, "Summarise lane diagnostics")
    assert summary["if"] == "always()"


def test_unit_lane_uploads_its_diagnostics__mutation_narrows_to_failure() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, UNIT_TEST_JOB, "Upload lane diagnostics")["if"] = "failure()"
    with pytest.raises(AssertionError):
        test_unit_lane_uploads_its_diagnostics(mutated)


def test_unit_lane_does_not_set_faulthandler_timeout(workflow: dict[str, Any]) -> None:
    """The subtle one: the ini option arms a watchdog that kills workers.

    pytest implements ``faulthandler_timeout`` with
    ``faulthandler.dump_traceback_later``, whose watchdog walks every thread's
    frames without holding the GIL. A sample landing inside one of the parsers
    this suite lives in — PyYAML, ruamel, Jinja2 — reads a frame that is being
    freed and takes the worker down with it, which xdist reports only as ``node
    down: Not properly terminated``.
    ``osprey_connectors_tests._harness.ci_diagnostics`` samples from an ordinary
    Python thread for that reason, and setting this option would put
    the unsafe watchdog back, on a per-test timer.

    Its dumps would also reach only the job log, which is exactly what a
    cancelled job truncates.
    """
    line = _unit_test_pytest_line(workflow)
    assert "faulthandler_timeout" not in line, (
        "the unit lane must not pass `-o faulthandler_timeout`: pytest implements it "
        "with faulthandler.dump_traceback_later, whose GIL-free watchdog segfaults "
        "workers that are inside a YAML or Jinja parser. See the docstring of "
        "osprey_connectors_tests._harness.ci_diagnostics."
    )


def test_unit_lane_passes_per_test_timeout(workflow: dict[str, Any]) -> None:
    """The complement of the pin above: no watchdog in-process, but a real cap.

    Without one, a single hung test is indistinguishable from "still running"
    until the step cap fires 40 minutes later and cancels the job — which
    truncates the log, so the run ends with no name for the test that hung
    (#743). ``--timeout=600`` turns that into a named failure inside ten
    minutes and still clears a cold image pull, so the container-starting
    files need no exemption.

    ``--timeout-method=signal`` is the other half. pytest-timeout's ``thread``
    method ends the test with ``os._exit`` of the worker, and a dead worker
    under ``--dist loadgroup`` takes the whole lane with it rather than one
    test — see the docstring of ``osprey_connectors_tests._harness.ci_diagnostics``.
    The cost of ``signal`` is that it cannot interrupt a hang inside a C extension holding the GIL;
    for that case the step cap is still the backstop.
    """
    line = _unit_test_pytest_line(workflow)
    missing = [flag for flag in PER_TEST_TIMEOUT_FLAGS if flag not in line]
    assert missing == [], (
        f"the '{UNIT_TEST_JOB}' job in .github/workflows/ci.yml must invoke pytest with "
        f"{' '.join(PER_TEST_TIMEOUT_FLAGS)} — missing {missing} in: {line.strip()!r}. "
        f"The timeout belongs on this line and not in [tool.pytest.ini_options]: an ini "
        f"default would also apply to every e2e lane, none of which passes a --timeout."
    )


def _sole_heavy_step(job: dict[str, Any]) -> dict[str, Any] | None:
    """The lane's single pytest-running step, or None if there isn't exactly one.

    Lanes with several heavy steps are excluded on purpose: capping them means
    splitting one job budget between them, which is a per-lane runtime judgement
    rather than something derivable, and a wrong split reds a healthy run.
    """
    heavy = [s for s in (job.get("steps") or []) if "uv run pytest" in str(s.get("run", ""))]
    return heavy[0] if len(heavy) == 1 else None


def test_capped_container_lanes_cap_their_heavy_step(workflow: dict[str, Any]) -> None:
    """Same argument as the unit lane, applied to the e2e lanes.

    Without a step cap the hang is killed by the JOB cap, which is a hard cancel
    — and the capture step never runs, so the containers are torn down by the
    runner with their logs unread. That is the exact blind spot the capture step
    exists to close, so a capped lane must not leave it open.
    """
    missing = []
    for name in sorted(_container_jobs(workflow)):
        job = _jobs(workflow)[name]
        job_cap = job.get("timeout-minutes")
        step = _sole_heavy_step(job)
        if not job_cap or step is None:
            continue
        step_cap = step.get("timeout-minutes")
        if step_cap is None or step_cap >= job_cap:
            missing.append((name, step.get("name"), step_cap, job_cap))
    assert missing == [], (
        "these lanes declare a job cap and have one heavy step, so that step needs its "
        f"own smaller `timeout-minutes`: {missing}"
    )


def test_capped_container_lanes_cap_their_heavy_step__mutation_drops_a_step_cap() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _sole_heavy_step(_jobs(mutated)[TILED_JOB])["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_capped_container_lanes_cap_their_heavy_step(mutated)


def test_capped_container_lanes_cap_their_heavy_step__mutation_raises_it_to_the_job_cap() -> None:
    """No headroom is the same as no cap — the capture step still never runs."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[TILED_JOB]
    _sole_heavy_step(job)["timeout-minutes"] = job["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_capped_container_lanes_cap_their_heavy_step(mutated)


def test_unit_lane_does_not_set_faulthandler_timeout__mutation_restores_the_ini_option() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace(
        " --dist loadgroup", " --dist loadgroup -o faulthandler_timeout=300"
    )
    assert step["run"] != original, "the unit line moved; mutation is stale"
    with pytest.raises(AssertionError):
        test_unit_lane_does_not_set_faulthandler_timeout(mutated)


def test_unit_lane_passes_per_test_timeout__mutation_drops_it() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    for flag in PER_TEST_TIMEOUT_FLAGS:
        step["run"] = step["run"].replace(f" {flag}", "")
    assert step["run"] != original, "timeout flags not found; mutation is stale"
    with pytest.raises(AssertionError):
        test_unit_lane_passes_per_test_timeout(mutated)


def test_unit_lane_passes_per_test_timeout__mutation_switches_to_the_thread_method() -> None:
    """The method is pinned, not just the presence of a timeout: ``thread`` is
    the variant that ``os._exit``s the xdist worker, turning "one hung test"
    into "the lane is wedged"."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace("--timeout-method=signal", "--timeout-method=thread")
    assert step["run"] != original, "method flag not found; mutation is stale"
    with pytest.raises(AssertionError):
        test_unit_lane_passes_per_test_timeout(mutated)


# The two-shape boot lane: podman-compose forced and PROVEN, the network axis
# run under it, and an epic-base trigger so it runs where the axis is built
# ---------------------------------------------------------------------------
#
# Two jobs are asserted here because the coverage is deliberately split in two.
# `two-shape-boot-e2e` is the real thing — it boots the dispatcher/worker pair
# twice, once per network shape, on a rootless podman runner. The parse-only
# step in `static-checks` is the cheap half: it renders the same kind of
# deployment and hands podman-compose OSPREY's own argv, so a merged document
# that no longer parses, or an env chain that stops interpolating, is caught in
# a Tier 0 job in seconds rather than after a ~20-minute image build.
#
# The provider is the load-bearing fact for both. `podman compose` delegates to
# whichever external provider it finds, and on a GitHub runner the first one is
# the docker-compose CLI plugin — the engine the docker lanes already cover,
# and one that accepts the `--project-directory` this argv deliberately omits.
# A lane that quietly ran the plugin would report green having tested nothing
# new, which is why the pin, the force and the assertion are each checked
# independently below.


def _step_names(wf: dict[str, Any], job_name: str) -> list[str]:
    return [step.get("name", "") for step in _jobs(wf)[job_name]["steps"]]


def test_two_shape_boot_job_runs_the_module_under_podman(workflow: dict[str, Any]) -> None:
    """The module reads its runtime off ``OSPREY_E2E_RUNTIME`` and defaults to
    docker. Without that variable the lane would install podman-compose, force
    it, assert it — and then run the whole boot against the runner's docker
    daemon, certifying the provider it was written to leave behind."""
    step = _find_named_step(workflow, TWO_SHAPE_JOB, TWO_SHAPE_BOOT_STEP)
    assert TWO_SHAPE_TEST_FILE in step["run"]
    assert step.get("env", {}).get("OSPREY_E2E_RUNTIME") == "podman", (
        f"the '{TWO_SHAPE_JOB}' boot step must set OSPREY_E2E_RUNTIME=podman; got {step.get('env')}"
    )


def test_two_shape_boot_job_runs_the_module_under_podman__mutation_drops_the_runtime() -> None:
    """Dropping the variable while leaving the step must fail — the exact shape
    of a lane that silently reverts to docker."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, TWO_SHAPE_BOOT_STEP)
    del step["env"]["OSPREY_E2E_RUNTIME"]
    assert TWO_SHAPE_TEST_FILE in step["run"]  # the module is still named
    with pytest.raises(AssertionError):
        test_two_shape_boot_job_runs_the_module_under_podman(mutated)


def test_two_shape_boot_job_runs_the_module_under_podman__mutation_drops_the_step() -> None:
    """A lane with no pytest step passes every other check in this section and
    runs nothing at all."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TWO_SHAPE_JOB]["steps"]
    _jobs(mutated)[TWO_SHAPE_JOB]["steps"] = [
        s for s in steps if s.get("name") != TWO_SHAPE_BOOT_STEP
    ]
    with pytest.raises(AssertionError, match="no step named"):
        test_two_shape_boot_job_runs_the_module_under_podman(mutated)


def _podman_compose_is_pinned_and_forced(wf: dict[str, Any], job_name: str) -> list[str]:
    """Which parts of the provider pin are MISSING from a job (empty = all).

    Three independent mechanisms, and none implies the others. An unpinned
    install drifts to whatever podman-compose release exists on the day. An
    unforced provider lets podman pick the docker-compose CLI plugin no matter
    which podman-compose sits beside it — podman's documented default order
    tries docker-compose FIRST, which is exactly how "we installed it" and "it
    executes" came apart in the neighbouring lane.

    The env override is the third because the file has one gap: podman reads a
    USER-level ``containers.conf.d/*.conf`` drop-in AFTER the user's
    ``containers.conf``, so such a drop-in would outrank the file this lane
    writes. ``PODMAN_COMPOSE_PROVIDER`` sidesteps the chain entirely. Nothing
    in this repo writes such a drop-in, so this is hardening against a future
    runner image rather than a fix for a live hole — an unpinned belt is how a
    belt quietly disappears.
    """
    serialized = json.dumps(_jobs(wf)[job_name])
    return [
        label
        for label, token in (
            ("version pin", PODMAN_COMPOSE_PIN),
            ("force", COMPOSE_PROVIDER_FORCE_KEY),
            ("env override", COMPOSE_PROVIDER_FORCE_ENV),
        )
        if token not in serialized
    ]


@pytest.mark.parametrize("job_name", [TWO_SHAPE_JOB, PARSE_ONLY_JOB])
def test_podman_compose_is_pinned_and_forced(workflow: dict[str, Any], job_name: str) -> None:
    """Both jobs that speak podman-compose must pin the version AND force
    podman to use it, for the reasons in this section's header."""
    assert _podman_compose_is_pinned_and_forced(workflow, job_name) == []


@pytest.mark.parametrize("job_name", [TWO_SHAPE_JOB, PARSE_ONLY_JOB])
def test_podman_compose_is_pinned_and_forced__mutation_unpins_the_version(job_name: str) -> None:
    """Installing an unpinned podman-compose must be reported: the floor this
    branch is written against is 1.0.6, and a lane that silently moved off it
    would report on an engine nobody chose."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][job_name] = json.loads(
        json.dumps(mutated["jobs"][job_name]).replace(PODMAN_COMPOSE_PIN, "podman-compose")
    )
    assert _podman_compose_is_pinned_and_forced(mutated, job_name) == ["version pin"]


@pytest.mark.parametrize("job_name", [TWO_SHAPE_JOB, PARSE_ONLY_JOB])
def test_podman_compose_is_pinned_and_forced__mutation_drops_the_force(job_name: str) -> None:
    """The dangerous half: podman-compose is installed, so the job LOOKS
    provider-correct, but nothing tells podman to prefer it over the
    docker-compose plugin already on the runner."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][job_name] = json.loads(
        json.dumps(mutated["jobs"][job_name]).replace(COMPOSE_PROVIDER_FORCE_KEY, "unrelated_key")
    )
    assert _podman_compose_is_pinned_and_forced(mutated, job_name) == ["force"]


@pytest.mark.parametrize("job_name", [TWO_SHAPE_JOB, PARSE_ONLY_JOB])
def test_podman_compose_is_pinned_and_forced__mutation_drops_the_env_override(
    job_name: str,
) -> None:
    """The quietest of the three to lose: the file still forces the provider, so
    nothing observable changes until a runner image ships a user-level
    containers.conf.d drop-in — at which point the lane silently runs the
    engine it was written to leave behind."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][job_name] = json.loads(
        json.dumps(mutated["jobs"][job_name]).replace(COMPOSE_PROVIDER_FORCE_ENV, "UNRELATED_VAR")
    )
    assert _podman_compose_is_pinned_and_forced(mutated, job_name) == ["env override"]


def _assert_probes_the_provider(run_text: str) -> None:
    """The content every provider-asserting step must carry, wherever it lives.

    Extracted so the parse-only twin is held to the lane's standard rather than
    inheriting its prose: deleting both assertions from the parse step's inline
    Python used to fail nothing, which is the same regression class the lane's
    own ordering test exists to prevent one job over.
    """
    assert "detect_compose_provider" in run_text, (
        "must probe through OSPREY's own detector, so it answers the same "
        "question the deploy asks rather than a lookalike"
    )
    assert "ComposeProvider.PODMAN_COMPOSE" in run_text, "must assert the provider it claims"
    # The parsed tuple, never the text: `podman-compose==1.0.6` on the install
    # line would satisfy a substring search over the job while the assertion
    # itself checked nothing.
    assert "(1, 0, 6)" in run_text, "must assert the parsed version, not just the banner"


#: Every step that claims the provider, in the job it belongs to. Both are held
#: to `_assert_probes_the_provider`; only the lane's is order-constrained,
#: because only the lane has an expensive boot to run afterwards.
PROVIDER_ASSERTION_STEPS = (
    (TWO_SHAPE_JOB, TWO_SHAPE_PROVIDER_STEP),
    (PARSE_ONLY_JOB, PARSE_ONLY_STEP),
)


@pytest.mark.parametrize(("job_name", "step_name"), PROVIDER_ASSERTION_STEPS)
def test_every_provider_claiming_step_actually_probes(
    workflow: dict[str, Any], job_name: str, step_name: str
) -> None:
    """Both jobs force podman-compose; both must SAY so falsifiably."""
    _assert_probes_the_provider(_find_named_step(workflow, job_name, step_name)["run"])


@pytest.mark.parametrize(("job_name", "step_name"), PROVIDER_ASSERTION_STEPS)
def test_every_provider_claiming_step_actually_probes__mutation_drops_the_asserts(
    job_name: str, step_name: str
) -> None:
    """Stripping the two assertions leaves a step that still runs the right
    engine — the pin and the force are asserted elsewhere — but no longer
    demonstrates it. That is precisely how a falsifiability check rots into a
    comment."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, job_name, step_name)
    step["run"] = "\n".join(
        line
        for line in step["run"].splitlines()
        if "ComposeProvider.PODMAN_COMPOSE" not in line and "(1, 0, 6)" not in line
    )
    with pytest.raises(AssertionError):
        test_every_provider_claiming_step_actually_probes(mutated, job_name, step_name)


def test_two_shape_lane_asserts_its_provider_before_the_boot(workflow: dict[str, Any]) -> None:
    """Step 0 is what makes the pin falsifiable rather than aspirational, and
    its POSITION is half of what it is worth. Placed after the boot it would
    still catch a wrong engine — but only after the lane had already spent the
    image build and reported the axis certified under it.

    The version is asserted as the parsed tuple the probe returns, not as text
    anywhere in the job: ``podman-compose==1.0.6`` in the install line above
    would satisfy a substring search while the assertion itself checked nothing.
    """
    names = _step_names(workflow, TWO_SHAPE_JOB)
    assert TWO_SHAPE_PROVIDER_STEP in names, f"no provider assertion step; got {names}"
    assert names.index(TWO_SHAPE_PROVIDER_STEP) < names.index(TWO_SHAPE_BOOT_STEP), (
        "the provider assertion must run BEFORE the boot it certifies"
    )
    _assert_probes_the_provider(
        _find_named_step(workflow, TWO_SHAPE_JOB, TWO_SHAPE_PROVIDER_STEP)["run"]
    )


def test_two_shape_lane_asserts_its_provider__mutation_moves_it_after_the_boot() -> None:
    """Reordering must fail even though every step survives — the assertion
    would then certify a boot that had already happened."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TWO_SHAPE_JOB]["steps"]
    probe = next(s for s in steps if s.get("name") == TWO_SHAPE_PROVIDER_STEP)
    steps.remove(probe)
    steps.append(probe)
    assert TWO_SHAPE_PROVIDER_STEP in _step_names(mutated, TWO_SHAPE_JOB)  # still present
    with pytest.raises(AssertionError, match="BEFORE the boot"):
        test_two_shape_lane_asserts_its_provider_before_the_boot(mutated)


def test_two_shape_lane_asserts_its_provider__mutation_drops_the_version_check() -> None:
    """A step 0 that recognizes podman-compose but accepts any version is the
    quiet half of the same failure: the lane keeps running after a release that
    changed the argv it certifies."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, TWO_SHAPE_PROVIDER_STEP)
    step["run"] = step["run"].replace("(1, 0, 6)", "info.version")
    assert "ComposeProvider.PODMAN_COMPOSE" in step["run"]  # the provider half survives
    with pytest.raises(AssertionError, match="parsed version"):
        test_two_shape_lane_asserts_its_provider_before_the_boot(mutated)


def test_e2e_lane_ignores_two_shape_boot(workflow: dict[str, Any]) -> None:
    """Named explicitly as well as swept up by the blanket ``dockerbuild``
    guard above: this module builds two images and binds real host ports under
    its host shape, so leaking it into the shared lane would collide with the
    lane's own stacks rather than merely double-execute."""
    assert _run_step_ignores_all(workflow, [TWO_SHAPE_TEST_FILE]) == []


def test_e2e_lane_ignores_two_shape_boot__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], TWO_SHAPE_TEST_FILE)
    assert _run_step_ignores_all(mutated, [TWO_SHAPE_TEST_FILE]) == [TWO_SHAPE_TEST_FILE]


def _accepts_an_epic_base(job_if: str) -> bool:
    """Whether a job's ``if:`` admits a PR whose base is an ``epic/*`` branch."""
    return "epic/" in job_if


def test_two_shape_lane_triggers_on_epic_base_prs(workflow: dict[str, Any]) -> None:
    """Both halves of the trigger, and neither is sufficient alone.

    The workflow must fire at all for a PR into an epic base — that is the
    ``pull_request.branches`` list — and this job's own ``if:`` must then admit
    it. The network axis and the podman-compose provider are built on phase
    branches whose PRs target ``epic/<name>``, so a lane scoped to main alone
    would skip on every PR that changes what it covers and first run once the
    work had already landed. That is the opposite of a gate.
    """
    triggers = workflow[True]["pull_request"]["branches"]
    assert "epic/**" in triggers, f"the workflow must fire on PRs into an epic base; got {triggers}"
    job_if = _jobs(workflow)[TWO_SHAPE_JOB]["if"]
    assert _accepts_an_epic_base(job_if), (
        f"'{TWO_SHAPE_JOB}' must admit PRs based on epic/*; got: {job_if!r}"
    )


def test_two_shape_lane_triggers_on_epic_base_prs__mutation_drops_the_job_clause() -> None:
    """Narrowing the job to main alone must fail even with the workflow-level
    trigger left intact — the lane would be present, wired, and permanently
    skipped on exactly the PRs it exists for."""
    mutated = copy.deepcopy(_load_workflow())
    job_if = _jobs(mutated)[TWO_SHAPE_JOB]["if"]
    # Surgical: only the epic disjunct goes, leaving a well-formed condition
    # that still admits same-repo PRs into main. A mutation that emptied the
    # whole `if:` would fail the assertion for the wrong reason.
    narrowed = job_if.replace(" || startsWith(github.event.pull_request.base.ref, 'epic/')", "")
    assert narrowed != job_if, "no epic clause in the job's if: — mutation is stale"
    assert "base.ref == 'main'" in narrowed, "the main clause must survive the narrowing"
    _jobs(mutated)[TWO_SHAPE_JOB]["if"] = narrowed
    assert "epic/**" in mutated[True]["pull_request"]["branches"]  # workflow trigger survives
    with pytest.raises(AssertionError, match="must admit PRs based on"):
        test_two_shape_lane_triggers_on_epic_base_prs(mutated)


def test_two_shape_lane_triggers_on_epic_base_prs__mutation_drops_the_workflow_trigger() -> None:
    """The other half: a job clause that would admit an epic base is inert if
    the workflow never runs on those pull requests at all."""
    mutated = copy.deepcopy(_load_workflow())
    mutated[True]["pull_request"]["branches"] = ["main"]
    assert _accepts_an_epic_base(_jobs(mutated)[TWO_SHAPE_JOB]["if"])  # job clause survives
    with pytest.raises(AssertionError, match="must fire on PRs into an epic base"):
        test_two_shape_lane_triggers_on_epic_base_prs(mutated)


#: Literal shapes with no flag grammar to respect.
#:
#: ``"*"`` is deliberately BLUNT: it fires on any asterisk anywhere in the job's
#: parsed form, shell comments inside ``run:`` blocks included. That is the
#: right trade for a container-ops guard — a false positive costs one reworded
#: comment and forces a human to look at a wildcard, while a false negative is a
#: glob-matched removal on a host holding someone else's containers. It caught
#: exactly one false positive in this lane (a comment naming a ``conf.d`` glob),
#: which was reworded rather than the check being loosened.
_PRUNE_LITERALS = ("prune", "*")

#: The flag half, matched on WORD BOUNDARIES rather than as a substring. A tuple
#: entry like ``" -a "`` cannot express "flag": it requires a trailing space, so
#: it misses the two most natural spellings — a flag at end of line (``podman
#: volume rm -a``) and a fused cluster (``podman rmi -af``). Both sailed past
#: every assertion in this module until a mutation was written for them.
#: ``-\w*a\w*`` catches any short cluster containing ``a``; the lookarounds are
#: what make it a flag rather than any substring.
_BLANKET_FLAG_RE = re.compile(r"(?<!\S)(-\w*a\w*|--all)(?!\S)")


def _strings_in(value: Any) -> Iterator[str]:
    """Every string leaf in a parsed-YAML structure.

    The flag match MUST run against these rather than against ``json.dumps`` of
    the job: serializing puts a closing quote immediately after a trailing
    flag, so ``podman volume rm -a`` becomes ``...rm -a"`` and the ``(?!\\S)``
    boundary can never fire. That is not hypothetical — it is why the first
    version of this guard passed its own mutations.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings_in(item)


def _blanket_removal_shapes(wf: dict[str, Any], job_name: str) -> list[str]:
    """Blanket-removal shapes present in a job (empty = exact-named throughout).

    Reads the job's PARSED form, so YAML comments are invisible to it — which is
    why the lane's prose may discuss prunes while the guard stays green. A step
    NAME carrying one of these would legitimately trip it.
    """
    found: list[str] = []
    for text in _strings_in(_jobs(wf)[job_name]):
        found.extend(shape for shape in _PRUNE_LITERALS if shape in text)
        found.extend(match.group(0) for match in _BLANKET_FLAG_RE.finditer(text))
    return found


def test_two_shape_lane_cleanup_is_exact_named(workflow: dict[str, Any]) -> None:
    """The module documents a container-ops guardrail — every resource it
    creates is exact-named, and nothing it runs is a prune, an ``-a``/``--all``
    removal or a wildcard match. The lane's belt-and-suspenders cleanup runs on
    a shared-shape runner and must honour the same rule: a blanket sweep here
    would be indistinguishable from the module's own teardown right up until it
    ran on a host that had something else on it."""
    found = _blanket_removal_shapes(workflow, TWO_SHAPE_JOB)
    assert found == [], f"'{TWO_SHAPE_JOB}' contains blanket-removal shape(s) {found}"


def test_two_shape_lane_cleanup_is_exact_named__mutation_adds_a_prune() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[TWO_SHAPE_JOB]["steps"].append(
        {"name": "sweep", "run": "podman system prune -f"}
    )
    with pytest.raises(AssertionError, match="blanket-removal"):
        test_two_shape_lane_cleanup_is_exact_named(mutated)


@pytest.mark.parametrize(
    "swept",
    [
        "podman volume rm -a",  # trailing flag: the substring tuple's blind spot
        "podman rmi -af",  # fused cluster: likewise
        "podman rm --all",
    ],
)
def test_two_shape_lane_cleanup_is_exact_named__mutation_adds_a_blanket_flag(swept: str) -> None:
    """The shapes the substring tuple missed. Each sweeps every volume or image
    on the host rather than the six this lane owns, and each left all of this
    module's assertions green before the boundary match replaced the tuple. The
    gap existed precisely because it had never been mutated."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[TWO_SHAPE_JOB]["steps"].append({"name": "sweep", "run": swept})
    with pytest.raises(AssertionError, match="blanket-removal"):
        test_two_shape_lane_cleanup_is_exact_named(mutated)


def test_blanket_flag_match_respects_word_boundaries() -> None:
    """The guard must not fire on the lane's legitimate argv. ``-f``, ``--type``
    and a hyphenated path are all things the real steps run; a guard that
    reddened on those would be reverted within a week, which is how a safety
    check dies."""
    for benign in ("podman rm -f name", "podman inspect --type container x", "/usr/bin/a-name"):
        assert _BLANKET_FLAG_RE.search(benign) is None, f"false positive on {benign!r}"
    for blanket in ("podman rmi -af", "podman volume rm -a", "podman rm --all"):
        assert _BLANKET_FLAG_RE.search(blanket) is not None, f"missed {blanket!r}"


# ---------------------------------------------------------------------------
# The lane spells the module's resource names and ports by hand — so they are
# pinned to the module's own constants rather than to a reader's memory
# ---------------------------------------------------------------------------


def _two_shape_module_constant(name: str) -> Any:
    """A module-level literal from ``tests/e2e/test_two_shape_boot.py``.

    Read with ``ast``, not imported: importing that module pulls in its e2e
    dependencies and executes its runtime-selection guard, neither of which
    this lane-wiring check has any business doing.
    """
    source = (CI_YML.parents[2] / TWO_SHAPE_TEST_FILE).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        # AnnAssign as well as Assign: `DISPATCHER_PORT: int = 21781` is a
        # spelling already used elsewhere in this codebase, and handling only
        # the bare form would make this helper RAISE the day someone annotates
        # a constant. Loud rather than silent, but still a needless break.
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name and node.value:
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{TWO_SHAPE_TEST_FILE} defines no module-level {name}")


def _two_shape_resource_names() -> list[str]:
    """The exact resource names the lane hardcodes, derived the way the module
    derives them. ``resolve_project_name`` is what turns the project name into
    the compose prefix that the volume and network carry, so it is called here
    rather than re-spelled.

    Ports are deliberately NOT in this list — see
    ``test_two_shape_lane_ports_match_the_module`` for why they need a narrower
    scope than these do.
    """
    from osprey.deployment.compose_generator import resolve_project_name

    project = _two_shape_module_constant("PROJECT_NAME")
    compose_project = resolve_project_name({"project_name": project})
    return [
        f"{project}-event-dispatcher",
        f"{project}-dispatch-worker-1",
        f"{compose_project}_dispatch_workspace_1",
        f"{compose_project}_osprey-network",
        f"{project}:local",
        f"{project}-dispatch:local",
    ]


def _two_shape_module_ports() -> list[str]:
    """The host ports the module binds: the dispatcher's and the worker's."""
    return [
        str(_two_shape_module_constant("DISPATCHER_PORT")),
        str(_two_shape_module_constant("WORKER_PORT")),
    ]


def test_two_shape_lane_names_match_the_module(workflow: dict[str, Any]) -> None:
    """The lane's cleanup names resources the module creates, and YAML cannot
    import a Python constant — so the two are pinned together here or not at
    all. The drift fails quietly: a cleanup naming a container that no longer
    exists removes nothing and reports success, leaving the real one stranded
    to break the next run.

    Job-wide scope is safe for these because no step NAME contains a container,
    volume, network or image name; the only place they appear is the argv that
    removes them.
    """
    job_text = json.dumps(_jobs(workflow)[TWO_SHAPE_JOB])
    missing = [name for name in _two_shape_resource_names() if name not in job_text]
    assert missing == [], (
        f"the '{TWO_SHAPE_JOB}' lane does not name {missing} — these come from "
        f"{TWO_SHAPE_TEST_FILE}'s own constants and must be kept in step with them"
    )


def test_two_shape_lane_names_match_the_module__mutation_drifts_a_name() -> None:
    mutated = copy.deepcopy(_load_workflow())
    volume = f"{_two_shape_module_constant('PROJECT_NAME')}_dispatch_workspace_1"
    mutated["jobs"][TWO_SHAPE_JOB] = json.loads(
        json.dumps(mutated["jobs"][TWO_SHAPE_JOB]).replace(volume, "stale_volume_name")
    )
    with pytest.raises(AssertionError, match="dispatch_workspace_1"):
        test_two_shape_lane_names_match_the_module(mutated)


def test_two_shape_lane_ports_match_the_module(workflow: dict[str, Any]) -> None:
    """Scoped to the pre-flight step's ``run:`` text, NOT to the job.

    The step's ``echo`` repeats every port it probes, so the numbers are in
    the job's JSON no matter what the ``ss`` loop actually probes, and a step
    title naming them would do the same. A job-wide check is therefore
    satisfied by the step's own description of itself: rewriting the loop to
    test 8020/9190 once left every assertion in this module green, which is
    exactly the drift this guard claims to prevent. Reading the assertion does
    not show that; running the mutation does. So the check reads the ``for``
    line alone.

    Same species as the ``or "{}"`` fallback caught one task over — an
    assertion whose subject is wide enough to include the thing that describes
    it, rather than only the thing that does it.
    """
    preflight_run = _find_named_step(workflow, TWO_SHAPE_JOB, PORT_PREFLIGHT_STEP)["run"]
    loops = [line for line in preflight_run.splitlines() if line.strip().startswith("for port in")]
    assert len(loops) == 1, f"the pre-flight step has no single `for port in` loop: {loops}"
    probed = loops[0].split(" in ", 1)[1].strip().removesuffix("; do").split()
    missing = [port for port in _two_shape_module_ports() if port not in probed]
    assert missing == [], (
        f"the pre-flight step's command does not probe {missing} — the module binds "
        f"those ports, and a pre-flight checking others reports 'free' about ports "
        f"nobody wanted"
    )


def test_two_shape_lane_ports_match_the_module__mutation_drifts_the_probed_ports() -> None:
    """The mutation that matters: rewrite the LOOP and leave the step's own
    ``echo`` alone. A whole-step replace would rewrite the echo too, curing the
    symptom the mutation is meant to expose."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, PORT_PREFLIGHT_STEP)
    loop = "for port in 21781 21791; do"
    assert loop in step["run"], "the pre-flight loop moved — mutation is stale"
    step["run"] = step["run"].replace(loop, "for port in 8020 9190; do")
    assert "21781" in step["run"], "the echo must still carry the real ports"
    with pytest.raises(AssertionError, match="does not probe"):
        test_two_shape_lane_ports_match_the_module(mutated)


def test_two_shape_lane_names_match_the_module__mutation_drops_the_network() -> None:
    """The network is the one resource ``osprey down`` normally removes, which
    is exactly why the belt-and-suspenders line matters: it exists for the run
    where ``down`` failed midway."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TWO_SHAPE_JOB]["steps"]
    cleanup = steps[-1]
    before = cleanup["run"]
    cleanup["run"] = "\n".join(line for line in before.splitlines() if "network rm" not in line)
    assert cleanup["run"] != before, "no network removal in the cleanup — mutation is stale"
    with pytest.raises(AssertionError, match="osprey-network"):
        test_two_shape_lane_names_match_the_module(mutated)


# ---------------------------------------------------------------------------
# Two pre-boot steps that exist to make a red lane READABLE
# ---------------------------------------------------------------------------
#
# Neither tests the feature, and they are load-bearing to different degrees.
#
# The port pre-flight is a real precondition: under the host shape those ports
# are bound on the runner itself, so anything already holding one makes the
# module's exposure assertion fail on a foreign socket — a red that accuses the
# feature for the runner's fault.
#
# The schema probe is informational. The module now reads the health field
# schema-driven and refuses in seconds naming both spellings if neither exists,
# so a spelling mismatch can no longer stall the lane; what the probe still
# supplies is WHICH spelling this podman uses, which no lane in this repo has
# ever established.
#
# Both are pinned anyway, because a lane whose red runs cannot be interpreted
# gets muted, and a muted lane is not a gate.

SCHEMA_PROBE_STEP = "Diagnostic — podman's inspect .State schema (health-field spelling)"
PORT_PREFLIGHT_STEP = "Pre-flight — the module's host ports must be free on the runner"


def test_two_shape_lane_diagnoses_before_it_boots(workflow: dict[str, Any]) -> None:
    """Both steps must exist AND precede the boot.

    Ordering is not stylistic. A step placed after a failing step does not run
    at all unless it carries its own ``if:``, so a pre-flight or a diagnostic
    parked below the boot would be missing from precisely the runs that need
    it — and the pre-flight would additionally be refusing a port collision
    only after the collision had already failed the test.
    """
    names = _step_names(workflow, TWO_SHAPE_JOB)
    for step_name in (PORT_PREFLIGHT_STEP, SCHEMA_PROBE_STEP):
        assert step_name in names, f"missing pre-boot step {step_name!r}; got {names}"
        assert names.index(step_name) < names.index(TWO_SHAPE_BOOT_STEP), (
            f"{step_name!r} must run BEFORE the boot it exists to explain"
        )


@pytest.mark.parametrize("step_name", [PORT_PREFLIGHT_STEP, SCHEMA_PROBE_STEP])
def test_two_shape_lane_diagnoses_before_it_boots__mutation_drops_a_step(step_name: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TWO_SHAPE_JOB]["steps"]
    _jobs(mutated)[TWO_SHAPE_JOB]["steps"] = [s for s in steps if s.get("name") != step_name]
    with pytest.raises(AssertionError, match="missing pre-boot step"):
        test_two_shape_lane_diagnoses_before_it_boots(mutated)


@pytest.mark.parametrize("step_name", [PORT_PREFLIGHT_STEP, SCHEMA_PROBE_STEP])
def test_two_shape_lane_diagnoses_before_it_boots__mutation_moves_a_step_late(
    step_name: str,
) -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TWO_SHAPE_JOB]["steps"]
    moved = next(s for s in steps if s.get("name") == step_name)
    steps.remove(moved)
    steps.append(moved)
    assert step_name in _step_names(mutated, TWO_SHAPE_JOB)  # still present
    with pytest.raises(AssertionError, match="BEFORE the boot"):
        test_two_shape_lane_diagnoses_before_it_boots(mutated)


def test_schema_probe_cannot_red_the_lane(workflow: dict[str, Any]) -> None:
    """It is a diagnostic, not an assertion. Every runtime call it makes is
    guarded and it ends in an explicit ``exit 0``, because reddening a gate
    over a throwaway busybox that would not start — or a registry hiccup
    pulling it — would fail the lane for something that says nothing about the
    feature under test.

    It probes BOTH spellings rather than asserting either: which one this
    podman presents is the open question, so the step's job is to answer it in
    the log, not to prejudge it.
    """
    run_text = _find_named_step(workflow, TWO_SHAPE_JOB, SCHEMA_PROBE_STEP)["run"]
    assert run_text.rstrip().endswith("exit 0"), (
        "the schema probe must end in `exit 0` so it can never fail the lane"
    )
    # The epilogue alone is NOT sufficient, and pinning only it was the gap.
    # `run:` executes under `bash -e`, so an unguarded `podman rm` on a runner
    # where the container never started aborts the step BEFORE the trailing
    # `exit 0` is ever reached — the step reds while still visibly ending in
    # `exit 0`. The guards are what make the epilogue reachable.
    assert run_text.count("|| true") >= 2, (
        "every podman call in the probe must be `|| true`-guarded, or bash -e "
        "aborts before the `exit 0` that is supposed to make this fail-open"
    )
    assert "if podman run" in run_text, (
        "the throwaway's own start must be conditional — a failed image pull is "
        "the most likely way this step does not get to run at all"
    )
    for field in (".State.Health", ".State.Healthcheck"):
        assert field in run_text, f"the probe must report on {field}"


def test_schema_probe_cannot_red_the_lane__mutation_drops_the_exit_guard() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, SCHEMA_PROBE_STEP)
    step["run"] = step["run"].rstrip()[: -len("exit 0")]
    with pytest.raises(AssertionError, match="never fail the lane"):
        test_schema_probe_cannot_red_the_lane(mutated)


def test_schema_probe_cannot_red_the_lane__mutation_drops_the_call_guards() -> None:
    """The subtle half: the step still ENDS in `exit 0`, so the epilogue check
    stays green, but under `bash -e` a failed `podman rm` aborts long before
    that line runs. A fail-open promise is only as good as its guards."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, SCHEMA_PROBE_STEP)
    step["run"] = step["run"].replace(" || true", "")
    assert step["run"].rstrip().endswith("exit 0"), "the epilogue must survive this mutation"
    with pytest.raises(AssertionError, match="bash -e"):
        test_schema_probe_cannot_red_the_lane(mutated)


def test_port_preflight_actually_refuses(workflow: dict[str, Any]) -> None:
    """The pre-flight is the one new step that MUST be able to red the lane.

    It is the mirror image of the schema probe above, and the pair is easy to
    confuse: both inspect the runner before the boot, but a diagnostic that
    fails is a bug while a precondition that cannot fail is decorative. Without
    a non-zero exit this step would print a busy port and carry on into the
    20-minute boot that the busy port has already doomed.
    """
    run_text = _find_named_step(workflow, TWO_SHAPE_JOB, PORT_PREFLIGHT_STEP)["run"]
    assert "exit 1" in run_text, (
        "the port pre-flight must REFUSE on a busy port, not merely report it"
    )


def test_port_preflight_actually_refuses__mutation_softens_the_exit() -> None:
    """Softening `exit 1` to `true` leaves a step that still probes, still
    prints, and never stops anything."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, PORT_PREFLIGHT_STEP)
    step["run"] = step["run"].replace("exit 1", "true")
    with pytest.raises(AssertionError, match="must REFUSE"):
        test_port_preflight_actually_refuses(mutated)


def test_schema_probe_cannot_red_the_lane__mutation_probes_only_one_spelling() -> None:
    """Probing only docker's spelling would report "no health field" on a
    podman that simply calls it something else — the same wrong conclusion the
    step exists to prevent, now printed with authority."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, SCHEMA_PROBE_STEP)
    step["run"] = step["run"].replace(".State.Healthcheck", ".State.Health")
    assert ".State.Health" in step["run"]  # the docker spelling survives
    with pytest.raises(AssertionError, match="Healthcheck"):
        test_schema_probe_cannot_red_the_lane(mutated)


def test_two_shape_lane_cleanup_runs_on_failure(workflow: dict[str, Any]) -> None:
    """Without ``if: always()`` the cleanup is present but skipped on exactly
    the runs it exists for — a failed boot is when something is left behind."""
    cleanup = _jobs(workflow)[TWO_SHAPE_JOB]["steps"][-1]
    assert cleanup.get("if") == "always()", (
        f"the '{TWO_SHAPE_JOB}' cleanup must carry `if: always()`; got {cleanup.get('if')!r}"
    )


def test_two_shape_lane_cleanup_runs_on_failure__mutation_drops_the_condition() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _jobs(mutated)[TWO_SHAPE_JOB]["steps"][-1]["if"]
    with pytest.raises(AssertionError, match="must carry"):
        test_two_shape_lane_cleanup_runs_on_failure(mutated)


def test_two_shape_boot_step_shares_step_zero_environment(workflow: dict[str, Any]) -> None:
    """No ``DOCKER_HOST`` export in the boot step — the one environment delta
    that could ever separate the step that PROVES the provider from the step
    that USES it.

    It cannot flip the provider today: selection is a file with an absolute
    path, and podman-compose drives the ``podman`` CLI, which ignores
    ``DOCKER_HOST``. That is an argument for deleting it rather than keeping
    it. A future provider that DID consult the variable would reintroduce the
    delegation this lane exists to prevent, with step 0 still green because
    step 0 never sees it. The neighbouring multi-user podman lane exports it
    legitimately — it runs the docker-compose plugin, which needs the socket —
    so this guard exists because copying from that lane is the obvious mistake.
    """
    boot_run = _find_named_step(workflow, TWO_SHAPE_JOB, TWO_SHAPE_BOOT_STEP)["run"]
    assert "DOCKER_HOST" not in boot_run, (
        "the boot step must not export DOCKER_HOST — see this test's docstring"
    )
    assert "podman system service" not in boot_run, (
        "the API socket belongs to the docker-compose-plugin lane, not this one"
    )


def test_two_shape_boot_step_shares_step_zero_environment__mutation_readds_the_socket() -> None:
    """Re-adding the neighbour's four lines must fail, whichever half returns."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, TWO_SHAPE_BOOT_STEP)
    step["run"] = (
        'podman system service --time=0 &\nexport DOCKER_HOST="unix://$SOCK"\n' + (step["run"])
    )
    with pytest.raises(AssertionError, match="must not export DOCKER_HOST"):
        test_two_shape_boot_step_shares_step_zero_environment(mutated)


# ---------------------------------------------------------------------------
# The parse-only counterpart: same provider, no containers, in a fast job
# ---------------------------------------------------------------------------


def test_parse_only_step_lives_in_a_fast_job(workflow: dict[str, Any]) -> None:
    """The step's whole value is being cheap enough to run on every push while
    the boot lane is PR-scoped. Parked in a lane that only runs on same-repo
    PRs it would add nothing the boot lane does not already cover, so the host
    job is asserted, not just the step's existence.

    ``static-checks`` qualifies because it carries no ``if:`` at all — it runs
    on pushes, schedules and fork PRs alike.
    """
    assert PARSE_ONLY_STEP in _step_names(workflow, PARSE_ONLY_JOB)
    assert "if" not in _jobs(workflow)[PARSE_ONLY_JOB], (
        f"'{PARSE_ONLY_JOB}' has grown an if: — the parse step no longer runs unconditionally"
    )


def test_parse_only_step_lives_in_a_fast_job__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[PARSE_ONLY_JOB]["steps"]
    _jobs(mutated)[PARSE_ONLY_JOB]["steps"] = [s for s in steps if s.get("name") != PARSE_ONLY_STEP]
    with pytest.raises(AssertionError):
        test_parse_only_step_lives_in_a_fast_job(mutated)


def test_parse_only_step_lives_in_a_fast_job__mutation_gates_the_host_job() -> None:
    """Giving the host job a PR-only ``if:`` must fail: the step would survive
    every other check here while quietly ceasing to run on pushes."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[PARSE_ONLY_JOB]["if"] = "github.event_name == 'pull_request'"
    assert PARSE_ONLY_STEP in _step_names(mutated, PARSE_ONLY_JOB)  # the step survives
    with pytest.raises(AssertionError, match="grown an if:"):
        test_parse_only_step_lives_in_a_fast_job(mutated)


def _parse_step_run(wf: dict[str, Any]) -> str:
    return _find_named_step(wf, PARSE_ONLY_JOB, PARSE_ONLY_STEP)["run"]


def test_parse_only_step_parses_osprey_own_argv(workflow: dict[str, Any]) -> None:
    """What separates this from a YAML lint: the document handed to
    podman-compose is written BY ``compose_base_cmd`` as a side effect of
    building the argv, and the argv is what the step runs. A hand-rolled
    ``podman-compose -f build/services/*.yml config`` would parse something
    real and prove nothing about the merged topology the deploy actually ships.
    """
    run_text = _parse_step_run(workflow)
    assert "compose_base_cmd" in run_text, "the step must run OSPREY's own argv, not a lookalike"
    # Spelled as the argv is built, NOT as a bare `"config" in run_text`: the
    # step's own body contains `load_config("build/config.yml")`, which supplies
    # that substring for free. The bare form stayed green when the subcommand
    # was rewritten to `ps` — an inert half of an otherwise load-bearing pair.
    assert '[*argv, "config"]' in run_text, (
        "the step must run the merged topology through `config`, not another subcommand"
    )


def test_parse_only_step_parses_osprey_own_argv__mutation_hand_rolls_the_argv() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, PARSE_ONLY_JOB, PARSE_ONLY_STEP)
    step["run"] = step["run"].replace("compose_base_cmd", "_hand_rolled_cmd")
    with pytest.raises(AssertionError, match="own argv"):
        test_parse_only_step_parses_osprey_own_argv(mutated)


def test_parse_only_step_parses_osprey_own_argv__mutation_swaps_the_subcommand() -> None:
    """`config` is what parses and interpolates; `ps` merely talks to the
    runtime. Swapping them leaves a step that still builds the real argv and
    still exits 0, having proved nothing about the merged document."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, PARSE_ONLY_JOB, PARSE_ONLY_STEP)
    step["run"] = step["run"].replace('[*argv, "config"]', '[*argv, "ps"]')
    assert "compose_base_cmd" in step["run"]  # the argv half survives untouched
    with pytest.raises(AssertionError, match="not another subcommand"):
        test_parse_only_step_parses_osprey_own_argv(mutated)


def test_parse_only_step_degrades_rather_than_reds_the_tier_0_gate(
    workflow: dict[str, Any],
) -> None:
    """``static-checks`` carries no ``if:`` — it gates every push, schedule and
    fork PR — so podman's absence from a future runner image must cost this
    extra coverage, not the job that gates everything.

    The skip is deliberately NOT the usual vacuous-green hazard, and the reason
    is worth stating: the two-shape lane runs the same pin UNguarded, so a
    runner that genuinely lost podman still reds there on the next PR. This
    guard narrows the blast radius; it does not make the coverage revocable.
    """
    pin_run = _find_named_step(workflow, PARSE_ONLY_JOB, PODMAN_PIN_STEP)["run"]
    assert _PODMAN_PRESENCE_GUARD_RE.search(pin_run), (
        f"the '{PARSE_ONLY_JOB}' pin step must tolerate a runner image without podman"
    )
    lane_pin = _find_named_step(workflow, TWO_SHAPE_JOB, PODMAN_PIN_STEP)["run"]
    assert not _PODMAN_PRESENCE_GUARD_RE.search(lane_pin), (
        f"the '{TWO_SHAPE_JOB}' pin step must NOT be guarded — an unguarded lane is "
        "what keeps the skip above from hiding a genuinely missing podman"
    )


def test_parse_only_step_degrades__mutation_drops_the_guard() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, PARSE_ONLY_JOB, PODMAN_PIN_STEP)
    step["run"] = _PODMAN_PRESENCE_GUARD_RE.sub("true", step["run"])
    with pytest.raises(AssertionError, match="without podman"):
        test_parse_only_step_degrades_rather_than_reds_the_tier_0_gate(mutated)


def test_podman_presence_guard_is_not_satisfied_by_podman_compose() -> None:
    """The lookahead's whole job, pinned directly.

    Both pin steps run ``command -v podman-compose``. A substring check reports
    the LANE as guarded on the strength of that line alone, which inverts the
    assertion above — it would then demand the opposite of what it means.
    """
    assert _PODMAN_PRESENCE_GUARD_RE.search(
        'PODMAN_COMPOSE_BIN="$(command -v podman-compose)"'
    ) is (None)
    for guarded in (
        "command -v podman >/dev/null",
        "command -v podman || exit 0",
        "command -v podman",
    ):
        assert _PODMAN_PRESENCE_GUARD_RE.search(guarded), f"missed {guarded!r}"


def test_parse_only_step_degrades__mutation_guards_the_lane_too() -> None:
    """Guarding the LANE as well would make podman's absence invisible
    everywhere — both halves would skip quietly and the gate would stay green
    with nothing behind it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, TWO_SHAPE_JOB, PODMAN_PIN_STEP)
    step["run"] = "if ! command -v podman >/dev/null 2>&1; then exit 0; fi\n" + step["run"]
    with pytest.raises(AssertionError, match="must NOT be guarded"):
        test_parse_only_step_degrades_rather_than_reds_the_tier_0_gate(mutated)


def test_parse_only_job_has_a_timeout(workflow: dict[str, Any]) -> None:
    """The job now shells out to `osprey init`, `osprey build` and a compose
    parse. None can hang in principle, but all are new shapes for a Tier 0 job
    that otherwise inherits the 6-hour default."""
    assert "timeout-minutes" in _jobs(workflow)[PARSE_ONLY_JOB], (
        f"'{PARSE_ONLY_JOB}' must bound its runtime"
    )


def test_parse_only_job_has_a_timeout__mutation_drops_it() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _jobs(mutated)[PARSE_ONLY_JOB]["timeout-minutes"]
    with pytest.raises(AssertionError, match="must bound its runtime"):
        test_parse_only_job_has_a_timeout(mutated)


def test_parse_only_step_proves_the_env_chain_interpolates(workflow: dict[str, Any]) -> None:
    """The interpolation half, and it is asserted in the direction that can
    fail. Seeding one value would only show that SOME env file was read; the
    step writes the same key into both chain members with different values, so
    the local one appearing — and the shared one not — is what distinguishes a
    merged file that preserved precedence from one that merely exists."""
    run_text = _parse_step_run(workflow)
    for sentinel in ("local-wins", "shared-loses"):
        assert sentinel in run_text, (
            f"the parse step must seed and assert the {sentinel!r} chain sentinel"
        )


def test_parse_only_step_proves_the_env_chain_interpolates__mutation_drops_a_sentinel() -> None:
    """Dropping the losing sentinel leaves a step that still reads an env file
    but can no longer tell which one won."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, PARSE_ONLY_JOB, PARSE_ONLY_STEP)
    step["run"] = step["run"].replace("shared-loses", "local-wins")
    assert "local-wins" in step["run"]  # the other sentinel survives
    with pytest.raises(AssertionError, match="shared-loses"):
        test_parse_only_step_proves_the_env_chain_interpolates(mutated)


# ---------------------------------------------------------------------------
# The proxied build inside dockerfile-e2e
# ---------------------------------------------------------------------------
#
# The nine template Dockerfiles bridge the uppercase proxy variables a facility
# hands a build into the lowercase ones apt reads. Nothing about that bridge is
# observable from a build's exit status on a runner with open egress: delete
# the export and apt simply fetches directly, and the build still succeeds. The
# lane therefore routes a real build through a tinyproxy and asserts on the
# PROXY'S ACCESS LOG, and what is pinned here is that it keeps doing exactly
# that — the failure mode these guards exist for is a sequence that survives as
# a green step while proving nothing.

DOCKERFILE_E2E_JOB = "dockerfile-e2e"
PROXY_START_STEP = "Start a logging forward proxy (tinyproxy)"
PROXY_CONTRACT_STEP = "Assert the uppercase-only proxy contract reaches the build"
PROXY_BUILD_STEP = "Build the web-terminal auth sidecar through the proxy"
PROXY_ASSERT_STEP = "Assert apt and pip traffic went through the proxy"
PROXY_STOP_STEP = "Stop the forward proxy"
PROXY_SEQUENCE = (
    PROXY_START_STEP,
    PROXY_CONTRACT_STEP,
    PROXY_BUILD_STEP,
    PROXY_ASSERT_STEP,
    PROXY_STOP_STEP,
)
#: The host-side path the proxy's access log is mounted to and read back from.
PROXY_ACCESS_LOG = "/tmp/osprey-proxy/access.log"
#: The apt mirror the template Dockerfiles rewrite their sources to. This is
#: the half that proves the BRIDGE: apt reads no uppercase proxy variable, so
#: its traffic can only reach the proxy through the Dockerfile's export.
PROXY_APT_HOST = "deb.debian.org"
#: The pip side — either host is enough to show the framework install was
#: proxied too.
PROXY_PIP_HOSTS = ("pypi.org", "files.pythonhosted.org")
#: The build context: small, and it uses both apt and pip.
PROXIED_BUILD_TARGET = "src/osprey/templates/modules/web_terminals/auth_sidecar"
_UPPERCASE_PROXY_ARG_RE = re.compile(r"--build-arg\s+(HTTP_PROXY|HTTPS_PROXY)=")
_LOWERCASE_PROXY_ARG_RE = re.compile(r"--build-arg\s+(http_proxy|https_proxy)=")


def _proxy_step_run(wf: dict[str, Any], step_name: str) -> str:
    return _find_named_step(wf, DOCKERFILE_E2E_JOB, step_name)["run"]


def test_dockerfile_e2e_carries_the_whole_proxied_sequence(workflow: dict[str, Any]) -> None:
    """All five steps, checked by name. They are one proof split across steps
    for readability, and each is independently droppable in a way that leaves
    the lane green: without the start step the build is unproxied, without the
    contract probe the log assertion can be satisfied by a runtime rather than
    by the Dockerfile, without the assertion the build's exit status is all
    that is left, and without the teardown a fixed-name container keeps port
    8888 on a shared runner."""
    for step_name in PROXY_SEQUENCE:
        _find_named_step(workflow, DOCKERFILE_E2E_JOB, step_name)


@pytest.mark.parametrize("dropped", PROXY_SEQUENCE)
def test_dockerfile_e2e_carries_the_whole_proxied_sequence__mutation_drops_one(
    dropped: str,
) -> None:
    """Dropping any single step must fail — otherwise the check is satisfied
    by the other four and the sequence can be dismantled one step at a time."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[DOCKERFILE_E2E_JOB]["steps"]
    steps.remove(_find_named_step(mutated, DOCKERFILE_E2E_JOB, dropped))
    with pytest.raises(AssertionError, match=re.escape(dropped)):
        test_dockerfile_e2e_carries_the_whole_proxied_sequence(mutated)


def test_proxy_assertion_reads_the_access_log(workflow: dict[str, Any]) -> None:
    """The assertion must be about LOG CONTENT. On this runner the internet is
    directly reachable, so a build whose bridge was deleted exits 0 all the
    same: an exit-status check would be a green step over a broken bridge.
    Both host classes are required, and separately — apt proves the lowercase
    bridge, pip proves the proxy was engaged for the framework install."""
    run_text = _proxy_step_run(workflow, PROXY_ASSERT_STEP)
    assert PROXY_ACCESS_LOG in run_text, (
        f"'{PROXY_ASSERT_STEP}' must read the proxy's access log at {PROXY_ACCESS_LOG}"
    )
    assert PROXY_APT_HOST in run_text, (
        f"'{PROXY_ASSERT_STEP}' must assert that {PROXY_APT_HOST} traffic reached the proxy"
    )
    assert any(host in run_text for host in PROXY_PIP_HOSTS), (
        f"'{PROXY_ASSERT_STEP}' must assert that pip traffic ({' or '.join(PROXY_PIP_HOSTS)}) "
        f"reached the proxy"
    )


def test_proxy_assertion_reads_the_access_log__mutation_checks_exit_status_instead() -> None:
    """The shape this guard exists for: an assertion step reduced to "the build
    exited 0", which on an open-egress runner is true whether or not anything
    was proxied."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_ASSERT_STEP)
    step["run"] = 'set -euo pipefail\necho "✅ the proxied build exited 0"\n'
    with pytest.raises(AssertionError, match="access log"):
        test_proxy_assertion_reads_the_access_log(mutated)


def test_proxy_assertion_reads_the_access_log__mutation_drops_only_the_apt_host() -> None:
    """Losing the apt half must fail on its own. It is the half that proves the
    bridge — pip honours the uppercase names unaided, so a pip-only assertion
    would stay green with every `export http_proxy=…` line deleted."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_ASSERT_STEP)
    original = step["run"]
    step["run"] = original.replace(PROXY_APT_HOST, "example.invalid")
    assert step["run"] != original, f"no {PROXY_APT_HOST} in the step — mutation is stale"
    assert any(host in step["run"] for host in PROXY_PIP_HOSTS)  # the pip half survives
    with pytest.raises(AssertionError, match=PROXY_APT_HOST):
        test_proxy_assertion_reads_the_access_log(mutated)


def test_proxy_assertion_reads_the_access_log__mutation_drops_only_the_pip_hosts() -> None:
    """And losing the pip half must fail too, so the two are independent."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_ASSERT_STEP)
    original = step["run"]
    for host in PROXY_PIP_HOSTS:
        step["run"] = step["run"].replace(host, "example.invalid")
    assert step["run"] != original, "no pip host in the step — mutation is stale"
    assert PROXY_APT_HOST in step["run"]  # the apt half survives
    with pytest.raises(AssertionError, match="pip traffic"):
        test_proxy_assertion_reads_the_access_log(mutated)


def test_proxied_build_passes_only_uppercase_proxy_build_args(workflow: dict[str, Any]) -> None:
    """Uppercase in, and nothing else. The lowercase spelling is what the
    Dockerfile itself is supposed to derive; handing it to the build directly
    would satisfy apt without the export ever running, which is the same
    vacuous green as asserting on exit status."""
    run_text = _proxy_step_run(workflow, PROXY_BUILD_STEP)
    passed = {match.group(1) for match in _UPPERCASE_PROXY_ARG_RE.finditer(run_text)}
    assert passed == {"HTTP_PROXY", "HTTPS_PROXY"}, (
        f"'{PROXY_BUILD_STEP}' must pass both uppercase proxy build-args; got {sorted(passed)}"
    )
    lowercase = _LOWERCASE_PROXY_ARG_RE.search(run_text)
    assert lowercase is None, (
        f"'{PROXY_BUILD_STEP}' passes a lowercase proxy build-arg "
        f"({lowercase.group(1) if lowercase else ''}) — the Dockerfile's bridge must be the "
        f"only source of it, or the lane proves nothing"
    )


def test_proxied_build_passes_only_uppercase_proxy_build_args__mutation_drops_https() -> None:
    """Dropping either uppercase arg must fail: pip and apt both reach their
    hosts over HTTPS, so HTTP_PROXY alone routes neither."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_BUILD_STEP)
    original = step["run"]
    step["run"] = re.sub(r"\s*--build-arg\s+HTTPS_PROXY=\S+\s*\\\n", "\n", original)
    assert step["run"] != original, "HTTPS_PROXY build-arg not found — mutation is stale"
    with pytest.raises(AssertionError, match="both uppercase proxy build-args"):
        test_proxied_build_passes_only_uppercase_proxy_build_args(mutated)


def test_proxied_build_passes_only_uppercase_proxy_build_args__mutation_adds_lowercase() -> None:
    """Adding the lowercase spelling alongside is the plausible "just make the
    build work" edit, and it silently retires the thing under test."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_BUILD_STEP)
    original = step["run"]
    step["run"] = original.replace(
        "--build-arg HTTP_PROXY=",
        "--build-arg http_proxy=http://127.0.0.1:8888 \\\n  --build-arg HTTP_PROXY=",
    )
    assert step["run"] != original, "HTTP_PROXY build-arg not found — mutation is stale"
    with pytest.raises(AssertionError, match="lowercase proxy build-arg"):
        test_proxied_build_passes_only_uppercase_proxy_build_args(mutated)


def test_proxied_build_uses_a_case_preserving_runtime(workflow: dict[str, Any]) -> None:
    """Measured, not assumed: docker's BuildKit expands `--build-arg
    HTTP_PROXY=…` into BOTH `HTTP_PROXY` and `http_proxy` inside RUN, so a
    docker build would satisfy the access-log assertion with the Dockerfile's
    bridge deleted. podman (buildah) passes the names it is given. The runtime
    is therefore part of the proof, and the contract probe step re-checks it on
    the runner every run."""
    run_text = _proxy_step_run(workflow, PROXY_BUILD_STEP)
    assert "podman build" in run_text, (
        f"'{PROXY_BUILD_STEP}' must build with podman — see the docstring"
    )
    assert "docker build" not in run_text, (
        f"'{PROXY_BUILD_STEP}' must not build with docker: BuildKit supplies the lowercase "
        f"proxy variables itself, which is precisely what the Dockerfile bridge is for"
    )


def test_proxied_build_uses_a_case_preserving_runtime__mutation_switches_to_docker() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_BUILD_STEP)
    original = step["run"]
    step["run"] = original.replace("podman build", "docker build")
    assert step["run"] != original, "no podman build in the step — mutation is stale"
    with pytest.raises(AssertionError):
        test_proxied_build_uses_a_case_preserving_runtime(mutated)


def test_contract_probe_rejects_a_runtime_that_injects_lowercase(workflow: dict[str, Any]) -> None:
    """The probe is what keeps the whole sequence honest across runtime
    upgrades, so its two halves are pinned: the uppercase value must be seen to
    arrive, and a lowercase one must be seen NOT to. A probe that only checked
    the first would go on passing on a runtime that supplies both."""
    run_text = _proxy_step_run(workflow, PROXY_CONTRACT_STEP)
    assert "HTTP_PROXY=" in run_text, (
        f"'{PROXY_CONTRACT_STEP}' must assert the uppercase value reaches the RUN environment"
    )
    assert "http_proxy=" in run_text, (
        f"'{PROXY_CONTRACT_STEP}' must fail when the runtime injects a lowercase proxy "
        f"variable of its own — without that half the log assertion can be satisfied by "
        f"the build runtime instead of by the Dockerfile"
    )


def test_contract_probe_rejects_a_runtime_that_injects_lowercase__mutation_drops_the_half() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_CONTRACT_STEP)
    original = step["run"]
    step["run"] = "\n".join(line for line in original.splitlines() if "http_proxy=" not in line)
    assert step["run"] != original, "no lowercase check in the probe — mutation is stale"
    assert "HTTP_PROXY=" in step["run"]  # the uppercase half survives
    with pytest.raises(AssertionError, match="lowercase proxy"):
        test_contract_probe_rejects_a_runtime_that_injects_lowercase(mutated)


def test_proxied_build_threads_the_version_args(workflow: dict[str, Any]) -> None:
    """The sidecar Dockerfile exits non-zero on an empty ``OSPREY_VERSION``,
    and a branch version is usually not on PyPI — ``OSPREY_DEV=1`` is what
    turns that pin miss into an unpinned prime. Both are threaded explicitly so
    the proxy bridge stays the only thing that can fail in this build."""
    run_text = _proxy_step_run(workflow, PROXY_BUILD_STEP)
    assert "--build-arg OSPREY_VERSION=" in run_text, (
        f"'{PROXY_BUILD_STEP}' must pass OSPREY_VERSION; the Dockerfile requires it"
    )
    assert "--build-arg OSPREY_DEV=1" in run_text, (
        f"'{PROXY_BUILD_STEP}' must pass OSPREY_DEV=1 so an unreleased pin falls back "
        f"to an unpinned prime instead of failing the build"
    )
    assert PROXIED_BUILD_TARGET in run_text, (
        f"'{PROXY_BUILD_STEP}' must build from {PROXIED_BUILD_TARGET}"
    )


@pytest.mark.parametrize(
    ("dropped", "message"),
    [
        ("--build-arg OSPREY_DEV=1", "OSPREY_DEV=1"),
        ("--build-arg OSPREY_VERSION=", "OSPREY_VERSION"),
    ],
)
def test_proxied_build_threads_the_version_args__mutation_drops_one(
    dropped: str, message: str
) -> None:
    """Either one missing turns a bridge failure and a version failure into the
    same red step, which is the reading problem this threading exists to avoid."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_BUILD_STEP)
    original = step["run"]
    step["run"] = original.replace(dropped, "--build-arg UNRELATED=")
    assert step["run"] != original, f"{dropped!r} not found — mutation is stale"
    with pytest.raises(AssertionError, match=message):
        test_proxied_build_threads_the_version_args(mutated)


def test_every_proxied_step_sets_pipefail(workflow: dict[str, Any]) -> None:
    """GitHub's default shell is ``bash -e`` WITHOUT pipefail, so a pipeline
    reports its last command's status. These steps grep for evidence of
    absence, which is exactly the shape a swallowed status turns into a silent
    pass."""
    for step_name in PROXY_SEQUENCE:
        run_text = _proxy_step_run(workflow, step_name)
        assert "set -euo pipefail" in run_text, f"'{step_name}' must open with `set -euo pipefail`"


def test_every_proxied_step_sets_pipefail__mutation_drops_it_from_one() -> None:
    """One step losing it is enough to fail — the guard must not be satisfied
    by its neighbours."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_ASSERT_STEP)
    original = step["run"]
    step["run"] = original.replace("set -euo pipefail", "set -e")
    assert step["run"] != original, "no pipefail in the step — mutation is stale"
    with pytest.raises(AssertionError, match=re.escape(PROXY_ASSERT_STEP)):
        test_every_proxied_step_sets_pipefail(mutated)


def test_proxy_teardown_runs_even_when_the_assertion_fails(workflow: dict[str, Any]) -> None:
    """The proxy is a fixed-name container holding a fixed port on a runner
    other steps share. A teardown that only runs on success leaves it behind
    for precisely the run that failed."""
    step = _find_named_step(workflow, DOCKERFILE_E2E_JOB, PROXY_STOP_STEP)
    assert step.get("if") == "always()", (
        f"'{PROXY_STOP_STEP}' must carry `if: always()`; got {step.get('if')!r}"
    )


def test_proxy_teardown_runs_even_when_the_assertion_fails__mutation_drops_the_condition() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, DOCKERFILE_E2E_JOB, PROXY_STOP_STEP)["if"]
    with pytest.raises(AssertionError, match="always"):
        test_proxy_teardown_runs_even_when_the_assertion_fails(mutated)


# ---------------------------------------------------------------------------
# (p) privilege-split-e2e gates the merge, and the gate has no orphan jobs
# ---------------------------------------------------------------------------

PRIVSPLIT_JOB = "privilege-split-e2e"

#: Jobs deliberately OUTSIDE ``all-checks-passed.needs``. Every entry needs a
#: reason: ``channel-finder-benchmarks`` is a manually dispatched quality score
#: (pinned by ``test_benchmarks_job_is_dispatch_gated_and_lane_ignores_it``),
#: ``tier2-model-benchmark`` is a manually dispatched model benchmark (pinned
#: by ``test_tier2_job_is_dispatch_gated_and_lane_ignores_it``), and the gate
#: cannot depend on itself. Anything else that is not in
#: ``needs`` can go red forever inside a green check — the 17 privilege-split
#: Docker tests did exactly that for a phase.
JOBS_OUTSIDE_THE_GATE = frozenset({BENCHMARKS_JOB, TIER2_JOB, GATE_JOB})


def _jobs_missing_from_the_gate(wf: dict[str, Any]) -> list[str]:
    needs = set(_jobs(wf)[GATE_JOB]["needs"])
    return sorted(set(_jobs(wf)) - needs - JOBS_OUTSIDE_THE_GATE)


def test_every_job_is_gated_or_deliberately_excluded(workflow: dict[str, Any]) -> None:
    """The reverse of ``test_gate_checks_every_needed_job``: that one proves
    ``needs`` ⊆ checked, this one proves jobs ⊆ ``needs`` ∪ allowlist. A new
    lane that nobody adds to ``needs`` is otherwise invisible — the roll-up
    stays green whatever the lane does."""
    assert _jobs_missing_from_the_gate(workflow) == []


def test_every_job_is_gated_or_deliberately_excluded__mutation_adds_orphan_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)["orphan-lane"] = {"runs-on": "ubuntu-latest", "steps": []}
    assert _jobs_missing_from_the_gate(mutated) == ["orphan-lane"]


# ---------------------------------------------------------------------------
# (q) posture-toggle browser test: named in the browser lane (same vacuous-
# green shape as (h)/(h2) — the unit lane skips it without chromium)
# ---------------------------------------------------------------------------

POSTURE_BROWSER_TEST_FILE = "tests/interfaces/web_terminal/test_posture_toggle_browser.py"


def test_posture_toggle_browser_test_runs_in_the_browser_lane(workflow: dict[str, Any]) -> None:
    """The toggle's only end-to-end proof (badge agrees with the store)
    must be NAMED in the lane's explicit file list; nowhere else in CI has
    Chromium installed."""
    assert POSTURE_BROWSER_TEST_FILE in _browser_lane_files(workflow)


def test_posture_toggle_browser_test_runs_in_the_browser_lane__mutation_drops_the_file() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, BROWSER_JOB, BROWSER_RUN_STEP)
    step["run"] = step["run"].replace(f"{POSTURE_BROWSER_TEST_FILE} \\\n", "")
    assert POSTURE_BROWSER_TEST_FILE not in _browser_lane_files(mutated)


# The control-target switch lane: registered in both halves of the gate
# ---------------------------------------------------------------------------
#
# This lane needs no secrets — two Channel Access servers and a deterministic
# switch between them — which makes it the easiest lane in the file to add and
# the easiest to leave half-wired. The triple below is the same one every other
# lane carries, for the same reason: a lane nothing reads is a lane nobody
# notices going red.


# ---------------------------------------------------------------------------
# The agent-driven control-target switch lane: the same triple, one tier riskier
# ---------------------------------------------------------------------------
#
# Its sibling above needs no secrets; this one needs the LLM key AND the SDK's
# pinned CLI, which is two more ways to go vacuously green. Two of those are
# covered elsewhere in this file and both cover this lane by construction: the
# ``CLI_DEPENDENT_JOBS`` parametrization (the pinned-CLI step) and
# ``_als_apg_probe_steps`` discovery (the pre-flight probe honours
# ``ALS_APG_BASE_URL``). What is left is the gate wiring, pinned here.


# ---------------------------------------------------------------------------
# The live virtual-accelerator lanes: every suite in tests/va/e2e/ is named by
# exactly one of them
# ---------------------------------------------------------------------------
#
# The directory's conftest skips every test in it unless
# ``OSPREY_VA_E2E_ENABLE`` is set, so a module here that no lane names is
# collected on every unit cell, green on every unit cell, and has never run a
# line of its own body. The lanes' explicit file lists are therefore checked
# against a glob over the tree rather than against each other: a module added
# to the directory is red here rather than silently never executed.

VA_LIVE_SUITE_DIR = "tests/va/e2e"


def _va_live_suites_on_disk() -> list[str]:
    """Every ``tests/va/e2e/test_*.py``, repo-relative, POSIX-spelled."""
    return sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (TESTS_ROOT / "va" / "e2e").glob("test_*.py")
        if "__pycache__" not in path.parts
    )


def _va_live_lane_named_files(wf: dict[str, Any], job_name: str) -> list[str]:
    """The pytest file arguments of *job_name*'s run step, one token at a time.

    Split on whitespace rather than searched for as substrings, for
    ``_browser_lane_named_files``' reason: a substring search lets
    ``test_full_sweep.py`` be satisfied by a file whose name merely ends with
    it.

    A list rather than a set: a file named twice in one lane is a duplicate
    the partition check below must still see.
    """
    run = _find_named_step(wf, job_name, VA_LIVE_LANE_RUN_STEPS[job_name])["run"]
    return [token for token in run.split() if token.endswith(".py")]


def _va_live_suites_named_by_any_lane(wf: dict[str, Any]) -> list[str]:
    return [f for job in VA_LIVE_LANE_RUN_STEPS for f in _va_live_lane_named_files(wf, job)]


def test_va_live_suite_discovery_has_a_floor() -> None:
    """A glob that finds nothing would make the guards below vacuous rather
    than red. One anchor per lane, so the floor covers all three parts of the
    partition rather than whichever part happens to be largest."""
    found = _va_live_suites_on_disk()
    assert f"{VA_LIVE_SUITE_DIR}/test_target_switch.py" in found, found
    assert f"{VA_LIVE_SUITE_DIR}/test_substrate_parity.py" in found, found
    assert f"{VA_LIVE_SUITE_DIR}/test_bluesky_lanes.py" in found, found


def test_every_live_va_suite_is_named_by_one_lane(workflow: dict[str, Any]) -> None:
    """The covering half of the partition, as one equality rather than two
    containment checks, so it catches both directions at once: a module on
    disk that no lane runs, and a lane naming a path that is not there."""
    named = _va_live_suites_named_by_any_lane(workflow)
    on_disk = _va_live_suites_on_disk()
    unrun = sorted(set(on_disk) - set(named))
    absent = sorted(set(named) - set(on_disk))
    assert sorted(named) == on_disk, (
        f"suites on disk that no live lane runs (the unit lane collects them and skips every "
        f"one, so nothing executes them anywhere): {unrun}; "
        f"paths named by a live lane that do not exist: {absent}"
    )


def test_every_live_va_suite_is_named_by_one_lane__mutation_drops_one_suite() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    step["run"] = step["run"].replace(f"{VA_LIVE_SUITE_DIR}/test_full_sweep.py ", "")
    with pytest.raises(AssertionError):
        test_every_live_va_suite_is_named_by_one_lane(mutated)


def test_no_live_va_suite_is_named_by_two_lanes(workflow: dict[str, Any]) -> None:
    """The disjointness half, and the reason the helper returns a list.

    Naming a module in two lanes is the cheap way to make the covering check
    above pass, and it buys a second container boot and a second copy of every
    assertion for nothing.
    """
    named = _va_live_suites_named_by_any_lane(workflow)
    repeats = sorted({f for f in named if named.count(f) > 1})
    assert sorted(named) == sorted(set(named)), (
        f"suites named by more than one live lane: {repeats}"
    )


def test_no_live_va_suite_is_named_by_two_lanes__mutation_names_a_suite_in_both_lanes() -> None:
    """Both halves fail on this mutation, which is what pins that they are
    genuinely independent rather than one check written twice."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    step["run"] = f"{step['run']} {VA_LIVE_SUITE_DIR}/test_target_switch.py\n"
    with pytest.raises(AssertionError):
        test_no_live_va_suite_is_named_by_two_lanes(mutated)
    with pytest.raises(AssertionError):
        test_every_live_va_suite_is_named_by_one_lane(mutated)


# The chain's boot half runs in the lane whose job builds the image it boots.
# The partition above is satisfied by any one lane naming the module, so the
# lane is pinned here by name, with the collection floor the run step's
# comment promises of every module it names.
CHAIN_BOOT_MODULE = f"{VA_LIVE_SUITE_DIR}/test_chain_boot.py"
_COLLECTED_FLOOR_RE = re.compile(r"^MIN_COLLECTED_TESTS = (\d+)$", re.MULTILINE)
_CHAIN_BOOT_FLOOR_GUARD = "assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)"


def test_chain_boot_suite_runs_in_the_va_live_run_step(
    workflow: dict[str, Any] | None = None,
) -> None:
    """The VA live job's run step names the chain's boot module by path."""
    wf = workflow if workflow is not None else _load_workflow()
    assert (REPO_ROOT / CHAIN_BOOT_MODULE).is_file()
    named = _va_live_lane_named_files(wf, VA_LIVE_JOB)
    assert CHAIN_BOOT_MODULE in named, f"'{VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB]}' runs {named}"


def test_chain_boot_suite_runs_in_the_va_live_run_step__mutation_moves_it_to_another_lane() -> None:
    """The partition still holds on this mutation, so only the pin sees it."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    step["run"] = step["run"].replace(f"{CHAIN_BOOT_MODULE} ", "")
    other = _find_named_step(mutated, TARGET_SWITCH_JOB, VA_LIVE_LANE_RUN_STEPS[TARGET_SWITCH_JOB])
    other["run"] = f"{other['run']} {CHAIN_BOOT_MODULE}\n"
    test_every_live_va_suite_is_named_by_one_lane(mutated)
    test_no_live_va_suite_is_named_by_two_lanes(mutated)
    with pytest.raises(AssertionError):
        test_chain_boot_suite_runs_in_the_va_live_run_step(mutated)


def _collected_floor_errors(source: str, *, guard: str) -> list[str]:
    """What a module's collection floor is missing, read from its text.

    Read rather than imported: a lane module may resolve its lane at import,
    and the floor is a fact about its source. ``guard`` is the text of the
    module's own assertion of its collection against the floor.
    """
    errors: list[str] = []
    match = _COLLECTED_FLOOR_RE.search(source)
    if match is None or int(match.group(1)) < 1:
        errors.append("declares no positive MIN_COLLECTED_TESTS")
    if guard not in source:
        errors.append("never asserts its collection against MIN_COLLECTED_TESTS")
    return errors


def test_chain_boot_module_carries_its_collection_floor() -> None:
    source = (REPO_ROOT / CHAIN_BOOT_MODULE).read_text(encoding="utf-8")
    assert _collected_floor_errors(source, guard=_CHAIN_BOOT_FLOOR_GUARD) == []


def test_chain_boot_module_carries_its_collection_floor__mutation_drops_the_floor() -> None:
    source = (REPO_ROOT / CHAIN_BOOT_MODULE).read_text(encoding="utf-8")
    guard = _CHAIN_BOOT_FLOOR_GUARD
    assert _collected_floor_errors(_COLLECTED_FLOOR_RE.sub("", source), guard=guard)
    assert _collected_floor_errors(source.replace(guard, "collected"), guard=guard)


# The measurement tools' live proofs run in the same lane as the chain: the
# guarded runs of `pyaml_measure` and check B. Each is pinned to the run step by
# path and carries the collection floor the step's comment promises.
MEASUREMENT_MODULES = (
    f"{VA_LIVE_SUITE_DIR}/test_pyaml_measure_guarded.py",
    f"{VA_LIVE_SUITE_DIR}/test_check_b_served.py",
)


@pytest.mark.parametrize("module", MEASUREMENT_MODULES)
def test_each_measurement_suite_runs_in_the_va_live_run_step(
    module: str, workflow: dict[str, Any]
) -> None:
    assert (REPO_ROOT / module).is_file()
    named = _va_live_lane_named_files(workflow, VA_LIVE_JOB)
    assert module in named, f"'{VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB]}' runs {named}"


@pytest.mark.parametrize("module", MEASUREMENT_MODULES)
def test_each_measurement_suite_runs_in_the_va_live_run_step__mutation_drops_it(
    module: str,
) -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    step["run"] = step["run"].replace(f"{module} ", "")
    with pytest.raises(AssertionError):
        test_each_measurement_suite_runs_in_the_va_live_run_step(module, mutated)


@pytest.mark.parametrize("module", MEASUREMENT_MODULES)
def test_each_measurement_module_carries_its_collection_floor(module: str) -> None:
    source = (REPO_ROOT / module).read_text(encoding="utf-8")
    assert _collected_floor_errors(source, guard=_CHAIN_BOOT_FLOOR_GUARD) == []


@pytest.mark.parametrize("job_name", sorted(VA_LIVE_LANE_RUN_STEPS))
def test_live_va_lane_enables_the_directory_flag(
    job_name: str, workflow: dict[str, Any] | None = None
) -> None:
    """Without the flag the directory conftest skips every test the lane
    names, and the lane goes green having built an image and measured nothing.

    The value is compared as a string because YAML would otherwise parse a
    bare ``1`` to an integer the shell never sees as the flag's expected
    value.
    """
    wf = workflow if workflow is not None else _load_workflow()
    step = _find_named_step(wf, job_name, VA_LIVE_LANE_RUN_STEPS[job_name])
    value = step.get("env", {}).get(VA_E2E_ENABLE_FLAG)
    assert value == "1", (
        f"{job_name}: the run step must set {VA_E2E_ENABLE_FLAG} to the string '1'; got {value!r}"
    )


@pytest.mark.parametrize("job_name", sorted(VA_LIVE_LANE_RUN_STEPS))
def test_live_va_lane_enables_the_directory_flag__mutation_drops_the_flag(job_name: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, job_name, VA_LIVE_LANE_RUN_STEPS[job_name])
    del step["env"][VA_E2E_ENABLE_FLAG]
    with pytest.raises(AssertionError):
        test_live_va_lane_enables_the_directory_flag(job_name, mutated)


@pytest.mark.parametrize("job_name", sorted(VA_LIVE_LANE_RUN_STEPS))
def test_live_va_lane_fails_on_any_skipped_test(
    job_name: str, workflow: dict[str, Any] | None = None
) -> None:
    """Each lane's zero-skip gate, pinned by content rather than by step name.

    The gate is what turns every way a lane can skip into a red, and pytest's
    exit code cannot express the difference: a fully skipped run exits 0. A
    gate reading a report the run step does not write is a gate over a file
    that is not there, so the two halves have to be pinned as one fact.
    """
    wf = workflow if workflow is not None else _load_workflow()
    run_step = _find_named_step(wf, job_name, VA_LIVE_LANE_RUN_STEPS[job_name])
    reports = _junit_reports_written_by([run_step])
    assert len(reports) == 1, (
        f"{job_name}: the run step must write exactly one --junitxml report; got {reports}"
    )
    gate = _find_named_step(wf, job_name, VA_LIVE_SKIP_GATE_STEP)["run"]
    assert reports[0] in gate, f"'{VA_LIVE_SKIP_GATE_STEP}' in {job_name} never reads {reports[0]}"
    assert 'get("skipped"' in gate, f"{job_name}: the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, f"{job_name}: the gate must fail the job, not just print"


@pytest.mark.parametrize("job_name", sorted(VA_LIVE_LANE_RUN_STEPS))
def test_live_va_lane_fails_on_any_skipped_test__mutation_drops_the_gate(job_name: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[job_name]
    job["steps"] = [s for s in job["steps"] if s.get("name") != VA_LIVE_SKIP_GATE_STEP]
    with pytest.raises(AssertionError):
        test_live_va_lane_fails_on_any_skipped_test(job_name, mutated)


# ---------------------------------------------------------------------------
# The real-facility lane, on the committed NSLS-II stand-in
# ---------------------------------------------------------------------------
#
# The real-facility lane installs an export that never enters the repository,
# so in CI it runs over a committed stand-in tree: the same import, build and
# boot, on the VA live job's image. It is a step of its own rather than a path
# in the run step above, because that step's suites are the tests/va/e2e
# partition and this module is not one of them. Without its variable the
# module skips by design, so its report goes through the same zero-skip gate.

ALS_LANE_STEP = "Run the real-facility lane on the NSLS-II stand-in"
ALS_LANE_MODULE = "tests/cli/test_als_lane.py"
ALS_STAND_IN_ENV = "OSPREY_ALS_LANE_STAND_IN"
ALS_STAND_IN = "tests/fixtures/mml/nsls2"
_ALS_LANE_FLOOR_GUARD = "len(collected) >= MIN_COLLECTED_TESTS"


def test_als_lane_runs_on_the_stand_in_in_the_va_live_job(
    workflow: dict[str, Any] | None = None,
) -> None:
    """The step exists, runs the lane module alone, and names the committed stand-in."""
    wf = workflow if workflow is not None else _load_workflow()
    step = _find_named_step(wf, VA_LIVE_JOB, ALS_LANE_STEP)
    named = [token for token in step["run"].split() if token.endswith(".py")]
    assert named == [ALS_LANE_MODULE], f"'{ALS_LANE_STEP}' runs {named}"
    assert "uv run pytest" in step["run"], step["run"]
    value = step.get("env", {}).get(ALS_STAND_IN_ENV)
    assert value == ALS_STAND_IN, f"'{ALS_LANE_STEP}' sets {ALS_STAND_IN_ENV} to {value!r}"
    assert (REPO_ROOT / ALS_STAND_IN / "imported" / "mml" / "mapping.yaml").is_file()

    names = _step_names(wf, VA_LIVE_JOB)
    image = names.index(next(n for n in names if n.startswith("Build the virtual accelerator")))
    assert names.index(ALS_LANE_STEP) > image, "the lane boots the image this job builds"


def test_als_lane_runs_on_the_stand_in__mutation_drops_the_variable() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, VA_LIVE_JOB, ALS_LANE_STEP)["env"][ALS_STAND_IN_ENV]
    with pytest.raises(AssertionError):
        test_als_lane_runs_on_the_stand_in_in_the_va_live_job(mutated)


def test_als_lane_runs_on_the_stand_in__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[VA_LIVE_JOB]
    job["steps"] = [s for s in job["steps"] if s.get("name") != ALS_LANE_STEP]
    with pytest.raises(AssertionError):
        test_als_lane_runs_on_the_stand_in_in_the_va_live_job(mutated)


def test_als_lane_report_goes_through_the_zero_skip_gate(
    workflow: dict[str, Any] | None = None,
) -> None:
    """One report of its own, read by the gate beside the VA run step's."""
    wf = workflow if workflow is not None else _load_workflow()
    reports = _junit_reports_written_by([_find_named_step(wf, VA_LIVE_JOB, ALS_LANE_STEP)])
    assert len(reports) == 1, f"'{ALS_LANE_STEP}' must write one --junitxml report; got {reports}"
    run_step = _find_named_step(wf, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    assert reports[0] not in _junit_reports_written_by([run_step]), reports
    gate = _find_named_step(wf, VA_LIVE_JOB, VA_LIVE_SKIP_GATE_STEP)["run"]
    assert reports[0] in gate, f"'{VA_LIVE_SKIP_GATE_STEP}' never reads {reports[0]}"


def test_als_lane_report_goes_through_the_zero_skip_gate__mutation_drops_it() -> None:
    mutated = copy.deepcopy(_load_workflow())
    (report,) = _junit_reports_written_by([_find_named_step(mutated, VA_LIVE_JOB, ALS_LANE_STEP)])
    gate = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_SKIP_GATE_STEP)
    gate["run"] = gate["run"].replace(report, "")
    with pytest.raises(AssertionError):
        test_als_lane_report_goes_through_the_zero_skip_gate(mutated)


def test_als_lane_module_carries_its_collection_floor() -> None:
    source = (REPO_ROOT / ALS_LANE_MODULE).read_text(encoding="utf-8")
    assert _collected_floor_errors(source, guard=_ALS_LANE_FLOOR_GUARD) == []


def test_als_lane_module_carries_its_collection_floor__mutation_drops_the_floor() -> None:
    source = (REPO_ROOT / ALS_LANE_MODULE).read_text(encoding="utf-8")
    guard = _ALS_LANE_FLOOR_GUARD
    assert _collected_floor_errors(_COLLECTED_FLOOR_RE.sub("", source), guard=guard)
    assert _collected_floor_errors(source.replace(guard, "collected"), guard=guard)


#: Minutes the VA live job keeps for everything outside its two pytest steps:
#: checkout, uv sync, the image build (a few minutes on a runner), the skip gate
#: and the capture step that must run after a step times out.
VA_LIVE_OUTSIDE_STEPS_MINUTES = 10


def test_va_live_step_caps_leave_the_job_room_for_the_build_and_the_capture(
    workflow: dict[str, Any] | None = None,
) -> None:
    """The two pytest steps' caps plus the outside budget fit under the job cap.

    The job has two heavy steps, so the generic one-heavy-step check passes it
    by. A step that hangs must fail by its own cap, with the image build
    already spent and the capture step still to run.
    """
    wf = workflow if workflow is not None else _load_workflow()
    job_cap = _jobs(wf)[VA_LIVE_JOB]["timeout-minutes"]
    run_step = _find_named_step(wf, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    als_step = _find_named_step(wf, VA_LIVE_JOB, ALS_LANE_STEP)
    run_cap = run_step.get("timeout-minutes")
    als_cap = als_step.get("timeout-minutes")
    assert isinstance(run_cap, int), f"'{run_step['name']}' has no int timeout-minutes: {run_cap!r}"
    assert isinstance(als_cap, int), f"'{ALS_LANE_STEP}' has no int timeout-minutes: {als_cap!r}"
    assert run_cap + als_cap + VA_LIVE_OUTSIDE_STEPS_MINUTES <= job_cap, (
        f"run step {run_cap} + real-facility step {als_cap} + outside "
        f"{VA_LIVE_OUTSIDE_STEPS_MINUTES} exceeds the job cap {job_cap}"
    )


def test_va_live_step_caps_leave_the_job_room__mutation_drops_the_real_facility_step_cap() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, VA_LIVE_JOB, ALS_LANE_STEP)["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_va_live_step_caps_leave_the_job_room_for_the_build_and_the_capture(mutated)


def test_va_live_step_caps_leave_the_job_room__mutation_raises_the_run_step_to_the_job_cap() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    job_cap = _jobs(mutated)[VA_LIVE_JOB]["timeout-minutes"]
    run_step = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    run_step["timeout-minutes"] = job_cap
    with pytest.raises(AssertionError):
        test_va_live_step_caps_leave_the_job_room_for_the_build_and_the_capture(mutated)


def test_va_live_step_caps_leave_the_job_room__mutation_lowers_the_job_to_the_two_steps() -> None:
    mutated = copy.deepcopy(_load_workflow())
    run_step = _find_named_step(mutated, VA_LIVE_JOB, VA_LIVE_LANE_RUN_STEPS[VA_LIVE_JOB])
    als_step = _find_named_step(mutated, VA_LIVE_JOB, ALS_LANE_STEP)
    job = _jobs(mutated)[VA_LIVE_JOB]
    job["timeout-minutes"] = run_step["timeout-minutes"] + als_step["timeout-minutes"]
    with pytest.raises(AssertionError):
        test_va_live_step_caps_leave_the_job_room_for_the_build_and_the_capture(mutated)


# ---------------------------------------------------------------------------
# The virtual accelerator image: one recipe, four lanes
# ---------------------------------------------------------------------------
#
# Four lanes boot a container of this image, and Actions jobs are isolated, so
# each of them builds one. What they share is the recipe and the layer cache,
# and every way of breaking that sharing is silent: a lane building under a
# different tag boots nothing and fails somewhere else, a lane whose cache key
# diverges pays for a second copy of identical layers, and a build that never
# rotates its export directory saves back the layers it restored instead of
# the ones it produced. None of the three reds a job on its own.
#
# Two more pieces sit either side of the recipe. The builder is the action's,
# and is the job's default afterwards, so a lane's own later build inherits
# it; the removal cannot be the action's at all, because a composite action
# has no post hook, so each lane calls a second action and says which of its
# own builds that call has to cover.


def _module_constant(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise AssertionError(f"{path} has no module-level `{name}` assignment")


def _action_step_scripts(source: str) -> list[dict[str, Any]]:
    loaded = yaml.safe_load(source)
    assert isinstance(loaded, dict), "the action parsed to no mapping"
    return list(loaded["runs"]["steps"])


def _tag_the_build_action_builds(source: str) -> str:
    """The word after the sole ``--tag`` in the build action's step scripts."""
    steps = _action_step_scripts(source)
    build = _sole_step_whose_script(steps, "--tag ", "builds the image under a `--tag`")
    tags = re.findall(r"--tag\s+(\S+)", str(steps[build]["run"]))
    assert len(tags) == 1, f"expected one `--tag <word>` in the build step; found {tags}"
    return tags[0]


def _tag_the_remove_action_removes(source: str) -> str:
    """The one fixed word of the removal loop, ahead of the caller's extra tags."""
    script = "\n".join(str(step.get("run", "")) for step in _action_step_scripts(source))
    loops = re.findall(r"^\s*for tag in (.*) \$EXTRA_TAGS; do$", script, flags=re.MULTILINE)
    assert len(loops) == 1, f"expected one `for tag in <tag> $EXTRA_TAGS; do` line; found {loops}"
    words = loops[0].split()
    assert len(words) == 1, f"the loop removes exactly one fixed tag; found {words}"
    return words[0]


def _tag_the_developer_builder_defaults_to(source: str) -> str:
    """The literal of the script's sole top-level ``IMAGE="<tag>"`` line."""
    tags = re.findall(r'^IMAGE="([^"$]+)"$', source, flags=re.MULTILINE)
    assert len(tags) == 1, f'expected one top-level `IMAGE="<tag>"` line; found {tags}'
    return tags[0]


VA_IMAGE_ACTION = "./.github/actions/build-virtual-accelerator-image"
VA_IMAGE_ACTION_YML = (
    REPO_ROOT / ".github" / "actions" / "build-virtual-accelerator-image" / "action.yml"
)
VA_REMOVE_ACTION = "./.github/actions/remove-virtual-accelerator-images"
VA_REMOVE_ACTION_YML = (
    REPO_ROOT / ".github" / "actions" / "remove-virtual-accelerator-images" / "action.yml"
)
#: Matched as a prefix, so a version bump of the setup action does not have
#: to be made here as well.
BUILDX_ACTION = "docker/setup-buildx-action@"
VA_CACHE_DIR = "/tmp/va-buildx-cache"
VA_CACHE_EXPORT_DIR = "/tmp/va-buildx-cache-new"
VA_CACHE_KEY_PREFIX = "va-full-buildx-"
#: Every workspace member's manifest, by glob rather than by name: the image
#: installs each member under packages/, so a member's dependency change must
#: invalidate the layers as surely as the framework's does, including a member
#: added after this key was written.
VA_CACHE_MEMBER_GLOB = "'packages/*/pyproject.toml'"
#: The fixture constant the built tag has to agree with, and the tag itself,
#: read from source rather than restated, for the reason the xdist pairing is
#: read from source.
VA_E2E_IMAGE_OWNER = REPO_ROOT / "tests" / "va" / "e2e" / "_image.py"
VA_E2E_IMAGE_CONSTANT = "DEFAULT_IMAGE"
VA_IMAGE_TAG = _module_constant(VA_E2E_IMAGE_OWNER, VA_E2E_IMAGE_CONSTANT)
#: The developer build that ``IMAGE_REMEDY`` in the owner module names.
VA_BUILD_SCRIPT = REPO_ROOT / "scripts" / "va" / "build_and_boot_check.sh"
#: Every file outside Python that builds or removes the image, and how its tag
#: is read from its own source text.
VA_TAG_SITES: dict[str, tuple[Path, Callable[[str], str]]] = {
    "build-action": (VA_IMAGE_ACTION_YML, _tag_the_build_action_builds),
    "remove-action": (VA_REMOVE_ACTION_YML, _tag_the_remove_action_removes),
    "build-script": (VA_BUILD_SCRIPT, _tag_the_developer_builder_defaults_to),
}


def _load_action_yml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded is not None, f"{path} parsed to None"
    return loaded


def _load_va_image_action() -> dict[str, Any]:
    return _load_action_yml(VA_IMAGE_ACTION_YML)


def _load_va_remove_action() -> dict[str, Any]:
    return _load_action_yml(VA_REMOVE_ACTION_YML)


@pytest.fixture()
def va_image_action() -> dict[str, Any]:
    return _load_va_image_action()


@pytest.fixture()
def va_remove_action() -> dict[str, Any]:
    return _load_va_remove_action()


def _va_image_jobs(wf: dict[str, Any]) -> set[str]:
    """Jobs with business in this image, derived rather than listed.

    Derived for the reason the diagnostics set is: a hand-kept list is what
    the next lane forgets to join. A job cannot inherit this image from
    another job, so naming it, building it and removing it are one
    requirement. Calling either shared action counts as naming it — a lane
    that calls both mentions the tag nowhere in its own body, because
    ``yaml.safe_load`` has already dropped the comments that do.
    """
    return {
        name
        for name, job in _jobs(wf).items()
        if any(
            needle in json.dumps(job)
            for needle in (VA_IMAGE_TAG, VA_IMAGE_ACTION, VA_REMOVE_ACTION)
        )
    }


def _sole_step_whose_script(steps: list[dict[str, Any]], needle: str, what: str) -> int:
    found = [i for i, step in enumerate(steps) if needle in str(step.get("run", ""))]
    assert len(found) == 1, f"expected exactly one step that {what}; found {len(found)}"
    return found[0]


def _steps_using(wf: dict[str, Any], job_name: str, action: str) -> list[int]:
    steps = _jobs(wf)[job_name]["steps"]
    return [i for i, step in enumerate(steps) if step.get("uses") == action]


def _va_action_steps(wf: dict[str, Any], job_name: str) -> list[int]:
    return _steps_using(wf, job_name, VA_IMAGE_ACTION)


def test_every_virtual_accelerator_lane_builds_through_the_shared_action(
    workflow: dict[str, Any],
) -> None:
    """Every lane that needs this image gets it from the one shared recipe.

    At the baseline the four are ``target-switch-e2e``,
    ``target-switch-agentic-e2e``, ``va-live-e2e`` and
    ``va-bluesky-deploy-e2e``, and the set is derived rather than listed: a
    fifth lane joins it by naming the image, not by being added here.
    """
    wrong = {
        name: len(_va_action_steps(workflow, name)) for name in sorted(_va_image_jobs(workflow))
    }
    wrong = {name: count for name, count in wrong.items() if count != 1}
    assert wrong == {}, (
        f"these jobs name {VA_IMAGE_TAG} but do not build it through the shared action "
        f"exactly once: {wrong}. Add `uses: {VA_IMAGE_ACTION}`."
    )
    overridden = sorted(
        name
        for name in _va_image_jobs(workflow)
        for index in _va_action_steps(workflow, name)
        if "with" in _jobs(workflow)[name]["steps"][index]
    )
    assert overridden == [], (
        f"the action declares no inputs, so a `with:` at the call sites in {overridden} is a "
        f"lane asking for a different image than its siblings build."
    )


def test_every_virtual_accelerator_lane_builds_through_the_shared_action__mutation_drops_the_action_from_a_lane() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[VA_LIVE_JOB]["steps"]
    # The job still names the tag in its cleanup step, so it stays in the set.
    del steps[_va_action_steps(mutated, VA_LIVE_JOB)[0]]
    with pytest.raises(AssertionError):
        test_every_virtual_accelerator_lane_builds_through_the_shared_action(mutated)


def test_every_virtual_accelerator_lane_builds_through_the_shared_action__mutation_overrides_the_tag_at_a_call_site() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[VA_BLUESKY_JOB]["steps"]
    steps[_va_action_steps(mutated, VA_BLUESKY_JOB)[0]]["with"] = {
        "tag": "osprey-va-full:experiment"
    }
    with pytest.raises(AssertionError):
        test_every_virtual_accelerator_lane_builds_through_the_shared_action(mutated)


def test_the_shared_action_pins_the_tag_the_cache_and_the_rotation(
    va_image_action: dict[str, Any],
) -> None:
    """The recipe itself, pinned where the two quiet failures live.

    ``--cache-to`` and the rotation have no other witness. Without the export
    the cache never advances; without the rotation the job's cache save writes
    back the layers it restored instead of the ones it produced. Both stay
    green, and both cost a full build on every run that follows.
    """
    assert va_image_action["runs"]["using"] == "composite"
    steps = va_image_action["runs"]["steps"]

    unshelled = [
        step.get("name") for step in steps if "run" in step and step.get("shell") != "bash"
    ]
    assert unshelled == [], (
        f"every `run:` step of a composite action needs `shell: bash`; {unshelled} lack it, "
        f"which is a workflow-parse error at the moment a lane calls the action."
    )

    cache = [i for i, step in enumerate(steps) if "actions/cache@" in str(step.get("uses", ""))]
    assert len(cache) == 1, f"expected exactly one cache step; found {len(cache)}"
    cache_with = steps[cache[0]]["with"]
    assert cache_with["path"] == VA_CACHE_DIR
    assert cache_with["key"].startswith(VA_CACHE_KEY_PREFIX), (
        f"a lane-specific cache key prefix buys a second copy of identical layers against "
        f"the repository's cache quota; got {cache_with['key']}"
    )
    for hashed in (
        "docker/virtual-accelerator/Containerfile",
        "'pyproject.toml'",
        VA_CACHE_MEMBER_GLOB,
    ):
        assert hashed in cache_with["key"], f"the cache key no longer hashes {hashed}"

    build = _sole_step_whose_script(steps, f"--tag {VA_IMAGE_TAG}", f"builds {VA_IMAGE_TAG}")
    script = steps[build]["run"]
    assert f"--cache-from type=local,src={VA_CACHE_DIR}" in script
    assert f"--cache-to type=local,dest={VA_CACHE_EXPORT_DIR},mode=max" in script

    rotate = _sole_step_whose_script(
        steps,
        f"mv {VA_CACHE_EXPORT_DIR} {VA_CACHE_DIR}",
        "rotates the export directory in",
    )
    assert rotate > build, (
        "the rotation must follow the build: run first, it deletes the restored cache and "
        "then fails its own `mv`."
    )


def test_the_shared_action_pins_the_tag_the_cache_and_the_rotation__mutation_names_one_member() -> (
    None
):
    """A key naming one member's manifest serves stale layers the day another
    member's dependencies move."""
    mutated = copy.deepcopy(_load_va_image_action())
    for step in mutated["runs"]["steps"]:
        if "actions/cache@" in str(step.get("uses", "")):
            step["with"]["key"] = step["with"]["key"].replace(
                VA_CACHE_MEMBER_GLOB, "'packages/osprey-connectors/pyproject.toml'"
            )
    with pytest.raises(AssertionError, match="no longer hashes"):
        test_the_shared_action_pins_the_tag_the_cache_and_the_rotation(mutated)


def test_the_shared_action_pins_the_tag_the_cache_and_the_rotation__mutation_drops_the_rotation() -> (
    None
):
    mutated = copy.deepcopy(_load_va_image_action())
    steps = mutated["runs"]["steps"]
    steps.pop(
        _sole_step_whose_script(
            steps, f"mv {VA_CACHE_EXPORT_DIR} {VA_CACHE_DIR}", "rotates the cache"
        )
    )
    with pytest.raises(AssertionError):
        test_the_shared_action_pins_the_tag_the_cache_and_the_rotation(mutated)


def test_the_shared_action_pins_the_tag_the_cache_and_the_rotation__mutation_changes_the_built_tag() -> (
    None
):
    mutated = copy.deepcopy(_load_va_image_action())
    steps = mutated["runs"]["steps"]
    build = _sole_step_whose_script(steps, f"--tag {VA_IMAGE_TAG}", "builds the image")
    steps[build]["run"] = steps[build]["run"].replace(VA_IMAGE_TAG, "osprey-va-full:experiment")
    with pytest.raises(AssertionError):
        test_the_shared_action_pins_the_tag_the_cache_and_the_rotation(mutated)


def test_the_shared_action_pins_the_tag_the_cache_and_the_rotation__mutation_gives_the_cache_a_lane_prefix() -> (
    None
):
    mutated = copy.deepcopy(_load_va_image_action())
    for step in mutated["runs"]["steps"]:
        if "actions/cache@" in str(step.get("uses", "")):
            step["with"]["key"] = f"target-switch-{step['with']['key']}"
    with pytest.raises(AssertionError):
        test_the_shared_action_pins_the_tag_the_cache_and_the_rotation(mutated)


def test_the_shared_action_sets_up_the_builder_before_it_builds(
    va_image_action: dict[str, Any],
) -> None:
    """The builder is a precondition of the build, not a companion to it.

    ``--cache-to type=local`` needs a driver the default docker builder does
    not provide, and it errors rather than skipping the export — so the setup
    is a precondition of the build, and a setup placed after it is no setup at
    all.
    """
    steps = va_image_action["runs"]["steps"]
    setup = [i for i, step in enumerate(steps) if BUILDX_ACTION in str(step.get("uses", ""))]
    assert len(setup) == 1, (
        f"expected exactly one `{BUILDX_ACTION}` step in the action; found {len(setup)}"
    )
    build = _sole_step_whose_script(steps, f"--tag {VA_IMAGE_TAG}", f"builds {VA_IMAGE_TAG}")
    assert setup[0] < build, (
        f"the builder setup is a precondition of the build, not a companion to it: it sits at "
        f"index {setup[0]} and the build at index {build}, and `--cache-to type=local` against "
        f"the default docker builder errors rather than skipping the export."
    )


def test_the_shared_action_sets_up_the_builder_before_it_builds__mutation_drops_the_builder_setup() -> (
    None
):
    mutated = copy.deepcopy(_load_va_image_action())
    steps = mutated["runs"]["steps"]
    steps.pop(next(i for i, step in enumerate(steps) if BUILDX_ACTION in str(step.get("uses", ""))))
    with pytest.raises(AssertionError):
        test_the_shared_action_sets_up_the_builder_before_it_builds(mutated)


def test_the_shared_action_sets_up_the_builder_before_it_builds__mutation_sets_the_builder_up_after_the_build() -> (
    None
):
    mutated = copy.deepcopy(_load_va_image_action())
    steps = mutated["runs"]["steps"]
    setup = next(i for i, step in enumerate(steps) if BUILDX_ACTION in str(step.get("uses", "")))
    steps.append(steps.pop(setup))
    with pytest.raises(AssertionError):
        test_the_shared_action_sets_up_the_builder_before_it_builds(mutated)


def test_no_virtual_accelerator_lane_sets_up_its_own_builder(workflow: dict[str, Any]) -> None:
    """The builder these lanes build on is the action's, and only the action's.

    The set is derived, so this constrains only the lanes that use this image:
    ``qmd-sidecar-e2e`` builds a different image, never calls the action and
    keeps its own setup step.
    """
    own = {}
    for name in sorted(_va_image_jobs(workflow)):
        setups = [
            step.get("name")
            for step in _jobs(workflow)[name]["steps"]
            if BUILDX_ACTION in str(step.get("uses", ""))
        ]
        if setups:
            own[name] = setups
    assert own == {}, (
        f"the action sets up the builder these lanes build on, so a lane that sets up its own "
        f"creates a second BuildKit container and hands its later builds a builder the "
        f"action's cache was not exported through: {own}"
    )


def test_no_virtual_accelerator_lane_sets_up_its_own_builder__mutation_gives_a_lane_its_own_builder() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[VA_LIVE_JOB]["steps"].append(
        {"name": "Set up Docker Buildx", "uses": "docker/setup-buildx-action@v3"}
    )
    with pytest.raises(AssertionError):
        test_no_virtual_accelerator_lane_sets_up_its_own_builder(mutated)


def test_every_lane_builds_after_the_shared_builder(workflow: dict[str, Any]) -> None:
    """A lane's own later build runs on the builder the action set up.

    These lanes have no builder of their own, so the one their bench IOC build
    runs on is the action's; a build placed before the action call gets the
    default docker driver and its ``--cache-to type=local`` errors out.
    """
    early = {}
    for name in sorted(_va_image_jobs(workflow)):
        steps = _jobs(workflow)[name]["steps"]
        action = _va_action_steps(workflow, name)
        if not action:
            continue
        builds = [
            i for i, step in enumerate(steps) if "docker buildx build" in str(step.get("run", ""))
        ]
        too_early = [i for i in builds if i < action[0]]
        if too_early:
            early[name] = too_early
    assert early == {}, (
        f"these lanes build before the action that sets the builder up, so the build gets the "
        f"default docker driver and its `--cache-to type=local` errors out: {early}"
    )


def test_every_lane_builds_after_the_shared_builder__mutation_moves_a_bench_build_before_the_action() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TARGET_SWITCH_JOB]["steps"]
    build = next(
        i for i, step in enumerate(steps) if "docker buildx build" in str(step.get("run", ""))
    )
    steps.insert(0, steps.pop(build))
    with pytest.raises(AssertionError):
        test_every_lane_builds_after_the_shared_builder(mutated)


@pytest.mark.parametrize("site", sorted(VA_TAG_SITES))
def test_every_image_site_names_the_tag_the_live_fixtures_boot(
    site: str, source: str | None = None
) -> None:
    """A pairing, not a second copy of a string.

    ``tests/va/e2e/_image.py`` assigns ``DEFAULT_IMAGE`` as a literal, and
    every module that boots the image boots it unless ``OSPREY_VA_E2E_IMAGE``
    names another, which nothing in CI sets. The build action, the removal
    action and the developer build script each name that tag in their own
    language, so a tag that drifts on any side leaves lanes building an image
    nothing starts, removing one nobody built, or a developer reproducing a
    lane against the wrong image. Each is read from source, as this module's
    other pins read theirs, and a site that stops naming a tag fails too.
    """
    path, read = VA_TAG_SITES[site]
    text = path.read_text(encoding="utf-8") if source is None else source
    image = _module_constant(VA_E2E_IMAGE_OWNER, VA_E2E_IMAGE_CONSTANT)
    tag = read(text)
    assert tag == image, (
        f"{path.relative_to(REPO_ROOT)} names {tag!r}, but the live fixtures boot "
        f"`{VA_E2E_IMAGE_CONSTANT}` = {image!r} from tests/va/e2e/_image.py."
    )


@pytest.mark.parametrize("site", sorted(VA_TAG_SITES))
def test_every_image_site_names_the_tag_the_live_fixtures_boot__mutation_names_a_tag_no_fixture_boots(
    site: str,
) -> None:
    path, _ = VA_TAG_SITES[site]
    mutated = path.read_text(encoding="utf-8").replace(VA_IMAGE_TAG, "osprey-va-full:experiment")
    with pytest.raises(AssertionError):
        test_every_image_site_names_the_tag_the_live_fixtures_boot(site, mutated)


@pytest.mark.parametrize("site", sorted(VA_TAG_SITES))
def test_every_image_site_names_the_tag_the_live_fixtures_boot__mutation_names_no_tag(
    site: str,
) -> None:
    path, _ = VA_TAG_SITES[site]
    mutated = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if VA_IMAGE_TAG not in line
    )
    with pytest.raises(AssertionError):
        test_every_image_site_names_the_tag_the_live_fixtures_boot(site, mutated)


@pytest.fixture()
def va_build_script_text() -> str:
    return VA_BUILD_SCRIPT.read_text(encoding="utf-8")


def test_the_developer_builder_takes_the_tag_as_an_argument(va_build_script_text: str) -> None:
    """The script builds and boots whatever tag it is given, and its own default otherwise.

    A developer reproducing a lane that boots another tag has to be able to
    build that tag with the remedy the live suites print, so ``IMAGE`` is
    assigned twice -- the default and the ``--tag`` branch -- and both the build
    and the run read it rather than a literal.
    """
    assignments = [
        line.strip() for line in va_build_script_text.splitlines() if re.match(r"\s*IMAGE=", line)
    ]
    assert len(assignments) == 2, (
        f"`IMAGE` is assigned in exactly two places, the default and the `--tag` branch; "
        f"found {assignments}"
    )
    assert any(re.fullmatch(r'IMAGE="\$\{?2\}?"', line) for line in assignments), (
        f"one assignment takes the word after `--tag`; found {assignments}"
    )
    lines = {line.strip() for line in va_build_script_text.splitlines()}
    assert any(line.startswith('-t "${IMAGE}"') for line in lines), (
        'the build tags the image `-t "${IMAGE}"`'
    )
    assert '"${IMAGE}" >/dev/null' in lines, 'the run starts `"${IMAGE}"`'


def test_the_developer_builder_takes_the_tag_as_an_argument__mutation_drops_the_tag_branch(
    va_build_script_text: str,
) -> None:
    kept = "\n".join(
        line
        for line in va_build_script_text.splitlines()
        if not re.match(r'\s*IMAGE="\$\{?2\}?"', line)
    )
    with pytest.raises(AssertionError):
        test_the_developer_builder_takes_the_tag_as_an_argument(kept)


@pytest.mark.parametrize(
    ("argv", "stop"),
    [(["--tag"], "--tag needs a value"), (["--no-such-option"], "unknown option")],
    ids=["tag-without-value", "unknown-option"],
)
def test_the_developer_builder_refuses_a_bad_argument_before_any_runtime(
    argv: list[str], stop: str
) -> None:
    refused = subprocess.run(
        ["bash", str(VA_BUILD_SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert refused.returncode == 1, refused.stdout + refused.stderr
    fatal = [line for line in refused.stderr.splitlines() if line.startswith("FATAL")]
    assert len(fatal) == 1 and stop in fatal[0], refused.stderr


@pytest.fixture()
def wiring_module_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def test_this_module_reads_the_image_from_its_owner(wiring_module_source: str) -> None:
    """The tag is read from ``tests/va/e2e/_image.py``, never restated here.

    A string constant here equal to the owner's value is a second copy that a
    rename on the owner's side leaves behind, and every pin in this block would
    then check the workflow against the stale copy. Constants only, so a
    comment naming the tag and the deliberately different replacement tags of
    the mutation twins do not count.
    """
    image = _module_constant(VA_E2E_IMAGE_OWNER, VA_E2E_IMAGE_CONSTANT)
    spelled = sorted(
        node.lineno
        for node in ast.walk(ast.parse(wiring_module_source))
        if isinstance(node, ast.Constant) and node.value == image
    )
    assert spelled == [], (
        f"lines {spelled} spell {image!r}; read it with "
        f"`_module_constant(VA_E2E_IMAGE_OWNER, VA_E2E_IMAGE_CONSTANT)`."
    )


def test_this_module_reads_the_image_from_its_owner__mutation_restates_the_tag() -> None:
    image = _module_constant(VA_E2E_IMAGE_OWNER, VA_E2E_IMAGE_CONSTANT)
    planted = f"VA_IMAGE_TAG = {image!r}\n"
    with pytest.raises(AssertionError):
        test_this_module_reads_the_image_from_its_owner(planted)


def test_every_virtual_accelerator_lane_removes_the_image_it_built(
    workflow: dict[str, Any],
) -> None:
    """The one piece that cannot move into the build action, pinned at each call site.

    A composite action has no ``post:`` hook, so the removal cannot be a step
    the build action contributes — the only always-run step an action can
    contribute is one that runs where the action runs, before the lane's tests
    rather than after them. It is a second action, and the ``if:`` belongs at
    the call site because a composite action cannot carry a condition on its
    caller's behalf.
    """
    missing = []
    for name in sorted(_va_image_jobs(workflow)):
        steps = _jobs(workflow)[name]["steps"]
        build = _va_action_steps(workflow, name)
        removals = [
            i
            for i in _steps_using(workflow, name, VA_REMOVE_ACTION)
            if steps[i].get("if") == "always()"
        ]
        if not removals or not build or max(removals) < build[0]:
            missing.append(name)
    assert missing == [], (
        f"these jobs build {VA_IMAGE_TAG} but never remove it on every outcome: {missing}. "
        f"Add `uses: {VA_REMOVE_ACTION}` with `if: always()` after the build."
    )


def test_every_virtual_accelerator_lane_removes_the_image_it_built__mutation_makes_a_removal_conditional() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TARGET_SWITCH_JOB]["steps"]
    for index in _steps_using(mutated, TARGET_SWITCH_JOB, VA_REMOVE_ACTION):
        # The removal survives, but stops running on a red lane.
        del steps[index]["if"]
    with pytest.raises(AssertionError):
        test_every_virtual_accelerator_lane_removes_the_image_it_built(mutated)


def test_each_lane_removes_every_image_it_built(workflow: dict[str, Any]) -> None:
    """``extra-tags`` is derived from the lane's own ``--tag`` flags.

    It is not a lane preference: it is the list of images this lane built that
    the shared removal cannot know about. So a lane that gains a second build
    gains its removal, and a lane that loses one stops asking for a removal
    with no subject. At the baseline the two switch lanes build
    ``bench-ioc:latest`` and pass it; the other two build nothing else and
    pass nothing.
    """
    unpaired = {}
    for name in sorted(_va_image_jobs(workflow)):
        steps = _jobs(workflow)[name]["steps"]
        built = {VA_IMAGE_TAG}
        for step in steps:
            built.update(re.findall(r"--tag\s+(\S+)", str(step.get("run", ""))))
        removed = {VA_IMAGE_TAG}
        for index in _steps_using(workflow, name, VA_REMOVE_ACTION):
            removed.update(str((steps[index].get("with") or {}).get("extra-tags", "")).split())
        if built != removed:
            unpaired[name] = {"built": sorted(built), "removed": sorted(removed)}
    assert unpaired == {}, (
        f"every image a lane builds is an image that lane removes, and `extra-tags` carries "
        f"the ones the shared removal cannot know about: {unpaired}"
    )


def test_each_lane_removes_every_image_it_built__mutation_drops_the_extra_tag() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[TARGET_SWITCH_JOB]["steps"]
    for index in _steps_using(mutated, TARGET_SWITCH_JOB, VA_REMOVE_ACTION):
        del steps[index]["with"]
    with pytest.raises(AssertionError):
        test_each_lane_removes_every_image_it_built(mutated)


def test_each_lane_removes_every_image_it_built__mutation_asks_for_a_tag_the_lane_never_built() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[VA_LIVE_JOB]["steps"]
    for index in _steps_using(mutated, VA_LIVE_JOB, VA_REMOVE_ACTION):
        steps[index]["with"] = {"extra-tags": "bench-ioc:latest"}
    with pytest.raises(AssertionError):
        test_each_lane_removes_every_image_it_built(mutated)


def test_the_removal_action_removes_exact_tags_only(va_remove_action: dict[str, Any]) -> None:
    """The guardrail the script's own comment claims, made load-bearing.

    Exact tags only, never a prune or a wildcard. The action removes images on
    behalf of four lanes at once, so a sweep introduced here would reach
    whatever else the runner holds — on shared self-hosted runners too.
    """
    assert va_remove_action["runs"]["using"] == "composite"
    steps = va_remove_action["runs"]["steps"]

    unshelled = [
        step.get("name") for step in steps if "run" in step and step.get("shell") != "bash"
    ]
    assert unshelled == [], (
        f"every `run:` step of a composite action needs `shell: bash`; {unshelled} lack it, "
        f"which is a workflow-parse error at the moment a lane calls the action."
    )

    inputs = va_remove_action.get("inputs") or {}
    assert set(inputs) == {"extra-tags"}, (
        f"the only thing a lane knows that the action does not is which further tags it built; "
        f"got {sorted(inputs)}"
    )
    assert not inputs["extra-tags"].get("required"), (
        "a lane that built only the shared image calls this action with no `with:` at all, so "
        "the input cannot be required."
    )
    assert inputs["extra-tags"].get("default") == "", (
        f"the default has to be the empty string: an empty unquoted expansion contributes zero "
        f"words and the loop runs once. Got {inputs['extra-tags'].get('default')!r}."
    )

    script = "\n".join(str(step.get("run", "")) for step in steps)
    assert VA_IMAGE_TAG in script, (
        f"every caller builds {VA_IMAGE_TAG}, so the action removes it without being asked."
    )
    for sweep in ("prune", "-a ", "--all", "*"):
        assert sweep not in script, (
            f"exact tags only: `{sweep}` in this script reaches whatever else the runner holds, "
            f"which is the container-ops guardrail every lane here observes."
        )


def test_the_removal_action_removes_exact_tags_only__mutation_sweeps_by_pattern() -> None:
    mutated = copy.deepcopy(_load_va_remove_action())
    mutated["runs"]["steps"][0]["run"] = "docker image prune -af"
    with pytest.raises(AssertionError):
        test_the_removal_action_removes_exact_tags_only(mutated)


def test_the_removal_action_removes_exact_tags_only__mutation_requires_the_extra_tags() -> None:
    mutated = copy.deepcopy(_load_va_remove_action())
    mutated["inputs"]["extra-tags"]["required"] = True
    del mutated["inputs"]["extra-tags"]["default"]
    with pytest.raises(AssertionError):
        test_the_removal_action_removes_exact_tags_only(mutated)


def test_no_lane_removes_the_virtual_accelerator_image_inline(workflow: dict[str, Any]) -> None:
    """The removal's counterpart to the inline-build guard.

    A fifth lane that pastes the removal rather than calling the action
    reintroduces the divergence the action ends, and nothing else would red.
    Scoped to this one tag on purpose: other lanes in this workflow
    legitimately run their own ``docker rmi``, and two of them even sweep by
    pattern, which is exactly why the exact-tags assertion above is scoped to
    the action rather than to the workflow.
    """
    inline = sorted(
        name
        for name, job in _jobs(workflow).items()
        for step in job.get("steps") or []
        if f"rmi -f {VA_IMAGE_TAG}" in str(step.get("run", ""))
    )
    assert inline == [], (
        f"these jobs remove {VA_IMAGE_TAG} inline: {inline}. Use "
        f"`uses: {VA_REMOVE_ACTION}` instead, so the tags removed stay paired with the tags "
        f"built."
    )


def test_no_lane_removes_the_virtual_accelerator_image_inline__mutation_pastes_the_removal_back() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[VA_LIVE_JOB]["steps"].append(
        {
            "name": "Clean up the target images",
            "if": "always()",
            "run": f"docker rmi -f {VA_IMAGE_TAG} || true",
        }
    )
    with pytest.raises(AssertionError):
        test_no_lane_removes_the_virtual_accelerator_image_inline(mutated)


def test_no_lane_builds_the_virtual_accelerator_image_inline(workflow: dict[str, Any]) -> None:
    """The guard against arriving back here by the route that got us here.

    A fifth lane that pastes the build block rather than calling the action
    reintroduces every divergence the action exists to prevent, and nothing
    else would red.
    """
    inline = sorted(
        name
        for name, job in _jobs(workflow).items()
        for step in job.get("steps") or []
        if f"--tag {VA_IMAGE_TAG}" in str(step.get("run", ""))
    )
    assert inline == [], (
        f"these jobs build {VA_IMAGE_TAG} inline: {inline}. Use `uses: {VA_IMAGE_ACTION}` "
        f"instead, so the tag, the cache key and the rotation stay one recipe."
    )


def test_no_lane_builds_the_virtual_accelerator_image_inline__mutation_pastes_the_build_back() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[VA_LIVE_JOB]["steps"].append(
        {
            "name": "Build the virtual accelerator image (staged context)",
            "run": f"docker buildx build --tag {VA_IMAGE_TAG} .",
        }
    )
    with pytest.raises(AssertionError):
        test_no_lane_builds_the_virtual_accelerator_image_inline(mutated)


# ---------------------------------------------------------------------------
# deploy-e2e skips Dependabot; lint proves the lockfile before sync can heal it
# ---------------------------------------------------------------------------

DEPLOY_JOB = "deploy-e2e"
LINT_JOB = "lint"
LOCK_CHECK_STEP = "Lockfile is current with pyproject"
INSTALL_STEP = "Install dependencies"
_DEPENDABOT_GUARD = "github.actor != 'dependabot[bot]'"


def test_deploy_e2e_skips_dependabot(workflow: dict[str, Any]) -> None:
    """Dependabot branches live in this repo, so the same-repo fork check does
    not exclude them — but Dependabot runs resolve secrets against the (empty)
    Dependabot store, and `osprey up`'s .env preflight then fails on the
    missing ALS_APG_API_KEY. The lane must carry the same actor guard and
    revalidation arm as every other secret-gated lane."""
    condition = _jobs(workflow)[DEPLOY_JOB]["if"]
    assert _DEPENDABOT_GUARD in condition
    assert "inputs.revalidate_secret_lanes" in condition


def test_deploy_e2e_skips_dependabot__mutation_reverts_to_fork_only_guard() -> None:
    """The exact guard this lane shipped with for months: same-repo check
    only, which Dependabot passes — and then reds every one of its PRs."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[DEPLOY_JOB]["if"] = (
        "github.event_name == 'pull_request'"
        " && github.event.pull_request.head.repo.full_name == github.repository"
    )
    with pytest.raises(AssertionError):
        test_deploy_e2e_skips_dependabot(mutated)


def test_gate_summary_names_the_deploy_lane(workflow: dict[str, Any]) -> None:
    """The Dependabot run summary enumerates the lanes the gate cannot vouch
    for; a skipped lane missing from that list is a silent coverage gap."""
    step = _find_named_step(workflow, GATE_JOB, "Check all jobs status")
    assert "Deploy E2E Test (build + deploy pipeline)" in step["run"]


def test_gate_summary_names_the_deploy_lane__mutation_drops_the_line() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [
        line
        for line in step["run"].splitlines(keepends=True)
        if "Deploy E2E Test (build + deploy pipeline)" not in line
    ]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    with pytest.raises(AssertionError):
        test_gate_summary_names_the_deploy_lane(mutated)


def _step_names(wf: dict[str, Any], job_name: str) -> list[str]:
    return [step.get("name", "") for step in _jobs(wf)[job_name]["steps"]]


def test_lint_checks_the_lockfile_before_installing(workflow: dict[str, Any]) -> None:
    """`uv lock --check` is the only place a pyproject/uv.lock split can
    surface: every other lane runs a plain `uv sync`, which re-locks a stale
    lockfile in the runner and goes green. Ordering is the pin's sharp edge —
    `uv sync` rewrites the lock on disk, so a check placed after the install
    step always passes."""
    step = _find_named_step(workflow, LINT_JOB, LOCK_CHECK_STEP)
    assert "uv lock --check" in step["run"]
    names = _step_names(workflow, LINT_JOB)
    assert names.index(LOCK_CHECK_STEP) < names.index(INSTALL_STEP)


def test_lint_checks_the_lockfile__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[LINT_JOB]["steps"]
    steps[:] = [s for s in steps if s.get("name") != LOCK_CHECK_STEP]
    with pytest.raises(AssertionError):
        test_lint_checks_the_lockfile_before_installing(mutated)


def test_lint_checks_the_lockfile__mutation_moves_it_after_sync() -> None:
    """The subtle regression: the step survives a refactor but lands after
    `uv sync`, which has just healed the very staleness it was checking for."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[LINT_JOB]["steps"]
    (check,) = [s for s in steps if s.get("name") == LOCK_CHECK_STEP]
    steps.remove(check)
    steps.insert(next(i for i, s in enumerate(steps) if s.get("name") == INSTALL_STEP) + 1, check)
    with pytest.raises(AssertionError):
        test_lint_checks_the_lockfile_before_installing(mutated)


# ---------------------------------------------------------------------------
# the changelog-fragment gate runs in lint, and the quick tier runs the same
# ---------------------------------------------------------------------------
#
# One rule, two callers: `scripts/changelog_fragments.py check` decides whether
# a PR that touches a gated directory carries a fragment in changelog.d/ and
# whether anyone hand-edited CHANGELOG.md's [Unreleased] block. The local check
# has to run the *same command* as the lane, or the pre-PR sweep goes green on
# work CI then rejects.

CHANGELOG_STEP = "Run changelog-fragment guard"
CHANGELOG_SCRIPT = "scripts/changelog_fragments.py"
#: How a local run names its base; the quick tier is the pre-commit check.
LOCAL_BASE = "$BASE"
#: What the CRITICAL block used to be: proof that CHANGELOG.md was *touched*,
#: which the fragment workflow makes both unsatisfiable (fragments are the
#: entry) and meaningless (touching the file proves nothing about content).
_CHANGELOG_TOUCH_GREP = "grep -q CHANGELOG.md"


@pytest.fixture()
def quick_source() -> str:
    return _tier_source("quick", LOCAL_BASE)


def test_lint_runs_the_changelog_fragment_guard(workflow: dict[str, Any]) -> None:
    """Four things at once, because each has its own way of silently going
    missing: the step exists and calls `check`; the base ref reaches the shell
    through ``env:`` and never through ``${{ }}`` inside ``run:`` (a branch
    name is attacker-controlled text — interpolated into the run line it is
    executed); the env value comes from the pull-request event rather than a
    hardcoded ``main``; and ``!cancelled()`` keeps a ruff failure above from
    hiding a missing fragment and buying a second full CI cycle."""
    step = _find_named_step(workflow, LINT_JOB, CHANGELOG_STEP)
    assert CHANGELOG_SCRIPT + " check --base" in step["run"]
    assert "${{" not in step["run"], (
        f"{CHANGELOG_STEP} interpolates an expression into `run:`: {step['run']!r}"
    )
    assert "github.event.pull_request.base.ref" in step["env"]["BASE_REF"]
    assert "!cancelled()" in step["if"]
    assert "pull_request" in step["if"]


def test_lint_runs_the_changelog_fragment_guard__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[LINT_JOB]["steps"]
    steps[:] = [s for s in steps if s.get("name") != CHANGELOG_STEP]
    with pytest.raises(AssertionError):
        test_lint_runs_the_changelog_fragment_guard(mutated)


def test_lint_runs_the_changelog_fragment_guard__mutation_interpolates_the_base_ref_into_run() -> (
    None
):
    """The regression the ``env:`` indirection exists to prevent: the same
    command, written the obvious way. A PR from a branch named with shell
    metacharacters then runs them on the lint runner."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, LINT_JOB, CHANGELOG_STEP)
    step["run"] = step["run"].replace(
        '"origin/$BASE_REF"', '"origin/${{ github.event.pull_request.base.ref }}"'
    )
    assert "${{" in step["run"], "the run line no longer uses $BASE_REF; this mutation is stale"
    with pytest.raises(AssertionError):
        test_lint_runs_the_changelog_fragment_guard(mutated)


def test_the_quick_tier_runs_the_same_gate(quick_source: str) -> None:
    """The local sweep must run the gate itself, not a proxy for it. Its old
    CRITICAL check grepped the diff for CHANGELOG.md, which under the fragment
    workflow is red on every ordinary PR and green on a PR that hand-edited
    [Unreleased] — wrong in both directions."""
    assert CHANGELOG_SCRIPT in quick_source
    assert "check --base" in quick_source
    assert _CHANGELOG_TOUCH_GREP not in quick_source


def test_the_quick_tier_runs_the_same_gate__mutation_restores_the_grep() -> None:
    source = _tier_source("quick", LOCAL_BASE)
    mutated = source.replace(
        f'python3 {CHANGELOG_SCRIPT} check --base "$BASE"',
        f"git diff $BASE...HEAD --name-only | {_CHANGELOG_TOUCH_GREP}",
    )
    assert mutated != source, "the gate invocation moved; this mutation is stale"
    with pytest.raises(AssertionError):
        test_the_quick_tier_runs_the_same_gate(mutated)


# ---------------------------------------------------------------------------
# the plugin version gate runs in lint as one call to the script
# ---------------------------------------------------------------------------
#
# One rule, one implementation: the lint step calls the script and states no
# part of the rule itself. `scripts/plugin_version.py check` decides whether a
# PR that changes the plugin tree carries a version its base does not; a copy
# of that rule written inline beside the call is a second answer to the same
# question, free to drift from the first.

PLUGIN_VERSION_STEP = "Run plugin version-bump gate"
PLUGIN_VERSION_SCRIPT = "scripts/plugin_version.py"


def test_lint_runs_the_plugin_version_check(workflow: dict[str, Any]) -> None:
    """The step calls ``check`` against the pull request's own base, the base
    reaches the shell through ``env:`` and never through ``${{ }}`` inside
    ``run:``, ``!cancelled()`` keeps an earlier failure from hiding a missing
    bump, and the run line restates no part of the rule."""
    step = _find_named_step(workflow, LINT_JOB, PLUGIN_VERSION_STEP)
    run = step["run"]
    assert PLUGIN_VERSION_SCRIPT + ' check --base "origin/$BASE_REF"' in run
    assert "${{" not in run, (
        f"{PLUGIN_VERSION_STEP} interpolates an expression into `run:`: {run!r}"
    )
    assert "github.event.pull_request.base.ref" in step["env"]["BASE_REF"]
    assert "!cancelled()" in step["if"]
    assert "pull_request" in step["if"]
    assert "git diff" not in run, f"{PLUGIN_VERSION_STEP} restates the rule inline: {run!r}"
    assert "json.load" not in run, f"{PLUGIN_VERSION_STEP} restates the rule inline: {run!r}"


def test_lint_runs_the_plugin_version_check__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[LINT_JOB]["steps"]
    steps[:] = [s for s in steps if s.get("name") != PLUGIN_VERSION_STEP]
    with pytest.raises(AssertionError):
        test_lint_runs_the_plugin_version_check(mutated)


def test_lint_runs_the_plugin_version_check__mutation_interpolates_the_base_ref_into_run() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, LINT_JOB, PLUGIN_VERSION_STEP)
    step["run"] = step["run"].replace(
        '"origin/$BASE_REF"', '"origin/${{ github.event.pull_request.base.ref }}"'
    )
    assert "${{" in step["run"], "the run line no longer uses $BASE_REF; this mutation is stale"
    with pytest.raises(AssertionError):
        test_lint_runs_the_plugin_version_check(mutated)


def test_lint_runs_the_plugin_version_check__mutation_restates_the_rule_inline() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, LINT_JOB, PLUGIN_VERSION_STEP)
    original = step["run"]
    step["run"] += '\ngit diff --name-only "origin/$BASE_REF...HEAD" -- plugins'
    assert step["run"] != original
    with pytest.raises(AssertionError):
        test_lint_runs_the_plugin_version_check(mutated)


def test_lint_runs_the_plugin_version_check__mutation_hardcodes_main() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, LINT_JOB, PLUGIN_VERSION_STEP)
    step["run"] = step["run"].replace('"origin/$BASE_REF"', "origin/main")
    assert "origin/main" in step["run"], "the run line moved; this mutation is stale"
    with pytest.raises(AssertionError):
        test_lint_runs_the_plugin_version_check(mutated)


def test_the_quick_tier_runs_the_plugin_version_check(quick_source: str) -> None:
    """The local check runs the same check as the lane, against its own base,
    or it goes green on a branch the lint job then refuses."""
    assert PLUGIN_VERSION_SCRIPT + ' check --base "$BASE"' in quick_source


def test_the_quick_tier_runs_the_plugin_version_check__mutation_drops_the_line() -> None:
    source = _tier_source("quick", LOCAL_BASE)
    mutated = "\n".join(
        line for line in source.splitlines() if PLUGIN_VERSION_SCRIPT + " check" not in line
    )
    assert mutated != source, "the check invocation moved; this mutation is stale"
    with pytest.raises(AssertionError):
        test_the_quick_tier_runs_the_plugin_version_check(mutated)


# ---------------------------------------------------------------------------
# the merge gate: each CI step that needs no secret or Docker calls one check
# ---------------------------------------------------------------------------
#
# scripts/check.py runs the merge gate from what the checks modules declare,
# and each check CI runs without a secret or a container is one step that calls
# it by id. Which step calls which check is read from ci.yml itself (see "One
# check set" below); the full tier runs each call's command too, so a
# contributor's `check.py full` is the command CI runs.

FRONTEND_JS_JOB = "frontend-js"
DOCS_JOB = "docs"
#: How a caller step names the comparison base of the two diff checks: the
#: pull request's base ref, read from ``env:``.
CI_BASE = "origin/$BASE_REF"


def _first_index(commands: list[str], predicate: Callable[[str], bool]) -> int:
    return next((i for i, command in enumerate(commands) if predicate(command)), len(commands))


def test_the_gate_checks_the_lockfile_before_any_uv_run(gate_source: str | None = None) -> None:
    """Any ``uv run`` re-locks a stale lockfile, so the check that the lockfile
    is current comes before the first of them, as it comes before the install
    step in CI."""
    source = _tier_source("full", CI_BASE) if gate_source is None else gate_source
    commands = source.splitlines()
    lock = _first_index(commands, lambda command: command == "uv lock --check")
    first_run = _first_index(commands, lambda command: "uv run " in command)
    assert lock < len(commands), "the full tier never checks the lockfile"
    assert lock < first_run, "the full tier runs `uv run` before `uv lock --check`"


def test_the_gate_checks_the_lockfile_before_any_uv_run__mutation_moves_it_after_ruff() -> None:
    lines = _tier_source("full", CI_BASE).splitlines()
    lines.remove("uv lock --check")
    ruff = max(i for i, line in enumerate(lines) if line.startswith("uv run ruff "))
    lines.insert(ruff + 1, "uv lock --check")
    with pytest.raises(AssertionError):
        test_the_gate_checks_the_lockfile_before_any_uv_run("\n".join(lines))


#: A vitest ``include`` glob's test-file extension set: ``*.test.{js,mjs}``.
_VITEST_INCLUDE_EXT_RE = re.compile(r"\.test\.(?:\{([^}]*)\}|(\w+))['\"]")
VITEST_CONFIG = CI_YML.parents[2] / "vitest.config.js"
#: Extensions a named test file could plausibly carry.
_CANDIDATE_EXTENSIONS = ("js", "mjs", "cjs", "ts", "jsx")

Router = Callable[[list[str]], dict[str, list[str]]]


def _vitest_include_extensions() -> set[str]:
    (include,) = re.findall(r"include:\s*\[([^\]]*)\]", VITEST_CONFIG.read_text(encoding="utf-8"))
    extensions: set[str] = set()
    for braced, single in _VITEST_INCLUDE_EXT_RE.findall(include):
        extensions.update(ext.strip() for ext in (braced or single).split(","))
    return extensions


def _routed_js_extensions(route: Router) -> set[str]:
    """The extensions the check command sends a named test file to vitest by."""
    routed = set()
    for ext in _CANDIDATE_EXTENSIONS:
        try:
            if route([f"tests/x.test.{ext}"])["vitest"]:
                routed.add(ext)
        except check_set.Refused:
            continue
    return routed


def test_the_gate_sends_vitest_exactly_the_files_vitest_includes(
    route: Router | None = None,
) -> None:
    """A JS test is sent to vitest by the extensions vitest's own ``include``
    lists: an extension the split lacks is a named test refused, one vitest
    lacks is a named test vitest never finds."""
    assert _vitest_include_extensions(), "vitest.config.js lists no test extension"
    routed = _routed_js_extensions(route or check_set._route_named)
    assert routed == _vitest_include_extensions()


def test_the_gate_sends_vitest_exactly_the_files_vitest_includes__mutation_drops_mjs() -> None:
    def narrowed(named: list[str]) -> dict[str, list[str]]:
        if any(path.endswith(".mjs") for path in named):
            raise check_set.Refused("no runner")
        return check_set._route_named(named)

    with pytest.raises(AssertionError):
        test_the_gate_sends_vitest_exactly_the_files_vitest_includes(narrowed)


def test_the_gate_hands_vitest_only_the_js_tests(route: Router | None = None) -> None:
    """Named tests reach vitest as the JS list, never every argument: a Python
    path handed to vitest is a filter that matches no file."""
    routed = (route or check_set._route_named)(["tests/a/test_b.py", "tests/c.test.js"])
    assert routed["vitest"] == ["tests/c.test.js"]
    assert routed["pytest"] == ["tests/a/test_b.py"]


def test_the_gate_hands_vitest_only_the_js_tests__mutation_hands_every_argument() -> None:
    def everything(named: list[str]) -> dict[str, list[str]]:
        return {"pytest": list(named), "vitest": list(named)}

    with pytest.raises(AssertionError):
        test_the_gate_hands_vitest_only_the_js_tests(everything)


def _run_check_with_tool_tripwires(
    tmp_path: Path, *args: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the check command with ``uv``, ``npm``, ``npx`` and ``node`` replaced
    by stand-ins that leave a mark, so a stop can be proven to come before any
    of them."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mark = tmp_path / "a tool ran"
    for tool in ("uv", "npm", "npx", "node"):
        stand_in = bin_dir / tool
        stand_in.write_text(f'#!/bin/sh\necho "$0" >> "{mark}"\nexit 99\n', encoding="utf-8")
        stand_in.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / CHECK_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return result, mark


def test_the_gate_refuses_a_test_no_runner_takes(tmp_path: Path) -> None:
    result, mark = _run_check_with_tool_tripwires(tmp_path, "full", "--", "tests/x.txt")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "check: refused: tests/x.txt: not a test file" in result.stderr
    assert not mark.exists(), f"a tool ran before the stop: {mark.read_text()}"


def test_the_gate_refuses_a_base_that_does_not_resolve(tmp_path: Path) -> None:
    result, mark = _run_check_with_tool_tripwires(tmp_path, "full", "--base", "no/such/ref")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "check: refused: no/such/ref does not resolve to a commit" in result.stderr
    assert not mark.exists(), f"a tool ran before the stop: {mark.read_text()}"


#: The written-down copies of a check a contributor or a skill runs by hand,
#: each held to the check's declared command.
DOCUMENTED_GATE_STEPS = (("plugins/osprey/skills/doc-sync/SKILL.md", "docs"),)


def _documented_commands(source: str) -> list[list[str]]:
    """The doc's shell lines, each split at ``&&`` into commands, as shell words."""
    commands = []
    for line in source.splitlines():
        for part in line.split("&&"):
            try:
                commands.append(shlex.split(part))
            except ValueError:
                continue
    return commands


@pytest.mark.parametrize(
    ("path", "check"), DOCUMENTED_GATE_STEPS, ids=[p for p, _ in DOCUMENTED_GATE_STEPS]
)
def test_documented_gate_steps_are_the_ci_line(
    path: str, check: str, doc_source: str | None = None
) -> None:
    """A documented copy of a check is the command a reader runs, so it is the
    declared command, not a variant of it."""
    declared = list(next(c for c in check_set.CHECKS if c.id == check).argv)
    source = (REPO_ROOT / path).read_text(encoding="utf-8") if doc_source is None else doc_source
    assert declared in _documented_commands(source), f"{path} no longer documents {declared!r}"


@pytest.mark.parametrize(
    ("path", "check"), DOCUMENTED_GATE_STEPS, ids=[p for p, _ in DOCUMENTED_GATE_STEPS]
)
def test_documented_gate_steps_are_the_ci_line__mutation_drops_a_flag(
    path: str, check: str
) -> None:
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    mutated = source.replace(' SPHINXOPTS="-W --keep-going"', "", 1)
    assert mutated != source, f"{path} documents no docs build; mutation is stale"
    with pytest.raises(AssertionError):
        test_documented_gate_steps_are_the_ci_line(path, check, mutated)


# ---------------------------------------------------------------------------
# (j) every dockerbuild-marked module reaches a lane that gates the merge
# ---------------------------------------------------------------------------
#
# Section (e) proves a marked module LEFT the shared e2e lane. Nothing proved it
# ARRIVED anywhere: the per-lane "job exists"/"needs"/"check_pr_lane" assertions
# above are hand-written one lane at a time, so a lane nobody remembered to pin
# could be deleted whole and this module stayed green. That is not hypothetical
# — it was true of both halves of the auth/audit wiring (the full-chain lane and
# the audit two-container step) until this section existed.
#
# The general guard below closes it for every marked module, present and future,
# in the same shape as (e): scan the marker, then require the file to be named by
# some job's run step, that job to appear in the gate's `needs:`, and the gate
# script to actually examine its result. The `needs:` half is self-enforcing (a
# dangling needs entry is a workflow error); the other two are not. The per-lane
# assertions that follow are redundant with it on purpose — a partial removal
# then names the lane it broke instead of only reporting a set.

FULL_CHAIN_JOB = "full-chain-auth-e2e"
FULL_CHAIN_TEST_FILE = "tests/e2e/test_full_chain_auth.py"
FULL_CHAIN_RUN_STEP = "Run full-chain auth E2E"
FULL_CHAIN_SKIP_GATE_STEP = "Fail the lane on any skipped test"
#: The two lanes full-chain-auth strictly subsumes: each builds ONE real image
#: (the deployed sidecar Dockerfile / the production project image), this one
#: builds both in a single job. Its budget must therefore exceed both of theirs,
#: which is the property pinned below — not the particular minute counts.
SINGLE_BUILD_AUTH_JOBS = ("auth-perimeter-e2e", "terminal-auth-multiuser-e2e")
PRIVSPLIT_JOB = "privilege-split-e2e"
PRIVSPLIT_TEST_FILE = "tests/e2e/test_privilege_split.py"
PRIVSPLIT_SKIP_GATE_STEP = "Fail the lane on a lane-wide skip"
AUDIT_2C_TEST_FILE = "tests/e2e/test_audit_two_container.py"
AUDIT_2C_STEP = "Run audit two-container mechanism E2E"
#: `success() || failure()`, never the default `success()`: the module runs as a
#: SECOND step behind test_privilege_split.py, so a red first step would skip it
#: — and a skipped step reads like a passing one in a run summary.
AUDIT_2C_STEP_IF = "success() || failure()"


def _jobs_running_e2e_file(wf: dict[str, Any], test_file: str) -> list[str]:
    """Job names with a step whose ``run`` actually EXECUTES *test_file*.

    ``--ignore=<file>`` is stripped before matching, so the shared ``e2e-tests``
    lane — which names every marked file precisely to skip it — is never counted
    as a home. That subtraction is what makes this the "arrived" half of the
    marker guard rather than a restatement of the "left" half.
    """
    homes = []
    for name, spec in _jobs(wf).items():
        for step in spec.get("steps") or []:
            if test_file in str(step.get("run", "")).replace(f"--ignore={test_file}", ""):
                homes.append(name)
                break
    return sorted(homes)


def _gate_checks_lane(wf: dict[str, Any], job: str) -> bool:
    """Does the gate script's ``check_pr_lane`` roll-up examine *job*'s result?"""
    pattern = rf'check_pr_lane\s+"[^"]*"\s+"\$\{{\{{\s*needs\.{re.escape(job)}\.result\s*\}}\}}"'
    return re.search(pattern, _gate_run_text(wf)) is not None


def _unwired_dockerbuild_modules(wf: dict[str, Any]) -> dict[str, list[str]]:
    """Marked e2e files with no lane that both runs them and gates the merge."""
    needs = _jobs(wf)[GATE_JOB]["needs"]
    unwired = {}
    for test_file in _dockerbuild_marked_e2e_files():
        homes = _jobs_running_e2e_file(wf, test_file)
        gating = [j for j in homes if j in needs and _gate_checks_lane(wf, j)]
        if not gating:
            unwired[test_file] = homes
    return unwired


def test_every_dockerbuild_marked_module_reaches_a_gating_lane(workflow: dict[str, Any]) -> None:
    """A marked module is ``--ignore``'d out of the shared lane, so the only
    thing that can run it is a job of its own — and the only thing that makes
    that job's result matter is the gate. Delete either end and the module
    silently runs nowhere, or runs and is never looked at, while every check
    stays green. This is the guard the e2e modules' own CI-HONESTY docstrings
    point at, in the general form: a lane added tomorrow is covered the day the
    marker lands, not whenever someone remembers to hand-write four assertions.
    """
    files = _dockerbuild_marked_e2e_files()
    assert files, "expected at least one dockerbuild-marked e2e file (marker scan broke?)"
    unwired = _unwired_dockerbuild_modules(workflow)
    assert unwired == {}, (
        "dockerbuild-marked e2e module(s) with no lane that runs them AND gates the "
        f"merge (file -> jobs that run it): {unwired} — each needs a job, a "
        f"`needs:` entry on '{GATE_JOB}', and a check_pr_lane line reading its result"
    )


def test_every_dockerbuild_marked_module_reaches_a_gating_lane__mutation_lane_stops_running_it() -> (
    None
):
    """The lane survives as a green job that no longer runs the module — the
    cheapest way to lose an e2e lane, and invisible to every other check here."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, FULL_CHAIN_JOB, FULL_CHAIN_RUN_STEP)["run"] = "echo LANE-DELETED\n"
    assert FULL_CHAIN_TEST_FILE in _unwired_dockerbuild_modules(mutated)
    assert AUDIT_2C_TEST_FILE not in _unwired_dockerbuild_modules(mutated)  # others survive
    with pytest.raises(AssertionError):
        test_every_dockerbuild_marked_module_reaches_a_gating_lane(mutated)


def test_every_dockerbuild_marked_module_reaches_a_gating_lane__mutation_gate_drops_the_need() -> (
    None
):
    """Dropping a lane's ``needs`` entry unhooks BOTH modules that ride it."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(PRIVSPLIT_JOB)
    assert sorted(_unwired_dockerbuild_modules(mutated)) == [
        AUDIT_2C_TEST_FILE,
        PRIVSPLIT_TEST_FILE,
    ]
    with pytest.raises(AssertionError):
        test_every_dockerbuild_marked_module_reaches_a_gating_lane(mutated)


def test_every_dockerbuild_marked_module_reaches_a_gating_lane__mutation_gate_stops_checking() -> (
    None
):
    """The one-line regression the finding named: the ``needs`` entry stays (so
    the roll-up still waits) but the gate stops reading the result, and a red
    lane sits inside a green merge gate."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if FULL_CHAIN_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert FULL_CHAIN_JOB in _jobs(mutated)[GATE_JOB]["needs"]  # the self-enforcing half survives
    with pytest.raises(AssertionError):
        test_every_dockerbuild_marked_module_reaches_a_gating_lane(mutated)


def test_full_chain_auth_job_runs_its_module(workflow: dict[str, Any]) -> None:
    """The named step is the lane's whole point; the generic guard above reports
    the module as homeless if it goes, this one says which step to look at."""
    assert (
        FULL_CHAIN_TEST_FILE
        in _find_named_step(workflow, FULL_CHAIN_JOB, FULL_CHAIN_RUN_STEP)["run"]
    )


def test_full_chain_auth_job_runs_its_module__mutation_drops_the_module() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FULL_CHAIN_JOB, FULL_CHAIN_RUN_STEP)
    step["run"] = step["run"].replace(FULL_CHAIN_TEST_FILE, "tests/e2e/test_something_else.py")
    with pytest.raises(AssertionError):
        test_full_chain_auth_job_runs_its_module(mutated)


def test_check_pr_lane_covers_full_chain_auth(workflow: dict[str, Any]) -> None:
    """Only lane in the repo that exercises login -> nginx -> sidecar -> render
    container end to end. Its result must be read where the merge is decided."""
    assert _gate_checks_lane(workflow, FULL_CHAIN_JOB), (
        f"the gate script has no check_pr_lane line reading needs.{FULL_CHAIN_JOB}.result"
    )


def test_check_pr_lane_covers_full_chain_auth__mutation_drops_the_check_line() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if FULL_CHAIN_JOB not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    with pytest.raises(AssertionError):
        test_check_pr_lane_covers_full_chain_auth(mutated)


def test_privilege_split_job_runs_the_audit_two_container_module(workflow: dict[str, Any]) -> None:
    """The audit two-container module rides as a SECOND step in an existing lane
    rather than a job of its own, so neither the ``needs`` half nor the
    check_pr_lane half says anything about it: delete the step and the lane keeps
    its green check. It must also run on a red first step — the default
    ``success()`` would skip it, and skipped reads like passed."""
    step = _find_named_step(workflow, PRIVSPLIT_JOB, AUDIT_2C_STEP)
    assert AUDIT_2C_TEST_FILE in step["run"]
    assert step.get("if") == AUDIT_2C_STEP_IF, (
        f"'{AUDIT_2C_STEP}' must carry `if: {AUDIT_2C_STEP_IF}` so a red "
        f"'{PRIVSPLIT_TEST_FILE}' step cannot hide it; got {step.get('if')!r}"
    )


def test_privilege_split_job_runs_the_audit_two_container_module__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[PRIVSPLIT_JOB]["steps"]
    kept = [s for s in steps if s.get("name") != AUDIT_2C_STEP]
    assert len(kept) == len(steps) - 1, "expected exactly one step to run the audit module"
    _jobs(mutated)[PRIVSPLIT_JOB]["steps"] = kept
    with pytest.raises(AssertionError):
        test_privilege_split_job_runs_the_audit_two_container_module(mutated)


def test_privilege_split_job_runs_the_audit_two_container_module__mutation_defaults_the_if() -> (
    None
):
    """The quiet half of the same regression: the step survives but only runs
    when the module ahead of it passed, so one red step hides the other."""
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, PRIVSPLIT_JOB, AUDIT_2C_STEP)["if"]
    with pytest.raises(AssertionError, match="must carry"):
        test_privilege_split_job_runs_the_audit_two_container_module(mutated)


def _lane_pytest_steps(wf: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return [s for s in _jobs(wf)[job]["steps"] if "uv run pytest" in str(s.get("run", ""))]


def _lane_junit_reports(wf: dict[str, Any], job: str) -> list[str]:
    return [r for step in _lane_pytest_steps(wf, job) for r in _JUNIT_RE.findall(step["run"])]


def test_full_chain_auth_job_fails_on_any_skipped_test(workflow: dict[str, Any]) -> None:
    """Every way this lane can skip is a misconfiguration, not an environment
    gap: the module's only skip paths are "no container runtime" and "daemon not
    responding", and this lane exists precisely to have both. pytest exits 0 on
    an all-skipped run, so the lane would report a green check over a stack it
    never built. The junit report closes that — zero skips, and a non-zero test
    count so an empty selection cannot pass the same check trivially."""
    steps = _lane_pytest_steps(workflow, FULL_CHAIN_JOB)
    reports = _lane_junit_reports(workflow, FULL_CHAIN_JOB)
    assert len(reports) == len(steps), (
        f"every pytest step in '{FULL_CHAIN_JOB}' must write a --junitxml report; got {reports}"
    )
    gate = _find_named_step(workflow, FULL_CHAIN_JOB, FULL_CHAIN_SKIP_GATE_STEP)["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{FULL_CHAIN_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"


def test_full_chain_auth_job_fails_on_any_skipped_test__mutation_drops_the_report() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _lane_pytest_steps(mutated, FULL_CHAIN_JOB)[0]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_full_chain_auth_job_fails_on_any_skipped_test(mutated)


def test_full_chain_auth_job_fails_on_any_skipped_test__mutation_gate_stops_failing() -> None:
    """A gate that prints the skip count without exiting non-zero is decorative:
    the job still reports success over a lane that ran nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FULL_CHAIN_JOB, FULL_CHAIN_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("sys.exit(1)", "pass")
    with pytest.raises(AssertionError, match="must fail the job"):
        test_full_chain_auth_job_fails_on_any_skipped_test(mutated)


def test_privilege_split_lane_fails_on_a_lane_wide_skip(workflow: dict[str, Any]) -> None:
    """Deliberately NOT the zero-skip gate the single-module lanes use.
    ``test_audit_two_container.py`` carries per-test guards for runtimes that
    remap ids or rewrite bind-mount ownership, each documenting the half of the
    property that still holds, and its own ``_degraded_host`` helper already
    FAILS rather than skips on a CI runner for the class it considers a lane
    misconfiguration. What no in-test guard can catch is the module skipping
    WHOLE — no runtime, an unusable mount source, a module-level skipif — because
    then nothing runs to fail. So a report that collected nothing, or in which
    every test skipped, must fail the lane."""
    steps = _lane_pytest_steps(workflow, PRIVSPLIT_JOB)
    reports = _lane_junit_reports(workflow, PRIVSPLIT_JOB)
    assert len(reports) == len(steps), (
        f"every pytest step in '{PRIVSPLIT_JOB}' must write a --junitxml report; got {reports}"
    )
    step = _find_named_step(workflow, PRIVSPLIT_JOB, PRIVSPLIT_SKIP_GATE_STEP)
    gate = step["run"]
    unread = [r for r in reports if r not in gate]
    assert unread == [], f"'{PRIVSPLIT_SKIP_GATE_STEP}' never reads: {unread}"
    assert 'get("skipped"' in gate, "the gate must read the junit skipped count"
    assert "skipped == total" in gate, "the gate must fail on a module that skipped in full"
    assert "sys.exit(1)" in gate, "the gate must fail the job, not just print"
    assert step.get("if") == AUDIT_2C_STEP_IF, (
        f"'{PRIVSPLIT_SKIP_GATE_STEP}' must carry `if: {AUDIT_2C_STEP_IF}`, or a red "
        "step ahead of it skips the very check that reads what the run proved"
    )


def test_privilege_split_lane_fails_on_a_lane_wide_skip__mutation_drops_a_report() -> None:
    """One module reported and one not is the silent-partial-fix shape."""
    mutated = copy.deepcopy(_load_workflow())
    step = _lane_pytest_steps(mutated, PRIVSPLIT_JOB)[1]
    step["run"] = _JUNIT_RE.sub("", step["run"])
    with pytest.raises(AssertionError, match="must write a --junitxml report"):
        test_privilege_split_lane_fails_on_a_lane_wide_skip(mutated)


def test_privilege_split_lane_fails_on_a_lane_wide_skip__mutation_gate_tolerates_all_skipped() -> (
    None
):
    """Downgrading the all-skipped arm to a warning is the whole regression: the
    lane goes back to reporting success over a module that built nothing."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, PRIVSPLIT_JOB, PRIVSPLIT_SKIP_GATE_STEP)
    step["run"] = step["run"].replace("skipped == total", "skipped < 0")
    with pytest.raises(AssertionError, match="skipped in full"):
        test_privilege_split_lane_fails_on_a_lane_wide_skip(mutated)


#: The pytest targets of the full-chain lane's run step: its own module and
#: nothing else, so the stack it builds once serves exactly the cases written
#: for it (the history proofs among them).
FULL_CHAIN_PYTEST_FILES = (FULL_CHAIN_TEST_FILE,)
FULL_CHAIN_PLAYWRIGHT_STEP = "Install Playwright chromium"


def _pytest_targets(run: str) -> tuple[str, ...]:
    """The ``tests/...py`` paths a step's ``pytest`` invocation names, in order."""
    return tuple(re.findall(r"(tests/[\w/.-]+\.py)", run))


def test_full_chain_auth_lane_runs_exactly_its_own_module(workflow: dict[str, Any]) -> None:
    run = _find_named_step(workflow, FULL_CHAIN_JOB, FULL_CHAIN_RUN_STEP)["run"]
    assert _pytest_targets(run) == FULL_CHAIN_PYTEST_FILES, (
        f"'{FULL_CHAIN_JOB}' runs {_pytest_targets(run)}; its stack is built for "
        f"{FULL_CHAIN_PYTEST_FILES} alone"
    )


def test_full_chain_auth_lane_runs_exactly_its_own_module__mutation_adds_a_file() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FULL_CHAIN_JOB, FULL_CHAIN_RUN_STEP)
    step["run"] = step["run"].replace(
        FULL_CHAIN_TEST_FILE, f"{FULL_CHAIN_TEST_FILE} {RECORD_E2E_FILE}"
    )
    with pytest.raises(AssertionError, match="alone"):
        test_full_chain_auth_lane_runs_exactly_its_own_module(mutated)


def test_full_chain_auth_lane_installs_chromium_before_its_module(
    workflow: dict[str, Any],
) -> None:
    """The history proofs drive a real browser against the lane's stack."""
    steps = _jobs(workflow)[FULL_CHAIN_JOB]["steps"]
    names = [step.get("name") for step in steps]
    assert FULL_CHAIN_PLAYWRIGHT_STEP in names, f"'{FULL_CHAIN_JOB}' installs no browser"
    install = steps[names.index(FULL_CHAIN_PLAYWRIGHT_STEP)]
    assert re.search(r"playwright install\b[^\n]*\bchromium\b", install.get("run", "")), install
    assert names.index(FULL_CHAIN_PLAYWRIGHT_STEP) < names.index(FULL_CHAIN_RUN_STEP), (
        f"'{FULL_CHAIN_JOB}' installs the browser after the module runs"
    )


def test_full_chain_auth_lane_installs_chromium_before_its_module__mutation_drops_it() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)[FULL_CHAIN_JOB]
    job["steps"] = [s for s in job["steps"] if s.get("name") != FULL_CHAIN_PLAYWRIGHT_STEP]
    with pytest.raises(AssertionError, match="installs no browser"):
        test_full_chain_auth_lane_installs_chromium_before_its_module(mutated)


def test_full_chain_auth_lane_caps_and_reads_the_playwright_install(
    workflow: dict[str, Any],
) -> None:
    """A trickling apt mirror has to fail the install in minutes, not after the
    lane's 85; the Playwright cache is restored before it and never saved here,
    so this lane cannot write a second copy of what the browser lane stores."""
    steps = _jobs(workflow)[FULL_CHAIN_JOB]["steps"]
    names = [step.get("name") for step in steps]
    install = names.index(FULL_CHAIN_PLAYWRIGHT_STEP)
    cap = steps[install].get("timeout-minutes")
    assert isinstance(cap, int) and cap <= 10, f"'{FULL_CHAIN_JOB}' install cap is {cap!r}"
    uses = [str(step.get("uses", "")) for step in steps]
    restores = [i for i, u in enumerate(uses) if u.startswith("actions/cache/restore@")]
    assert len(restores) == 1 and restores[0] < install, (
        f"'{FULL_CHAIN_JOB}' restores the Playwright cache at {restores}, install at {install}"
    )
    assert not any(u.startswith(("actions/cache/save@", "actions/cache@")) for u in uses), (
        f"'{FULL_CHAIN_JOB}' saves a cache of its own"
    )


def test_full_chain_auth_lane_caps_and_reads_the_playwright_install__mutation_uncapped() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, FULL_CHAIN_JOB, FULL_CHAIN_PLAYWRIGHT_STEP)["timeout-minutes"]
    with pytest.raises(AssertionError, match="install cap"):
        test_full_chain_auth_lane_caps_and_reads_the_playwright_install(mutated)


def test_full_chain_auth_lane_caps_and_reads_the_playwright_install__mutation_saves() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[FULL_CHAIN_JOB]["steps"]
    steps.append({"uses": "actions/cache/save@v6", "with": {"path": "~/apt-archives"}})
    with pytest.raises(AssertionError, match="saves a cache"):
        test_full_chain_auth_lane_caps_and_reads_the_playwright_install(mutated)


def test_full_chain_auth_lane_outbudgets_the_single_build_lanes(workflow: dict[str, Any]) -> None:
    """The lane builds BOTH real framework images — the deployed sidecar
    Dockerfile and the production project image — where each sibling below
    builds one, and its neighbours measure a single such build at 7-11 minutes
    cold. Held to the SAME cap as a one-build lane, the worst legal in-test path
    runs past the ceiling and the lane reds on wall clock rather than on
    anything it asserts. What is pinned is the relation, not the minute counts:
    strictly more work must carry strictly more budget, at both the job cap and
    the step cap under it."""
    jobs = _jobs(workflow)
    job_cap = jobs[FULL_CHAIN_JOB]["timeout-minutes"]
    step_cap = _sole_heavy_step(jobs[FULL_CHAIN_JOB])["timeout-minutes"]
    for sibling in SINGLE_BUILD_AUTH_JOBS:
        sibling_job_cap = jobs[sibling]["timeout-minutes"]
        sibling_step_cap = _sole_heavy_step(jobs[sibling])["timeout-minutes"]
        assert job_cap > sibling_job_cap, (
            f"'{FULL_CHAIN_JOB}' builds two images where '{sibling}' builds one, but "
            f"carries the same or less job budget ({job_cap} vs {sibling_job_cap} minutes)"
        )
        assert step_cap > sibling_step_cap, (
            f"'{FULL_CHAIN_JOB}''s pytest step carries the same or less budget than "
            f"'{sibling}''s ({step_cap} vs {sibling_step_cap} minutes)"
        )


def test_full_chain_auth_lane_outbudgets_the_single_build_lanes__mutation_back_to_sibling_cap() -> (
    None
):
    """The shipped-before state: two builds held to a one-build lane's ceiling."""
    mutated = copy.deepcopy(_load_workflow())
    sibling = _jobs(mutated)[SINGLE_BUILD_AUTH_JOBS[0]]
    job = _jobs(mutated)[FULL_CHAIN_JOB]
    job["timeout-minutes"] = sibling["timeout-minutes"]
    _sole_heavy_step(job)["timeout-minutes"] = _sole_heavy_step(sibling)["timeout-minutes"]
    with pytest.raises(AssertionError, match="job budget"):
        test_full_chain_auth_lane_outbudgets_the_single_build_lanes(mutated)


#: The full-chain lane's measured caps, in minutes: the job, and its one pytest
#: step under it. Measured with the history page deployed (see the job's
#: comment for the wall clock); a change to either is a re-measurement.
FULL_CHAIN_JOB_CAP = 85
FULL_CHAIN_STEP_CAP = 75


def test_full_chain_auth_lane_caps_are_the_measured_ones(workflow: dict[str, Any]) -> None:
    job = _jobs(workflow)[FULL_CHAIN_JOB]
    caps = (job["timeout-minutes"], _sole_heavy_step(job)["timeout-minutes"])
    assert caps == (FULL_CHAIN_JOB_CAP, FULL_CHAIN_STEP_CAP), (
        f"'{FULL_CHAIN_JOB}' carries caps {caps}, not the measured "
        f"{(FULL_CHAIN_JOB_CAP, FULL_CHAIN_STEP_CAP)}: re-measure the lane and pin both"
    )


def test_full_chain_auth_lane_caps_are_the_measured_ones__mutation_step_cap() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _sole_heavy_step(_jobs(mutated)[FULL_CHAIN_JOB])["timeout-minutes"] = FULL_CHAIN_STEP_CAP + 1
    with pytest.raises(AssertionError, match="re-measure"):
        test_full_chain_auth_lane_caps_are_the_measured_ones(mutated)


#: The full-chain lane's exact-name cleanup and what its history page leaves:
#: the page's container and the project image ``osprey up`` builds, which the
#: page runs (``tests/e2e/test_full_chain_auth.py`` pins the render's tag).
FULL_CHAIN_CLEANUP_STEP = "Clean up any stranded full-chain-auth resources"
FULL_CHAIN_HISTORY_CONTAINER = "osprey-e2e-full-chain-auth-history"
FULL_CHAIN_PROJECT_IMAGE = "osprey-e2e-full-chain-auth:local"


def test_full_chain_auth_cleanup_removes_the_history_page_and_its_image(
    workflow: dict[str, Any],
) -> None:
    run = _cleanup_step_run(workflow, FULL_CHAIN_JOB, FULL_CHAIN_CLEANUP_STEP)
    assert FULL_CHAIN_HISTORY_CONTAINER in _CONTAINER_REMOVAL.findall(run), (
        f"'{FULL_CHAIN_CLEANUP_STEP}' never removes {FULL_CHAIN_HISTORY_CONTAINER}"
    )
    assert re.search(rf"(?m)^\s*docker rmi -f {re.escape(FULL_CHAIN_PROJECT_IMAGE)}\b", run), (
        f"'{FULL_CHAIN_CLEANUP_STEP}' never removes the history page's image "
        f"{FULL_CHAIN_PROJECT_IMAGE}"
    )


def test_full_chain_auth_cleanup_removes_the_history_page_and_its_image__mutation_drops_rmi() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FULL_CHAIN_JOB, FULL_CHAIN_CLEANUP_STEP)
    original = step["run"]
    step["run"] = original.replace(f"docker rmi -f {FULL_CHAIN_PROJECT_IMAGE} || true\n", "")
    assert step["run"] != original, "no project-image removal — mutation is stale"
    with pytest.raises(AssertionError, match="history page's image"):
        test_full_chain_auth_cleanup_removes_the_history_page_and_its_image(mutated)


# ---------------------------------------------------------------------------
# Model-spending lanes: `full-ci` on a pull request, on demand on main
# ---------------------------------------------------------------------------
#
# The SPENDING_LANES jobs drive real Claude sessions through the gateway.
# Their cost scales as turns x context times push volume (the measurement is
# recorded in docs/source/contributing/workflow.rst, "Model-Spending Lanes"),
# so the lever is WHEN, not WHAT: each of these lanes runs on a labeled PR or on
# the revalidation dispatch — and on nothing else. Both arms are pinned
# below, and so is the absence of a third: a schedule arm (or a `schedule`
# trigger on the workflow) spends model tokens with nobody watching and
# turns red where nobody reads it. Dropping the label clause is the other
# regression — the lane is back on every push, and so is the bill.

SPENDING_LANES = frozenset(
    {
        "agentic-per-preset",
        "e2e-tests",
        "dispatch-deploy-e2e",
        "single-user-narrowing-e2e",
        "host-record-e2e",
        "dispatch-overlay-e2e",
        "scan-agentic-e2e",
        "nextcloud-talk-bridge-e2e",
        "gchat-bridge-e2e",
        "target-switch-agentic-e2e",
    }
)
# Lanes that carry the secret but spend nothing worth gating: deploy-e2e needs
# the key only for `osprey up`'s .env preflight and never calls a model;
# dockerfile-e2e drives a single haiku turn; the two benchmark jobs run only on
# a deliberate run_benchmarks dispatch, never on a PR. Gating them by label would drop
# per-PR deploy coverage for no saving, so they keep the per-PR shape — and a
# NEW secret-bearing lane must be listed in one set or the other, on purpose.
SECRET_FREE_LANES = frozenset({"deploy-e2e", "dockerfile-e2e", BENCHMARKS_JOB, TIER2_JOB})
FULL_CI_LABEL_CLAUSE = "contains(github.event.pull_request.labels.*.name, 'full-ci')"
SCHEDULE_ARM = "github.event_name == 'schedule'"
# The pre-gating shape: the PR conjunction closed right after the actor guard.
# Its absence is what proves the label clause sits INSIDE that conjunction,
# not OR-ed beside it (where it would admit every PR again).
_UNLABELED_PR_ARM_CLOSE = "&& github.actor != 'dependabot[bot]')"


def _secret_bearing_jobs(wf: dict[str, Any]) -> set[str]:
    return {name for name in _jobs(wf) if _job_declares_secret(wf, name, SECRET_TOKEN)}


def test_every_secret_bearing_lane_declares_its_spend_posture(workflow: dict[str, Any]) -> None:
    """A lane that holds the gateway key is either a spending lane (label /
    revalidation) or listed as secret-free. No third category: a new
    agentic lane that nobody classifies would run on every push by default,
    which is exactly how the bill got where it was."""
    unclassified = _secret_bearing_jobs(workflow) - SPENDING_LANES - SECRET_FREE_LANES
    assert unclassified == set(), (
        f"secret-bearing lanes with no declared posture: {sorted(unclassified)}"
    )
    # and the classification is not stale the other way either
    assert SPENDING_LANES <= _secret_bearing_jobs(workflow)


def test_every_secret_bearing_lane_declares_its_spend_posture__mutation_adds_unclassified_lane() -> (
    None
):
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"]["new-agentic-lane"] = copy.deepcopy(mutated["jobs"]["scan-agentic-e2e"])
    with pytest.raises(AssertionError, match="no declared posture"):
        test_every_secret_bearing_lane_declares_its_spend_posture(mutated)


@pytest.mark.parametrize("lane", sorted(SPENDING_LANES))
def test_spending_lane_runs_only_under_label_or_revalidation(
    workflow: dict[str, Any], lane: str
) -> None:
    """Both arms, no schedule arm, and the label INSIDE the pull-request arm."""
    condition = _jobs(workflow)[lane]["if"]
    assert FULL_CI_LABEL_CLAUSE in condition, f"{lane}: no full-ci label clause"
    assert SCHEDULE_ARM not in condition, f"{lane}: schedule arm — an unattended run"
    assert _DEPENDABOT_GUARD in condition, f"{lane}: Dependabot guard dropped"
    assert "workflow_dispatch" in condition, f"{lane}: dispatch arm dropped"
    assert _UNLABELED_PR_ARM_CLOSE not in condition, (
        f"{lane}: the pull-request arm closes before the label clause — "
        "an unlabeled PR would run this lane again"
    )


def test_spending_lane_gating__mutation_drops_the_label_clause() -> None:
    """The regression that restores the bill: the PR arm admits every push."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)["scan-agentic-e2e"]
    job["if"] = job["if"].replace(f"\n  && {FULL_CI_LABEL_CLAUSE}", "")
    assert FULL_CI_LABEL_CLAUSE not in job["if"], "mutation is stale"
    with pytest.raises(AssertionError, match="label clause|closes before"):
        test_spending_lane_runs_only_under_label_or_revalidation(mutated, "scan-agentic-e2e")


def test_spending_lane_gating__mutation_ors_the_label_beside_the_pr_arm() -> None:
    """The subtler regression: the clause is present but OR-ed outside the
    pull-request conjunction, which admits every same-repo PR exactly as
    before and reads as gated."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)["scan-agentic-e2e"]
    job["if"] = job["if"].replace(
        f"\n  && {FULL_CI_LABEL_CLAUSE})", f")\n|| {FULL_CI_LABEL_CLAUSE}"
    )
    assert _UNLABELED_PR_ARM_CLOSE in job["if"], "mutation is stale"
    with pytest.raises(AssertionError, match="closes before the label clause"):
        test_spending_lane_runs_only_under_label_or_revalidation(mutated, "scan-agentic-e2e")


def test_spending_lane_gating__mutation_adds_a_schedule_arm() -> None:
    """The other failure: a lane that runs unattended."""
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)["agentic-per-preset"]
    job["if"] = job["if"] + f" || {SCHEDULE_ARM}"
    with pytest.raises(AssertionError, match="schedule arm"):
        test_spending_lane_runs_only_under_label_or_revalidation(mutated, "agentic-per-preset")


def test_spending_lanes_name_no_base_ref(workflow: dict[str, Any]) -> None:
    """The label is the only pull-request gate on a spending lane: no clause
    names the base branch. Widening the workflow trigger to another base
    therefore cannot start a spending lane on an unlabeled pull request."""
    scoped = sorted(lane for lane in SPENDING_LANES if "base.ref" in _jobs(workflow)[lane]["if"])
    assert scoped == [], f"spending lanes whose if: names a base ref: {scoped}"


def test_spending_lanes_name_no_base_ref__mutation_adds_an_integration_base() -> None:
    mutated = copy.deepcopy(_load_workflow())
    job = _jobs(mutated)["scan-agentic-e2e"]
    job["if"] = job["if"] + " || startsWith(github.event.pull_request.base.ref, 'integration/')"
    with pytest.raises(AssertionError, match="names a base ref"):
        test_spending_lanes_name_no_base_ref(mutated)


def test_workflow_has_no_schedule_trigger(workflow: dict[str, Any]) -> None:
    """No cron at all: every run of this workflow is one somebody asked for
    (a push, a pull request, a dispatch), so every red has a reader. A
    schedule would also be the only trigger that can spend model tokens with
    nobody in the loop; the lane `if:` checks above pin the arm, this pins
    the event."""
    assert "schedule" not in workflow[True], "workflow declares a schedule trigger"


def test_workflow_has_no_schedule_trigger__mutation_adds_a_cron() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated[True]["schedule"] = [{"cron": "0 7 * * *"}]
    with pytest.raises(AssertionError, match="schedule trigger"):
        test_workflow_has_no_schedule_trigger(mutated)


def test_secret_free_lanes_keep_the_per_pr_shape(workflow: dict[str, Any]) -> None:
    """deploy-e2e and dockerfile-e2e are the per-PR proof that a render builds
    and boots; they cost runner minutes, not tokens, and must not hide behind
    the label."""
    for lane in ("deploy-e2e", "dockerfile-e2e"):
        condition = _jobs(workflow)[lane]["if"]
        assert FULL_CI_LABEL_CLAUSE not in condition, f"{lane}: label-gated but spends nothing"
        assert "pull_request" in condition


def test_secret_free_lanes_keep_the_per_pr_shape__mutation_gates_deploy_by_label() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)["deploy-e2e"]["if"] = _jobs(mutated)["scan-agentic-e2e"]["if"]
    with pytest.raises(AssertionError, match="spends nothing"):
        test_secret_free_lanes_keep_the_per_pr_shape(mutated)


def test_workflow_fires_when_the_full_ci_label_is_applied(workflow: dict[str, Any]) -> None:
    """A label clause in a job's ``if:`` is inert unless the workflow subscribes
    to the ``labeled`` activity: with the three default types only, adding
    ``full-ci`` starts nothing and the lanes wait for the next push."""
    types = workflow[True]["pull_request"]["types"]
    assert "labeled" in types, f"pull_request.types must include 'labeled'; got {types}"
    # the defaults survive — the workflow still runs on open / push / reopen
    assert {"opened", "synchronize", "reopened"} <= set(types)


def test_workflow_fires_when_the_full_ci_label_is_applied__mutation_drops_types() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated[True]["pull_request"]["types"]
    with pytest.raises(KeyError):
        test_workflow_fires_when_the_full_ci_label_is_applied(mutated)


def test_workflow_fires_when_the_full_ci_label_is_applied__mutation_drops_labeled() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated[True]["pull_request"]["types"] = ["opened", "synchronize", "reopened"]
    with pytest.raises(AssertionError, match="must include 'labeled'"):
        test_workflow_fires_when_the_full_ci_label_is_applied(mutated)


#: The gate step's two skipped-lanes blocks, by the start of each heading it echoes.
DEPENDABOT_SUMMARY_HEADING = "### ⚠️ Secret-dependent lanes did not run"
UNLABELED_SUMMARY_HEADING = "### ℹ️ Model-spending lanes did not run"
SUMMARY_HEADINGS = (DEPENDABOT_SUMMARY_HEADING, UNLABELED_SUMMARY_HEADING)
SUMMARY_BULLET = re.compile(r'echo "- (.+)"')


def _summary_block_lanes(run: str, heading: str) -> list[str]:
    """The lanes one summary block prints, a line naming two split into its two."""
    block = run.partition(f'echo "{heading}')[2].partition('} >> "$GITHUB_STEP_SUMMARY"')[0]
    bullets = (SUMMARY_BULLET.fullmatch(line.strip()) for line in block.splitlines())
    return [lane for bullet in bullets if bullet for lane in bullet[1].split(" / ")]


def test_gate_summary_tells_an_unlabeled_pr_what_did_not_run(workflow: dict[str, Any]) -> None:
    """Same rule as the Dependabot block: a green gate with skipped lanes says
    so where a reviewer sees it, under each lane's real name, and prints the one
    command that runs them."""
    step = _find_named_step(workflow, GATE_JOB, "Check all jobs status")
    assert step["env"]["HAS_FULL_CI"] == "${{ " + FULL_CI_LABEL_CLAUSE + " }}"
    assert "--add-label full-ci" in step["run"]
    prefixes = _required_lane_prefixes().values()
    for heading in SUMMARY_HEADINGS:
        lanes = _summary_block_lanes(step["run"], heading)
        for prefix in prefixes:
            assert lanes.count(prefix) == 1, f"{prefix!r} must be one bullet under {heading!r}"
    # The label gates these lanes and no other, so its block names nothing else.
    assert sorted(_summary_block_lanes(step["run"], UNLABELED_SUMMARY_HEADING)) == sorted(prefixes)


def test_gate_summary_tells_an_unlabeled_pr_what_did_not_run__mutation_drops_the_command() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    step["run"] = step["run"].replace("--add-label full-ci", "")
    with pytest.raises(AssertionError):
        test_gate_summary_tells_an_unlabeled_pr_what_did_not_run(mutated)


@pytest.mark.parametrize("heading", SUMMARY_HEADINGS, ids=("dependabot", "unlabeled"))
def test_gate_summary_tells_an_unlabeled_pr_what_did_not_run__mutation_drops_a_lane(
    heading: str,
) -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    before, marker, after = step["run"].partition(heading)
    step["run"] = before + marker + after.replace('echo "- Host Record E2E"', "", 1)
    with pytest.raises(AssertionError, match="Host Record E2E"):
        test_gate_summary_tells_an_unlabeled_pr_what_did_not_run(mutated)


# ---------------------------------------------------------------------------
# The tag gate in release.yml reads these lanes by their rendered job names
# ---------------------------------------------------------------------------
#
# `release.yml`'s `verify-full-ci` job refuses to publish a tag unless every
# spending lane concluded `success` on the tagged tree, and it finds those
# lanes by matching a prefix of each job's rendered `name:`. That string lives
# in one workflow and is defined in another, so a rename would otherwise go
# unnoticed until it failed a release — which is the worst possible moment.
# Both directions are pinned here: the gate covers every spending lane, and
# every prefix still names exactly one job in ci.yml.

RELEASE_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"
RELEASE_GATE_JOB = "verify-full-ci"
#: The gate both release workflows run, and the env var each one lists its lanes in.
GREEN_LANES_SCRIPT = "scripts/ci/require_green_lanes.py"
GREEN_LANES_ENV = "REQUIRED_LANES"


@functools.cache
def _green_lanes_module() -> Any:
    """``scripts/ci/require_green_lanes.py``, loaded by path (``scripts/`` is no package)."""
    path = Path(__file__).resolve().parents[2] / GREEN_LANES_SCRIPT
    spec = importlib.util.spec_from_file_location("require_green_lanes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_step(release: dict[str, Any], job: str) -> dict[str, Any]:
    """The one step of *job* that runs the shared lane gate."""
    steps = [s for s in release["jobs"][job]["steps"] if GREEN_LANES_SCRIPT in s.get("run", "")]
    assert len(steps) == 1, f"{job} must run {GREEN_LANES_SCRIPT} in exactly one step"
    return steps[0]


def _gate_lanes(release: dict[str, Any], job: str) -> dict[str, str]:
    """``{ci.yml job id: rendered-name prefix}`` the gate in *job* requires, read by
    the gate script's own parser so the test and the gate cannot read it apart."""
    step = _gate_step(release, job)
    assert f'--lanes "${GREEN_LANES_ENV}"' in step["run"], (
        f"{job} does not pass ${GREEN_LANES_ENV} to the gate"
    )
    return _green_lanes_module().parse_lanes(step["env"][GREEN_LANES_ENV])


def _required_lane_prefixes() -> dict[str, str]:
    return _gate_lanes(yaml.safe_load(RELEASE_YML.read_text()), RELEASE_GATE_JOB)


def test_the_tag_gate_covers_every_spending_lane() -> None:
    """A lane added to the label gate but not to the tag gate is a lane a
    release can skip silently."""
    assert set(_required_lane_prefixes()) == set(SPENDING_LANES)


def _assert_lane_prefixes_match(
    workflow: dict[str, Any], lanes: dict[str, str], source: str = "release.yml"
) -> None:
    jobs = _jobs(workflow)
    for lane, prefix in lanes.items():
        assert jobs[lane]["name"].startswith(prefix), (
            f"{lane}: {source} looks for {prefix!r}, ci.yml renders {jobs[lane]['name']!r}"
        )
        others = [n for n, j in jobs.items() if n != lane and j.get("name", "").startswith(prefix)]
        assert others == [], f"{prefix!r} also matches {others}"


def test_the_tag_gate_names_match_the_rendered_job_names(workflow: dict[str, Any]) -> None:
    """Each prefix identifies its own lane and nothing else. The matrix lane
    renders a job for every preset, so a prefix is the only stable handle."""
    _assert_lane_prefixes_match(workflow, _required_lane_prefixes())


def test_the_tag_gate_names_match__mutation_renames_a_lane() -> None:
    """The regression this exists for: ci.yml renames a lane, release.yml is
    left looking for a job that no run will ever contain, and the gate then
    fails a release instead of a test."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)["scan-agentic-e2e"]["name"] = "Scan Stack (agentic)"
    with pytest.raises(AssertionError, match="release.yml looks for"):
        test_the_tag_gate_names_match_the_rendered_job_names(mutated)


def test_the_tag_gate_blocks_the_publish() -> None:
    """The gate is only a gate if the jobs that build and publish wait on it."""
    release = yaml.safe_load(RELEASE_YML.read_text())
    for job in ("build", "publish-to-pypi"):
        assert RELEASE_GATE_JOB in release["jobs"][job]["needs"], (
            f"{job} does not wait for {RELEASE_GATE_JOB}"
        )
    assert release["jobs"][RELEASE_GATE_JOB]["permissions"]["actions"] == "read", (
        "reading another workflow's runs needs the actions:read scope"
    )


# ---------------------------------------------------------------------------
# The publish uploads every member before the framework, and can be resumed
# ---------------------------------------------------------------------------
#
# The framework declares a hard dependency on each workspace member, and PyPI
# accepts files one at a time. Uploading the framework first would leave it live
# with a dependency nothing satisfies whenever a member upload is refused (an
# unregistered trusted publisher, a name conflict). Members go first, from a
# directory holding only their files; the framework goes last. `skip-existing`
# on both lets a re-pushed tag finish a publish that stopped halfway instead of
# failing on "File already exists".

PUBLISH_JOB = "publish-to-pypi"
PUBLISH_ACTION = "pypa/gh-action-pypi-publish"


def _load_release() -> dict[str, Any]:
    return yaml.safe_load(RELEASE_YML.read_text())


def _publish_steps(release: dict[str, Any]) -> list[dict[str, Any]]:
    steps = release["jobs"][PUBLISH_JOB]["steps"]
    return [s for s in steps if str(s.get("uses", "")).startswith(PUBLISH_ACTION)]


def _packages_dir(step: dict[str, Any]) -> str:
    return str(step.get("with", {}).get("packages-dir", "dist/")).rstrip("/")


def test_the_publish_uploads_members_before_the_framework(
    release: dict[str, Any] | None = None,
) -> None:
    """The first upload carries only member files; the framework's upload is the last."""
    release = release if release is not None else _load_release()
    publishes = _publish_steps(release)
    assert len(publishes) >= 2, "members and the framework must upload in separate steps"
    member_dir = _packages_dir(publishes[0])
    assert member_dir != "dist", "the first upload must not be the whole dist/"
    assert _packages_dir(publishes[-1]) == "dist", "the framework uploads last, from dist/"

    steps = release["jobs"][PUBLISH_JOB]["steps"]
    first_publish = steps.index(publishes[0])
    stagers = [s for s in steps[:first_publish] if member_dir in s.get("run", "")]
    assert stagers, f"no step before the first upload stages files into {member_dir}/"
    script = stagers[-1]["run"]
    assert "osprey_framework-" in script and "mv " in script, (
        "the member directory must be staged by moving every file except the framework's"
    )


def test_every_publish_step_skips_existing_files(release: dict[str, Any] | None = None) -> None:
    """A re-pushed tag must be able to complete a partial publish."""
    release = release if release is not None else _load_release()
    publishes = _publish_steps(release)
    assert publishes, f"{PUBLISH_JOB} has no {PUBLISH_ACTION} step"
    for step in publishes:
        assert step.get("with", {}).get("skip-existing") is True, (
            f"{step.get('name')!r} must set skip-existing: true"
        )


def test_the_publish_order__mutation_one_upload_of_all_of_dist() -> None:
    """The single upload over dist/ is the shape that goes framework-first."""
    mutated = copy.deepcopy(_load_release())
    steps = mutated["jobs"][PUBLISH_JOB]["steps"]
    publishes = _publish_steps(mutated)
    for step in publishes[:-1]:
        steps.remove(step)
    with pytest.raises(AssertionError, match="separate steps"):
        test_the_publish_uploads_members_before_the_framework(mutated)


def test_the_publish_order__mutation_framework_first() -> None:
    mutated = copy.deepcopy(_load_release())
    steps = mutated["jobs"][PUBLISH_JOB]["steps"]
    first, last = _publish_steps(mutated)[0], _publish_steps(mutated)[-1]
    i, j = steps.index(first), steps.index(last)
    steps[i], steps[j] = last, first
    with pytest.raises(AssertionError):
        test_the_publish_uploads_members_before_the_framework(mutated)


def test_the_publish_order__mutation_drops_skip_existing() -> None:
    mutated = copy.deepcopy(_load_release())
    del _publish_steps(mutated)[0]["with"]["skip-existing"]
    with pytest.raises(AssertionError, match="skip-existing"):
        test_every_publish_step_skips_existing_files(mutated)


# Only the upstream repository publishes. A version tag is both the version
# source and the publish trigger, so any copy of the repository that receives
# one would otherwise run the whole release chain. Every job carries the
# repository guard as a top-level conjunct of its `if:`, so the guard fails
# closed: a later edit that rewires `needs` cannot un-gate a publish.

UPSTREAM_REPOSITORY = "als-apg/osprey"
_REPOSITORY_GUARD = f"github.repository == '{UPSTREAM_REPOSITORY}'"


def _top_level_conjuncts(expression: str) -> list[str] | None:
    """Split an Actions expression on `&&` outside parentheses and quotes.

    Returns None when the expression has a top-level `||`: an OR'd guard
    does not fail closed, so it never counts as a guard.
    """
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    parts: list[str] = []
    depth = 0
    quoted = False
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "'":
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        elif not quoted and depth == 0 and text.startswith("||", i):
            return None
        elif not quoted and depth == 0 and text.startswith("&&", i):
            parts.append(text[start:i])
            start = i + 2
            i += 1
        i += 1
    parts.append(text[start:])
    return [" ".join(part.split()) for part in parts]


# The same invariant over every workflow: whatever a job is called and
# whichever file holds it, a job that publishes (to PyPI, a release, gh-pages or
# a container registry) runs only upstream. Publishers are found by the shape
# of their steps, never by a list of file names, so a new workflow is covered
# the moment it exists. The guard is product behaviour on every copy of the
# repository, not scaffolding for one copy: a mirror or fork builds and tests
# but publishes nothing.

WORKFLOWS_DIR = RELEASE_YML.parent

_PUBLISHING_ACTIONS = (
    "pypa/gh-action-pypi-publish@",
    "peaceiris/actions-gh-pages@",
    "softprops/action-gh-release@",
)


def _all_workflows() -> dict[str, dict[str, Any]]:
    """Every workflow under `.github/workflows/`, keyed by file name."""
    paths = sorted([*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")])
    return {path.name: yaml.safe_load(path.read_text()) for path in paths}


def _run_publishes(script: str) -> bool:
    """Whether a `run:` script pushes an artifact somewhere public."""
    if "docker push" in script or "buildx imagetools create" in script:
        return True
    if "docker buildx build" in script and "--push" in script:
        return True
    if "gh release create" in script or "twine upload" in script:
        return True
    return "git push" in script and "gh-pages" in script


def _step_publishes(step: dict[str, Any]) -> bool:
    uses = step.get("uses")
    run = step.get("run")
    return (isinstance(uses, str) and uses.startswith(_PUBLISHING_ACTIONS)) or (
        isinstance(run, str) and _run_publishes(run)
    )


def _step_name(step: dict[str, Any], index: int) -> str:
    return str(step.get("name") or f"step {index}")


def _publishing_steps(job: dict[str, Any]) -> list[str]:
    """Names of the steps in `job` that publish, judged by declared step shape."""
    return [
        _step_name(step, index)
        for index, step in enumerate(job.get("steps") or [])
        if _step_publishes(step)
    ]


def _has_repository_guard(condition: object) -> bool:
    conjuncts = _top_level_conjuncts(condition) if isinstance(condition, str) else None
    return conjuncts is not None and _REPOSITORY_GUARD in conjuncts


def _calls_a_publisher(job: dict[str, Any], workflows: dict[str, dict[str, Any]]) -> bool:
    """Whether a job-level `uses:` calls a local workflow that has a publishing job."""
    uses = job.get("uses")
    if not isinstance(uses, str) or not uses.startswith("./.github/workflows/"):
        return False
    called = workflows.get(uses.removeprefix("./.github/workflows/"))
    if called is None:
        return False
    return any(
        _publishing_steps(inner) or _calls_a_publisher(inner, workflows)
        for inner in (called.get("jobs") or {}).values()
    )


def _unguarded_publishers(workflows: dict[str, dict[str, Any]]) -> list[str]:
    """Publishing jobs and steps whose `if:` does not require the upstream repository.

    A job is named `<file>:<job>`, a step `<file>:<job>/<step name>`. A guard on
    the job covers its steps; otherwise each publishing step needs its own.
    """
    unguarded = []
    for file_name, workflow in workflows.items():
        for job_name, job in (workflow.get("jobs") or {}).items():
            if _has_repository_guard(job.get("if")):
                continue
            if _calls_a_publisher(job, workflows):
                unguarded.append(f"{file_name}:{job_name}")
            for index, step in enumerate(job.get("steps") or []):
                if _step_publishes(step) and not _has_repository_guard(step.get("if")):
                    unguarded.append(f"{file_name}:{job_name}/{_step_name(step, index)}")
    return unguarded


def _publishing_jobs(workflows: dict[str, dict[str, Any]]) -> list[str]:
    return [
        f"{file_name}:{job_name}"
        for file_name, workflow in workflows.items()
        for job_name, job in (workflow.get("jobs") or {}).items()
        if _publishing_steps(job) or _calls_a_publisher(job, workflows)
    ]


def test_every_publishing_job_runs_only_upstream() -> None:
    """A mirror or fork publishes nothing, from any workflow, on any trigger."""
    workflows = _all_workflows()
    assert _publishing_jobs(workflows), "the publisher detector matches no job at all"
    assert _unguarded_publishers(workflows) == []


def test_every_publishing_job_runs_only_upstream__mutation_unguards_the_docs_deploy() -> None:
    workflows = _all_workflows()
    for step in workflows["docs.yml"]["jobs"]["build"]["steps"]:
        if step.get("name") == "Deploy to GitHub Pages":
            step["if"] = "steps.plan.outputs.deploy == 'true'"
    assert _unguarded_publishers(workflows) == ["docs.yml:build/Deploy to GitHub Pages"]


def test_every_publishing_job_runs_only_upstream__mutation_ors_the_guard() -> None:
    workflows = _all_workflows()
    for step in workflows["docs.yml"]["jobs"]["build"]["steps"]:
        if step.get("name") == "Deploy to GitHub Pages":
            step["if"] = f"{_REPOSITORY_GUARD} || steps.plan.outputs.deploy == 'true'"
    assert _unguarded_publishers(workflows) == ["docs.yml:build/Deploy to GitHub Pages"]


def test_every_publishing_job_runs_only_upstream__mutation_unguards_the_nextcloud_fixture() -> None:
    workflows = _all_workflows()
    del workflows["build-nextcloud-fixture.yml"]["jobs"]["build"]["if"]
    assert _unguarded_publishers(workflows) == [
        "build-nextcloud-fixture.yml:build/Build and push the fixture"
    ]


def test_every_publishing_job_runs_only_upstream__mutation_unguards_the_pypi_publish() -> None:
    workflows = _all_workflows()
    workflows["release.yml"]["jobs"]["publish-to-pypi"]["if"] = (
        "startsWith(github.ref, 'refs/tags/v')"
    )
    assert _unguarded_publishers(workflows) == [
        "release.yml:publish-to-pypi/Publish workspace members to PyPI",
        "release.yml:publish-to-pypi/Publish framework to PyPI",
    ]


def test_every_publishing_job_runs_only_upstream__mutation_adds_a_pushing_job_to_ci() -> None:
    workflows = _all_workflows()
    workflows["ci.yml"]["jobs"]["push-image"] = {
        "runs-on": "ubuntu-latest",
        "steps": [{"name": "Push", "run": "docker push x"}],
    }
    assert _unguarded_publishers(workflows) == ["ci.yml:push-image/Push"]


# ---------------------------------------------------------------------------
# osprey-connectors releases from its own tags, through its own workflow
# ---------------------------------------------------------------------------
#
# The connectors package has its own calendar stream on osprey-connectors-v2*
# tags, so it publishes from release-connectors.yml and release.yml stops
# building it. What is pinned: the connectors workflow fires on those tags and
# nothing else (no dispatch); its publish waits on the CI gate, the build and
# the install check, under its own PyPI environment, with the OIDC scope on
# that job alone; its version check refuses a build that is not the tag; and
# the framework release builds exactly its lockstep members.

CONNECTORS_RELEASE_YML = RELEASE_YML.with_name("release-connectors.yml")
CONNECTORS_GATE_JOB = "verify-ci"
CONNECTORS_BUILD_JOB = "build"
CONNECTORS_INSTALL_JOB = "verify-install"
CONNECTORS_TAG_GUARD = (
    f"{_REPOSITORY_GUARD} && startsWith(github.ref, 'refs/tags/osprey-connectors-v2')"
)
CONNECTORS_ENVIRONMENT = "pypi-connectors"
CONNECTORS_VERSION_STEP = "Verify built versions match tag"
LEAN_WALK_SCRIPT = "scripts/ci/connectors_lean_import_walk.py"
#: The always-on lanes that prove the package. The backward-compatibility lanes
#: join the gate whenever ci.yml carries them; the test below fails if one is missing.
CONNECTORS_LANES = frozenset({"test", "lint", "package", "connectors-standalone"})
CONNECTORS_COMPAT_LANES = frozenset({"connectors-compat", "framework-at-connectors-floor"})


def _load_connectors_release() -> dict[str, Any]:
    return yaml.safe_load(CONNECTORS_RELEASE_YML.read_text())


def _github_tag_glob_matches(pattern: str, tag: str) -> bool:
    """A GitHub tag filter: ``*`` matches any run of characters but ``/``."""
    regex = "".join("[^/]*" if c == "*" else re.escape(c) for c in pattern)
    return re.fullmatch(regex, tag) is not None


def _tag_filters(release: dict[str, Any]) -> list[str]:
    # YAML 1.1 reads the bare `on:` key as True.
    return release[True]["push"]["tags"]


@pytest.mark.parametrize(
    ("tag", "connectors", "framework"),
    [
        ("osprey-connectors-v2026.10.0", True, False),
        ("osprey-connectors-v2026.10.0b1", True, False),
        ("osprey-connectors-v2027.1.3rc2", True, False),
        ("osprey-connectors-v0.1.0", False, False),  # retired; PyPI carries 0.1.0
        ("v2026.10.0", False, True),
        ("v2026.9.0b3", False, True),
        ("checkpoint-2026-10-01", False, False),
    ],
)
def test_each_release_workflow_fires_only_on_its_own_tags(
    tag: str, connectors: bool, framework: bool
) -> None:
    fires = {
        "connectors": any(
            _github_tag_glob_matches(p, tag) for p in _tag_filters(_load_connectors_release())
        ),
        "framework": any(_github_tag_glob_matches(p, tag) for p in _tag_filters(_load_release())),
    }
    assert fires == {"connectors": connectors, "framework": framework}, tag


def test_the_connectors_release_has_no_other_trigger() -> None:
    """No dispatch: nothing publishes a commit nobody tagged."""
    triggers = _load_connectors_release()[True]
    assert triggers == {"push": {"tags": ["osprey-connectors-v2*"]}}


def test_every_connectors_job_runs_only_on_a_connectors_tag(
    release: dict[str, Any] | None = None,
) -> None:
    release = release if release is not None else _load_connectors_release()
    for name, job in release["jobs"].items():
        assert job.get("if") == CONNECTORS_TAG_GUARD, (
            f"{name} is not guarded on the repository and the tag ref"
        )


def test_every_connectors_job_runs_only_on_a_connectors_tag__mutation_drops_a_guard() -> None:
    mutated = _load_connectors_release()
    del mutated["jobs"]["publish-to-pypi"]["if"]
    with pytest.raises(AssertionError, match="publish-to-pypi is not guarded"):
        test_every_connectors_job_runs_only_on_a_connectors_tag(mutated)


def test_the_connectors_publish_is_gated_and_scoped(release: dict[str, Any] | None = None) -> None:
    """The upload waits on the CI gate, the build and the install check; it
    runs in the package's own environment, which the PyPI trusted publisher
    names; and it alone holds the OIDC scope."""
    release = release if release is not None else _load_connectors_release()
    publish = release["jobs"]["publish-to-pypi"]
    assert {CONNECTORS_GATE_JOB, CONNECTORS_BUILD_JOB, CONNECTORS_INSTALL_JOB} <= set(
        publish["needs"]
    ), f"the publish needs {publish['needs']}"
    assert publish["environment"]["name"] == CONNECTORS_ENVIRONMENT, (
        f"the publish runs in environment {publish['environment']['name']!r}"
    )
    assert publish.get("permissions") == {"id-token": "write"}, "publish must hold id-token"
    assert "id-token" not in release.get("permissions", {}), (
        "id-token is granted at workflow level, so every job can mint a PyPI token"
    )
    holders = [n for n, j in release["jobs"].items() if "id-token" in j.get("permissions", {})]
    assert holders == ["publish-to-pypi"], f"id-token held by {holders}"
    test_every_publish_step_skips_existing_files({"jobs": {PUBLISH_JOB: publish}})


def test_the_connectors_publish__mutation_workflow_level_id_token() -> None:
    mutated = _load_connectors_release()
    mutated["permissions"]["id-token"] = "write"
    with pytest.raises(AssertionError, match="workflow level"):
        test_the_connectors_publish_is_gated_and_scoped(mutated)


def test_the_connectors_publish__mutation_ungated() -> None:
    mutated = _load_connectors_release()
    mutated["jobs"]["publish-to-pypi"]["needs"] = [CONNECTORS_BUILD_JOB]
    with pytest.raises(AssertionError, match="the publish needs"):
        test_the_connectors_publish_is_gated_and_scoped(mutated)


def test_the_connectors_publish__mutation_framework_environment() -> None:
    mutated = _load_connectors_release()
    mutated["jobs"]["publish-to-pypi"]["environment"]["name"] = "pypi"
    with pytest.raises(AssertionError, match="runs in environment 'pypi'"):
        test_the_connectors_publish_is_gated_and_scoped(mutated)


def test_the_connectors_gate_requires_the_connectors_lanes(
    workflow: dict[str, Any], release: dict[str, Any] | None = None
) -> None:
    """The four always-on lanes, plus every compatibility lane ci.yml has."""
    release = release if release is not None else _load_connectors_release()
    lanes = _gate_lanes(release, CONNECTORS_GATE_JOB)
    expected = CONNECTORS_LANES | (CONNECTORS_COMPAT_LANES & set(_jobs(workflow)))
    assert set(lanes) == expected, f"the connectors gate requires {sorted(lanes)}"
    _assert_lane_prefixes_match(workflow, lanes, CONNECTORS_RELEASE_YML.name)


def test_the_connectors_gate__mutation_a_compat_lane_lands_ungated() -> None:
    """ci.yml carries a compatibility lane the connectors gate does not list."""
    release = _load_connectors_release()
    step = _gate_step(release, CONNECTORS_GATE_JOB)
    step["env"][GREEN_LANES_ENV] = "\n".join(
        line
        for line in step["env"][GREEN_LANES_ENV].splitlines()
        if not line.strip().startswith("connectors-compat=")
    )
    with pytest.raises(AssertionError, match="the connectors gate requires"):
        test_the_connectors_gate_requires_the_connectors_lanes(_load_workflow(), release)


def test_the_connectors_gate__mutation_renames_a_lane() -> None:
    workflow = _load_workflow()
    _jobs(workflow)["connectors-standalone"]["name"] = "Standalone connectors suite"
    with pytest.raises(AssertionError, match="release-connectors.yml looks for"):
        test_the_connectors_gate_requires_the_connectors_lanes(workflow)


def test_the_connectors_gate_blocks_the_build() -> None:
    release = _load_connectors_release()
    assert release["jobs"][CONNECTORS_BUILD_JOB]["needs"] == [CONNECTORS_GATE_JOB]
    assert release["jobs"][CONNECTORS_GATE_JOB]["permissions"]["actions"] == "read"


def test_both_release_workflows_run_the_shared_lane_gate() -> None:
    for release, job in (
        (_load_release(), RELEASE_GATE_JOB),
        (_load_connectors_release(), CONNECTORS_GATE_JOB),
    ):
        step = _gate_step(release, job)
        assert "python3 - <<" not in step["run"], f"{job} still carries an inline gate"


def _run_version_check(tmp_path: Path, version: str, files: list[str]) -> int:
    step = next(
        s
        for s in _load_connectors_release()["jobs"][CONNECTORS_BUILD_JOB]["steps"]
        if s.get("name") == CONNECTORS_VERSION_STEP
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in files:
        (dist / name).write_text("")
    env = {**os.environ, "VERSION": version}
    return subprocess.run(
        ["bash", "-c", step["run"]], cwd=tmp_path, env=env, capture_output=True, text=True
    ).returncode


@pytest.mark.parametrize(
    ("files", "ok"),
    [
        (
            ["osprey_connectors-2026.10.0-py3-none-any.whl", "osprey_connectors-2026.10.0.tar.gz"],
            True,
        ),
        (
            [
                "osprey_connectors-2026.9.0b3.post4-py3-none-any.whl",
                "osprey_connectors-2026.9.0b3.post4.tar.gz",
            ],
            False,
        ),
        (["osprey_connectors-2026.10.0-py3-none-any.whl"], False),  # no sdist
        (
            [
                "osprey_connectors-2026.10.0-py3-none-any.whl",
                "osprey_connectors-2026.10.0.tar.gz",
                "osprey_framework-2026.10.0-py3-none-any.whl",
            ],
            False,
        ),
    ],
)
def test_the_connectors_version_check_refuses_anything_but_the_tag(
    tmp_path: Path, files: list[str], ok: bool
) -> None:
    assert (_run_version_check(tmp_path, "2026.10.0", files) == 0) is ok


def test_the_connectors_build_builds_the_package_alone() -> None:
    steps = _load_connectors_release()["jobs"][CONNECTORS_BUILD_JOB]["steps"]
    runs = [s.get("run", "") for s in steps]
    assert "uv build --package osprey-connectors" in runs
    assert not any("--all-packages" in r for r in runs)
    assert any("${TAG_NAME#osprey-connectors-v}" in r for r in runs), (
        "the version is not derived from the tag minus its prefix"
    )


def _lean_walk_invocations(steps: list[dict[str, Any]]) -> list[str]:
    return [s["run"] for s in steps if LEAN_WALK_SCRIPT in s.get("run", "")]


def test_both_lean_wheel_checks_run_the_shared_walk(
    package_steps: tuple[tuple[str, ...], ...] | None = None,
) -> None:
    """The wheel a tag publishes passes the check every PR runs, isolated from the
    checkout. Every PR runs it through the package check, whose steps are
    declared once, in ``PACKAGE_STEPS`` of scripts/check.py."""
    steps = framework_checks.PACKAGE_STEPS if package_steps is None else package_steps
    walks = [step for step in steps if LEAN_WALK_SCRIPT in step]
    assert walks == [("{scratch}/lean/bin/python", "-I", LEAN_WALK_SCRIPT)], (
        f"the package check must run {LEAN_WALK_SCRIPT} once, in isolated mode from the "
        f"bare venv; it runs {walks}"
    )
    assert (
        "uv",
        "pip",
        "install",
        "--python",
        "{scratch}/lean/bin/python",
        "dist/osprey_connectors-*.whl",
    ) in steps
    runs = _lean_walk_invocations(
        _load_connectors_release()["jobs"][CONNECTORS_INSTALL_JOB]["steps"]
    )
    where = "release-connectors.yml verify-install"
    assert len(runs) == 1, f"{where} does not run {LEAN_WALK_SCRIPT}"
    assert f"/tmp/lean-check/bin/python -I {LEAN_WALK_SCRIPT}" in runs[0], (
        f"{where} must run the walk in isolated mode from the bare venv"
    )
    assert (
        "uv pip install --python /tmp/lean-check/bin/python dist/osprey_connectors-*.whl" in runs[0]
    )


def test_both_lean_wheel_checks_run_the_shared_walk__mutation_inline_probe() -> None:
    """The regression: the package check walks the wheel with a copy of its own."""
    steps = tuple(
        ("{scratch}/lean/bin/python", "-c", "import osprey_connectors")
        if LEAN_WALK_SCRIPT in step
        else step
        for step in framework_checks.PACKAGE_STEPS
    )
    with pytest.raises(AssertionError, match="must run scripts/ci/connectors_lean_import_walk"):
        test_both_lean_wheel_checks_run_the_shared_walk(steps)


def test_the_connectors_jobs_that_read_tags_fetch_them(
    release: dict[str, Any] | None = None,
) -> None:
    """The gate walks the tag's parents and the build describes the package's
    tags; a shallow checkout has neither and builds a wrong version silently."""
    release = release if release is not None else _load_connectors_release()
    for job in (CONNECTORS_GATE_JOB, CONNECTORS_BUILD_JOB):
        checkout = next(
            s
            for s in release["jobs"][job]["steps"]
            if str(s.get("uses", "")).startswith("actions/checkout@")
        )
        assert checkout.get("with", {}).get("fetch-depth") == 0, f"{job} checks out shallow"


def test_the_connectors_jobs_that_read_tags_fetch_them__mutation_shallow_build() -> None:
    mutated = _load_connectors_release()
    for step in mutated["jobs"][CONNECTORS_BUILD_JOB]["steps"]:
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            step.pop("with")
    with pytest.raises(AssertionError, match="build checks out shallow"):
        test_the_connectors_jobs_that_read_tags_fetch_them(mutated)


def test_both_release_workflows_extract_notes_with_the_shared_script() -> None:
    for release, changelog in (
        (_load_release(), "CHANGELOG.md"),
        (_load_connectors_release(), "packages/osprey-connectors/CHANGELOG.md"),
    ):
        runs = " ".join(s.get("run", "") for s in release["jobs"]["create-github-release"]["steps"])
        assert f'python3 scripts/ci/release_notes.py {changelog} "$VERSION"' in runs
        assert "sed -n" not in runs, "an inline sed extractor drops a last section's last line"


def test_the_connectors_release_notes_come_from_its_own_changelog() -> None:
    job = _load_connectors_release()["jobs"]["create-github-release"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "packages/osprey-connectors/CHANGELOG.md" in runs
    create = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("softprops/"))
    assert create["with"]["make_latest"] is False, "a connectors release must not become Latest"
    assert create["with"]["name"].startswith("osprey-connectors v")
    assert job["permissions"] == {"contents": "write"}


def _framework_build_commands(release: dict[str, Any]) -> list[str]:
    step = next(s for s in release["jobs"]["build"]["steps"] if s.get("name") == "Build packages")
    return [line.strip() for line in step["run"].splitlines() if line.strip()]


def test_the_framework_release_builds_its_lockstep_members_only(
    release: dict[str, Any] | None = None,
) -> None:
    """Every workspace member but osprey-connectors rides the v-tag; a member
    added later fails here until the release builds it."""
    release = release if release is not None else _load_release()
    members = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["uv"]["workspace"][
        "members"
    ]
    lockstep = sorted(Path(m).name for m in members if Path(m).name != "osprey-connectors")
    expected = ["uv build", *(f"uv build --package {m}" for m in lockstep)]
    assert _framework_build_commands(release) == expected, (
        f"the framework release builds {_framework_build_commands(release)}"
    )
    text = yaml.safe_dump(release)
    assert "--all-packages" not in text, "release.yml builds every package again"


def test_the_framework_release_builds__mutation_all_packages() -> None:
    mutated = _load_release()
    step = next(s for s in mutated["jobs"]["build"]["steps"] if s.get("name") == "Build packages")
    step["run"] = "uv build --all-packages\n"
    with pytest.raises(AssertionError, match="the framework release builds"):
        test_the_framework_release_builds_its_lockstep_members_only(mutated)


def test_the_framework_version_check_refuses_a_connectors_artifact(tmp_path: Path) -> None:
    step = next(
        s
        for s in _load_release()["jobs"]["build"]["steps"]
        if s.get("name") == CONNECTORS_VERSION_STEP
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    env = {**os.environ, "GITHUB_REF": "refs/tags/v2026.10.0"}

    def run() -> int:
        return subprocess.run(
            ["bash", "-c", step["run"]], cwd=tmp_path, env=env, capture_output=True, text=True
        ).returncode

    for name in (
        "osprey_framework-2026.10.0-py3-none-any.whl",
        "pyaml_cs_osprey-2026.10.0-py3-none-any.whl",
    ):
        (dist / name).write_text("")
    assert run() == 0
    (dist / "osprey_connectors-2026.10.0-py3-none-any.whl").write_text("")
    assert run() != 0


# ---------------------------------------------------------------------------
# Both release workflows: least privilege, tag guards, tags on main
# ---------------------------------------------------------------------------
#
# The jobs that install from PyPI and run what they installed must hold no
# write scope, so nothing at workflow level writes: only the PyPI upload mints
# an OIDC token and only the GitHub release writes contents. Every framework
# job is guarded on a tag ref, not a ref name a branch could carry. And both
# gates refuse a tag main cannot reach, which the lane gate alone lets through
# whenever the off-main commit's tree had a green run.

#: ``{job: write scopes}`` each release workflow may grant, and nothing else.
RELEASE_WRITE_SCOPES = {
    "publish-to-pypi": {"id-token"},
    "create-github-release": {"contents"},
}
FRAMEWORK_TAG_GUARD = f"{_REPOSITORY_GUARD} && startsWith(github.ref, 'refs/tags/v')"
ON_MAIN_STEP = "Require the tagged commit on main"


def _both_releases() -> list[tuple[str, dict[str, Any], str]]:
    """``(file name, workflow, gate job)`` for each release workflow."""
    return [
        (RELEASE_YML.name, _load_release(), RELEASE_GATE_JOB),
        (CONNECTORS_RELEASE_YML.name, _load_connectors_release(), CONNECTORS_GATE_JOB),
    ]


def _write_scopes(release: dict[str, Any]) -> dict[str, set[str]]:
    return {
        name: scopes
        for name, job in release["jobs"].items()
        if (scopes := {s for s, v in (job.get("permissions") or {}).items() if v == "write"})
    }


def _assert_least_privilege(name: str, release: dict[str, Any]) -> None:
    assert release.get("permissions") == {"contents": "read"}, (
        f"{name} grants {release.get('permissions')} at workflow level"
    )
    assert _write_scopes(release) == RELEASE_WRITE_SCOPES, (
        f"{name} write scopes: {_write_scopes(release)}"
    )


@pytest.mark.parametrize("index", [0, 1], ids=["release.yml", "release-connectors.yml"])
def test_each_release_workflow_writes_only_where_it_must(index: int) -> None:
    name, release, _ = _both_releases()[index]
    _assert_least_privilege(name, release)


def test_release_least_privilege__mutation_workflow_level_write() -> None:
    """The shape release.yml had: both scopes for every job."""
    release = _load_release()
    release["permissions"] = {"contents": "write", "id-token": "write"}
    with pytest.raises(AssertionError, match="at workflow level"):
        _assert_least_privilege("release.yml", release)


def test_release_least_privilege__mutation_a_job_escalates() -> None:
    release = _load_release()
    release["jobs"]["verify-pypi-resolvable"]["permissions"] = {"id-token": "write"}
    with pytest.raises(AssertionError, match="write scopes"):
        _assert_least_privilege("release.yml", release)


def test_the_reusable_install_docs_check_needs_no_write_scope() -> None:
    """A called workflow gets at most its caller's scopes; it must not need more."""
    called = yaml.safe_load((WORKFLOWS_DIR / "validate-install-docs.yml").read_text())
    for job in called["jobs"].values():
        assert not _write_scopes({"jobs": {"_": job}}), job


def test_every_framework_release_job_runs_only_on_a_version_tag(
    release: dict[str, Any] | None = None,
) -> None:
    """`github.ref_name` is also a branch's name; `refs/tags/v` is a tag only."""
    release = release if release is not None else _load_release()
    for name, job in release["jobs"].items():
        assert job.get("if") == FRAMEWORK_TAG_GUARD, (
            f"{name} is not guarded on the repository and the tag ref"
        )


def test_every_framework_release_job__mutation_ref_name_guard() -> None:
    mutated = _load_release()
    mutated["jobs"]["build"]["if"] = f"{_REPOSITORY_GUARD} && startsWith(github.ref_name, 'v')"
    with pytest.raises(AssertionError, match="build is not guarded"):
        test_every_framework_release_job_runs_only_on_a_version_tag(mutated)


def _on_main_step(release: dict[str, Any], job: str) -> dict[str, Any]:
    steps = release["jobs"][job]["steps"]
    found = [s for s in steps if s.get("name") == ON_MAIN_STEP]
    assert len(found) == 1, f"{job} has no {ON_MAIN_STEP!r} step"
    assert steps.index(found[0]) < steps.index(_gate_step(release, job)), (
        f"{job} checks main only after the lane gate"
    )
    return found[0]


def test_both_release_gates_require_the_tag_on_main() -> None:
    """One check, two copies held equal, ahead of the lane gate it covers for."""
    steps = [_on_main_step(release, job) for _, release, job in _both_releases()]
    assert steps[0]["run"] == steps[1]["run"], "the two on-main checks differ"
    assert steps[0]["env"] == steps[1]["env"]


def test_both_release_gates_require_the_tag_on_main__mutation_drops_one() -> None:
    release = _load_connectors_release()
    steps = release["jobs"][CONNECTORS_GATE_JOB]["steps"]
    steps[:] = [s for s in steps if s.get("name") != ON_MAIN_STEP]
    with pytest.raises(AssertionError, match="has no"):
        _on_main_step(release, CONNECTORS_GATE_JOB)


def _git_in(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_the_on_main_check_refuses_a_commit_main_cannot_reach(tmp_path: Path) -> None:
    """Run against a scratch origin: a commit on main passes, one on a side branch does not."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git_in(origin, "init", "-q", "-b", "main")
    _git_in(origin, "commit", "-q", "--allow-empty", "-m", "a")
    _git_in(origin, "commit", "-q", "--allow-empty", "-m", "b")
    on_main = _git_in(origin, "rev-parse", "HEAD")
    _git_in(origin, "checkout", "-q", "-b", "side", "HEAD~1")
    _git_in(origin, "commit", "-q", "--allow-empty", "-m", "c")
    off_main = _git_in(origin, "rev-parse", "HEAD")
    clone = tmp_path / "clone"
    _git_in(tmp_path, "clone", "-q", "--no-checkout", str(origin), str(clone))
    _git_in(clone, "fetch", "-q", "origin", off_main)

    script = _on_main_step(_load_release(), RELEASE_GATE_JOB)["run"]

    def run(sha: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "TAG_SHA": sha, "TAG_NAME": "v2026.10.0"}
        return subprocess.run(
            ["bash", "-c", script], cwd=clone, env=env, capture_output=True, text=True
        )

    assert run(on_main).returncode == 0
    refused = run(off_main)
    assert refused.returncode == 1
    assert "v2026.10.0" in refused.stdout and "is not on main" in refused.stdout


# ---------------------------------------------------------------------------
# The type check's stubs are declared where CI installs them: the dev extra
# ---------------------------------------------------------------------------
#
# An import in `src/` that ships no inline types needs its stub distribution in
# the `dev` extra. Without it the checker reports a per-module `import-untyped`
# error ("Library stubs not installed for ...") and gives every value from that
# library the type `Any`, and `scripts/mypy_gate.py` refuses to judge a run
# carrying one: it exits 2 rather than pass a report in which those values are
# `Any`. So an undeclared stub distribution turns the type-check lane red
# instead of quietly widening the modules that import it.


#: Stub distributions the `dev` extra carries, keyed by the import each one
#: serves. Silencing the diagnostic instead — disabling the `import-untyped`
#: code for these modules — buys a green gate over a report in which every
#: value from these libraries is `Any`, which is the opposite of checking the
#: modules that import them.
STUB_DISTRIBUTIONS = {
    "types-PyYAML": "yaml",
    "types-Markdown": "markdown",
    "types-aiofiles": "aiofiles",
}


def _dev_extra(pyproject: dict[str, Any]) -> list[str]:
    return pyproject["project"]["optional-dependencies"]["dev"]


def _declared_names(deps: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(dep).name) for dep in deps}


def test_the_type_check_declares_a_stub_for_every_stubless_import(
    pyproject: dict[str, Any],
) -> None:
    """Every import named here appears in `src/` and ships no types of its own,
    so an undeclared stub distribution costs the run its verdict: the checker
    reports `import-untyped` for that import and the gate refuses to judge
    what it produced."""
    declared = _declared_names(_dev_extra(pyproject))
    for distribution, module in STUB_DISTRIBUTIONS.items():
        assert canonicalize_name(distribution) in declared, (
            f"the `dev` extra must declare {distribution} — without it every "
            f"`import {module}` in src/ reports `import-untyped` and the type-check "
            f"gate refuses to judge the run"
        )


@pytest.mark.parametrize("distribution", sorted(STUB_DISTRIBUTIONS))
def test_the_type_check_declares_a_stub__mutation_drops_one(distribution: str) -> None:
    """Dropping any one of the three must fail: one absent stub is refused by
    the gate exactly as thoroughly as three are."""
    mutated = _load_pyproject()
    mutated["project"]["optional-dependencies"]["dev"] = [
        dep
        for dep in _dev_extra(mutated)
        if canonicalize_name(Requirement(dep).name) != canonicalize_name(distribution)
    ]
    with pytest.raises(AssertionError, match=distribution):
        test_the_type_check_declares_a_stub_for_every_stubless_import(mutated)


def test_the_type_checker_is_pinned_to_one_minor(pyproject: dict[str, Any]) -> None:
    """The gate holds the tree to zero errors under one checker, so the checker is
    named as precisely as that verdict. A floor alone lets a new minor report errors
    no diff introduced, and the gate would blame the change."""
    requirements = [
        Requirement(dep)
        for dep in _dev_extra(pyproject)
        if canonicalize_name(Requirement(dep).name) == canonicalize_name("mypy")
    ]
    assert requirements, "the `dev` extra must declare mypy"
    operators = {spec.operator for spec in requirements[0].specifier}
    assert ">=" in operators, "the mypy requirement must carry a lower bound"
    assert "<" in operators, (
        "the mypy requirement must carry an upper bound — an unpinned minor "
        "can red the gate on a change that introduced nothing"
    )


def test_the_type_checker_is_pinned__mutation_drops_the_ceiling() -> None:
    """A bare `mypy` is the state this guard exists to refuse."""
    mutated = _load_pyproject()
    mutated["project"]["optional-dependencies"]["dev"] = [
        "mypy" if canonicalize_name(Requirement(dep).name) == canonicalize_name("mypy") else dep
        for dep in _dev_extra(mutated)
    ]
    with pytest.raises(AssertionError):
        test_the_type_checker_is_pinned_to_one_minor(mutated)


# ---------------------------------------------------------------------------
# The type check completes: it does not follow an installed package the
# configured Python cannot parse
# ---------------------------------------------------------------------------
#
# A syntax error raised while mypy parses a followed dependency is fatal to the
# whole run: the checker stops with "errors prevented further checking" and
# examines no module. Nothing in this tree imports
# sphinx; mypy reaches it through bokeh's own property module, bokeh ships
# types, so `ignore_missing_imports` never applies. Sphinx's source uses syntax
# newer than the `python_version` this project targets, and that floor is the
# point of the check — so the package mypy cannot parse is the thing that gives,
# not the version it is parsed against.


def _mypy_overrides(pyproject: dict[str, Any]) -> list[dict[str, Any]]:
    return pyproject["tool"]["mypy"]["overrides"]


def test_the_type_check_completes_over_the_declared_targets(
    pyproject: dict[str, Any],
) -> None:
    """Pinned so the block cannot be dropped as mysterious: without it the run
    ends in "errors prevented further checking" and no module is examined."""
    skipped = [
        override for override in _mypy_overrides(pyproject) if "sphinx.*" in override["module"]
    ]
    assert skipped, "[tool.mypy] must carry an override for `sphinx.*`"
    for override in skipped:
        assert override.get("follow_imports") == "skip", (
            'the `sphinx.*` override must set follow_imports = "skip" — '
            "ignore_missing_imports does not apply to a package that ships types"
        )


def test_the_type_check_completes__mutation_drops_the_override() -> None:
    mutated = _load_pyproject()
    mutated["tool"]["mypy"]["overrides"] = [
        override for override in _mypy_overrides(mutated) if "sphinx.*" not in override["module"]
    ]
    with pytest.raises(AssertionError, match="sphinx"):
        test_the_type_check_completes_over_the_declared_targets(mutated)


def test_the_type_check_completes__mutation_weakens_the_override() -> None:
    """Swapping the skip for `ignore_missing_imports` must fail: bokeh's
    dependency ships types, so the import is never missing and the run still
    follows it into syntax it cannot parse."""
    mutated = _load_pyproject()
    for override in _mypy_overrides(mutated):
        if "sphinx.*" in override["module"]:
            del override["follow_imports"]
            override["ignore_missing_imports"] = True
    with pytest.raises(AssertionError, match="follow_imports"):
        test_the_type_check_completes_over_the_declared_targets(mutated)


# ---------------------------------------------------------------------------
# The type check resolves the same packages however it is invoked
# ---------------------------------------------------------------------------
#
# `osprey_connectors` lives in this repository under `packages/`, but it also
# sits in the environment as an installed wheel. Which of the two mypy resolves
# decides whether values crossing that boundary carry their real types or
# collapse to `Any` — and with `warn_return_any` on, the collapsed ones surface
# as `no-any-return` at every return site that touches them. Declaring the
# source roots is what makes a run naming a single file report the same thing a
# whole-tree run reports.


#: The roots `mypy_path` must carry, in the order it searches them: the
#: framework source and the sibling package's source, not its installed wheel.
MYPY_SOURCE_ROOTS = ("src", "packages/osprey-connectors/src")


def test_the_type_check_declares_its_source_roots(pyproject: dict[str, Any]) -> None:
    """`explicit_package_bases` is half the pair: without it mypy derives a
    module's name from its own directory rather than from these roots, and the
    roots buy nothing."""
    mypy_config = pyproject["tool"]["mypy"]
    assert mypy_config.get("explicit_package_bases") is True, (
        "[tool.mypy] must set explicit_package_bases = true so module names are "
        "derived from the declared roots"
    )
    declared = mypy_config.get("mypy_path", "").split(":")
    for root in MYPY_SOURCE_ROOTS:
        assert root in declared, (
            f"mypy_path must name {root!r} — without it a run naming a single file "
            "resolves less than a whole-tree run and reports errors it does not"
        )


def test_the_type_check_declares_its_source_roots__mutation_drops_the_bases() -> None:
    mutated = _load_pyproject()
    del mutated["tool"]["mypy"]["explicit_package_bases"]
    with pytest.raises(AssertionError, match="explicit_package_bases"):
        test_the_type_check_declares_its_source_roots(mutated)


@pytest.mark.parametrize("root", MYPY_SOURCE_ROOTS)
def test_the_type_check_declares_its_source_roots__mutation_drops_a_root(root: str) -> None:
    mutated = _load_pyproject()
    mutated["tool"]["mypy"]["mypy_path"] = ":".join(
        entry for entry in MYPY_SOURCE_ROOTS if entry != root
    )
    with pytest.raises(AssertionError, match=re.escape(root)):
        test_the_type_check_declares_its_source_roots(mutated)


# ---------------------------------------------------------------------------
# mypy target drift guard
# ---------------------------------------------------------------------------

MYPY_JOB = "lint"
MYPY_STEP = "Run mypy (type checking)"


def _load_mypy_gate() -> Any:
    """The gate script, loaded by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "_mypy_gate_probe", CI_YML.parents[2] / "scripts" / "mypy_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def test_the_type_check_step_runs_the_gate(workflow: dict[str, Any]) -> None:
    """The build runs the type check through the gate.

    The gate reads the declared trees and refuses a run without the declared stubs,
    which a bare ``mypy`` invocation does neither of.
    """
    run = _find_named_step(workflow, MYPY_JOB, MYPY_STEP)["run"]
    assert "scripts/mypy_gate.py" in run, (
        "the type-check step must run scripts/mypy_gate.py, not a bare mypy"
    )


def test_the_type_check_step_runs_the_gate__mutation_restores_a_bare_run() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, MYPY_JOB, MYPY_STEP)
    step["run"] = "uv run mypy src/ packages/osprey-connectors/src/ --no-error-summary"
    with pytest.raises(AssertionError):
        test_the_type_check_step_runs_the_gate(mutated)


def test_the_type_check_step_is_not_advisory(workflow: dict[str, Any]) -> None:
    """A check whose result cannot fail the job reports to no one.

    Both ways of discarding it are refused: ``continue-on-error`` on the step, and a
    shell suffix that swallows the command's own status.
    """
    step = _find_named_step(workflow, MYPY_JOB, MYPY_STEP)
    assert "continue-on-error" not in step, "the type-check step must be able to fail the lint job"
    run = step["run"]
    assert "|| true" not in run and "|| :" not in run, (
        "the type-check step must not swallow its own exit status"
    )


def test_the_type_check_step_is_not_advisory__mutation_restores_continue_on_error() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, MYPY_JOB, MYPY_STEP)["continue-on-error"] = True
    with pytest.raises(AssertionError):
        test_the_type_check_step_is_not_advisory(mutated)


def test_the_type_check_step_is_not_advisory__mutation_restores_the_suffix() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, MYPY_JOB, MYPY_STEP)
    step["run"] = step["run"].rstrip() + " || true"
    with pytest.raises(AssertionError):
        test_the_type_check_step_is_not_advisory(mutated)


def test_the_gate_reads_its_targets_from_the_declared_files(pyproject: dict[str, Any]) -> None:
    """CI and a bare local ``mypy`` check the same trees, by construction.

    The build names no trees at all any more, so there is nothing for CI and a local
    run to disagree about. What has to hold instead is that the gate keeps deriving
    them from ``[tool.mypy] files`` rather than growing a list of its own.
    """
    gate = _load_mypy_gate()
    declared = list(pyproject["tool"]["mypy"]["files"])
    assert gate.declared_targets(CI_YML.parents[2] / "pyproject.toml") == declared


def test_the_gate_reads_its_targets_from_the_declared_files__mutation_drops_a_tree(
    tmp_path: Path,
) -> None:
    pyproject = _load_pyproject()
    trimmed = [
        entry
        for entry in pyproject["tool"]["mypy"]["files"]
        if entry != "packages/osprey-connectors/src"
    ]
    mutated = tmp_path / "pyproject.toml"
    mutated.write_text(f"[tool.mypy]\nfiles = {json.dumps(trimmed)}\n")
    gate = _load_mypy_gate()
    assert gate.declared_targets(mutated) != pyproject["tool"]["mypy"]["files"]


# ---------------------------------------------------------------------------
# the local package check judges this tree's artifacts, by twine's own verdict
# ---------------------------------------------------------------------------
#
# ci.yml's `package` job gets both properties from the runner for nothing: a
# fresh checkout hands it an empty dist/, and a `run:` step fails its job on a
# non-zero exit. A local run of the check is handed neither. dist/ is
# gitignored, so it outlives a branch switch and holds the artifacts of other
# commits; and a shell pipeline reports the status of its LAST command, so a
# verdict read through a grep is the grep's verdict rather than the checker's.

PACKAGE_BUILD_STEP = ("uv", "build")
TWINE_CHECK_STEP = ("uvx", "twine", "check", "dist/*")
DIST_CLEAN = "clean dist"


def _package_events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What ``_check_package`` does, in order: the dist/ clean and each command."""
    events: list[str] = []

    class _Shutil:
        @staticmethod
        def rmtree(path: object, ignore_errors: bool = False) -> None:
            if Path(str(path)) == check_set.ROOT / "dist":
                events.append(DIST_CLEAN)

    def _run(argv: list[str], *_: object) -> int:
        events.append(" ".join(argv))
        return 0

    monkeypatch.setattr(framework_checks, "shutil", _Shutil)
    monkeypatch.setattr(check_set.kit, "run", _run)
    assert framework_checks._check_package(check_set.Context()) == 0
    return events


def _dist_clean_missing(events: list[str]) -> list[str]:
    """What the package check's clean is missing (empty = it precedes the build)."""
    build = " ".join(PACKAGE_BUILD_STEP)
    first = next((i for i, event in enumerate(events) if event == build), None)
    if first is None:
        return ["a package build"]
    if DIST_CLEAN not in events[:first]:
        return ["a dist/ clean before the build"]
    return []


def test_the_package_check_builds_into_an_empty_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check judges what this tree built, not what the directory happens to
    hold. `uv build` writes into dist/ without clearing it and dist/ is
    gitignored, so without the clean the twine check can be answered by a
    wheel from another commit — and answered green."""
    assert _dist_clean_missing(_package_events(monkeypatch)) == []


def test_the_package_check_builds_into_an_empty_dist__mutation_drops_the_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _package_events(monkeypatch)
    mutated = [event for event in events if event != DIST_CLEAN]
    assert mutated != events, "no dist/ clean in the package check; this mutation is stale"
    assert _dist_clean_missing(mutated) == ["a dist/ clean before the build"]


def test_the_package_check_builds_into_an_empty_dist__mutation_moves_it_after_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order is the whole rule: a clean that runs after the build deletes the
    artifacts the next step was going to check."""
    events = _package_events(monkeypatch)
    events.insert(2, events.pop(events.index(DIST_CLEAN)))
    assert _dist_clean_missing(events) == ["a dist/ clean before the build"]


def _twine_verdict_missing(steps: tuple[tuple[str, ...], ...]) -> list[str]:
    """What the twine check is missing (empty = its own status decides)."""
    if TWINE_CHECK_STEP not in steps:
        return ["a twine check"]
    return [
        f"twine's own status on {step!r}"
        for step in steps
        if any("twine" in part for part in step) and step != TWINE_CHECK_STEP
    ]


def test_the_package_check_reports_twines_own_verdict() -> None:
    """The checker decides, and says why. A pipeline reports its last command's
    status, so anything downstream of twine answers for it — and twine reports
    per artifact, so a downstream match on a passing line is satisfied by one
    good artifact beside a broken one."""
    assert _twine_verdict_missing(framework_checks.PACKAGE_STEPS) == []


def test_the_package_check_reports_twines_own_verdict__mutation_restores_the_grep() -> None:
    steps = framework_checks.PACKAGE_STEPS
    piped = ("sh", "-c", 'uvx twine check dist/* 2>&1 | grep -q "PASSED"')
    mutated = tuple(piped if step == TWINE_CHECK_STEP else step for step in steps)
    assert mutated != steps, "the twine step moved; this mutation is stale"
    assert _twine_verdict_missing(mutated) != []


# ---------------------------------------------------------------------------
# a broken documentation link fails the run, by the checker's own status
# ---------------------------------------------------------------------------
#
# A check can fail open in three independent ways: the status a pipeline
# reports is its LAST command's, so a verdict read through a grep or an
# `|| true` is not the checker's; a step marked `continue-on-error` reports a
# failure and passes the job; and a check that is in no tier a contributor runs
# cannot reach the verdict at all.

LINKCHECK_CHECK = "linkcheck"
LINKCHECK_COMMAND = "uv run make linkcheck"
LINKCHECK_STEP = "Check for broken links"
DOCS_BUILD_STEP = "Build documentation"
DOCS_UPLOAD_STEP = "Upload documentation artifact"
_SHELLS = frozenset({"sh", "bash", "zsh"})


def _linkcheck_verdict_missing(commands: list[list[str]]) -> list[str]:
    """What the full tier's link check is missing (empty = sphinx's own status decides).

    *commands* are the argument vectors the tier resolves to. The check
    command runs each one without a shell and fails the run on any non-zero
    exit, so the checker's status decides exactly when it is its own command.
    """
    checks = [argv for argv in commands if LINKCHECK_COMMAND in shlex.join(argv)]
    if not checks:
        return ["a link check"]
    return [
        f"the checker's own status on {shlex.join(argv)!r}"
        for argv in checks
        if argv[0] in _SHELLS or shlex.join(argv) != LINKCHECK_COMMAND
    ]


def _full_tier_commands() -> list[list[str]]:
    return check_set.resolve(["full"])


def test_the_full_tier_blocks_on_a_broken_documentation_link() -> None:
    """A link the docs publish and no reader can follow fails the run a push
    needs, by the checker's own status and in the checker's own words."""
    assert _linkcheck_verdict_missing(_full_tier_commands()) == []
    linkcheck = {check.id: check for check in check_set.CHECKS}[LINKCHECK_CHECK]
    assert linkcheck.workdir == "docs"


def test_the_full_tier_blocks_on_a_broken_documentation_link__mutation_restores_the_grep() -> None:
    piped = f'{LINKCHECK_COMMAND} 2>&1 | grep -q "build succeeded"'
    mutated = [
        ["sh", "-c", piped] if shlex.join(argv) == LINKCHECK_COMMAND else argv
        for argv in _full_tier_commands()
    ]
    assert _linkcheck_verdict_missing(mutated) == [
        f"the checker's own status on {shlex.join(['sh', '-c', piped])!r}"
    ]


def test_the_full_tier_blocks_on_a_broken_documentation_link__mutation_drops_the_check() -> None:
    """A link check no tier runs is advisory, whatever its verdict says."""
    commands = _full_tier_commands()
    mutated = [argv for argv in commands if shlex.join(argv) != LINKCHECK_COMMAND]
    assert mutated != commands, "no link check in the full tier; mutation is stale"
    assert _linkcheck_verdict_missing(mutated) == ["a link check"]


def _linkcheck_step_problems(wf: dict[str, Any]) -> list[str]:
    """What lets the docs job's link check pass over a broken link (empty = nothing)."""
    step = _find_named_step(wf, DOCS_JOB, LINKCHECK_STEP)
    problems = []
    if "continue-on-error" in step:
        problems.append("the link-check step carries continue-on-error")
    if str(step.get("run", "")).strip() != f"python3 {CHECK_SCRIPT} run {LINKCHECK_CHECK}":
        problems.append(f"the link-check step is not exactly a call of {CHECK_SCRIPT}")
    return problems


def test_the_link_check_can_fail_the_docs_job(raw_workflow: dict[str, Any]) -> None:
    """The docs job runs the link check as the check command declares it, and
    its failure fails the job."""
    assert _linkcheck_step_problems(raw_workflow) == []


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"continue-on-error": True}, "the link-check step carries continue-on-error"),
        (
            {"run": f"python3 {CHECK_SCRIPT} run {LINKCHECK_CHECK} || true"},
            f"the link-check step is not exactly a call of {CHECK_SCRIPT}",
        ),
    ],
)
def test_the_link_check_can_fail_the_docs_job__mutation(
    change: dict[str, Any], problem: str
) -> None:
    mutated = _load_raw_workflow()
    _find_named_step(mutated, DOCS_JOB, LINKCHECK_STEP).update(change)
    assert _linkcheck_step_problems(mutated) == [problem]


def test_docs_preview_uploads_before_the_link_check(workflow: dict[str, Any]) -> None:
    """A broken link must not cost the reader the preview that shows it: the
    built pages are uploaded after the build and before the link check, so a
    red link check still fails the job and the preview is still published."""
    names = _step_names(workflow, DOCS_JOB)
    for step in (DOCS_BUILD_STEP, DOCS_UPLOAD_STEP, LINKCHECK_STEP):
        assert step in names, f"no {step!r} step in the {DOCS_JOB} job; got {names}"
    assert (
        names.index(DOCS_BUILD_STEP) < names.index(DOCS_UPLOAD_STEP) < names.index(LINKCHECK_STEP)
    ), "the documentation preview must upload after the build and BEFORE the link check"


def test_docs_preview_uploads_before_the_link_check__mutation_moves_it_last() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[DOCS_JOB]["steps"]
    upload = next(s for s in steps if s.get("name") == DOCS_UPLOAD_STEP)
    steps.remove(upload)
    steps.append(upload)
    with pytest.raises(AssertionError, match="BEFORE the link check"):
        test_docs_preview_uploads_before_the_link_check(mutated)


# ---------------------------------------------------------------------------
# the secret scan runs the gitleaks the pre-commit hook pins
# ---------------------------------------------------------------------------

PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
GITLEAKS_HOOK_REPO = "https://github.com/gitleaks/gitleaks"
#: The lint job's two steps for the secret scan: the binary, then the call.
GITLEAKS_INSTALL_STEP = "Install gitleaks"
GITLEAKS_SCAN_STEP = "Run gitleaks (secret scan)"


def _hook_gitleaks_rev(config: dict[str, Any]) -> str:
    """The ``rev`` of the gitleaks hook repository in a parsed pre-commit config."""
    revs = [repo["rev"] for repo in config["repos"] if repo.get("repo") == GITLEAKS_HOOK_REPO]
    assert len(revs) == 1, f"expected one {GITLEAKS_HOOK_REPO} entry; got {revs}"
    return str(revs[0])


def _hook_and_ci_gitleaks_disagree(config: dict[str, Any], wf: dict[str, Any]) -> list[str]:
    step = _find_named_step(wf, LINT_JOB, GITLEAKS_INSTALL_STEP)
    hook, installed = _hook_gitleaks_rev(config), f"v{step['env']['GITLEAKS_VERSION']}"
    return [] if hook == installed else [f"the hook runs gitleaks {hook}; CI installs {installed}"]


def test_ci_installs_the_gitleaks_the_hook_pins(workflow: dict[str, Any]) -> None:
    """The secret scan CI runs and the pre-commit hook's scan use one gitleaks.
    Two versions carry different rules, so a commit the hook passed can fail
    the scan. The hook's ``rev`` is the repository's statement of the version:
    CI installs that release, and raising gitleaks moves both together."""
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    assert _hook_and_ci_gitleaks_disagree(config, workflow) == []


def test_ci_installs_the_gitleaks_the_hook_pins__mutation_the_hook_moves_ahead() -> None:
    wf = _load_workflow()
    installed = _find_named_step(wf, LINT_JOB, GITLEAKS_INSTALL_STEP)["env"]["GITLEAKS_VERSION"]
    config = {"repos": [{"repo": GITLEAKS_HOOK_REPO, "rev": "v99.0.0"}, {"repo": "x", "rev": "v1"}]}
    assert _hook_and_ci_gitleaks_disagree(config, wf) == [
        f"the hook runs gitleaks v99.0.0; CI installs v{installed}"
    ]
    config["repos"][0]["rev"] = f"v{installed}"
    assert _hook_and_ci_gitleaks_disagree(config, wf) == []


# ---------------------------------------------------------------------------
# CLI tool inventory check runs on every push
# ---------------------------------------------------------------------------


def test_cli_tool_inventory_check_runs_on_every_push(workflow: dict[str, Any]) -> None:
    """The recorded inventories are re-proved against the real builds on every
    run; a job-level ``if:`` would let a stale inventory pass unseen."""
    step = _find_named_step(workflow, INVENTORY_JOB, INVENTORY_STEP)
    assert "scripts/cli_tool_inventory.py --check" in step.get("run", ""), (
        f"'{INVENTORY_STEP}' must run the inventory check"
    )
    assert "if" not in _jobs(workflow)[INVENTORY_JOB], f"'{INVENTORY_JOB}' must carry no `if:`"


def test_cli_tool_inventory_check_runs_on_every_push__mutation_drops_the_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[INVENTORY_JOB]["steps"]
    steps[:] = [s for s in steps if s.get("name") != INVENTORY_STEP]
    with pytest.raises(AssertionError):
        test_cli_tool_inventory_check_runs_on_every_push(mutated)


def test_cli_tool_inventory_check_runs_on_every_push__mutation_writes_instead() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, INVENTORY_JOB, INVENTORY_STEP)
    step["run"] = step["run"].replace("--check", "--write")
    with pytest.raises(AssertionError):
        test_cli_tool_inventory_check_runs_on_every_push(mutated)


# ---------------------------------------------------------------------------
# Web-terminal cleanup steps address containers by the compose project name
# ---------------------------------------------------------------------------

#: Every belt-and-suspenders cleanup step that removes web-terminal containers,
#: with the compose project names its lane deploys. A web-terminal container is
#: ``<project>-web-<user>`` / ``<project>-nginx`` / ``<project>-auth``, so an
#: exact-named removal spelled any other way removes nothing the lane created.
WEB_TERMINAL_CLEANUP_STEPS: dict[tuple[str, str], tuple[str, ...]] = {
    (
        "multi-user-deploy-lifecycle-e2e",
        "Clean up any stranded lifecycle-e2e resources",
    ): ("osprey-e2e-mus-p3",),
    (
        "multi-user-deploy-lifecycle-e2e-podman",
        "Clean up any stranded lifecycle-e2e resources",
    ): ("osprey-e2e-mus-p3",),
    (
        "qmd-sidecar-e2e",
        "Clean up any stranded shared-bundle e2e resources",
    ): ("osprey-e2e-mus-p3",),
    (
        "auth-perimeter-e2e",
        "Clean up any stranded auth-perimeter resources",
    ): ("osprey-e2e-auth-perimeter",),
    (
        "terminal-auth-multiuser-e2e",
        "Clean up any stranded terminal-auth-multiuser resources",
    ): ("osprey-e2e-token-multiuser", "osprey-e2e-open-multiuser"),
    (
        "full-chain-auth-e2e",
        "Clean up any stranded full-chain-auth resources",
    ): ("osprey-e2e-full-chain-auth",),
    (
        "jupyter-panel-e2e",
        "Clean up any stranded notebook-panel resources",
    ): ("osprey-e2e-jnb",),
}

#: The lanes whose deploy builds the auth sidecar, which is ``<project>-auth``
#: running ``<project>-auth:local``.
AUTH_SIDECAR_CLEANUP_STEPS = (
    ("auth-perimeter-e2e", "Clean up any stranded auth-perimeter resources"),
    ("full-chain-auth-e2e", "Clean up any stranded full-chain-auth resources"),
)

_CONTAINER_REMOVAL = re.compile(r'\b(?:docker|podman) rm -f "?([^"\s]+)"?')


def _cleanup_step_run(wf: dict[str, Any], job: str, step_name: str) -> str:
    return _find_named_step(wf, job, step_name).get("run", "")


def test_web_terminal_cleanup_removes_project_named_containers(
    workflow: dict[str, Any],
) -> None:
    """Each cleanup step removes its lane's per-user and nginx containers by the
    compose project name the same step tears down."""
    for (job, step_name), projects in WEB_TERMINAL_CLEANUP_STEPS.items():
        run = _cleanup_step_run(workflow, job, step_name)
        removed = _CONTAINER_REMOVAL.findall(run)
        for project in projects:
            assert f"compose -p {project} down" in run, (
                f"'{job}' / '{step_name}' does not tear down compose project {project}"
            )
            assert any(name.startswith(f"{project}-web-") for name in removed), (
                f"'{job}' / '{step_name}' removes no {project}-web-<user> container"
            )
            assert f"{project}-nginx" in removed, (
                f"'{job}' / '{step_name}' does not remove {project}-nginx"
            )


def test_web_terminal_cleanup_removes_only_project_named_containers(
    workflow: dict[str, Any],
) -> None:
    """No exact-named removal in these steps is spelled off anything but one of
    the step's own compose project names."""
    for (job, step_name), projects in WEB_TERMINAL_CLEANUP_STEPS.items():
        run = _cleanup_step_run(workflow, job, step_name)
        removed = _CONTAINER_REMOVAL.findall(run)
        assert removed, f"'{job}' / '{step_name}' removes no container"
        strays = [
            name
            for name in removed
            if not any(name.startswith(f"{project}-") for project in projects)
        ]
        assert not strays, (
            f"'{job}' / '{step_name}' removes containers not named by its project: {strays}"
        )


def test_auth_sidecar_cleanup_removes_project_named_sidecar(workflow: dict[str, Any]) -> None:
    """The auth lanes remove ``<project>-auth`` and its ``<project>-auth:local``
    image."""
    for job, step_name in AUTH_SIDECAR_CLEANUP_STEPS:
        (project,) = WEB_TERMINAL_CLEANUP_STEPS[(job, step_name)]
        run = _cleanup_step_run(workflow, job, step_name)
        assert f"{project}-auth" in _CONTAINER_REMOVAL.findall(run), (
            f"'{job}' does not remove the {project}-auth container"
        )
        assert f"rmi -f {project}-auth:local" in run, (
            f"'{job}' does not remove the {project}-auth:local image"
        )


def test_web_terminal_cleanup_removes_project_named_containers__mutation_prefix_spelling() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(
        mutated, "auth-perimeter-e2e", "Clean up any stranded auth-perimeter resources"
    )
    original = step["run"]
    step["run"] = original.replace("osprey-e2e-auth-perimeter-nginx", "authe2e-nginx")
    assert step["run"] != original, "no project-named nginx removal — mutation is stale"
    with pytest.raises(AssertionError):
        test_web_terminal_cleanup_removes_project_named_containers(mutated)


def test_web_terminal_cleanup_removes_only_project_named_containers__mutation_stray() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(
        mutated,
        "multi-user-deploy-lifecycle-e2e",
        "Clean up any stranded lifecycle-e2e resources",
    )
    step["run"] += "\ndocker rm -f e2e-nginx || true\n"
    with pytest.raises(AssertionError):
        test_web_terminal_cleanup_removes_only_project_named_containers(mutated)


def test_auth_sidecar_cleanup_removes_project_named_sidecar__mutation_prefix_image() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(
        mutated, "full-chain-auth-e2e", "Clean up any stranded full-chain-auth resources"
    )
    original = step["run"]
    step["run"] = original.replace("osprey-e2e-full-chain-auth-auth:local", "fullchain-auth:local")
    assert step["run"] != original, "no project-named sidecar image — mutation is stale"
    with pytest.raises(AssertionError):
        test_auth_sidecar_cleanup_removes_project_named_sidecar(mutated)


# ---------------------------------------------------------------------------
# The compose projects each stack lane's cleanup step owes
# ---------------------------------------------------------------------------

#: The belt-and-suspenders cleanup step of a lane whose modules deploy a stack
#: under their own compose projects. A killed run otherwise leaves that stack
#: holding its ports and volumes on a self-hosted or reused runner.
STACK_CLEANUP_STEP = "Clean up any stranded stack resources"

#: The compose projects of the lanes whose modules deploy a stack, each from
#: its module's ``PROJECT_NAME``. The shared ``e2e-tests`` lane can run any of
#: them through its ``E2E_FILTER`` input, so its cleanup step names them all.
_STACK_LANE_PROJECTS: dict[str, tuple[str, ...]] = {
    "bluesky-queue-e2e": ("queue-e2e",),
    "bluesky-deploy-e2e": ("osprey-e2e-bluesky-deploy",),
    "bluesky-catalog-e2e": ("osprey-e2e-catalog",),
    "tiled-roundtrip-e2e": ("osprey-e2e-tiled",),
    "bluesky-web-deploy-e2e": ("panels-proj",),
    "scan-agentic-e2e": ("plan-agentic",),
    "orm-roundtrip-e2e": (
        "orm-roundtrip",
        "grid-scan-roundtrip",
        "bump-roundtrip",
        "preflight-probe",
    ),
    "bluesky-sandbox-escape-e2e": ("sandbox-escape",),
    "dispatch-deploy-e2e": ("osprey-e2e-dispatch",),
    "dispatch-overlay-e2e": ("osprey-e2e-overlay",),
    "single-user-narrowing-e2e": ("suproj",),
    "host-record-e2e": ("osprey-e2e-host-record",),
    "va-substrate-equivalence-e2e": ("osprey-e2e-va-substrate",),
    "va-bluesky-deploy-e2e": ("lane-e2e",),
    RECORD_E2E_JOB: ("osprey-e2e-agent-record",),
}

#: Where each lane's projects come from: the module constant they must equal.
_STACK_PROJECT_SOURCES: dict[str, str] = {
    "queue-e2e": "tests/e2e/test_bluesky_queue_e2e.py",
    "osprey-e2e-bluesky-deploy": "tests/e2e/test_bluesky_deploy.py",
    "osprey-e2e-catalog": "tests/e2e/test_bluesky_catalog_e2e.py",
    "osprey-e2e-tiled": "tests/e2e/test_tiled_roundtrip.py",
    "panels-proj": "tests/e2e/test_bluesky_web_deploy.py",
    "plan-agentic": "tests/e2e/test_plan_stack_agentic.py",
    "orm-roundtrip": "tests/e2e/test_orm_roundtrip.py",
    "grid-scan-roundtrip": "tests/e2e/test_grid_scan_roundtrip.py",
    "bump-roundtrip": "tests/e2e/test_bump_roundtrip.py",
    "preflight-probe": "tests/e2e/test_preflight_probe.py",
    "sandbox-escape": "tests/e2e/test_bluesky_sandbox_escape_e2e.py",
    "osprey-e2e-dispatch": "tests/e2e/test_dispatch_deploy.py",
    "osprey-e2e-overlay": "tests/e2e/test_dispatch_overlay_visibility.py",
    "suproj": "tests/e2e/test_single_user_narrowing_e2e.py",
    "osprey-e2e-va-substrate": "tests/e2e/test_va_substrate_equivalence.py",
    "lane-e2e": "tests/va/e2e/test_bluesky_lanes.py",
    "osprey-e2e-agent-record": RECORD_E2E_FILE,
    "osprey-e2e-host-record": "tests/e2e/test_host_record_e2e.py",
}

#: Every such cleanup step, with the compose projects its lane deploys: the
#: stack lanes above, plus the lanes whose cleanup step carries its own name.
STACK_CLEANUP_STEPS: dict[tuple[str, str], tuple[str, ...]] = {
    (
        "multi-user-deploy-lifecycle-e2e",
        "Clean up any stranded lifecycle-e2e resources",
    ): ("osprey-e2e-mus-p3",),
    (
        "multi-user-deploy-lifecycle-e2e-podman",
        "Clean up any stranded lifecycle-e2e resources",
    ): ("osprey-e2e-mus-p3",),
    (
        "full-chain-auth-e2e",
        "Clean up any stranded full-chain-auth resources",
    ): ("osprey-e2e-full-chain-auth",),
    (
        "jupyter-panel-e2e",
        "Clean up any stranded notebook-panel resources",
    ): ("osprey-e2e-jnb",),
    **{(job, STACK_CLEANUP_STEP): projects for job, projects in _STACK_LANE_PROJECTS.items()},
    ("e2e-tests", STACK_CLEANUP_STEP): tuple(
        project for projects in _STACK_LANE_PROJECTS.values() for project in projects
    ),
}


def test_stack_lane_projects_match_their_modules() -> None:
    """A renamed ``PROJECT_NAME`` would leave the cleanup removing names no
    deploy uses any more."""
    stale = {
        project: source
        for project, source in _STACK_PROJECT_SOURCES.items()
        if _module_constant(REPO_ROOT / source, "PROJECT_NAME") != project
    }
    assert stale == {}, f"cleanup projects that differ from their module's PROJECT_NAME: {stale}"
    listed = {p for projects in _STACK_LANE_PROJECTS.values() for p in projects}
    assert listed == set(_STACK_PROJECT_SOURCES)


# ---------------------------------------------------------------------------
# Every lane that can bring up a stack removes the agent record by exact name
# ---------------------------------------------------------------------------

#: The step that tears down what ``deploy-e2e``'s ``osprey up`` started, and
#: the step that does the same for ``archiver-world-e2e``.
DEPLOY_STACK_CLEANUP_STEP = "Clean up any stranded hello-world resources"
ARCHIVER_CLEANUP_STEP = "Clean up any stranded archiver-world resources"
TWO_SHAPE_CLEANUP_STEP = "Clean up any stranded two-shape resources"


def _deploy_e2e_project(wf: dict[str, Any]) -> str:
    """The project ``deploy-e2e`` creates, read off its own ``osprey init``."""
    for step in _jobs(wf)[DEPLOY_JOB]["steps"]:
        found = re.search(r"\bosprey init (\S+)", str(step.get("run", "")))
        if found:
            return found.group(1)
    raise AssertionError(f"'{DEPLOY_JOB}' runs no `osprey init <project>`")


def _record_cleanup_steps(wf: dict[str, Any]) -> dict[tuple[str, str], tuple[str, ...]]:
    """Every belt-and-suspenders cleanup step of a lane that can bring up a
    stack, with the compose projects that lane deploys.

    The agent record renders into every deployment, so every lane that runs
    ``osprey up`` (directly or through its module) owes it, whichever other
    services the lane also deploys.
    """
    return {
        **STACK_CLEANUP_STEPS,
        **WEB_TERMINAL_CLEANUP_STEPS,
        (WRITEBACK_JOB, WRITEBACK_CLEANUP_STEP): tuple(
            f"{WRITEBACK_PROJECT_PREFIX}{case}" for case in sorted(_writeback_project_cases())
        ),
        (TWO_SHAPE_JOB, TWO_SHAPE_CLEANUP_STEP): (_two_shape_module_constant("PROJECT_NAME"),),
        (ARCHIVER_JOB, ARCHIVER_CLEANUP_STEP): (
            _module_constant(REPO_ROOT / ARCHIVER_TEST_FILE, "PROJECT_NAME"),
        ),
        (DEPLOY_JOB, DEPLOY_STACK_CLEANUP_STEP): (_deploy_e2e_project(wf),),
    }


#: Lanes that run a docker-backed module or carry a docker cleanup step but
#: never run ``osprey up``, so no agent record can exist there to strand.
RECORD_CLEANUP_EXEMPT: dict[str, str] = {
    "dockerfile-e2e": "runs the generated image on its own; no compose deployment",
    "execution-environment-parity-e2e": "builds and runs the project image only",
    "graph-reseed-e2e": "runs the graph store container only",
    "nextcloud-talk-bridge-e2e": "runs fixture containers and the bridge process only",
    "target-switch-agentic-e2e": "runs the VA and an agent turn; no osprey up deployment",
    "privilege-split-e2e": "starts hand-named containers and sweeps their name prefix",
}

_OSPREY_UP = re.compile(r"\bosprey (?:deploy )?up\b")
_COMPOSE_DOWN = re.compile(r"\b(?:docker|podman)[ -]compose\b[^\n]*\bdown\b")


def _expand_shell_loops(run: str) -> str:
    """*run* with each ``for <var> in <cases>; do … done`` loop unrolled, and
    each plain ``name="…"`` assignment in the unrolled body substituted, so a
    removal spelled ``"${project}-agent-record"`` reads as the names it removes.
    """
    loop = re.compile(r"for (\w+) in ([^;\n]+); do\n(.*?)\ndone", re.DOTALL)

    def _unroll(match: re.Match[str]) -> str:
        var, cases, body = match.group(1), match.group(2).split(), match.group(3)
        out = []
        for case in cases:
            text = body.replace(f"${{{var}}}", case).replace(f"${var}", case)
            for name, value in re.findall(r'^\s*(\w+)="([^"$]*)"\s*$', text, re.MULTILINE):
                text = text.replace(f"${{{name}}}", value).replace(f"${name}", value)
            out.append(text)
        return "\n".join(out)

    return loop.sub(_unroll, run)


def _record_cleanup_gaps(run: str, project: str) -> list[str]:
    """What *run* fails to remove of *project*'s agent record, its ClickHouse
    store and the store's data volume.

    The volume counts as removed when the step names it exactly or sweeps the
    project's own volume prefix (``--filter "name=^<project>_"``).
    """
    from osprey.deployment.compose_generator import resolve_project_name

    expanded = _expand_shell_loops(run)
    removed = set(_CONTAINER_REMOVAL.findall(expanded))
    name = resolve_project_name({"project_name": project})
    gaps = [
        container
        for container in (f"{name}-agent-record", f"{name}-agent-record-clickhouse")
        if container not in removed
    ]
    volume = f"{name}_agent_record_clickhouse_data"
    swept = f'name=^{name}_"' in expanded
    if not swept and not re.search(
        rf"volume rm (?:-f )?\"?{re.escape(volume)}\"?(?:\s|$)", expanded
    ):
        gaps.append(volume)
    return gaps


def _stack_capable_jobs(wf: dict[str, Any]) -> set[str]:
    """Every job that can bring up a stack: it runs a docker-backed e2e module,
    runs ``osprey up`` itself, or carries an ``if: always()`` docker or podman
    cleanup step."""
    jobs = {job for f in _dockerbuild_marked_e2e_files() for job in _jobs_running_e2e_file(wf, f)}
    for job, spec in _jobs(wf).items():
        for step in spec.get("steps") or []:
            run = str(step.get("run", ""))
            if _OSPREY_UP.search(run):
                jobs.add(job)
            if step.get("if") == "always()" and (
                _CONTAINER_REMOVAL.search(run)
                or _COMPOSE_DOWN.search(run)
                or re.search(r"\b(?:docker|podman) ps -aq\b", run)
            ):
                jobs.add(job)
    return jobs


def test_every_stack_lane_owes_the_record_cleanup(workflow: dict[str, Any]) -> None:
    """A lane that can bring up a stack and has no record cleanup leaves the
    record container holding the next lane's port and volume after a failed
    run. Each such lane is either in the record-cleanup table or exempt with
    a reason; an exemption for a lane outside the rule is stale."""
    covered = {job for job, _ in _record_cleanup_steps(workflow)}
    unowned = sorted(_stack_capable_jobs(workflow) - covered - set(RECORD_CLEANUP_EXEMPT))
    assert unowned == [], f"stack lanes with no agent-record cleanup: {unowned}"
    stale = sorted(
        job
        for job in RECORD_CLEANUP_EXEMPT
        if job in covered or job not in _stack_capable_jobs(workflow)
    )
    assert stale == [], f"record-cleanup exemptions that no longer apply: {stale}"


def test_stack_lanes_remove_the_agent_record_by_name(workflow: dict[str, Any]) -> None:
    """Each cleanup step removes ``<project>-agent-record``,
    ``<project>-agent-record-clickhouse`` and the store's data volume for every
    project its lane deploys."""
    gaps = {
        job: missing
        for (job, step_name), projects in _record_cleanup_steps(workflow).items()
        if (
            missing := [
                gap
                for project in projects
                for gap in _record_cleanup_gaps(
                    _cleanup_step_run(workflow, job, step_name), project
                )
            ]
        )
    }
    assert gaps == {}, f"cleanup steps that do not remove the agent record: {gaps}"


def test_record_cleanup_steps_run_always_after_the_failure_capture(
    workflow: dict[str, Any],
) -> None:
    """The record cleanup runs on a failed or cancelled run, and only after the
    failure capture has read the containers' logs."""
    misplaced = []
    for job, step_name in _record_cleanup_steps(workflow):
        steps = workflow["jobs"][job]["steps"]
        step = _find_named_step(workflow, job, step_name)
        captures = [
            i for i, s in enumerate(steps) if "capture-ci-diagnostics" in str(s.get("uses", ""))
        ]
        if step.get("if") != "always()" or any(i > steps.index(step) for i in captures):
            misplaced.append(job)
    assert misplaced == [], (
        f"record cleanup not always() or before the failure capture: {misplaced}"
    )


def test_stack_lanes_remove_the_agent_record_by_name__mutation_drops_the_cleanup_step() -> None:
    """A stack lane whose cleanup step is gone is named by the rule."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _jobs(mutated)[ARCHIVER_JOB]["steps"]
    before = len(steps)
    steps[:] = [s for s in steps if s.get("name") != ARCHIVER_CLEANUP_STEP]
    assert len(steps) == before - 1, "no archiver-world cleanup step — mutation is stale"
    with pytest.raises(AssertionError, match=ARCHIVER_JOB):
        test_stack_lanes_remove_the_agent_record_by_name(mutated)


def test_every_stack_lane_owes_the_record_cleanup__mutation_adds_an_unowned_lane() -> None:
    """A new lane that runs ``osprey up`` with no table entry is named."""
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)["stray-stack-e2e"] = {
        "steps": [{"name": "boot", "run": "uv run osprey up -d --dev"}]
    }
    with pytest.raises(AssertionError, match="stray-stack-e2e"):
        test_every_stack_lane_owes_the_record_cleanup(mutated)


def test_every_stack_lane_owes_the_record_cleanup__mutation_stale_exemption() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _jobs(mutated)["graph-reseed-e2e"]
    with pytest.raises(AssertionError, match="graph-reseed-e2e"):
        test_every_stack_lane_owes_the_record_cleanup(mutated)


@pytest.mark.parametrize(
    ("job", "step_name", "dropped"),
    [
        ("bluesky-queue-e2e", STACK_CLEANUP_STEP, "queue-e2e-agent-record"),
        ("e2e-tests", STACK_CLEANUP_STEP, "suproj-agent-record-clickhouse"),
        (
            "auth-perimeter-e2e",
            "Clean up any stranded auth-perimeter resources",
            "osprey-e2e-auth-perimeter-agent-record",
        ),
        (TWO_SHAPE_JOB, TWO_SHAPE_CLEANUP_STEP, "osprey-e2e-two-shape-agent-record-clickhouse"),
        (DEPLOY_JOB, DEPLOY_STACK_CLEANUP_STEP, "hello-world-test-agent-record"),
        (ARCHIVER_JOB, ARCHIVER_CLEANUP_STEP, "archiver-world-agent-record"),
    ],
)
def test_stack_lanes_remove_the_agent_record_by_name__mutation_drops_a_removal(
    job: str, step_name: str, dropped: str
) -> None:
    """Dropping one record removal from one lane is a gap naming that lane."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, job, step_name)
    before = step["run"]
    step["run"] = "\n".join(
        line for line in before.splitlines() if not line.rstrip().endswith(f"{dropped} || true")
    )
    assert step["run"] != before, f"no {dropped} removal in '{job}' — mutation is stale"
    with pytest.raises(
        AssertionError, match=rf"'{re.escape(job)}': \[[^\]]*'{re.escape(dropped)}'"
    ):
        test_stack_lanes_remove_the_agent_record_by_name(mutated)


def test_stack_lanes_remove_the_agent_record_by_name__mutation_drops_the_loop_removal() -> None:
    """The write-back lane removes its records inside its per-project loop."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, WRITEBACK_JOB, WRITEBACK_CLEANUP_STEP)
    before = step["run"]
    step["run"] = "\n".join(
        line for line in before.splitlines() if '"${project}-agent-record" ' not in line
    )
    assert step["run"] != before, "no looped record removal — mutation is stale"
    with pytest.raises(AssertionError, match="osprey-e2e-wb-render-agent-record'"):
        test_stack_lanes_remove_the_agent_record_by_name(mutated)


def test_stack_lanes_remove_the_agent_record_by_name__mutation_drops_the_record_volume() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, ARCHIVER_JOB, ARCHIVER_CLEANUP_STEP)
    before = step["run"]
    step["run"] = "\n".join(
        line for line in before.splitlines() if "agent_record_clickhouse_data" not in line
    )
    assert step["run"] != before, "no record volume removal — mutation is stale"
    with pytest.raises(AssertionError, match="archiver-world_agent_record_clickhouse_data"):
        test_stack_lanes_remove_the_agent_record_by_name(mutated)


def test_stack_lanes_remove_the_agent_record_by_name__mutation_drops_the_volume_sweep() -> None:
    """The notebook-panel lane removes its volumes by project prefix only."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(
        mutated, "jupyter-panel-e2e", "Clean up any stranded notebook-panel resources"
    )
    before = step["run"]
    step["run"] = before.replace('"name=^osprey-e2e-jnb_"', '"name=^osprey-e2e-jnb-web_"')
    assert step["run"] != before, "no project volume sweep — mutation is stale"
    with pytest.raises(AssertionError, match="osprey-e2e-jnb_agent_record_clickhouse_data"):
        test_stack_lanes_remove_the_agent_record_by_name(mutated)


def test_record_cleanup_steps_run_always_after_the_failure_capture__mutation_drops_always() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, DEPLOY_JOB, DEPLOY_STACK_CLEANUP_STEP)["if"]
    with pytest.raises(AssertionError, match=DEPLOY_JOB):
        test_record_cleanup_steps_run_always_after_the_failure_capture(mutated)


def test_expand_shell_loops_unrolls_cases_and_assignments() -> None:
    run = (
        "for case in a b; do\n"
        '  project="p-${case}"\n'
        '  docker rm -f "${project}-agent-record" || true\n'
        "done\n"
    )
    assert _CONTAINER_REMOVAL.findall(_expand_shell_loops(run)) == [
        "p-a-agent-record",
        "p-b-agent-record",
    ]


_YAML_KEY_LINE = re.compile(r"^\s*[A-Za-z_][\w-]*:(?:\s|$)")


def _last_script_line(run: str) -> str:
    """The last non-blank line of a run script, where a spliced key lands."""
    lines = [line for line in run.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def test_no_run_script_absorbs_a_step_key(workflow: dict[str, Any]) -> None:
    """A step key spliced below a `run: |` block becomes the script's last
    line: the YAML still parses, the key is lost, and bash exits 127 on it.
    Only the last line is read, so a heredoc's own ``try:`` or ``else:`` is
    never mistaken for one."""
    absorbed = [
        (job_name, step.get("name"), line)
        for job_name, job in workflow["jobs"].items()
        for step in job.get("steps", [])
        if _YAML_KEY_LINE.match(line := _last_script_line(str(step.get("run", ""))))
    ]
    assert absorbed == [], f"run scripts holding a bare `key:` line: {absorbed}"


def test_no_run_script_absorbs_a_step_key__mutation_splices_a_key() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DEPLOY_JOB, DEPLOY_STACK_CLEANUP_STEP)
    step["run"] += "engine: docker\n"
    with pytest.raises(AssertionError, match="engine: docker"):
        test_no_run_script_absorbs_a_step_key(mutated)


def test_no_run_script_absorbs_a_step_key__a_heredoc_block_is_not_a_key() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DEPLOY_JOB, DEPLOY_STACK_CLEANUP_STEP)
    step["run"] = "python - <<'PY'\ntry:\n    pass\nfinally:\n    pass\nPY\n" + step["run"]
    test_no_run_script_absorbs_a_step_key(mutated)


# ---------------------------------------------------------------------------
# the graph-reseed lane: its own job, secret-free, out of the shared lane, gated
# ---------------------------------------------------------------------------

GRAPH_RESEED_JOB = "graph-reseed-e2e"
GRAPH_RESEED_TEST_FILE = "tests/e2e/test_graph_reseed_stamp.py"
GRAPH_RESEED_STEP = "Run graph reseed stamp E2E"


def test_graph_reseed_job_runs_its_file_in_a_named_step(workflow: dict[str, Any]) -> None:
    step = _find_named_step(workflow, GRAPH_RESEED_JOB, GRAPH_RESEED_STEP)
    assert f"pytest {GRAPH_RESEED_TEST_FILE}" in step["run"]


def test_graph_reseed_job_runs_its_file_in_a_named_step__mutation_drops_job() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][GRAPH_RESEED_JOB]
    with pytest.raises((AssertionError, KeyError)):
        test_graph_reseed_job_runs_its_file_in_a_named_step(mutated)


def test_e2e_tests_ignores_the_graph_reseed_file(workflow: dict[str, Any]) -> None:
    step = _find_named_step(workflow, E2E_TESTS_JOB, "Run E2E tests")
    assert f"--ignore={GRAPH_RESEED_TEST_FILE}" in step["run"]


def test_e2e_tests_ignores_the_graph_reseed_file__mutation_drops_ignore() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, E2E_TESTS_JOB, "Run E2E tests")
    step["run"] = _drop_ignore_line(step["run"], GRAPH_RESEED_TEST_FILE)
    with pytest.raises(AssertionError):
        test_e2e_tests_ignores_the_graph_reseed_file(mutated)


# ---------------------------------------------------------------------------
# The pyAML control-system member: coverage, wheel and release
# ---------------------------------------------------------------------------
#
# `pyaml-cs-osprey` is the second workspace member, and every place the first
# member is wired by name is a place the second can silently be missing from:
# the coverage cell measures only the source trees it names, the package job
# walks only the wheels it builds, and the release gate offers only the sibling
# wheels it stages. Each is pinned here with a mutation that removes exactly
# that wiring.

PYAML_COVERAGE_FLAG = "--cov=packages/pyaml-cs-osprey/src/pyaml_cs_osprey"
PACKAGE_JOB = "package"
PYAML_WHEEL_GLOBS = (
    "dist/osprey_framework-*.whl",
    "dist/osprey_connectors-*.whl",
    "dist/pyaml_cs_osprey-*.whl",
)
DOCK_ENGINE_COVERAGE_FLAG = "--cov=packages/dock-engine/src/dock_engine"
#: What the framework wheel must carry for ``dock_engine``: the module itself
#: (bundled from packages/dock-engine, not a wheel of its own) and the package
#: data ``dock_engine._data`` reads through ``importlib.resources``.
DOCK_ENGINE_WHEEL_DATA = (
    "dock_engine/__init__.py",
    "dock_engine/cases/validate/",
    "dock_engine/manifest.json",
)
RELEASE_VERIFY_JOB = "verify-pypi-resolvable"
RELEASE_SIBLING_DIR = "/tmp/sibling"


def _coverage_cell_flags(wf: dict[str, Any]) -> list[str]:
    """The ``--cov=`` flags the coverage cell's ``COV_ARGS`` carries."""
    run_text = _find_named_step(wf, UNIT_TEST_JOB, "Run unit tests")["run"]
    found = re.findall(r'COV_ARGS="([^"]+)"', run_text)
    assert len(found) == 1, f"expected one non-empty COV_ARGS assignment; got {found}"
    return [flag for flag in found[0].split() if flag.startswith("--cov=")]


def test_the_coverage_cell_measures_the_pyaml_member(workflow: dict[str, Any]) -> None:
    """Coverage measures only the trees named on the command line, so a member
    left off it reports nothing -- not zero, nothing -- and the upload stays green."""
    flags = _coverage_cell_flags(workflow)
    assert PYAML_COVERAGE_FLAG in flags, (
        f"the coverage cell does not measure the pyAML member; COV_ARGS names {flags}"
    )
    assert "--cov=packages/osprey-connectors/src/osprey_connectors" in flags


def test_the_coverage_cell_measures_the_pyaml_member__mutation_drops_the_flag() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    assert PYAML_COVERAGE_FLAG in step["run"], "the flag moved; this mutation is stale"
    step["run"] = step["run"].replace(f" {PYAML_COVERAGE_FLAG}", "")
    with pytest.raises(AssertionError, match="does not measure the pyAML member"):
        test_the_coverage_cell_measures_the_pyaml_member(mutated)


def test_the_coverage_cell_measures_the_dock_engine_member(workflow: dict[str, Any]) -> None:
    """The dock-engine member's tests run in the unit lane, so its tree is on
    the coverage command line beside the other members'."""
    flags = _coverage_cell_flags(workflow)
    assert DOCK_ENGINE_COVERAGE_FLAG in flags, (
        f"the coverage cell does not measure the dock-engine member; COV_ARGS names {flags}"
    )


def test_the_coverage_cell_measures_the_dock_engine_member__mutation_drops_the_flag() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    assert DOCK_ENGINE_COVERAGE_FLAG in step["run"], "the flag moved; this mutation is stale"
    step["run"] = step["run"].replace(f" {DOCK_ENGINE_COVERAGE_FLAG}", "")
    with pytest.raises(AssertionError, match="does not measure the dock-engine member"):
        test_the_coverage_cell_measures_the_dock_engine_member(mutated)


PackageSteps = tuple[tuple[str, ...], ...]
PYAML_BUILD_STEP = ("uv", "build", "--package", "pyaml-cs-osprey")
LEAN_PYTHON = "{scratch}/lean/bin/python"
PYAML_PYTHON = "{scratch}/pyaml/bin/python"
PACKAGE_UPLOAD_STEP = "Upload package artifacts"
#: What the uploaded ``package`` artifact holds: the framework's own artifacts.
FRAMEWORK_ARTIFACTS = "dist/osprey_framework-*"


def _step_index(
    steps: PackageSteps, predicate: Callable[[tuple[str, ...]], bool], what: str
) -> int:
    for i, step in enumerate(steps):
        if predicate(step):
            return i
    raise AssertionError(f"the package check has no step that {what}")


def _installs_into(python: str) -> Callable[[tuple[str, ...]], bool]:
    return lambda step: step[:3] == ("uv", "pip", "install") and python in step


def test_the_package_job_import_walks_the_pyaml_wheel(
    workflow: dict[str, Any], steps: PackageSteps | None = None
) -> None:
    """The pyAML wheel is built and every module in it imported, in a venv that
    holds the framework and connectors wheels too -- it imports the framework
    by design -- while the lean check keeps the connectors wheel alone. The
    package check runs it after twine has judged the framework-only dist/, and
    the job uploads the framework's artifacts alone."""
    steps = framework_checks.PACKAGE_STEPS if steps is None else steps
    twine = _step_index(steps, lambda step: step == TWINE_CHECK_STEP, "runs twine")
    build = _step_index(steps, lambda step: step == PYAML_BUILD_STEP, "builds the pyAML wheel")
    lean = _step_index(steps, lambda step: step[0] == LEAN_PYTHON, "runs the lean probe")
    walk = _step_index(steps, lambda step: step[0] == PYAML_PYTHON, "import-walks the pyAML wheel")
    install = _step_index(steps, _installs_into(PYAML_PYTHON), "fills the pyAML venv")
    lean_install = _step_index(steps, _installs_into(LEAN_PYTHON), "fills the lean venv")
    upload = _find_named_step(workflow, PACKAGE_JOB, PACKAGE_UPLOAD_STEP)
    assert twine < build and walk > lean and upload["with"]["path"] == FRAMEWORK_ARTIFACTS, (
        "the pyAML wheel must be built after twine judges the framework-only dist/, the "
        "upload must carry the framework-only dist/, and its import-walk must not precede "
        "the lean check"
    )
    for glob in PYAML_WHEEL_GLOBS:
        assert glob in steps[install], f"the pyAML import-walk venv does not install {glob}"
    script = steps[walk][-1]
    assert "pkgutil.walk_packages(pyaml_cs_osprey.__path__" in script
    assert "onerror=_raise" in script, (
        "walk_packages swallows a package import error unless onerror raises"
    )
    lean_words = " ".join(steps[lean_install])
    assert "osprey_framework" not in lean_words and "pyaml_cs_osprey" not in lean_words, (
        "the lean check must install the connectors wheel alone; a framework in its venv "
        "satisfies every osprey.* import it exists to refuse"
    )


def _replace_step(steps: PackageSteps, index: int, new: tuple[str, ...]) -> PackageSteps:
    return (*steps[:index], new, *steps[index + 1 :])


def test_the_package_job_import_walks_the_pyaml_wheel__mutation_drops_the_step() -> None:
    steps = tuple(s for s in framework_checks.PACKAGE_STEPS if s[0] != PYAML_PYTHON)
    with pytest.raises(AssertionError, match="no step that"):
        test_the_package_job_import_walks_the_pyaml_wheel(_load_workflow(), steps)


def test_the_package_job_import_walks_the_pyaml_wheel__mutation_walks_without_framework() -> None:
    steps = framework_checks.PACKAGE_STEPS
    index = _step_index(steps, _installs_into(PYAML_PYTHON), "fills the pyAML venv")
    narrowed = tuple(word for word in steps[index] if word != "dist/osprey_framework-*.whl")
    with pytest.raises(AssertionError, match="does not install dist/osprey_framework"):
        test_the_package_job_import_walks_the_pyaml_wheel(
            _load_workflow(), _replace_step(steps, index, narrowed)
        )


def test_the_package_job_import_walks_the_pyaml_wheel__mutation_lean_venv_gets_framework() -> None:
    steps = framework_checks.PACKAGE_STEPS
    index = _step_index(steps, _installs_into(LEAN_PYTHON), "fills the lean venv")
    widened = (*steps[index], "dist/osprey_framework-*.whl")
    with pytest.raises(AssertionError, match="connectors wheel alone"):
        test_the_package_job_import_walks_the_pyaml_wheel(
            _load_workflow(), _replace_step(steps, index, widened)
        )


def test_the_package_job_import_walks_the_pyaml_wheel__mutation_built_before_twine() -> None:
    steps = [s for s in framework_checks.PACKAGE_STEPS if s != PYAML_BUILD_STEP]
    steps.insert(steps.index(TWINE_CHECK_STEP), PYAML_BUILD_STEP)
    with pytest.raises(AssertionError, match="framework-only dist/"):
        test_the_package_job_import_walks_the_pyaml_wheel(_load_workflow(), tuple(steps))


def test_the_package_job_import_walks_the_pyaml_wheel__mutation_uploads_all_of_dist() -> None:
    mutated = _load_workflow()
    _find_named_step(mutated, PACKAGE_JOB, PACKAGE_UPLOAD_STEP)["with"]["path"] = "dist/"
    with pytest.raises(AssertionError, match="framework-only dist/"):
        test_the_package_job_import_walks_the_pyaml_wheel(mutated)


def _checks_dock_engine(step: tuple[str, ...]) -> bool:
    return step[:2] == ("python3", "-c") and "dock_engine" in step[2]


def test_the_package_job_checks_the_framework_wheel_for_dock_engine(
    workflow: dict[str, Any], steps: PackageSteps | None = None
) -> None:
    """``dock_engine`` ships inside the framework wheel, not as a wheel of its own,
    so the package check reads the framework wheel's file list for the module and
    the package data ``dock_engine._data`` needs. No dock-engine wheel is built
    anywhere in the check or the job."""
    steps = framework_checks.PACKAGE_STEPS if steps is None else steps
    check = _step_index(steps, _checks_dock_engine, "checks the framework wheel for dock_engine")
    assert "dist/osprey_framework-*.whl" in steps[check], (
        "the check does not read the framework wheel"
    )
    for member in DOCK_ENGINE_WHEEL_DATA:
        assert member in steps[check][2], f"the dock-engine check does not look for {member}"
    job_text = " ".join(" ".join(step) for step in steps) + json.dumps(_jobs(workflow)[PACKAGE_JOB])
    assert "osprey-dock-engine" not in job_text and "osprey_dock_engine" not in job_text, (
        "the package job builds or installs a dock-engine wheel; the engine ships inside "
        "the framework wheel and has no distribution of its own"
    )


def test_the_package_job_checks_the_framework_wheel_for_dock_engine__mutation_drops_the_step() -> (
    None
):
    steps = tuple(s for s in framework_checks.PACKAGE_STEPS if not _checks_dock_engine(s))
    with pytest.raises(AssertionError, match="no step that"):
        test_the_package_job_checks_the_framework_wheel_for_dock_engine(_load_workflow(), steps)


@pytest.mark.parametrize("member", DOCK_ENGINE_WHEEL_DATA)
def test_the_package_job_checks_the_framework_wheel_for_dock_engine__mutation_skips_data(
    member: str,
) -> None:
    steps = framework_checks.PACKAGE_STEPS
    index = _step_index(steps, _checks_dock_engine, "checks the framework wheel")
    step = steps[index]
    assert member in step[2], f"the check no longer names {member}; mutation is stale"
    mutated = _replace_step(steps, index, (*step[:2], step[2].replace(member, ""), *step[3:]))
    with pytest.raises(AssertionError, match=f"does not look for {re.escape(member)}"):
        test_the_package_job_checks_the_framework_wheel_for_dock_engine(_load_workflow(), mutated)


def test_the_package_job_checks_the_framework_wheel_for_dock_engine__mutation_builds_a_wheel() -> (
    None
):
    steps = (*framework_checks.PACKAGE_STEPS, ("uv", "build", "--package", "osprey-dock-engine"))
    with pytest.raises(AssertionError, match="no distribution of its own"):
        test_the_package_job_checks_the_framework_wheel_for_dock_engine(_load_workflow(), steps)


def _release_staging_script(release: dict[str, Any]) -> str:
    """The sibling-staging lines of the release gate, up to the install itself."""
    steps = release["jobs"][RELEASE_VERIFY_JOB]["steps"]
    staging = [s for s in steps if RELEASE_SIBLING_DIR in str(s.get("run", ""))]
    assert len(staging) == 1, f"expected one step staging {RELEASE_SIBLING_DIR}; got {staging}"
    script = staging[0]["run"]
    assert "uv tool install" in script, "the staging step no longer installs the framework"
    return script.split("uv tool install", 1)[0]


def _staged_siblings(release: dict[str, Any], tmp_path: Path) -> list[str]:
    """Run the staging lines against a dist/ holding every artifact a tag builds,
    and return what they offer as find-links."""
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in (
        "osprey_framework-2026.9.1-py3-none-any.whl",
        "osprey_framework-2026.9.1.tar.gz",
        "osprey_connectors-2026.9.1-py3-none-any.whl",
        "osprey_connectors-2026.9.1.tar.gz",
        "pyaml_cs_osprey-2026.9.1-py3-none-any.whl",
        "pyaml_cs_osprey-2026.9.1.tar.gz",
    ):
        (dist / name).write_text("")
    sibling = tmp_path / "sibling"
    script = _release_staging_script(release).replace(RELEASE_SIBLING_DIR, str(sibling))
    subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path, check=True)
    return sorted(p.name for p in sibling.iterdir())


def _assert_release_stages_every_member_wheel(release: dict[str, Any], tmp_path: Path) -> None:
    # osprey-connectors is absent even from a dist/ that holds it: it releases
    # on its own tags, so the framework's floor on it must resolve from PyPI.
    assert _staged_siblings(release, tmp_path) == [
        "pyaml_cs_osprey-2026.9.1-py3-none-any.whl",
    ], "the release gate stages the wrong sibling set"


def _with_staging(release: dict[str, Any], staging: str) -> dict[str, Any]:
    """``release`` with the staging lines replaced and the install kept."""
    mutated = copy.deepcopy(release)
    for step in mutated["jobs"][RELEASE_VERIFY_JOB]["steps"]:
        if RELEASE_SIBLING_DIR in str(step.get("run", "")):
            tail = step["run"].split("uv tool install", 1)[1]
            step["run"] = f"mkdir -p {RELEASE_SIBLING_DIR}\n{staging}\nuv tool install{tail}"
    return mutated


def test_the_release_gate_stages_every_member_wheel_but_the_framework(tmp_path: Path) -> None:
    """The framework depends on every lockstep member, so a sibling left out of
    the find-links fails the gate on the tag that first ships it; the
    framework's own wheel, any sdist, or a connectors wheel staged there would
    let an unpublished pin resolve."""
    _assert_release_stages_every_member_wheel(yaml.safe_load(RELEASE_YML.read_text()), tmp_path)


def test_the_release_gate_stages_every_member_wheel__mutation_connectors_only(
    tmp_path: Path,
) -> None:
    mutated = _with_staging(
        yaml.safe_load(RELEASE_YML.read_text()),
        f"cp dist/osprey_connectors-*.whl {RELEASE_SIBLING_DIR}/",
    )
    with pytest.raises(AssertionError, match="wrong sibling set"):
        _assert_release_stages_every_member_wheel(mutated, tmp_path)


def test_the_release_gate_stages_every_member_wheel__mutation_offers_connectors(
    tmp_path: Path,
) -> None:
    """Staging by exclusion of the framework alone, as before the split."""
    mutated = _with_staging(
        yaml.safe_load(RELEASE_YML.read_text()),
        f'for wheel in dist/*.whl; do case "$(basename "$wheel")" in '
        f'osprey_framework-*) ;; *) cp "$wheel" {RELEASE_SIBLING_DIR}/ ;; esac; done',
    )
    with pytest.raises(AssertionError, match="wrong sibling set"):
        _assert_release_stages_every_member_wheel(mutated, tmp_path)


def test_the_release_gate_stages_every_member_wheel__mutation_whole_dist(tmp_path: Path) -> None:
    mutated = _with_staging(
        yaml.safe_load(RELEASE_YML.read_text()), f"cp dist/* {RELEASE_SIBLING_DIR}/"
    )
    with pytest.raises(AssertionError, match="wrong sibling set"):
        _assert_release_stages_every_member_wheel(mutated, tmp_path)


# ---------------------------------------------------------------------------
# The two-uid simulator log case runs in the unit lane's Linux cells
# ---------------------------------------------------------------------------
#
# The case that two uids sharing a log directory's group both append to one
# model log runs two `docker run --user` writers, so only a Linux runner with an
# engine can run it, and it skips everywhere else. A skip on the cells that must
# run it would be silent, so the lane, the shard and the case's own name are
# pinned here.

TWO_UID_LOG_FILE = "tests/deployment/test_var_simulator_provisioning.py"
TWO_UID_LOG_CASE = "test_two_uids_in_one_group_append_to_one_log_from_two_containers"
TWO_UID_LOG_OS = "ubuntu-latest"
TWO_UID_LOG_SHARD = "rest-a"


def test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell(
    workflow: dict[str, Any], unit: Any = None
) -> None:
    """The unit lane's ubuntu `rest-a` cells collect the case, deselect it by no
    marker, and leave `CI` to the runner, which is what turns a missing engine
    into a failure there."""
    lane = f"the '{UNIT_TEST_JOB}' job in .github/workflows/ci.yml"
    job = _jobs(workflow)[UNIT_TEST_JOB]
    matrix = job["strategy"]["matrix"]
    assert TWO_UID_LOG_OS in matrix["os"], (
        f"{lane} has no {TWO_UID_LOG_OS} cell for {TWO_UID_LOG_FILE}"
    )
    assert TWO_UID_LOG_SHARD in matrix["shard"], (
        f"{lane} has no '{TWO_UID_LOG_SHARD}' shard to collect {TWO_UID_LOG_FILE}"
    )
    assert _shard_reaches(TWO_UID_LOG_SHARD, TWO_UID_LOG_FILE, unit), (
        f"{lane}'s '{TWO_UID_LOG_SHARD}' shard does not collect {TWO_UID_LOG_FILE}"
    )
    exclude = matrix.get("exclude", "")
    exclude_text = (
        exclude if isinstance(exclude, str) else json.dumps(exclude, separators=(",", ":"))
    )
    assert '"os":"ubuntu-latest"' not in exclude_text, (
        f"{lane} excludes an {TWO_UID_LOG_OS} cell that runs {TWO_UID_LOG_FILE}"
    )
    step = _find_named_step(workflow, UNIT_TEST_JOB, "Run unit tests")
    # The ignores the pinned shard runs under: the line every shard shares and
    # the shard's own arguments, not the ignores that spell another shard.
    shard_line = f"{_unit_test_pytest_line(workflow)} {_unit_shards(unit)[TWO_UID_LOG_SHARD]}"
    for ignored in re.findall(r"--ignore=([^\s\"';]+)", shard_line):
        assert not (
            TWO_UID_LOG_FILE == ignored or TWO_UID_LOG_FILE.startswith(ignored.rstrip("/") + "/")
        ), f"{lane} ignores {ignored}, which holds {TWO_UID_LOG_FILE}"
    selections = re.findall(r'-m "[^"]*"', _unit_test_pytest_line(workflow))
    assert selections == [UNIT_LANE_DESELECT], (
        f"{lane} selects {selections} rather than {UNIT_LANE_DESELECT}, "
        f"which can deselect {TWO_UID_LOG_FILE}"
    )
    for where, env in (("job", job.get("env")), ("step", step.get("env"))):
        assert "CI" not in (env or {}), (
            f"{lane} sets CI in its {where} env, which decides whether {TWO_UID_LOG_FILE} "
            f"fails or skips without an engine"
        )


def test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell__mutation_ignores_the_file() -> (
    None
):
    mutated = _unit_lane_with(
        _load_workflow(), f" --ignore={TWO_UID_LOG_FILE} {UNIT_LANE_DESELECT}"
    )
    with pytest.raises(AssertionError, match="ignores tests/deployment/test_var_simulator"):
        test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell(mutated)


def test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell__mutation_ignores_tests_deployment() -> (
    None
):
    mutated = _unit_lane_with(_load_workflow(), f" --ignore=tests/deployment {UNIT_LANE_DESELECT}")
    with pytest.raises(AssertionError, match="ignores tests/deployment, which holds"):
        test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell(mutated)


def test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell__mutation_drops_the_ubuntu_cells() -> (
    None
):
    mutated = _load_workflow()
    matrix = _jobs(mutated)[UNIT_TEST_JOB]["strategy"]["matrix"]
    assert matrix["os"] != ["macos-latest"], "the unit lane is macOS-only; mutation is stale"
    matrix["os"] = ["macos-latest"]
    with pytest.raises(AssertionError, match=f"has no {TWO_UID_LOG_OS} cell"):
        test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell(mutated)


def test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell__mutation_deselects_by_another_marker() -> (
    None
):
    mutated = _unit_lane_with(_load_workflow(), f' {UNIT_LANE_DESELECT[:-1]} and not slow"')
    with pytest.raises(AssertionError, match="which can deselect"):
        test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell(mutated)


def test_the_two_uid_log_case_is_a_docker_group_case(source: str | None = None) -> None:
    """The case carries the docker xdist group, so the lane's scheduler keeps
    it with the other container-starting tests."""
    text = (REPO_ROOT / TWO_UID_LOG_FILE).read_text() if source is None else source
    cases = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == TWO_UID_LOG_CASE
    ]
    assert cases, f"{TWO_UID_LOG_FILE} has no case named {TWO_UID_LOG_CASE}"
    grouped = [
        dec
        for dec in cases[0].decorator_list
        if isinstance(dec, ast.Call)
        and ast.unparse(dec.func) == "pytest.mark.xdist_group"
        and [ast.literal_eval(arg) for arg in dec.args] == ["docker"]
    ]
    assert grouped, f"{TWO_UID_LOG_FILE}::{TWO_UID_LOG_CASE} is not in the docker xdist group"


def test_the_two_uid_log_case_is_a_docker_group_case__mutation_renames_the_case() -> None:
    source = (REPO_ROOT / TWO_UID_LOG_FILE).read_text()
    mutated = source.replace(f"def {TWO_UID_LOG_CASE}(", f"def {TWO_UID_LOG_CASE}_renamed(")
    assert mutated != source, f"{TWO_UID_LOG_CASE} not found; mutation is stale"
    with pytest.raises(AssertionError, match="has no case named"):
        test_the_two_uid_log_case_is_a_docker_group_case(mutated)


def test_the_two_uid_log_case_is_a_docker_group_case__mutation_drops_the_group() -> None:
    source = (REPO_ROOT / TWO_UID_LOG_FILE).read_text()
    mutated = source.replace('@pytest.mark.xdist_group("docker")\n', "", 1)
    assert mutated != source, "docker group decorator not found; mutation is stale"
    with pytest.raises(AssertionError, match="not in the docker xdist group"):
        test_the_two_uid_log_case_is_a_docker_group_case(mutated)


# ---------------------------------------------------------------------------
# The unit lane's shards cover tests/ exactly once
# ---------------------------------------------------------------------------
#
# The unit lane runs each matrix cell as several jobs, one per set of
# directories, named by `matrix.shard` in "Run unit tests" and defined by the
# check's ``UNIT_SELECTION.shards``. A directory two shards name runs twice and
# costs a slow runner its margin under the step cap; a directory no shard names
# runs nowhere with every check green. The last shard is spelled as ignores of
# every directory another shard names, so a new directory falls into it; what
# is pinned here is that the spellings still partition the tree.


def _unit_shards(unit: Any = None) -> dict[str, str]:
    """Each shard's own pytest selection, its paths and its own ignores, keyed
    by its name. The ignores every shard shares are left out: they narrow the
    lane, not the partition."""
    unit = unit or framework_checks.UNIT_SELECTION
    return {
        name: " ".join(["pytest", *paths, *(f"--ignore={path}" for path in ignores)])
        for name, (paths, ignores) in unit.shards.items()
    }


def _shard_reaches(shard: str, test_file: str, unit: Any = None) -> bool:
    return _invocation_reaches(_unit_shards(unit).get(shard, ""), test_file)


def _test_tree_entries(root: Path = TESTS_ROOT) -> list[str]:
    """The top level of ``tests/`` as the shards see it: every directory that
    holds a test module somewhere below it, and every test module directly in
    ``tests/``, as repo-relative posix paths, plus each package suite in
    :data:`PACKAGE_TEST_ROOTS`."""
    entries = {
        package for package in PACKAGE_TEST_ROOTS if any((root.parent / package).rglob("test_*.py"))
    }
    for pattern in ("test_*.py", "*_test.py"):
        for path in root.rglob(pattern):
            if "__pycache__" in path.parts:
                continue
            top = path.relative_to(root).parts[0]
            entries.add(f"tests/{top}")
    return sorted(entries)


def _shard_coverage_problems(wf: dict[str, Any], entries: list[str], unit: Any = None) -> list[str]:
    """Why the shards do not partition *entries* (empty = each entry in exactly
    one shard, and the declared shards are the matrix's shards).

    The partition is checked at the top level of ``tests/``, so it holds only
    while every shard path and ignore is a top-level entry: an ignore one level
    deeper takes its files out of the shard that names the directory while no
    other shard picks them up. A shard that reaches below the top level is
    refused outright rather than checked file by file."""
    shards = _unit_shards(unit)
    matrix = sorted(_jobs(wf)[UNIT_TEST_JOB]["strategy"]["matrix"]["shard"])
    problems = []
    if sorted(shards) != matrix:
        problems.append(f"declared shards {sorted(shards)} are not the matrix shards {matrix}")
    for name, command in shards.items():
        for token in command.split()[1:]:
            path = token.removeprefix("--ignore=").rstrip("/")
            if path.startswith("tests/") and path.count("/") > 1:
                problems.append(
                    f"shard '{name}' names {token}, below a top-level test directory; "
                    f"a shard is a set of whole tests/<dir> entries"
                )
    for entry in entries:
        # A probe file below a directory entry; a root module is its own probe.
        probe = entry if entry.endswith(".py") else f"{entry}/test_probe.py"
        holders = [name for name, command in shards.items() if _invocation_reaches(command, probe)]
        if len(holders) != 1:
            problems.append(f"{entry} is in {len(holders)} shards: {holders}")
    return problems


def test_the_unit_shards_cover_the_test_tree_exactly_once(
    workflow: dict[str, Any], unit: Any = None
) -> None:
    entries = _test_tree_entries()
    assert "tests/cli" in entries and len(entries) > 10, f"test tree scan broke: {entries}"
    assert set(PACKAGE_TEST_ROOTS) <= set(entries), f"package suite scan broke: {entries}"
    problems = _shard_coverage_problems(workflow, entries, unit)
    assert problems == [], "the unit lane's shards do not partition tests/:\n" + "\n".join(problems)


def _with_shard(shard: str, paths: tuple[str, ...], ignores: tuple[str, ...]) -> Any:
    """The declared selection with *shard* redefined."""
    unit = framework_checks.UNIT_SELECTION
    assert shard in unit.shards, f"no '{shard}' shard declared; mutation is stale"
    return replace(unit, shards={**unit.shards, shard: (paths, ignores)})


def _last_shard() -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """The shard spelled by exclusion: its name, paths and ignored directories."""
    for name, (paths, ignores) in framework_checks.UNIT_SELECTION.shards.items():
        if ignores:
            return name, paths, ignores
    raise AssertionError("no unit shard is spelled by exclusion")


def test_the_unit_shards_cover_the_test_tree__mutation_drops_an_ignore() -> None:
    """The last shard forgets one directory another shard names: it runs twice."""
    name, paths, ignores = _last_shard()
    dropped = ignores[-1]
    mutated = _with_shard(name, paths, ignores[:-1])
    with pytest.raises(AssertionError, match=rf"{re.escape(dropped)} is in 2 shards"):
        test_the_unit_shards_cover_the_test_tree_exactly_once(_load_workflow(), mutated)


def test_the_unit_shards_cover_the_test_tree__mutation_lists_the_last_shard() -> None:
    """The last shard is spelled as a list of directories instead of by
    exclusion: every directory nobody listed runs nowhere."""
    name, _, _ = _last_shard()
    mutated = _with_shard(name, ("tests/pty", "tests/va"), ())
    with pytest.raises(AssertionError, match=r"is in 0 shards"):
        test_the_unit_shards_cover_the_test_tree_exactly_once(_load_workflow(), mutated)


def test_the_unit_shards_cover_the_test_tree__mutation_new_directory_lands_once() -> None:
    """The positive half of spelling the last shard by exclusion: a directory
    added to the tree is in exactly one shard without anyone editing ci.yml."""
    entries = [*_test_tree_entries(), "tests/a_directory_added_later"]
    assert _shard_coverage_problems(_load_workflow(), entries) == []


def test_the_unit_shards_cover_the_test_tree__mutation_drops_a_matrix_shard() -> None:
    mutated = _load_workflow()
    _jobs(mutated)[UNIT_TEST_JOB]["strategy"]["matrix"]["shard"].pop()
    with pytest.raises(AssertionError, match="are not the matrix shards"):
        test_the_unit_shards_cover_the_test_tree_exactly_once(mutated)


# ---------------------------------------------------------------------------
# The unit lane's macOS cells are one trimmed job each
# ---------------------------------------------------------------------------
#
# The account's macOS ceiling is 5 concurrent jobs, so a macOS cell is one
# `include` job running the check's `macos` trim, never os x shard.

UNIT_MACOS_TRIM = "macos"

#: macOS jobs per event against the ceiling of 5: one push and three PRs fit.
UNIT_MACOS_BUDGET = {"pull_request": 1, "other": 2}


def _unit_macos_cells(wf: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """The unit matrix's `include` cells per event, from the two JSON arrays
    in its fromJSON expression (pull request first)."""
    include = str(_jobs(wf)[UNIT_TEST_JOB]["strategy"]["matrix"].get("include", ""))
    arrays = re.findall(r"'(\[[^']*\])'", include)
    assert re.search(r"github\.event_name\s*==\s*'pull_request'\s*&&", include) and (
        len(arrays) == 2
    ), f"unit matrix include is not the per-event fromJSON: {include!r}"
    return {"pull_request": json.loads(arrays[0]), "other": json.loads(arrays[1])}


def _macos_cell_problems(wf: dict[str, Any]) -> list[str]:
    matrix = _jobs(wf)[UNIT_TEST_JOB]["strategy"]["matrix"]
    problems = []
    if "macos-latest" in matrix["os"]:
        problems.append(
            f"macos-latest is on the os axis, so each shard is a macOS job: {matrix['os']}"
        )
    if UNIT_MACOS_TRIM in matrix["shard"]:
        problems.append(f"'{UNIT_MACOS_TRIM}' is on the shard axis, so ubuntu runs it too")
    if UNIT_MACOS_TRIM not in framework_checks.UNIT_SELECTION.trims:
        problems.append(f"the check declares no '{UNIT_MACOS_TRIM}' trim")
    cells = _unit_macos_cells(wf)
    for event, event_cells in cells.items():
        for cell in event_cells:
            if cell != {**cell, "os": "macos-latest", "shard": UNIT_MACOS_TRIM}:
                problems.append(f"{event} cell is not one macOS '{UNIT_MACOS_TRIM}' job: {cell}")
        if len(event_cells) > UNIT_MACOS_BUDGET[event]:
            problems.append(
                f"{event} runs {len(event_cells)} macOS jobs, over its "
                f"{UNIT_MACOS_BUDGET[event]} of the account's 5"
            )
    # Paired with the ubuntu coverage cell, a PR macOS failure isolates to the OS.
    coverage = _coverage_cell_version(wf)
    pr_versions = [cell.get("python-version") for cell in cells["pull_request"]]
    if pr_versions != [coverage]:
        problems.append(
            f"the pull request's macOS cell runs {pr_versions}, not the coverage cell's {coverage}"
        )
    return problems


def test_the_unit_macos_cells_are_one_trimmed_job_each(workflow: dict[str, Any]) -> None:
    assert _macos_cell_problems(workflow) == []


def test_the_unit_macos_cells_are_one_trimmed_job_each__mutation_puts_macos_on_the_axis() -> None:
    mutated = _load_workflow()
    _jobs(mutated)[UNIT_TEST_JOB]["strategy"]["matrix"]["os"].append("macos-latest")
    assert any("on the os axis" in p for p in _macos_cell_problems(mutated))


def test_the_unit_macos_cells_are_one_trimmed_job_each__mutation_runs_a_shard() -> None:
    mutated = _load_workflow()
    matrix = _jobs(mutated)[UNIT_TEST_JOB]["strategy"]["matrix"]
    matrix["include"] = matrix["include"].replace('"shard":"macos"', '"shard":"cli"', 1)
    assert any("is not one macOS 'macos' job" in p for p in _macos_cell_problems(mutated))


def test_the_unit_macos_cells_are_one_trimmed_job_each__mutation_moves_the_pr_cell() -> None:
    """Off the coverage cell's interpreter, a PR macOS failure no longer
    isolates to the OS."""
    mutated = _load_workflow()
    matrix = _jobs(mutated)[UNIT_TEST_JOB]["strategy"]["matrix"]
    original = matrix["include"]
    matrix["include"] = original.replace(
        '&& \'[{"python-version":"3.12"', '&& \'[{"python-version":"3.11"', 1
    )
    assert matrix["include"] != original, "PR macOS cell not found -- mutation is stale"
    assert any("not the coverage cell's" in p for p in _macos_cell_problems(mutated))


def test_the_unit_macos_cells_are_one_trimmed_job_each__mutation_adds_a_pr_cell() -> None:
    """A second PR macOS cell lets one push and three PRs exceed the ceiling."""
    mutated = _load_workflow()
    matrix = _jobs(mutated)[UNIT_TEST_JOB]["strategy"]["matrix"]
    original = matrix["include"]
    cell = '{"python-version":"3.11","os":"macos-latest","shard":"macos"}'
    matrix["include"] = original.replace("&& '[", f"&& '[{cell},", 1)
    assert matrix["include"] != original, "PR include array not found -- mutation is stale"
    assert any("over its 1" in p for p in _macos_cell_problems(mutated))


def test_the_unit_lane_runs_the_two_uid_log_case__mutation_moves_it_to_another_shard() -> None:
    """The case's directory leaves the shard the pin names, so the pinned
    shard name no longer says where the case runs."""
    directory = "/".join(TWO_UID_LOG_FILE.split("/")[:2])
    paths, ignores = framework_checks.UNIT_SELECTION.shards[TWO_UID_LOG_SHARD]
    kept = tuple(path for path in paths if path != directory)
    assert len(kept) == len(paths) - 1, (
        f"{directory} not named by '{TWO_UID_LOG_SHARD}'; mutation is stale"
    )
    mutated = _with_shard(TWO_UID_LOG_SHARD, kept, ignores)
    with pytest.raises(AssertionError, match="shard does not collect"):
        test_the_unit_lane_runs_the_two_uid_log_case_on_a_linux_cell(_load_workflow(), mutated)


def test_the_unit_shards_cover_the_test_tree__mutation_ignores_below_the_top_level() -> None:
    """A shard ignores a subdirectory of a directory it names: those files run in
    no shard, and a top-level check alone would still see the directory once."""
    paths, ignores = framework_checks.UNIT_SELECTION.shards[TWO_UID_LOG_SHARD]
    deeper = "tests/deployment/web_terminals"
    assert "tests/deployment" in paths, "rest-a no longer names tests/deployment"
    mutated = _with_shard(TWO_UID_LOG_SHARD, paths, (*ignores, deeper))
    with pytest.raises(AssertionError, match="below a top-level test directory"):
        test_the_unit_shards_cover_the_test_tree_exactly_once(_load_workflow(), mutated)


# ---------------------------------------------------------------------------
# Connectors compatibility lanes: Lane A runs HEAD connectors under every
# supported framework release, Lane B runs HEAD framework at its connectors
# floor, and the nightly runs Lane A and each release's own tests. Each stands
# down until the cut-over, and the stand-down has to lift by data alone
# ---------------------------------------------------------------------------

COMPAT_PLAN_JOB = "connectors-compat-plan"
COMPAT_JOB = "connectors-compat"
FLOOR_COMPAT_JOB = "framework-at-connectors-floor"
COMPAT_LISTED = "matrix.framework != ''"
COMPAT_UNLISTED = "matrix.framework == ''"
FLOOR_ARMED = "needs.connectors-compat-plan.outputs.floor-applicable == 'true'"
FLOOR_DISARMED = "needs.connectors-compat-plan.outputs.floor-applicable != 'true'"
COMPAT_MATRIX_SCRIPT = REPO_ROOT / "scripts" / "ci" / "connectors_compat_matrix.py"
COMPAT_LANE = "python3 scripts/ci/connectors_compat_lane.py"
NIGHTLY_YML = REPO_ROOT / ".github" / "workflows" / "connectors-compat-nightly.yml"
NIGHTLY_PLAN_JOB = "plan"
NIGHTLY_LANE_JOBS = ("connectors-compat", "release-tests")
NIGHTLY_REPORT_JOB = "report-failure"


def _compat_matrix_module() -> Any:
    spec = importlib.util.spec_from_file_location("connectors_compat_matrix", COMPAT_MATRIX_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_nightly() -> dict[str, Any]:
    with NIGHTLY_YML.open() as f:
        return yaml.safe_load(f)


@pytest.fixture()
def nightly() -> dict[str, Any]:
    return _load_nightly()


def _on(wf: dict[str, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True.
    return wf.get("on", wf.get(True))


def _stand_down_problems(wf: dict[str, Any], job_name: str, armed: str, disarmed: str) -> list[str]:
    """Why the job's one stand-down step is not the only thing its gate decides.

    The first step runs when disarmed and does nothing but say so; every other
    step runs exactly when armed. Anything else is a lane that can report
    success with part of it switched off once the gate lifts.
    """
    steps = _jobs(wf)[job_name]["steps"]
    problems = []
    if steps[0].get("if") != disarmed or "uses" in steps[0]:
        problems.append(f"first step {steps[0].get('name')!r} is not the '{disarmed}' stand-down")
    for step in steps[1:]:
        if step.get("if") != armed:
            problems.append(f"step {step.get('name')!r} runs on {step.get('if')!r}, not '{armed}'")
    return problems


def _runs(wf: dict[str, Any], job_name: str) -> str:
    return " ".join(str(step.get("run", "")) for step in _jobs(wf)[job_name]["steps"])


def test_the_compat_lanes_read_the_plan(workflow: dict[str, Any]) -> None:
    jobs = _jobs(workflow)
    assert jobs[COMPAT_JOB]["needs"] == COMPAT_PLAN_JOB
    assert jobs[FLOOR_COMPAT_JOB]["needs"] == COMPAT_PLAN_JOB
    assert jobs[COMPAT_JOB]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.connectors-compat-plan.outputs.matrix) }}"
    )
    plan_runs = [step.get("run", "") for step in jobs[COMPAT_PLAN_JOB]["steps"]]
    assert "python3 scripts/ci/connectors_compat_matrix.py github-output" in plan_runs


@pytest.mark.parametrize("source", ["ci", "nightly"])
def test_the_plan_reads_the_full_history(source: str) -> None:
    """The plan holds the declarations to every tag reachable from HEAD, which a
    shallow checkout hides (the script refuses one)."""
    wf, job = (
        (_load_workflow(), COMPAT_PLAN_JOB)
        if source == "ci"
        else (_load_nightly(), NIGHTLY_PLAN_JOB)
    )
    checkout = _jobs(wf)[job]["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout.get("with", {}).get("fetch-depth") == 0


def test_the_compat_lanes_carry_no_job_level_condition(workflow: dict[str, Any]) -> None:
    """A job-level ``if:`` would skip the lane, which the gate reads as red; the
    stand-down is a step so the lane is green for the reason its log states."""
    for job_name in (COMPAT_PLAN_JOB, COMPAT_JOB, FLOOR_COMPAT_JOB):
        assert "if" not in _jobs(workflow)[job_name], job_name


def test_lane_a_stands_down_only_on_the_empty_cell(workflow: dict[str, Any]) -> None:
    assert _stand_down_problems(workflow, COMPAT_JOB, COMPAT_LISTED, COMPAT_UNLISTED) == []


def test_lane_a_stands_down_only_on_the_empty_cell__mutation_gates_a_step_elsewhere() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[COMPAT_JOB]["steps"][-1]["if"] = "github.event_name == 'push'"
    assert len(_stand_down_problems(mutated, COMPAT_JOB, COMPAT_LISTED, COMPAT_UNLISTED)) == 1


def test_lane_b_stands_down_only_while_the_floor_predates_independence(
    workflow: dict[str, Any],
) -> None:
    assert _stand_down_problems(workflow, FLOOR_COMPAT_JOB, FLOOR_ARMED, FLOOR_DISARMED) == []


def test_lane_b_stands_down_only_while__mutation_ungates_a_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[FLOOR_COMPAT_JOB]["steps"][-1].pop("if")
    assert len(_stand_down_problems(mutated, FLOOR_COMPAT_JOB, FLOOR_ARMED, FLOOR_DISARMED)) == 1


def test_the_empty_cell_prints_the_reason_the_plan_gives() -> None:
    """One wording, the script's: the stand-down step echoes the cell's reason."""
    for wf, job in [(_load_workflow(), COMPAT_JOB)] + [
        (_load_nightly(), j) for j in NIGHTLY_LANE_JOBS
    ]:
        step = _jobs(wf)[job]["steps"][0]
        assert step["env"] == {"REASON": "${{ matrix.reason }}"}, job
        assert step["run"].strip() == 'echo "$REASON"', job
    assert _compat_matrix_module().matrix([])["include"][0]["reason"]


def test_lane_a_runs_the_shared_steps(workflow: dict[str, Any]) -> None:
    """The nightly's dry run exercises Lane A only if both run the same driver;
    the driver runs every compat script ``python -I`` (test_connectors_compat_lane.py)."""
    for wf, job in ((workflow, COMPAT_JOB), (_load_nightly(), "connectors-compat")):
        runs = _runs(wf, job)
        assert f"{COMPAT_LANE} install" in runs, job
        assert f"{COMPAT_LANE} check" in runs, job
        assert "osprey_connectors_tests/compat" not in runs, job


def test_lane_a_installs_the_wheel_built_at_head(workflow: dict[str, Any]) -> None:
    runs = _runs(workflow, COMPAT_JOB)
    assert "uv build --package osprey-connectors --wheel --out-dir dist/" in runs
    assert "--connectors head" in runs


def test_lane_b_installs_the_published_metadata_at_the_floor(workflow: dict[str, Any]) -> None:
    """Only the connectors leave the workspace. ``--no-sources`` drops every
    workspace source (``--no-sources-package osprey-connectors`` alone leaves the
    root's pyaml-cs-osprey one, which ``uv pip install`` of the root refuses), so
    the workspace pyaml-cs-osprey is named by its path or it would come from PyPI."""
    install = _find_named_step(
        workflow, FLOOR_COMPAT_JOB, "Install the framework with the floor connectors"
    )["run"]
    assert "--no-sources " in install
    assert "-e ./packages/pyaml-cs-osprey " in install
    assert '"osprey-connectors==$FLOOR"' in install


def test_lane_b_holds_the_suites_to_the_floor(workflow: dict[str, Any]) -> None:
    runs = _runs(workflow, FLOOR_COMPAT_JOB)
    assert (
        f"{COMPAT_LANE} assert-installed --python .venv/bin/python --module osprey_connectors "
        '--connectors-version "$FLOOR"'
    ) in runs
    assert (
        f"{COMPAT_LANE} pytest --python .venv/bin/python --installed osprey_connectors --" in runs
    )
    scan = _find_named_step(
        workflow, FLOOR_COMPAT_JOB, "Scan the framework for connectors names the floor lacks"
    )["run"]
    assert scan.startswith(".venv/bin/python -I ")
    assert "src/osprey packages/pyaml-cs-osprey/src/pyaml_cs_osprey" in scan


def test_the_stand_down_lifts_once_the_floor_is_an_independent_release() -> None:
    """The declarations agree with the tags this checkout carries. Only the plan
    job is authoritative for the tag rules: a shallow checkout, or one with no
    framework tag after the last paired release, checks what it can."""
    compat = _compat_matrix_module()
    try:
        tags: list[str] | None = compat.reachable_tags()
    except ValueError:
        tags = None
    if tags is not None and not compat.framework_releases(tags):
        tags = None
    assert (
        compat.problems(
            compat.supported_frameworks(),
            compat.retired_frameworks(),
            compat.connectors_floor(),
            tags,
        )
        == []
    )


def test_the_stand_down_lifts__mutation_raises_the_floor_without_its_tag() -> None:
    compat = _compat_matrix_module()
    assert compat.problems([], [], compat.FIRST_INDEPENDENT_CONNECTORS, []) != []


# -- the nightly ---------------------------------------------------------------


def test_the_nightly_dispatch_names_a_release_and_the_connectors(nightly: dict[str, Any]) -> None:
    inputs = _on(nightly)["workflow_dispatch"]["inputs"]
    assert inputs["framework"]["default"] == ""
    assert inputs["connectors"]["default"] == "head"
    assert nightly["env"]["CONNECTORS"] == "${{ inputs.connectors || 'head' }}"
    plan = _jobs(nightly)[NIGHTLY_PLAN_JOB]["steps"][-1]
    assert plan["env"] == {"DISPATCHED": "${{ inputs.framework }}"}
    assert plan["run"] == (
        'python3 scripts/ci/connectors_compat_matrix.py github-output --framework "$DISPATCHED"'
    )


@pytest.mark.parametrize("job", NIGHTLY_LANE_JOBS)
def test_the_nightly_lanes_stand_down_only_on_the_empty_cell(
    nightly: dict[str, Any], job: str
) -> None:
    jobs = _jobs(nightly)
    assert "if" not in jobs[job], job
    assert jobs[job]["needs"] == NIGHTLY_PLAN_JOB
    assert jobs[job]["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.matrix) }}"
    assert _stand_down_problems(nightly, job, COMPAT_LISTED, COMPAT_UNLISTED) == []


def test_the_nightly_lanes_stand_down__mutation_ungates_a_step() -> None:
    mutated = _load_nightly()
    _jobs(mutated)["release-tests"]["steps"][-1].pop("if")
    assert len(_stand_down_problems(mutated, "release-tests", COMPAT_LISTED, COMPAT_UNLISTED)) == 1


def test_the_release_tests_run_guarded_against_the_installed_packages(
    nightly: dict[str, Any],
) -> None:
    runs = _runs(nightly, "release-tests")
    assert f'{COMPAT_LANE} install --venv "$RELEASE_DIR/.venv"' in runs
    assert "--extra dev" in runs
    assert f"{COMPAT_LANE} replace-connectors-tree" in runs
    assert "--installed osprey --installed osprey_connectors --" in runs
    assert "rm -rf" not in runs, "the release's sources stay: its tests read them"


def test_only_the_report_job_may_write_issues(nightly: dict[str, Any]) -> None:
    assert nightly["permissions"] == {"contents": "read"}
    jobs = _jobs(nightly)
    for name, job in jobs.items():
        if name != NIGHTLY_REPORT_JOB:
            assert "issues" not in json.dumps(job.get("permissions", {})), name
            assert "gh issue" not in json.dumps(job), name
    assert jobs[NIGHTLY_REPORT_JOB]["permissions"] == {"actions": "read", "issues": "write"}


def test_the_report_job_files_for_a_failed_scheduled_run(nightly: dict[str, Any]) -> None:
    """Every other job is needed, the plan's failure included, and a dispatched
    dry run files nothing."""
    report = _jobs(nightly)[NIGHTLY_REPORT_JOB]
    assert set(report["needs"]) == set(_jobs(nightly)) - {NIGHTLY_REPORT_JOB}
    assert report["if"] == "failure() && github.event_name == 'schedule'"
    assert all("uses" not in step for step in report["steps"]), "the report job checks nothing out"


def test_every_nightly_checkout_drops_its_credentials(nightly: dict[str, Any]) -> None:
    checkouts = [
        (name, step)
        for name, job in _jobs(nightly).items()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkouts) == 4
    for name, step in checkouts:
        assert step["with"]["persist-credentials"] is False, (name, step.get("name"))


# ---------------------------------------------------------------------------
# One check set: the caller jobs run scripts/check.py and spell no check
# ---------------------------------------------------------------------------
#
# A check is declared once, in scripts/check.py, and CI is one of its callers.
# Three statements keep that true, each read from ci.yml as written: no step of
# a caller job spells a check, every check of the full tier is called by one,
# and every job the roll-up gate requires is a caller or a lane the declaration
# names.

#: A `run:` of a caller job that prepares the job or reports on it, and judges
#: nothing.
_SETUP_RUN_RES = (
    re.compile(r"uv python install \S+(?: \S+)*"),
    re.compile(r"uv sync(?: --extra [\w-]+)*"),
    re.compile(r"npm ci"),
    re.compile(r"sudo apt-get update && sudo apt-get install -y --no-install-recommends [\w -]+"),
    re.compile(
        r"python3 scripts/ci/diag_summary\.py ci-diag >> \"\$GITHUB_STEP_SUMMARY\" \|\| true"
    ),
    re.compile(
        r'curl -sSfL -o "\$RUNNER_TEMP/gitleaks\.tar\.gz" '
        r'"https://github\.com/gitleaks/gitleaks/releases/download/v\$\{GITLEAKS_VERSION\}/'
        r'gitleaks_\$\{GITLEAKS_VERSION\}_linux_x64\.tar\.gz"\n'
        r'echo "\$\{GITLEAKS_SHA256\}  \$RUNNER_TEMP/gitleaks\.tar\.gz" \| sha256sum -c -\n'
        r'sudo tar -C /usr/local/bin -xzf "\$RUNNER_TEMP/gitleaks\.tar\.gz" gitleaks'
    ),
)
#: The steps of a caller job that prepare or report in ways no pattern above
#: states, by job and step name; each has its own tests above. None on main: a
#: caller job a branch adds names its own here.
CALLER_SETUP_STEPS: dict[str, tuple[str, ...]] = {}
#: The commands the checks run. One of these in a caller step is a check
#: spelled a second time.
LEAF_COMMANDS = (
    "ruff ",
    "mypy",
    "pytest",
    "tsc",
    "eslint",
    "actionlint",
    "vitest",
    "lint-imports",
    "npm run ",
    "build.mjs",
    "changelog_fragments.py",
    "check_config_keys.py",
    "plugin_version.py",
    "live_ca/gate.py",
    "uv lock --check",
    "git ls-files -ci",
    "make html",
    "make linkcheck",
    "twine",
    "uv build",
    "gitleaks git",
)
#: Full-tier checks no caller step names, each with the reason CI needs no step
#: for it.
CHECKS_WITHOUT_A_CI_STEP = {
    # Its files are test modules inside the parallel sweep, so the `pytest`
    # caller runs them on every CI run; a step of its own would run them twice.
    "invariant-tests": "its files are inside the parallel sweep the pytest caller runs",
    # The wall-clock budgets run alone in the lane that hosts their marker
    # (LANE_LEAVING_MARKERS), on a runner with no xdist workers beside them.
    "pytest-wallclock": "its suite runs serially in the lane that hosts the wallclock marker",
}
#: The `if:` clause that puts a step on the runners that meet the venue.
VENUE_RUNNER_CLAUSES = {framework_checks.LIVE_CA_VENUE: "matrix.os == 'ubuntu-latest'"}


@pytest.fixture()
def raw_workflow() -> dict[str, Any]:
    return _load_raw_workflow()


def _leaf_commands(run: object) -> list[str]:
    """The check commands a ``run:`` script spells outside its check.py calls.

    A log file's name is not a command: ``ci-diag/pytest.log`` is where the
    unit step tees its output.
    """
    words = [
        word
        for line in _uncommented_lines(run)
        for word in _CHECK_CALL_RE.sub(" ", line).split()
        if not word.endswith(".log")
    ]
    text = " ".join(words) + " "
    return [leaf for leaf in LEAF_COMMANDS if leaf in text]


def _caller_steps(wf: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every step of a caller job that runs a script, with its job."""
    return [
        (job, step)
        for job in check_set.CALLER_JOBS
        for step in _jobs(wf)[job]["steps"]
        if "run" in step
    ]


def _addopts_words() -> list[str]:
    """The words of pyproject's ``[tool.pytest.ini_options].addopts``."""
    return shlex.split(_load_pyproject()["tool"]["pytest"]["ini_options"]["addopts"])


def _callers_repeating_addopts(wf: dict[str, Any]) -> list[str]:
    """Each caller's pass-through word that addopts already gives pytest."""
    configured = set(_addopts_words())
    problems = []
    for job, step in _caller_steps(wf):
        for call in _check_calls(step):
            passed = call[call.index("--") + 1 :] if "--" in call else []
            problems.extend(
                f"{job}: '{step.get('name')}' passes {word}"
                for word in passed
                if word in configured
            )
    return problems


def test_no_caller_repeats_the_configured_pytest_options(raw_workflow: dict[str, Any]) -> None:
    """pyproject's addopts is the one home of pytest's verbosity and traceback
    style. A caller that passes one of its words again runs pytest with it
    twice, so ``-v`` becomes ``-vv`` and a failing log buries the failure."""
    assert _addopts_words(), "pyproject sets no addopts; this pin is stale"
    assert _callers_repeating_addopts(raw_workflow) == []


def test_no_caller_repeats_the_configured_pytest_options__mutation_adds_v() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace(" -- --durations=50", " -- -v --durations=50")
    assert step["run"] != original, "the unit step's flags moved; mutation is stale"
    with pytest.raises(AssertionError):
        test_no_caller_repeats_the_configured_pytest_options(mutated)


#: The one CI step that passes execution flags through the check command: the
#: unit sweep, whose durations, timeout, workers and coverage flags are not yet
#: part of the pytest check's declaration. Every other step leaves execution to
#: the declaration, so each check runs on CI as it runs on a laptop.
EXECUTION_FLAGS_EXCEPTION = (UNIT_TEST_JOB, "Run unit tests")


def _steps_passing_execution_flags(wf: dict[str, Any]) -> list[str]:
    """Each step outside the exception that passes words after ``--`` to the
    check command, and the exception when it no longer passes any."""
    problems = []
    exception_passes = False
    for job, body in _jobs(wf).items():
        for step in body.get("steps", []):
            for call in _check_calls(step):
                if "--" not in call:
                    continue
                if (job, step.get("name")) == EXECUTION_FLAGS_EXCEPTION:
                    exception_passes = True
                    continue
                passed = " ".join(call[call.index("--") + 1 :])
                problems.append(
                    f"{job}: '{step.get('name')}' passes {passed!r} through {CHECK_SCRIPT}"
                )
    if not exception_passes:
        problems.append(
            f"{EXECUTION_FLAGS_EXCEPTION} passes no execution flags; drop the exception"
        )
    return problems


def test_no_ci_step_passes_execution_flags_through_the_check_command(
    raw_workflow: dict[str, Any],
) -> None:
    """Execution flags are a laptop affordance: on CI a check runs with what its
    declaration says, verbosity included, set in one place."""
    assert _steps_passing_execution_flags(raw_workflow) == []


def test_no_ci_step_passes_execution_flags__mutation_the_pty_step_passes_v() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, UNIT_TEST_JOB, TERMINAL_HOST.step)
    step["run"] = f"{step['run'].rstrip()} -- -v"
    assert _steps_passing_execution_flags(mutated) == [
        f"test: '{TERMINAL_HOST.step}' passes '-v' through {CHECK_SCRIPT}"
    ]


def test_no_ci_step_passes_execution_flags__mutation_the_sweep_drops_its_flags() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, *EXECUTION_FLAGS_EXCEPTION)
    old = " -- --durations=50 --timeout=600 --timeout-method=signal -n 4 $COV_ARGS"
    assert step["run"].count(old) == 1, f"{old!r} moved — mutation is stale"
    step["run"] = step["run"].replace(old, "")
    assert _steps_passing_execution_flags(mutated) == [
        f"{EXECUTION_FLAGS_EXCEPTION} passes no execution flags; drop the exception"
    ]


def _is_setup(job: str, step: dict[str, Any]) -> bool:
    return step.get("name") in CALLER_SETUP_STEPS.get(job, ())


def _spelled_checks(wf: dict[str, Any]) -> list[str]:
    """What is wrong with the caller jobs' steps as callers, one line per defect."""
    problems = []
    for job, step in _caller_steps(wf):
        where = f"{job}: '{step.get('name')}'"
        calls = _check_calls(step)
        run = str(step["run"]).strip()
        setup = _is_setup(job, step)
        if setup and calls:
            problems.append(f"{where} is declared a setup step and calls {CHECK_SCRIPT}")
        elif not calls and not setup and not any(p.fullmatch(run) for p in _SETUP_RUN_RES):
            problems.append(f"{where} is neither a setup step nor a call of {CHECK_SCRIPT}")
        for call in calls:
            if call[:1] != ["run"]:
                problems.append(f"{where} must call `{CHECK_SCRIPT} run ID`; got {call}")
                continue
            try:
                _resolved(call)
            except (check_set.Refused, SystemExit) as refusal:
                problems.append(f"{where} makes a call {CHECK_SCRIPT} refuses: {refusal}")
        spelled = _leaf_commands(step["run"])
        if spelled:
            problems.append(f"{where} spells {spelled}")
    return problems


def test_no_caller_step_spells_a_check(raw_workflow: dict[str, Any]) -> None:
    """Two spellings of one check drift apart, and the one a contributor runs
    locally is then not the one that gates the merge. So a caller job's step
    either prepares the job or names a check, and the command it names is one
    the declaration accepts."""
    assert _caller_steps(raw_workflow), "the caller jobs run no script; this scan is stale"
    assert _spelled_checks(raw_workflow) == []


def test_no_caller_step_spells_a_check__mutation_restores_the_ruff_command() -> None:
    mutated = _load_raw_workflow()
    _find_named_step(mutated, LINT_JOB, "Run ruff (linting)")["run"] = "uv run ruff check ."
    assert _spelled_checks(mutated) == [
        f"lint: 'Run ruff (linting)' is neither a setup step nor a call of {CHECK_SCRIPT}",
        "lint: 'Run ruff (linting)' spells ['ruff ']",
    ]


def test_no_caller_step_spells_a_check__mutation_restores_the_bundle_check() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, FRONTEND_JS_JOB, "Check the dock-engine bundle is current")
    step["run"] = "node packages/dock-engine/scripts/build.mjs --check"
    assert _spelled_checks(mutated) == [
        f"frontend-js: 'Check the dock-engine bundle is current' is neither a setup step "
        f"nor a call of {CHECK_SCRIPT}",
        "frontend-js: 'Check the dock-engine bundle is current' spells ['build.mjs']",
    ]


def test_no_caller_step_spells_a_check__mutation_spells_actionlint() -> None:
    mutated = _load_raw_workflow()
    _find_named_step(mutated, LINT_JOB, "Run actionlint (workflow lint)")["run"] = (
        "uv run actionlint"
    )
    assert _spelled_checks(mutated) == [
        f"lint: 'Run actionlint (workflow lint)' is neither a setup step nor a call of "
        f"{CHECK_SCRIPT}",
        "lint: 'Run actionlint (workflow lint)' spells ['actionlint']",
    ]


def test_no_caller_step_spells_a_check__mutation_spells_the_secret_scan() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, LINT_JOB, GITLEAKS_SCAN_STEP)
    step["run"] = 'gitleaks git --log-opts="origin/$BASE_REF..HEAD" --no-banner --redact'
    assert _spelled_checks(mutated) == [
        f"lint: '{GITLEAKS_SCAN_STEP}' is neither a setup step nor a call of {CHECK_SCRIPT}",
        f"lint: '{GITLEAKS_SCAN_STEP}' spells ['gitleaks git']",
    ]


def test_no_caller_step_spells_a_check__mutation_spells_one_beside_a_call() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, FRONTEND_JS_JOB, "Run JS tests (vitest)")
    step["run"] = f"{step['run'].rstrip()} && npm run test:js"
    assert _spelled_checks(mutated) == ["frontend-js: 'Run JS tests (vitest)' spells ['npm run ']"]


def test_no_caller_step_spells_a_check__mutation_passes_a_selection() -> None:
    """A selection flag after ``--`` would make the step a second declaration
    of what the sweep runs; the command refuses it."""
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run unit tests")
    original = step["run"]
    step["run"] = original.replace(" -n 4 $COV_ARGS", " -n 4 -k smoke $COV_ARGS")
    assert step["run"] != original, "the unit step's flags moved; mutation is stale"
    (problem,) = _spelled_checks(mutated)
    assert problem.startswith(f"test: 'Run unit tests' makes a call {CHECK_SCRIPT} refuses: -k")


def test_no_caller_step_spells_a_check__mutation_runs_a_tier() -> None:
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, LINT_JOB, MYPY_STEP)
    step["run"] = step["run"].replace("run mypy", "quick")
    (problem,) = _spelled_checks(mutated)
    assert problem.startswith(f"lint: '{MYPY_STEP}' must call `{CHECK_SCRIPT} run ID`")


def test_no_caller_step_spells_a_check__mutation_adds_a_step_that_is_neither() -> None:
    mutated = _load_raw_workflow()
    _jobs(mutated)["docs"]["steps"].append({"name": "Count pages", "run": "ls docs/build/html"})
    assert _spelled_checks(mutated) == [
        f"docs: 'Count pages' is neither a setup step nor a call of {CHECK_SCRIPT}"
    ]


def test_a_log_file_name_is_not_a_spelled_check() -> None:
    assert _leaf_commands("x 2>&1 | tee ci-diag/pytest.log\ngrep -m1 y ci-diag/pytest.log") == []
    assert _leaf_commands("uv run pytest tests/  # serial") == ["pytest"]
    assert _leaf_commands("echo ok  # was: uv run mypy src/") == []
    assert _leaf_commands("python3 scripts/check.py run pytest mypy | tee out.log") == []


def _called_checks(wf: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Each check id a caller step names after ``run``, with the steps that name it."""
    by_id = {check.id for check in check_set.CHECKS}
    called: dict[str, list[dict[str, Any]]] = {}
    for _job, step in _caller_steps(wf):
        for call in _check_calls(step):
            for word in call[1:] if call[:1] == ["run"] else []:
                if word in by_id:
                    called.setdefault(word, []).append(step)
    return called


def _swept(path: str, unit: Any) -> bool:
    """Whether the parallel sweep collects the test module at *path*."""
    ignored = any(
        path == entry or path.startswith(entry.rstrip("/") + "/") for entry, _host in unit.ignores
    )
    rooted = any(path.startswith(root.rstrip("/") + "/") for root in unit.roots)
    return rooted and not path.startswith(TERMINAL_HOST.path) and not ignored


def _checks_missing_from_ci(wf: dict[str, Any], unit: Any = None) -> list[str]:
    """What is wrong with CI as a caller of the full tier, one line per defect."""
    unit = unit or framework_checks.UNIT_SELECTION
    called = _called_checks(wf)
    problems = []
    for check in check_set.CHECKS:
        if "full" not in check.tiers:
            continue
        steps = called.get(check.id, [])
        if check.id in CHECKS_WITHOUT_A_CI_STEP:
            if steps:
                problems.append(f"{check.id}: a caller step runs it; drop its exemption")
        elif not steps:
            problems.append(f"{check.id}: no caller step runs it")
        elif check.venue:
            clause = VENUE_RUNNER_CLAUSES[check.venue]
            off_venue = [step.get("name") for step in steps if clause not in str(step.get("if"))]
            if off_venue:
                problems.append(f"{check.id}: {off_venue} is not held to {clause}")
    declared = {check.id for check in check_set.CHECKS}
    problems += [
        f"{name}: exempted and not declared"
        for name in sorted(set(CHECKS_WITHOUT_A_CI_STEP) - declared)
    ]
    problems += [
        f"invariant-tests: {path} is outside the parallel sweep"
        for path in framework_checks.INVARIANT_TESTS
        if not _swept(path, unit)
    ]
    return problems


def test_every_full_check_is_called_on_ci(raw_workflow: dict[str, Any]) -> None:
    """A check the full tier runs and CI does not is a check a pushed change
    passes without. Every full-tier check is named by a caller step, and a
    check bound to a venue by a step held to runners that meet it; each
    exemption is named beside its reason in ``CHECKS_WITHOUT_A_CI_STEP``, and
    the reason of ``invariant-tests`` is a fact this test holds: every file of
    it is collected by the parallel sweep."""
    assert _checks_missing_from_ci(raw_workflow) == []


def test_every_full_check_is_called_on_ci__mutation_the_sweep_ignores_an_invariant_test() -> None:
    """The exemption of ``invariant-tests`` holds only while the sweep runs
    its files: an ignore over one of them leaves it run by nothing on CI."""
    unit = framework_checks.UNIT_SELECTION
    path = "tests/deployment/test_ci_workflow_wiring.py"
    narrowed = dataclasses.replace(unit, ignores=(*unit.ignores, (path, "none")))
    assert _checks_missing_from_ci(_load_raw_workflow(), narrowed) == [
        f"invariant-tests: {path} is outside the parallel sweep"
    ]


def test_every_full_check_is_called_on_ci__mutation_drops_the_mypy_step() -> None:
    mutated = _load_raw_workflow()
    steps = _jobs(mutated)[LINT_JOB]["steps"]
    steps[:] = [step for step in steps if step.get("name") != MYPY_STEP]
    assert _checks_missing_from_ci(mutated) == ["mypy: no caller step runs it"]


def test_every_full_check_is_called_on_ci__mutation_drops_the_bundle_step() -> None:
    mutated = _load_raw_workflow()
    steps = _jobs(mutated)[FRONTEND_JS_JOB]["steps"]
    steps[:] = [s for s in steps if s.get("name") != "Check the dock-engine bundle is current"]
    assert _checks_missing_from_ci(mutated) == ["dock-engine-bundle: no caller step runs it"]


def test_every_full_check_is_called_on_ci__mutation_runs_live_ca_on_every_cell() -> None:
    """Off its venue the check reports that it did not run and exits clean, so
    a step on a macOS cell would be green over nothing."""
    mutated = _load_raw_workflow()
    step = _find_named_step(mutated, UNIT_TEST_JOB, "Run live Channel Access suites in isolation")
    del step["if"]
    (problem,) = _checks_missing_from_ci(mutated)
    assert problem.startswith("live-ca: ['Run live Channel Access suites in isolation']")


def test_every_full_check_is_called_on_ci__mutation_calls_an_exempted_check() -> None:
    mutated = _load_raw_workflow()
    _jobs(mutated)[LINT_JOB]["steps"].append(
        {
            "name": "Run the wall-clock budgets",
            "run": "uv run python scripts/check.py run pytest-wallclock",
        }
    )
    assert _checks_missing_from_ci(mutated) == [
        "pytest-wallclock: a caller step runs it; drop its exemption"
    ]


@dataclass(frozen=True)
class CallerCall:
    """One ``check.py run`` call a caller step makes, as ci.yml spells it."""

    job: str
    step: str
    call: tuple[str, ...]

    @property
    def checks(self) -> tuple[str, ...]:
        declared = {check.id for check in check_set.CHECKS}
        return tuple(word for word in self.call[1:] if word in declared)


def _gate_calls(wf: dict[str, Any]) -> list[CallerCall]:
    """Every call a caller step makes that the full tier spells the same way.

    A call that passes execution flags or a shard, or names a check the full
    tier passes execution flags to, runs with flags of its own on CI; the
    lane-leaving-marker tests above hold those.
    """
    flagged = {check.id for check in check_set.CHECKS if check.tier_flags.get("full")}
    calls = []
    for job, step in _caller_steps(wf):
        for call in _check_calls(step):
            row = CallerCall(job, str(step.get("name")), tuple(call))
            if "--" in call or "--shard" in call or flagged & set(row.checks):
                continue
            calls.append(row)
    return calls


#: Read from ci.yml as written: the rows are the caller steps, not a table of them.
GATE_CALLS = _gate_calls(_load_raw_workflow())


def _ci_commands(row: CallerCall) -> list[str]:
    """The command lines *row*'s call resolves to, one shell line each."""
    return [_render(argv).replace("\n", " ") for argv in check_set.resolve(list(row.call))]


def test_the_caller_steps_make_gate_calls() -> None:
    assert GATE_CALLS, "no caller step calls a check the full tier spells alike; scan is stale"
    assert len({row.step for row in GATE_CALLS}) == len(GATE_CALLS)


@pytest.mark.parametrize("row", GATE_CALLS, ids=lambda row: row.step)
def test_the_gate_runs_each_ci_step_once(row: CallerCall, gate_source: str | None = None) -> None:
    """The full tier runs the call's checks, and each of its commands exactly
    once. A check implemented in Python runs no command of its own."""
    assert set(row.checks) <= {check.id for check in check_set.tier_checks("full")}
    source = _tier_source("full", CI_BASE) if gate_source is None else gate_source
    lines = source.splitlines()
    for command in _ci_commands(row):
        count = lines.count(command)
        assert count == 1, f"the full tier runs {command!r} {count} times, not once"


#: The calls whose checks run at least one command.
_COMMAND_CALLS = tuple(row for row in GATE_CALLS if check_set.resolve(list(row.call)))


@pytest.mark.parametrize("row", _COMMAND_CALLS, ids=lambda row: row.step)
def test_the_gate_runs_each_ci_step_once__mutation_drops_the_line(row: CallerCall) -> None:
    source = _tier_source("full", CI_BASE)
    first = _ci_commands(row)[0]
    mutated = "\n".join(line for line in source.splitlines() if line != first)
    assert mutated != source, f"the full tier has no {first!r}; mutation is stale"
    with pytest.raises(AssertionError):
        test_the_gate_runs_each_ci_step_once(row, mutated)


def test_the_gate_runs_each_ci_step_once__mutation_hardcodes_a_second_base() -> None:
    (row,) = [row for row in GATE_CALLS if row.step == CHANGELOG_STEP]
    source = _tier_source("full", CI_BASE)
    (command,) = _ci_commands(row)
    mutated = source.replace(command, command + " --base origin/main", 1)
    assert mutated != source, f"the full tier has no {command!r}; mutation is stale"
    with pytest.raises(AssertionError):
        test_the_gate_runs_each_ci_step_once(row, mutated)


def test_a_new_caller_step_is_a_gate_call_without_a_table_row() -> None:
    """A step a branch adds is held the moment it lands in ci.yml: nothing
    beside the workflow lists it."""
    mutated = _load_raw_workflow()
    _jobs(mutated)[LINT_JOB]["steps"].append(
        {
            "name": "List what git ignores, again",
            "run": "python3 scripts/check.py run tracked-ignored",
        }
    )
    assert "List what git ignores, again" in {row.step for row in _gate_calls(mutated)}


def _unknown_required_jobs(wf: dict[str, Any]) -> list[str]:
    """What is wrong with the roll-up gate's needs as the declaration reads them."""
    jobs = _jobs(wf)
    needs = jobs[GATE_JOB]["needs"]
    lanes = {lane.job for lane in check_set.LANES}
    callers = set(check_set.CALLER_JOBS)
    problems = [
        f"{job}: required, and neither a caller job nor a lane {CHECK_SCRIPT} names"
        for job in needs
        if job not in callers | lanes
    ]
    problems += [
        f"{job}: a caller job the gate does not require" for job in sorted(callers - set(needs))
    ]
    problems += [
        f"{job}: a caller job the workflow does not define" for job in sorted(callers - set(jobs))
    ]
    return problems


def test_every_required_job_is_a_caller_or_a_named_lane(raw_workflow: dict[str, Any]) -> None:
    """The check command names every CI lane it does not run. A required job
    it does not know is a job a clean local verdict says nothing about, and
    says nothing about saying nothing."""
    assert _unknown_required_jobs(raw_workflow) == []


def test_every_required_job_is_a_caller_or_a_named_lane__mutation_requires_an_unnamed_job() -> None:
    mutated = _load_raw_workflow()
    _jobs(mutated)["release-notes"] = {"runs-on": "ubuntu-latest", "steps": []}
    _jobs(mutated)[GATE_JOB]["needs"].append("release-notes")
    assert _unknown_required_jobs(mutated) == [
        f"release-notes: required, and neither a caller job nor a lane {CHECK_SCRIPT} names"
    ]


def test_every_required_job_is_a_caller_or_a_named_lane__mutation_gate_drops_a_caller() -> None:
    mutated = _load_raw_workflow()
    _jobs(mutated)[GATE_JOB]["needs"].remove("frontend-js")
    assert _unknown_required_jobs(mutated) == [
        "frontend-js: a caller job the gate does not require"
    ]


# ---------------------------------------------------------------------------
# Wiring every dedicated lane shares
#
# Each dedicated lane must exist, most must hold no LLM secret, and each must
# gate the merge. The per-lane sections above pin what is particular to a lane;
# these tables pin the common contracts once, a row per lane.
# ---------------------------------------------------------------------------


#: Lanes whose job must exist under exactly this name.
_DEDICATED_LANES = [
    TILED_JOB,
    VA_JOB,
    ORM_JOB,
    OVERLAY_JOB,
    FLOOR_JOB,
    NEXTCLOUD_JOB,
    GCHAT_JOB,
    TEAMS_JOB,
    ARCHIVER_JOB,
    TWO_SHAPE_JOB,
    FULL_CHAIN_JOB,
]


@pytest.mark.parametrize("job", _DEDICATED_LANES)
def test_dedicated_lane_job_exists(workflow: dict[str, Any], job: str) -> None:
    assert job in _jobs(workflow)


@pytest.mark.parametrize("job", _DEDICATED_LANES)
def test_dedicated_lane_job_exists__mutation_drops_job(job: str) -> None:
    """Dropping the job from the parsed dict must fail the existence check."""
    mutated = copy.deepcopy(_load_workflow())
    del mutated["jobs"][job]
    with pytest.raises(AssertionError):
        test_dedicated_lane_job_exists(mutated, job)


#: The lanes that must never hold the LLM gateway key, each with the reason a
#: secret appearing there would be wrong.
_SECRET_FREE_LANES = [
    TILED_JOB,
    VA_JOB,
    # The ORM roundtrip drives the bridge HTTP API directly — an LLM secret
    # appearing in its job would mean the lane's scope silently grew.
    ORM_JOB,
    # The Teams lane is deterministic end to end: loopback fakes for the token
    # endpoint, the Bot Connector and the queue receiver, and no agentic tier at
    # all. Handing it the gateway key would buy nothing and would move it into
    # the spend-posture classification, where it would have to be gated by
    # label — turning a lane that can honestly run on every pull request into
    # one that runs on a few.
    TEAMS_JOB,
    # Every test this lane selects carries ``model_free``, so the directory's
    # provider refusal does not apply to it; the workflow-level
    # ``OSPREY_E2E_PROVIDER`` is still present but not needed here. A gateway
    # key would buy nothing and would move the lane into the label-gated
    # posture it was built to leave.
    NO_MODEL_JOB,
    # Every stage drives the bridge HTTP API and the osprey CLI directly — an
    # LLM secret appearing here would mean the lane's scope silently grew.
    QUEUE_JOB,
    # Every assertion in that file is mechanical — seed duration, store size, a
    # setpoint round-trip, a noise-band step, document counts — and the
    # archiver read goes through the connector rather than an agent turn.
    ARCHIVER_JOB,
    # Every answer this lane reads comes off the runtime — published ports,
    # ``/proc/net/tcp``, container environment, the rendered compose document.
    TWO_SHAPE_JOB,
    # Every assertion in that file reads the store or the render; no model is
    # reached.
    GRAPH_RESEED_JOB,
]


@pytest.mark.parametrize("job", _SECRET_FREE_LANES)
def test_secret_free_lane_has_no_llm_secret(workflow: dict[str, Any], job: str) -> None:
    assert not _job_declares_secret(workflow, job, SECRET_TOKEN)


@pytest.mark.parametrize("job", _SECRET_FREE_LANES)
def test_secret_free_lane_has_no_llm_secret__mutation_adds_secret(job: str) -> None:
    """Injecting the secret into a step env must flip the check to failing."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][job]["steps"].append(
        {"name": "inject", "env": {"ALS_APG_API_KEY": "${{ secrets.ALS_APG_API_KEY }}"}}
    )
    with pytest.raises(AssertionError):
        test_secret_free_lane_has_no_llm_secret(mutated, job)


def test_secret_free_lane_has_no_llm_secret__mutation_survives_lookalike() -> None:
    """A differently-spelled secret expression must NOT trip the assertion —
    proves the check matches the real token, not any 'secrets.' substring."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][TILED_JOB]["steps"].append(
        {"name": "unrelated", "env": {"CODECOV_TOKEN": "${{ secrets.CODECOV_TOKEN }}"}}
    )
    assert not _job_declares_secret(mutated, TILED_JOB, SECRET_TOKEN)


#: Lanes pinned on the ``needs:`` entry alone.
_NEEDED_LANES = [TILED_JOB, VA_JOB, NEXTCLOUD_JOB, FULL_CHAIN_JOB]


@pytest.mark.parametrize("job", _NEEDED_LANES)
def test_all_checks_passed_needs_lane(workflow: dict[str, Any], job: str) -> None:
    """A single job nobody depends on can go red forever without blocking a merge."""
    assert job in _jobs(workflow)[GATE_JOB]["needs"]


@pytest.mark.parametrize("job", _NEEDED_LANES)
def test_all_checks_passed_needs_lane__mutation_drops_needs_entry(job: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(job)
    with pytest.raises(AssertionError):
        test_all_checks_passed_needs_lane(mutated, job)


#: The lanes pinned on both halves of the gate.
_GATED_LANES = [
    GCHAT_JOB,
    TEAMS_JOB,
    NO_MODEL_JOB,
    # These two travel with the lane: if ``bluesky-queue-e2e`` cannot go green
    # in the PR, the job, its gate entry AND its guard rows are dropped together
    # and re-landed together in the follow-up. A gate entry left behind without
    # its lane blocks every merge; a lane left behind without its gate entry is
    # unwatched.
    QUEUE_JOB,
    # Same drop-together/re-land-together discipline as the queue lane. This
    # lane is the expensive one: nobody re-reads its log once the roll-up is
    # green.
    SCAN_AGENTIC_JOB,
    ARCHIVER_JOB,
    TWO_SHAPE_JOB,
    # The only EXECUTED proof that the render zone is root-owned and the served
    # process runs as uid 1000.
    PRIVSPLIT_JOB,
    TARGET_SWITCH_JOB,
    # Its own row rather than folded into the sibling's: these two lanes can be
    # added, reverted or re-landed independently — they share only the two
    # images — and a gate entry that quietly covers "the target-switch lane"
    # without saying which one is exactly how the expensive half, five agent
    # sessions on real API budget, goes unwatched.
    TARGET_SWITCH_AGENTIC_JOB,
    GRAPH_RESEED_JOB,
    RECORD_E2E_JOB,
]


@pytest.mark.parametrize("job", _GATED_LANES)
def test_all_checks_passed_gates_on_lane(workflow: dict[str, Any], job: str) -> None:
    """``needs:`` alone is not a gate — the roll-up runs ``if: always()``, so a
    needed job that failed still lets it start; the ``check_pr_lane`` line is
    what turns the result into an exit code. Both halves are pinned."""
    assert job in _jobs(workflow)[GATE_JOB]["needs"]
    assert f"needs.{job}.result" in _gate_run_text(workflow)


@pytest.mark.parametrize("job", _GATED_LANES)
def test_all_checks_passed_gates_on_lane__mutation_drops_needs_entry(job: str) -> None:
    mutated = copy.deepcopy(_load_workflow())
    _jobs(mutated)[GATE_JOB]["needs"].remove(job)
    with pytest.raises(AssertionError):
        test_all_checks_passed_gates_on_lane(mutated, job)


@pytest.mark.parametrize("job", [lane for lane in _GATED_LANES if lane != PRIVSPLIT_JOB])
def test_all_checks_passed_gates_on_lane__mutation_drops_check_pr_lane_line(job: str) -> None:
    """The dangerous half: the ``needs`` entry stays (so the gate waits for the
    job) while the line that reads its result is gone — the lane could go red
    forever inside a green check."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, GATE_JOB, "Check all jobs status")
    kept = [line for line in step["run"].splitlines(keepends=True) if job not in line]
    assert len(kept) == len(step["run"].splitlines()) - 1, "expected exactly one line dropped"
    step["run"] = "".join(kept)
    assert job in _jobs(mutated)[GATE_JOB]["needs"]  # the needs entry survives
    with pytest.raises(AssertionError):
        test_all_checks_passed_gates_on_lane(mutated, job)

"""One removal set, two verbs — and a report shaped for each verb's reader.

``osprey reset`` and ``osprey init --reset`` run the same wipe: both plan with
``plan_reset`` and remove with ``execute_reset``, so neither can touch anything
the other would not. What differs — deliberately, and pinned here — is the
report. Standalone ``reset`` is a confirmed destruction: it prints the full
plan (WILL BE REMOVED / WILL BE KEPT) because that text is what the operator
types the confirmation against. Chained ``init --reset`` runs unconfirmed
inside another verb's phase record, so it reports the executed outcome as
condensed phase steps — what was removed, what was kept — and never pastes a
confirmation document into the middle of a run.

The equality that used to be asserted line-by-line on the printed plan is now
asserted where it actually lives: on the planned removal set both paths hand to
``execute_reset``. A transport cannot re-wrap or truncate an object.

Nothing here reaches a container runtime, and nothing here removes anything:
the runtime is the recording fake ``tests/deployment/test_reset_scoping``
already specifies, and ``execute_reset`` is stubbed out. That is not only for
speed — it is what lets both halves plan against a repo in *identical* state,
which is the precondition that makes removal-set equality a claim about the
verbs rather than about two different deployments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from osprey.cli import init_cmd
from osprey.cli.main import cli
from osprey.deployment import reset as reset_mod
from osprey.deployment.compose_generator import REPO_ID_LABEL, repo_identity
from osprey.deployment.reset import ResetPlan, RuntimeProbe, confirmation_token
from tests.deployment import test_reset_scoping as scoping

FakeRuntime = scoping.FakeRuntime

#: The deployment name both paths work on. It is the directory name, which is
#: where a repo with no ``build/`` gets its compose project name from -- so an
#: initialized-but-not-yet-built repo and a reset of that same repo resolve the
#: same project, and the plan's header line matches on both.
PROJECT = "demo"

#: The first line of every rendered full plan, and the last. Their presence is
#: what marks the standalone verb's output; their absence is what marks the
#: chained verb's.
PLAN_OPENS_WITH = "osprey reset "
PLAN_CLOSES_WITH = "Reset complete."

#: A heading deep inside the full plan body. It is a heading rather than a
#: resource name, so it is there in every full plan this file drives.
PLAN_HEADING = "WILL BE REMOVED"


def ours(repo_root: Path) -> dict[str, str]:
    """Labels a resource created by a deploy of ``repo_root`` would carry.

    ``scoping.ours`` hardcodes that module's project name; this is the same two
    labels against the project name used here.
    """
    return {
        reset_mod.COMPOSE_PROJECT_LABEL: PROJECT,
        REPO_ID_LABEL: repo_identity(repo_root.resolve()),
    }


def removal_set(plan: ResetPlan) -> dict[str, list[str]]:
    """The plan's removal set, as comparable names.

    What ``execute_reset`` iterates, reduced to the identity of each entry:
    equality of two of these is equality of what the two verbs would remove.
    """
    return {
        "containers": sorted(r.name for r in plan.containers),
        "volumes": sorted(r.name for r in plan.volumes),
        "images": sorted(r.name for r in plan.images),
        "paths": sorted(str(p) for p in plan.paths),
        "env_keys": sorted(key for block in plan.env_blocks for key in block.keys),
    }


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """Where the deployment goes. It must not exist yet -- ``init`` creates it."""
    return tmp_path / PROJECT


@pytest.fixture
def runtime(target: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    """A runtime holding one container and one volume labelled for ``target``.

    A plan with nothing on it takes the nothing-to-do path on both verbs and
    would make every comparison below pass on an empty removal list, so the
    fake is stocked: the plan under test has resources, files and a kept
    section.
    """
    fake = FakeRuntime(
        containers={f"{PROJECT}-dispatch": ours(target)},
        volumes={f"{PROJECT}_dispatch_workspace": ours(target)},
    )
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: RuntimeProbe("docker", run=fake))
    return fake


@pytest.fixture
def no_destruction(monkeypatch: pytest.MonkeyPatch) -> list[ResetPlan]:
    """Stub the removals, and record the plan each verb was about to run.

    The recorder is what makes both "the gate held" (an empty list) and "the
    two verbs remove the same things" (recorded plans) observable. Stubbing
    here rather than at the runtime is also what keeps the repo's files where
    they are, so the second path plans the same deployment the first one did.
    """
    reached: list[ResetPlan] = []

    def record(plan: ResetPlan, **kwargs: object) -> None:
        reached.append(plan)

    monkeypatch.setattr(reset_mod, "execute_reset", record)
    return reached


@pytest.fixture
def no_survivor_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence ``init``'s post-reset survivor probe.

    It builds a real :class:`RuntimeProbe` from the configured runtime rather
    than going through the seam the rest of this file injects at, so on a host
    with a live daemon it would ask a real docker about a repo that only exists
    in ``tmp_path``. What it guards is pinned in ``tests/cli/test_init_verb.py``;
    it is not this file's subject.
    """
    monkeypatch.setattr(init_cmd, "_surviving_project_resources", lambda repo_root: [])


def run_init_reset(target: Path) -> Result:
    """``osprey init <target> --reset``, through the group that installs the gate.

    Invoked on the top-level ``cli`` rather than on the ``init`` command
    directly, because the altitude gate is installed by the group callback and
    the default view is exactly what the gate makes.
    """
    return CliRunner().invoke(
        cli,
        ["init", str(target), "--preset", "hello-world", "--no-git", "--reset"],
        catch_exceptions=False,
    )


def run_reset(repo_root: Path, *args: str) -> Result:
    """``osprey reset --repo <repo_root>``, with whatever extra flags."""
    return CliRunner().invoke(
        cli, ["reset", "--repo", str(repo_root), *args], catch_exceptions=False
    )


# ---------------------------------------------------------------------------
# One removal set, two verbs
# ---------------------------------------------------------------------------


def test_both_verbs_execute_the_same_removal_set(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check
) -> None:
    """The two verbs hand ``execute_reset`` the same plan, entry for entry.

    Ordering matters: ``init --reset`` runs first and leaves the repo it just
    created, and because nothing was actually removed the ``reset`` below plans
    that same repo in that same state. Equality is therefore a statement about
    the two verbs' planning, which is the only thing that differs between the
    calls.
    """
    from_init = run_init_reset(target)
    assert from_init.exit_code == 0, from_init.output

    from_reset = run_reset(target, "-y")
    assert from_reset.exit_code == 0, from_reset.output

    assert len(no_destruction) == 2
    init_removals, reset_removals = (removal_set(plan) for plan in no_destruction)
    assert init_removals == reset_removals
    # Guard on the guard: an empty removal set would make the equality above
    # a comparison of two empty lists.
    assert init_removals["containers"] and init_removals["volumes"]


def test_the_standalone_verb_still_prints_the_full_plan(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check
) -> None:
    """``osprey reset`` keeps the confirmation document, whole.

    The full plan is what an operator confirms a destruction against, so the
    standalone verb opens with the header, carries the removal heading, and
    closes with the completion line.
    """
    run_init_reset(target)

    result = run_reset(target, "-y")

    assert result.exit_code == 0, result.output
    assert PLAN_OPENS_WITH in result.stdout
    assert PLAN_HEADING in result.stdout
    assert PLAN_CLOSES_WITH in result.stdout


def test_init_reset_reports_condensed_steps_not_the_plan(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check
) -> None:
    """The chained reset is phase steps: what went, what stayed — no document.

    The full plan is a confirmation surface, and nothing is being confirmed
    inside ``init --reset``; pasting it into the phase record is exactly the
    clutter the condensed form replaces.
    """
    result = run_init_reset(target)

    assert result.exit_code == 0, result.output
    assert PLAN_HEADING not in result.stdout
    assert PLAN_OPENS_WITH not in result.stdout
    flowed = " ".join(result.stdout.split())
    assert "removed 1 container, 1 volume" in flowed
    assert "kept the audit log" in flowed


def test_the_condensed_steps_are_on_the_default_view_not_the_log(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check, terminal_probe
) -> None:
    """The historical bug, re-pinned against the condensed form.

    Routed through ``logger.key_info`` the old plan was an INFO record, which
    the altitude gate drops — so the destructive verb that never asks also
    never said what it was taking. The condensed outcome is phase record now:
    on stdout, and not painted by the log handler.
    """
    result = run_init_reset(target)

    assert result.exit_code == 0, result.output
    assert "removed 1 container, 1 volume" in " ".join(result.stdout.split())
    assert "removed 1 container" not in terminal_probe.rendered_text
    # Witness: the record sink is live, so the absence above is an observation
    # rather than an empty buffer agreeing with everything.
    logging.getLogger("osprey.test.reset_plan_parity").info("plan parity witness")
    assert any("plan parity witness" in message for message in terminal_probe.messages)


def test_an_empty_plan_reports_nothing_to_do_and_removes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_destruction
) -> None:
    """``reset_for_reinit`` on a repo with nothing to take does not execute.

    A fresh ``init`` always leaves *something* on the plan (the state
    directories it just made), so this is the direct contract test for the
    branch — the phase note it feeds is pinned on the CLI below.
    """
    repo = tmp_path / PROJECT
    repo.mkdir()
    fake = FakeRuntime()

    outcome = reset_mod.reset_for_reinit(repo, probe=RuntimeProbe("docker", run=fake))

    assert outcome is reset_mod.ResetOutcome.NOTHING_TO_DO
    assert no_destruction == []


def test_a_reset_with_nothing_to_remove_says_so_on_the_phase_line(
    target: Path, monkeypatch: pytest.MonkeyPatch, no_survivor_check
) -> None:
    """An empty plan closes the phase with its note instead of printing a plan."""
    monkeypatch.setattr(
        reset_mod, "reset_for_reinit", lambda repo_root, **kw: reset_mod.ResetOutcome.NOTHING_TO_DO
    )

    result = run_init_reset(target)

    assert result.exit_code == 0, result.output
    flowed = " ".join(result.stdout.split())
    assert "nothing from a previous run to remove" in flowed
    assert PLAN_HEADING not in result.stdout


# ---------------------------------------------------------------------------
# The typed gate, on both paths
# ---------------------------------------------------------------------------


def test_the_typed_gate_still_refuses_a_wrong_answer(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check, monkeypatch
) -> None:
    """``osprey reset`` without ``-y`` still destroys nothing on a mistyped token."""
    run_init_reset(target)
    no_destruction.clear()
    monkeypatch.setattr("builtins.input", lambda prompt: "not-the-token")

    result = run_reset(target)

    assert result.exit_code == 1
    assert "confirmation did not match" in result.stdout
    assert no_destruction == [], "a declined reset must not reach the removals"


def test_the_typed_gate_still_accepts_the_token(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check, monkeypatch
) -> None:
    """And the right token still gets through it, on the same path."""
    run_init_reset(target)
    no_destruction.clear()
    monkeypatch.setattr("builtins.input", lambda prompt: confirmation_token(target))

    result = run_reset(target)

    assert result.exit_code == 0, result.output
    assert len(no_destruction) == 1


def test_init_reset_waives_the_prompt_rather_than_answering_it(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check, monkeypatch
) -> None:
    """``--reset`` is the unattended path: nothing may ask.

    Worth pinning next to the two above: the gate is *waived* here, by a flag
    the operator typed, and not silently answered on their behalf somewhere in
    the plumbing. A prompt reached by an unattended run would hang it.
    """

    def refuse(prompt: str) -> str:
        raise AssertionError(f"init --reset must not prompt, but asked: {prompt!r}")

    monkeypatch.setattr("builtins.input", refuse)

    result = run_init_reset(target)

    assert result.exit_code == 0, result.output
    assert len(no_destruction) == 1

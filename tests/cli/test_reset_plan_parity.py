"""The destruction plan reads the same whichever verb is about to destroy.

``osprey reset`` and ``osprey init --reset`` run the same wipe through the same
``reset_deployment`` call, but until now they handed it two different printers:
``reset`` passed the CLI's echo path, ``init`` passed ``logger.key_info``. An
INFO record is what the altitude gate drops on a normal run, so the plan
listing every container, volume and file about to go simply did not appear on
the ``init --reset`` path -- on the one command in OSPREY that destroys data
without asking, because ``--reset`` means unattended.

Both call sites now pass :func:`osprey.cli.output.report`, and this file pins
that as an equality rather than as two separate presence checks: the same lines
in the same order, sliced out of each path's stdout by the SAME rule. A
transport that re-wrapped at the console width, swallowed a blank separator or
dropped the trailing summary would satisfy every substring assertion ever
written about a plan and fail the comparison below.

Nothing here reaches a container runtime, and nothing here removes anything:
the runtime is the recording fake ``tests/deployment/test_reset_scoping``
already specifies, and ``execute_reset`` is stubbed out. That is not only for
speed -- it is what lets both halves plan against a repo in *identical* state,
which is the precondition that makes line-identity a claim about the transport
rather than about two different deployments.
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
from osprey.deployment.reset import RuntimeProbe, confirmation_token
from tests.deployment import test_reset_scoping as scoping

FakeRuntime = scoping.FakeRuntime

#: The deployment name both paths work on. It is the directory name, which is
#: where a repo with no ``build/`` gets its compose project name from -- so an
#: initialized-but-not-yet-built repo and a reset of that same repo resolve the
#: same project, and the plan's header line matches on both.
PROJECT = "demo"

#: The first line of every rendered plan, and the last. The slice between them
#: is the plan; everything outside it belongs to whichever verb was running.
PLAN_OPENS_WITH = "osprey reset "
PLAN_CLOSES_WITH = "Reset complete."

#: A line deep inside the plan body, used where the question is presence rather
#: than equality. It is a heading rather than a resource name, so it is there in
#: every plan this file drives.
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


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """Where the deployment goes. It must not exist yet -- ``init`` creates it."""
    return tmp_path / PROJECT


@pytest.fixture
def runtime(target: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    """A runtime holding one container and one volume labelled for ``target``.

    A plan with nothing on it returns early with "nothing to reset" and would
    make the comparison below pass on an empty removal list, so the fake is
    stocked: the plan under test has resources, files and a kept section.
    """
    fake = FakeRuntime(
        containers={f"{PROJECT}-dispatch": ours(target)},
        volumes={f"{PROJECT}_dispatch_workspace": ours(target)},
    )
    monkeypatch.setattr(reset_mod, "_default_probe", lambda root: RuntimeProbe("docker", run=fake))
    return fake


@pytest.fixture
def no_destruction(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Stub the removals, and record that they were reached.

    The recorder is what makes "the gate held" observable: a declined reset must
    leave this list empty. Stubbing here rather than at the runtime is also what
    keeps the repo's files where they are, so the second path plans the same
    deployment the first one did.
    """
    reached: list[object] = []

    def record(plan: object, **kwargs: object) -> None:
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


def plan_lines(stdout: str) -> list[str]:
    """The rendered destruction plan, sliced out of a verb's whole output.

    The same rule is applied to both paths -- open at the header line, close at
    the completion line, keep everything between. Anything the plan lost in
    transport is lost from the slice too, which is the point: the slice is not
    allowed to be a filter that agrees with itself.
    """
    lines = stdout.splitlines()
    opens = [i for i, line in enumerate(lines) if line.startswith(PLAN_OPENS_WITH)]
    closes = [i for i, line in enumerate(lines) if line.startswith(PLAN_CLOSES_WITH)]
    assert opens, f"no plan header in:\n{stdout}"
    assert closes, f"no plan completion line in:\n{stdout}"
    return lines[opens[0] : closes[-1] + 1]


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
# FR4: one plan, two verbs
# ---------------------------------------------------------------------------


def test_the_plan_is_line_identical_on_both_paths(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check
) -> None:
    """The two verbs print the same plan, line for line and in the same order.

    Ordering matters: ``init --reset`` runs first and leaves the repo it just
    created, and because nothing was actually removed the ``reset`` below plans
    that same repo in that same state. Equality is therefore a statement about
    the printer, which is the only thing that differs between the two calls.
    """
    from_init = run_init_reset(target)
    assert from_init.exit_code == 0, from_init.output

    from_reset = run_reset(target, "-y")
    assert from_reset.exit_code == 0, from_reset.output

    init_plan = plan_lines(from_init.stdout)
    reset_plan = plan_lines(from_reset.stdout)

    assert init_plan == reset_plan
    # Guard on the guard. A plan of two lines, or one whose longest line is
    # comfortably inside a console width, would let a truncating or re-wrapping
    # transport through the equality above without anyone noticing.
    assert len(init_plan) > 20
    assert max(len(line) for line in init_plan) > 100


def test_the_reset_verbs_whole_output_is_the_plan(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check
) -> None:
    """The slice is not dropping anything on the side where it can be checked.

    ``osprey reset`` prints its plan and nothing else, so on that path the
    sliced plan must be the entire stdout. That is what stops
    :func:`plan_lines` from quietly becoming a filter that makes any two
    outputs agree.
    """
    run_init_reset(target)

    result = run_reset(target, "-y")

    assert result.exit_code == 0, result.output
    assert "".join(f"{line}\n" for line in plan_lines(result.stdout)) == result.stdout


def test_the_plan_is_printed_on_the_default_init_reset_view(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check, terminal_probe
) -> None:
    """The bug: on a normal run the plan reached the operator on neither channel.

    Routed through ``logger.key_info`` the plan was an INFO record, which the
    altitude gate drops -- so the destructive verb that never asks also never
    said what it was taking. Both halves are asserted, because either alone
    passes for the wrong reason: the plan is on stdout NOW, and it is no longer
    a log record at all (an absence the gate would have faked either way, so it
    is checked against ``records``, which the gate never touches).
    """
    result = run_init_reset(target)

    assert result.exit_code == 0, result.output
    assert PLAN_HEADING in result.stdout
    assert not any(PLAN_HEADING in message for message in terminal_probe.messages)
    # Witness: the record sink is live, so the absence above is an observation
    # rather than an empty buffer agreeing with everything.
    logging.getLogger("osprey.test.reset_plan_parity").info("plan parity witness")
    assert any("plan parity witness" in message for message in terminal_probe.messages)


def test_the_plan_survives_the_verbose_reporter_on_both_paths(
    target: Path, runtime: FakeRuntime, no_destruction, no_survivor_check
) -> None:
    """``-v`` installs the NullReporter, which swallows the phase record only.

    The plan is echo-class on both paths now, so the flag that turns the run
    into a transcript must not be the flag that loses the destruction list.
    """
    from_init = CliRunner().invoke(
        cli,
        ["-v", "init", str(target), "--preset", "hello-world", "--no-git", "--reset"],
        catch_exceptions=False,
    )
    assert from_init.exit_code == 0, from_init.output

    from_reset = run_reset(target, "-y")

    assert PLAN_HEADING in from_init.stdout
    assert plan_lines(from_init.stdout) == plan_lines(from_reset.stdout)


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
    """``--reset`` is the unattended path: it passes ``assume_yes``, so nothing asks.

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

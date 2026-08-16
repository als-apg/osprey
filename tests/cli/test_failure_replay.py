"""What the lifecycle verbs print when a captured child fails or is interrupted.

Capturing a subprocess to a spool file is only honest if the failure path gives
that output back. Two shapes are asserted here, both with a real child process
writing a real spool:

* a child that exits non-zero — its spool is replayed **in full**, beneath the
  phase's own ✗ line and **before** the verb's error message, exactly once;
* a run interrupted by SIGINT — the partial spool is **named in one line** and
  never replayed, because an interrupt is not a diagnosis and the operator did
  not ask to read a half-written log.

Neither of those is wired at the verb's ``except`` blocks: the replay comes from
``Phase.__exit__``, which fails the open phase with whatever spool the capture
helper last recorded on it. That is what makes it work for the paths whose
failure is not a ``CapturedProcessError`` at all. The verb handlers' part is the
negative one — not replaying a second time, and not printing a second failure
marker over the phase's.

Output is read with ``capfd`` and the verbs are driven through ``cli.main``
rather than ``CliRunner``: the question is what reached the terminal's file
descriptors while a real child was writing to a file beside them, and a runner
that swaps ``sys.stdout`` for a buffer cannot answer it.
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path

import click
import pytest

from osprey.cli.main import cli

#: Written by the child into its spool. Deliberately absent from the argv that
#: produces it — the error message quotes the command, and a marker that also
#: appeared there would make "the spool was printed once" unaskable.
POISON = "the-child-said-this"


def run_verb(*args: str) -> int:
    """Run one verb in this process and return the exit status a shell would see.

    ``standalone_mode=False`` keeps Click from swallowing the abort into
    ``sys.exit``, so the status is decided here instead of by an exception that
    would take the test session with it.
    """
    try:
        cli.main(args=list(args), prog_name="osprey", standalone_mode=False)
    except click.Abort:
        return 1
    except SystemExit as exit_:  # `down` propagates the runtime's own status
        return int(exit_.code or 0)
    return 0


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A directory standing in for a built deployment repo."""
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text("{}\n")
    (tmp_path / "poison.txt").write_text(f"{POISON}\n")
    return tmp_path


@pytest.fixture
def wide(monkeypatch):
    """Take Rich's 80-column default out of the assertions.

    The reporter soft-wraps, but the refusal paths print through the plain
    console, and a wrapped ✗ line would be a test about terminal width.
    """
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def stubs(monkeypatch, repo: Path):
    """Stub everything `up`/`down` reach outside the CLI layer."""
    from osprey.cli import deploy_cmd, repo_resolver
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(repo_resolver, "find_repo_root", lambda start=None: repo)
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "ensure_repo_env", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "load_project_config", lambda *a, **k: {})
    monkeypatch.setattr(
        container_lifecycle, "as_built_config_path", lambda root: repo / "build" / "config.yml"
    )


@pytest.fixture
def errors_on_stdout(monkeypatch):
    """Point the renderer's trouble stream at stdout, so its order against the replay is visible.

    The reporter prints to stdout and ``_abort`` renders to stderr. Two streams
    carry no order relative to one another, and "the spool came first" is
    exactly what has to be asserted — so for the duration of the test the stderr
    console is a stdout-bound one, and the failure lands on the stream the
    replay uses. Production behaviour is untouched.

    Redirects the console the module already owns rather than substituting a
    fresh one. ``set_theme`` re-themes ``styles.err_console`` in place by
    popping the theme it pushed last, and a substitute built here would have
    nothing on its stack to pop — the theme initialisation that runs on every
    ``cli.main`` would die on it. Rich resolves ``_file or sys.stderr`` per
    write, so setting it redirects and clearing it back to ``None`` restores the
    dynamic lookup exactly as it was.

    The redirect target is a proxy rather than ``sys.stdout`` itself, because
    pytest's capture swaps that object out from under a long-lived reference and
    closes the one captured here.
    """
    import sys

    from osprey.cli import styles

    class _LiveStdout:
        """Whatever ``sys.stdout`` is at the moment of the write."""

        def __getattr__(self, name: str):
            return getattr(sys.stdout, name)

    monkeypatch.setattr(styles.err_console, "_file", _LiveStdout())


def poisoned(*args, **kwargs) -> None:
    """Stand-in for the verb's real work: one captured child that exits 3."""
    from osprey.deployment.subprocess_capture import run_captured

    repo_root = Path(args[0])
    run_captured(
        ["sh", "-c", "cat poison.txt; exit 3"],
        cwd=repo_root,
        spool_name="probe",
        repo_root=repo_root,
    )


def spools(repo: Path) -> list[Path]:
    """The spool files this run left in the repo."""
    return sorted((repo / "var" / "logs").glob("probe-*.log"))


# ---------------------------------------------------------------------------
# A captured child that exits non-zero
# ---------------------------------------------------------------------------


def test_a_failed_captured_step_replays_its_spool_before_the_error(
    monkeypatch, capfd, wide, stubs, errors_on_stdout, repo
):
    """The order is the whole point.

    An error message that arrives before the output it is about leaves the
    operator scrolling back for the reason, which is the state the capture was
    supposed to remove.
    """
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "down_deployment", poisoned)

    status = run_verb("down")

    assert status != 0
    out = capfd.readouterr().out
    assert POISON in out, out
    assert out.index("✗ Stopping") < out.index(POISON)
    assert out.index(POISON) < out.index("✗ Could not stop this deployment")


def test_the_error_message_names_the_spool_it_replayed(monkeypatch, capfd, wide, stubs, repo):
    """The replay is for now; the path is for the operator who comes back later.

    Read off stderr, where the failure is rendered. This used to read the log
    record instead, because the log console wrapped a long path across lines and
    an assertion on it would have been an assertion about terminal width; the
    renderer prints unwrapped, so the path arrives whole and can be asserted
    where the operator actually reads it.
    """
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "down_deployment", poisoned)

    status = run_verb("down")

    assert status != 0
    spool = spools(repo)[0]
    assert str(spool) in capfd.readouterr().err


def test_a_failed_captured_step_replays_its_spool_only_once(
    monkeypatch, capfd, wide, stubs, errors_on_stdout, repo
):
    """The phase teardown replays. A handler that replayed too would print the
    whole log twice, which for a compose build is thousands of duplicated
    lines — and would read as two separate failures."""
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "down_deployment", poisoned)

    status = run_verb("down")

    assert status != 0
    assert capfd.readouterr().out.count(POISON) == 1


def test_a_start_that_fails_on_a_captured_child_replays_it_too(
    monkeypatch, capfd, wide, stubs, errors_on_stdout, repo
):
    """Same property on the other verb, whose phase is opened separately."""
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "up_as_built", poisoned)

    status = run_verb("up", "-d")

    assert status != 0
    out = capfd.readouterr().out
    assert out.count(POISON) == 1
    assert out.index("✗ Starting") < out.index(POISON)
    assert out.index(POISON) < out.index("✗ Deployment failed")


# ---------------------------------------------------------------------------
# A captured child interrupted with SIGINT
# ---------------------------------------------------------------------------


@pytest.fixture
def alarm_sigint():
    """Deliver a real SIGINT to this process shortly after the child starts.

    An alarm rather than a helper thread: ``SIGALRM`` reaches the main thread,
    which is the one parked in the child's ``waitpid``, so the interrupt lands
    inside the capture instead of after it. What the handler then raises is the
    genuine signal — the ``KeyboardInterrupt`` comes from Python's own SIGINT
    handling, not from the test.
    """

    def fire(_signum, _frame):
        signal.raise_signal(signal.SIGINT)

    previous = signal.signal(signal.SIGALRM, fire)
    try:
        yield lambda seconds=0.4: signal.setitimer(signal.ITIMER_REAL, seconds)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX signal delivery")
def test_an_interrupted_capture_names_its_partial_spool_and_does_not_replay_it(
    monkeypatch, capfd, wide, stubs, alarm_sigint, repo
):
    """Ctrl-C is a decision, not a diagnosis.

    The operator who stops a deploy has not asked to read the log of the half
    of it that ran; they have asked for it to stop. So the spool is named in one
    line and left on disk, and replaying it here would bury the one thing worth
    saying under the output of a run nobody is waiting on any more.
    """
    from osprey.deployment import container_lifecycle
    from osprey.deployment.subprocess_capture import run_captured

    def interrupted(*args, **kwargs) -> None:
        repo_root = Path(args[0])
        alarm_sigint()
        run_captured(
            ["sh", "-c", "cat poison.txt; sleep 30"],
            cwd=repo_root,
            spool_name="probe",
            repo_root=repo_root,
        )

    monkeypatch.setattr(container_lifecycle, "down_deployment", interrupted)

    status = run_verb("down")

    assert status != 0
    out = capfd.readouterr().out

    # The child got far enough to write, and what it wrote survived the kill.
    spool = spools(repo)[0]
    assert POISON in spool.read_text()

    # One line, naming that file, from the phase's own teardown.
    named = [line for line in out.splitlines() if "partial output:" in line]
    assert len(named) == 1, out
    assert named[0].lstrip().startswith("⚠ Stopping")
    assert str(spool) in named[0]

    # And no replay: the content stayed in the file it was written to.
    assert POISON not in out
    assert f"--- {spool} ---" not in out


# ---------------------------------------------------------------------------
# One failure marker per failure
# ---------------------------------------------------------------------------


def test_a_dev_refusal_prints_one_failure_marker(monkeypatch, capfd, wide, stubs, repo):
    """The ✗ belongs to the phase that stopped, and there is one of those.

    The handler's job is the reason and the remedy; a second ✗ in front of them
    reads as a second thing having gone wrong. So the phase owns the only mark,
    on stdout, and the handler's block continues it markless on the trouble
    stream — one failure to read, two records to anything reading the streams
    apart.
    """
    from osprey.deployment import container_lifecycle
    from osprey.deployment.errors import DevModeUnavailableError

    def refuse(*args, **kwargs):
        raise DevModeUnavailableError("this render carries no wheel", "osprey build --dev")

    monkeypatch.setattr(container_lifecycle, "up_as_built", refuse)

    status = run_verb("up", "-d", "--dev")

    assert status != 0
    captured = capfd.readouterr()
    out, err = captured.out, captured.err
    assert out.count("✗") == 1, out  # the phase's mark, and only it
    assert err.count("✗") == 0, err  # the continuation adds none
    # The positive half: without it the counts above pass on an empty stream.
    # Summary, reason and remedy are the three lines of the one trouble shape.
    assert "--dev cannot be honored" in err
    assert "this render carries no wheel" in err
    assert "osprey build --dev" in err


def test_an_unexpected_error_wears_its_own_marker(monkeypatch, capfd, wide, stubs, repo):
    """A refusal continues the phase's mark; an unexpected error does not.

    The two cases differ in what the handler has to say. A refusal restates the
    failure the phase already marked — same fact, second voice — so it continues
    that line markless. An unexpected error is a SECOND fact: the phase says
    which step stopped, and the handler says that the verb hit something it does
    not have a refusal for. Two facts, two marks, and this pins them landing on
    the two streams they belong to rather than doubling up on one.
    """
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(container_lifecycle, "down_deployment", poisoned)

    status = run_verb("down")

    assert status != 0
    captured = capfd.readouterr()
    out, err = captured.out, captured.err
    assert out.count("✗") == 1, out  # the phase's, on the reporter's stream
    assert "✗ Stopping" in out
    assert err.count("✗") == 1, err  # the handler's, on the trouble stream
    assert "✗ Could not stop this deployment" in err


def test_a_captured_build_failure_is_not_reported_as_unexpected(monkeypatch, caplog, tmp_path):
    """A child that exited non-zero is a build failure with a log to read.

    The catch-all it used to land in called it "Unexpected error" and put a ✗ in
    front of it — a second marker over the phase's own, under a heading that
    tells the operator nothing about what to do next.
    """
    from osprey.cli import build_cmd, build_profile
    from osprey.deployment.errors import CapturedProcessError

    spool = tmp_path / "var" / "logs" / "compose-config-20260101-000000.log"

    def boom(*args, **kwargs):
        raise CapturedProcessError(["docker", "compose", "config"], 1, spool)

    monkeypatch.setattr(build_cmd, "find_repo_root", lambda repo=None: tmp_path)
    monkeypatch.setattr(build_profile, "resolve_build_document", boom)

    with caplog.at_level(logging.ERROR), pytest.raises(click.Abort):
        build_cmd._build_repo(
            tmp_path, stream=False, skip_lifecycle=True, skip_deps=True, runtime_root=None
        )

    assert "Build failed:" in caplog.text
    assert str(spool) in caplog.text
    assert "Unexpected error" not in caplog.text
    assert "✗" not in caplog.text

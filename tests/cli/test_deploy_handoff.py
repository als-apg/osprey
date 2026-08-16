"""The deploy verbs' interactive prompt asks its question on a free terminal.

``ensure_repo_env`` is the one place a start verb stops to ask something: with
no ``.env`` and the provider's key exported, it offers to seed the file. The
reporter is already installed and already drawing by then, and a live region
repaints on its own schedule — over the question, while the operator is
reading it, in a terminal where the answer is being typed.

So the prompt runs inside ``suspended()``: the region comes down for exactly
as long as the question is on screen, and goes back up however the block ends.
These tests pin the call site, not the seam — that the prompt is wrapped at
all, that the wrap is a no-op where nothing is rendering, and that a prompt
the operator abandons still gives the terminal back.
"""

from __future__ import annotations

import stat
from pathlib import Path

import click
import pytest

from osprey.cli import deploy_cmd, phase_reporter
from osprey.cli.phase_reporter import PhaseReporter, current_reporter, install_reporter

#: The provider whose key the seed offer harvests, and the variable it lives in.
_PROVIDER = "anthropic"
_SECRET_VAR = "ANTHROPIC_API_KEY"


@pytest.fixture(autouse=True)
def parked_monitor(monkeypatch):
    """Park the monitor's tick, so no thread of it lands mid-test.

    A reporter that is rendering runs a real thread on the real clock. One
    tick arriving while these tests are counting seam calls appends work
    nobody asked about and silences the assertions after it.
    """
    monkeypatch.setattr(phase_reporter, "_MONITOR_INTERVAL", 3600.0)


@pytest.fixture(autouse=True)
def restore_installed_reporter():
    """Leave the module singleton exactly as the test found it."""
    original = current_reporter()
    yield
    install_reporter(original)


class SeamRecorder(PhaseReporter):
    """A reporter that IS rendering, recording every seam call in order.

    ``is_rendering`` is what decides whether ``suspended()`` has anything to
    do, so claiming it here is what makes the real seam act — the block is
    exercised, not stubbed out.
    """

    def __init__(self) -> None:
        super().__init__(color=False)
        self.events: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        """Swallow the line: only the seam calls are being counted."""

    def is_rendering(self) -> bool:
        return True

    def stop_rendering(self) -> None:
        self.events.append("stop")
        super().stop_rendering()

    def start_rendering(self) -> None:
        self.events.append("start")
        super().start_rendering()


def _repo(tmp_path: Path) -> Path:
    """A deployment repo with no ``.env`` — the state that triggers the offer."""
    root = tmp_path / "repo"
    root.mkdir()
    return root


def _config() -> dict:
    return {"claude_code": {"provider": _PROVIDER}}


@pytest.fixture
def offered(monkeypatch):
    """Everything ``ensure_repo_env`` needs to reach its prompt."""
    monkeypatch.setenv(_SECRET_VAR, "sk-from-the-shell")
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: True)


def _answer(monkeypatch, reporter, answer=True):
    """Answer the prompt, recording where in the seam it was asked."""

    def _confirm(text, **kwargs):
        reporter.events.append("prompt")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(deploy_cmd.click, "confirm", _confirm)


def test_the_seed_prompt_is_asked_with_the_terminal_given_back(offered, monkeypatch, tmp_path):
    """The question and the region want the same rows of the same terminal.

    Ordering is the whole assertion: a prompt asked before the region came
    down, or after it went back up, is the bug — and a test that only checked
    that both happened would pass either way.
    """
    reporter = SeamRecorder()
    install_reporter(reporter)
    _answer(monkeypatch, reporter)

    deploy_cmd.ensure_repo_env(_repo(tmp_path), _config())

    assert reporter.events == ["stop", "prompt", "start"]


def test_an_accepted_offer_still_seeds_the_file(offered, monkeypatch, tmp_path):
    """The wrap changes where the question is asked, and nothing else.

    The seed is the point of the prompt: a refactor that suspended the
    reporter but stopped acting on the answer would leave the operator
    agreeing to something that never happened.
    """
    reporter = SeamRecorder()
    install_reporter(reporter)
    _answer(monkeypatch, reporter)
    repo = _repo(tmp_path)

    deploy_cmd.ensure_repo_env(repo, _config())

    env_path = repo / ".env"
    assert f"{_SECRET_VAR}=sk-from-the-shell" in env_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_a_declined_offer_refuses_with_the_terminal_back(offered, monkeypatch, tmp_path):
    """The refusal that follows a "no" is a plain message, printed after the
    region is back under the reporter's own control rather than into the
    middle of a frame."""
    reporter = SeamRecorder()
    install_reporter(reporter)
    _answer(monkeypatch, reporter, answer=False)
    repo = _repo(tmp_path)

    with pytest.raises(click.Abort):
        deploy_cmd.ensure_repo_env(repo, _config())

    assert reporter.events == ["stop", "prompt", "start"]
    assert not (repo / ".env").exists()


def test_an_abandoned_prompt_still_gives_the_terminal_back(offered, monkeypatch, tmp_path):
    """A prompt the operator Ctrl-Cs is exactly the one that must not leave
    the rest of the run rendering nothing — so the wrap has to be a ``with``
    block, not a stop-then-start pair around the call."""
    reporter = SeamRecorder()
    install_reporter(reporter)
    _answer(monkeypatch, reporter, answer=click.Abort())

    with pytest.raises(click.Abort):
        deploy_cmd.ensure_repo_env(_repo(tmp_path), _config())

    assert reporter.events == ["stop", "prompt", "start"]


def test_the_prompt_path_is_a_no_op_with_nothing_rendering(offered, monkeypatch, tmp_path):
    """Off a terminal, under ``--verbose``, or from a library caller there is
    no region to take down. The call site asks no questions about which
    reporter it got, so the seam has to put back exactly what it took away —
    here, nothing at all.
    """
    reporter = SeamRecorder()
    monkeypatch.setattr(SeamRecorder, "is_rendering", lambda self: False)
    install_reporter(reporter)
    _answer(monkeypatch, reporter)
    repo = _repo(tmp_path)

    deploy_cmd.ensure_repo_env(repo, _config())

    assert reporter.events == ["prompt"], "a reporter drawing nothing was restarted"
    assert (repo / ".env").exists()


def test_the_default_reporter_survives_the_prompt(offered, monkeypatch, tmp_path):
    """The seams live on the base class, so the quiet default answers them too.

    This is the shape every non-verb caller reaches the prompt with — a
    missing definition here is an ``AttributeError`` in production, on the one
    path that exists to keep a deployment from starting with empty secrets.
    """
    monkeypatch.setattr(deploy_cmd.click, "confirm", lambda text, **kwargs: True)
    repo = _repo(tmp_path)

    deploy_cmd.ensure_repo_env(repo, _config())

    assert (repo / ".env").exists()

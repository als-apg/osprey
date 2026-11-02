"""The landing-page demo: its storyboard and the probes only it needs.

The storyboard asks for a 3D plot, rotates it, asks for a correlation plot and
posts it to the logbook through the one approval prompt the take answers. The
generic beats come from :mod:`~docs.screenshots.video_beats`; the rest are
custom beats on :class:`Landing`, one per take, which carries what an earlier
beat found (the plot ids, the cards already opened, the approved call) to the
later ones.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from docs.screenshots import (
    capture,
    video_check_approval,
    video_check_plot,
    video_check_stall,
    video_overlay,
    video_probes,
    video_take,
)
from docs.screenshots.video_beats import Beat, focus, hold, prompt
from docs.screenshots.video_probes import ProbeFailed
from docs.screenshots.video_take import Take

PROMPT_PLOT = (
    "Make a 3D scatter plot of the horizontal positions of the first three BPMs "
    "in the storage ring over the last 24 hours"
)
PROMPT_CORR = (
    "Now give me a full correlation plot of all horizontal BPMs in the first sector "
    "together with the vacuum"
)
PROMPT_POST = "Post this plot to the logbook"

# The one tool call the storyboard approves. Any other approval prompt means
# the agent went somewhere the video must not show, so the take is discarded.
APPROVED_TOOL = "entry_create"

PLOT_BUDGET_MS = 600_000
APPROVAL_BUDGET_MS = 180_000
# Without a draft this long after the approving Enter, the keypress may have
# been swallowed; a single guarded retry Enter follows.
DRAFT_RETRY_AFTER_MS = 15_000
DRAFT_BUDGET_MS = 90_000
OPEN_PLOT_BUDGET_MS = 30_000
# Each card that lands while the agent works is opened in real time: its row
# must show in the hub sidebar's artifact list, then its preview render in the
# artifacts viewer, then it is held on screen.
CARD_BUDGET_MS = 10_000
PREVIEW_BUDGET_MS = 15_000
OPEN_HOLD_MS = 1_500
# The correlation plot is held on screen before the post prompt is typed.
CORR_HOLD_MS = 3_000
# How long the approved entry_create call may take to appear in the transcript.
ATTACH_BUDGET_MS = 10_000

APPROVAL_HOLD_MS = 2_000
CLOSE_HOLD_MS = 3_000

# The rotate beat must turn the camera at least this far around the z axis.
MIN_AZIMUTH_CHANGE_DEG = 90.0
# Each drag sweeps this share of the plot width; drags repeat until the
# camera has turned far enough.
DRAG_FRACTION = 0.5
MAX_DRAGS = 3
# Each step is a browser round trip played in real time, so the glide stays
# short; the camera is read once the scene has drawn the last step.
DRAG_STEPS = 20
DRAG_SETTLE_MS = 300
# Real time the rotate beat may take before the take is discarded: it plays
# unsped, and a browser crawling under load would stretch the video by minutes.
ROTATE_BUDGET_S = 15.0

ARTIFACT_FETCH_TIMEOUT_S = 10.0


def _fetch_html(artifact_port: int, path: str, secret: str | None = None) -> str:
    """Return the artifact server's file at ``path`` as text."""
    request = capture.artifact_request(artifact_port, path, secret)
    with urllib.request.urlopen(request, timeout=ARTIFACT_FETCH_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "replace")


def _has_tool_result(transcript: Path | None, tool_use_id: str) -> bool:
    """Whether the transcript holds a ``tool_result`` for ``tool_use_id``.

    Results are either top-level ``tool_result`` entries or blocks inside a
    user message's ``content`` list. An unreadable transcript counts as having
    one, so the guarded retry never fires on missing evidence.
    """
    if transcript is None:
        return True
    try:
        lines = Path(transcript).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return True
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "tool_result" and entry.get("tool_use_id") == tool_use_id:
            return True
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") == tool_use_id
                ):
                    return True
    return False


def _tool_input(transcript: Path | None, tool_use_id: str | None) -> dict | None:
    """The ``input`` of the ``tool_use`` block ``tool_use_id`` in the transcript, if any."""
    if transcript is None or tool_use_id is None:
        return None
    try:
        lines = Path(transcript).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        message = entry.get("message") if isinstance(entry, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == tool_use_id
            ):
                tool_input = block.get("input")
                return tool_input if isinstance(tool_input, dict) else {}
    return None


class _PreviewReplaced(Exception):
    """The plot preview being read is mid-swap: it has no on-screen box."""


def _on_screen_box(frame: Any) -> dict:
    box = frame.frame_element().bounding_box()
    if not box:
        raise _PreviewReplaced
    return box


class Landing:
    """The landing storyboard's state on one :class:`~docs.screenshots.video_take.Take`."""

    def __init__(self, take: Take) -> None:
        self.take = take
        self.ctx = take.ctx
        self.page = take.page
        self.timeline = take.timeline
        self.opened: set[str] = set()
        # Per agent phase ("plot", "correlate"): the result found so far and the
        # stall count shared by the phase's waits and its card openings.
        self.found: dict[str, str | None] = {}
        self.stalls: dict[str, video_check_stall.StallWatch] = {}
        self.phase_stall: video_check_stall.StallWatch | None = None
        # The approval-log mark taken before each prompt, by prompt.
        self.marks: dict[str, datetime] = {}
        self.plot_frame: Any = None
        self.corr_id: str | None = None
        self.approved: video_check_approval.Ask | None = None

    @classmethod
    def of(cls, take: Take) -> Landing:
        """The take's landing state, made on first use."""
        if "landing" not in take.memo:
            take.memo["landing"] = cls(take)
        return take.memo["landing"]

    # The engine's plumbing, under the names the beats below use.

    @property
    def transcript(self) -> Path | None:
        return self.take.transcript

    @property
    def hook_log(self) -> Callable[[datetime], list]:
        return self.take.hook_log

    def turn(self) -> str:
        return self.take.turn()

    def watch(self, *args: Any, **kwargs: Any) -> Callable[[], None]:
        return self.take.watch(*args, **kwargs)

    def click(self, locator: Any) -> None:
        self.take.click(locator)

    def press_enter(self, step_name: str) -> None:
        self.take.press_enter(step_name)

    def step(self, name: str, caption: str, fast_forward: bool = False) -> Any:
        return self.take.step(name, caption, fast_forward)

    # -- probes ------------------------------------------------------------

    def asks(self, mark: datetime) -> list[video_check_approval.Ask]:
        """Asks since ``mark``; fails the take on any unnamed or foreign ask."""
        records = self.hook_log(mark)
        asks = video_check_approval.asks_in(records)
        if len(video_check_approval.ask_records(records)) > len(asks):
            raise ProbeFailed("approve", "an approval prompt without a tool name was logged")
        foreign = [a.name for a in asks if a.name != APPROVED_TOOL]
        if foreign:
            raise ProbeFailed("approve", f"unexpected approval prompt for {', '.join(foreign)}")
        return asks

    def plot_id(self) -> str | None:
        port = self.ctx.stack.artifact_port
        secret = self.ctx.stack.operator_secret
        return video_check_plot.find_plot_artifact(
            capture.fetch_artifacts(port, secret),
            self.ctx.before_ids,
            lambda path: _fetch_html(port, path, secret),
        )

    def arrivals(self) -> list[dict]:
        """Cards new since the prompt and not yet opened, oldest first."""
        fresh = [
            a
            for a in capture.fetch_artifacts(
                self.ctx.stack.artifact_port, self.ctx.stack.operator_secret
            )
            if a.get("id") and a["id"] not in self.ctx.before_ids and a["id"] not in self.opened
        ]
        return sorted(fresh, key=lambda a: str(a.get("timestamp") or ""))

    def wait_for_result(
        self,
        step_name: str,
        ask_mark: datetime,
        find: Callable[[], str | None],
        what: str,
    ) -> tuple[str, Any]:
        """Wait while the agent works, until a new card lands or its result is done.

        Returns ``("card", artifact)`` for the oldest card not yet opened, or
        ``("result", artifact_id)`` once the turn has ended and ``find`` names
        the new ``what``. One stall count spans every call of a phase, so
        opening cards never resets it.
        """
        if step_name not in self.stalls:
            self.stalls[step_name] = video_check_stall.StallWatch(
                self.timeline.clock, video_take.STALL_S
            )
        self.phase_stall = self.stalls[step_name]
        watch = self.watch(step_name, video_check_stall.PLOT_DIALOGS, self.phase_stall)
        ticks = max(1, -(-PLOT_BUDGET_MS // video_probes.POLL_MS))
        for _ in range(ticks):
            if video_check_approval.ask_record_count(self.hook_log, ask_mark):
                raise ProbeFailed(
                    step_name, f"an approval prompt appeared while waiting for the {what}"
                )
            state = self.turn()
            if state == "errored":
                raise ProbeFailed(step_name, "the agent's turn ended in an API error")
            fresh = self.arrivals()
            if fresh:
                self.refuse_empty_plot(step_name, fresh[0])
                return "card", fresh[0]
            if self.found.get(step_name) is None:
                self.found[step_name] = find()
            if state == "ended":
                if self.found.get(step_name) is None:
                    raise ProbeFailed(step_name, f"the turn ended without a new {what}")
                return "result", self.found[step_name]
            watch()
            self.page.wait_for_timeout(video_probes.POLL_MS)
        missing = "end of the turn" if self.found.get(step_name) else f"new {what}"
        raise ProbeFailed(
            step_name,
            f"no {missing} within {PLOT_BUDGET_MS} ms; last lines:\n"
            f"{video_probes.terminal_tail(self.page)}",
        )

    def refuse_empty_plot(self, step_name: str, artifact: dict) -> None:
        """Fail ``step_name`` when a new plot card has nothing drawn in it.

        The agent sometimes saves a scratch plot of its own; its card would open
        onto empty axes, which the video must not show. Checked while the wait
        is sped up, before the card's real-time opening.
        """
        if artifact.get("artifact_type") != video_check_plot.PLOT_ARTIFACT_TYPE:
            return
        port, secret = self.ctx.stack.artifact_port, self.ctx.stack.operator_secret
        try:
            html = _fetch_html(port, video_check_plot.artifact_file_path(artifact), secret)
        except OSError:
            return
        if video_check_plot.plot_is_empty(html):
            title = artifact.get("title") or artifact["id"]
            raise ProbeFailed(step_name, f"card {title!r} ({artifact['id']}) is an empty plot")

    def agent_phase(
        self,
        step_name: str,
        caption: str,
        ask_mark: datetime,
        find: Callable[[], str | None],
        what: str,
    ) -> str:
        """Fast-forward through the agent's work, opening each card as it lands.

        Returns the id of the new ``what`` once the turn has ended with it.
        """
        while True:
            with self.step(step_name, caption, fast_forward=True):
                kind, found = self.wait_for_result(step_name, ask_mark, find, what)
            if kind == "result":
                return found
            with self.step("open-artifact", caption):
                self.open_artifact(found)

    def correlation_id(self) -> str | None:
        port = self.ctx.stack.artifact_port
        secret = self.ctx.stack.operator_secret
        return video_check_plot.find_correlation_artifact(
            capture.fetch_artifacts(port, secret),
            self.ctx.before_ids | {self.found.get("plot") or ""},
            lambda path: _fetch_html(port, path, secret),
        )

    def open_correlation(self, corr_id: str) -> None:
        """Bring the correlation plot's preview to the front and hold it.

        It usually still shows from when its card landed; it is clicked again
        only when a later card took the preview, since re-rendering a large
        plot costs seconds of real-time footage.
        """
        if video_probes.preview_shows(self.page, corr_id):
            self.page.wait_for_timeout(CORR_HOLD_MS)
            return
        self.click(self.page.locator(video_probes.artifact_row(corr_id)).first)
        if not video_take.poll(
            self.page,
            lambda: video_probes.preview_shows(self.page, corr_id),
            OPEN_PLOT_BUDGET_MS,
        ):
            raise ProbeFailed(
                "open-correlation",
                f"correlation plot {corr_id} did not render; "
                f"{video_probes.preview_state(self.page)}",
            )
        self.page.wait_for_timeout(CORR_HOLD_MS)

    def check_attachment(self, approved: video_check_approval.Ask, corr_id: str) -> None:
        """Fail ``post`` unless the approved call attaches the correlation plot alone.

        Checked before the approving Enter, so a wrong entry is never created.
        """
        tool_input: dict | None = None
        for _ in range(max(1, -(-ATTACH_BUDGET_MS // video_probes.POLL_MS))):
            tool_input = _tool_input(self.transcript, approved.tool_use_id)
            if tool_input is not None:
                break
            self.page.wait_for_timeout(video_probes.POLL_MS)
        if tool_input is None:
            raise ProbeFailed(
                "post", f"no {APPROVED_TOOL} call {approved.tool_use_id} in the transcript"
            )
        ids = tool_input.get("artifact_ids")
        if not ids:
            raise ProbeFailed(
                "post", "the entry has no artifact_ids; it must attach the correlation plot"
            )
        if corr_id not in ids:
            raise ProbeFailed(
                "post", f"the entry attaches {ids}, not the correlation plot {corr_id}"
            )
        plot_id = self.found.get("plot")
        if plot_id and plot_id in ids:
            raise ProbeFailed("post", f"the entry attaches the 3D plot too: {ids}")

    def open_artifact(self, artifact: dict) -> None:
        """Click a newly landed artifact's sidebar row and hold once it has rendered.

        Fails ``open-artifact`` with the preview pane's state and the terminal's
        last lines when the row never shows or its preview never renders; the
        stall watch and the dialog check keep running throughout.
        """
        artifact_id = artifact["id"]
        self.opened.add(artifact_id)
        title = artifact.get("title") or artifact_id
        watch = self.watch("open-artifact", video_check_stall.PLOT_DIALOGS, self.phase_stall)

        def wait(check: Callable[[], bool], budget_ms: int, what: str) -> None:
            for _ in range(max(1, -(-budget_ms // video_probes.POLL_MS))):
                if check():
                    return
                watch()
                self.page.wait_for_timeout(video_probes.POLL_MS)
            if check():
                return
            raise ProbeFailed(
                "open-artifact",
                f"{what} for {title!r} ({artifact_id}) within {budget_ms} ms; "
                f"{video_probes.preview_state(self.page)}; last lines:\n"
                f"{video_probes.terminal_tail(self.page)}",
            )

        wait(lambda: video_probes.card_shown(self.page, artifact_id), CARD_BUDGET_MS, "no row")
        self.click(self.page.locator(video_probes.artifact_row(artifact_id)).first)
        wait(
            lambda: video_probes.preview_shows(self.page, artifact_id),
            PREVIEW_BUDGET_MS,
            "no rendered preview",
        )
        self.page.wait_for_timeout(OPEN_HOLD_MS)

    def open_plot(self, plot_id: str) -> Any:
        self.click(self.page.locator(video_probes.artifact_row(plot_id)).first)
        frame: list[Any] = []

        def ready() -> bool:
            candidate = self.shown_plot()
            frame[:] = [candidate] if candidate is not None else []
            return candidate is not None

        if not video_take.poll(self.page, ready, OPEN_PLOT_BUDGET_MS):
            raise ProbeFailed("open-plot", f"plot {plot_id} did not render a 3D scene")
        return frame[0]

    def shown_plot(self) -> Any | None:
        """The plot preview's frame once it shows a 3D scene on screen, else None.

        The gallery re-renders its preview on a focus change or a list refresh:
        mid-swap the old frame is detached and the new one has no box yet, and
        both read as "not yet".
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            candidate = video_probes.plot_frame(self.page)
            video_probes.camera_azimuth(candidate)
            if not candidate.frame_element().bounding_box():
                return None
        except ProbeFailed:
            return None
        except PlaywrightError as exc:
            if "detached" not in str(exc):
                raise
            return None
        return candidate

    def on_plot(self, frame: Any, use: Callable[[Any], Any]) -> tuple[Any, Any]:
        """Apply ``use`` to the plot preview's frame; return the frame and the result.

        The gallery re-renders its preview on a focus change or a list refresh,
        which replaces the iframe and detaches the frame held so far; the
        preview showing now is found and ``use`` runs once more on it.
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            return frame, use(frame)
        except _PreviewReplaced:
            pass
        except PlaywrightError as exc:
            if "detached" not in str(exc):
                raise
        fresh: list[Any] = []

        def ready() -> bool:
            candidate = self.shown_plot()
            fresh[:] = [candidate] if candidate is not None else []
            return candidate is not None

        if not video_take.poll(self.page, ready, OPEN_PLOT_BUDGET_MS):
            raise ProbeFailed("rotate", "the plot preview was replaced and never came back")
        return fresh[0], use(fresh[0])

    def rotate(self, frame: Any) -> None:
        started = self.timeline.clock()
        frame, before = self.on_plot(frame, video_probes.camera_azimuth)
        change = 0.0
        for drag in range(1, MAX_DRAGS + 1):
            frame, box = self.on_plot(frame, _on_screen_box)
            cx = box["x"] + box["width"] / 2
            cy = box["y"] + box["height"] / 2
            half = box["width"] * DRAG_FRACTION / 2
            frame, _armed = self.on_plot(frame, video_probes.arm_relayout)
            video_overlay.move_cursor(self.page, cx - half, cy)
            self.page.mouse.down()
            video_overlay.move_cursor(self.page, cx + half, cy, steps=DRAG_STEPS)
            self.page.mouse.up()
            self.page.wait_for_timeout(DRAG_SETTLE_MS)
            frame, _fired = self.on_plot(frame, video_probes.await_relayout)
            frame, after = self.on_plot(frame, video_probes.camera_azimuth)
            change = video_probes.azimuth_change(before, after)
            elapsed = self.timeline.clock() - started
            print(
                f"rotate drag {drag}: camera moved {change:.0f} deg, {elapsed:.1f} s in", flush=True
            )
            if elapsed > ROTATE_BUDGET_S:
                raise ProbeFailed(
                    "rotate",
                    f"{elapsed:.0f} s spent, over the {ROTATE_BUDGET_S:g} s real-time budget "
                    f"(camera moved {change:.0f} deg)",
                )
            if change >= MIN_AZIMUTH_CHANGE_DEG:
                return
        raise ProbeFailed(
            "rotate",
            f"camera azimuth moved {change:.0f} deg after {MAX_DRAGS} drags; "
            f"need {MIN_AZIMUTH_CHANGE_DEG:.0f}",
        )

    def wait_for_approval(self, ask_mark: datetime) -> video_check_approval.Ask:
        watch = self.watch("post", video_check_stall.NOTICE_DIALOGS)
        ticks = max(1, -(-APPROVAL_BUDGET_MS // video_probes.POLL_MS))
        for _ in range(ticks):
            asks = self.asks(ask_mark)
            if asks:
                return asks[0]
            state = self.turn()
            if state == "errored":
                raise ProbeFailed("post", "the agent's turn ended in an API error")
            if state == "ended":
                raise ProbeFailed("post", f"the turn ended without an {APPROVED_TOOL} approval")
            watch()
            self.page.wait_for_timeout(video_probes.POLL_MS)
        raise ProbeFailed("post", f"no {APPROVED_TOOL} approval within {APPROVAL_BUDGET_MS} ms")

    def wait_for_draft(self, ask_mark: datetime, approved: video_check_approval.Ask) -> None:
        """Wait for the saved draft, pressing Enter once more only when it is safe.

        The retry fires once, after :data:`DRAFT_RETRY_AFTER_MS` without a
        draft, and only when the approved call has no ``tool_result`` yet and
        no other approval prompt was logged: then the first Enter cannot have
        reached the approval box, and the second cannot approve anything else.
        """
        watch = self.watch("draft", video_check_stall.NOTICE_DIALOGS)
        retry_tick = -(-DRAFT_RETRY_AFTER_MS // video_probes.POLL_MS)
        ticks = max(1, -(-DRAFT_BUDGET_MS // video_probes.POLL_MS))
        for tick in range(ticks):
            asks = self.asks(ask_mark)
            if video_probes.draft_ready(self.page):
                return
            if self.turn() == "errored":
                raise ProbeFailed("post", "the agent's turn ended in an API error")
            if (
                tick == retry_tick
                and approved.tool_use_id is not None
                and asks == [approved]
                and not _has_tool_result(self.transcript, approved.tool_use_id)
            ):
                self.press_enter("draft")
            watch()
            self.page.wait_for_timeout(video_probes.POLL_MS)
        raise ProbeFailed("draft", f"no logbook draft within {DRAFT_BUDGET_MS} ms")


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------


def _mark(name: str) -> Callable[[Take], None]:
    """Note the approval-log time before prompt *name*; no beat of its own."""

    def run(take: Take) -> None:
        Landing.of(take).marks[name] = video_check_approval.log_mark()

    return run


def _phase(name: str, caption: str, what: str, find: str) -> Callable[[Take], None]:
    """Fast-forward through one agent phase, opening each card as it lands."""

    def run(take: Take) -> None:
        state = Landing.of(take)
        state.found[name] = state.agent_phase(
            name, caption, state.marks[name], getattr(state, find), what
        )

    return run


def _open_plot(take: Take) -> None:
    state = Landing.of(take)
    state.plot_frame = state.open_plot(state.found["plot"])


def _rotate(take: Take) -> None:
    state = Landing.of(take)
    state.rotate(state.plot_frame)


def _open_correlation(take: Take) -> None:
    state = Landing.of(take)
    state.open_correlation(state.found["correlate"])


def _post(take: Take) -> None:
    state = Landing.of(take)
    state.approved = state.wait_for_approval(state.marks["post"])
    state.check_attachment(state.approved, state.found["correlate"])


def _approve(take: Take) -> None:
    take.press_enter("approve")


def _draft(take: Take) -> None:
    state = Landing.of(take)
    assert state.approved is not None
    state.wait_for_draft(state.marks["post"], state.approved)


def _close(take: Take) -> None:
    if not video_probes.reveal_attachment(take.page):
        raise ProbeFailed("close", "the saved draft shows no attachment to reveal")
    take.page.wait_for_timeout(CLOSE_HOLD_MS)


ASK = "Ask in plain language"
CORRELATE = "Ask for a correlation"
POST = "Post it to the logbook"
APPROVE = "Writes wait for the operator's approval"

LANDING = (
    focus("focus-1", ASK),
    _mark("plot"),
    prompt("prompt-1", ASK, PROMPT_PLOT),
    _phase("plot", "The agent reads the archiver and plots", "3D BPM scatter plot", "plot_id"),
    Beat("open-plot", "An interactive 3D plot", _open_plot),
    Beat("rotate", "An interactive 3D plot", _rotate),
    focus("focus-2", CORRELATE),
    _mark("correlate"),
    prompt("prompt-2", CORRELATE, PROMPT_CORR),
    _phase(
        "correlate",
        "The agent correlates the BPMs with the vacuum",
        "correlation plot",
        "correlation_id",
    ),
    Beat("open-correlation", "A correlation plot", _open_correlation),
    focus("focus-3", POST),
    _mark("post"),
    prompt("prompt-3", POST, PROMPT_POST),
    Beat("post", "The agent drafts the entry", _post, fast_forward=True),
    hold("approval-hold", APPROVE, APPROVAL_HOLD_MS),
    Beat("approve", APPROVE, _approve),
    Beat("draft", "Saving the draft", _draft, fast_forward=True),
    Beat("close", "Draft saved with the plot attached", _close),
)

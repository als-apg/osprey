"""Take orchestration for the landing-page demo video recorder.

A *take* is one recording attempt of the storyboard in one theme. Every take
starts from the same clean state, built by :func:`start_take`:

1. close the previous take's browser context;
2. kill every terminal PTY through ``POST /api/terminal/restart``;
3. open a fresh 1920x1080 context carrying the overlay init script, and listen
   for the ``/ws/terminal`` websocket;
4. load the UI once with ``?token&theme``, in the preset's own layout with the
   vertical left rail (never reloaded: a reload
   resumes the dead session and gets ``transcript_missing``);
5. record the terminal session id from the ``session_info`` frame;
6. wait for the dock's boot layout to settle, then widen the terminal;
7. click the terminal and wait for the Claude Code REPL;
8. snapshot the artifact ids, then start the CDP screencast into a
   :class:`~docs.screenshots.video_frames.FrameSink`.

:func:`record_take` then plays the storyboard, one :func:`~docs.screenshots.
video_timeline.step` per beat, and raises :class:`ProbeFailed` naming the beat
whose check failed. :func:`record_theme` records one take and closes it when
it fails; the CLI retries a failed take on a fresh stack.

Every wait is a ``page.wait_for_timeout`` poll: sync Playwright only pumps its
event loop inside Playwright calls, so a plain sleep would freeze the page and
starve the websocket and screencast handlers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docs.screenshots import (
    capture,
    video_check_approval,
    video_check_plot,
    video_check_stall,
    video_check_turn,
    video_overlay,
    video_probes,
)
from docs.screenshots.video_frames import FrameSink
from docs.screenshots.video_overlay import OVERLAY_JS
from docs.screenshots.video_probes import ProbeFailed
from docs.screenshots.video_timeline import Step, Timeline, step

if TYPE_CHECKING:
    from playwright.sync_api import Browser

VIEWPORT = {"width": 1920, "height": 1080}

TERMINAL_WS_PATH = "/ws/terminal"

# Share of the dock workspace the terminal group is widened to, so the typed
# prompts and the agent's answer stay legible next to the artifacts panel.
TERMINAL_WIDTH_FRACTION = 0.45

GOTO_TIMEOUT_MS = 30_000
CLICK_TIMEOUT_MS = 30_000
MOUNT_BUDGET_MS = 30_000
SESSION_INFO_BUDGET_MS = 30_000
SETTLE_BUDGET_MS = 10_000
# Time for the terminal to refit its rows after the group is resized.
WIDEN_SETTLE_MS = 500
RESTART_TIMEOUT_S = 10.0

SCREENCAST_PARAMS = {
    "format": "jpeg",
    "quality": 90,
    "maxWidth": VIEWPORT["width"],
    "maxHeight": VIEWPORT["height"],
}

_MOUNTED_JS = "() => !!document.querySelector('#terminal-container')"

# Resolves once the dock has finished arranging the shell: the boot layout is
# announced final AND the console has held its place for one frame after it.
# The dock re-parents the terminal card twice during boot and each move takes
# focus away, so nothing may be clicked or typed before this resolves.
DOCK_SETTLED_JS = """
async (budgetMs) => {
  const dock = await import('/static/js/dock-workspace.js');
  const place = () => {
    let out = '';
    for (let n = document.querySelector('#operator-container'); n; n = n.parentElement) {
      out += '>' + (n.id || n.className || n.tagName);
    }
    return out;
  };
  const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
  const deadline = Date.now() + budgetMs;
  let previous = null;
  while (Date.now() < deadline) {
    await frame();
    const current = place();
    if (dock.bootLayoutSettled() && current === previous) return true;
    previous = current;
  }
  return false;
}
"""

# Resize the terminal's dock group to a share of the workspace width. Returns
# false when the dock or its terminal panel is missing.
WIDEN_TERMINAL_JS = """
async (fraction) => {
  const dock = await import('/static/js/dock-workspace.js');
  const api = dock.getDockApi();
  if (!api) return false;
  const panel = api.getPanel('terminal');
  if (!panel) return false;
  panel.group.api.setSize({ width: Math.round(fraction * api.width) });
  return true;
}
"""


@dataclass
class VideoStack:
    """The running tutorial stack and web terminal a take is recorded on.

    ``current`` is the take whose browser context is still open; the next
    :func:`start_take` closes it first.
    """

    project_dir: Path
    artifact_port: int
    base_url: str
    operator_secret: str
    work_dir: Path
    # The Claude Code config dir the web terminal's session runs under.
    claude_config_dir: Path | None = None
    browser: Any = None
    current: TakeContext | None = None

    @property
    def build_dir(self) -> Path:
        return self.project_dir / "build"


@dataclass
class TakeContext:
    """Everything one take needs after bootstrap, and its teardown."""

    stack: VideoStack
    theme: str
    context: Any
    page: Any
    frames_dir: Path
    sink: FrameSink
    session_id: str | None = None
    before_ids: set[str] = field(default_factory=set)
    cdp: Any = None
    screencasting: bool = False
    closed: bool = False

    def stop_screencast(self) -> None:
        """Stop the screencast and write the held frame. Safe to call twice."""
        if self.screencasting and self.cdp is not None:
            self.screencasting = False
            try:
                self.cdp.send("Page.stopScreencast")
            except Exception as exc:
                print(f"WARNING: could not stop the screencast: {exc}")
        self.sink.flush()

    def close(self) -> None:
        """Stop the screencast and close the browser context. Safe to call twice."""
        if self.closed:
            return
        self.stop_screencast()
        self.closed = True
        try:
            self.context.close()
        except Exception as exc:
            print(f"WARNING: could not close the take's browser context: {exc}")


def restart_terminal(stack: VideoStack) -> None:
    """Kill every terminal PTY so the take starts a fresh Claude Code session.

    The operator secret goes in a header: a cookie-only POST carries no Origin
    header from urllib and is refused by the strict origin check.
    """
    request = urllib.request.Request(
        f"{stack.base_url}/api/terminal/restart",
        data=b"",
        method="POST",
        headers={capture.OPERATOR_SECRET_HEADER: stack.operator_secret},
    )
    try:
        with urllib.request.urlopen(request, timeout=RESTART_TIMEOUT_S) as resp:
            status = getattr(resp, "status", 200)
    except (urllib.error.URLError, OSError) as exc:
        raise ProbeFailed("restart", f"POST /api/terminal/restart failed: {exc}") from exc
    if status != 200:
        raise ProbeFailed("restart", f"POST /api/terminal/restart returned HTTP {status}")


def _watch_session_info(take: TakeContext) -> Any:
    """Return a ``page.on('websocket')`` handler that records the session id."""

    def on_frame(payload: Any) -> None:
        if take.session_id is not None:
            return
        if isinstance(payload, bytes | bytearray):
            payload = payload.decode("utf-8", "replace")
        try:
            message = json.loads(payload)
        except (TypeError, ValueError):
            return
        if isinstance(message, dict) and message.get("type") == "session_info":
            session_id = message.get("session_id")
            if session_id:
                take.session_id = str(session_id)

    def on_websocket(ws: Any) -> None:
        if TERMINAL_WS_PATH in ws.url:
            ws.on("framereceived", on_frame)

    return on_websocket


def _poll(page: Any, check: Any, budget_ms: int) -> bool:
    """Poll ``check()`` every probe tick until it is true or ``budget_ms`` runs out."""
    ticks = max(1, -(-budget_ms // video_probes.POLL_MS))
    for _ in range(ticks):
        if check():
            return True
        page.wait_for_timeout(video_probes.POLL_MS)
    return bool(check())


def _settle_and_widen(page: Any) -> None:
    if not _poll(page, lambda: page.evaluate(_MOUNTED_JS), MOUNT_BUDGET_MS):
        raise ProbeFailed("layout", "#terminal-container never mounted")
    if not page.evaluate(DOCK_SETTLED_JS, SETTLE_BUDGET_MS):
        raise ProbeFailed("layout", f"dock boot layout did not settle in {SETTLE_BUDGET_MS} ms")
    if not page.evaluate(WIDEN_TERMINAL_JS, TERMINAL_WIDTH_FRACTION):
        raise ProbeFailed("layout", "dock API or terminal panel unavailable; cannot widen")
    page.wait_for_timeout(WIDEN_SETTLE_MS)


def _start_screencast(take: TakeContext) -> None:
    cdp = take.context.new_cdp_session(take.page)
    take.cdp = cdp

    def on_screencast_frame(payload: dict[str, Any]) -> None:
        if not take.screencasting:
            return  # a late frame from a take that has stopped recording
        take.sink.on_frame(
            payload,
            lambda: cdp.send("Page.screencastFrameAck", {"sessionId": payload["sessionId"]}),
        )

    cdp.on("Page.screencastFrame", on_screencast_frame)
    cdp.send("Page.startScreencast", dict(SCREENCAST_PARAMS))
    take.screencasting = True


def start_take(browser: Browser | Any, stack: VideoStack, theme: str) -> TakeContext:
    """Bootstrap one take in ``theme`` and return it with the screencast running.

    The new take becomes ``stack.current`` as soon as its context exists, so a
    take that fails part-way is still closed by the next call. Raises
    :class:`ProbeFailed` naming the step that gave up.
    """
    if stack.current is not None:
        stack.current.close()
        stack.current = None

    restart_terminal(stack)

    frames_dir = stack.work_dir / theme / "frames"
    context = browser.new_context(viewport=dict(VIEWPORT))
    context.add_init_script(OVERLAY_JS)
    page = context.new_page()
    take = TakeContext(
        stack=stack,
        theme=theme,
        context=context,
        page=page,
        frames_dir=frames_dir,
        sink=FrameSink(frames_dir),
    )
    stack.current = take
    page.on("websocket", _watch_session_info(take))

    token = urllib.parse.quote(stack.operator_secret, safe="")
    page.goto(
        f"{stack.base_url}/?token={token}&theme={theme}",
        wait_until="domcontentloaded",
        timeout=GOTO_TIMEOUT_MS,
    )

    if not _poll(page, lambda: take.session_id is not None, SESSION_INFO_BUDGET_MS):
        raise ProbeFailed(
            "session-info",
            f"no session_info frame on {TERMINAL_WS_PATH} within {SESSION_INFO_BUDGET_MS} ms",
        )

    _settle_and_widen(page)

    page.locator("#terminal-container").click(timeout=CLICK_TIMEOUT_MS)
    video_probes.await_repl_ready(page)

    take.before_ids = capture.artifact_ids(stack.artifact_port, stack.operator_secret)
    _start_screencast(take)
    return take


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------

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

# Per-keystroke delay, slow enough to read along, fast enough not to drag.
TYPE_DELAY_MS = 45

PLOT_BUDGET_MS = 600_000
APPROVAL_BUDGET_MS = 180_000
# Without a draft this long after the approving Enter, the keypress may have
# been swallowed; a single guarded retry Enter follows.
DRAFT_RETRY_AFTER_MS = 15_000
DRAFT_BUDGET_MS = 90_000
OPEN_PLOT_BUDGET_MS = 30_000
# Each card that lands while the agent works is opened in real time: the card
# must show in the gallery, then its preview render, then it is held on screen.
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

# Seconds without any progress (screen or transcripts) before a wait on the
# agent gives up, however much of its budget is left.
STALL_S = video_check_stall.STALL_S
# Terminal lines quoted when a wait gives up on a stall or a dialog.
DUMP_LINES = 15


def _fetch_html(artifact_port: int, path: str, secret: str | None = None) -> str:
    """Return the artifact server's file at ``path`` as text."""
    request = capture.artifact_request(artifact_port, path, secret)
    with urllib.request.urlopen(request, timeout=ARTIFACT_FETCH_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "replace")


def _ask_record_count(path: Path, offset: int) -> int:
    """Count every complete ``status == "ask"`` record after ``offset``, named or not."""
    try:
        with Path(path).open("rb") as fh:
            fh.seek(max(offset, 0))
            data = fh.read()
    except FileNotFoundError:
        return 0
    count = 0
    for raw in data.split(b"\n")[:-1]:
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(record, dict) and record.get("status") == "ask":
            count += 1
    return count


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


class _Take:
    """One run of the storyboard over a bootstrapped :class:`TakeContext`."""

    def __init__(self, ctx: TakeContext, timeline: Timeline) -> None:
        self.ctx = ctx
        self.page = ctx.page
        self.timeline = timeline
        self.hook_log = video_check_approval.hook_log_path(ctx.stack.project_dir)
        self.transcript: Path | None = None
        self.submitted_at = 0.0
        self.opened: set[str] = set()
        # Per agent phase ("plot", "correlate"): the result found so far and the
        # stall count shared by the phase's waits and its card openings.
        self.found: dict[str, str | None] = {}
        self.stalls: dict[str, video_check_stall.StallWatch] = {}
        self.phase_stall: video_check_stall.StallWatch | None = None

    # -- step plumbing -----------------------------------------------------

    def _enter(self, current: Step) -> None:
        print(f"{self.ctx.theme}: {current.name}", flush=True)
        video_overlay.set_caption(self.page, current.caption)

    def _exit(self, _current: Step) -> None:
        video_overlay.set_caption(self.page, None)
        self.ctx.sink.flush()

    def step(self, name: str, caption: str, fast_forward: bool = False) -> Any:
        return step(
            self.timeline,
            name,
            caption=caption,
            fast_forward=fast_forward,
            on_enter=self._enter,
            on_exit=self._exit,
        )

    # -- probes ------------------------------------------------------------

    def turn(self) -> str:
        if self.transcript is None and self.ctx.session_id:
            self.transcript = video_check_turn.transcript_path(
                self.ctx.stack.build_dir,
                self.ctx.session_id,
                self.ctx.stack.claude_config_dir,
            )
        return video_check_turn.turn_state(
            self.transcript, self.submitted_at, now=self.timeline.clock
        )

    def asks(self, offset: int) -> list[video_check_approval.Ask]:
        """Asks after ``offset``; fails the take on any unnamed or foreign ask."""
        asks = video_check_approval.find_asks(self.hook_log, offset)
        if _ask_record_count(self.hook_log, offset) > len(asks):
            raise ProbeFailed("approve", "an approval prompt without a tool name was logged")
        foreign = [a.name for a in asks if a.name != APPROVED_TOOL]
        if foreign:
            raise ProbeFailed("approve", f"unexpected approval prompt for {', '.join(foreign)}")
        return asks

    def watch(
        self,
        step_name: str,
        dialogs: tuple[str, ...],
        stall: video_check_stall.StallWatch | None = None,
    ) -> Callable[[], None]:
        """A per-tick check that fails ``step_name`` on a dialog or a stall.

        Call the returned function once per poll tick of a wait on the agent.
        Passing ``stall`` carries one quiet-time count across several waits.
        """
        if stall is None:
            stall = video_check_stall.StallWatch(self.timeline.clock, STALL_S)

        def check() -> None:
            try:
                screen = video_probes.xterm_text(self.page)
            except Exception:
                screen = ""
            dump = video_probes.last_lines(screen, DUMP_LINES)
            dialog = video_check_stall.dialog_on_screen(screen, dialogs)
            if dialog is not None:
                raise ProbeFailed(
                    step_name, f"an unanswered dialog is on screen ({dialog!r}):\n{dump}"
                )
            self.turn()  # resolves the transcript path once the session has one
            stall.observe(screen, video_check_stall.transcript_bytes(self.transcript))
            if stall.stalled:
                raise ProbeFailed(
                    step_name, f"no progress for {stall.quiet_s:.0f} s; last lines:\n{dump}"
                )

        return check

    def plot_id(self) -> str | None:
        port = self.ctx.stack.artifact_port
        secret = self.ctx.stack.operator_secret
        return video_check_plot.find_plot_artifact(
            capture.fetch_artifacts(port, secret),
            self.ctx.before_ids,
            lambda path: _fetch_html(port, path, secret),
        )

    # -- actions -----------------------------------------------------------

    def click(self, locator: Any) -> None:
        """Glide the overlay cursor to ``locator`` and click it."""
        box = locator.bounding_box()
        if box:
            video_overlay.move_cursor(
                self.page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            )
        locator.click(timeout=CLICK_TIMEOUT_MS)

    def require_focus(self, step_name: str) -> None:
        if not video_probes.terminal_focused(self.page):
            raise ProbeFailed(step_name, "keyboard focus is not on the terminal")

    def press_enter(self, step_name: str) -> None:
        self.require_focus(step_name)
        self.page.keyboard.press("Enter")

    def submit(self, step_name: str, prompt: str) -> None:
        self.require_focus(step_name)
        self.page.keyboard.type(prompt, delay=TYPE_DELAY_MS)
        self.require_focus(step_name)
        self.submitted_at = self.timeline.clock()
        self.page.keyboard.press("Enter")

    def focus_terminal(self) -> None:
        self.click(self.page.locator("#terminal-container").first)

    # -- beats -------------------------------------------------------------

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
        ask_offset: int,
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
            self.stalls[step_name] = video_check_stall.StallWatch(self.timeline.clock, STALL_S)
        self.phase_stall = self.stalls[step_name]
        watch = self.watch(step_name, video_check_stall.PLOT_DIALOGS, self.phase_stall)
        ticks = max(1, -(-PLOT_BUDGET_MS // video_probes.POLL_MS))
        for _ in range(ticks):
            if _ask_record_count(self.hook_log, ask_offset):
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
        ask_offset: int,
        find: Callable[[], str | None],
        what: str,
    ) -> str:
        """Fast-forward through the agent's work, opening each card as it lands.

        Returns the id of the new ``what`` once the turn has ended with it.
        """
        while True:
            with self.step(step_name, caption, fast_forward=True):
                kind, found = self.wait_for_result(step_name, ask_offset, find, what)
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
        card = (
            self.page.frame_locator(video_probes.ARTIFACTS_IFRAME)
            .locator(f'[data-id="{corr_id}"]')
            .first
        )
        self.click(card)
        if not _poll(
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
        """Click a newly landed card and hold once its preview has rendered.

        Fails ``open-artifact`` with the preview pane's state and the terminal's
        last lines when the card never shows or its preview never renders; the
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

        wait(lambda: video_probes.card_shown(self.page, artifact_id), CARD_BUDGET_MS, "no card")
        card = (
            self.page.frame_locator(video_probes.ARTIFACTS_IFRAME)
            .locator(f'[data-id="{artifact_id}"]')
            .first
        )
        self.click(card)
        wait(
            lambda: video_probes.preview_shows(self.page, artifact_id),
            PREVIEW_BUDGET_MS,
            "no rendered preview",
        )
        self.page.wait_for_timeout(OPEN_HOLD_MS)

    def open_plot(self, plot_id: str) -> Any:
        item = (
            self.page.frame_locator(video_probes.ARTIFACTS_IFRAME)
            .locator(f'[data-id="{plot_id}"]')
            .first
        )
        self.click(item)
        frame: list[Any] = []

        def ready() -> bool:
            candidate = self.shown_plot()
            frame[:] = [candidate] if candidate is not None else []
            return candidate is not None

        if not _poll(self.page, ready, OPEN_PLOT_BUDGET_MS):
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

        if not _poll(self.page, ready, OPEN_PLOT_BUDGET_MS):
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

    def wait_for_approval(self, ask_offset: int) -> video_check_approval.Ask:
        watch = self.watch("post", video_check_stall.NOTICE_DIALOGS)
        ticks = max(1, -(-APPROVAL_BUDGET_MS // video_probes.POLL_MS))
        for _ in range(ticks):
            asks = self.asks(ask_offset)
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

    def wait_for_draft(self, ask_offset: int, approved: video_check_approval.Ask) -> None:
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
            asks = self.asks(ask_offset)
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

    # -- storyboard ----------------------------------------------------------

    def run(self) -> None:
        ask_caption = "Ask in plain language"
        with self.step("focus-1", ask_caption):
            self.focus_terminal()
        plot_offset = video_check_approval.log_offset(self.hook_log)
        with self.step("prompt-1", ask_caption):
            self.submit("prompt-1", PROMPT_PLOT)
        plot_id = self.agent_phase(
            "plot",
            "The agent reads the archiver and plots",
            plot_offset,
            self.plot_id,
            "3D BPM scatter plot",
        )
        with self.step("open-plot", "An interactive 3D plot"):
            frame = self.open_plot(plot_id)
        with self.step("rotate", "An interactive 3D plot"):
            self.rotate(frame)
        corr_caption = "Ask for a correlation"
        with self.step("focus-2", corr_caption):
            self.focus_terminal()
        corr_offset = video_check_approval.log_offset(self.hook_log)
        with self.step("prompt-2", corr_caption):
            self.submit("prompt-2", PROMPT_CORR)
        corr_id = self.agent_phase(
            "correlate",
            "The agent correlates the BPMs with the vacuum",
            corr_offset,
            self.correlation_id,
            "correlation plot",
        )
        with self.step("open-correlation", "A correlation plot"):
            self.open_correlation(corr_id)
        post_caption = "Post it to the logbook"
        with self.step("focus-3", post_caption):
            self.focus_terminal()
        post_offset = video_check_approval.log_offset(self.hook_log)
        with self.step("prompt-3", post_caption):
            self.submit("prompt-3", PROMPT_POST)
        with self.step("post", "The agent drafts the entry", fast_forward=True):
            approved = self.wait_for_approval(post_offset)
            self.check_attachment(approved, corr_id)
        approve_caption = "Writes wait for the operator's approval"
        with self.step("approval-hold", approve_caption):
            self.page.wait_for_timeout(APPROVAL_HOLD_MS)
        with self.step("approve", approve_caption):
            self.press_enter("approve")
        with self.step("draft", "Saving the draft", fast_forward=True):
            self.wait_for_draft(post_offset, approved)
        with self.step("close", "Draft saved with the plot attached"):
            if not video_probes.reveal_attachment(self.page):
                raise ProbeFailed("close", "the saved draft shows no attachment to reveal")
            self.page.wait_for_timeout(CLOSE_HOLD_MS)


def record_take(ctx: TakeContext, timeline: Timeline) -> None:
    """Play the storyboard on a bootstrapped take, recording each beat in ``timeline``.

    Raises :class:`ProbeFailed` naming the beat whose check failed; the take
    is then worthless and :func:`record_theme` starts another.
    """
    _Take(ctx, timeline).run()


class ThemeFailed(Exception):
    """Every take of a theme failed; names the theme and the last failed step."""

    def __init__(self, theme: str, step: str, detail: str, attempts: int) -> None:
        super().__init__(
            f"theme {theme}: all {attempts} takes failed; last failure at step {step}: {detail}"
        )
        self.theme = theme
        self.step = step
        self.detail = detail


def record_theme(
    stack: VideoStack,
    theme: str,
    clock: Callable[[], float] | None = None,
) -> tuple[TakeContext, Timeline]:
    """Record one take of ``theme`` on ``stack``.

    Returns the take, with its screencast stopped and every frame on disk, and
    its timeline. A failed check raises :class:`ProbeFailed` naming the beat,
    and so does a browser call that gives up mid-take; the take is closed
    first. Retrying is the caller's call, on a fresh stack. ``clock``
    overrides the timeline's epoch clock.
    """
    from playwright.sync_api import Error as PlaywrightError

    timeline = Timeline(theme=theme) if clock is None else Timeline(theme=theme, clock=clock)
    try:
        take = start_take(stack.browser, stack, theme)
        record_take(take, timeline)
    except (ProbeFailed, PlaywrightError) as exc:
        if stack.current is not None:
            stack.current.close()
            stack.current = None
        if isinstance(exc, PlaywrightError):
            where = timeline.steps[-1].name if timeline.steps else "bootstrap"
            raise ProbeFailed(where, f"browser error: {exc}") from exc
        raise
    take.stop_screencast()
    return take, timeline

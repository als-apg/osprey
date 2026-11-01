"""Take orchestration for the docs video recorder.

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

:func:`record_take` then plays a video's storyboard on a :class:`Take`, one
:func:`~docs.screenshots.video_timeline.step` per beat, and raises
:class:`ProbeFailed` naming the beat whose check failed. :func:`record_theme` records one take and closes it when
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docs.screenshots import (
    capture,
    video_check_approval,
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
# The first request after launch waits for the server's startup to finish,
# which takes tens of seconds on a loaded host.
RESTART_TIMEOUT_S = 60.0

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
    header from urllib and is refused by the strict origin check. The body is
    an empty JSON object: the web gate refuses a POST body of any other type.
    """
    request = urllib.request.Request(
        f"{stack.base_url}/api/terminal/restart",
        data=b"{}",
        method="POST",
        headers={
            capture.OPERATOR_SECRET_HEADER: stack.operator_secret,
            "Content-Type": "application/json",
        },
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


def _settle_and_widen(page: Any) -> None:
    if not poll(page, lambda: page.evaluate(_MOUNTED_JS), MOUNT_BUDGET_MS):
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

    if not poll(page, lambda: take.session_id is not None, SESSION_INFO_BUDGET_MS):
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
# The take a storyboard plays on
# ---------------------------------------------------------------------------

# Per-keystroke delay, slow enough to read along, fast enough not to drag.
TYPE_DELAY_MS = 45

# Seconds without any progress (screen or transcripts) before a wait on the
# agent gives up, however much of its budget is left.
STALL_S = video_check_stall.STALL_S
# Terminal lines quoted when a wait gives up on a stall or a dialog.
DUMP_LINES = 15

# A storyboard is the beats of one video, played in order on one take. Each beat
# is a callable taking the :class:`Take`: a :class:`~docs.screenshots.
# video_beats.Beat` from the shared primitives, or a video's own function.
Storyboard = Sequence[Callable[["Take"], None]]


def poll(page: Any, check: Any, budget_ms: int) -> bool:
    """Poll ``check()`` every probe tick until it is true or ``budget_ms`` runs out."""
    ticks = max(1, -(-budget_ms // video_probes.POLL_MS))
    for _ in range(ticks):
        if check():
            return True
        page.wait_for_timeout(video_probes.POLL_MS)
    return bool(check())


class Take:
    """One run of a storyboard over a bootstrapped :class:`TakeContext`.

    It holds what every video's beats share: the step plumbing that shows a
    beat's caption and stamps its span, the agent's turn state, the stall and
    dialog watch, and the cursor, focus and typing actions. ``memo`` is where a
    video's own beats keep what a later beat needs.
    """

    def __init__(self, ctx: TakeContext, timeline: Timeline) -> None:
        self.ctx = ctx
        self.page = ctx.page
        self.timeline = timeline
        self.hook_log = video_check_approval.record_log(ctx.stack.project_dir)
        self.transcript: Path | None = None
        self.submitted_at = 0.0
        self.memo: dict[str, Any] = {}

    # -- step plumbing -----------------------------------------------------

    def _enter(self, current: Step) -> None:
        print(f"{self.ctx.theme}: {current.name}", flush=True)
        video_overlay.set_caption(self.page, current.caption)

    def _exit(self, _current: Step) -> None:
        video_overlay.set_caption(self.page, None)
        self.ctx.sink.flush()

    def step(self, name: str, caption: str | None, fast_forward: bool = False) -> Any:
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


def record_take(ctx: TakeContext, timeline: Timeline, storyboard: Storyboard) -> None:
    """Play *storyboard* on a bootstrapped take, recording each beat in ``timeline``.

    Raises :class:`ProbeFailed` naming the beat whose check failed; the take
    is then worthless and :func:`record_theme` starts another.
    """
    take = Take(ctx, timeline)
    for beat in storyboard:
        beat(take)


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
    storyboard: Storyboard,
    clock: Callable[[], float] | None = None,
) -> tuple[TakeContext, Timeline]:
    """Record one take of *storyboard* in ``theme`` on ``stack``.

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
        record_take(take, timeline, storyboard)
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

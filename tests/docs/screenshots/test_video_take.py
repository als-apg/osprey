"""Unit tests for the demo-video take orchestration.

CI-safe: the browser, context, page, websocket and CDP session are small fakes
that record every call into one shared event log, ``urlopen`` and the artifact
snapshot are monkeypatched, and the ``osprey web`` launch runs against stubbed
``subprocess`` calls. No browser, server or container is started.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docs.screenshots import capture, video_probes, video_take
from docs.screenshots.video_overlay import OVERLAY_JS
from docs.screenshots.video_probes import ProbeFailed
from docs.screenshots.video_take import (
    DOCK_SETTLED_JS,
    TERMINAL_WIDTH_FRACTION,
    WIDEN_TERMINAL_JS,
    VideoStack,
    record_take,
    record_theme,
    start_take,
)
from docs.screenshots.video_timeline import Timeline

SECRET = "s3cret/+=token"

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeWebSocket:
    def __init__(self, url: str) -> None:
        self.url = url
        self.handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def emit(self, payload) -> None:
        for handler in self.handlers.get("framereceived", []):
            handler(payload)


class FakeLocator:
    def __init__(self, log: list, selector: str) -> None:
        self.log = log
        self.selector = selector

    @property
    def first(self) -> FakeLocator:
        return self

    def bounding_box(self) -> dict:
        return {"x": 100.0, "y": 200.0, "width": 400.0, "height": 300.0}

    def click(self, **kwargs) -> None:
        self.log.append(("click", self.selector))


class FakeFrameLocator:
    def __init__(self, log: list, selector: str) -> None:
        self.log = log
        self.selector = selector

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self.log, f"{self.selector} >> {selector}")


class FakeKeyboard:
    def __init__(self, log: list, world=None) -> None:
        self.log = log
        self.world = world

    def press(self, key: str) -> None:
        self.log.append(("press", key))
        if self.world is not None:
            self.world.on_press(key)

    def type(self, text: str, delay=None) -> None:
        self.log.append(("type", text, delay))
        if self.world is not None:
            self.world.on_type(text)


class FakeMouse:
    def __init__(self, log: list, world=None) -> None:
        self.log = log
        self.world = world

    def move(self, x, y) -> None:
        self.log.append(("mouse.move", x, y))

    def down(self) -> None:
        self.log.append(("mouse.down",))

    def up(self) -> None:
        self.log.append(("mouse.up",))
        if self.world is not None:
            self.world.on_drag()


class FakePage:
    def __init__(self, log: list, scenario: dict) -> None:
        self.log = log
        self.scenario = scenario
        self.handlers: dict[str, list] = {}
        self.world = scenario.get("world")
        self.keyboard = FakeKeyboard(log, self.world)
        self.mouse = FakeMouse(log, self.world)
        self.sockets: list[FakeWebSocket] = []

    def on(self, event: str, handler) -> None:
        self.log.append(("page.on", event))
        self.handlers.setdefault(event, []).append(handler)

    def _open_socket(self, url: str) -> FakeWebSocket:
        ws = FakeWebSocket(url)
        self.sockets.append(ws)
        for handler in self.handlers.get("websocket", []):
            handler(ws)
        return ws

    def goto(self, url: str, **kwargs) -> None:
        self.log.append(("goto", url, kwargs))
        # Unrelated websocket first: its frames must be ignored.
        other = self._open_socket("ws://127.0.0.1/ws/other")
        other.emit(json.dumps({"type": "session_info", "session_id": "WRONG"}))
        terminal = self._open_socket("ws://127.0.0.1/ws/terminal?x=1")
        terminal.emit(b"\x00binary-noise")
        terminal.emit(json.dumps({"type": "output", "data": "hi"}))
        if self.scenario.get("session_info", True):
            terminal.emit(json.dumps({"type": "session_info", "session_id": "sess-123"}))
            terminal.emit(json.dumps({"type": "session_info", "session_id": "later"}))

    def reload(self, **kwargs) -> None:
        self.log.append(("reload",))

    def evaluate(self, script: str, arg=None):
        if script == video_take._MOUNTED_JS:
            self.log.append(("mounted?",))
            return True
        if script == DOCK_SETTLED_JS:
            self.log.append(("settle", arg))
            return self.scenario.get("settled", True)
        if script == WIDEN_TERMINAL_JS:
            self.log.append(("widen", arg))
            return self.scenario.get("widened", True)
        if script == video_probes._XTERM_TEXT_JS:
            self.log.append(("xterm_text",))
            if self.world is not None and self.world.screen is not None:
                return self.world.screen
            return self.scenario.get("xterm", "? for shortcuts")
        if script == video_probes._FOCUS_JS:
            self.log.append(("focused?",))
            return self.world.focus if self.world is not None else True
        if script == "(t) => window.__demoCaption(t)":
            self.log.append(("caption", arg))
            return None
        if script == "([x, y]) => window.__demoCursor(x, y)":
            return None
        raise AssertionError(f"unexpected evaluate: {script[:60]!r}")

    def wait_for_timeout(self, ms) -> None:
        self.log.append(("wait", ms))
        if self.world is not None:
            self.world.advance(ms)

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self.log, selector)

    def frame_locator(self, selector: str) -> FakeFrameLocator:
        return FakeFrameLocator(self.log, selector)


class FakeCDP:
    def __init__(self, log: list) -> None:
        self.log = log
        self.handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def send(self, method: str, params=None):
        self.log.append(("cdp", method, params))
        return {}

    def emit(self, event: str, payload) -> None:
        for handler in self.handlers.get(event, []):
            handler(payload)


class FakeContext:
    def __init__(self, log: list, scenario: dict, viewport: dict) -> None:
        self.log = log
        self.viewport = viewport
        self.page = FakePage(log, scenario)
        self.cdp = FakeCDP(log)
        self.init_scripts: list[str] = []
        self.closed = False

    def add_init_script(self, script: str) -> None:
        self.log.append(("init_script",))
        self.init_scripts.append(script)

    def new_page(self) -> FakePage:
        self.log.append(("new_page",))
        return self.page

    def new_cdp_session(self, page) -> FakeCDP:
        assert page is self.page
        return self.cdp

    def close(self) -> None:
        self.log.append(("context.close", id(self)))
        self.closed = True


class FakeBrowser:
    def __init__(self, log: list, scenario: dict | None = None) -> None:
        self.log = log
        self.scenario = scenario or {}
        self.contexts: list[FakeContext] = []

    def new_context(self, **kwargs) -> FakeContext:
        self.log.append(("new_context", kwargs))
        ctx = FakeContext(self.log, self.scenario, kwargs.get("viewport"))
        self.contexts.append(ctx)
        return ctx


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture
def env(monkeypatch, tmp_path):
    log: list = []
    requests: list = []

    def fake_urlopen(request, **_kwargs):
        requests.append(request)
        log.append(("restart", request.full_url))
        return FakeResponse()

    monkeypatch.setattr(video_take.urllib.request, "urlopen", fake_urlopen)

    def fake_ids(port, secret=None):
        log.append(("artifact_ids", port, secret))
        return {"a1", "a2"}

    monkeypatch.setattr(capture, "artifact_ids", fake_ids)
    stack = VideoStack(
        project_dir=tmp_path / "proj",
        artifact_port=4242,
        base_url="http://127.0.0.1:9999",
        operator_secret=SECRET,
        work_dir=tmp_path / "work",
    )
    return SimpleNamespace(log=log, requests=requests, stack=stack)


def _index(log: list, name: str) -> int:
    for i, entry in enumerate(log):
        if entry[0] == name:
            return i
    raise AssertionError(f"{name} not in log: {log}")


# ---------------------------------------------------------------------------
# start_take
# ---------------------------------------------------------------------------


def test_bootstrap_runs_the_fr2_sequence_in_order(env) -> None:
    browser = FakeBrowser(env.log)
    start_take(browser, env.stack, "dark")
    order = [
        "restart",
        "new_context",
        "init_script",
        "page.on",
        "goto",
        "settle",
        "widen",
        "click",
        "xterm_text",
        "artifact_ids",
        "cdp",
    ]
    positions = [_index(env.log, name) for name in order]
    assert positions == sorted(positions), list(zip(order, positions, strict=True))


def test_bootstrap_restart_posts_with_secret_header(env) -> None:
    start_take(FakeBrowser(env.log), env.stack, "dark")
    (request,) = env.requests
    assert request.full_url == "http://127.0.0.1:9999/api/terminal/restart"
    assert request.get_method() == "POST"
    assert request.get_header("X-osprey-terminal-secret") == SECRET


def test_bootstrap_restart_failure_names_the_step(env, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise video_take.urllib.error.URLError("refused")

    monkeypatch.setattr(video_take.urllib.request, "urlopen", boom)
    browser = FakeBrowser(env.log)
    with pytest.raises(ProbeFailed) as exc:
        start_take(browser, env.stack, "dark")
    assert exc.value.step == "restart"
    assert browser.contexts == []


def test_bootstrap_fresh_1920x1080_context_with_overlay(env) -> None:
    browser = FakeBrowser(env.log)
    take = start_take(browser, env.stack, "light")
    (ctx,) = browser.contexts
    assert ctx.viewport == {"width": 1920, "height": 1080}
    assert ctx.init_scripts == [OVERLAY_JS]
    assert take.context is ctx and take.page is ctx.page
    assert env.stack.current is take


def test_bootstrap_closes_the_previous_context(env) -> None:
    browser = FakeBrowser(env.log)
    first = start_take(browser, env.stack, "dark")
    second = start_take(browser, env.stack, "dark")
    assert browser.contexts[0].closed
    assert not browser.contexts[1].closed
    assert first.closed and env.stack.current is second
    close_at = _index(env.log, "context.close")
    restarts = [i for i, e in enumerate(env.log) if e[0] == "restart"]
    assert restarts[0] < close_at < restarts[1]
    # The first take's screencast is stopped before its context closes.
    stops = [i for i, e in enumerate(env.log) if e[:2] == ("cdp", "Page.stopScreencast")]
    assert stops and stops[0] < close_at


def test_bootstrap_loads_once_with_token_theme_and_the_default_rail(env) -> None:
    # The preset's own layout, with its vertical left rail.
    start_take(FakeBrowser(env.log), env.stack, "light")
    gotos = [e for e in env.log if e[0] == "goto"]
    assert len(gotos) == 1
    assert not [e for e in env.log if e[0] == "reload"]
    url = gotos[0][1]
    parsed = urllib.parse.urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}" == "http://127.0.0.1:9999"
    query = urllib.parse.parse_qs(parsed.query)
    assert query == {"token": [SECRET], "theme": ["light"]}
    assert urllib.parse.quote(SECRET, safe="") in url


def test_bootstrap_records_the_terminal_session_id(env) -> None:
    take = start_take(FakeBrowser(env.log), env.stack, "dark")
    assert take.session_id == "sess-123"


def test_bootstrap_missing_session_info_fails_the_take(env) -> None:
    browser = FakeBrowser(env.log, {"session_info": False})
    with pytest.raises(ProbeFailed) as exc:
        start_take(browser, env.stack, "dark")
    assert exc.value.step == "session-info"
    # The half-built take stays current so the next start_take closes it.
    assert env.stack.current is not None
    assert env.stack.current.context is browser.contexts[0]
    waits = [e for e in env.log if e[0] == "wait"]
    assert waits and all(e[1] == video_probes.POLL_MS for e in waits)


def test_bootstrap_waits_for_settled_layout_then_widens(env) -> None:
    start_take(FakeBrowser(env.log), env.stack, "dark")
    settle = next(e for e in env.log if e[0] == "settle")
    widen = next(e for e in env.log if e[0] == "widen")
    assert settle[1] > 0
    assert widen[1] == TERMINAL_WIDTH_FRACTION == pytest.approx(0.45)
    assert "bootLayoutSettled()" in DOCK_SETTLED_JS
    assert "requestAnimationFrame" in DOCK_SETTLED_JS
    assert "getDockApi()" in WIDEN_TERMINAL_JS
    assert "getPanel('terminal')" in WIDEN_TERMINAL_JS


@pytest.mark.parametrize("scenario", [{"settled": False}, {"widened": False}])
def test_bootstrap_layout_failure_names_the_step(env, scenario) -> None:
    with pytest.raises(ProbeFailed) as exc:
        start_take(FakeBrowser(env.log, scenario), env.stack, "dark")
    assert exc.value.step == "layout"
    assert not [e for e in env.log if e[0] == "click"]


def test_bootstrap_clicks_terminal_then_awaits_repl(env) -> None:
    browser = FakeBrowser(env.log, {"xterm": "Quick safety check\nYes, I trust this folder"})
    with pytest.raises(ProbeFailed) as exc:
        # The fake never reaches the REPL marker, so the probe runs out its budget.
        start_take(browser, env.stack, "dark")
    assert exc.value.step == "repl-ready"
    assert ("click", "#terminal-container") in env.log
    assert [e for e in env.log if e[0] == "press"] == [("press", "Enter")]
    assert not [e for e in env.log if e[0] == "cdp"]


def test_bootstrap_snapshots_artifacts_before_the_screencast(env) -> None:
    take = start_take(FakeBrowser(env.log), env.stack, "dark")
    assert take.before_ids == {"a1", "a2"}
    assert ("artifact_ids", 4242, SECRET) in env.log
    start = next(e for e in env.log if e[:2] == ("cdp", "Page.startScreencast"))
    assert start[2]["format"] == "jpeg"
    assert (start[2]["maxWidth"], start[2]["maxHeight"]) == (1920, 1080)
    assert take.screencasting


def test_a_frame_after_the_screencast_stopped_is_dropped(env) -> None:
    # The browser can deliver one last frame after the take stopped recording,
    # even after its context closed: acking it would fail, and writing it
    # would add footage to a take that is already finished.
    from playwright.sync_api import Error as PlaywrightError

    browser = FakeBrowser(env.log)
    take = start_take(browser, env.stack, "dark")
    take.close()
    cdp = browser.contexts[0].cdp

    def closed(_method, _params=None):
        raise PlaywrightError("CDPSession.send: Target page, context or browser has been closed")

    cdp.send = closed
    cdp.emit(
        "Page.screencastFrame",
        {
            "data": base64.b64encode(b"late").decode(),
            "metadata": {"timestamp": 1.0},
            "sessionId": 9,
        },
    )
    assert take.sink.frames == []


def test_bootstrap_screencast_frames_reach_the_sink_and_are_acked(env) -> None:
    browser = FakeBrowser(env.log)
    take = start_take(browser, env.stack, "dark")
    cdp = browser.contexts[0].cdp
    cdp.emit(
        "Page.screencastFrame",
        {
            "data": base64.b64encode(b"jpeg-bytes").decode(),
            "metadata": {"timestamp": 1000.0},
            "sessionId": 7,
        },
    )
    assert ("cdp", "Page.screencastFrameAck", {"sessionId": 7}) in env.log
    ((path, t),) = take.sink.frames
    assert t == 1000.0
    assert path.read_bytes() == b"jpeg-bytes"
    assert take.frames_dir in path.parents
    assert env.stack.work_dir in path.parents


def test_bootstrap_close_stops_screencast_and_is_idempotent(env) -> None:
    browser = FakeBrowser(env.log)
    take = start_take(browser, env.stack, "dark")
    take.close()
    take.close()
    stops = [e for e in env.log if e[:2] == ("cdp", "Page.stopScreencast")]
    closes = [e for e in env.log if e[0] == "context.close"]
    assert len(stops) == 1 and len(closes) == 1
    assert browser.contexts[0].closed


def test_bootstrap_build_dir_is_under_the_project(env) -> None:
    assert env.stack.build_dir == env.stack.project_dir / "build"


def test_bootstrap_video_take_never_sleeps() -> None:
    source = Path(video_take.__file__).read_text()
    assert not re.search(r"\btime\.sleep\b|\bimport time\b", source)


# ---------------------------------------------------------------------------
# capture.web_terminal: the shared ``osprey web`` launch
# ---------------------------------------------------------------------------


def _stub_web_launch(monkeypatch, *, ready: bool = True):
    monkeypatch.setattr(capture, "verify_vendor_bundles", lambda: ([], []))
    popen = Mock()
    run = Mock()
    monkeypatch.setattr(capture.subprocess, "Popen", popen)
    monkeypatch.setattr(capture.subprocess, "run", run)
    monkeypatch.setattr(capture, "free_port", lambda: 5151)
    monkeypatch.setattr(capture, "mint_secret", lambda: SECRET)

    def wait_for_port(*_args, **_kwargs):
        if not ready:
            raise RuntimeError("never listened")

    monkeypatch.setattr(capture, "wait_for_port", wait_for_port)
    return popen, run


def test_bootstrap_web_terminal_exposes_secret_and_disables_tour(monkeypatch, tmp_path) -> None:
    popen, run = _stub_web_launch(monkeypatch)
    with capture.web_terminal(tmp_path) as web:
        assert web.port == 5151
        assert web.base_url == "http://127.0.0.1:5151"
        assert web.operator_secret == SECRET
        run.assert_not_called()
    (args,), kwargs = popen.call_args
    assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(web.claude_config_dir)
    assert args == ["osprey", "web", "--repo", str(tmp_path), "--detach", "--port", "5151"]
    child_env = kwargs["env"]
    assert child_env[capture.OPERATOR_SECRET_ENV] == SECRET
    assert child_env["OSPREY_WEB_TOUR"] == "never"
    (stop_args,), _ = run.call_args
    assert stop_args == ["osprey", "web", "stop", "--repo", str(tmp_path)]
    popen.return_value.terminate.assert_called_once()


def test_bootstrap_web_terminal_not_ready_skips_and_tears_down(monkeypatch, tmp_path) -> None:
    popen, run = _stub_web_launch(monkeypatch, ready=False)
    with pytest.raises(capture.ScreenshotSkip), capture.web_terminal(tmp_path):
        pytest.fail("body must not run")
    run.assert_called_once()
    popen.return_value.terminate.assert_called_once()


def test_bootstrap_web_terminal_missing_cli_skips(monkeypatch, tmp_path) -> None:
    popen, run = _stub_web_launch(monkeypatch)
    popen.side_effect = FileNotFoundError("osprey")
    with pytest.raises(capture.ScreenshotSkip), capture.web_terminal(tmp_path):
        pytest.fail("body must not run")
    run.assert_not_called()


def test_bootstrap_web_terminal_stop_failure_warns(monkeypatch, tmp_path, capsys) -> None:
    _, run = _stub_web_launch(monkeypatch)
    run.side_effect = capture.subprocess.CalledProcessError(1, "osprey web stop")
    with capture.web_terminal(tmp_path):
        pass
    assert "WARNING: `osprey web stop` failed" in capsys.readouterr().out


def test_bootstrap_web_terminal_isolates_claude_code_config(monkeypatch, tmp_path) -> None:
    # The session must not load the operator's own hooks, plugins, output style
    # or memory: it gets a fresh, pre-seeded config dir of its own.
    popen, _ = _stub_web_launch(monkeypatch)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/operator/own/claude")
    with capture.web_terminal(tmp_path):
        config_dir = Path(popen.call_args.kwargs["env"]["CLAUDE_CONFIG_DIR"])
        assert config_dir != Path("/operator/own/claude")
        state = json.loads((config_dir / ".claude.json").read_text())
        assert state["hasCompletedOnboarding"] is True
        render_dir = str((tmp_path / "build").resolve())
        assert state["projects"][render_dir]["hasTrustDialogAccepted"] is True
    assert not config_dir.exists(), "the isolated config dir is removed on exit"


def _write_web_pid(project_dir, pid: int) -> None:
    from osprey.cli.web_cmd import PID_FILE

    pid_path = project_dir / PID_FILE
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n")


def test_bootstrap_web_terminal_kills_a_server_that_outlives_its_stop(monkeypatch, tmp_path):
    _stub_web_launch(monkeypatch)
    _write_web_pid(tmp_path, 4242)
    monkeypatch.setattr(capture, "_WEB_STOP_GRACE_S", 0.0)
    signals: list[tuple[int, int]] = []

    def kill(pid, sig):
        signals.append((pid, sig))  # signal 0 probes: the server is still alive

    monkeypatch.setattr(capture.os, "kill", kill)
    with capture.web_terminal(tmp_path):
        pass
    assert (4242, capture.signal.SIGKILL) in signals


def test_bootstrap_web_terminal_leaves_a_stopped_server_alone(monkeypatch, tmp_path):
    _stub_web_launch(monkeypatch)
    _write_web_pid(tmp_path, 4242)
    signals: list[tuple[int, int]] = []

    def kill(pid, sig):
        signals.append((pid, sig))
        raise ProcessLookupError(pid)

    monkeypatch.setattr(capture.os, "kill", kill)
    with capture.web_terminal(tmp_path):
        pass
    assert (4242, capture.signal.SIGKILL) not in signals


# ---------------------------------------------------------------------------
# record_take / record_theme: the FR1 storyboard against a scripted world
# ---------------------------------------------------------------------------

ENTRY_CREATE = ("mcp__ariel__entry_create", "tool=entry_create", "tu-create")
ENTRY_PUBLISH = ("mcp__ariel__entry_publish", "tool=entry_publish", "tu-publish")
SELECTIVE = ("execute", None, "tu-exec")
UNNAMED = (None, None, "tu-anon")

BEATS = [
    ("focus-1", False),
    ("prompt-1", False),
    ("plot", True),
    # The plot card itself arrives while the agent works, and is opened.
    ("open-artifact", False),
    ("plot", True),
    ("open-plot", False),
    ("rotate", False),
    ("focus-2", False),
    ("prompt-2", False),
    ("correlate", True),
    # The correlation plot's card lands while the agent works, and is opened.
    ("open-artifact", False),
    ("correlate", True),
    ("open-correlation", False),
    ("focus-3", False),
    ("prompt-3", False),
    ("post", True),
    ("approval-hold", False),
    ("approve", False),
    ("draft", True),
    ("close", False),
]


def _ask_line(tool, detail, tool_use_id) -> str:
    rec = {"ts": "2026-09-24T10:00:00", "hook": "approval", "status": "ask"}
    if tool is not None:
        rec["tool"] = tool
    if detail is not None:
        rec["detail"] = detail
    if tool_use_id is not None:
        rec["tool_use_id"] = tool_use_id
    return json.dumps(rec) + "\n"


class FakePlotFrame:
    def __init__(self, world=None) -> None:
        self.world = world

    def frame_element(self):
        from playwright.sync_api import Error as PlaywrightError

        plan = self.world.plan if self.world is not None else {}
        if plan.get("browser_error"):
            raise PlaywrightError("Target page, context or browser has been closed")
        if self.world is not None and self.world.detach_pending:
            self.world.detach_pending = False
            raise PlaywrightError("Frame.frame_element: Frame has been detached.")
        if self.world is not None and self.world.boxless_pending:
            self.world.boxless_pending = False
            return SimpleNamespace(bounding_box=lambda: None)
        return SimpleNamespace(
            bounding_box=lambda: {"x": 1000.0, "y": 100.0, "width": 800.0, "height": 600.0}
        )


class World:
    """The agent, artifact server, hook log and transcript, scripted per take.

    Each take plan is a dict; keys (all optional):
    ``plot`` ("3d" | "2d"), ``plot_error``, ``ask_during_plot``, ``asks`` (list
    of (tool, detail, tool_use_id)), ``post_error``, ``draft_after_enters``
    (approval Enters needed before the draft appears; 0 = never),
    ``tool_result`` ("entry" | "block"), ``late_ask``, ``drag_deg``,
    ``arrivals`` (``(id, artifact_type, seconds_after_prompt)`` cards that
    land in the gallery before the plot), ``no_card`` / ``no_preview`` (ids
    whose card never shows / whose preview never renders),
    ``dialog_at`` (``(seconds, screen)`` shown while the agent keeps working),
    ``empty_plot_card`` (the agent saves a plot with nothing drawn in it),
    ``no_attachment`` (the draft shows no attached image to scroll to),
    ``corr`` ("ok" | "missing" | "3d" | "error": what the correlation prompt
    yields), ``attach`` (which plots the entry_create call attaches: a list of
    "corr" / "3d", or None for a call with no ``artifact_ids``),
    ``detach_once`` (the gallery replaces the plot preview once, detaching the
    frame the take holds), ``detach_on_open`` (the same while the plot is
    opened), ``boxless_once`` (the replaced preview has no on-screen box yet),
    ``browser_error`` (every plot frame call fails), ``drag_s`` (seconds each
    drag takes in the browser),
    ``stuck`` (the agent never answers the plot prompt), ``plot_dialog`` and
    ``post_dialog`` (screen text shown instead of an answer), ``busy_s`` (the
    screen keeps changing, the plot lands after this many seconds).
    """

    def __init__(self, stack: VideoStack, tmp_path: Path, plans: list[dict]) -> None:
        self.t = 1000.0
        self.plans = plans
        self.take = -1
        self.focus = True
        self.hook_log = video_take.video_check_approval.hook_log_path(stack.project_dir)
        self.hook_log.parent.mkdir(parents=True, exist_ok=True)
        self.transcript = tmp_path / "transcript.jsonl"
        self.transcript.write_text("")
        self.artifacts: list[dict] = []
        self.html: dict[str, str] = {}
        self.turn = "running"
        self.draft = False
        self.azimuth = 0.0
        self.pending: list = []
        self.screen: str | None = None
        self.detach_pending = False
        self.boxless_pending = False
        self.detach_on_open = False
        self.log: list = []
        self.asked = False
        self.approval_enters = 0

    @property
    def plan(self) -> dict:
        return self.plans[self.take]

    def clock(self) -> float:
        return self.t

    def later(self, seconds: float, fn) -> None:
        self.pending.append((self.t + seconds, fn))

    def advance(self, ms) -> None:
        self.t += ms / 1000
        due = [p for p in self.pending if p[0] <= self.t]
        self.pending = [p for p in self.pending if p[0] > self.t]
        for _, fn in due:
            fn()

    def write_asks(self, *asks) -> None:
        with self.hook_log.open("a") as fh:
            fh.writelines(_ask_line(*a) for a in asks)
        self.asked = True

    def _set(self, **kw):
        return lambda: self.__dict__.update(kw)

    def _stamp(self) -> str:
        return datetime.fromtimestamp(self.t, tz=UTC).isoformat()

    def _add_card(self, artifact_id: str, artifact_type: str) -> None:
        self.artifacts.append(
            {
                "id": artifact_id,
                "filename": f"{artifact_id}.dat",
                "artifact_type": artifact_type,
                "tool_source": "agent",
                "title": f"title of {artifact_id}",
                "timestamp": self._stamp(),
            }
        )

    def _add_empty_plot(self) -> None:
        artifact_id = f"scratch-{self.take}"
        self.artifacts.append(
            {
                "id": artifact_id,
                "filename": "scratch.html",
                "artifact_type": "plot_html",
                "tool_source": "create_interactive_plot",
                "title": "scratch probe",
                "timestamp": self._stamp(),
            }
        )
        self.html[f"/files/{artifact_id}/scratch.html"] = (
            '<script>Plotly.newPlot("g", [{"type": "scatter", "x": [], "y": []}], {}, {});</script>'
        )

    def clicked(self, artifact_id: str) -> bool:
        return any(e[0] == "click" and f'data-id="{artifact_id}"' in e[1] for e in self.log)

    def card_shown(self, artifact_id: str) -> bool:
        ids = {a["id"] for a in self.artifacts}
        return artifact_id in ids and artifact_id not in self.plan.get("no_card", ())

    def selected(self) -> str | None:
        clicks = [e[1] for e in self.log if e[0] == "click" and 'data-id="' in e[1]]
        return clicks[-1].split('data-id="')[1].split('"')[0] if clicks else None

    def preview_shows(self, artifact_id: str) -> bool:
        # The preview pane shows the card clicked last.
        return self.selected() == artifact_id and artifact_id not in self.plan.get("no_preview", ())

    def _add_plot(self) -> None:
        artifact_id = f"plot-{self.take}"
        filename = "plot.html"
        self.artifacts.append(
            {
                "id": artifact_id,
                "filename": filename,
                "artifact_type": "plot_html",
                "tool_source": "create_interactive_plot",
                "title": "the plot",
                "timestamp": self._stamp(),
            }
        )
        trace = "scatter3d" if self.plan.get("plot", "3d") == "3d" else "scatter"
        self.html[f"/files/{artifact_id}/{filename}"] = f"{trace} BPM1 BPM2 BPM3"
        self.turn = "ended"

    def _add_corr(self) -> None:
        artifact_id = f"corr-{self.take}"
        self.artifacts.append(
            {
                "id": artifact_id,
                "filename": "corr.html",
                "artifact_type": "plot_html",
                "tool_source": "create_interactive_plot",
                "title": "the correlation",
                "timestamp": self._stamp(),
            }
        )
        kind = "heatmap" if self.plan.get("corr", "ok") != "3d" else "scatter3d"
        labels = [f"SR:DIAG:BPM:0{i}:POSITION:X" for i in range(1, 5)]
        labels.append("SR:VAC:GAUGE:SR01:PRESSURE:RB")
        trace = {"type": kind, "x": labels, "y": labels, "z": [[1]]}
        self.html[f"/files/{artifact_id}/corr.html"] = (
            f'<script>Plotly.newPlot("g", {json.dumps([trace])}, {{}}, {{}});</script>'
        )
        self.turn = "ended"

    def _write_entry_call(self) -> None:
        attach = self.plan.get("attach", ["corr"])
        if attach is None:
            tool_input = {"subject": "s"}
        else:
            ids = {"corr": f"corr-{self.take}", "3d": f"plot-{self.take}"}
            tool_input = {"subject": "s", "artifact_ids": [ids[a] for a in attach]}
        use = {"type": "tool_use", "id": "tu-create", "name": ENTRY_CREATE[0], "input": tool_input}
        entry = {"type": "assistant", "message": {"role": "assistant", "content": [use]}}
        with self.transcript.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def on_type(self, text: str) -> None:
        if text == video_take.PROMPT_CORR:
            self.turn = "running"
            corr = self.plan.get("corr", "ok")
            if corr == "error":
                self.later(2, self._set(turn="errored"))
            elif corr == "stuck":
                pass
            elif corr == "missing":
                self.later(4, self._set(turn="ended"))
            else:
                self.later(4, self._add_corr)
            if self.plan.get("card_after_corr"):
                self.later(4.5, lambda: self._add_card(f"tbl-{self.take}", "markdown"))
            return
        if text == video_take.PROMPT_PLOT:
            self.take += 1
            self.pending = []
            self.turn, self.draft, self.asked, self.approval_enters = "running", False, False, 0
            self.azimuth = 0.0
            self.detach_pending = bool(self.plan.get("detach_once"))
            self.boxless_pending = bool(self.plan.get("boxless_once"))
            self.detach_on_open = bool(self.plan.get("detach_on_open"))
            if self.plan.get("plot_error"):
                self.later(2, self._set(turn="errored"))
                return
            if self.plan.get("ask_during_plot"):
                self.later(1, lambda: self.write_asks(ENTRY_CREATE))
            if self.plan.get("dialog_at"):
                at_s, text = self.plan["dialog_at"]
                self.later(at_s, self._set(screen=text))
            if self.plan.get("empty_plot_card"):
                self.later(1.0, self._add_empty_plot)
            for artifact_id, artifact_type, at_s in self.plan.get("arrivals", ()):
                self.later(
                    at_s,
                    lambda i=artifact_id, k=artifact_type: self._add_card(f"{i}-{self.take}", k),
                )
            if self.plan.get("plot_dialog"):
                self.later(3, self._set(screen=self.plan["plot_dialog"]))
                return
            if self.plan.get("stuck"):
                return
            if self.plan.get("busy_s"):
                for i in range(int(self.plan["busy_s"]) // 10):
                    self.later(10 * (i + 1), self._set(screen=f"⏺ step {i}"))
                self.later(self.plan["busy_s"], self._add_plot)
                return
            self.later(5, self._add_plot)
        elif text == video_take.PROMPT_POST:
            self.turn = "running"
            if self.plan.get("post_error"):
                self.later(2, self._set(turn="errored"))
                return
            if self.plan.get("post_dialog"):
                self.later(3, self._set(screen=self.plan["post_dialog"]))
                return
            self._write_entry_call()
            self.later(3, lambda: self.write_asks(*self.plan.get("asks", [ENTRY_CREATE])))

    def on_press(self, key: str) -> None:
        if key != "Enter" or not self.asked:
            return
        self.approval_enters += 1
        kind = self.plan.get("tool_result")
        if kind and self.approval_enters == 1:
            if kind == "entry":
                entry = {"type": "tool_result", "tool_use_id": "tu-create"}
            else:
                entry = {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "tu-create"}],
                    },
                }
            with self.transcript.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
        if self.plan.get("late_ask") and self.approval_enters == 1:
            self.later(1, lambda: self.write_asks(("mcp__ariel__entry_create", None, "tu-2")))
        if self.approval_enters == self.plan.get("draft_after_enters", 1):
            self.later(1, self._set(draft=True))

    def camera(self) -> float:
        if self.detach_on_open:
            from playwright.sync_api import Error as PlaywrightError

            self.detach_on_open = False
            raise PlaywrightError("Frame.evaluate: Frame was detached")
        return self.azimuth

    def on_drag(self) -> None:
        self.t += self.plan.get("drag_s", 0.0)
        self.azimuth += self.plan.get("drag_deg", 120.0)


@pytest.fixture
def world_env(env, monkeypatch, tmp_path):
    holder: dict = {}

    def make(*plans: dict) -> World:
        world = World(env.stack, tmp_path, list(plans))
        world.log = env.log
        holder["world"] = world
        return world

    def w() -> World:
        return holder["world"]

    monkeypatch.setattr(capture, "osprey_version", lambda: "9.9-test")

    def fetch_artifacts(_port, secret=None):
        holder["artifact_secrets"].add(secret)
        return list(w().artifacts)

    def fetch_html(_port, path, secret=None):
        holder["artifact_secrets"].add(secret)
        return w().html[path]

    def artifact_ids(_port, secret=None):
        holder["artifact_secrets"].add(secret)
        return {a["id"] for a in w().artifacts}

    holder["artifact_secrets"] = set()
    monkeypatch.setattr(capture, "fetch_artifacts", fetch_artifacts)
    monkeypatch.setattr(capture, "artifact_ids", artifact_ids)
    monkeypatch.setattr(video_take, "_fetch_html", fetch_html)

    def transcript_path(build, sid, config_dir=None):
        holder["transcript_lookup"] = (build, sid, config_dir)
        return w().transcript

    monkeypatch.setattr(video_take.video_check_turn, "transcript_path", transcript_path)
    env.lookups = holder
    monkeypatch.setattr(
        video_take.video_check_turn, "turn_state", lambda path, submitted, **kw: w().turn
    )
    monkeypatch.setattr(video_probes, "plot_frame", lambda page: FakePlotFrame(w()))
    monkeypatch.setattr(video_probes, "camera_azimuth", lambda frame: w().camera())
    monkeypatch.setattr(video_probes, "arm_relayout", lambda frame: True)
    monkeypatch.setattr(video_probes, "await_relayout", lambda frame, timeout_ms=2000: True)
    monkeypatch.setattr(video_probes, "draft_ready", lambda page: w().draft)
    monkeypatch.setattr(video_probes, "card_shown", lambda page, i: w().card_shown(i))
    monkeypatch.setattr(video_probes, "preview_shows", lambda page, i: w().preview_shows(i))
    monkeypatch.setattr(video_probes, "preview_state", lambda page: "preview: empty")
    monkeypatch.setattr(
        video_probes,
        "reveal_attachment",
        lambda page: env.log.append(("reveal-attachment",)) or not w().plan.get("no_attachment"),
    )
    env.make_world = make
    return env


def _run_take(env, world: World, theme: str = "dark"):
    browser = FakeBrowser(env.log, {"world": world})
    take = start_take(browser, env.stack, theme)
    timeline = Timeline(theme=theme, clock=world.clock, version="x")
    env.log.clear()
    return take, timeline


def _enters(log: list) -> int:
    return sum(1 for e in log if e == ("press", "Enter"))


def test_record_take_runs_the_fr1_beats_in_order(world_env) -> None:
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert [(s.name, s.fast_forward) for s in timeline.steps] == BEATS
    assert all(s.error is None and s.caption for s in timeline.steps)
    starts = [s.start for s in timeline.steps]
    assert starts == sorted(starts)
    typed = [e for e in world_env.log if e[0] == "type"]
    assert [e[1] for e in typed] == [
        video_take.PROMPT_PLOT,
        video_take.PROMPT_CORR,
        video_take.PROMPT_POST,
    ]
    assert all(e[2] and e[2] > 0 for e in typed)
    # Three prompt submissions plus exactly one approval.
    assert _enters(world_env.log) == 4
    steps = {s.name: s for s in timeline.steps}
    assert steps["approval-hold"].duration == pytest.approx(video_take.APPROVAL_HOLD_MS / 1000)
    assert steps["close"].duration == pytest.approx(video_take.CLOSE_HOLD_MS / 1000)
    # The speed-up badge is burned in at encode time; the page is never asked.
    assert not [e for e in world_env.log if e[0] == "badge"]
    # The artifact is opened from the artifacts panel before the drag.
    click = _index(world_env.log, "click")
    assert any(
        e[0] == "click" and 'data-id="plot-0"' in e[1] and video_probes.ARTIFACTS_IFRAME in e[1]
        for e in world_env.log
    )
    assert click < _index(world_env.log, "mouse.down")


def test_record_take_reads_the_session_transcript_from_its_config_dir(world_env, tmp_path):
    world_env.stack.claude_config_dir = tmp_path / "session-config"
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert world_env.lookups["transcript_lookup"] == (
        world_env.stack.build_dir,
        "sess-123",
        tmp_path / "session-config",
    )


def test_record_take_authenticates_to_the_artifact_server(world_env) -> None:
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert world_env.lookups["artifact_secrets"] == {SECRET}


def test_bootstrap_snapshots_artifacts_with_the_operator_secret(env) -> None:
    start_take(FakeBrowser(env.log), env.stack, "dark")
    assert ("artifact_ids", 4242, SECRET) in env.log


ARRIVALS = [("cf", "markdown", 1.0), ("arch", "json", 3.0)]


def _clicked_ids(log: list) -> list[str]:
    return [
        e[1].split('data-id="')[1].split('"')[0]
        for e in log
        if e[0] == "click" and 'data-id="' in e[1]
    ]


def test_record_take_opens_each_new_card_in_arrival_order(world_env) -> None:
    world = world_env.make_world({"arrivals": ARRIVALS})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    # Channel finder, archiver, the plot as it lands, the plot beat; then the
    # correlation plot as it lands, still on screen for its own beat.
    assert _clicked_ids(world_env.log) == ["cf-0", "arch-0", "plot-0", "plot-0", "corr-0"]
    names = [s.name for s in timeline.steps]
    assert names[:8] == [
        "focus-1",
        "prompt-1",
        "plot",
        "open-artifact",
        "plot",
        "open-artifact",
        "plot",
        "open-artifact",
    ]
    opens = [s for s in timeline.steps if s.name == "open-artifact"]
    assert all(not s.fast_forward for s in opens)
    # Each opening keeps the caption of the wait it interrupts.
    assert [s.caption for s in opens] == ["The agent reads the archiver and plots"] * 3 + [
        "The agent correlates the BPMs with the vacuum"
    ]
    assert all(s.duration >= video_take.OPEN_HOLD_MS / 1000 for s in opens)
    assert timeline.steps[-1].name == "close"


def test_record_take_holds_each_preview_only_after_it_renders(world_env) -> None:
    world = world_env.make_world({"arrivals": ARRIVALS[:1]})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    log = world_env.log
    click = next(i for i, e in enumerate(log) if e[0] == "click" and "cf-0" in e[1])
    assert ("wait", video_take.OPEN_HOLD_MS) in log[click:]


def test_record_take_never_clicks_cards_from_before_the_prompt(world_env) -> None:
    world = world_env.make_world({"arrivals": ARRIVALS})
    world.artifacts.append({"id": "a1", "artifact_type": "markdown", "timestamp": "2026"})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert "a1" not in _clicked_ids(world_env.log)


@pytest.mark.parametrize(
    ("plan", "budget_ms"),
    [
        ({"arrivals": ARRIVALS, "no_preview": ["arch-0"]}, "PREVIEW_BUDGET_MS"),
        ({"arrivals": ARRIVALS, "no_card": ["arch-0"]}, "CARD_BUDGET_MS"),
    ],
    ids=["preview-never-renders", "card-never-shows"],
)
def test_record_take_a_card_that_never_opens_fails_fast(world_env, plan, budget_ms) -> None:
    world = world_env.make_world(plan)
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "open-artifact"
    assert "title of arch-0" in exc.value.detail
    assert "preview: empty" in exc.value.detail and "? for shortcuts" in exc.value.detail
    last = timeline.steps[-1]
    assert last.name == "open-artifact"
    assert last.duration <= getattr(video_take, budget_ms) / 1000 + 1


def test_record_take_an_empty_plot_card_fails_the_take_before_it_is_shown(world_env) -> None:
    # A plot card with nothing drawn would put empty axes on screen; the take
    # is discarded and retaken rather than published.
    world = world_env.make_world({"empty_plot_card": True})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "plot"
    assert "empty plot" in exc.value.detail and "scratch probe" in exc.value.detail
    assert "scratch-0" not in _clicked_ids(world_env.log)


def test_record_take_a_dialog_while_a_card_opens_fails_at_once(world_env) -> None:
    world = world_env.make_world(
        {
            "arrivals": ARRIVALS,
            "no_preview": ["cf-0"],
            "dialog_at": (1.5, "Notice … Enter to continue"),
        }
    )
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "open-artifact"
    assert "dialog" in exc.value.detail
    assert timeline.steps[-1].duration < 3


# --- the correlation beat ------------------------------------------------------


def _step(timeline, name: str):
    return next(s for s in timeline.steps if s.name == name)


def test_record_take_shows_the_correlation_plot_before_posting(world_env) -> None:
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    typed = [e[1] for e in world_env.log if e[0] == "type"]
    assert typed.index(video_take.PROMPT_CORR) < typed.index(video_take.PROMPT_POST)
    shown = _step(timeline, "open-correlation")
    assert shown.caption == "A correlation plot"
    assert shown.duration >= video_take.CORR_HOLD_MS / 1000
    waits = [s for s in timeline.steps if s.name == "correlate"]
    assert waits and all(s.fast_forward for s in waits)
    assert {s.caption for s in waits} == {"The agent correlates the BPMs with the vacuum"}
    assert _step(timeline, "prompt-2").caption == "Ask for a correlation"
    assert _step(timeline, "prompt-3").caption == "Post it to the logbook"


def test_the_correlation_beat_reclicks_only_when_another_card_took_the_preview(world_env):
    world = world_env.make_world({"card_after_corr": True})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    ids = _clicked_ids(world_env.log)
    assert ids[-3:] == ["corr-0", "tbl-0", "corr-0"]
    assert timeline.steps[-1].name == "close"


def test_the_correlation_prompt_names_sector_bpms_and_vacuum() -> None:
    prompt = video_take.PROMPT_CORR.lower()
    assert "correlation" in prompt and "first sector" in prompt
    assert "horizontal bpms" in prompt and "vacuum" in prompt


@pytest.mark.parametrize(
    ("corr", "detail"),
    [
        ("missing", "without a new correlation plot"),
        ("3d", "without a new correlation plot"),
        ("error", "API error"),
    ],
)
def test_record_take_no_correlation_plot_fails_the_correlate_beat(world_env, corr, detail):
    world = world_env.make_world({"corr": corr})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "correlate"
    assert detail in exc.value.detail
    assert timeline.steps[-1].duration < 10


def test_record_take_a_silent_correlation_wait_fails_at_the_stall_threshold(world_env):
    world = world_env.make_world({"corr": "stuck"})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "correlate"
    assert "no progress" in exc.value.detail
    assert timeline.steps[-1].duration < video_take.STALL_S + 2


@pytest.mark.parametrize(
    ("attach", "detail"),
    [
        (["3d"], "not the correlation plot"),
        (["corr", "3d"], "the 3D plot too"),
        (None, "no artifact_ids"),
    ],
    ids=["3d-only", "both", "none"],
)
def test_record_take_the_entry_must_attach_only_the_correlation_plot(world_env, attach, detail):
    # Checked before the approving Enter, so a wrong entry is never created.
    world = world_env.make_world({"attach": attach})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "post"
    assert detail in exc.value.detail
    assert _enters(world_env.log) == 3


def test_record_take_the_closing_hold_shows_the_attachment(world_env) -> None:
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    log = world_env.log
    reveal = log.index(("reveal-attachment",))
    close_caption = log.index(("caption", "Draft saved with the plot attached"))
    assert close_caption < reveal
    assert ("wait", video_take.CLOSE_HOLD_MS) in log[reveal:]


def test_record_take_fails_the_close_when_the_draft_has_no_attachment(world_env) -> None:
    # The caption says the plot is attached; a draft without it fails the take.
    world = world_env.make_world({"no_attachment": True})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "close" and "attachment" in exc.value.detail


def test_record_take_announces_each_beat(world_env, capsys) -> None:
    # A live run is followed from its log: each beat is named as it starts.
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("dark: ")] == [
        f"dark: {name}" for name, _ in BEATS
    ]


def test_record_take_checks_focus_before_typing(world_env) -> None:
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    world.focus = False
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "prompt-1"
    assert not [e for e in world_env.log if e[0] in ("type", "press")]
    assert timeline.steps[-1].name == "prompt-1" and timeline.steps[-1].error


def test_record_take_rotates_at_least_90_degrees(world_env) -> None:
    world = world_env.make_world({"drag_deg": 50.0})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert sum(1 for e in world_env.log if e[0] == "mouse.up") == 2


def test_record_take_drags_in_one_glide_then_lets_the_camera_settle(world_env) -> None:
    # The rotate beat plays in real time: every browser round trip inside the
    # drag lengthens the video, so the glide carries no waits of its own.
    world = world_env.make_world({})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    log = world_env.log
    down, up = _index(log, "mouse.down"), _index(log, "mouse.up")
    assert not [e for e in log[down:up] if e[0] == "wait"]
    assert sum(1 for e in log[down:up] if e[0] == "mouse.move") == video_take.DRAG_STEPS
    assert log[up + 1] == ("wait", video_take.DRAG_SETTLE_MS)


def test_record_take_rotate_survives_a_replaced_plot_preview(world_env) -> None:
    # The gallery re-renders the preview (a focus change, a list refresh), and
    # the frame the take held is detached: the take finds the new one.
    world = world_env.make_world({"detach_once": True})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert timeline.steps[-1].name == "close"
    assert not world.detach_pending


@pytest.mark.parametrize("plan", [{"detach_on_open": True}, {"boxless_once": True}])
def test_record_take_a_preview_replaced_mid_swap_is_found_again(world_env, plan) -> None:
    world = world_env.make_world(plan)
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert timeline.steps[-1].name == "close"


def test_record_theme_turns_a_browser_error_into_a_failed_step(world_env, capsys) -> None:
    # A browser call that gives up mid-take fails the take like a failed check,
    # naming the beat it was in, so the caller retries it on a fresh stack.
    world = world_env.make_world({"browser_error": True})
    world_env.stack.browser = FakeBrowser(world_env.log, {"world": world})
    with pytest.raises(ProbeFailed) as exc:
        record_theme(world_env.stack, "dark", clock=world.clock)
    assert exc.value.step == "open-plot" and "browser error" in exc.value.detail
    assert world_env.stack.current is None
    assert all(c.closed for c in world_env.stack.browser.contexts)
    assert "WARNING" not in capsys.readouterr().out


def test_record_take_a_slow_rotation_fails_fast_instead_of_filling_the_video(world_env):
    # The rotate beat plays in real time: a browser that crawls under load
    # would stretch the video by minutes, so the take is discarded instead.
    world = world_env.make_world({"drag_deg": 50.0, "drag_s": 9.0})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "rotate"
    assert "real-time budget" in exc.value.detail
    assert timeline.steps[-1].duration <= video_take.ROTATE_BUDGET_S + 10


def test_record_take_logs_each_drag(world_env, capsys) -> None:
    world = world_env.make_world({"drag_deg": 50.0})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    drags = [line for line in capsys.readouterr().out.splitlines() if "rotate drag" in line]
    assert len(drags) == 2 and "deg" in drags[0]


def test_record_take_too_small_rotation_fails(world_env) -> None:
    world = world_env.make_world({"drag_deg": 20.0})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "rotate"
    assert sum(1 for e in world_env.log if e[0] == "mouse.up") == video_take.MAX_DRAGS


def test_record_take_any_ask_while_plotting_fails(world_env) -> None:
    world = world_env.make_world({"ask_during_plot": True})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "plot"
    assert _enters(world_env.log) == 1


def test_record_take_ignores_asks_logged_before_prompt_1(world_env) -> None:
    world = world_env.make_world({})
    world.write_asks(ENTRY_PUBLISH, UNNAMED)
    world.asked = False
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert timeline.steps[-1].name == "close"


@pytest.mark.parametrize(
    ("plan", "step_name"), [({"plot_error": True}, "plot"), ({"post_error": True}, "post")]
)
def test_record_take_errored_turn_fails_at_once(world_env, plan, step_name) -> None:
    world = world_env.make_world(plan)
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == step_name
    # Failed within a few seconds of the error, not at the end of the budget.
    assert timeline.steps[-1].duration < 5


def test_record_take_a_silent_agent_fails_at_the_stall_threshold(world_env) -> None:
    world = world_env.make_world({"stuck": True})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "plot"
    assert "no progress" in exc.value.detail
    assert "? for shortcuts" in exc.value.detail
    duration = timeline.steps[-1].duration
    assert video_take.STALL_S <= duration < video_take.STALL_S + 2


@pytest.mark.parametrize(
    ("plan", "step_name"),
    [
        ({"plot_dialog": "Billing notice … Enter to continue · Esc to cancel"}, "plot"),
        ({"plot_dialog": "Do you want to proceed?\n❯ 1. Yes"}, "plot"),
        ({"post_dialog": "Billing notice … Enter to continue"}, "post"),
    ],
    ids=["plot-notice", "plot-permission", "post-notice"],
)
def test_record_take_an_unanswerable_dialog_fails_at_once(world_env, plan, step_name) -> None:
    world = world_env.make_world(plan)
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == step_name
    assert "dialog" in exc.value.detail and world.screen.splitlines()[0] in exc.value.detail
    assert timeline.steps[-1].duration < 5
    # Nothing was typed into the dialog.
    assert _enters(world_env.log) == (1 if step_name == "plot" else 3)


def test_record_take_a_long_but_busy_turn_is_no_stall(world_env) -> None:
    world = world_env.make_world({"busy_s": 3 * video_take.STALL_S})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert timeline.steps[-1].name == "close"


def test_record_take_2d_plot_fails_the_plot_step(world_env) -> None:
    world = world_env.make_world({"plot": "2d"})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "plot"


@pytest.mark.parametrize(
    "asks",
    [[ENTRY_PUBLISH], [SELECTIVE], [UNNAMED], [ENTRY_CREATE, ENTRY_PUBLISH]],
    ids=["entry_publish", "selective", "unnamed", "entry_create+publish"],
)
def test_record_take_foreign_ask_fails_without_keypress(world_env, asks) -> None:
    world = world_env.make_world({"asks": asks})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "approve"
    # Only the three prompt submissions: nothing was approved.
    assert _enters(world_env.log) == 3
    assert "approve" not in [s.name for s in timeline.steps]


def test_record_take_retries_enter_once_when_first_was_swallowed(world_env) -> None:
    world = world_env.make_world({"draft_after_enters": 2})
    take, timeline = _run_take(world_env, world)
    record_take(take, timeline)
    assert _enters(world_env.log) == 5
    assert timeline.steps[-1].name == "close"
    draft = next(s for s in timeline.steps if s.name == "draft")
    assert draft.duration >= video_take.DRAFT_RETRY_AFTER_MS / 1000


def test_record_take_never_retries_more_than_once(world_env) -> None:
    world = world_env.make_world({"draft_after_enters": 0})
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "draft"
    assert _enters(world_env.log) == 5


@pytest.mark.parametrize(
    "plan",
    [
        {"draft_after_enters": 0, "tool_result": "entry"},
        {"draft_after_enters": 0, "tool_result": "block"},
        {"draft_after_enters": 0, "late_ask": True},
    ],
    ids=["tool_result-entry", "tool_result-block", "newer-ask"],
)
def test_record_take_no_retry_enter_when_unsafe(world_env, plan) -> None:
    world = world_env.make_world(plan)
    take, timeline = _run_take(world_env, world)
    with pytest.raises(ProbeFailed) as exc:
        record_take(take, timeline)
    assert exc.value.step == "draft"
    assert _enters(world_env.log) == 4


def test_record_theme_records_one_take(world_env) -> None:
    world = world_env.make_world({})
    world_env.stack.browser = FakeBrowser(world_env.log, {"world": world})
    take, timeline = record_theme(world_env.stack, "dark", clock=world.clock)
    (context,) = world_env.stack.browser.contexts
    assert take.context is context
    assert [(s.name, s.fast_forward) for s in timeline.steps] == BEATS
    assert timeline.theme == "dark" and timeline.version == "9.9-test"
    assert not take.screencasting


def test_record_theme_a_failed_take_raises_and_is_closed(world_env, capsys) -> None:
    # One take per call: a failed check ends it, and the caller decides on a
    # retry (on a fresh stack).
    world = world_env.make_world({"plot": "2d"})
    world_env.stack.browser = FakeBrowser(world_env.log, {"world": world})
    with pytest.raises(ProbeFailed) as exc:
        record_theme(world_env.stack, "light", clock=world.clock)
    assert exc.value.step == "plot"
    assert all(c.closed for c in world_env.stack.browser.contexts)
    assert world_env.stack.current is None
    assert "WARNING" not in capsys.readouterr().out


def test_record_theme_no_sleep_in_any_video_module() -> None:
    here = Path(video_take.__file__).parent
    files = sorted(here.glob("video_*.py"))
    assert files
    offenders = [f.name for f in files if re.search(r"\btime\.sleep\b", f.read_text())]
    assert offenders == []


def _fake_session_config(monkeypatch, tmp_path):
    config_dir = tmp_path / "claude-config"
    transcript = config_dir / "projects" / "demo" / "session.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type": "user"}\n')
    monkeypatch.setattr(capture, "_isolated_claude_config", lambda project_dir: config_dir)
    return config_dir


def test_web_terminal_keeps_the_session_record_when_asked(monkeypatch, tmp_path) -> None:
    # A failed take is diagnosed from its transcript, which lives in the
    # session's own config dir and would otherwise go with it.
    _stub_web_launch(monkeypatch)
    config_dir = _fake_session_config(monkeypatch, tmp_path)
    keep = tmp_path / "kept"
    monkeypatch.setenv(capture.KEEP_ENV, str(keep))
    with capture.web_terminal(tmp_path / "repo"):
        pass
    kept = list(keep.rglob("session.jsonl"))
    assert len(kept) == 1
    assert kept[0].read_text() == '{"type": "user"}\n'
    assert not config_dir.exists()


def test_web_terminal_keeps_nothing_by_default(monkeypatch, tmp_path) -> None:
    _stub_web_launch(monkeypatch)
    config_dir = _fake_session_config(monkeypatch, tmp_path)
    monkeypatch.delenv(capture.KEEP_ENV, raising=False)
    with capture.web_terminal(tmp_path / "repo"):
        pass
    assert not config_dir.exists()
    assert list(tmp_path.rglob("session.jsonl")) == []


def test_the_demo_session_starts_in_auto_mode(monkeypatch, tmp_path) -> None:
    # The agent hands work to background subagents, which cannot show a
    # permission prompt; in manual mode their tool calls wait on nobody.
    from osprey.agent_runner import claude_state

    def seed(_build_dir, env):
        config_dir = Path(env["CLAUDE_CONFIG_DIR"])
        (config_dir / "settings.json").write_text(json.dumps({"model": "seeded"}))

    monkeypatch.setattr(claude_state, "seed_claude_state", seed)
    monkeypatch.setattr(capture.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()

    config_dir = capture._isolated_claude_config(tmp_path / "repo")

    settings = json.loads((config_dir / "settings.json").read_text())
    assert settings["permissions"]["defaultMode"] == "auto"
    assert settings["skipAutoPermissionPrompt"] is True
    assert settings["model"] == "seeded"


def test_the_gateway_billing_notice_is_already_acknowledged(monkeypatch, tmp_path) -> None:
    # Auto mode behind a gateway opens a one-time notice that waits for Enter
    # and blocks the tool call it interrupted; a scripted take cannot answer it.
    from osprey.agent_runner import claude_state

    def seed(_build_dir, env):
        config_dir = Path(env["CLAUDE_CONFIG_DIR"])
        (config_dir / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True}))

    monkeypatch.setattr(claude_state, "seed_claude_state", seed)
    monkeypatch.setattr(capture.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()
    monkeypatch.setattr(capture, "_now_ms", lambda: 1_790_000_000_500)

    config_dir = capture._isolated_claude_config(tmp_path / "repo")

    state = json.loads((config_dir / ".claude.json").read_text())
    assert state["autoModeClassifierBillingNoticeAcknowledgedAt"] == 1_790_000_000_500
    assert state["hasCompletedOnboarding"] is True


def test_web_terminal_serves_its_front_end_from_local_bundles(monkeypatch, tmp_path) -> None:
    # A take must not depend on a public CDN answering: the terminal's own
    # scripts are served from the checkout's vendored bundles.
    popen, _ = _stub_web_launch(monkeypatch)
    with capture.web_terminal(tmp_path):
        pass
    (_args,), kwargs = popen.call_args
    assert kwargs["env"]["OSPREY_OFFLINE"] == "1"


def test_missing_local_bundles_skip_with_the_fetch_command(monkeypatch, tmp_path) -> None:
    popen, _ = _stub_web_launch(monkeypatch)
    monkeypatch.setattr(
        capture, "verify_vendor_bundles", lambda: ([], ["MISSING: static/vendor/xterm.min.js"])
    )
    with pytest.raises(capture.ScreenshotSkip, match="osprey vendor fetch"):
        with capture.web_terminal(tmp_path):
            pytest.fail("body must not run")
    popen.assert_not_called()

"""The watchdog dialog in a real browser, its flags raised on the real path.

Real here: chromium, the web terminal app and its chat route, the pinned Claude
Code it launches, the rendered trigger hook that posts each prompt to the app's
intake, two watchdogs and their decision client, the hold record, the Stop that
interrupts the turn. Replaced: the model (:mod:`tests.e2e.scripted_model`), the Jev
service (a local server behind ``api.jev`` answering with the flag mass a test
asks for) and the event dispatcher behind the EVENTS panel.

Each prompt is the page's own, typed in the chat's composer. The terminal runs a
stand-in shell, so a session reaches it by a reload of a tab that opted into it, and a prompt
typed in it is the post its trigger hook makes to the intake. A flag
opens after its prompt, so a scripted turn that must meet the flag waits at the
model until the record shows both.

Run:
    uv run pytest tests/interfaces/test_watchdog_e2e.py -m browser -q

The browser lane reads the junit report of this file and fails on any skip.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import yaml

from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV
from osprey.audit import writer as audit_writer
from osprey.interfaces import _serving
from osprey.interfaces._serving import run_app_server
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, WEB_PORT_ENV
from osprey.interfaces.web_auth import DISPATCHER_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal.operator_session import (
    build_operator_child_env,
    session_hold_root_of,
)
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.models.jev.config import TYPESAFE_KEY_ENV
from osprey.services.watchdog.verdict import SURFACE_SAFETY_WATCHDOG
from osprey_connectors.session_hold import is_held, open_flags
from tests.e2e import sdk_helpers
from tests.e2e.claude_code.conftest import _point_at_safety_limits_db, _serve_safety_channels
from tests.e2e.conftest import reset_registry_between_tests, scripted_model  # noqa: F401
from tests.e2e.scripted_model import ScriptedCall
from tests.e2e.sdk_helpers import (
    cli_mcp_ready_env,
    enable_writes_in_project,
    init_project,
    render_dir,
    trust_render,
)
from tests.e2e.test_watchdog_hold_scripted import (
    WATCHDOG,
    Conductor,
    FakeDispatcher,
    FakeJev,
    _point_jev_at,
    read,
)
from tests.interfaces._hub import LONG_LIVED_SHELL, hub_app
from tests.interfaces.web_terminal._surface import arm_terminal_tab

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page

with contextlib.suppress(ImportError):
    from playwright.sync_api import expect

pytestmark = [pytest.mark.browser, pytest.mark.slow, pytest.mark.usefixtures("real_agent_cli")]

#: A second watchdog over the same prompt, so one prompt raises two flags.
SECOND_WATCHDOG = "second-look"
WATCHDOGS = {WATCHDOG, SECOND_WATCHDOG}

CHIP = "#control-target-chip"
CHIP_STATE = f"{CHIP} .ctc-state"
DIALOG = ".watchdog-alert-dialog"
FLAG = f"{DIALOG} .watchdog-alert-flag"
STOP = f"{DIALOG} .watchdog-alert-stop"
CONTINUE = f"{DIALOG} .watchdog-alert-continue"
ANSWER_ROUTE = "**/api/watchdog/flags/answer"

OPERATOR = "#operator-container"
COMPOSER = f"{OPERATOR} .op-input-area textarea"
TURN_STOP_BUTTON = f"{OPERATOR} .op-stop-btn"
TERMINAL = "#terminal-container .xterm"

#: A page's own settling, once what it waits for exists.
TIMEOUT_MS = 15_000
#: A first prompt launches the agent and its MCP servers, then the watchdogs judge it.
TURN_TIMEOUT_MS = 120_000
FLAG_TIMEOUT_S = 90.0
HTTP_TIMEOUT_S = 180.0

EXECUTE = "mcp__python__execute"
#: What the process table calls a script the executor runs.
SANDBOX_COMMAND = "wrapped_script.py"

#: Outlasts a Stop on its own: a group of its own is what names it a started
#: command, and ignoring the interrupt is what keeps it there. It ends by itself
#: so that nothing outlives the test by much.
STUBBORN_SCRIPT = (
    "import os, signal, time\n"
    "os.setsid()\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "time.sleep(120)\n"
)


@dataclass
class Served:
    """The web terminal over one deployment, which can be stopped and started again."""

    repo: Path
    config_path: Path
    watched: Path
    events: dict[str, Any]
    conductor: Conductor
    jev: FakeJev
    audit: list[dict[str, Any]] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)
    app: Any = None
    base_url: str = ""
    _running: ExitStack | None = None

    def start(self, port: int | None = None) -> None:
        """Serve a new app over the same config, on *port* when one is named."""
        running = ExitStack()
        app = running.enter_context(
            hub_app(
                self.watched,
                enabled_panels=(),
                custom_panels=[self.events],
                shell_command=LONG_LIVED_SHELL,
                config_path=self.config_path,
                project_dir=render_dir(self.repo),
            )
        )
        bind = _serving.listening_socket
        with pytest.MonkeyPatch.context() as patch:
            if port is not None:
                patch.setattr(_serving, "listening_socket", lambda host, _port: bind(host, port))
            self.base_url = running.enter_context(run_app_server(app))
        self.app = app
        self._running = running

    def stop(self) -> None:
        running, self._running = self._running, None
        if running is None:
            return
        try:
            with self.client() as client:
                for key in self.keys:
                    client.delete(f"/api/chat/{key}")
        finally:
            running.close()

    def restart(self) -> None:
        """Stop the app and bring a new one up on the same port, as a redeploy would."""
        port = int(self.base_url.rsplit(":", 1)[1])
        self.stop()
        self.start(port)

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=HTTP_TIMEOUT_S,
            headers={OPERATOR_SECRET_HEADER: get_web_credentials(self.app).operator_secret},
        )

    @property
    def root(self) -> Path:
        root = session_hold_root_of(self.app)
        assert root is not None, "the render runs no watchdog"
        return root

    def flags(self, key: str) -> list[Any]:
        return open_flags(key, root=self.root)

    def until_both_flagged(self, key: str) -> Callable[[], None]:
        """What holds a scripted turn back until each watchdog's flag is on the record."""

        def wait() -> None:
            deadline = time.monotonic() + FLAG_TIMEOUT_S
            while time.monotonic() < deadline:
                if {flag.source for flag in self.flags(key)} == WATCHDOGS:
                    return
                time.sleep(0.05)
            raise AssertionError(f"both flags did not open within {FLAG_TIMEOUT_S} s")

        return wait

    def next_prompt(
        self,
        script: list[list[ScriptedCall]],
        *,
        flagging: bool,
        before_turn: dict[int, Callable[[], None]] | None = None,
    ) -> str:
        prompt = f"Carry out the scripted steps ({uuid.uuid4().hex})."
        self.jev.flagging = flagging
        self.conductor.next_prompt(prompt, script, before_turn)
        return prompt

    def answers(self, key: str) -> list[dict[str, Any]]:
        """The answers filed for *key*: every audit entry that is not a verdict."""
        return [
            entry
            for entry in self.audit
            if entry["surface"] == SURFACE_SAFETY_WATCHDOG
            and entry.get("session") == key
            and entry["decision"] not in {"flagged", "clear", "unavailable"}
        ]


@pytest.fixture(autouse=True)
def _claude_state_of_the_test(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Claude Code state directory of the test's own, which nothing of the developer's reaches."""
    config_dir = tmp_path_factory.mktemp("claude-config")
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setattr(sdk_helpers, "harness_config_dir", config_dir)


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant repo with two watchdogs on and writes armed."""
    root = tmp_path_factory.mktemp("watchdog-browser")
    repo = init_project(root, "watchdog-browser", provider="als-apg")
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["watchdogs"][WATCHDOG]["enabled"] = True
    profile["watchdogs"][SECOND_WATCHDOG] = dict(profile["watchdogs"][WATCHDOG])
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    shutil.copytree(repo / "watchdogs" / WATCHDOG, repo / "watchdogs" / SECOND_WATCHDOG)
    enable_writes_in_project(repo)
    _serve_safety_channels(repo, root / "served")
    _point_at_safety_limits_db(repo)
    return repo


@pytest.fixture
def served(
    deployment: Path,
    scripted_model: Callable[[Any], Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Served]:
    """The app served over the deployment, with the model and Jev replaced."""
    jev = FakeJev()
    dispatcher = FakeDispatcher()
    conductor = Conductor(scripted_model([]))
    conductor.install()

    config_path = _point_jev_at(deployment, jev.url)
    render = render_dir(deployment)
    trust_render(deployment)
    # The deployment's own agent-data tree, as a launch from its render resolves it.
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv(TYPESAFE_KEY_ENV, "fake-jev-key")
    monkeypatch.setenv(DISPATCHER_TOKEN_ENV, "fake-dispatcher-token")
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "fake-launch-token")
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "0")
    for name, value in cli_mcp_ready_env().items():
        monkeypatch.setenv(name, value)

    def chat_child_env(*args: Any, **kwargs: Any) -> dict[str, str]:
        # The chat's own environment, hold stamps included, on the scripted credential.
        return {**build_operator_child_env(*args, **kwargs), **sdk_helpers.sdk_env(render)}

    monkeypatch.setattr(chat_routes, "build_operator_child_env", chat_child_env)

    stack = Served(
        repo=deployment,
        config_path=config_path,
        watched=tmp_path / "watched",
        events={
            "id": "events",
            "label": "EVENTS",
            "url": dispatcher.url,
            "configDefined": True,
        },
        conductor=conductor,
        jev=jev,
    )

    record = audit_writer.record

    def keep_and_record(**fields: Any) -> Any:
        stack.audit.append(dict(fields))
        return record(**fields)

    monkeypatch.setattr(audit_writer, "record", keep_and_record)

    stack.start()
    # Read when a child starts, so the trigger hook posts to this app.
    monkeypatch.setenv(WEB_PORT_ENV, stack.base_url.rsplit(":", 1)[1])
    try:
        yield stack
    finally:
        try:
            stack.stop()
        finally:
            dispatcher.stop()
            jev.stop()


@pytest.fixture
def context(
    chromium_browser: Browser,
    served: Served,  # noqa: ARG001 - its port names the session cookie this profile is given
) -> Iterator[BrowserContext]:
    """One browser profile: the session pointer lives in its storage across tabs."""
    browser_context = chromium_browser.new_context()
    try:
        yield browser_context
    finally:
        browser_context.close()


def _open_page(context: BrowserContext, served: Served) -> tuple[Page, str]:
    """Open a tab on the app; the tab and the session key it settled on."""
    page = context.new_page()
    page.goto(served.base_url, wait_until="domcontentloaded")
    expect(page.locator(CHIP)).to_be_visible(timeout=TIMEOUT_MS)
    key = _session_key(page)
    served.keys.add(key)
    return page, key


def _session_key(page: Page) -> str:
    page.wait_for_function(
        "async () => (await import('/static/js/terminal.js')).getCurrentSessionId() !== null",
        timeout=TIMEOUT_MS,
    )
    return page.evaluate(
        "async () => (await import('/static/js/terminal.js')).getCurrentSessionId()"
    )


def _type_prompt(page: Page, prompt: str) -> None:
    composer = page.locator(COMPOSER)
    expect(composer).to_be_enabled(timeout=TURN_TIMEOUT_MS)
    composer.fill(prompt)
    composer.press("Enter")


def _flag_in_composer(
    page: Page,
    served: Served,
    key: str,
    script: list[list[ScriptedCall]] | None = None,
    *,
    before_turn: dict[int, Callable[[], None]] | None = None,
) -> str:
    """Type a prompt both watchdogs flag; returns once its dialog shows both flags."""
    prompt = served.next_prompt(
        script if script is not None else [[read()]],
        flagging=True,
        before_turn=before_turn or {0: served.until_both_flagged(key)},
    )
    _type_prompt(page, prompt)
    _assert_dialog_names_both_watchdogs(page, prompt)
    return prompt


def _assert_dialog_names_both_watchdogs(page: Page, prompt: str) -> None:
    expect(page.locator(FLAG)).to_have_count(2, timeout=TURN_TIMEOUT_MS)
    expect(page.locator(DIALOG)).to_have_count(1)
    flags = page.locator(FLAG)
    text = [flags.nth(index).inner_text() for index in range(2)]
    for name in WATCHDOGS:
        assert [item for item in text if name in item], f"{name} is not on the dialog: {text}"
    for item in text:
        assert prompt in item, f"the prompt is not on the flag: {item}"
    assert not re.search(r"\bsafe\b", page.inner_text("body"), re.IGNORECASE)


def _assert_chip_held(page: Page) -> None:
    expect(page.locator(CHIP_STATE)).to_have_text("held", timeout=TIMEOUT_MS)
    expect(page.locator(CHIP)).to_have_attribute("data-held", "true", timeout=TIMEOUT_MS)


def _assert_released(page: Page, served: Served, key: str) -> None:
    expect(page.locator(DIALOG)).to_have_count(0, timeout=TIMEOUT_MS)
    expect(page.locator(CHIP)).to_have_attribute("data-held", "false", timeout=TIMEOUT_MS)
    expect(page.locator(CHIP_STATE)).not_to_have_text("held", timeout=TIMEOUT_MS)
    assert served.flags(key) == []


def _answer(page: Page, button: str) -> dict[str, Any]:
    """Click a dialog button; the answer the server gave it."""
    with page.expect_response(ANSWER_ROUTE, timeout=HTTP_TIMEOUT_S * 1_000) as answered:
        page.locator(button).click()
    assert answered.value.ok, answered.value.status
    return answered.value.json()


def _wait_started_command(served: Served, key: str) -> None:
    """Block until the agent's script runs in a group of its own, its approval hook gone by."""
    session = served.app.state.operator_registry.get_chat_session(key)
    deadline = time.monotonic() + TURN_TIMEOUT_MS / 1_000
    while time.monotonic() < deadline:
        if SANDBOX_COMMAND in {group.label for group in session.started_commands()}:
            return
        time.sleep(0.1)
    raise AssertionError(f"the script never started: {served.conductor.results}")


def _wait_terminal_session(served: Served, key: str) -> None:
    """Block until the terminal runs a child under *key*: only then can its hook post."""
    deadline = time.monotonic() + TURN_TIMEOUT_MS / 1_000
    while served.app.state.pty_registry.get_session(key) is None:
        assert time.monotonic() < deadline, "the terminal never started a session under the key"
        time.sleep(0.1)


def _wait_turn_over(page: Page, served: Served) -> None:
    deadline = time.monotonic() + TURN_TIMEOUT_MS / 1_000
    while not served.conductor.returned:
        assert time.monotonic() < deadline, "the agent never returned its last scripted turn"
        page.wait_for_timeout(50)
    expect(page.locator(TURN_STOP_BUTTON)).to_be_hidden(timeout=TURN_TIMEOUT_MS)


def test_a_flagged_prompt_raises_one_dialog_for_two_watchdogs_and_continue_releases(
    context: BrowserContext, served: Served
) -> None:
    """Typed in the chat: both flags in one dialog, chip held, then not."""
    page, key = _open_page(context, served)
    expect(page.locator(CHIP_STATE)).not_to_have_text("held", timeout=TIMEOUT_MS)

    _flag_in_composer(page, served, key)

    _assert_chip_held(page)
    assert is_held(key, root=served.root)

    answer = _answer(page, CONTINUE)

    assert len(answer["released"]) == 2
    _assert_released(page, served, key)
    answers = served.answers(key)
    assert {entry["subject"] for entry in answers} == WATCHDOGS
    assert {entry["decision"] for entry in answers} == {"continue"}


def test_the_terminal_shows_the_held_session_and_a_flag_raised_under_it(
    context: BrowserContext, served: Served
) -> None:
    """The session moves to the terminal still held; its next prompt reaches the intake there."""
    page, key = _open_page(context, served)
    prompt = _flag_in_composer(page, served, key)
    _wait_turn_over(page, served)

    # The dialog takes every click, so the tab reaches the terminal the way one
    # that already opted into it does: on a reload.
    arm_terminal_tab(page)
    page.reload(wait_until="domcontentloaded")
    expect(page.locator(TERMINAL)).to_be_visible(timeout=TIMEOUT_MS)
    assert _session_key(page) == key
    _wait_terminal_session(served, key)

    _assert_dialog_names_both_watchdogs(page, prompt)
    _assert_chip_held(page)
    _answer(page, CONTINUE)
    _assert_released(page, served, key)

    # What the trigger hook posts for a prompt typed in this terminal.
    typed = f"A prompt typed in the terminal ({uuid.uuid4().hex})."
    served.jev.flagging = True
    with served.client() as client:
        posted = client.post(
            "/api/watchdog/prompt",
            json={
                "session_key": key,
                "record_session_id": key,
                "submission_id": str(uuid.uuid4()),
                "prompt": typed,
            },
        )
    assert posted.status_code == 202, posted.text

    _assert_dialog_names_both_watchdogs(page, typed)
    _assert_chip_held(page)
    _answer(page, CONTINUE)
    _assert_released(page, served, key)


def test_stop_interrupts_a_running_turn_and_releases(
    context: BrowserContext, served: Served
) -> None:
    """The model has not answered when Stop lands: the turn ends and its next step never comes."""
    page, key = _open_page(context, served)
    reached = threading.Event()
    never = threading.Event()

    def park_the_turn() -> None:
        reached.set()
        never.wait(timeout=HTTP_TIMEOUT_S)

    try:
        _flag_in_composer(
            page,
            served,
            key,
            [[read()], [read()]],
            before_turn={0: served.until_both_flagged(key), 1: park_the_turn},
        )
        assert reached.wait(timeout=TURN_TIMEOUT_MS / 1_000), "the turn never reached its model"
        expect(page.locator(TURN_STOP_BUTTON)).to_be_visible(timeout=TIMEOUT_MS)

        answer = _answer(page, STOP)

        assert len(answer["released"]) == 2
        assert answer["stop_unconfirmed"] == []
        _assert_released(page, served, key)
        expect(page.locator(TURN_STOP_BUTTON)).to_be_hidden(timeout=TIMEOUT_MS)
        expect(page.locator(COMPOSER)).to_be_enabled(timeout=TIMEOUT_MS)
        assert 1 not in served.conductor.delivered
        assert {entry["decision"] for entry in served.answers(key)} == {"stop"}
    finally:
        never.set()


def test_an_idle_stop_records_and_releases_and_the_session_goes_on(
    context: BrowserContext, served: Served
) -> None:
    """Nothing runs: the Stop still files its answer, and the next prompt runs in the session."""
    page, key = _open_page(context, served)
    _flag_in_composer(page, served, key)
    _wait_turn_over(page, served)

    answer = _answer(page, STOP)

    assert len(answer["released"]) == 2
    _assert_released(page, served, key)
    answers = served.answers(key)
    assert {entry["subject"] for entry in answers} == WATCHDOGS
    assert {entry["decision"] for entry in answers} == {"stop"}

    prompt = served.next_prompt([[read()]], flagging=False)
    _type_prompt(page, prompt)
    _wait_turn_over(page, served)
    served.conductor.assert_consumed()
    expect(page.locator(DIALOG)).to_have_count(0)


def test_a_stop_a_running_command_outlives_keeps_the_dialog_until_continue(
    context: BrowserContext, served: Served
) -> None:
    """The agent's script ignores the interrupt in a group of its own: the flags stay and name it."""
    page, key = _open_page(context, served)
    _flag_in_composer(
        page,
        served,
        key,
        [
            [read()],
            [
                ScriptedCall(
                    EXECUTE,
                    {
                        "code": STUBBORN_SCRIPT,
                        "description": "Keep going",
                        "execution_mode": "readonly",
                    },
                )
            ],
        ],
    )
    _wait_started_command(served, key)

    answer = _answer(page, STOP)

    assert answer["released"] == []
    assert len(answer["stop_unconfirmed"]) == 2
    flags = page.locator(FLAG)
    expect(flags).to_have_count(2, timeout=TIMEOUT_MS)
    expect(flags.first).to_contain_text("turn did not stop", timeout=TIMEOUT_MS)
    expect(flags.first).to_contain_text(SANDBOX_COMMAND)
    expect(page.locator(CONTINUE)).to_be_enabled()
    _assert_chip_held(page)
    assert is_held(key, root=served.root)
    assert {entry["decision"] for entry in served.answers(key)} == {"stop_unconfirmed"}

    answer = _answer(page, CONTINUE)

    assert len(answer["released"]) == 2
    _assert_released(page, served, key)
    continued = [entry for entry in served.answers(key) if entry["decision"] == "continue"]
    assert {entry["reason"] for entry in continued} == {"after_stop_unconfirmed"}


def test_the_hold_survives_reload_a_closed_tab_and_a_restart(
    context: BrowserContext, served: Served
) -> None:
    """Nothing but an answer drops it: each way of leaving comes back to the same dialog."""
    page, key = _open_page(context, served)
    prompt = _flag_in_composer(page, served, key)
    _wait_turn_over(page, served)
    flag_ids = {flag.flag_id for flag in served.flags(key)}

    page.reload(wait_until="domcontentloaded")
    _assert_dialog_names_both_watchdogs(page, prompt)
    _assert_chip_held(page)

    page.close()
    assert is_held(key, root=served.root)
    page, again = _open_page(context, served)
    assert again == key
    _assert_dialog_names_both_watchdogs(page, prompt)
    _assert_chip_held(page)

    page.close()
    served.restart()
    assert {flag.flag_id for flag in served.flags(key)} == flag_ids
    page, again = _open_page(context, served)
    assert again == key
    _assert_dialog_names_both_watchdogs(page, prompt)
    _assert_chip_held(page)

    answer = _answer(page, CONTINUE)

    assert set(answer["released"]) == flag_ids
    _assert_released(page, served, key)

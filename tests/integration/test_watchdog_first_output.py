"""A watchdog costs a prompt's first output at most 100 ms, even with Jev hung.

The app is the real web terminal on its real event loop, the chat turn goes through
``POST /api/chat``, and the runner is the real one with the real decision engine,
pointed at a socket that listens and never accepts. Only the agent is a stub: it
sleeps, then speaks, so what is timed is the app's part of the wait.

Each turn with the watchdog is paired with one without it on the same server and
the same chat. The median of the paired differences is the claim. A wall-clock
budget, so it runs alone (``wallclock``).
"""

from __future__ import annotations

import asyncio
import json
import socket
import statistics
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import yaml
from claude_agent_sdk import TextBlock

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import get_web_credentials
from osprey.models.jev.config import TYPESAFE_KEY_ENV
from osprey.services.watchdog.config import WatchdogDefinition, parse_watchdogs_block
from osprey.services.watchdog.runner import WatchdogRunner
from tests.interfaces.web_terminal._chat_server import _FakeSDKClient, live_chat_server
from tests.interfaces.web_terminal._fakes import assistant_message, result_message

BUDGET_SECONDS = 0.100
PAIRS = 15
AGENT_SLEEP_SECONDS = 0.05

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"
PRESET_WATCHDOG = "guard-circumvention"
#: The name the agent runner constructs its client under.
CLIENT = "osprey.agent_runner.session.ClaudeSDKClient"


class _SleepingAgent(_FakeSDKClient):
    """Answers every prompt with one line, after a fixed think."""

    async def receive_response(self):
        await self._pending.get()
        await asyncio.sleep(AGENT_SLEEP_SECONDS)
        yield assistant_message([TextBlock("first output")])
        yield result_message()


def _decision_watchdog() -> WatchdogDefinition:
    document = yaml.safe_load((PRESET_DIR / "control-assistant.yml").read_text(encoding="utf-8"))
    definitions = parse_watchdogs_block(document["watchdogs"], base_dir=PRESET_DIR)
    by_name = {definition.name: definition for definition in definitions}
    return replace(by_name[PRESET_WATCHDOG], enabled=True)


async def _seconds_to_first_output(client: httpx.AsyncClient, chat_id: str, prompt: str) -> float:
    """From sending *prompt* to the agent's first text frame; the turn is then drained."""
    started = time.perf_counter()
    elapsed: float | None = None
    async with client.stream(
        "POST", "/api/chat", json={"chat_id": chat_id, "prompt": prompt}
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            frame: dict[str, Any] = json.loads(line[len("data: ") :])
            if elapsed is None and frame.get("type") == "text":
                elapsed = time.perf_counter() - started
    assert elapsed is not None, "the turn produced no output"
    return elapsed


async def _with_watchdog(client: httpx.AsyncClient, chat_id: str, prompt: str) -> float:
    """A turn whose prompt reaches the intake as the trigger hook's post does."""

    async def intake() -> None:
        posted = await client.post(
            "/api/watchdog/prompt",
            json={
                "session_key": chat_id,
                "record_session_id": chat_id,
                "submission_id": str(uuid.uuid4()),
                "prompt": prompt,
            },
        )
        assert posted.status_code == 202, posted.text
        assert posted.json() == {"accepted": True, "duplicate": False}

    elapsed, _posted = await asyncio.gather(
        _seconds_to_first_output(client, chat_id, prompt), intake()
    )
    return elapsed


def _accepted_a_connection(hung: socket.socket, timeout_s: float) -> bool:
    """Whether the decision client reached the hung socket's backlog."""
    hung.settimeout(timeout_s)
    try:
        connection, _address = hung.accept()
    except TimeoutError:
        return False
    connection.close()
    return True


@pytest.mark.wallclock
async def test_first_output_is_within_100_ms_of_no_watchdog_with_jev_hung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TYPESAFE_KEY_ENV, "fake-jev-key")
    definition = _decision_watchdog()
    # Listening and never accepting: a connection lands in the backlog and hangs.
    with (
        socket.socket() as hung,
        live_chat_server(tmp_path) as (base_url, app),
        patch(CLIENT, _SleepingAgent),
    ):
        hung.bind(("127.0.0.1", 0))
        hung.listen(128)
        url = f"http://127.0.0.1:{hung.getsockname()[1]}"
        config = {"api": {"jev": {"route": "typesafe", "routes": {"typesafe": {"base_url": url}}}}}
        root = tmp_path / "var" / "state" / "session_holds"
        verdicts: list[Any] = []
        runner = WatchdogRunner(
            [definition],
            recorder=verdicts.append,
            hold_root=root,
            notify=lambda _frame: None,
            config=config,
        )
        chat_id = str(uuid.uuid4())
        headers = {OPERATOR_SECRET_HEADER: get_web_credentials(app).operator_secret}
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60.0) as client:
            # The child and its pool entry exist before anything is timed.
            await _seconds_to_first_output(client, chat_id, "warm up")

            deltas = []
            for index in range(PAIRS):
                app.state.watchdog_runner = None
                app.state.session_hold_root = None
                without = await _seconds_to_first_output(client, chat_id, f"plain {index}")

                app.state.session_hold_root = root
                app.state.watchdog_runner = runner
                watched = await _with_watchdog(client, chat_id, f"watched {index}")
                deltas.append(watched - without)

        assert statistics.median(deltas) <= BUDGET_SECONDS, sorted(deltas)
        # The budget means nothing unless the judging really was in flight.
        assert await asyncio.to_thread(_accepted_a_connection, hung, 10.0), verdicts

"""App factories for the multi-user pages the screenshot registry captures.

Each factory takes no argument and returns an ASGI app a
``standalone_interface`` recipe boots: the page a deployment serves, built by
the code that serves it, on state that never changes between runs.

* :func:`landing_app` and :func:`sign_in_app` — the landing page and the
  password sign-in page of the ``demo`` preset. Both come from one throwaway
  ``osprey init`` + ``osprey build`` (:func:`demo_web_tier`): the landing page
  is the one the build renders, and the sign-in service runs on the environment
  the build gives it. No container starts.
* :func:`history_app` — the history page on a canned agent record, behind a
  stand-in for the web tier that signs the request in as ``alice``.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docs.screenshots.recipes import HISTORY_SESSION

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import FastAPI

# The demo pins the light theme; a theme family lets both pages follow the
# browser's colour scheme, so one build serves the light and the dark capture.
_THEME_FAMILY = "main"
_PROJECT_NAME = "docshots-pages"


@dataclass(frozen=True)
class WebTier:
    """What the demo build renders for its web tier."""

    landing_html: str
    auth_env: Mapping[str, str]
    """The sign-in service's environment, as its compose entry carries it."""


@cache
def demo_web_tier() -> WebTier:
    """Render the demo preset's web tier once per run, in a throwaway repo.

    Raises :class:`~docs.screenshots.capture.ScreenshotSkip` when the
    ``osprey`` CLI is missing or a step fails.
    """
    import yaml
    from docs.screenshots.capture import ScreenshotSkip, _run_stack_step

    from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
    from osprey.deployment.web_terminals.render import render_web_terminals

    build_root = Path(tempfile.mkdtemp(prefix="osprey-docshot-pages-"))
    project_dir = build_root / _PROJECT_NAME
    try:
        _run_stack_step(
            [
                "osprey",
                "init",
                str(project_dir),
                "--preset",
                "demo",
                "--no-git",
                "--set",
                f"config.web.theme={_THEME_FAMILY}",
            ],
            cwd=None,
            what="create the demo repo",
        )
        _run_stack_step(
            ["osprey", "build", "--skip-deps", "--dev"], cwd=project_dir, what="build the demo"
        )
        config = yaml.safe_load((project_dir / "build" / "config.yml").read_text(encoding="utf-8"))
        rendered = render_web_terminals(config, **resolve_render_inputs(config, project_dir))
    finally:
        shutil.rmtree(build_root, ignore_errors=True)

    compose = yaml.safe_load(rendered["docker-compose.web.yml"]) or {}
    auth = (compose.get("services") or {}).get("auth") or {}
    auth_env = dict(pair.split("=", 1) for pair in auth.get("environment") or [] if "=" in pair)
    if not auth_env:
        raise ScreenshotSkip("the demo build rendered no sign-in service")
    return WebTier(landing_html=rendered["nginx/landing.html"], auth_env=auth_env)


def landing_app() -> FastAPI:
    """Serve the landing page the demo build renders, at ``/``."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    html = demo_web_tier().landing_html
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def landing() -> str:
        return html

    return app


def sign_in_app() -> FastAPI:
    """The sign-in service on the demo build's environment.

    The build leaves the signing secret to the deployment's ``.env.auth``; the
    page needs one to be served, and no capture signs in with it.
    """
    from osprey.interfaces.web_auth import mint_secret
    from osprey.services.auth_sidecar.app import ENV_SESSION_SECRET, create_app

    return create_app({**demo_web_tier().auth_env, ENV_SESSION_SECRET: mint_secret()})


# ---------------------------------------------------------------------------
# The history page, on a canned record
# ---------------------------------------------------------------------------

_PERSON = "alice"
_PROMPT = "BPM 5 reads high in x. Is it the orbit or the monitor? Fix it if it is the orbit."
_ANSWER = (
    "BPM 5 reads 0.41 mm in x while BPM 4 and BPM 6 read 0.02 mm and 0.03 mm, so the offset "
    "is local to BPM 5. The 14 A corrector step that would cancel it is outside the "
    "channel's limits (-12 A to 12 A) and was refused. Nothing was written."
)


def _node(kind: str, second: int, body: Any, **extra: Any) -> dict[str, Any]:
    ts = f"2026-01-15T10:15:{second:02d}+00:00"
    return {
        "key": f"{kind}-{second:02d}-{extra.pop('suffix', '')}",
        "kind": kind,
        "ts": ts,
        "person": _PERSON,
        "identity": _PERSON,
        "session_id": HISTORY_SESSION,
        "body": body,
        "incomplete": False,
        "decision": None,
        "result": None,
        "subagent": None,
        **extra,
    }


def _call(
    second: int,
    name: str,
    tool_input: dict[str, Any],
    result: str,
    *,
    decision: str = "accept",
    source: str = "config",
) -> dict[str, Any]:
    return _node(
        "tool_call",
        second,
        {"id": f"toolu-{second:02d}", "name": name, "input": tool_input},
        decision=_node("decision", second, {"decision": decision, "source": source}),
        result=_node(
            "tool_result",
            second,
            {"tool_use_id": f"toolu-{second:02d}", "content": [{"type": "text", "text": result}]},
        ),
    )


def _read(second: int, channel: str, value: str) -> dict[str, Any]:
    return _call(second, "mcp__controls__channel_read", {"channel": channel}, value)


def _timeline() -> list[dict[str, Any]]:
    return [
        _node("prompt", 0, {"text": _PROMPT}),
        _node(
            "thinking",
            2,
            {
                "text": "Read BPM 5 first, then its neighbours. A bump shows on several "
                "monitors; one monitor reading alone points at the monitor."
            },
        ),
        _read(4, "SR:DIAG:BPM:05:POSITION:X", "0.41 mm"),
        _node(
            "tool_call",
            5,
            {
                "id": "toolu-05",
                "name": "Agent",
                "input": {"description": "Read the neighbouring BPMs"},
            },
            decision=_node("decision", 5, {"decision": "accept", "source": "config"}),
            result=_node("subagent", 9, {"agent_id": "a-1", "status": "completed"}),
            subagent={
                "agent_id": "a-1",
                "timeline": [
                    _node("thinking", 6, {"text": "BPM 4 and BPM 6 bracket BPM 5."}, suffix="a"),
                    _read(7, "SR:DIAG:BPM:04:POSITION:X", "0.02 mm"),
                    _read(8, "SR:DIAG:BPM:06:POSITION:X", "0.03 mm"),
                ],
            },
        ),
        _call(
            10,
            "mcp__controls__channel_write",
            {"channel": "SR:MAG:HCM:05:CURRENT:SP", "value": 14.0},
            "refused: 14.0 A is outside the limits [-12.0, 12.0] A",
            decision="reject",
        ),
        _node("answer", 12, {"text": _ANSWER}),
    ]


class _CannedRecord:
    """The record reads the history page makes, answered from :func:`_timeline`."""

    def list_sessions(self, **_: Any) -> Any:
        from osprey.agent_record.client import ClientAnswer

        nodes = _timeline()
        row = {
            "session_id": HISTORY_SESSION,
            "persons": [_PERSON],
            "identities": [_PERSON],
            "first_ts": nodes[0]["ts"],
            "last_ts": nodes[-1]["ts"],
            "entries": len(nodes),
            "prompt": _PROMPT,
            "run_ids": [],
            "terminal_session": "",
        }
        return ClientAnswer(data=[row], complete=True)

    def session_users(self, **_: Any) -> Any:
        from osprey.agent_record.client import ClientAnswer

        user = {"person": _PERSON, "sessions": 1, "last_ts": _timeline()[-1]["ts"]}
        return ClientAnswer(data=[user], complete=True)

    def timeline(self, session_id: str, **_: Any) -> Any:
        from osprey.agent_record.client import ClientAnswer

        nodes = _timeline() if session_id == HISTORY_SESSION else []
        return ClientAnswer(data=nodes, complete=True)


def history_app() -> Any:
    """The history page under ``/history``, signed in as ``alice`` (own sessions).

    The wrapper does what the web tier does in a deployment: strips the
    ``/history`` prefix and adds the sign-in service's answer as headers.
    """
    from osprey.history import viewer
    from osprey.history.app import create_app

    app = create_app(env={}, client_for=lambda _viewer: _CannedRecord())
    prefix = viewer.ROOT_PATH
    signed_in = {
        viewer.PERSON_HEADER: _PERSON,
        viewer.SCOPE_HEADER: "own",
        viewer.VIEW_HEADER: f"own:{_PERSON}",
        viewer.ENTRY_HEADER: _PERSON,
        viewer.CHOICES_HEADER: f"{_PERSON}={_PERSON}",
    }
    headers = [(name.lower().encode(), value.encode()) for name, value in signed_in.items()]

    async def web_tier(scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == prefix or path.startswith(f"{prefix}/")):
            kept = [
                h for h in scope["headers"] if not h[0].lower().startswith(b"x-osprey-history-")
            ]
            scope = dict(scope, path=path[len(prefix) :] or "/", headers=kept + headers)
        await app(scope, receive, send)

    return web_tier

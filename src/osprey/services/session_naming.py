"""Automatic session names: a short title made from the operator's first prompt.

:func:`name_session` asks the project's provider for a name in one plain chat
completion on its fast model (the model Claude Code's ``haiku`` alias runs), and
trims the reply to at most :data:`MAX_TITLE_WORDS` words and
:data:`MAX_TITLE_CHARS` characters. Any failure — a timeout, an unresolved
provider, a refused or empty reply — returns ``None`` and is logged; the caller
keeps the first-prompt title. No agent runs, so nothing reaches the session's
record or transcripts.

:class:`SessionNamer` is the hub's scheduler. ``GET /api/sessions`` hands it the
rows it listed; it marks the eligible ones pending (in memory, in
``app.state.session_naming_pending``) and names them in the background, one at a
time, so the listing never waits on the model. Each result is recorded in the
session meta as ``done`` or ``failed`` and announced with one ``session_named``
frame. A failure is not retried again during the same hub start; a failure
recorded by an earlier start is retried once.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from osprey.stores.session_ledger import aware_time

logger = logging.getLogger(__name__)

#: The hub page frame sent once per naming result.
SESSION_NAMED_FRAME = "session_named"
#: Seconds one naming call may take before it counts as failed.
NAMING_TIMEOUT_S = 20.0
#: The most tokens a naming reply may take: a title, with room for a reasoning
#: model's thinking before it.
NAMING_MAX_TOKENS = 256
#: The longest automatic title, in words and in characters.
MAX_TITLE_WORDS = 6
MAX_TITLE_CHARS = 60
#: How much of the first prompt the model is shown.
PROMPT_CHARS = 2000
#: Seconds the shutdown waits for a cancelled naming to unwind.
_SHUTDOWN_GRACE_S = 5.0
#: The Claude Code alias whose model names sessions, on every provider.
NAMING_ALIAS = "haiku"

_SYSTEM_PROMPT = (
    "You name conversations. Reply with a short title for the conversation that "
    f"starts with the user's message: at most {MAX_TITLE_WORDS} words, no quotes, "
    "no trailing punctuation, nothing else."
)

_EDGE_JUNK = "\"'`*_#>«»“”‘’ \t"
_TRAILING_PUNCT = ".,;:!?…-–—"


def trim_title(reply: str | None) -> str | None:
    """A model reply as a title: its first line, unquoted, ≤ 6 words and ≤ 60 chars.

    ``None`` when nothing usable is left.
    """
    if not isinstance(reply, str):
        return None
    line = next((ln for ln in reply.splitlines() if ln.strip()), "")
    line = re.sub(r"^\s*(title|name)\s*:\s*", "", line, flags=re.IGNORECASE)
    words = line.strip(_EDGE_JUNK).split()[:MAX_TITLE_WORDS]
    text = " ".join(words)
    if len(text) > MAX_TITLE_CHARS:
        cut = text[:MAX_TITLE_CHARS]
        space = cut.rfind(" ")
        text = cut[:space] if space > 0 else cut
    text = text.strip(_EDGE_JUNK).rstrip(_TRAILING_PUNCT).strip(_EDGE_JUNK)
    return text or None


def _naming_route(project_dir: Path) -> tuple[str, str, dict[str, Any]] | None:
    """The project's provider, its ``haiku`` alias model and its ``api.providers`` entry.

    The alias model is always an id the provider serves (or its main model), so
    a chat completion can ask for it as is. ``None`` when the project resolves
    no provider.
    """
    from osprey.agent_runner import primitives
    from osprey.models.config import get_provider_config

    spec = primitives._resolve_project_spec(project_dir)
    if spec is None:
        return None
    model = spec.alias_models.get(NAMING_ALIAS) or spec.default_model_id
    provider_config = get_provider_config(spec.provider, str(project_dir / "config.yml"))
    return spec.provider, model, provider_config


async def name_session(
    prompt: str, *, project_dir: Path, timeout: float = NAMING_TIMEOUT_S
) -> str | None:
    """A short name for the session that began with *prompt*, or ``None``.

    Args:
        prompt: The operator's first prompt.
        project_dir: The rendered project the provider and model resolve from.
        timeout: Seconds the call may take.

    Returns:
        The trimmed title; ``None`` on any failure, which is logged.
    """
    text = " ".join((prompt or "").split())[:PROMPT_CHARS]
    if not text:
        return None
    try:
        from osprey.models.completion import aget_chat_completion
        from osprey.models.messages import ChatCompletionRequest, ChatMessage

        route = await asyncio.to_thread(_naming_route, Path(project_dir))
        if route is None:
            logger.warning("Session naming found no provider in %s", project_dir)
            return None
        provider, model, provider_config = route
        reply = await asyncio.wait_for(
            aget_chat_completion(
                chat_request=ChatCompletionRequest(
                    messages=[
                        ChatMessage(role="system", content=_SYSTEM_PROMPT),
                        ChatMessage(role="user", content=text),
                    ]
                ),
                provider=provider,
                model_id=model,
                provider_config=provider_config,
                max_tokens=NAMING_MAX_TOKENS,
            ),
            timeout,
        )
        title = trim_title(reply if isinstance(reply, str) else None)
        if title is None:
            logger.warning("Session naming returned no usable title")
        return title
    except TimeoutError:
        logger.warning("Session naming timed out after %g s", timeout)
        return None
    except Exception as exc:
        logger.warning("Session naming failed: %s: %s", type(exc).__name__, exc)
        return None


def _real_prompt(prompt: Any) -> str:
    """*prompt* when it is the operator's own words, else ``""``."""
    from osprey.agent_record.reader import is_bookkeeping_prompt

    if not isinstance(prompt, str) or not prompt.strip():
        return ""
    return "" if is_bookkeeping_prompt({"text": prompt}) else prompt


#: What names a session: ``(prompt, project_dir=...) -> title | None``.
Namer = Callable[..., Awaitable[str | None]]


class SessionNamer:
    """The hub's naming scheduler: one naming in flight, results as frames.

    Args:
        app: The hub app; its state holds the session meta, the pending set and
            the broadcaster.
        project_dir: The rendered project the naming calls resolve from.
        namer: What names one session; :func:`name_session` by default.
    """

    def __init__(self, app: Any, *, project_dir: Path, namer: Namer | None = None) -> None:
        self._app = app
        self._project_dir = Path(project_dir)
        self._namer = namer or name_session
        self._queue: deque[tuple[str, str, str]] = deque()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._worker: asyncio.Task[Any] | None = None
        self._failed: set[str] = set()
        self._closing = False
        if not isinstance(getattr(app.state, "session_naming_pending", None), set):
            app.state.session_naming_pending = set()

    @property
    def pending(self) -> set[str]:
        pending: set[str] = self._app.state.session_naming_pending
        return pending

    def eligible(self, row: Mapping[str, Any], since: datetime | None) -> str:
        """The prompt to name *row* from, or ``""`` when it is not to be named.

        Named: not deleted, a non-bookkeeping first prompt, not ``truncated``, no auto title,
        ``first_ts ≥ naming_since``, not pending, not failed this hub start.
        """
        from osprey.interfaces.web_terminal import session_meta

        tid = row.get("transcript_id")
        if not isinstance(tid, str) or not tid or tid in self.pending or tid in self._failed:
            return ""
        if row.get("truncated"):
            return ""
        prompt = _real_prompt(row.get("prompt"))
        if not prompt:
            return ""
        first = aware_time(row.get("first_ts"))
        if since is None or first is None or first < since:
            return ""
        meta = session_meta.get(self._app, tid)
        if meta.get("deleted") or meta.get("auto_title") or meta.get("naming") == "done":
            return ""
        return prompt

    def schedule(self, rows: Iterable[Mapping[str, Any]]) -> list[str]:
        """Mark the eligible *rows* pending and queue them; never waits on a model.

        Returns:
            The transcript ids newly scheduled.
        """
        if self._closing:
            return []
        from osprey.interfaces.web_terminal import session_meta

        since = aware_time(session_meta.naming_since(self._app))
        scheduled: list[str] = []
        for row in rows:
            prompt = self.eligible(row, since)
            if not prompt:
                continue
            tid = row["transcript_id"]
            self.pending.add(tid)
            title = row.get("title")
            self._queue.append((tid, prompt, title if isinstance(title, str) else ""))
            scheduled.append(tid)
        if scheduled and (self._worker is None or self._worker.done()):
            self._worker = self._spawn(self._drain_queue())
        return scheduled

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._forget)
        return task

    def _forget(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Session naming worker failed: %s", type(task.exception()).__name__)

    async def _drain_queue(self) -> None:
        while self._queue and not self._closing:
            tid, prompt, shown = self._queue.popleft()
            try:
                await self._name_one(tid, prompt, shown)
            except asyncio.CancelledError:
                self.pending.discard(tid)
                raise
            except Exception:
                logger.exception("Session naming of %s failed", tid)
                self.pending.discard(tid)
                self._failed.add(tid)

    async def _name_one(self, tid: str, prompt: str, shown: str) -> None:
        from osprey.interfaces.web_terminal import session_meta

        try:
            title = await self._namer(prompt, project_dir=self._project_dir)
        except Exception as exc:
            logger.warning("Session naming failed: %s: %s", type(exc).__name__, exc)
            title = None
        title = trim_title(title) if title is not None else None
        state = "done" if title else "failed"
        if state == "failed":
            self._failed.add(tid)
        entry = await asyncio.to_thread(session_meta.set_auto_title, self._app, tid, title, state)
        self.pending.discard(tid)
        broadcaster = getattr(self._app.state, "broadcaster", None)
        if broadcaster is None:
            return
        try:
            broadcaster.broadcast(
                {
                    "type": SESSION_NAMED_FRAME,
                    "transcript_id": tid,
                    "title": entry.get("title") or shown,
                    "naming": state,
                }
            )
        except Exception:  # a frame is a hint; the result is already durable
            logger.warning("Could not send the session_named frame", exc_info=True)

    async def wait_idle(self) -> None:
        """Return once everything scheduled so far has a result (or was cancelled)."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))

    async def shutdown(self) -> None:
        """Cancel what is queued and in flight; nothing pending outlives the hub."""
        self._closing = True
        self._queue.clear()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            _, stuck = await asyncio.wait(set(self._tasks), timeout=_SHUTDOWN_GRACE_S)
            if stuck:
                logger.warning("Session naming shutdown left %d task(s) unfinished", len(stuck))
        self.pending.clear()

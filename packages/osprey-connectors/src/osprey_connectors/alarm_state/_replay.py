"""A recorded alarm feed: topic messages replayed from a file through the real fold.

The file is JSONL, one topic message per line: ``{"key", "value", "time"}``, with
``time`` an ISO-8601 string. A message with no ``time`` keeps what its value
records. Times are rebased at load so the newest equals the load moment, so a
replay never ages; a trailing ``{"feed": "stopped"}`` line is what makes one stale.
A line that cannot be read fails the load: half a feed could read as a clear state.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey_connectors.alarm_state._contract import AlarmConfigRecord, AlarmState, AlarmStateRecord
from osprey_connectors.alarm_state._fold import (
    FEED_EMPTY,
    FEED_LOADING,
    FeedSnapshot,
    config_record,
    fold_message,
    state_record,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable

logger = get_logger("alarm_replay")

_STOPPED_MARKER = {"feed": "stopped"}


def _now() -> datetime:
    return datetime.now(UTC)


class ReplayAlarmState(AlarmState):
    """Answer from one replay file, loaded once by :meth:`start`.

    ``settings.replay_file`` is an absolute path or one relative to ``project_dir``.
    A ``demo:<name>`` still here was not resolved: only the framework can do that.
    A failed load answers ``Unavailable`` with its reason, so no record exists for
    ``stale`` to mark; the stopped marker is the one way a record reads stale. A
    read during the load waits for it up to its deadline.
    """

    def __init__(self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: Any) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        self._task: asyncio.Task[None] | None = None
        self._snapshot: FeedSnapshot | None = None
        self._stopped = False
        self._failed: Unavailable | None = None

    async def start(self) -> None:
        path = self._replay_path()
        if isinstance(path, Unavailable):
            return
        if self._task is None:
            self._task = asyncio.create_task(self._load(path), name="alarm-replay-load")
        # wait, not await: a cancelled caller leaves the load running for the next
        # one, and aclose cancelling the load does not raise in this caller.
        await asyncio.wait({self._task})

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._snapshot = None
        self._stopped = False
        self._failed = None

    async def state_for(
        self, channel: str, deadline: float
    ) -> AlarmStateRecord | Unavailable | None:
        await self._loading(deadline)
        snapshot = self._answering()
        if isinstance(snapshot, Unavailable):
            return snapshot
        return state_record(snapshot, channel, stale=self._stopped)

    async def config_for(self, channel: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        await self._loading(deadline)
        snapshot = self._answering()
        return snapshot if isinstance(snapshot, Unavailable) else config_record(snapshot, channel)

    async def probe(self, deadline: float) -> str | Unavailable:  # noqa: ARG002
        snapshot = self._loaded()
        if isinstance(snapshot, Unavailable):
            return snapshot
        evidence = f"read {snapshot.messages} messages from the replay file"
        return f"{evidence}; the feed stopped" if self._stopped else evidence

    async def _loading(self, deadline: float) -> None:
        """Wait up to *deadline* seconds for a load that has begun and not ended."""
        task = self._task
        if task is None or task.done() or deadline <= 0:
            return
        # wait, not wait_for: a deadline that runs out leaves the load running.
        await asyncio.wait({task}, timeout=deadline)

    def _answering(self) -> FeedSnapshot | Unavailable:
        snapshot = self._loaded()
        if isinstance(snapshot, FeedSnapshot) and snapshot.messages == 0:
            return Unavailable(FEED_EMPTY)
        return snapshot

    def _loaded(self) -> FeedSnapshot | Unavailable:
        path = self._replay_path()
        if isinstance(path, Unavailable):
            return path
        if self._failed is not None:
            return self._failed
        return self._snapshot if self._snapshot is not None else Unavailable(FEED_LOADING)

    def _replay_path(self) -> Path | Unavailable:
        value = self.spec.settings.get("replay_file")
        if not value:
            return Unavailable("alarm_state.replay_file is not set")
        if str(value).startswith("demo:"):
            return Unavailable(f"alarm replay {value} was not resolved to a file")
        path = Path(value)
        if path.is_absolute():
            return path
        if self.project_dir is None:
            return Unavailable(f"alarm replay file {value} is relative, with no project directory")
        return self.project_dir / path

    async def _load(self, path: Path) -> None:
        try:
            self._snapshot, self._stopped = await asyncio.to_thread(_read_replay, path)
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            logger.warning("alarm replay not loaded: %s", reason)
            self._failed = Unavailable(f"alarm replay file cannot be read: {reason}")


@dataclass(frozen=True)
class _Message:
    key: Any
    value: Any
    time: datetime | None


def _read_replay(path: Path) -> tuple[FeedSnapshot, bool]:
    """Fold the file at *path*; the second answer is whether it ends with the stopped marker."""
    messages: list[_Message] = []
    stopped = False
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                stopped = parsed == _STOPPED_MARKER
                if not stopped:
                    messages.append(_message(parsed))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path} line {number}: {exc}") from exc

    snapshot = FeedSnapshot()
    times = [message.time for message in messages if message.time is not None]
    shift = _now() - max(times) if times else None
    for message in messages:
        rebased = None
        if message.time is not None and shift is not None:
            rebased = _iso(message.time + shift)
        fold_message(snapshot, message.key, message.value, time=rebased)
    return snapshot, stopped


def _message(parsed: Any) -> _Message:
    if not isinstance(parsed, dict):
        raise ValueError("a message is a JSON object")
    value = parsed.get("value")
    if value is not None and not isinstance(value, dict | str):
        raise ValueError("a message's value is a JSON object, a string or null")
    recorded = parsed.get("time")
    if recorded is None:
        return _Message(parsed.get("key"), value, None)
    if not isinstance(recorded, str):
        raise ValueError("a message's time is an ISO-8601 string")
    moment = datetime.fromisoformat(recorded)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return _Message(parsed.get("key"), value, moment)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

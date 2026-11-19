"""A recorded alarm feed: topic messages replayed from a file through the real fold.

The file is JSONL, one topic message per line: ``{"key", "value", "time"}``, with
``time`` an ISO-8601 string with a zone (``Z`` or an offset). A message with no
``time`` keeps what its value records. Times are rebased at load so the newest
equals the load moment, so a replay never ages; a trailing ``{"feed": "stopped"}``
line is what makes one stale. A line that cannot be read, a time with no zone
among them, fails the load: half a feed could read as a clear state.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime, ConfigDict, ValidationError, field_validator
from typing_extensions import override

from osprey_connectors.alarm_state._contract import (
    AlarmCondition,
    AlarmConfiguration,
    AlarmState,
    AlarmStates,
)
from osprey_connectors.alarm_state._fold import (
    FEED_EMPTY,
    FEED_LOADING,
    FeedSnapshot,
    config_record,
    fold_message,
    state_record,
    state_rows,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.model import FrozenModel, parse_instant
from osprey_connectors.service_answers import BackendSpec, NotListed, Unavailable

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

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        self._task: asyncio.Task[None] | None = None
        self._snapshot: FeedSnapshot | None = None
        self._stopped = False
        self._loaded_at: datetime | None = None
        self._failed: Unavailable | None = None

    @override
    async def start(self) -> None:
        path = self._replay_path()
        if isinstance(path, Unavailable):
            return
        if self._task is None:
            self._task = asyncio.create_task(self._load(path), name="alarm-replay-load")
        # wait, not await: a cancelled caller leaves the load running for the next
        # one, and aclose cancelling the load does not raise in this caller.
        await asyncio.wait({self._task})

    @override
    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._snapshot = None
        self._stopped = False
        self._loaded_at = None
        self._failed = None

    @override
    async def state_for(
        self, address: str, *, timeout_s: float
    ) -> AlarmCondition | NotListed | Unavailable:
        await self._loading(timeout_s)
        snapshot = self._answering()
        if isinstance(snapshot, Unavailable):
            return snapshot
        return state_record(snapshot, address, stale=self._stopped)

    @override
    async def states(self, *, timeout_s: float) -> AlarmStates | Unavailable:
        """Every channel's state; ``last_contact_at`` is the load moment, the newest rebased time."""
        await self._loading(timeout_s)
        snapshot = self._answering()
        if isinstance(snapshot, Unavailable):
            return snapshot
        rows = state_rows(snapshot, stale=self._stopped)
        return AlarmStates(rows=rows, stale=self._stopped, last_contact_at=self._loaded_at)

    @override
    async def config_for(
        self, address: str, *, timeout_s: float
    ) -> AlarmConfiguration | Unavailable:
        await self._loading(timeout_s)
        snapshot = self._answering()
        return snapshot if isinstance(snapshot, Unavailable) else config_record(snapshot, address)

    @override
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        snapshot = self._loaded()
        if isinstance(snapshot, Unavailable):
            return snapshot
        evidence = f"read {snapshot.messages} messages from the replay file"
        return f"{evidence}; the feed stopped" if self._stopped else evidence

    async def _loading(self, timeout_s: float) -> None:
        """Wait up to *timeout_s* seconds for a load that has begun and not ended."""
        task = self._task
        if task is None or task.done() or timeout_s <= 0:
            return
        # wait, not wait_for: a deadline that runs out leaves the load running.
        await asyncio.wait({task}, timeout=timeout_s)

    def _answering(self) -> FeedSnapshot | Unavailable:
        snapshot = self._loaded()
        if isinstance(snapshot, FeedSnapshot) and snapshot.messages == 0:
            return Unavailable(cause="unreachable", reason=FEED_EMPTY)
        return snapshot

    def _loaded(self) -> FeedSnapshot | Unavailable:
        path = self._replay_path()
        if isinstance(path, Unavailable):
            return path
        if self._failed is not None:
            return self._failed
        return (
            self._snapshot
            if self._snapshot is not None
            else Unavailable(cause="not_ready", reason=FEED_LOADING)
        )

    def _replay_path(self) -> Path | Unavailable:
        value = self.spec.settings.get("replay_file")
        if not value:
            return Unavailable(cause="misconfigured", reason="alarm_state.replay_file is not set")
        if not isinstance(value, str | os.PathLike):
            return Unavailable(
                cause="misconfigured", reason="alarm_state.replay_file is not a path"
            )
        if str(value).startswith("demo:"):
            return Unavailable(
                cause="misconfigured", reason=f"alarm replay {value} was not resolved to a file"
            )
        path = Path(value)
        if path.is_absolute():
            return path
        if self.project_dir is None:
            return Unavailable(
                cause="misconfigured",
                reason=f"alarm replay file {value} is relative, with no project directory",
            )
        return self.project_dir / path

    async def _load(self, path: Path) -> None:
        try:
            loaded = await asyncio.to_thread(_read_replay, path)
            self._snapshot, self._stopped, self._loaded_at = loaded
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            logger.warning("alarm replay not loaded: %s", reason)
            self._failed = Unavailable(
                cause="failed", reason=f"alarm replay file cannot be read: {reason}"
            )


class _Message(FrozenModel):
    """One topic message of the file; a key it does not name is ignored.

    ``time`` is read with :func:`~osprey_connectors.model.parse_instant`. A time
    with no zone names no instant, so it fails the load rather than being read
    in a guessed one.
    """

    model_config = ConfigDict(extra="ignore")

    key: str | None = None
    value: dict[str, object] | str | None = None
    time: AwareDatetime | None = None

    @field_validator("time", mode="before")
    @classmethod
    def _iso_time(cls, recorded: object) -> object:
        return parse_instant(recorded) if isinstance(recorded, str) else recorded


def _read_replay(path: Path) -> tuple[FeedSnapshot, bool, datetime]:
    """Fold the file at *path*.

    The second answer is whether it ends with the stopped marker, the third the
    load moment, which the newest message time is rebased to.
    """
    messages: list[_Message] = []
    stopped = False
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                parsed: object = json.loads(line)
                stopped = parsed == _STOPPED_MARKER
                if not stopped:
                    messages.append(_Message.model_validate(parsed))
            except ValidationError as exc:
                raise ValueError(f"{path} line {number}: {_refused(exc)}") from exc
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path} line {number}: {exc}") from exc

    snapshot = FeedSnapshot()
    loaded_at = _now()
    times = [message.time for message in messages if message.time is not None]
    shift = loaded_at - max(times) if times else None
    for message in messages:
        rebased = None
        if message.time is not None and shift is not None:
            rebased = message.time + shift
        fold_message(snapshot, message.key, message.value, event_at=rebased)
    return snapshot, stopped, loaded_at


def _refused(error: ValidationError) -> str:
    """Why a line is not a message: each field refused, and the reason."""
    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc']) or 'a message'}: {detail['msg']}"
        for detail in error.errors(include_url=False)
    )

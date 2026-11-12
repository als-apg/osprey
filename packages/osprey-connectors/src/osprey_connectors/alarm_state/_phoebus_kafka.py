"""The Phoebus alarm server's live state, read from its compacted Kafka topic.

The topic is read from the beginning with no consumer group, so the service sees
no member and no committed offset, and nothing is ever produced. ``aiokafka`` is
an optional extra, imported only when a consumer is about to be built.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from osprey_connectors.alarm_state._contract import (
    AlarmConfigRecord,
    AlarmState,
    AlarmStateRecord,
)
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

logger = get_logger("phoebus_alarm_state")

KAFKA_CLIENT_MISSING = "Kafka client not installed (install osprey-connectors[alarm-state])"
SETTINGS_MISSING = "alarm_state.settings needs bootstrap_servers and topic"

#: Seconds after which a feed that has been silent reads as stale.
_STALE_WINDOW_S = 60.0
_POLL_TIMEOUT_MS = 1000
_OFFSET_CHECK_S = _STALE_WINDOW_S / 4
_RETRY_S = 5.0
#: How long a load may go without a message before the broker counts as lost. aiokafka
#: retries a lost connection inside ``getmany`` and ``position``, so neither raises.
_LOAD_DEADLINE_S = _STALE_WINDOW_S
_now = time.monotonic


class PhoebusAlarmState(AlarmState):
    """Follow one alarm configuration's topic and answer from the folded snapshot.

    ``consumer_factory(settings)`` returns an unstarted consumer with aiokafka's
    surface (``start``, ``stop``, ``assignment``, ``end_offsets``,
    ``seek_to_beginning``, ``position``, ``getmany``), already subscribed to the
    topic. A topic re-read after a lost consumer folds into a new snapshot, so the
    old one keeps answering until the new one has caught up. A read of a started
    feed waits up to its deadline for the load to catch up or to find the broker
    unreachable.
    """

    def __init__(
        self,
        spec: BackendSpec,
        *,
        project_dir: Path | None = None,
        consumer_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        **seams: Any,
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        self._consumer_factory = consumer_factory
        self._task: asyncio.Task[None] | None = None
        self._snapshot = FeedSnapshot()
        self._caught_up = False
        self._unreachable: str | None = None
        #: Set once a read has its answer: caught up, or the broker found unreachable.
        self._settled = asyncio.Event()
        self._last_contact = 0.0
        self._cannot_start: Unavailable | None = None

    async def start(self) -> None:
        if self._task is None and self._blocked() is None:
            self._task = asyncio.create_task(self._run(), name="phoebus-alarm-state")

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._snapshot = FeedSnapshot()
        self._caught_up = False
        self._unreachable = None
        self._settled.clear()

    async def state_for(
        self, channel: str, deadline: float
    ) -> AlarmStateRecord | Unavailable | None:
        await self._settling(deadline)
        unanswered = self._unanswered()
        if unanswered is not None:
            return unanswered
        return state_record(self._snapshot, channel, stale=self._stale())

    async def config_for(self, channel: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        await self._settling(deadline)
        unanswered = self._unanswered()
        return unanswered if unanswered is not None else config_record(self._snapshot, channel)

    async def probe(self, deadline: float) -> str | Unavailable:  # noqa: ARG002
        before_catch_up = self._before_catch_up()
        if before_catch_up is not None:
            return before_catch_up
        age = _now() - self._last_contact
        return f"read {self._snapshot.messages} messages; last contact {age:.0f} s ago"

    async def _settling(self, deadline: float) -> None:
        """Wait up to *deadline* seconds for a started feed to catch up or be found unreachable."""
        if self._task is None or self._settled.is_set() or deadline <= 0:
            return
        with suppress(TimeoutError):
            async with asyncio.timeout(deadline):
                await self._settled.wait()

    def _unanswered(self) -> Unavailable | None:
        before_catch_up = self._before_catch_up()
        if before_catch_up is not None:
            return before_catch_up
        return Unavailable(FEED_EMPTY) if self._snapshot.messages == 0 else None

    def _before_catch_up(self) -> Unavailable | None:
        blocked = self._blocked()
        if blocked is not None:
            return blocked
        if self._caught_up:
            return None
        return Unavailable(self._unreachable or FEED_LOADING)

    def _blocked(self) -> Unavailable | None:
        """Why no consumer can be built; decided once, on first need, not in the constructor."""
        if self._cannot_start is None and self._consumer_factory is None:
            settings = self.spec.settings
            if not (settings.get("bootstrap_servers") and settings.get("topic")):
                self._cannot_start = Unavailable(SETTINGS_MISSING)
            else:
                try:
                    importlib.import_module("aiokafka")
                except ImportError:
                    self._cannot_start = Unavailable(KAFKA_CLIENT_MISSING)
        return self._cannot_start

    def _stale(self) -> bool:
        return _now() - self._last_contact > _STALE_WINDOW_S

    async def _run(self) -> None:
        while True:
            try:
                await self._follow()
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                logger.warning("alarm feed lost: %s", reason)
                self._unreachable = f"alarm broker cannot be reached: {reason}"
                self._settled.set()
            await asyncio.sleep(_RETRY_S)

    async def _follow(self) -> None:
        consumer = (self._consumer_factory or _aiokafka_consumer)(self.spec.settings)
        try:
            await consumer.start()
            partitions = consumer.assignment()
            targets = await consumer.end_offsets(partitions)
            await consumer.seek_to_beginning()
            fresh = FeedSnapshot()
            loading = True
            last_check = _now()
            while True:
                batches = await consumer.getmany(timeout_ms=_POLL_TIMEOUT_MS)
                received = False
                for records in batches.values():
                    for record in records:
                        fold_message(fresh, record.key, record.value)
                        received = True
                if loading:
                    if received:
                        last_check = _now()
                    if await _reached(consumer, targets):
                        loading = False
                        self._snapshot = fresh
                        self._caught_up = True
                        self._unreachable = None
                        self._settled.set()
                        self._last_contact = last_check = _now()
                    elif _now() - last_check >= _LOAD_DEADLINE_S:
                        raise ConnectionError(f"no message in {_LOAD_DEADLINE_S:g} s of loading")
                elif received:
                    self._last_contact = last_check = _now()
                elif _now() - last_check >= _OFFSET_CHECK_S:
                    last_check = _now()
                    try:
                        await consumer.end_offsets(partitions)
                    except Exception as exc:
                        logger.debug("alarm feed end-offset check failed: %s", exc)
                    else:
                        self._last_contact = _now()
        finally:
            with suppress(Exception):
                await consumer.stop()


async def _reached(consumer: Any, targets: Mapping[Any, int]) -> bool:
    # position waits without end for a partition with no valid position.
    try:
        async with asyncio.timeout(_LOAD_DEADLINE_S):
            for partition, target in targets.items():
                if await consumer.position(partition) < target:
                    return False
    except TimeoutError:
        return False
    return True


def _aiokafka_consumer(settings: Mapping[str, Any]) -> Any:
    aiokafka = importlib.import_module("aiokafka")
    return aiokafka.AIOKafkaConsumer(
        settings["topic"],
        bootstrap_servers=settings["bootstrap_servers"],
        group_id=None,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )

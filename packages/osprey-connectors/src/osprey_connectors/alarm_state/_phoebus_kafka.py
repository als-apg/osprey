"""The Phoebus alarm server's live state, read from its compacted Kafka topic.

The topic is read from the beginning with no consumer group, so the service sees
no member and no committed offset, and nothing is ever produced. ``aiokafka`` is
an optional extra, imported only when a consumer is about to be built.

A broker that asks for a login is named one under ``settings.auth``: ``username``
with ``password_env``, sent as SASL (``auth.mechanism``, ``PLAIN`` unless set).
A login is sent over TLS, verified against ``settings.tls.ca_bundle`` or, without
one, the process trust store. Only ``auth.cleartext: true`` sends it unencrypted,
for a listener that offers no TLS. Without a login, ``tls.ca_bundle`` alone turns
TLS on. Reading needs one right on the broker, ``Read`` on the topic: with no
consumer group there is no group to authorize.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typing_extensions import override

from osprey_connectors.alarm_state._contract import (
    AlarmConfigRecord,
    AlarmState,
    AlarmStateRecord,
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
from osprey_connectors.connection import ConnectionSettings, read_connection_settings
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable

logger = get_logger("phoebus_alarm_state")

KAFKA_CLIENT_MISSING = "Kafka client not installed (install osprey-connectors[alarm-state])"
SETTINGS_MISSING = "alarm_state.settings needs bootstrap_servers and topic"

_WHERE = "pv_directory.alarm_state.settings"
_BEARER_REFUSED = "a Kafka broker takes `auth.username` and `auth.password_env`, sent as SASL"
#: The SASL mechanisms a password login can be sent with; the first is the default.
SASL_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512")

#: Seconds after which a feed that has been silent reads as stale.
_STALE_WINDOW_S = 60.0
_POLL_TIMEOUT_MS = 1000
_OFFSET_CHECK_S = _STALE_WINDOW_S / 4
_RETRY_S = 5.0
#: How long a load may go without a message before the broker counts as lost. aiokafka
#: retries a lost connection inside ``getmany`` and ``position``, so neither raises.
_LOAD_DEADLINE_S = _STALE_WINDOW_S
_now = time.monotonic


def _wall_now() -> datetime:
    return datetime.now(UTC)


class _Record(Protocol):
    """One topic message, as the consumer hands it over."""

    @property
    def key(self) -> str | bytes | None: ...

    @property
    def value(self) -> str | bytes | None: ...


class _Consumer(Protocol):
    """The part of aiokafka's ``AIOKafkaConsumer`` read here; a partition is opaque."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def assignment(self) -> object: ...

    async def end_offsets(self, partitions: object) -> Mapping[object, int]: ...

    async def seek_to_beginning(self) -> object: ...

    async def position(self, partition: object) -> int: ...

    async def getmany(self, *, timeout_ms: int) -> Mapping[object, Sequence[_Record]]: ...


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
        consumer_factory: Callable[[Mapping[str, object]], _Consumer] | None = None,
        **seams: object,
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
        #: The wall-clock time of ``_last_contact``, as ISO-8601 UTC; ``None`` before any.
        self._last_contact_at: str | None = None
        self._cannot_start: Unavailable | None = None

    @override
    async def start(self) -> None:
        if self._task is None and self._blocked() is None:
            self._task = asyncio.create_task(self._run(), name="phoebus-alarm-state")

    @override
    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._snapshot = FeedSnapshot()
        self._caught_up = False
        self._unreachable = None
        self._settled.clear()

    @override
    async def state_for(
        self, channel: str, deadline: float
    ) -> AlarmStateRecord | Unavailable | None:
        await self._settling(deadline)
        unanswered = self._unanswered()
        if unanswered is not None:
            return unanswered
        return state_record(self._snapshot, channel, stale=self._stale())

    @override
    async def states(self, deadline: float) -> AlarmStates | Unavailable:
        await self._settling(deadline)
        unanswered = self._unanswered()
        if unanswered is not None:
            return unanswered
        stale = self._stale()
        rows = state_rows(self._snapshot, stale=stale)
        return AlarmStates(rows=rows, stale=stale, last_contact=self._last_contact_at)

    @override
    async def config_for(self, channel: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        await self._settling(deadline)
        unanswered = self._unanswered()
        return unanswered if unanswered is not None else config_record(self._snapshot, channel)

    @override
    async def probe(self, deadline: float) -> str | Unavailable:
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
                return self._cannot_start
            try:
                _read_login(settings)
            except ValueError as exc:
                self._cannot_start = Unavailable(str(exc))
            else:
                try:
                    importlib.import_module("aiokafka")
                except ImportError:
                    self._cannot_start = Unavailable(KAFKA_CLIENT_MISSING)
        return self._cannot_start

    def _stale(self) -> bool:
        return _now() - self._last_contact > _STALE_WINDOW_S

    def _contact(self) -> float:
        """Record contact with the feed now, on both clocks; returns the monotonic time."""
        self._last_contact = _now()
        moment = _wall_now().isoformat(timespec="milliseconds")
        self._last_contact_at = moment.replace("+00:00", "Z")
        return self._last_contact

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
                        last_check = self._contact()
                    elif _now() - last_check >= _LOAD_DEADLINE_S:
                        raise ConnectionError(f"no message in {_LOAD_DEADLINE_S:g} s of loading")
                elif received:
                    last_check = self._contact()
                elif _now() - last_check >= _OFFSET_CHECK_S:
                    last_check = _now()
                    try:
                        await consumer.end_offsets(partitions)
                    except Exception as exc:
                        logger.debug("alarm feed end-offset check failed: %s", exc)
                    else:
                        self._contact()
        finally:
            with suppress(Exception):
                await consumer.stop()


async def _reached(consumer: _Consumer, targets: Mapping[object, int]) -> bool:
    # position waits without end for a partition with no valid position.
    try:
        async with asyncio.timeout(_LOAD_DEADLINE_S):
            for partition, target in targets.items():
                if await consumer.position(partition) < target:
                    return False
    except TimeoutError:
        return False
    return True


def _read_login(settings: Mapping[str, object]) -> tuple[ConnectionSettings, str, bool]:
    """The block's login and CA, the SASL mechanism, and whether the login goes unencrypted.

    Raises:
        ValueError: ``auth`` or ``tls`` breaks the connection shape, names a
            bearer token or a mechanism this backend does not send, or asks for
            a cleartext login beside a CA.
    """
    connection = read_connection_settings(
        settings,
        where=_WHERE,
        logins=frozenset({"password"}),
        unsupported_because=_BEARER_REFUSED,
        extra_auth_keys=frozenset({"mechanism", "cleartext"}),
    )
    auth = settings.get("auth")
    auth = auth if isinstance(auth, Mapping) else {}
    mechanism = auth.get("mechanism", SASL_MECHANISMS[0])
    if mechanism not in SASL_MECHANISMS:
        raise ValueError(f"`{_WHERE}.auth.mechanism` must be one of {', '.join(SASL_MECHANISMS)}")
    cleartext = auth.get("cleartext", False)
    if not isinstance(cleartext, bool):
        raise ValueError(f"`{_WHERE}.auth.cleartext` must be true or false")
    if cleartext and connection.ca_bundle is not None:
        raise ValueError(
            f"`{_WHERE}.auth.cleartext` sends the login unencrypted, and `{_WHERE}.tls.ca_bundle` "
            "names a CA for TLS: set one"
        )
    return connection, str(mechanism), cleartext


def _login_options(settings: Mapping[str, object]) -> dict[str, object]:
    """The consumer options a login and a CA add; none for an open plaintext broker.

    The password is read from the environment here, when the consumer is built.
    A login is encrypted unless ``auth.cleartext`` says otherwise; with no login,
    only a named CA turns TLS on.

    Raises:
        ValueError: As :func:`_read_login`, or the CA file cannot be used.
        ConnectionError: The variable ``auth.password_env`` names is unset.
    """
    connection, mechanism, cleartext = _read_login(settings)
    credential = connection.resolve_credential()
    encrypted = connection.ca_bundle is not None or (credential is not None and not cleartext)
    options: dict[str, object] = {}
    if encrypted:
        options["ssl_context"] = connection.ssl_context()
    if credential is not None:
        options["security_protocol"] = "SASL_SSL" if encrypted else "SASL_PLAINTEXT"
        options["sasl_mechanism"] = mechanism
        options["sasl_plain_username"] = credential.username
        options["sasl_plain_password"] = credential.secret
    elif encrypted:
        options["security_protocol"] = "SSL"
    return options


def _aiokafka_consumer(settings: Mapping[str, object]) -> _Consumer:
    aiokafka = importlib.import_module("aiokafka")
    consumer: _Consumer = aiokafka.AIOKafkaConsumer(
        settings["topic"],
        bootstrap_servers=settings["bootstrap_servers"],
        group_id=None,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        **_login_options(settings),
    )
    return consumer

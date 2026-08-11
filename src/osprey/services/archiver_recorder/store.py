"""Writing samples into the archive collection.

The document shape is not this module's invention: it is the one
``MongoDBArchiverConnector`` reads — one document per timestamp, ``date`` plus
a field per PV, and a PV absent from a document simply contributes no sample
for that channel. Writing anything else here would produce an archive the
product cannot read.

The one field the reader never sees is ``expireAt``, the per-document TTL
stamp. Retention is expressed per document rather than per collection because
the archive holds three kinds of history with three different lifetimes: dense
recent samples, coarse older ones, and scenario event windows that must not
expire at all. A collection-wide TTL could only express one of those. The TTL
index itself (``expireAfterSeconds=0`` on ``expireAt``) is created by the base
seeder during the staged bring-up, before this service starts — and a document
with no ``expireAt`` never expires under it, which is exactly how the seeder
protects an event window.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from .config import RecorderSettings

logger = logging.getLogger(__name__)


def expiry_for(timestamp: datetime, settings: RecorderSettings) -> datetime:
    """When the sample taken at ``timestamp`` should expire.

    Two tiers, decided by the timestamp alone. A sample that lands on the tail
    grid — an exact multiple of ``recorder_tail_cadence_sec`` since the epoch —
    is kept for the full retention span; every other sample is kept for the hot
    span only.

    That single rule is what keeps steady-state storage bounded without ever
    losing the recorded record. As the dense copy of a stretch of history ages
    past the hot span, what survives is one sample per tail interval — the same
    density the seeder already used for history of that age, so the archive
    thins out uniformly instead of ending abruptly at the point where recording
    began.

    Because the grid is absolute (multiples since the epoch, not since this
    process started), a restart lands on the same instants the previous run
    did, and on the same instants the seeder used.
    """
    on_tail_grid = int(timestamp.timestamp()) % settings.tail_cadence_sec == 0
    span = (
        timedelta(days=settings.retention_days)
        if on_tail_grid
        else timedelta(hours=settings.hot_span_hours)
    )
    return timestamp + span


class ArchiveWriter:
    """A live connection to the archive collection, held open for the process.

    Deliberately synchronous: pymongo's own client is, the recorder writes once
    per cadence rather than continuously, and the caller runs the write off the
    event loop.
    """

    def __init__(self, settings: RecorderSettings, password: str) -> None:
        self._settings = settings
        self._password = password
        # Untyped because pymongo is imported inside connect(): the module has
        # to stay importable wherever the optional extra is absent.
        self._client: Any = None
        self._collection: Any = None

    def connect(self) -> None:
        """Open the connection and prove the server answers.

        Raises:
            ConnectionError: if the store cannot be reached or authenticated.
                Fatal at startup — a recorder that cannot write is not degraded,
                it is pointless, and failing loudly is what puts the cause in
                ``docker logs`` instead of leaving a silently empty archive.
        """
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError

        settings = self._settings
        try:
            client: MongoClient = MongoClient(
                host=settings.host,
                port=settings.port,
                username=settings.username,
                password=self._password,
                authSource=settings.auth_database,
                serverSelectionTimeoutMS=settings.timeout_sec * 1000,
            )
            client.admin.command("ping")
        except PyMongoError as exc:
            raise ConnectionError(
                f"cannot reach the archive store at {settings.host}:{settings.port} "
                f"as {settings.username!r}: {exc}"
            ) from exc

        self._client = client
        self._collection = client[settings.database][settings.collection]
        logger.info(
            "Archive store connected: %s:%s/%s.%s",
            settings.host,
            settings.port,
            settings.database,
            settings.collection,
        )

    def write_sample(self, timestamp: datetime, values: Mapping[str, float]) -> None:
        """Store one instant's readings as one document.

        Merges rather than replaces, and stamps the expiry only when it creates
        the document. Both matter for coexistence with the seeded half of the
        collection: merging leaves any value already stored at this instant for
        a channel this recorder does not read, and stamping on insert only
        means a document whose expiry the scenario seeder deliberately removed
        (an event window, protected from retention) never gets one put back.
        """
        if self._collection is None:
            raise RuntimeError("ArchiveWriter.connect() has not been called")
        if not values:
            return
        self._collection.update_one(
            {"date": timestamp},
            {
                "$set": dict(values),
                "$setOnInsert": {"expireAt": expiry_for(timestamp, self._settings)},
            },
            upsert=True,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None

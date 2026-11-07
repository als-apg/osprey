"""A throwaway archive store, written the way OSPREY's recorder and seeder write it.

The recorder adapter reads a collection two OSPREY processes write: the
archiver-recorder, one document per instant, and the scenario seeder, which also
leaves one undated manifest document. Samples here go through the recorder's own
:class:`~osprey.services.archiver_recorder.store.ArchiveWriter`, and the manifest
carries the seeder's ``_id`` and no ``date``, so the adapter reads what a
deployment holds rather than a shape the tests invented.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tests._mongo_container import MONGO_AUTH_DB, started_mongo

from osprey.services.archiver_recorder.config import RecorderSettings
from osprey.services.archiver_recorder.store import ArchiveWriter
from osprey_connectors.simulation.archive import DATE_FIELD, EXPIRE_FIELD, MANIFEST_ID

#: The collection every test reads, as a deployment names it.
COLLECTION = "pv_history"

#: The environment variable the adapter's ``api_key_env`` names.
PASSWORD_ENV = "OSPREY_TEST_RECORDER_PW"

USERNAME = "osprey"
PASSWORD = "recorder-test-pw"


@dataclass(frozen=True)
class Store:
    """A started MongoDB the recorder can write to."""

    host: str
    port: int

    def url(self, database: str) -> str:
        """The adapter's URL for *database*; the password comes from :data:`PASSWORD_ENV`."""
        return f"mongodb://{USERNAME}@{self.host}:{self.port}/{database}?authSource={MONGO_AUTH_DB}"

    def client(self) -> Any:
        from pymongo import MongoClient

        return MongoClient(
            self.host,
            self.port,
            username=USERNAME,
            password=PASSWORD,
            authSource=MONGO_AUTH_DB,
            tz_aware=True,
        )

    def writer(self, database: str) -> ArchiveWriter:
        """The recorder's writer, connected to *database*'s :data:`COLLECTION`."""
        settings = RecorderSettings(
            host=self.host,
            port=self.port,
            database=database,
            collection=COLLECTION,
            auth_source=MONGO_AUTH_DB,
            username=USERNAME,
            password_env=PASSWORD_ENV,
            timeout_s=10,
            cadence_sec=1,
            tail_cadence_sec=60,
            poll_sec=1,
            hot_span_hours=24,
            retention_days=30,
        )
        writer = ArchiveWriter(settings, PASSWORD)
        writer.connect()
        return writer


def started_store() -> Iterator[Store]:
    """A :class:`Store`, skipped where no container engine or mongodb extra exists."""
    with started_mongo("recorder store", username=USERNAME, password=PASSWORD) as (host, port):
        yield Store(host, port)


def prepare_collection(collection: Any) -> None:
    """Create the indexes the seeder creates: ``date`` and the ``expireAt`` TTL."""
    collection.create_index(DATE_FIELD)
    collection.create_index(EXPIRE_FIELD, expireAfterSeconds=0)


def utc_slot(epoch_seconds: float) -> datetime:
    """The recorder's sample instant for *epoch_seconds*: whole seconds, UTC."""
    return datetime.fromtimestamp(int(epoch_seconds), tz=UTC)


def seed_samples(
    writer: ArchiveWriter, samples: Iterable[tuple[datetime, Mapping[str, float]]]
) -> None:
    """Write each ``(instant, {pv: value})`` pair through the recorder."""
    for timestamp, values in samples:
        writer.write_sample(timestamp, values)


def write_manifest(collection: Any) -> None:
    """The seeder's undated manifest document, replacing any previous one."""
    collection.replace_one(
        {"_id": MANIFEST_ID},
        {
            "_id": MANIFEST_ID,
            "fingerprint": {"source": "osprey-tests"},
            "seeded_at": datetime.now(UTC),
            "touched_windows": [],
        },
        upsert=True,
    )

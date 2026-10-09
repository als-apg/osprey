"""A watch pass against a llama-server that never answers.

One watch pass is what ``osprey ariel watch`` runs per poll: the scheduler's
``poll_once`` followed by ``run_catchup`` under the poll's catch-up budget. Two
silent-server cases run it on a fresh scratch database holding two entries
with one copied picture each:

* the whole server is silent (every route accepts and never answers): the pass
  returns within the 5 s health timeout + 1 s, no status is touched and the
  module is reported ``unreachable``;
* ``/v1/models`` answers the configured alias but ``/v1/embeddings`` hangs, with
  ``timeout_seconds: 2``: the pass ends after two consecutive timeouts, within
  2 x 2 s + 5 s + 1 s, with one WARNING and no row in the picture table.

In both, the next poll still ingests a new entry: a dead model server never
stops ingestion.

Fakes, at their boundaries: the source adapter hands out queued entries, the
attachment fetch answers the two recorded probe pictures, and every fetched
picture is its own rendition. ``image_embedding`` runs the real llama-cpp
adapter against the shared ``llama_stub``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import image_table_name
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.availability import HEALTH_TIMEOUT_S
from osprey.services.ariel_search.enhancement.image_embedding import module as embed_mod
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from tests.services.ariel_search.llama_stub import FIXTURES
from tests.services.ariel_search.llama_stub import MODEL as LLAMA_MODEL

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

DIMS = 1024
EMBED = "image_embedding"
ORIGINS = frozenset({("https", "h.example", 443)})
TS = datetime(2026, 9, 1, tzinfo=UTC)
PICTURES = ("orbit_kick", "tunnel_temp")
EMBED_TIMEOUT_S = 2
#: Slack on every wall-clock bound.
SLACK_S = 1.0


# --- fakes -------------------------------------------------------------------------


class _Adapter(FacilityAdapter):
    """An http source on ``h.example``; each poll hands out what was queued since."""

    queue: list[dict[str, Any]] = []

    @property
    def source_system_name(self) -> str:
        return "Stub"

    def attachment_origins(self) -> frozenset:
        return ORIGINS

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        entries, _Adapter.queue[:] = list(_Adapter.queue), []
        for entry in entries:
            yield entry


# --- helpers -----------------------------------------------------------------------


def _url(name: str) -> str:
    return f"https://h.example/files/{name}.png"


def _picture_bytes(url: str) -> bytes:
    return (FIXTURES / url.rsplit("/", 1)[-1]).read_bytes()


def _entry(entry_id: str, picture: str | None, minutes: int) -> dict[str, Any]:
    attachments = (
        [{"url": _url(picture), "type": "image/png", "filename": f"{picture}.png"}]
        if picture
        else []
    )
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": TS + timedelta(minutes=minutes),
        "author": "tester",
        "raw_text": f"beam note {entry_id}",
        "attachments": attachments,
        "metadata": {},
        "enhancement_status": {},
    }


def _config_dict(uri: str, embed_url: str, **embed: Any) -> dict[str, Any]:
    return {
        "database": {"uri": uri},
        "attachments": {"copy_on_ingest": "images"},
        "ingestion": {
            "adapter": "generic_json",
            "source_url": "https://h.example/api",
            "watch": {"require_initial_ingest": False},
        },
        "search_modules": {"keyword": {"enabled": True}, "hybrid": {"enabled": True}},
        "enhancement_modules": {
            EMBED: {
                "enabled": True,
                "provider": {"name": "llama-cpp", "base_url": embed_url},
                "model": LLAMA_MODEL,
                "dimensions": DIMS,
                **embed,
            }
        },
    }


def _fake_prepare(monkeypatch) -> None:
    """Every fetched picture is its own rendition; no render worker starts."""
    from osprey.services.ariel_search.attachments import prepare as prepare_mod

    async def _prepare(data, **_kwargs):
        return prepare_mod.PreparedPicture(
            mime_type="image/png",
            skip_reason=None,
            rendition_bytes=bytes(data),
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256=hashlib.sha256(data).hexdigest(),
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


def _statuses(uri: str) -> dict[str, dict[str, Any]]:
    with psycopg.connect(uri) as conn:
        rows = conn.execute("SELECT entry_id, enhancement_status FROM enhanced_entries").fetchall()
    return {r[0]: r[1] or {} for r in rows}


def _copied(uri: str) -> int:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT count(*) FROM attachment_files WHERE copy_status = 'copied'"
        ).fetchone()
    assert row is not None
    return row[0]


def _image_rows(uri: str) -> int:
    with psycopg.connect(uri) as conn:
        row = conn.execute(f"SELECT count(*) FROM {image_table_name(LLAMA_MODEL, DIMS)}").fetchone()
    assert row is not None
    return row[0]


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


class _Watch:
    """One store and its scheduler; :meth:`pass_` runs one watch pass."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.config = ARIELConfig.from_dict(raw)
        self.pool: Any = None
        self.scheduler: Any = None

    async def __aenter__(self) -> _Watch:
        from osprey.services.ariel_search.database import ARIELRepository
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.migrations import run_migrations
        from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

        self.pool = await create_connection_pool(self.config.database)
        await run_migrations(self.pool, self.config)
        repository = ARIELRepository(self.pool, self.config)
        self.scheduler = IngestionScheduler(config=self.config, repository=repository)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.pool.close()

    async def poll(self) -> Any:
        return await self.scheduler.poll_once()

    async def pass_(self) -> float:
        """``poll_once`` then ``run_catchup``, as ``run_watch`` runs them; returns seconds."""
        started = time.monotonic()
        polled = await self.poll()
        await ops.run_catchup(
            self.raw,
            budget_s=ops.catchup_budget(self.raw, polled.duration_seconds),
            stop_event=self.scheduler._stop_event,
        )
        return time.monotonic() - started


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, attachment_fetch):
    """Fresh availability and offload state; the source and its pictures faked."""
    availability.reset_availability()
    _offload.reset_offload_state()
    _Adapter.queue[:] = []
    monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: True)
    monkeypatch.setattr(
        "osprey.services.ariel_search.ingestion.get_adapter", lambda config: _Adapter(config)
    )
    attachment_fetch.respond(lambda url, *_a, **_k: FetchOutcome(data=_picture_bytes(url)))
    _fake_prepare(monkeypatch)
    yield
    _Adapter.queue[:] = []
    availability.reset_availability()
    _offload.reset_offload_state()


def _queue_two_pictured_entries() -> None:
    _Adapter.queue[:] = [
        _entry("silent-1", PICTURES[0], 0),
        _entry("silent-2", PICTURES[1], 1),
    ]


async def _next_poll_ingests_a_new_entry(watch: _Watch, uri: str) -> None:
    _Adapter.queue[:] = [_entry("after-1", None, 10)]
    result = await watch.poll()
    assert result.entries_added == 1
    assert "after-1" in _statuses(uri)


# --- the two silent-server cases ---------------------------------------------------


async def test_a_silent_server_ends_the_pass_on_the_health_timeout(
    scratch_database, llama_stub, caplog
):
    stub = llama_stub()
    stub.silent = True
    _queue_two_pictured_entries()

    async with _Watch(_config_dict(scratch_database, stub.url)) as watch:
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            elapsed = await watch.pass_()

        assert elapsed < HEALTH_TIMEOUT_S + SLACK_S, elapsed
        assert _copied(scratch_database) == len(PICTURES)
        statuses = _statuses(scratch_database)
        assert set(statuses) == {"silent-1", "silent-2"}
        assert all(EMBED not in s for s in statuses.values()), statuses
        assert availability.current_reason(EMBED) == "unreachable"
        assert stub.embeddings == []
        assert _image_rows(scratch_database) == 0

        await _next_poll_ingests_a_new_entry(watch, scratch_database)


async def test_a_hanging_embeddings_route_ends_the_pass_after_two_timeouts(
    scratch_database, llama_stub, caplog
):
    stub = llama_stub()
    stub.hang = True
    _queue_two_pictured_entries()
    raw = _config_dict(scratch_database, stub.url, timeout_seconds=EMBED_TIMEOUT_S)

    async with _Watch(raw) as watch:
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            elapsed = await watch.pass_()

        bound = 2 * EMBED_TIMEOUT_S + HEALTH_TIMEOUT_S + SLACK_S
        assert elapsed < bound, elapsed
        assert _copied(scratch_database) == len(PICTURES)
        assert len(stub.embeddings) == 2
        assert availability.current_reason(EMBED) == "transient"
        warned = _warnings(caplog)
        assert len(warned) == 1, [r.getMessage() for r in warned]
        assert EMBED in warned[0].getMessage()
        assert _image_rows(scratch_database) == 0

        await _next_poll_ingests_a_new_entry(watch, scratch_database)

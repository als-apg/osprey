"""Acceptance rows of the picture-caption pipeline, each against a real PostgreSQL schema.

Pins how captioning sits between the poll and the catch-up: a poll never waits
on the vision model, one catch-up pass finishes a call that overruns its
budget, a stopped pass keeps what it captioned, a picture deleted mid-call
gets no caption and costs no attempt, disabling the module changes nothing a
reader sees, and a caption written in one pass reaches the qmd mirror in the
next.

Every case runs on a fresh scratch database. The real ``ImageCaptionModule``
runs with its vision call replaced by a fake that may sleep, stop the pass or
delete a row; the call runs on a daemon thread, so the fake touches the
database through its own synchronous connection.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.attachments.compose import caption_model_id
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.attachments.summaries import build_attachment_summaries
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import CopyRendition
from osprey.services.ariel_search.enhancement import _offload, availability, image_driver
from osprey.services.ariel_search.enhancement.image_caption import module as caption_mod
from osprey.services.ariel_search.enhancement.qmd_export.writer import mirror_path
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import ingest_one
from osprey.services.ariel_search.search.keyword import keyword_search

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

MODEL = "vis-caption-1"
CAPTION = "image_caption"
ORIGINS = frozenset({("https", "h.example", 443)})
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
RENDITION = CopyRendition(data=b"png", mime_type="image/png", width=1, height=1, sha256="cd" * 32)
TS = datetime(2026, 9, 1, tzinfo=UTC)
REPLY = "Klystron arc trace on the scope.\nVisible text: KLY-3"


# --- helpers -----------------------------------------------------------------------


class _Adapter(FacilityAdapter):
    """An http source whose pictures live on ``h.example``; it yields ``entries``."""

    def __init__(self, config: ARIELConfig, entries: list[dict] | None = None) -> None:
        super().__init__(config)
        self.entries = entries or []

    @property
    def source_system_name(self) -> str:
        return "Stub"

    def attachment_origins(self) -> frozenset:
        return ORIGINS

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in list(self.entries):
            yield dict(entry)


class _FakeVision:
    """Stands in for the vision call; ``step(call_number)`` runs before each reply."""

    def __init__(self, step: Callable[[int], None] | None = None, reply: str = REPLY) -> None:
        self.step = step
        self.reply = reply
        self.calls = 0

    def __call__(self, **_kwargs: Any) -> str:
        self.calls += 1
        if self.step is not None:
            self.step(self.calls)
        return self.reply


def _config_dict(uri: str, *, enabled: bool = True, mirror: Path | None = None) -> dict:
    """The raw ``ariel`` block: caption module (and optionally qmd_export) on *uri*."""
    modules: dict[str, Any] = {
        CAPTION: {
            "enabled": enabled,
            "provider": "openai",
            "model": {"model_id": MODEL},
            "timeout_seconds": 60,
        }
    }
    if mirror is not None:
        modules["qmd_export"] = {"enabled": True, "mirror_path": str(mirror)}
    return {
        "database": {"uri": uri},
        "attachments": {"copy_on_ingest": "images"},
        "ingestion": {
            "adapter": "generic_json",
            "source_url": "https://h.example/api",
            "watch": {"require_initial_ingest": False},
        },
        "search_modules": {"keyword": {"enabled": True}},
        "enhancement_modules": modules,
    }


def _url(name: str) -> str:
    return f"https://h.example/files/{name}"


def _entry(entry_id: str, names: list[str]) -> dict:
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": TS,
        "author": "tester",
        "raw_text": "beam lost at 14:02",
        "attachments": [{"url": _url(n), "type": "image/png", "filename": n} for n in names],
        "metadata": {},
        "enhancement_status": {},
    }


def _id(entry_id: str, name: str) -> str:
    aid = attachment_id_for(entry_id, {"url": _url(name)})
    assert aid is not None
    return aid


def _fake_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render every sniffed picture instantly."""
    from osprey.imaging.formats import sniff
    from osprey.services.ariel_search.attachments import prepare as prepare_mod

    async def _prepare(data, **_kwargs):
        sniffed = sniff(data)
        if not sniffed.is_image:
            return prepare_mod.PreparedPicture(sniffed.mime, sniffed.skip_reason)
        return prepare_mod.PreparedPicture(
            mime_type=sniffed.mime,
            skip_reason=None,
            rendition_bytes=b"r",
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256="cd" * 32,
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


async def _seed_copied(repo, uri: str, entry_id: str, names: list[str]) -> list[str]:
    """Store an entry with one PNG per name, every picture copied with a rendition."""
    cfg = ARIELConfig.from_dict(_config_dict(uri))
    await ingest_one(_entry(entry_id, names), _Adapter(cfg), repo, [], cfg, None)
    ids = [_id(entry_id, n) for n in names]
    for aid in ids:
        written = await repo.apply_copy_outcome(
            entry_id,
            aid,
            copy_status="copied",
            data=PNG_MAGIC,
            mime_type="image/png",
            size_bytes=len(PNG_MAGIC),
            rendition=RENDITION,
        )
        assert written == "copied"
    return ids


def _row(uri: str, entry_id: str) -> dict[str, Any]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status, attachment_captions, attachment_text"
            " FROM enhanced_entries WHERE entry_id = %s",
            (entry_id,),
        ).fetchone()
    assert row is not None
    return {"status": row[0] or {}, "captions": row[1] or {}, "text": row[2]}


def _caption_status(uri: str, entry_id: str) -> dict[str, Any]:
    return _row(uri, entry_id)["status"].get(CAPTION) or {}


def _captioned(uri: str, entry_id: str) -> set[str]:
    """Attachment ids holding a caption under ``MODEL``."""
    captions = _row(uri, entry_id)["captions"]
    return {aid for aid, per in captions.items() if isinstance(per, dict) and MODEL in per}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh availability and offload state; a configured provider and a healthy listing."""
    availability.reset_availability()
    _offload.reset_offload_state()
    monkeypatch.setattr("osprey.models.config.get_provider_config", lambda name: {"api_key": "k"})
    monkeypatch.setattr(
        caption_mod, "probe_models_endpoint", lambda *a, **k: HealthResult(True, "served", None)
    )
    yield
    availability.reset_availability()
    _offload.reset_offload_state()


@pytest.fixture
async def repo(scratch_database):
    """An ``ARIELRepository`` on a fresh scratch database at today's schema."""
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    cfg = ARIELConfig.from_dict(_config_dict(scratch_database))
    pool = await create_connection_pool(cfg.database)
    try:
        await run_migrations(pool, cfg)
        yield ARIELRepository(pool, cfg)
    finally:
        await pool.close()


def _vision(monkeypatch, fake: _FakeVision) -> _FakeVision:
    monkeypatch.setattr(caption_mod, "_chat_completion", fake)
    return fake


# --- the poll never waits on the model ------------------------------------------------


class TestPollLeavesCaptionsToTheCatchup:
    async def test_poll_with_a_ten_second_model_returns_fast_and_leaves_the_picture_pending(
        self, repo, scratch_database, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search import ingestion as ingestion_pkg
        from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        fake = _vision(monkeypatch, _FakeVision(lambda _n: time.sleep(10)))
        cfg = ARIELConfig.from_dict(_config_dict(scratch_database))
        adapter = _Adapter(cfg, [_entry("poll-1", ["a.png"])])
        monkeypatch.setattr(ingestion_pkg, "get_adapter", lambda _config: adapter)
        scheduler = IngestionScheduler(cfg, repo)

        started = time.monotonic()
        result = await scheduler.poll_once()
        elapsed = time.monotonic() - started

        assert result.entries_added == 1
        assert elapsed < 1.0
        assert fake.calls == 0
        rows = (await repo.get_attachment_rows(["poll-1"]))["poll-1"]
        assert [r["copy_status"] for r in rows] == ["copied"]
        assert _captioned(scratch_database, "poll-1") == set()
        assert _caption_status(scratch_database, "poll-1").get("status") != "complete"
        pending = await repo.get_incomplete_entries(module_name=CAPTION, limit=10, marker=MODEL)
        assert [e["entry_id"] for e in pending] == ["poll-1"]


# --- the catch-up pass ---------------------------------------------------------------


class TestCatchupPass:
    async def test_a_call_taking_twice_the_budget_completes_on_the_first_pass(
        self, repo, scratch_database, monkeypatch
    ):
        budget = 1.0
        monkeypatch.setattr(image_driver, "MIN_PICTURE_SECONDS", 0.5)
        (aid,) = await _seed_copied(repo, scratch_database, "slow-1", ["a.png"])
        fake = _vision(monkeypatch, _FakeVision(lambda _n: time.sleep(2 * budget)))

        started = time.monotonic()
        await ops.run_catchup(_config_dict(scratch_database), budget_s=budget, stop_event=None)
        elapsed = time.monotonic() - started

        assert fake.calls == 1
        assert elapsed >= 2 * budget
        assert _captioned(scratch_database, "slow-1") == {aid}
        status = _caption_status(scratch_database, "slow-1")
        assert (status.get("status"), status.get("marker")) == ("complete", MODEL)
        assert not status.get("attempts")

    async def test_a_pass_stopped_after_picture_two_keeps_captions_one_and_two(
        self, repo, scratch_database, monkeypatch
    ):
        ids = await _seed_copied(repo, scratch_database, "stop-1", ["a.png", "b.png", "c.png"])
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _stop_on_second(call: int) -> None:
            if call == 2:
                loop.call_soon_threadsafe(stop.set)

        fake = _vision(monkeypatch, _FakeVision(_stop_on_second))

        await ops.run_catchup(_config_dict(scratch_database), budget_s=None, stop_event=stop)

        assert fake.calls == 2
        assert _captioned(scratch_database, "stop-1") == {ids[0], ids[1]}
        status = _caption_status(scratch_database, "stop-1")
        assert status.get("status") != "complete"
        assert not status.get("attempts")

    async def test_a_picture_deleted_during_its_call_gets_no_caption_and_costs_no_attempt(
        self, repo, scratch_database, monkeypatch
    ):
        (aid,) = await _seed_copied(repo, scratch_database, "gone-1", ["a.png"])
        text_before = _row(scratch_database, "gone-1")["text"]

        def _delete(_call: int) -> None:
            with psycopg.connect(scratch_database, autocommit=True) as conn:
                conn.execute("DELETE FROM attachment_files WHERE attachment_id = %s", (aid,))

        fake = _vision(monkeypatch, _FakeVision(_delete))

        await ops.run_catchup(_config_dict(scratch_database), budget_s=None, stop_event=None)

        assert fake.calls == 1
        row = _row(scratch_database, "gone-1")
        assert aid not in row["captions"]
        assert row["text"] == text_before
        assert not (row["status"].get(CAPTION) or {}).get("attempts")


# --- disabling the module changes nothing a reader sees ------------------------------


async def _reader_view(repo, entry_id: str, cfg: ARIELConfig) -> tuple[Any, Any, Any]:
    """``attachment_text``, the summary caption and the keyword hit's matched ids."""
    entry = await repo.get_entry(entry_id)
    assert entry is not None
    rows = (await repo.get_attachment_rows([entry_id]))[entry_id]
    (summary,) = build_attachment_summaries(
        entry, rows, None, (), file_source=False, model_id=caption_model_id(cfg)
    )
    hits = await keyword_search("klystron", repo, cfg)
    assert isinstance(hits, list)
    matched = [h[0].get("_matched_attachment_ids") for h in hits if h[0]["entry_id"] == entry_id]
    return entry.get("attachment_text"), summary.get("caption"), matched


class TestDisableAndReingest:
    async def test_reingest_with_the_module_disabled_leaves_text_summary_and_matches(
        self, repo, scratch_database, monkeypatch
    ):
        (aid,) = await _seed_copied(repo, scratch_database, "dis-1", ["a.png"])
        _vision(monkeypatch, _FakeVision())
        await ops.run_catchup(_config_dict(scratch_database), budget_s=None, stop_event=None)
        enabled = ARIELConfig.from_dict(_config_dict(scratch_database))
        disabled = ARIELConfig.from_dict(_config_dict(scratch_database, enabled=False))

        before = await _reader_view(repo, "dis-1", enabled)
        text, caption, matched = before
        assert "Klystron arc trace" in (text or "")
        assert "Klystron arc trace" in (caption or "")
        assert matched == [[aid]]

        outcome = await ingest_one(
            _entry("dis-1", ["a.png"]), _Adapter(disabled), repo, [], disabled, None
        )
        assert outcome.attachments_recorded

        assert await _reader_view(repo, "dis-1", disabled) == before


# --- a caption reaches the qmd mirror on the next pass -------------------------------


class TestQmdMirror:
    async def test_a_caption_written_in_pass_n_is_in_the_mirror_after_pass_n_plus_one(
        self, repo, scratch_database, monkeypatch, tmp_path
    ):
        await _seed_copied(repo, scratch_database, "qmd-1", ["a.png"])
        fake = _vision(monkeypatch, _FakeVision())
        config = _config_dict(scratch_database, mirror=tmp_path)
        entry = await repo.get_entry("qmd-1")
        assert entry is not None
        path = mirror_path(tmp_path, entry)

        await ops.run_catchup(config, budget_s=None, stop_event=None)

        assert fake.calls == 1
        first = path.read_text()
        assert "Klystron arc trace" not in first

        await ops.run_catchup(config, budget_s=None, stop_event=None)

        assert fake.calls == 1
        second = path.read_text()
        assert "Captions:" in second
        assert "Klystron arc trace" in second

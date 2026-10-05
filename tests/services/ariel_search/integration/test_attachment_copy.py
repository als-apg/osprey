"""Integration tests for recording attachment rows from an entry's locked JSONB.

``record_rows`` and ``record_and_compose`` run against the real PostgreSQL schema
(testcontainer or the shared test database) inside a transaction holding the
entry lock, the way ingest and backfill call them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.attachments import copy as copy_mod
from osprey.services.ariel_search.attachments.copy import record_and_compose, record_rows
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.ingestion.base import FacilityAdapter

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

#: Rows this file seeds, one family recorded before any write.
PREFIX = "attcopy-"

STATUS_WITH_KEYS = {
    "image_caption": {"status": "complete"},
    "image_embedding": {"status": "complete"},
    "text_embedding": {"status": "complete"},
    "qmd_export": {"status": "complete"},
}


@pytest.fixture(autouse=True)
def _seed_prefix(seeded_prefixes):
    seeded_prefixes.add(PREFIX)


@pytest.fixture(autouse=True)
def _no_fetch(monkeypatch):
    """Recording rows never fetches a picture."""

    async def _refuse(*_args, **_kwargs):
        raise AssertionError("record_rows must not fetch")

    monkeypatch.setattr(copy_mod, "fetch_attachment_bytes", _refuse)


class _HttpAdapter(FacilityAdapter):
    """An http source: no file base."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():  # never yields
            yield entry


class _FileAdapter(_HttpAdapter):
    """A file source: relative attachment paths resolve under a base."""

    def attachment_file_base(self) -> Path | None:
        return Path("/nonexistent-base")


def _config(mode: str = "images", *, model_id: str | None = None) -> ARIELConfig:
    data: dict = {
        "database": {"uri": "postgresql://test"},
        "attachments": {"copy_on_ingest": mode},
    }
    if model_id is not None:
        data["enhancement_modules"] = {
            "image_caption": {"enabled": False, "model": {"provider": "x", "model_id": model_id}}
        }
    return ARIELConfig.from_dict(data)


def _http(config: ARIELConfig) -> _HttpAdapter:
    return _HttpAdapter(config)


async def _seed_entry(
    repository,
    entry_id: str,
    attachments: list,
    *,
    status: dict | None = None,
    attachment_text: str | None = None,
    captions: dict | None = None,
) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text,
                attachments, metadata, enhancement_status,
                attachment_text, attachment_captions
            ) VALUES (
                %(entry_id)s, 'test', %(ts)s, 'tester', 'x',
                %(attachments)s::jsonb, '{}'::jsonb, %(status)s::jsonb,
                %(text)s, %(captions)s::jsonb
            )
            """,
            {
                "entry_id": entry_id,
                "ts": datetime.now(UTC),
                "attachments": json.dumps(attachments),
                "status": json.dumps(status or {}),
                "text": attachment_text,
                "captions": json.dumps(captions) if captions is not None else None,
            },
        )


async def _set_attachments(repository, entry_id: str, attachments: list) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            "UPDATE enhanced_entries SET attachments = %(a)s::jsonb WHERE entry_id = %(e)s",
            {"a": json.dumps(attachments), "e": entry_id},
        )


async def _locked_row(conn, entry_id: str) -> dict:
    result = await conn.execute(
        """
        SELECT attachments, attachment_text, attachment_captions
        FROM enhanced_entries WHERE entry_id = %(e)s FOR UPDATE
        """,
        {"e": entry_id},
    )
    row = await result.fetchone()
    return {"attachments": row[0], "attachment_text": row[1], "attachment_captions": row[2]}


async def _record(repository, entry_id: str, config: ARIELConfig, adapter) -> list[str]:
    async with repository.pool.connection() as conn, conn.transaction():
        locked = await _locked_row(conn, entry_id)
        return await record_rows(conn, entry_id, locked["attachments"], config, adapter)


async def _compose(repository, entry_id: str, config: ARIELConfig, adapter) -> list[str]:
    async with repository.pool.connection() as conn, conn.transaction():
        locked = await _locked_row(conn, entry_id)
        return await record_and_compose(conn, entry_id, locked, config, adapter)


async def _rows(repository, entry_id: str) -> dict[str, dict]:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT attachment_id, filename, mime_type, data, size_bytes, source_url,
                   copy_status, skip_reason
            FROM attachment_files WHERE entry_id = %(e)s
            """,
            {"e": entry_id},
        )
        cols = [
            "attachment_id",
            "filename",
            "mime_type",
            "data",
            "size_bytes",
            "source_url",
            "copy_status",
            "skip_reason",
        ]
        return {r[0]: dict(zip(cols, r, strict=True)) for r in await result.fetchall()}


async def _entry(repository, entry_id: str) -> dict:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT enhancement_status, attachment_text, attachment_captions
            FROM enhanced_entries WHERE entry_id = %(e)s
            """,
            {"e": entry_id},
        )
        row = await result.fetchone()
        return {"status": row[0], "text": row[1], "captions": row[2]}


def _id(entry_id: str, url: str) -> str:
    aid = attachment_id_for(entry_id, {"url": url})
    assert aid is not None
    return aid


class TestRecordRows:
    async def test_record_rows_none_mode_gives_one_placeholder_per_fetchable_attachment(
        self, repository
    ):
        entry = f"{PREFIX}none-1"
        a, b = "https://h.example/a.png", "https://h.example/doc.pdf"
        await _seed_entry(
            repository,
            entry,
            [
                {"url": a, "type": "image/png", "filename": "a.png"},
                {"url": b, "type": "application/pdf"},
                {"url": "rel/c.png", "type": "image/png"},  # not fetchable on http
                {"url": "", "filename": "header-only"},
                {"url": "/api/attachments/native-1", "type": "image/png"},
                {"url": 123},
                None,
            ],
        )
        cfg = _config("none")
        keep = await _record(repository, entry, cfg, _http(cfg))

        rows = await _rows(repository, entry)
        assert set(rows) == {_id(entry, a), _id(entry, b)}
        for row in rows.values():
            assert row["copy_status"] == "skipped"
            assert row["skip_reason"] == "copy_on_ingest_mode"
            assert row["data"] is None and row["size_bytes"] is None
        assert rows[_id(entry, a)]["mime_type"] == "image/png"
        assert rows[_id(entry, b)]["mime_type"] == "application/pdf"
        assert rows[_id(entry, b)]["source_url"] == b
        assert sorted(keep) == sorted([a, b])

    async def test_record_rows_images_mode_pending_and_declared_ineligible(self, repository):
        entry = f"{PREFIX}img-1"
        png = "https://h.example/dir/shot.png?x=1"
        pdf = "https://h.example/report.pdf"
        bad = "https://h.example/weird"
        slash = "https://h.example/dir/"
        await _seed_entry(
            repository,
            entry,
            [
                {"url": png, "type": "IMAGE/PNG"},
                {"url": pdf, "type": "application/pdf", "filename": "Report.pdf"},
                {"url": bad, "type": "image/png; charset=x", "filename": 42},
                {"url": slash, "type": None, "filename": ""},
            ],
        )
        cfg = _config("images")
        await _record(repository, entry, cfg, _http(cfg))

        rows = await _rows(repository, entry)
        r_png = rows[_id(entry, png)]
        assert (r_png["copy_status"], r_png["skip_reason"]) == ("pending", None)
        assert r_png["mime_type"] == "image/png"
        assert r_png["filename"] == "shot.png"
        assert r_png["source_url"] == png

        r_pdf = rows[_id(entry, pdf)]
        assert (r_pdf["copy_status"], r_pdf["skip_reason"]) == ("skipped", "copy_on_ingest_mode")
        assert r_pdf["mime_type"] == "application/pdf"
        assert r_pdf["filename"] == "Report.pdf"

        r_bad = rows[_id(entry, bad)]
        assert r_bad["copy_status"] == "pending"
        assert r_bad["mime_type"] is None
        assert r_bad["filename"] == "42"

        r_slash = rows[_id(entry, slash)]
        assert r_slash["copy_status"] == "pending"
        assert r_slash["filename"] == "attachment"

    async def test_record_rows_relative_path_on_file_source_gets_pending_row(self, repository):
        entry = f"{PREFIX}file-1"
        await _seed_entry(repository, entry, [{"url": "pics/c.png", "type": "image/png"}])
        cfg = _config("images")
        keep = await _record(repository, entry, cfg, _FileAdapter(cfg))

        rows = await _rows(repository, entry)
        row = rows[_id(entry, "pics/c.png")]
        assert row["copy_status"] == "pending"
        assert row["filename"] == "c.png"
        assert keep == ["pics/c.png"]

    async def test_record_rows_unfetchable_url_makes_no_row_and_stays_out_of_keep(self, repository):
        entry = f"{PREFIX}unfetch-1"
        await _seed_entry(
            repository,
            entry,
            [
                {"url": "javascript:alert(1)"},
                {"url": "../escape.png"},
                {"url": "/abs/path.png"},
            ],
        )
        cfg = _config("all")
        keep = await _record(repository, entry, cfg, _FileAdapter(cfg))
        assert keep == []
        assert await _rows(repository, entry) == {}

    async def test_record_rows_native_png_plus_upstream_leave_exactly_two_rows(self, repository):
        entry = f"{PREFIX}native-1"
        native_url = "/api/attachments/attcopy-native-1-png"
        upstream = "https://h.example/up.png"
        await _seed_entry(
            repository,
            entry,
            [{"url": native_url, "type": "image/png"}, {"url": upstream, "type": "image/png"}],
        )
        await repository.insert_native_attachment(
            entry, "attcopy-native-1-png", filename="n.png", mime_type="image/png", data=b"\x89PNG"
        )
        cfg = _config("images")
        keep1 = await _record(repository, entry, cfg, _http(cfg))
        keep2 = await _compose(repository, entry, cfg, _http(cfg))

        rows = await _rows(repository, entry)
        assert set(rows) == {"attcopy-native-1-png", _id(entry, upstream)}
        assert rows["attcopy-native-1-png"]["source_url"] is None
        assert rows["attcopy-native-1-png"]["copy_status"] == "copied"
        assert keep1 == keep2 == [upstream]

    async def test_record_rows_on_conflict_leaves_existing_rows_and_clears_keys_only_on_insert(
        self, repository
    ):
        entry = f"{PREFIX}conflict-1"
        url = "https://h.example/a.png"
        await _seed_entry(repository, entry, [{"url": url, "type": "image/png"}])
        aid = _id(entry, url)
        async with repository.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO attachment_files (
                    attachment_id, entry_id, filename, mime_type, data, size_bytes,
                    source_url, copy_status
                ) VALUES (%(a)s, %(e)s, 'kept.png', 'image/png', %(d)s, 3, %(u)s, 'copied')
                """,
                {"a": aid, "e": entry, "d": b"abc", "u": url},
            )
            await conn.execute(
                "UPDATE enhanced_entries SET enhancement_status = %(s)s::jsonb "
                "WHERE entry_id = %(e)s",
                {"s": json.dumps(STATUS_WITH_KEYS), "e": entry},
            )
        cfg = _config("images")
        await _record(repository, entry, cfg, _http(cfg))

        row = (await _rows(repository, entry))[aid]
        assert (row["copy_status"], row["filename"], row["data"]) == ("copied", "kept.png", b"abc")
        status = (await _entry(repository, entry))["status"]
        assert "image_caption" in status and "image_embedding" in status

        # A new item inserts a row, which clears the image keys.
        await _set_attachments(
            repository,
            entry,
            [{"url": url, "type": "image/png"}, {"url": "https://h.example/b.png"}],
        )
        await _record(repository, entry, cfg, _http(cfg))
        status = (await _entry(repository, entry))["status"]
        assert "image_caption" not in status and "image_embedding" not in status
        assert "text_embedding" in status


class TestRecordAndCompose:
    async def test_record_rows_compose_deletes_dropped_url_and_prunes_captions(self, repository):
        entry = f"{PREFIX}drop-1"
        a, c = "https://h.example/a.png", "https://h.example/c.png"
        model = "vision-x"
        await _seed_entry(repository, entry, [{"url": a}, {"url": c}])
        cfg = _config("images", model_id=model)
        await _compose(repository, entry, cfg, _http(cfg))
        aid, cid = _id(entry, a), _id(entry, c)
        assert set(await _rows(repository, entry)) == {aid, cid}

        captions = {
            aid: {model: {"caption": "beam a", "visible_text": ""}},
            cid: {model: {"caption": "beam c", "visible_text": "BPM"}},
        }
        async with repository.pool.connection() as conn:
            await conn.execute(
                "UPDATE enhanced_entries SET attachment_captions = %(c)s::jsonb "
                "WHERE entry_id = %(e)s",
                {"c": json.dumps(captions), "e": entry},
            )
        await _set_attachments(
            repository, entry, [{"url": "", "filename": "hdr"}, {"url": c, "filename": "c.png"}]
        )
        keep = await _compose(repository, entry, cfg, _http(cfg))

        assert keep == [c]
        assert set(await _rows(repository, entry)) == {cid}
        stored = await _entry(repository, entry)
        assert set(stored["captions"]) == {cid}
        assert stored["text"] == (
            f"[picture c.png - machine caption by {model}] beam c Visible text: BPM"
        )

    async def test_record_rows_compose_header_only_list_deletes_every_source_row(self, repository):
        entry = f"{PREFIX}hdr-1"
        a = "https://h.example/a.png"
        await _seed_entry(repository, entry, [{"url": a}])
        cfg = _config("images")
        await _compose(repository, entry, cfg, _http(cfg))
        assert len(await _rows(repository, entry)) == 1

        await _set_attachments(repository, entry, [{"url": ""}])
        assert await _compose(repository, entry, cfg, _http(cfg)) == []
        assert await _rows(repository, entry) == {}

    async def test_record_rows_compose_empty_list_deletes_nothing(self, repository):
        entry = f"{PREFIX}empty-1"
        a = "https://h.example/a.png"
        await _seed_entry(repository, entry, [{"url": a}])
        cfg = _config("images")
        await _compose(repository, entry, cfg, _http(cfg))

        await _set_attachments(repository, entry, [])
        assert await _compose(repository, entry, cfg, _http(cfg)) == []
        assert set(await _rows(repository, entry)) == {_id(entry, a)}

    async def test_record_rows_compose_writes_text_and_clears_text_keys_only_on_change(
        self, repository
    ):
        entry = f"{PREFIX}text-1"
        a = "https://h.example/a.png"
        await _seed_entry(
            repository,
            entry,
            [{"url": a, "filename": "a.png", "caption": "upstream words"}],
            status={"text_embedding": {"status": "complete"}, "qmd_export": {"status": "x"}},
        )
        cfg = _config("images")
        await _compose(repository, entry, cfg, _http(cfg))

        stored = await _entry(repository, entry)
        assert stored["text"] == "[picture a.png - upstream caption] upstream words"
        assert "text_embedding" not in stored["status"]
        assert "qmd_export" not in stored["status"]

        # Unchanged text: the text keys a later run wrote stay.
        async with repository.pool.connection() as conn:
            await conn.execute(
                "UPDATE enhanced_entries SET enhancement_status = %(s)s::jsonb "
                "WHERE entry_id = %(e)s",
                {"s": json.dumps(STATUS_WITH_KEYS), "e": entry},
            )
        await _compose(repository, entry, cfg, _http(cfg))
        stored = await _entry(repository, entry)
        assert "text_embedding" in stored["status"]
        assert "qmd_export" in stored["status"]

    async def test_record_rows_compose_null_captions_stay_null(self, repository):
        entry = f"{PREFIX}nullcap-1"
        await _seed_entry(repository, entry, [{"url": "https://h.example/a.png"}])
        cfg = _config("images")
        await _compose(repository, entry, cfg, _http(cfg))
        stored = await _entry(repository, entry)
        assert stored["captions"] is None
        assert stored["text"] is None


# --- copy_entry ------------------------------------------------------------------

ORIGINS = frozenset({("https", "h.example", 443)})
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
HTML = b"<!DOCTYPE html><html><head><title>Login</title></head><body>sign in</body></html>"


def _fake_prepare(monkeypatch) -> None:
    """Render every sniffed picture instantly; non-images answer from the sniff."""
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


def _copy_run(cfg: ARIELConfig, **overrides) -> copy_mod.CopyRun:
    return copy_mod.CopyRun(_http(cfg), ORIGINS, **overrides)


async def _pending(repository, entry_id: str, names: list[str], cfg: ARIELConfig) -> list[str]:
    """Seed an entry with one declared-PNG attachment per name and record its rows."""
    urls = [f"https://h.example/files/{name}" for name in names]
    await _seed_entry(repository, entry_id, [{"url": u, "type": "image/png"} for u in urls])
    await _record(repository, entry_id, cfg, _http(cfg))
    return [_id(entry_id, u) for u in urls]


async def _state(repository, entry_id: str) -> dict[str, dict]:
    async with repository.pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT attachment_id, copy_status, skip_reason, copy_attempts, mime_type,
                   data, rendition_sha256
            FROM attachment_files WHERE entry_id = %(e)s
            """,
            {"e": entry_id},
        )
        cols = [
            "attachment_id",
            "copy_status",
            "skip_reason",
            "copy_attempts",
            "mime_type",
            "data",
            "rendition_sha256",
        ]
        return {r[0]: dict(zip(cols, r, strict=True)) for r in await result.fetchall()}


def _sleepy_server(seconds: float):
    async def _fetch(*_args, **kwargs):
        kwargs["on_sent"]()
        await asyncio.sleep(seconds)
        return FetchOutcome(data=PNG_MAGIC)

    return _fetch


async def _insert_native(repository, entry_id: str, attachment_id: str, data: bytes, mime: str):
    """A B1-era native row: source_url NULL, copied, original stored, no rendition."""
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, data, size_bytes
            ) VALUES (%(a)s, %(e)s, 'native', %(m)s, %(d)s, %(s)s)
            """,
            {"a": attachment_id, "e": entry_id, "m": mime, "d": data, "s": len(data)},
        )


@pytest.mark.timeout(60)
class TestCopyEntry:
    @pytest.fixture(autouse=True)
    def _no_fetch(self):
        """copy_entry tests answer fetches through the conftest fake instead."""

    async def test_copy_entry_ten_poll_outage_then_serving_copies_on_poll_eleven(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        entry, cfg = f"{PREFIX}outage-1", _config("images")
        ids = await _pending(repository, entry, [f"{i}.png" for i in range(8)], cfg)
        attachment_fetch.respond(FetchOutcome(transient=True, host_up=False))
        for _poll in range(10):
            before = len(attachment_fetch.calls)
            await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg))
            assert len(attachment_fetch.calls) - before <= 5
            state = await _state(repository, entry)
            assert all(state[i]["copy_status"] == "pending" for i in ids)
            assert all(state[i]["copy_attempts"] == 0 for i in ids)

        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg))
        state = await _state(repository, entry)
        assert all(state[i]["copy_status"] == "copied" for i in ids)
        assert all(bytes(state[i]["data"]) == PNG_MAGIC for i in ids)

    async def test_copy_entry_refusing_host_ends_eight_day_row_only(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        entry, cfg = f"{PREFIX}age-1", _config("images")
        old, fresh = await _pending(repository, entry, ["old.png", "new.png"], cfg)
        async with repository.pool.connection() as conn:
            await conn.execute(
                "UPDATE attachment_files SET created_at = now() - interval '8 days' "
                "WHERE attachment_id = %(a)s",
                {"a": old},
            )
        attachment_fetch.respond(FetchOutcome(transient=True, host_up=False))
        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg))
        state = await _state(repository, entry)
        assert (state[old]["copy_status"], state[old]["skip_reason"]) == ("skipped", "fetch_failed")
        assert (state[fresh]["copy_status"], state[fresh]["copy_attempts"]) == ("pending", 0)

    async def test_copy_entry_twenty_slow_pictures_all_copied_without_fetch_failed(
        self, repository, attachment_fetch, monkeypatch
    ):
        # Scaled: a 60 s deadline over 20 s pictures becomes 0.6 s over 0.2 s.
        _fake_prepare(monkeypatch)
        entry, cfg = f"{PREFIX}slow-1", _config("images")
        ids = await _pending(repository, entry, [f"{i}.png" for i in range(20)], cfg)
        attachment_fetch.respond(_sleepy_server(0.2))
        charged = 0
        for _poll in range(12):
            await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg, entry_deadline_s=0.6))
            state = await _state(repository, entry)
            total = sum(s["copy_attempts"] for s in state.values())
            assert total - charged <= 4
            charged = total
            assert not any(s["skip_reason"] == "fetch_failed" for s in state.values())
            if all(s["copy_status"] == "copied" for s in state.values()):
                break
        state = await _state(repository, entry)
        assert all(state[i]["copy_status"] == "copied" for i in ids)

    async def test_copy_entry_twenty_five_pictures_over_three_polls_copy_twenty(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        entry, cfg = f"{PREFIX}budget-1", _config("images")
        ids = await _pending(repository, entry, [f"{i}.png" for i in range(25)], cfg)
        attachment_fetch.respond(_sleepy_server(0.2))
        for _poll in range(3):
            await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg, entry_deadline_s=0.6))
        state = await _state(repository, entry)
        assert [state[i]["copy_status"] for i in ids[:20]] == ["copied"] * 20
        assert [state[i]["skip_reason"] for i in ids[20:]] == ["per_entry_limit"] * 5

    @pytest.mark.parametrize("mode", ["images", "all"])
    async def test_copy_entry_declared_png_html_body_is_source_refused_then_backfilled(
        self, repository, attachment_fetch, monkeypatch, mode
    ):
        _fake_prepare(monkeypatch)
        entry, cfg = f"{PREFIX}login-{mode}", _config(mode)
        (aid,) = await _pending(repository, entry, ["shot.png"], cfg)
        attachment_fetch.respond(FetchOutcome(data=HTML))
        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg))
        row = (await _state(repository, entry))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("skipped", "source_refused")
        assert row["mime_type"] == "text/html"
        assert row["data"] is None

        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg), retry_skipped=True)
        row = (await _state(repository, entry))[aid]
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert bytes(row["data"]) == PNG_MAGIC
        assert row["rendition_sha256"] is not None


@pytest.mark.timeout(60)
class TestCopyEntryNativeRows:
    """B1-era native rows (``source_url`` NULL) are rendered from their stored bytes."""

    @pytest.fixture(autouse=True)
    def _no_fetch(self):
        """The fetch fake fails the test on any call; none is expected here."""

    @pytest.fixture
    async def no_worker_left(self):
        from osprey.imaging import render

        await render.close_render_worker()
        yield
        await render.close_render_worker()

    @pytest.mark.usefixtures("no_worker_left")
    @pytest.mark.parametrize("mode", ["images", "none"])
    async def test_copy_entry_renders_b1_native_png_with_zero_fetches(
        self, repository, attachment_fetch, mode
    ):
        import io

        image = pytest.importorskip("PIL.Image")
        out = io.BytesIO()
        image.new("RGB", (16, 12), (10, 120, 200)).save(out, "PNG")
        png = out.getvalue()

        entry, cfg = f"{PREFIX}native-png-{mode}", _config(mode)
        aid = f"{PREFIX}nat-png-{mode}"
        await _seed_entry(repository, entry, [])
        await _insert_native(repository, entry, aid, png, "image/png")

        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg))
        row = (await _state(repository, entry))[aid]
        assert attachment_fetch.calls == []
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert row["rendition_sha256"] is not None
        assert bytes(row["data"]) == png
        assert not [r for r in await repository.get_copy_rows(entry) if _render_only(r)]

    async def test_copy_entry_b1_native_pdf_is_reserved_and_kept(
        self, repository, attachment_fetch, monkeypatch
    ):
        from osprey.imaging import render

        async def _no_render(*_args, **_kwargs):
            raise AssertionError("a PDF never reaches the render worker")

        monkeypatch.setattr(render, "render_isolated", _no_render)
        entry, cfg = f"{PREFIX}native-pdf", _config("images")
        aid = f"{PREFIX}nat-pdf"
        await _seed_entry(repository, entry, [])
        await _insert_native(repository, entry, aid, PDF, "image/png")  # client-declared

        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg))
        row = (await _state(repository, entry))[aid]
        assert attachment_fetch.calls == []
        assert (row["copy_status"], row["skip_reason"]) == ("copied", "reserved_format")
        assert row["mime_type"] == "application/pdf"
        assert bytes(row["data"]) == PDF
        assert not [r for r in await repository.get_copy_rows(entry) if _render_only(r)]
        assert await repository.get_copy_source(aid) is None


def _render_only(row: dict) -> bool:
    return (
        row["copy_status"] == "copied"
        and row["rendition_sha256"] is None
        and row["skip_reason"] is None
        and row["has_data"]
    )


# --- the poll's copy retry step --------------------------------------------------

#: Entries the retry tests seed sit after every other row of the shared database,
#: so the step's newest-first walk reaches them first.
FUTURE = datetime(2200, 1, 1, tzinfo=UTC)
#: Every attachment the retry tests seed is served from here; others refuse.
SERVING = "https://h.example/files/"
REFUSING = "https://down.example/files/"


class _RetryAdapter(_HttpAdapter):
    """An http source whose attachments live on the two retry-test hosts; no new entries."""

    def attachment_origins(self) -> frozenset:
        return frozenset({("https", "h.example", 443), ("https", "down.example", 443)})


async def _retry_entry(
    repository, entry_id: str, names: list[str], cfg: ARIELConfig, *, at: datetime, base=SERVING
) -> list[str]:
    """Seed an entry stamped ``at`` with one declared-PNG attachment per name; record rows."""
    urls = [f"{base}{name}" for name in names]
    await _seed_entry(repository, entry_id, [{"url": u, "type": "image/png"} for u in urls])
    async with repository.pool.connection() as conn:
        await conn.execute(
            "UPDATE enhanced_entries SET timestamp = %(t)s WHERE entry_id = %(e)s",
            {"t": at, "e": entry_id},
        )
    await _record(repository, entry_id, cfg, _http(cfg))
    return [_id(entry_id, u) for u in urls]


def _retry_scheduler(repository, monkeypatch, cfg: ARIELConfig, scope: str):
    """A scheduler whose polls fetch no entries, so each poll is only its retry step.

    The run ledger is stubbed so the shared database gains no ingestion runs, and
    the candidate walk is narrowed to the entries whose id starts with *scope*:
    the shared database carries other tests' entries, which the step must not
    copy here.
    """
    from unittest.mock import AsyncMock

    from osprey.services.ariel_search import enhancement
    from osprey.services.ariel_search import ingestion as ingestion_pkg
    from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

    adapter = _RetryAdapter(cfg)
    monkeypatch.setattr(ingestion_pkg, "get_adapter", lambda _config: adapter)
    monkeypatch.setattr(enhancement, "create_enhancers_from_config", lambda _config: [])
    monkeypatch.setattr(repository, "get_last_successful_run", AsyncMock(return_value=None))
    monkeypatch.setattr(repository, "start_ingestion_run", AsyncMock(return_value=1))
    monkeypatch.setattr(repository, "complete_ingestion_run", AsyncMock())
    monkeypatch.setattr(repository, "fail_ingestion_run", AsyncMock())

    real_walk = repository.get_copy_retry_candidates

    async def _own_rows(after, limit):
        return [p for p in await real_walk(after, limit) if p[1].startswith(scope)]

    monkeypatch.setattr(repository, "get_copy_retry_candidates", _own_rows)
    return IngestionScheduler(cfg, repository)


def _retry_config() -> ARIELConfig:
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "attachments": {"copy_on_ingest": "images"},
            "ingestion": {
                "adapter": "generic_json",
                "source_url": "https://h.example/api",
                "watch": {"require_initial_ingest": False},
            },
        }
    )


def _by_host(url, *_args, **_kwargs):
    if url.startswith(SERVING):
        return FetchOutcome(data=PNG_MAGIC)
    return FetchOutcome(transient=True, host_up=False)


@pytest.mark.timeout(60)
class TestPollCopyRetryStep:
    @pytest.fixture(autouse=True)
    def _no_fetch(self):
        """Retry-step tests answer fetches through the conftest fake instead."""

    async def test_retry_step_copies_a_timed_out_picture_on_the_next_poll(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        cfg = _retry_config()
        entry = scope = f"{PREFIX}retry-timeout"
        (aid,) = await _retry_entry(repository, entry, ["late.png"], cfg, at=FUTURE)

        # Ingest-time copy: the fetch runs past the entry deadline, the row stays pending.
        attachment_fetch.respond(_sleepy_server(0.5))
        await copy_mod.copy_entry(repository, entry, cfg, _copy_run(cfg, entry_deadline_s=0.05))
        assert (await _state(repository, entry))[aid]["copy_status"] == "pending"

        attachment_fetch.respond(_by_host)
        scheduler = _retry_scheduler(repository, monkeypatch, cfg, scope)
        await scheduler.poll_once()

        row = (await _state(repository, entry))[aid]
        assert row["copy_status"] == "copied"
        assert bytes(row["data"]) == PNG_MAGIC
        assert row["rendition_sha256"] is not None

    async def test_retry_step_renders_ten_pictures_copied_while_the_worker_was_broken(
        self, repository, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search.attachments import prepare as prepare_mod

        async def _broken(*_args, **_kwargs):
            raise prepare_mod.RenderUnavailable("worker cannot start", exit_code=1)

        monkeypatch.setattr(prepare_mod, "prepare_picture", _broken)
        cfg = _retry_config()
        scope = f"{PREFIX}retry-render-"
        entries = [f"{scope}{i}" for i in range(2)]
        ids = []
        for i, entry in enumerate(entries):
            ids += await _retry_entry(
                repository,
                entry,
                [f"r{i}-{n}.png" for n in range(5)],
                cfg,
                at=FUTURE - timedelta(minutes=i),
            )
        attachment_fetch.respond(_by_host)
        scheduler = _retry_scheduler(repository, monkeypatch, cfg, scope)
        await scheduler.poll_once()

        state = {k: v for e in entries for k, v in (await _state(repository, e)).items()}
        assert len(ids) == 10
        assert all(state[i]["copy_status"] == "copied" for i in ids)
        assert all(state[i]["rendition_sha256"] is None for i in ids)

        _fake_prepare(monkeypatch)
        fetches = len(attachment_fetch.calls)
        await scheduler.poll_once()

        state = {k: v for e in entries for k, v in (await _state(repository, e)).items()}
        assert all(state[i]["rendition_sha256"] is not None for i in ids)
        assert len(attachment_fetch.calls) == fetches

    async def test_retry_step_skipped_while_a_backfill_holds_the_copy_lock(
        self, repository, attachment_fetch, monkeypatch, database_url
    ):
        import psycopg

        _fake_prepare(monkeypatch)
        cfg = _retry_config()
        entry = scope = f"{PREFIX}retry-locked"
        (aid,) = await _retry_entry(repository, entry, ["held.png"], cfg, at=FUTURE)
        attachment_fetch.respond(_by_host)
        scheduler = _retry_scheduler(repository, monkeypatch, cfg, scope)

        async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as other:
            await other.execute(
                "SELECT pg_advisory_lock(hashtextextended(%(k)s, 0))", {"k": "ariel_copy"}
            )
            result = await scheduler.poll_once()

        assert result.entries_added == 0
        assert attachment_fetch.calls == []
        assert (await _state(repository, entry))[aid]["copy_status"] == "pending"

        # Released: the next poll copies it.
        await scheduler.poll_once()
        assert (await _state(repository, entry))[aid]["copy_status"] == "copied"

    async def test_retry_step_older_entry_copied_within_two_polls_behind_twenty_stuck(
        self, repository, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        cfg = _retry_config()
        scope = f"{PREFIX}retry-starve-"
        stuck = [f"{scope}stuck-{i:02d}" for i in range(20)]
        for i, entry in enumerate(stuck):
            await _retry_entry(
                repository,
                entry,
                [f"s{i}.png"],
                cfg,
                at=FUTURE + timedelta(minutes=i + 1),
                base=REFUSING,
            )
        older = f"{scope}older"
        (aid,) = await _retry_entry(repository, older, ["o.png"], cfg, at=FUTURE)
        attachment_fetch.respond(_by_host)
        scheduler = _retry_scheduler(repository, monkeypatch, cfg, scope)

        copied_at = None
        for poll in (1, 2):
            await scheduler.poll_once()
            if (await _state(repository, older))[aid]["copy_status"] == "copied":
                copied_at = poll
                break
        assert copied_at is not None and copied_at <= 2
        for entry in stuck:
            rows = (await _state(repository, entry)).values()
            assert all(r["copy_status"] == "pending" for r in rows)

    async def test_retry_step_b1_schema_poll_completes_and_skips_the_step(
        self, repository, attachment_fetch, monkeypatch
    ):
        from unittest.mock import AsyncMock

        from osprey.services.ariel_search.database.repository import SchemaFacts

        cfg = _retry_config()
        entry = scope = f"{PREFIX}retry-b1"
        (aid,) = await _retry_entry(repository, entry, ["b1.png"], cfg, at=FUTURE)
        attachment_fetch.respond(_by_host)
        scheduler = _retry_scheduler(repository, monkeypatch, cfg, scope)
        monkeypatch.setattr(
            repository,
            "schema_facts",
            AsyncMock(return_value=SchemaFacts(has_v2_fts=False, has_copy_state=False)),
        )

        result = await scheduler.poll_once()

        assert result.entries_failed == 0
        repository.complete_ingestion_run.assert_awaited_once()
        assert attachment_fetch.calls == []
        assert (await _state(repository, entry))[aid]["copy_status"] == "pending"


# --- backfill ----------------------------------------------------------------------
#
# Backfill walks the whole store, so these tests run on a fresh scratch database
# each: the shared database carries every other module's entries.

BACKFILL_BASE = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
async def scratch_repo(scratch_database):
    """An ``ARIELRepository`` on a fresh scratch database at today's schema."""
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    cfg = ARIELConfig.from_dict({"database": {"uri": scratch_database}})
    pool = await create_connection_pool(cfg.database)
    try:
        await run_migrations(pool, cfg)
        yield ARIELRepository(pool, cfg)
    finally:
        await pool.close()


async def _seed_at(repository, entry_id: str, attachments: list, *, minutes: int) -> None:
    async with repository.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text,
                attachments, metadata, enhancement_status
            ) VALUES (
                %(e)s, 'test', %(ts)s, 'tester', 'x', %(a)s::jsonb, '{}'::jsonb, '{}'::jsonb
            )
            """,
            {
                "e": entry_id,
                "ts": BACKFILL_BASE + timedelta(minutes=minutes),
                "a": json.dumps(attachments),
            },
        )


def _png_item(name: str) -> dict:
    return {"url": f"https://h.example/files/{name}", "type": "image/png"}


def _ingest_entry(entry_id: str, attachments: list, *, minutes: int) -> dict:
    return {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": BACKFILL_BASE + timedelta(minutes=minutes),
        "author": "tester",
        "raw_text": "x",
        "attachments": attachments,
        "metadata": {},
        "enhancement_status": {},
    }


class _OriginHttpAdapter(_HttpAdapter):
    """An http source whose attachments live on ``h.example``."""

    def attachment_origins(self):
        return ORIGINS


async def _backfill(repository, cfg: ARIELConfig, **kwargs):
    from osprey.services.ariel_search.cli_operations import backfill_store

    return await backfill_store(repository, _OriginHttpAdapter(cfg), cfg, **kwargs)


@pytest.mark.timeout(60)
class TestBackfill:
    @pytest.fixture(autouse=True)
    def _no_fetch(self):
        """Backfill tests answer fetches through the conftest fake instead."""

    @pytest.mark.timeout(10)
    async def test_backfill_page_with_one_pending_picture_completes_quickly(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg = _config("images")
        await _seed_at(scratch_repo, "bf-one", [_png_item("one.png")], minutes=0)

        result = await _backfill(scratch_repo, cfg)

        assert result.status == "done"
        assert (result.entries, result.fetches, result.copied) == (1, 1, 1)
        (row,) = (await _state(scratch_repo, "bf-one")).values()
        assert row["copy_status"] == "copied"

    async def test_backfill_interleaved_reingest_never_resurrects_a_removed_url(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        """``ingest_one([a,c])`` lands between the page commit and the copy of ``[a,b]``."""
        from osprey.services.ariel_search.ingestion.ingest import ingest_one

        _fake_prepare(monkeypatch)
        cfg = _config("images")
        adapter = _OriginHttpAdapter(cfg)
        a, b, c = _png_item("a.png"), _png_item("b.png"), _png_item("c.png")
        await _seed_at(scratch_repo, "bf-older", [a, b], minutes=0)
        await _seed_at(scratch_repo, "bf-newer", [_png_item("n.png")], minutes=10)
        reingested: list[bool] = []

        async def _serve(url, *_args, **_kwargs):
            if url.endswith("/n.png") and not reingested:
                # The newest entry is copied first; the older one's page is
                # already committed with rows {a, b}.
                assert set(await _state(scratch_repo, "bf-older")) == {
                    _id("bf-older", a["url"]),
                    _id("bf-older", b["url"]),
                }
                await ingest_one(
                    _ingest_entry("bf-older", [a, c], minutes=0),
                    adapter,
                    scratch_repo,
                    [],
                    cfg,
                    None,
                )
                reingested.append(True)
            return FetchOutcome(data=PNG_MAGIC)

        attachment_fetch.respond(_serve)

        result = await _backfill(scratch_repo, cfg)

        assert reingested == [True]
        assert result.record_failed == result.copy_failed == 0
        rows = await _state(scratch_repo, "bf-older")
        assert set(rows) == {_id("bf-older", a["url"]), _id("bf-older", c["url"])}
        assert all(r["copy_status"] == "copied" for r in rows.values())
        assert not [call for call in attachment_fetch.calls if call["url"] == b["url"]]

    async def test_backfill_after_reingest_before_the_page_drops_the_removed_url(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search.ingestion.ingest import ingest_one

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg = _config("images")
        adapter = _OriginHttpAdapter(cfg)
        a, b, c = _png_item("a.png"), _png_item("b.png"), _png_item("c.png")
        await _seed_at(scratch_repo, "bf-pre", [a, b], minutes=0)
        await _compose(scratch_repo, "bf-pre", cfg, adapter)  # rows {a, b} pending
        await ingest_one(
            _ingest_entry("bf-pre", [a, c], minutes=0), adapter, scratch_repo, [], cfg, None
        )

        await _backfill(scratch_repo, cfg)

        rows = await _state(scratch_repo, "bf-pre")
        assert set(rows) == {_id("bf-pre", a["url"]), _id("bf-pre", c["url"])}
        assert not [call for call in attachment_fetch.calls if call["url"] == b["url"]]

    @pytest.mark.parametrize("wait", [True, False])
    async def test_backfill_two_concurrent_runs_fetch_each_picture_once(
        self, scratch_repo, attachment_fetch, monkeypatch, wait
    ):
        from collections import Counter

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(_sleepy_server(0.05))
        cfg = _config("images")
        for i in range(3):
            await _seed_at(
                scratch_repo,
                f"bf-conc-{i}",
                [_png_item(f"{i}-{j}.png") for j in range(2)],
                minutes=i,
            )

        first, second = await asyncio.gather(
            _backfill(scratch_repo, cfg, wait=wait), _backfill(scratch_repo, cfg, wait=wait)
        )

        counts = Counter(call["url"] for call in attachment_fetch.calls)
        assert len(counts) == 6
        assert set(counts.values()) == {1}
        statuses = sorted([first.status, second.status])
        assert statuses == (["done", "done"] if wait else ["done", "locked"])
        for i in range(3):
            rows = (await _state(scratch_repo, f"bf-conc-{i}")).values()
            assert all(r["copy_status"] == "copied" for r in rows)

    async def test_backfill_and_ingest_one_looped_on_one_entry_never_deadlock(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search.ingestion.ingest import ingest_one

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        cfg = _config("images")
        adapter = _OriginHttpAdapter(cfg)
        lists = [
            [_png_item("a.png"), _png_item("b.png")],
            [_png_item("a.png"), _png_item("c.png")],
        ]
        await _seed_at(scratch_repo, "bf-loop", lists[0], minutes=0)

        async def _ingest_loop() -> None:
            async with copy_mod.CopyRun(adapter, ORIGINS) as run:
                for i in range(15):
                    outcome = await ingest_one(
                        _ingest_entry("bf-loop", lists[i % 2], minutes=0),
                        adapter,
                        scratch_repo,
                        [],
                        cfg,
                        run,
                    )
                    assert outcome.attachments_recorded

        async def _backfill_loop() -> list:
            return [await _backfill(scratch_repo, cfg, wait=True) for _ in range(15)]

        _, results = await asyncio.gather(_ingest_loop(), _backfill_loop())

        assert all(r.record_failed == 0 and r.copy_failed == 0 for r in results)
        urls = {r["attachment_id"] for r in (await _state(scratch_repo, "bf-loop")).values()}
        assert urls == {_id("bf-loop", lists[0][0]["url"]), _id("bf-loop", lists[0][1]["url"])}

    async def test_backfill_legacy_unfetchable_urls_create_no_pending_row(
        self, scratch_repo, scratch_database, attachment_fetch
    ):
        """``/rel/x.png`` on a generic http source and an eLog-adapter url with ``..`` get no row."""
        from osprey.services.ariel_search.cli_operations import backfill_store
        from osprey.services.ariel_search.ingestion import get_adapter

        als_prefix = get_adapter(
            ARIELConfig.from_dict(
                {
                    "database": {"uri": scratch_database},
                    "ingestion": {"adapter": "als_logbook", "source_url": "/fake/path.jsonl"},
                }
            )
        ).attachment_url_prefix

        await _seed_at(scratch_repo, "bf-rel", [{"url": "/rel/x.png"}], minutes=0)
        await _seed_at(
            scratch_repo,
            "bf-als",
            [{"url": f"{als_prefix}attachments/../secret.png"}],
            minutes=1,
        )

        for adapter_name, source in (
            ("generic_json", "https://h.example/logbook.json"),
            ("als_logbook", f"{als_prefix}logbook.json"),
        ):
            cfg = ARIELConfig.from_dict(
                {
                    "database": {"uri": scratch_database},
                    "ingestion": {"adapter": adapter_name, "source_url": source},
                    "attachments": {"copy_on_ingest": "images"},
                }
            )
            result = await backfill_store(scratch_repo, get_adapter(cfg), cfg)
            assert result.status == "done"
            assert result.fetches == 0

        assert await _state(scratch_repo, "bf-rel") == {}
        assert await _state(scratch_repo, "bf-als") == {}
        assert attachment_fetch.calls == []

    async def test_backfill_on_b1_store_after_migrate_composes_upstream_caption(
        self, scratch_database, attachment_fetch, monkeypatch
    ):
        """A B1-era entry with an upstream caption: migrate, then backfill."""
        import psycopg

        from osprey.services.ariel_search import cli_operations as ops
        from osprey.services.ariel_search.database.attachment_migration import (
            AttachmentFilesCopyStateMigration,
        )
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextColumnsMigration,
        )
        from osprey.services.ariel_search.database.connection import create_connection_pool

        _fake_prepare(monkeypatch)
        attachment_fetch.respond(FetchOutcome(data=PNG_MAGIC))
        config_dict = {
            "database": {"uri": scratch_database},
            "ingestion": {"adapter": "generic_json", "source_url": "https://h.example/l.json"},
            "attachments": {"copy_on_ingest": "images"},
        }
        assert await ops.run_migrate(dict(config_dict)) == []

        cfg = ARIELConfig.from_dict(config_dict)
        pool = await create_connection_pool(cfg.database)
        try:
            for migration in (
                AttachmentFilesCopyStateMigration(),
                AttachmentTextColumnsMigration(),
            ):
                async with pool.connection() as conn, conn.transaction():
                    await migration.down(conn)
                    await conn.execute(
                        "DELETE FROM ariel_migrations WHERE name = %s", (migration.name,)
                    )
        finally:
            await pool.close()

        item = {
            "url": "https://h.example/files/spot.png",
            "type": "image/png",
            "filename": "spot.png",
            "caption": "Beam spot on screen 3",
        }
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text,
                    attachments, metadata, enhancement_status
                ) VALUES ('bf-b1', 'test', %s, 'tester', 'x', %s::jsonb, '{}'::jsonb, %s::jsonb)
                """,
                (
                    BACKFILL_BASE,
                    json.dumps([item]),
                    json.dumps({"text_embedding": {"status": "complete"}}),
                ),
            )

        refused = await ops.run_backfill(dict(config_dict))
        assert refused.status == "no_copy_state"

        assert await ops.run_migrate(dict(config_dict)) == []
        result = await ops.run_backfill(dict(config_dict))

        assert result.status == "done"
        assert result.copied == 1
        with psycopg.connect(scratch_database) as conn:
            row = conn.execute(
                "SELECT attachment_text, enhancement_status FROM enhanced_entries "
                "WHERE entry_id = 'bf-b1'"
            ).fetchone()
        assert row is not None
        text, status = row
        assert "[picture spot.png - upstream caption] Beam spot on screen 3" in text
        assert "text_embedding" not in status


# --- backfill --retry-decoder-failed ---------------------------------------------------
#
# A ``decoder_failed`` row keeps its stored original and has no rendition. The flag
# clears the reason (and any rendition column) in one transaction per entry; the
# backfill's render-only pass then renders the original again, with no fetch.

RETRY_CAPTION_MODEL = "vis-caption-1"
RETRY_REPLY = "Beam spot on screen 3.\nVisible text: SCR-3"


def _retry_config_dict(uri: str, embed_url: str) -> dict:
    """Both picture modules on *uri*: llama-cpp embeddings and an openai caption model."""
    from tests.services.ariel_search.llama_stub import MODEL as LLAMA_MODEL

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
            "image_embedding": {
                "enabled": True,
                "provider": {"name": "llama-cpp", "base_url": embed_url},
                "model": LLAMA_MODEL,
                "dimensions": 1024,
            },
            "image_caption": {
                "enabled": True,
                "provider": "openai",
                "model": {"model_id": RETRY_CAPTION_MODEL},
                "timeout_seconds": 60,
            },
        },
    }


async def _seed_content_skip(repository, entry_id: str, name: str, code: str, cfg) -> str:
    """Store one copied PNG original that the content check set aside as *code*."""
    url = f"https://h.example/files/{name}"
    await _seed_entry(repository, entry_id, [{"url": url, "type": "image/png", "filename": name}])
    await _record(repository, entry_id, cfg, _http(cfg))
    aid = _id(entry_id, url)
    written = await repository.apply_copy_outcome(
        entry_id,
        aid,
        copy_status="copied",
        data=PNG_MAGIC,
        mime_type="image/png",
        size_bytes=len(PNG_MAGIC),
        skip_reason=code,
    )
    assert written == "copied"
    return aid


def _invoke_backfill_cli(config_dict: dict, args: list[str]):
    from unittest.mock import patch

    from click.testing import CliRunner

    from osprey.cli.ariel import ariel_group

    def _get(key, default=None, *_a, **_k):
        return config_dict if key == "ariel" else default

    with patch("osprey.cli.ariel.get_config_value", side_effect=_get):
        return CliRunner().invoke(
            ariel_group, ["attachments", "backfill", *args], catch_exceptions=False
        )


@pytest.mark.timeout(180)
class TestBackfillRetryDecoderFailed:
    @pytest.fixture(autouse=True)
    def _seed_prefix(self):
        """Every case seeds a scratch database only; the shared one is never written."""

    @pytest.fixture(autouse=True)
    def _no_fetch(self):
        """Fetches answer through the conftest fake, which records every call."""

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch):
        """Fresh module availability and offload state; a configured caption provider."""
        from osprey.models.providers.health import HealthResult
        from osprey.services.ariel_search.enhancement import _offload, availability
        from osprey.services.ariel_search.enhancement.image_caption import module as caption_mod

        availability.reset_availability()
        _offload.reset_offload_state()
        monkeypatch.setattr(
            "osprey.models.config.get_provider_config", lambda name: {"api_key": "k"}
        )
        monkeypatch.setattr(
            caption_mod,
            "probe_models_endpoint",
            lambda *a, **k: HealthResult(True, "served", None),
        )
        monkeypatch.setattr(caption_mod, "_chat_completion", lambda **_k: RETRY_REPLY)
        yield
        availability.reset_availability()
        _offload.reset_offload_state()

    @pytest.fixture
    def stub(self, llama_stub):
        return llama_stub()

    @pytest.fixture
    async def image_repo(self, scratch_database, stub):
        """A repository on a fresh scratch database with the image table migrated."""
        from osprey.services.ariel_search.database import ARIELRepository
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.migrations import run_migrations

        cfg = ARIELConfig.from_dict(_retry_config_dict(scratch_database, stub.url))
        pool = await create_connection_pool(cfg.database)
        try:
            await run_migrations(pool, cfg)
            yield ARIELRepository(pool, cfg)
        finally:
            await pool.close()

    async def test_retry_decoder_failed_flag_rerenders_and_the_picture_gets_caption_and_vector(
        self, image_repo, scratch_database, stub, attachment_fetch, monkeypatch
    ):
        import psycopg

        from osprey.services.ariel_search import cli_operations as ops
        from osprey.services.ariel_search.database.migrations import image_table_name
        from tests.services.ariel_search.llama_stub import MODEL as LLAMA_MODEL

        _fake_prepare(monkeypatch)
        config_dict = _retry_config_dict(scratch_database, stub.url)
        cfg = ARIELConfig.from_dict(config_dict)
        aid = await _seed_content_skip(image_repo, "rdf-1", "spot.png", "decoder_failed", cfg)

        # Without the flag the content skip is terminal: nothing is rendered.
        plain = await _backfill(image_repo, cfg)
        assert (plain.rendered, plain.decoder_reset) == (0, 0)
        assert (await _state(image_repo, "rdf-1"))[aid]["skip_reason"] == "decoder_failed"

        result = await asyncio.to_thread(
            _invoke_backfill_cli, config_dict, ["--retry-decoder-failed"]
        )

        assert result.exit_code == 0, result.output
        assert "backfill --retry-decoder-failed" in result.output
        assert "1 rendered" in result.output
        assert "Decoder failures reset for a re-render: 1" in result.output
        row = (await _state(image_repo, "rdf-1"))[aid]
        assert row["copy_status"] == "copied"
        assert row["skip_reason"] is None
        assert row["rendition_sha256"] == "cd" * 32
        assert row["data"] == PNG_MAGIC

        await ops.run_enhance(config_dict, module=None, force=False, limit=100)

        with psycopg.connect(scratch_database) as conn:
            caption_row = conn.execute(
                "SELECT attachment_captions FROM enhanced_entries WHERE entry_id = 'rdf-1'"
            ).fetchone()
            vectors = dict(
                conn.execute(
                    f"SELECT attachment_id, embedding IS NOT NULL "
                    f"FROM {image_table_name(LLAMA_MODEL, 1024)}"
                ).fetchall()
            )
        assert caption_row is not None
        assert caption_row[0][aid][RETRY_CAPTION_MODEL]["caption"] == "Beam spot on screen 3."
        assert vectors == {aid: True}
        assert attachment_fetch.calls == []

    async def test_retry_decoder_failed_resets_only_decoder_failed_and_dry_run_writes_nothing(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        _fake_prepare(monkeypatch)
        cfg = _config("images")
        broken = await _seed_content_skip(scratch_repo, "rdf-a", "a.png", "decoder_failed", cfg)
        other = await _seed_content_skip(scratch_repo, "rdf-b", "b.png", "format_mismatch", cfg)

        dry = await _backfill(scratch_repo, cfg, dry_run=True, retry_decoder_failed=True)
        assert dry.decoder_reset == 0
        assert (await _state(scratch_repo, "rdf-a"))[broken]["skip_reason"] == "decoder_failed"

        result = await _backfill(scratch_repo, cfg, retry_decoder_failed=True)

        assert (result.decoder_reset, result.rendered, result.copy_failed) == (1, 1, 0)
        assert (await _state(scratch_repo, "rdf-a"))[broken]["skip_reason"] is None
        assert (await _state(scratch_repo, "rdf-a"))[broken]["rendition_sha256"] == "cd" * 32
        assert (await _state(scratch_repo, "rdf-b"))[other]["skip_reason"] == "format_mismatch"
        assert attachment_fetch.calls == []

    async def test_retry_decoder_failed_picture_failing_again_is_set_aside_again(
        self, scratch_repo, attachment_fetch, monkeypatch
    ):
        from osprey.services.ariel_search.attachments import prepare as prepare_mod

        async def _still_broken(_data, **_kwargs):
            return prepare_mod.PreparedPicture("image/png", "decoder_failed")

        monkeypatch.setattr(prepare_mod, "prepare_picture", _still_broken)
        cfg = _config("images")
        aid = await _seed_content_skip(scratch_repo, "rdf-c", "c.png", "decoder_failed", cfg)

        for _ in range(2):
            result = await _backfill(scratch_repo, cfg, retry_decoder_failed=True)
            assert result.decoder_reset == 1
            row = (await _state(scratch_repo, "rdf-c"))[aid]
            assert (row["skip_reason"], row["rendition_sha256"]) == ("decoder_failed", None)
            assert row["data"] == PNG_MAGIC
        assert attachment_fetch.calls == []

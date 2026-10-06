"""Tests for the pure decision helpers of the attachment copy step."""

from __future__ import annotations

import pytest

from osprey.services.ariel_search.attachments import copy as copy_mod
from osprey.services.ariel_search.attachments.copy import (
    eligible_for_mode,
    still_skipped,
    validated_declared_type,
)
from osprey.services.ariel_search.attachments.formats import CONFIG_SKIP_REASONS
from osprey.services.ariel_search.config import ARIELConfig, AttachmentsConfig

MB = 1024 * 1024
ORIGINS = frozenset({("https", "logbook.example.org", 443)})


def _config(mode: str = "images", max_file_mb: int = 10) -> ARIELConfig:
    config = ARIELConfig(database=None)  # type: ignore[arg-type]
    config.attachments = AttachmentsConfig(copy_on_ingest=mode, max_file_mb=max_file_mb)  # type: ignore[arg-type]
    return config


def _row(
    source_url: str = "https://logbook.example.org/files/a.png",
    mime_type: str | None = "image/png",
    size_bytes: int | None = None,
    skip_reason: str | None = None,
) -> dict:
    return {
        "attachment_id": "att-" + "0" * 24,
        "source_url": source_url,
        "filename": "a.png",
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "copy_status": "skipped" if skip_reason else "pending",
        "skip_reason": skip_reason,
    }


@pytest.mark.real_fetch
def test_copy_module_binds_the_fetcher_at_module_level():
    from osprey.services.ariel_search.attachments.fetch import fetch_attachment_bytes

    assert copy_mod.fetch_attachment_bytes is fetch_attachment_bytes


# --- validated_declared_type -------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("image/png", "image/png"),
        ("IMAGE/PNG", "image/png"),
        ("Image/Svg+Xml", "image/svg+xml"),
        ("application/vnd.ms-excel", "application/vnd.ms-excel"),
        ("application/octet-stream", "application/octet-stream"),
        ("x/" + "a" * 98, "x/" + "a" * 98),  # exactly 100 chars
    ],
)
def test_declared_type_accepts_valid(value, expected):
    assert validated_declared_type(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        42,
        b"image/png",
        "image",
        "image/",
        "/png",
        "image/png; charset=utf-8",
        "image/png [image not sent]",
        " image/png",
        "image/png\n",
        "-image/png",
        "image/.png",
        "image/png/extra",
        "x/" + "a" * 99,  # 101 chars
        "imagé/png",
    ],
)
def test_declared_type_rejects_invalid(value):
    assert validated_declared_type(value) is None


# --- eligible_for_mode -------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "mode", "expected"),
    [
        ("image/png", "images", True),
        ("IMAGE/JPEG", "images", True),
        (None, "images", True),
        ("", "images", True),
        ("application/octet-stream", "images", True),
        ("garbage type", "images", True),  # invalid declared type reads as missing
        ("application/pdf", "images", False),
        ("text/html", "images", False),
        ("application/pdf", "all", True),
        ("image/png", "all", True),
        (None, "all", True),
        ("image/png", "none", False),
        (None, "none", False),
        ("application/pdf", "none", False),
    ],
)
def test_declared_eligible_for_mode(declared, mode, expected):
    assert eligible_for_mode(declared, mode) is expected


# --- still_skipped -----------------------------------------------------------


@pytest.mark.parametrize(
    ("row_kwargs", "mode", "max_mb", "origins", "file_source", "expected"),
    [
        # copy_on_ingest_mode
        ({}, "none", 10, ORIGINS, False, "copy_on_ingest_mode"),
        ({"mime_type": None}, "none", 10, ORIGINS, False, "copy_on_ingest_mode"),
        ({"mime_type": "application/pdf"}, "images", 10, ORIGINS, False, "copy_on_ingest_mode"),
        ({"mime_type": "application/pdf"}, "all", 10, ORIGINS, False, None),
        ({"mime_type": "image/png"}, "images", 10, ORIGINS, False, None),
        ({"mime_type": None}, "images", 10, ORIGINS, False, None),
        ({"mime_type": "application/octet-stream"}, "images", 10, ORIGINS, False, None),
        # origin_not_allowed
        (
            {"source_url": "https://evil.example.com/a.png"},
            "images",
            10,
            ORIGINS,
            False,
            "origin_not_allowed",
        ),
        (
            {"source_url": "https://logbook.example.org:8443/a.png"},
            "all",
            10,
            ORIGINS,
            False,
            "origin_not_allowed",
        ),
        ({"source_url": "HTTPS://LOGBOOK.example.org:443/a.png"}, "all", 10, ORIGINS, False, None),
        ({}, "all", 10, frozenset(), False, "origin_not_allowed"),
        # relative path on a file source is never origin_not_allowed
        ({"source_url": "files/a.png"}, "images", 10, frozenset(), True, None),
        ({"source_url": "files/a.png"}, "images", 10, ORIGINS, True, None),
        ({"source_url": "sub/dir/a.png"}, "all", 10, frozenset(), True, None),
        # an absolute http(s) url on a file source still meets the origin rule
        (
            {"source_url": "https://evil.example.com/a.png"},
            "images",
            10,
            ORIGINS,
            True,
            "origin_not_allowed",
        ),
        # a url without an origin on an http source can never be fetched
        ({"source_url": "files/a.png"}, "images", 10, ORIGINS, False, "origin_not_allowed"),
        (
            {"source_url": "ftp://logbook.example.org/a.png"},
            "all",
            10,
            ORIGINS,
            False,
            "origin_not_allowed",
        ),
        # size_cap
        ({"size_bytes": 10 * MB + 1}, "images", 10, ORIGINS, False, "size_cap"),
        ({"size_bytes": 10 * MB}, "images", 10, ORIGINS, False, None),
        ({"size_bytes": 10 * MB + 1}, "all", 20, ORIGINS, False, None),
        ({"size_bytes": 2**31 - 1}, "all", 10, ORIGINS, False, "size_cap"),
        ({"size_bytes": None}, "all", 1, ORIGINS, False, None),
        # per_entry_limit is decided by the budget check, never here
        ({"skip_reason": "per_entry_limit"}, "images", 10, ORIGINS, False, None),
        ({"skip_reason": "per_entry_limit"}, "all", 10, ORIGINS, False, None),
    ],
)
def test_still_skipped_table(row_kwargs, mode, max_mb, origins, file_source, expected):
    row = _row(**row_kwargs)
    result = still_skipped(row, _config(mode, max_mb), origins, file_source=file_source)
    assert result == expected
    if result is not None:
        assert result in CONFIG_SKIP_REASONS


@pytest.mark.parametrize("prior", sorted(CONFIG_SKIP_REASONS))
def test_still_skipped_recorded_code_alone_never_decides(prior):
    # A row is re-evaluated from what it recorded, not from its stored code.
    row = _row(skip_reason=prior)
    assert still_skipped(row, _config("images"), ORIGINS, file_source=False) is None


@pytest.mark.parametrize("prior", sorted(CONFIG_SKIP_REASONS))
def test_still_skipped_switching_to_none_skips_every_config_code(prior):
    row = _row(skip_reason=prior)
    assert still_skipped(row, _config("none"), ORIGINS, file_source=False) == "copy_on_ingest_mode"


def test_still_skipped_sniffed_non_image_copies_after_switch_to_all():
    row = _row(mime_type="application/pdf", skip_reason="copy_on_ingest_mode")
    assert (
        still_skipped(row, _config("images"), ORIGINS, file_source=False) == "copy_on_ingest_mode"
    )
    assert still_skipped(row, _config("all"), ORIGINS, file_source=False) is None


def test_still_skipped_origin_cleared_after_allowlist_grows():
    row = _row(source_url="https://other.example.net/a.png", skip_reason="origin_not_allowed")
    grown = ORIGINS | {("https", "other.example.net", 443)}
    assert still_skipped(row, _config("images"), ORIGINS, file_source=False) == "origin_not_allowed"
    assert still_skipped(row, _config("images"), grown, file_source=False) is None


def test_still_skipped_size_cap_cleared_after_cap_raised():
    row = _row(size_bytes=10 * MB + 1, skip_reason="size_cap")
    assert still_skipped(row, _config("images", 10), ORIGINS, file_source=False) == "size_cap"
    assert still_skipped(row, _config("images", 11), ORIGINS, file_source=False) is None


def test_still_skipped_accepts_an_attachments_config_directly():
    row = _row(mime_type="application/pdf")
    attachments = AttachmentsConfig(copy_on_ingest="images")
    assert still_skipped(row, attachments, ORIGINS, file_source=False) == "copy_on_ingest_mode"


# --- copy_entry ----------------------------------------------------------------

import asyncio  # noqa: E402
import inspect  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

from osprey.imaging.formats import is_viewable  # noqa: E402
from osprey.services.ariel_search.attachments import attachment_id_for  # noqa: E402
from osprey.services.ariel_search.attachments import prepare as prepare_mod  # noqa: E402
from osprey.services.ariel_search.attachments.copy import (  # noqa: E402
    CopyRun,
    HostBreaker,
    copy_entry,
)
from osprey.services.ariel_search.attachments.fetch import FetchOutcome  # noqa: E402
from osprey.services.ariel_search.attachments.prepare import (  # noqa: E402
    PreparedPicture,
    RenderUnavailable,
)

ENTRY = "copy-unit-1"
HOST = "https://logbook.example.org"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
HTML = b"<!DOCTYPE html><html><head><title>Login</title></head><body>sign in</body></html>"
SVG = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
BINARY = bytes(range(0, 32)) * 4


class _Adapter:
    """Just enough of a FacilityAdapter for copy_entry: an http source."""

    def attachment_file_base(self) -> Path | None:
        return None


class _FakeRepo:
    """The copy statements over in-memory rows, recording every write."""

    def __init__(self, items: list[dict], rows: list[dict] | None = None) -> None:
        self.attachments = items
        self.rows: dict[str, dict] = {}
        self.data: dict[str, bytes] = {}
        self.outcomes: list[tuple[str, dict]] = []
        self.renders: list[tuple[str, dict]] = []
        for row in rows if rows is not None else [self.pending(item) for item in items]:
            self.rows[row["attachment_id"]] = row

    @staticmethod
    def pending(item: dict, **overrides) -> dict:
        aid = attachment_id_for(ENTRY, item)
        row = {
            "attachment_id": aid,
            "source_url": item.get("url"),
            "filename": "f",
            "mime_type": validated_declared_type(item.get("type")),
            "size_bytes": None,
            "copy_status": "pending",
            "skip_reason": None,
            "copy_attempts": 0,
            "rendition_sha256": None,
            "has_data": False,
            "created_at": datetime.now(UTC),
        }
        row.update(overrides)
        return row

    async def get_entry(self, entry_id):
        return {"entry_id": entry_id, "attachments": self.attachments}

    async def get_copy_rows(self, _entry_id):
        # Reverse of list order, so the JSONB ordering is what the code relies on.
        return [dict(r) for r in reversed(list(self.rows.values()))]

    async def count_copied_attachments(self, _entry_id):
        copied = [r for r in self.rows.values() if r["copy_status"] == "copied"]
        return len(copied), sum(r["size_bytes"] or 0 for r in copied)

    async def apply_copy_outcome(self, _entry_id, attachment_id, **kw):
        self.outcomes.append((attachment_id, kw))
        row = self.rows[attachment_id]
        row.update(
            copy_status=kw["copy_status"],
            skip_reason=kw.get("skip_reason"),
            mime_type=kw.get("mime_type"),
            size_bytes=kw.get("size_bytes"),
            has_data=kw.get("data") is not None,
            rendition_sha256=kw["rendition"].sha256 if kw.get("rendition") else None,
        )
        if kw.get("copy_attempts") is not None:
            row["copy_attempts"] = kw["copy_attempts"]
        if kw.get("data") is not None:
            self.data[attachment_id] = kw["data"]
        else:
            self.data.pop(attachment_id, None)
        return kw["copy_status"]

    async def apply_render_outcome(self, _entry_id, attachment_id, **kw):
        self.renders.append((attachment_id, kw))
        row = self.rows[attachment_id]
        row.update(
            skip_reason=kw.get("skip_reason"),
            mime_type=kw.get("mime_type"),
            rendition_sha256=kw["rendition"].sha256 if kw.get("rendition") else None,
        )
        return True

    async def get_copy_source(self, attachment_id):
        row = self.rows[attachment_id]
        if row["copy_status"] == "copied" and row["rendition_sha256"] is None:
            return self.data[attachment_id], row["mime_type"]
        return None

    def row(self, item: dict) -> dict:
        return self.rows[attachment_id_for(ENTRY, item)]


def _item(name: str, declared: str | None = None, host: str = HOST) -> dict:
    item: dict = {"url": f"{host}/files/{name}"}
    if declared is not None:
        item["type"] = declared
    return item


def _run(**overrides) -> CopyRun:
    return CopyRun(_Adapter(), ORIGINS, **overrides)  # type: ignore[arg-type]


def _serve(bodies: dict[str, bytes]):
    """A fetcher answering each url with its body (by basename)."""

    async def _fetch(url, *_args, **kwargs):
        kwargs["on_sent"]()
        return FetchOutcome(data=bodies[url.rsplit("/", 1)[1]])

    return _fetch


@pytest.fixture
def renderer(monkeypatch):
    """Fake the render worker: a PNG renders unless told to answer a content skip."""
    state = {"calls": 0, "skip": None, "unavailable": False}

    async def _prepare(data, **_kwargs):
        from osprey.imaging.formats import sniff

        sniffed = sniff(data)
        if not sniffed.is_image:
            return PreparedPicture(mime_type=sniffed.mime, skip_reason=sniffed.skip_reason)
        state["calls"] += 1
        if state["unavailable"]:
            raise RenderUnavailable("no worker", exit_code=3)
        if state["skip"]:
            return PreparedPicture(mime_type=sniffed.mime, skip_reason=state["skip"])
        return PreparedPicture(
            mime_type=sniffed.mime,
            skip_reason=None,
            rendition_bytes=b"r",
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256="ab" * 32,
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)
    return state


# (mode, declared, name, body, render skip, status, skip_reason, mime, data kept, rendition)
OUTCOME_TABLE = [
    (
        "images",
        OCTET_STREAM_T := "application/octet-stream",
        "a.bin",
        BINARY,
        None,
        "skipped",
        "copy_on_ingest_mode",
        "application/octet-stream",
        False,
        False,
    ),
    (
        "images",
        OCTET_STREAM_T,
        "doc.pdf",
        PDF,
        None,
        "skipped",
        "copy_on_ingest_mode",
        "application/pdf",
        False,
        False,
    ),
    (
        "images",
        None,
        "plot.svg",
        SVG,
        None,
        "skipped",
        "copy_on_ingest_mode",
        "image/svg+xml",
        False,
        False,
    ),
    (
        "images",
        None,
        "page",
        HTML,
        None,
        "skipped",
        "copy_on_ingest_mode",
        "text/html",
        False,
        False,
    ),
    (
        "all",
        "application/pdf",
        "doc.pdf",
        PDF,
        None,
        "copied",
        "reserved_format",
        "application/pdf",
        True,
        False,
    ),
    ("all", None, "plot.svg", SVG, None, "copied", "reserved_format", "image/svg+xml", True, False),
    (
        "all",
        "application/x-foo",
        "a.bin",
        BINARY,
        None,
        "copied",
        "not_an_image",
        "application/octet-stream",
        True,
        False,
    ),
    ("all", None, "page", HTML, None, "copied", "not_an_image", "text/html", True, False),
    ("images", "image/png", "a.png", PNG, None, "copied", None, "image/png", True, True),
    ("all", None, "a.dat", PNG, None, "copied", None, "image/png", True, True),
    (
        "images",
        "image/png",
        "a.png",
        PNG,
        "decoder_failed",
        "copied",
        "decoder_failed",
        "image/png",
        True,
        False,
    ),
    # A declared picture whose bytes are a web page: a login or proxy page.
    (
        "images",
        "image/png",
        "a.png",
        HTML,
        None,
        "skipped",
        "source_refused",
        "text/html",
        False,
        False,
    ),
    (
        "all",
        "image/png",
        "a.png",
        HTML,
        None,
        "skipped",
        "source_refused",
        "text/html",
        False,
        False,
    ),
    ("all", None, "shot.jpg", HTML, None, "skipped", "source_refused", "text/html", False, False),
]


@pytest.mark.usefixtures("renderer")
class TestCopyEntryOutcomeTable:
    @pytest.mark.parametrize(
        (
            "mode",
            "declared",
            "name",
            "body",
            "render_skip",
            "status",
            "skip",
            "mime",
            "kept",
            "rendered",
        ),
        OUTCOME_TABLE,
    )
    async def test_copy_entry_outcome_table(
        self,
        attachment_fetch,
        renderer,
        mode,
        declared,
        name,
        body,
        render_skip,
        status,
        skip,
        mime,
        kept,
        rendered,
    ):
        renderer["skip"] = render_skip
        attachment_fetch.respond(_serve({name: body}))
        item = _item(name, declared)
        repo = _FakeRepo([item])

        await copy_entry(repo, ENTRY, _config(mode), _run())  # type: ignore[arg-type]

        row = repo.row(item)
        assert (row["copy_status"], row["skip_reason"], row["mime_type"]) == (status, skip, mime)
        assert (repo.data.get(row["attachment_id"]) == body) is kept
        assert (row["rendition_sha256"] is not None) is rendered
        # Every copied row carrying a skip_reason is not viewable.
        assert is_viewable(row) is (status == "copied" and skip is None and rendered)
        assert len(attachment_fetch.calls) == 1

    async def test_copy_entry_declared_picture_html_then_backfill_serving_copies(
        self, attachment_fetch
    ):
        bodies = {"a.png": HTML}
        attachment_fetch.respond(_serve(bodies))
        item = _item("a.png", "image/png")
        repo = _FakeRepo([item])
        await copy_entry(repo, ENTRY, _config("images"), _run())  # type: ignore[arg-type]
        assert repo.row(item)["skip_reason"] == "source_refused"

        bodies["a.png"] = PNG
        await copy_entry(repo, ENTRY, _config("images"), _run())  # type: ignore[arg-type]
        assert repo.row(item)["skip_reason"] == "source_refused"  # plain poll: not retried
        await copy_entry(repo, ENTRY, _config("images"), _run(), retry_skipped=True)  # type: ignore[arg-type]
        row = repo.row(item)
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert is_viewable(row)

    @pytest.mark.parametrize("mode", ["images", "all"])
    @pytest.mark.parametrize("name", ["a.png", "attachment?id=7"])
    @pytest.mark.parametrize("retry", ["still_html", "transient"])
    async def test_copy_entry_source_refused_retry_keeps_declared_picture(
        self, attachment_fetch, mode, name, retry
    ):
        # A refused retry (the source still serves a login page, or a host-up
        # transient) leaves the row source_refused, even when the URL carries no
        # image extension; a plain poll leaves it alone; backfill against a
        # serving source then copies it.
        bodies = {name: HTML}
        attachment_fetch.respond(_serve(bodies))
        item = _item(name, "image/png")
        repo = _FakeRepo([item])
        await copy_entry(repo, ENTRY, _config(mode), _run())  # type: ignore[arg-type]
        assert repo.row(item)["skip_reason"] == "source_refused"

        if retry == "transient":
            attachment_fetch.respond(FetchOutcome(transient=True, host_up=True))
        await copy_entry(repo, ENTRY, _config(mode), _run(), retry_skipped=True)  # type: ignore[arg-type]
        row = repo.row(item)
        assert (row["copy_status"], row["skip_reason"]) == ("skipped", "source_refused")
        assert repo.data.get(row["attachment_id"]) is None

        attachment_fetch.respond(_serve(bodies))
        bodies[name] = PNG
        await copy_entry(repo, ENTRY, _config(mode), _run())  # type: ignore[arg-type]
        assert repo.row(item)["skip_reason"] == "source_refused"  # plain poll: not retried
        await copy_entry(repo, ENTRY, _config(mode), _run(), retry_skipped=True)  # type: ignore[arg-type]
        row = repo.row(item)
        assert (row["copy_status"], row["skip_reason"]) == ("copied", None)
        assert is_viewable(row)


@pytest.mark.usefixtures("renderer")
class TestCopyEntryDecisions:
    async def test_copy_entry_none_mode_writes_config_codes_with_no_fetch(self, attachment_fetch):
        items = [_item(f"{i}.png", "image/png") for i in range(3)]
        repo = _FakeRepo(items)
        await copy_entry(repo, ENTRY, _config("none"), _run(), retry_skipped=True)  # type: ignore[arg-type]
        assert attachment_fetch.calls == []
        assert [repo.row(i)["skip_reason"] for i in items] == ["copy_on_ingest_mode"] * 3

    async def test_copy_entry_count_budget_fills_in_list_order(self, attachment_fetch):
        items = [_item(f"{i}.png", "image/png") for i in range(5)]
        attachment_fetch.respond(_serve({f"{i}.png": PNG for i in range(5)}))
        repo = _FakeRepo(items)
        await copy_entry(repo, ENTRY, _config(), _run(max_per_entry=2))  # type: ignore[arg-type]
        assert [repo.row(i)["copy_status"] for i in items] == ["copied"] * 2 + ["skipped"] * 3
        assert [repo.row(i)["skip_reason"] for i in items[2:]] == ["per_entry_limit"] * 3
        assert [c["url"] for c in attachment_fetch.calls] == [i["url"] for i in items[:2]]

    async def test_copy_entry_default_budget_is_twenty(self, attachment_fetch):
        items = [_item(f"{i}.png", "image/png") for i in range(22)]
        attachment_fetch.respond(_serve({f"{i}.png": PNG for i in range(22)}))
        repo = _FakeRepo(items)
        await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
        statuses = [repo.row(i)["copy_status"] for i in items]
        assert statuses.count("copied") == 20
        assert [repo.row(i)["skip_reason"] for i in items[20:]] == ["per_entry_limit"] * 2

    async def test_copy_entry_byte_budget_full_fetches_nothing(self, attachment_fetch):
        done = _FakeRepo.pending(_item("old.png"), copy_status="copied", size_bytes=4 * MB)
        new = _item("new.png", "image/png")
        repo = _FakeRepo([_item("old.png"), new], rows=[done, _FakeRepo.pending(new)])
        await copy_entry(repo, ENTRY, _config(max_file_mb=1), _run())  # type: ignore[arg-type]
        assert attachment_fetch.calls == []
        assert repo.row(new)["skip_reason"] == "per_entry_limit"

    async def test_copy_entry_byte_budget_checked_on_fetched_size(self, attachment_fetch):
        big = PNG + b"\x00" * (MB - len(PNG))
        items = [_item(f"{i}.png", "image/png") for i in range(5)]
        attachment_fetch.respond(_serve({f"{i}.png": big for i in range(5)}))
        repo = _FakeRepo(items)
        await copy_entry(repo, ENTRY, _config(max_file_mb=1), _run(semaphore=asyncio.Semaphore(1)))  # type: ignore[arg-type]
        assert [repo.row(i)["copy_status"] for i in items].count("copied") == 4
        assert repo.row(items[4])["skip_reason"] == "per_entry_limit"
        assert repo.row(items[4])["has_data"] is False

    @pytest.mark.parametrize(
        ("outcome", "code", "size"),
        [
            (FetchOutcome(code="source_gone"), "source_gone", None),
            (FetchOutcome(code="source_refused"), "source_refused", None),
            (FetchOutcome(code="size_cap", observed_size=10 * MB + 1), "size_cap", 10 * MB + 1),
        ],
    )
    async def test_copy_entry_records_fetch_skip_codes(self, attachment_fetch, outcome, code, size):
        attachment_fetch.respond(outcome)
        item = _item("a.png", "image/png")
        repo = _FakeRepo([item])
        await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
        row = repo.row(item)
        assert (row["copy_status"], row["skip_reason"], row["size_bytes"]) == (
            "skipped",
            code,
            size,
        )

    async def test_copy_entry_host_up_transient_charges_until_fetch_failed(self, attachment_fetch):
        attachment_fetch.respond(FetchOutcome(transient=True, host_up=True))
        item = _item("a.png", "image/png")
        repo = _FakeRepo([item])
        for attempt in range(1, 5):
            await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
            assert (repo.row(item)["copy_status"], repo.row(item)["copy_attempts"]) == (
                "pending",
                attempt,
            )
        await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
        row = repo.row(item)
        assert (row["copy_status"], row["skip_reason"], row["copy_attempts"]) == (
            "skipped",
            "fetch_failed",
            5,
        )

    async def test_copy_entry_connect_failure_burns_no_attempt_unless_row_is_old(
        self, attachment_fetch
    ):
        attachment_fetch.respond(FetchOutcome(transient=True, host_up=False))
        fresh, old = _item("new.png", "image/png"), _item("old.png", "image/png")
        rows = [
            _FakeRepo.pending(fresh),
            _FakeRepo.pending(old, created_at=datetime.now(UTC) - timedelta(days=8)),
        ]
        repo = _FakeRepo([fresh, old], rows=rows)
        await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
        assert (repo.row(fresh)["copy_status"], repo.row(fresh)["copy_attempts"]) == ("pending", 0)
        assert [aid for aid, _ in repo.outcomes] == [repo.row(old)["attachment_id"]]
        assert repo.row(old)["skip_reason"] == "fetch_failed"

    async def test_copy_entry_render_unavailable_stores_without_rendition_and_stops_rendering(
        self, attachment_fetch, renderer
    ):
        renderer["unavailable"] = True
        items = [_item(f"{i}.png", "image/png") for i in range(3)]
        attachment_fetch.respond(_serve({f"{i}.png": PNG for i in range(3)}))
        repo = _FakeRepo(items)
        run = _run()
        await copy_entry(repo, ENTRY, _config(), run)  # type: ignore[arg-type]
        assert run.render_available is False
        assert renderer["calls"] == 1
        for item in items:
            row = repo.row(item)
            assert (row["copy_status"], row["skip_reason"], row["rendition_sha256"]) == (
                "copied",
                None,
                None,
            )
            assert repo.data[row["attachment_id"]] == PNG

    async def test_copy_entry_render_only_rows_skip_network_in_every_mode(self, attachment_fetch):
        for mode in ("images", "all", "none"):
            png, pdf = _item("n.png"), _item("n.pdf")
            rows = [
                _FakeRepo.pending(png, copy_status="copied", has_data=True, size_bytes=len(PNG)),
                _FakeRepo.pending(pdf, copy_status="copied", has_data=True, size_bytes=len(PDF)),
            ]
            repo = _FakeRepo([png, pdf], rows=rows)
            repo.data = {rows[0]["attachment_id"]: PNG, rows[1]["attachment_id"]: PDF}
            await copy_entry(repo, ENTRY, _config(mode), _run(max_per_entry=0))  # type: ignore[arg-type]
            assert attachment_fetch.calls == []
            assert repo.outcomes == []
            assert repo.row(png)["rendition_sha256"] is not None
            assert repo.row(pdf)["skip_reason"] == "reserved_format"
            assert repo.row(pdf)["mime_type"] == "application/pdf"
            assert repo.row(pdf)["copy_status"] == "copied"

    async def test_copy_entry_retry_skipped_takes_source_and_config_rows_only(
        self, attachment_fetch
    ):
        attachment_fetch.respond(_serve({"g.png": PNG, "c.png": PNG, "d.png": PNG}))
        gone, cfg, content = (
            _item("g.png", "image/png"),
            _item("c.png", "image/png"),
            _item("d.png", "image/png"),
        )
        rows = [
            _FakeRepo.pending(gone, copy_status="skipped", skip_reason="source_gone"),
            _FakeRepo.pending(cfg, copy_status="skipped", skip_reason="per_entry_limit"),
            _FakeRepo.pending(content, copy_status="skipped", skip_reason="decoder_failed"),
        ]
        repo = _FakeRepo([gone, cfg, content], rows=rows)
        await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
        assert attachment_fetch.calls == []
        await copy_entry(repo, ENTRY, _config(), _run(), retry_skipped=True)  # type: ignore[arg-type]
        assert sorted(c["url"] for c in attachment_fetch.calls) == sorted([gone["url"], cfg["url"]])
        assert repo.row(content)["skip_reason"] == "decoder_failed"

    async def test_copy_entry_passes_run_session_and_on_sent(self, attachment_fetch):
        attachment_fetch.respond(_serve({"a.png": PNG}))
        sentinel = object()
        repo = _FakeRepo([_item("a.png", "image/png")])
        await copy_entry(repo, ENTRY, _config(), _run(session=sentinel))  # type: ignore[arg-type]
        (call,) = attachment_fetch.calls
        assert call["session"] is sentinel
        assert callable(call["on_sent"])
        assert 0 < call["total"] <= 60


def test_copy_entry_module_classifies_no_markup_itself():
    source = inspect.getsource(copy_mod)
    for needle in ("<html", "<!doctype", "text/html", "_MARKUP"):
        assert needle not in source.lower()
    assert "is_markup(" in source


async def test_copy_entry_run_owns_one_session():
    import aiohttp

    class _NetAdapter(_Adapter):
        def _create_connector(self):
            return aiohttp.TCPConnector()

    run = CopyRun(_NetAdapter(), ORIGINS)  # type: ignore[arg-type]
    async with run as entered:
        assert entered is run
        session = run.session
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed
    assert session.closed
    assert run.session is None


# --- breaker -------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


H = ("https", "logbook.example.org", 443)


def test_copy_entry_breaker_trips_on_five_consecutive_transients():
    breaker = HostBreaker(5, 60.0, clock=_Clock())
    for _ in range(4):
        assert breaker.allow(H)
        breaker.record(H, transient=True)
    assert not breaker.is_open(H)
    breaker.record(H, transient=True)
    assert breaker.is_open(H) and not breaker.allow(H)


def test_copy_entry_breaker_interleaved_successes_never_trip():
    breaker = HostBreaker(5, 60.0, clock=_Clock())
    for _ in range(20):
        for _ in range(4):
            breaker.record(H, transient=True)
        breaker.record(H, transient=False)
    assert not breaker.is_open(H) and breaker.allow(H)


def test_copy_entry_breaker_cooldown_lets_one_probe_through():
    clock = _Clock()
    breaker = HostBreaker(5, 60.0, clock=clock)
    for _ in range(5):
        breaker.record(H, transient=True)
    clock.now = 59.0
    assert not breaker.allow(H)
    clock.now = 60.0
    assert breaker.allow(H)
    assert not breaker.allow(H)  # one probe at a time
    breaker.record(H, transient=True)  # the probe failed: re-tripped
    assert not breaker.allow(H)
    clock.now = 120.0
    assert breaker.allow(H)
    breaker.record(H, transient=False)  # the probe answered: closed
    assert not breaker.is_open(H) and breaker.allow(H) and breaker.allow(H)


def test_copy_entry_breaker_released_probe_can_be_retried():
    clock = _Clock()
    breaker = HostBreaker(1, 1.0, clock=clock)
    breaker.record(H, transient=True)
    clock.now = 1.0
    assert breaker.allow(H)
    breaker.release(H)
    assert breaker.allow(H)


def test_copy_entry_run_constants_are_fields_with_production_defaults():
    run = _run()
    assert run.entry_deadline_s == 60.0
    assert run.pending_max_age == timedelta(days=7)
    assert run.breaker_threshold == 5
    assert run.breaker_cooldown_s == 60.0
    assert run.max_per_entry == 20
    assert run.breaker is not None and run.breaker.threshold == 5
    assert run.semaphore._value == 4


@pytest.mark.timeout(60)
@pytest.mark.usefixtures("renderer")
class TestCopyEntryScaled:
    async def test_copy_entry_down_host_costs_five_connects_then_holds(self, attachment_fetch):
        attachment_fetch.respond(FetchOutcome(transient=True, host_up=False))
        items = [_item(f"{i}.png", "image/png") for i in range(12)]
        repo = _FakeRepo(items)
        await copy_entry(repo, ENTRY, _config(), _run())  # type: ignore[arg-type]
        assert len(attachment_fetch.calls) == 5
        assert all(repo.row(i)["copy_status"] == "pending" for i in items)
        assert repo.outcomes == []

    async def test_copy_entry_five_interleaved_transients_never_trip(self, attachment_fetch):
        seen: list[str] = []

        async def _fetch(url, *_args, **kwargs):
            seen.append(url)
            kwargs["on_sent"]()
            if len(seen) % 2:  # 1st, 3rd, ... fail; the others succeed
                return FetchOutcome(transient=True, host_up=True)
            return FetchOutcome(data=PNG)

        attachment_fetch.respond(_fetch)
        items = [_item(f"{i}.png", "image/png") for i in range(10)]
        repo = _FakeRepo(items)
        run = _run(semaphore=asyncio.Semaphore(1))
        await copy_entry(repo, ENTRY, _config(), run)  # type: ignore[arg-type]
        assert len(seen) == 10
        assert not run.breaker.is_open(H)  # type: ignore[union-attr]

    async def test_copy_entry_host_recovers_after_scaled_cooldown(self, attachment_fetch):
        state = {"up": False}

        async def _fetch(*_args, **kwargs):
            if not state["up"]:
                return FetchOutcome(transient=True, host_up=False)
            kwargs["on_sent"]()
            return FetchOutcome(data=PNG)

        attachment_fetch.respond(_fetch)
        items = [_item(f"{i}.png", "image/png") for i in range(8)]
        repo = _FakeRepo(items)
        run = _run(breaker_cooldown_s=0.2)
        await copy_entry(repo, ENTRY, _config(), run)  # type: ignore[arg-type]
        assert len(attachment_fetch.calls) == 5
        state["up"] = True
        await copy_entry(repo, ENTRY, _config(), run)  # type: ignore[arg-type]
        assert len(attachment_fetch.calls) == 5  # still cooling down
        await asyncio.sleep(0.25)
        await copy_entry(repo, ENTRY, _config(), run)  # type: ignore[arg-type]
        assert all(repo.row(i)["copy_status"] == "copied" for i in items)

    async def test_copy_entry_deadline_charges_only_sent_requests(self, attachment_fetch):
        async def _hang(*_args, **kwargs):
            kwargs["on_sent"]()
            await asyncio.sleep(0.9)
            return FetchOutcome(data=PNG)

        attachment_fetch.respond(_hang)
        items = [_item(f"{i}.png", "image/png") for i in range(6)]
        repo = _FakeRepo(items)
        loop = asyncio.get_running_loop()
        started = loop.time()
        await copy_entry(repo, ENTRY, _config(), _run(entry_deadline_s=0.6))  # type: ignore[arg-type]
        assert loop.time() - started < 0.85
        attempts = [repo.row(i)["copy_attempts"] for i in items]
        assert attempts == [1, 1, 1, 1, 0, 0]
        assert all(repo.row(i)["copy_status"] == "pending" for i in items)

"""Tests for the attachment summary builder and ``file_source_for``."""

from __future__ import annotations

import json
import logging

import pytest

from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.attachments import summaries as summaries_mod
from osprey.services.ariel_search.attachments.summaries import (
    LISTING_ONLY_KEYS,
    SUMMARY_KEYS,
    build_attachment_summaries,
    build_attachment_summary_pairs,
    file_source_for,
)
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import AdapterNotFoundError

ENTRY = "entry-42"
MODEL = "vision-model"
NATIVE_ID = "att-0123456789ab"
SHA = "f" * 64


def _item(url, filename="beam.png", type_="image/png", caption=None) -> dict:
    item = {"url": url, "type": type_, "filename": filename}
    if caption is not None:
        item["caption"] = caption
    return item


def _row(item, **overrides) -> dict:
    row = {
        "attachment_id": attachment_id_for(ENTRY, item),
        "entry_id": ENTRY,
        "filename": item.get("filename"),
        "mime_type": "image/png",
        "source_url": item.get("url"),
        "copy_status": "copied",
        "skip_reason": None,
        "rendition_sha256": SHA,
    }
    row.update(overrides)
    return row


def _entry(items, captions=None) -> dict:
    entry = {"entry_id": ENTRY, "attachments": items}
    if captions is not None:
        entry["attachment_captions"] = captions
    return entry


def _build(entry, rows=(), limit=None, matched=(), *, file_source=False, **kw):
    return build_attachment_summaries(
        entry,
        list(rows) if rows is not None else None,
        limit,
        matched,
        file_source=file_source,
        **kw,
    )


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_strings(v)


class TestKeys:
    def test_summary_keys_shape(self):
        assert SUMMARY_KEYS == (
            "attachment_id",
            "filename",
            "mime_type",
            "viewable",
            "copy_status",
            "skip_reason",
            "caption",
            "caption_source",
            "visible_text",
            "url",
        )
        assert LISTING_ONLY_KEYS == ("caption_truncated", "visible_text_truncated")

    def test_emitted_keys_are_known(self):
        item = _item("https://h/x.png", caption="c" * 300)
        out = _build(_entry([item]), [_row(item)])
        assert set(out[0]) <= set(SUMMARY_KEYS) | set(LISTING_ONLY_KEYS)


class TestJoin:
    def test_copied_row_is_viewable_with_id(self):
        item = _item("https://h/beam.png")
        [s] = _build(_entry([item]), [_row(item)])
        assert s == {
            "attachment_id": attachment_id_for(ENTRY, item),
            "filename": "beam.png",
            "mime_type": "image/png",
            "viewable": True,
            "copy_status": "copied",
            "url": "https://h/beam.png",
        }

    def test_native_item_joins_by_parsed_id_and_has_no_url(self):
        item = _item(f"/api/attachments/{NATIVE_ID}")
        row = _row(item, source_url=None)
        assert row["attachment_id"] == NATIVE_ID
        [s] = _build(_entry([item]), [row])
        assert s["attachment_id"] == NATIVE_ID
        assert s["viewable"] is True
        assert "url" not in s

    def test_skipped_row_carries_reason(self):
        item = _item("https://h/doc.pdf", filename="doc.pdf", type_="application/pdf")
        row = _row(
            item,
            mime_type="application/pdf",
            copy_status="skipped",
            skip_reason="copy_on_ingest_mode",
            rendition_sha256=None,
        )
        [s] = _build(_entry([item]), [row])
        assert s["copy_status"] == "skipped"
        assert s["skip_reason"] == "copy_on_ingest_mode"
        assert s["viewable"] is False
        assert s["attachment_id"] == row["attachment_id"]

    def test_copied_without_rendition_not_viewable(self):
        item = _item("https://h/beam.png")
        [s] = _build(_entry([item]), [_row(item, rendition_sha256=None)])
        assert s["viewable"] is False
        assert s["copy_status"] == "copied"

    def test_fetchable_item_without_row_is_pending_without_id(self):
        item = _item("https://h/beam.png")
        [s] = _build(_entry([item]), [])
        assert s["copy_status"] == "pending"
        assert "attachment_id" not in s
        assert "skip_reason" not in s
        assert s["viewable"] is False

    def test_empty_url_is_no_source_url(self):
        [s] = _build(_entry([_item("", type_=None, filename="hdr")]), [])
        assert s["copy_status"] == "skipped"
        assert s["skip_reason"] == "no_source_url"
        assert "attachment_id" not in s
        assert s["filename"] == "hdr"

    def test_null_declared_type_is_octet_stream(self):
        [s] = _build(_entry([_item("https://h/x", type_=None)]), [])
        assert s["mime_type"] == "application/octet-stream"

    def test_null_row_mime_is_octet_stream(self):
        item = _item("https://h/x", type_=None)
        [s] = _build(_entry([item]), [_row(item, mime_type=None)])
        assert s["mime_type"] == "application/octet-stream"
        assert s["viewable"] is False

    def test_ornl_header_only_items(self):
        items = [{"url": "", "filename": f"header-{i}", "type": None} for i in range(3)]
        out = _build(_entry(items), [])
        assert len(out) == 3
        for s in out:
            assert s["copy_status"] == "skipped"
            assert s["skip_reason"] == "no_source_url"
            assert "attachment_id" not in s

    def test_attachments_as_json_string(self):
        item = _item("https://h/beam.png")
        [s] = _build({"entry_id": ENTRY, "attachments": json.dumps([item])}, [_row(item)])
        assert s["viewable"] is True

    def test_non_mapping_items_ignored(self):
        assert _build(_entry(["x", 3, None]), []) == []

    def test_filename_falls_back_to_url_basename(self):
        item = {"url": "https://h/dir/scope.png", "type": "image/png"}
        [s] = _build(_entry([item]), [])
        assert s["filename"] == "scope.png"


class TestNonFetchable:
    def test_relative_path_on_http_source_is_no_source_url(self):
        [s] = _build(_entry([_item("/rel/x.png")]), [], file_source=False)
        assert s["copy_status"] == "skipped"
        assert s["skip_reason"] == "no_source_url"
        assert "attachment_id" not in s

    def test_dot_dot_url_is_no_source_url(self):
        [s] = _build(_entry([_item("https://h/a/../x")]), [], file_source=False)
        assert s["skip_reason"] == "no_source_url"
        assert "attachment_id" not in s

    def test_relative_path_on_file_source_stays_pending(self):
        [s] = _build(_entry([_item("pics/x.png")]), [], file_source=True)
        assert s["copy_status"] == "pending"
        assert "attachment_id" not in s
        assert "url" not in s

    def test_row_wins_over_predicate(self):
        item = _item("pics/x.png")
        [s] = _build(_entry([item]), [_row(item)], file_source=False)
        assert s["viewable"] is True
        assert s["attachment_id"] == attachment_id_for(ENTRY, item)
        assert s["copy_status"] == "copied"

    def test_native_item_without_row_is_pending(self):
        [s] = _build(_entry([_item(f"/api/attachments/{NATIVE_ID}")]), [])
        assert s["copy_status"] == "pending"
        assert "attachment_id" not in s


class TestFallback:
    def test_rows_none_everything_pending(self):
        items = [
            _item("https://h/a.png"),
            _item(f"/api/attachments/{NATIVE_ID}"),
            _item("", type_=None),
            _item("/rel/x.png"),
        ]
        out = _build(_entry(items), rows=None)
        assert len(out) == 4
        for s in out:
            assert s["copy_status"] == "pending"
            assert s["viewable"] is False
            assert "attachment_id" not in s
            assert "skip_reason" not in s


class TestCaptions:
    def test_upstream_caption(self):
        item = _item("https://h/a.png", caption="beam spot")
        [s] = _build(_entry([item]), [_row(item)])
        assert s["caption"] == "beam spot"
        assert s["caption_source"] == "upstream"
        assert "visible_text" not in s

    def test_model_caption_wins_when_model_id_set(self):
        item = _item("https://h/a.png", caption="upstream text")
        aid = attachment_id_for(ENTRY, item)
        captions = {aid: {MODEL: {"caption": "model text", "visible_text": "QX-77"}}}
        [s] = _build(_entry([item], captions), [_row(item)], model_id=MODEL)
        assert s["caption"] == "model text"
        assert s["caption_source"] == f"model:{MODEL}"
        assert s["visible_text"] == "QX-77"

    def test_model_caption_ignored_without_model_id(self):
        item = _item("https://h/a.png", caption="upstream text")
        aid = attachment_id_for(ENTRY, item)
        captions = {aid: {MODEL: {"caption": "model text", "visible_text": "QX-77"}}}
        [s] = _build(_entry([item], captions), [_row(item)], model_id=None)
        assert s["caption"] == "upstream text"
        assert s["caption_source"] == "upstream"
        assert "visible_text" not in s

    def test_error_object_ignored(self):
        item = _item("https://h/a.png", caption="upstream text")
        aid = attachment_id_for(ENTRY, item)
        captions = {aid: {MODEL: {"error": "timeout", "attempts": 2}}}
        [s] = _build(_entry([item], captions), [_row(item)], model_id=MODEL)
        assert s["caption_source"] == "upstream"

    def test_no_caption_at_all(self):
        [s] = _build(_entry([_item("https://h/a.png")]), [])
        assert "caption" not in s and "caption_source" not in s

    def test_listing_truncates_to_200(self):
        item = _item("https://h/a.png")
        aid = attachment_id_for(ENTRY, item)
        captions = {aid: {MODEL: {"caption": "c" * 500, "visible_text": "v" * 300}}}
        [s] = _build(_entry([item], captions), [_row(item)], model_id=MODEL)
        assert s["caption"] == "c" * 200
        assert s["caption_truncated"] is True
        assert s["visible_text"] == "v" * 200
        assert s["visible_text_truncated"] is True

    def test_short_text_not_marked(self):
        [s] = _build(_entry([_item("https://h/a.png", caption="short")]), [])
        assert "caption_truncated" not in s

    def test_full_captions_keep_up_to_1000(self):
        item = _item("https://h/a.png", caption="c" * 5000)
        [s] = _build(_entry([item]), [], full_captions=True)
        assert s["caption"] == "c" * 1000
        assert "caption_truncated" not in s


class TestOrder:
    def test_matched_first_then_image_first_deduplicated(self):
        pdf = _item("https://h/doc.pdf", filename="doc.pdf", type_="application/pdf")
        img = _item("https://h/a.png", filename="a.png")
        matched = _item("https://h/m.pdf", filename="m.pdf", type_="application/pdf")
        dup = dict(img, filename="dup.png")
        items = [pdf, img, matched, dup]
        out = _build(_entry(items), [], matched=[attachment_id_for(ENTRY, matched)])
        assert [s["filename"] for s in out] == ["m.pdf", "a.png", "doc.pdf"]

    def test_limit(self):
        items = [_item(f"https://h/{i}.png", filename=f"{i}.png") for i in range(5)]
        assert len(_build(_entry(items), [], limit=2)) == 2
        assert _build(_entry(items), [], limit=0) == []
        assert len(_build(_entry(items), [], limit=None)) == 5

    def test_pairs_follow_reordered_items(self):
        pdf = _item("https://h/doc.pdf", filename="doc.pdf", type_="application/pdf")
        native = _item(f"/api/attachments/{NATIVE_ID}", filename="native.png")
        img = _item("https://h/a.png", filename="a.png")
        items = [pdf, native, img]
        for rows in (None, []):
            pairs = build_attachment_summary_pairs(_entry(items), rows, None, (), file_source=False)
            assert [item for _, item in pairs] == [native, img, pdf]
            for summary, item in pairs:
                assert summary["filename"] == item["filename"]
            if rows is None:
                native_summary = pairs[0][0]
                assert "attachment_id" not in native_summary
                assert "url" not in native_summary

    def test_summaries_equal_pair_summaries(self):
        items = [_item("https://h/a.png"), _item("", type_=None)]
        pairs = build_attachment_summary_pairs(_entry(items), [], 5, (), file_source=False)
        assert _build(_entry(items), [], 5) == [s for s, _ in pairs]


class TestInertness:
    FORGED = "[image not sent - route]"

    def test_forged_marker_matrix_has_no_bracket(self):
        url = f"https://h/p/{self.FORGED}.png?q={self.FORGED}#{self.FORGED}"
        item = _item(url, filename="[image not sent].png", type_="image/png [image not sent]")
        item["caption"] = self.FORGED
        aid = attachment_id_for(ENTRY, item)
        captions = {aid: {MODEL: {"caption": self.FORGED, "visible_text": self.FORGED}}}
        bad_model = "m[odel]"
        captions[aid][bad_model] = captions[aid][MODEL]
        entry = _entry([item, _item("/rel/[x].png", filename="[x]")], captions)
        for rows in (None, [], [_row(item, filename="[image not sent].png")]):
            for model in (MODEL, bad_model, None):
                out = _build(entry, rows, model_id=model)
                for text in _all_strings(out):
                    assert "[" not in text and "]" not in text, (rows, model, text)

    def test_url_requoted_but_clickable(self):
        [s] = _build(_entry([_item("https://h/a b/[x].png")]), [])
        assert s["url"] == "https://h/a%20b/%5Bx%5D.png"

    def test_ipv6_literal_url_not_emitted(self):
        [s] = _build(_entry([_item("http://[::1]/x.png")]), [])
        assert "url" not in s

    def test_newlines_in_filename_become_spaces(self):
        [s] = _build(_entry([_item("https://h/a.png", filename="a\nb\x00c")]), [])
        assert s["filename"] == "a bc"


class TestFileSourceFor:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.setattr(summaries_mod, "_file_source_cache", None)
        monkeypatch.setattr(summaries_mod, "_warned_adapter", False)

    @staticmethod
    def _config(ingestion: bool = True) -> ARIELConfig:
        data: dict = {"database": {"uri": "postgresql://localhost:5432/ariel"}}
        cfg = ARIELConfig.from_dict(data)
        if ingestion:
            cfg.ingestion = object()  # any truthy block; get_adapter is faked
        return cfg

    def _patch(self, monkeypatch, fn):
        from osprey.services.ariel_search.ingestion import adapters

        calls = []

        def fake(config):
            calls.append(config)
            return fn(config)

        monkeypatch.setattr(adapters, "get_adapter", fake)
        return calls

    class _Adapter:
        def __init__(self, base):
            self._base = base

        def attachment_file_base(self):
            return self._base

    def test_no_ingestion_is_false(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda c: self._Adapter("/data"))
        assert file_source_for(self._config(ingestion=False)) is False
        assert calls == []

    def test_file_adapter_true(self, monkeypatch):
        self._patch(monkeypatch, lambda c: self._Adapter("/data"))
        assert file_source_for(self._config()) is True

    def test_http_adapter_false(self, monkeypatch):
        self._patch(monkeypatch, lambda c: self._Adapter(None))
        assert file_source_for(self._config()) is False

    def test_adapter_not_found_false_with_one_warning(self, monkeypatch, caplog):
        def boom(_config):
            raise AdapterNotFoundError("nope", adapter_name="x", available_adapters=[])

        self._patch(monkeypatch, boom)
        caplog.set_level(logging.WARNING, logger="ariel")
        assert file_source_for(self._config()) is False
        assert file_source_for(self._config()) is False  # new config, cache miss
        warnings = [r for r in caplog.records if "ariel.ingestion.adapter" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING

    def test_constructor_error_false(self, monkeypatch):
        def boom(_config):
            raise ValueError("source_url missing")

        self._patch(monkeypatch, boom)
        assert file_source_for(self._config()) is False

    def test_cached_per_config(self, monkeypatch):
        calls = self._patch(monkeypatch, lambda c: self._Adapter("/data"))
        cfg = self._config()
        assert file_source_for(cfg) is True
        assert file_source_for(cfg) is True
        assert len(calls) == 1
        other = self._config()
        assert file_source_for(other) is True
        assert len(calls) == 2

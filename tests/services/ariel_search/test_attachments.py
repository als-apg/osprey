"""Tests for ARIEL attachment processing module."""

import hashlib
import io
import re
from unittest.mock import AsyncMock

import pytest

from osprey.services.ariel_search import attachments as attachments_module
from osprey.services.ariel_search.attachments import (
    ATTACHMENT_ID_RE,
    DEFAULT_MAX_ATTACHMENT_MB,
    AttachmentValidationError,
    attachment_id_for,
    fetchable_url,
    generate_attachment_id,
    guess_mime_type,
    is_native_item,
    process_attachments_for_entry,
    read_local_file,
    store_native_attachment,
    validate_file_size,
)
from osprey.services.ariel_search.attachments import prepare as prepare_module
from osprey.services.ariel_search.attachments.prepare import PreparedPicture, RenderUnavailable
from osprey.services.ariel_search.database.repository import CopyRendition, SchemaFacts

_DEFAULT_MAX_BYTES = DEFAULT_MAX_ATTACHMENT_MB * 1024 * 1024


class TestValidateFileSize:
    """Tests for validate_file_size."""

    def test_valid_size(self):
        """Files under the limit pass validation."""
        validate_file_size(1024, "small.txt")

    def test_exact_limit(self):
        """Files at exactly the limit pass validation."""
        validate_file_size(_DEFAULT_MAX_BYTES, "exact.bin")

    def test_exceeds_limit(self):
        """Files over the limit raise AttachmentValidationError."""
        with pytest.raises(AttachmentValidationError, match="exceeds"):
            validate_file_size(_DEFAULT_MAX_BYTES + 1, "big.bin")


class TestGuessMimeType:
    """Tests for guess_mime_type."""

    def test_png(self):
        assert guess_mime_type("photo.png") == "image/png"

    def test_jpeg(self):
        assert guess_mime_type("photo.jpg") == "image/jpeg"

    def test_pdf(self):
        assert guess_mime_type("doc.pdf") == "application/pdf"

    def test_unknown(self):
        result = guess_mime_type("data.xyz123")
        # Unknown extensions return None
        assert result is None


class TestGenerateAttachmentId:
    """Tests for generate_attachment_id."""

    def test_prefix(self):
        aid = generate_attachment_id()
        assert aid.startswith("att-")

    def test_length(self):
        aid = generate_attachment_id()
        # "att-" + 12 hex chars = 16 total
        assert len(aid) == 16

    def test_uniqueness(self):
        ids = {generate_attachment_id() for _ in range(100)}
        assert len(ids) == 100


class TestReadLocalFile:
    """Tests for read_local_file."""

    def test_reads_file(self, tmp_path):
        """Reading a valid file returns data, filename, and mime_type."""
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)

        data, filename, mime_type = read_local_file(str(f))
        assert data == b"\x89PNG" + b"\x00" * 100
        assert filename == "test.png"
        assert mime_type == "image/png"

    def test_file_not_found(self):
        """Nonexistent file raises AttachmentValidationError."""
        with pytest.raises(AttachmentValidationError, match="not found"):
            read_local_file("/nonexistent/path/file.txt")

    def test_directory_rejected(self, tmp_path):
        """Directories are rejected."""
        with pytest.raises(AttachmentValidationError, match="Not a file"):
            read_local_file(str(tmp_path))

    def test_oversized_file(self, tmp_path):
        """Files exceeding the size limit are rejected."""
        f = tmp_path / "huge.bin"
        f.write_bytes(b"\x00" * (_DEFAULT_MAX_BYTES + 1))

        with pytest.raises(AttachmentValidationError, match="exceeds"):
            read_local_file(str(f))


class TestProcessAttachmentsForEntry:
    """Tests for process_attachments_for_entry."""

    async def test_processes_files(self, tmp_path):
        """Processing valid files stores them and returns AttachmentInfo list."""
        f1 = tmp_path / "image.png"
        f1.write_bytes(b"\x89PNG" + b"\x00" * 50)

        f2 = tmp_path / "notes.txt"
        f2.write_bytes(b"Some notes here")

        mock_repo = _fake_repo(SchemaFacts(False, False))

        result = await process_attachments_for_entry(
            entry_id="test-entry-1",
            file_paths=[str(f1), str(f2)],
            repository=mock_repo,
        )

        assert len(result) == 2
        assert mock_repo.store_attachment.call_count == 2
        mock_repo.insert_native_attachment.assert_not_called()

        # Check returned AttachmentInfo
        assert result[0]["filename"] == "image.png"
        assert result[0]["type"] == "image/png"
        assert result[0]["url"].startswith("/api/attachments/att-")

        assert result[1]["filename"] == "notes.txt"
        assert result[1]["type"] == "text/plain"

    async def test_validation_fails_before_storing(self, tmp_path):
        """If any file fails validation, no attachments are stored."""
        good = tmp_path / "good.txt"
        good.write_bytes(b"ok")

        mock_repo = _fake_repo(SchemaFacts(True, True))

        with pytest.raises(AttachmentValidationError, match="not found"):
            await process_attachments_for_entry(
                entry_id="test-entry-2",
                file_paths=[str(good), "/nonexistent/bad.txt"],
                repository=mock_repo,
            )

        # No store calls should have been made
        mock_repo.store_attachment.assert_not_called()
        mock_repo.insert_native_attachment.assert_not_called()


def _section(section: object):
    """A ``get_config_value`` double that serves only the ``ariel.attachments`` block."""

    def get_config_value(path, *_args, **_kwargs):
        assert path == "ariel.attachments", path
        return section

    return get_config_value


class TestTheAttachmentCapIsAConfigKey:
    """``ariel.attachments.max_file_mb``: default, override, refusal.

    Attachments are BYTEA rows in the same Postgres the logbook lives in, so
    both directions of this number are a site storage decision. The block is
    parsed by ``AttachmentsConfig``, the one parser of ``ariel.attachments.*``.
    """

    def test_default_when_no_config_is_primed(self, monkeypatch):
        """A standalone ARIEL reads no config and still has a bound."""
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no config")),
        )

        assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES

    def test_an_absent_block_is_the_default(self, monkeypatch):
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section({}))

        assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES

    def test_configured_value_is_read_in_megabytes(self, monkeypatch):
        """The key is authored in MB; validation compares bytes."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section({"max_file_mb": 50}))

        assert attachments_module.max_attachment_bytes() == 50 * 1024 * 1024
        validate_file_size(40 * 1024 * 1024, "trace.bin")
        with pytest.raises(AttachmentValidationError, match="exceeds"):
            validate_file_size(50 * 1024 * 1024 + 1, "trace.bin")

    @pytest.mark.parametrize("bad", [0, -1, True, "10", None])
    def test_an_unusable_cap_falls_back_to_the_default(self, monkeypatch, bad):
        """A nonsense cap keeps the documented bound rather than removing it."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section({"max_file_mb": bad}))

        assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES

    def test_a_non_number_cap_warns_and_keeps_ten_megabytes(self, monkeypatch, caplog):
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section({"max_file_mb": "x"}))

        with caplog.at_level("WARNING"):
            assert attachments_module.max_attachment_bytes() == 10 * 1024 * 1024
        assert "ariel.attachments.max_file_mb" in caplog.text

    @pytest.mark.parametrize("section", [50, "10", ["max_file_mb", 50], True])
    def test_a_non_mapping_block_is_logged_and_the_default_kept(self, monkeypatch, caplog, section):
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section(section))

        with caplog.at_level("WARNING"):
            assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES
        assert "ariel.attachments" in caplog.text

    @pytest.mark.parametrize(
        "section",
        [
            {"copy_on_ingest": "bogus", "max_file_mb": 50},
            {"allowed_origins": "elog.example.org"},
            {"allowed_origins": ["elog.example.org"]},
            {"view": {"enabled": "no"}},
        ],
    )
    def test_a_malformed_sibling_does_not_break_validation(self, monkeypatch, caplog, section):
        """The size check keeps a bound when another attachments knob is malformed."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section(section))

        with caplog.at_level("WARNING"):
            assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES
            validate_file_size(_DEFAULT_MAX_BYTES, "trace.bin")
            with pytest.raises(AttachmentValidationError, match="10 MB limit"):
                validate_file_size(_DEFAULT_MAX_BYTES + 1, "trace.bin")
        assert "ariel.attachments" in caplog.text

    def test_the_refusal_names_the_configured_limit(self, monkeypatch):
        """The operator is told the number in force, not a framework literal."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _section({"max_file_mb": 50}))

        with pytest.raises(AttachmentValidationError, match="50 MB limit"):
            validate_file_size(60 * 1024 * 1024, "trace.bin")


# ---------------------------------------------------------------------------
# Package layout and the JSONB-item -> row id helpers
# ---------------------------------------------------------------------------


class TestAttachmentsIsAPackage:
    """The module became a package without changing its public surface."""

    def test_package_keeps_every_public_name(self):
        for name in (
            "DEFAULT_MAX_ATTACHMENT_MB",
            "AttachmentValidationError",
            "max_attachment_bytes",
            "validate_file_size",
            "guess_mime_type",
            "generate_attachment_id",
            "read_local_file",
            "process_attachments_for_entry",
        ):
            assert hasattr(attachments_module, name), name
        assert hasattr(attachments_module, "__path__")

    def test_formats_reexports_the_imaging_registry(self):
        from osprey.imaging import formats as imaging_formats
        from osprey.services.ariel_search.attachments import formats

        for name in (
            "sniff",
            "ACCEPTED",
            "RENDITION_MAX_BYTES",
            "CONTENT_SKIP_REASONS",
            "CONFIG_SKIP_REASONS",
            "SOURCE_SKIP_REASONS",
            "skip_reason_text",
            "is_viewable",
            "VIEWABLE_SQL",
        ):
            assert getattr(formats, name) is getattr(imaging_formats, name), name


class TestIsNativeItem:
    """A native item's url is exactly ``/api/attachments/<id>``."""

    def test_native_url(self):
        assert is_native_item({"url": "/api/attachments/att-0123456789ab"})

    @pytest.mark.parametrize(
        "url",
        [
            "/api/attachments/att-0123456789ab?x=1",
            "/api/attachments/att-0123456789ab#frag",
            "/api/attachments/att-0123456789ab/",
            "/api/attachments/a/b",
            "/api/attachments//att-0123456789ab",
            "/api/attachments/",
            "https://h/api/attachments/att-0123456789ab",
            "api/attachments/att-0123456789ab",
            "",
        ],
    )
    def test_non_native_urls(self, url):
        assert not is_native_item({"url": url})

    @pytest.mark.parametrize("item", [{}, {"url": None}, {"url": 5}, None, "str", []])
    def test_missing_or_non_string_url(self, item):
        assert not is_native_item(item)


class TestAttachmentIdFor:
    """Native items map to their own id; other urls to a deterministic id."""

    def test_native_item_returns_parsed_id(self):
        item = {"url": "/api/attachments/att-0123456789ab"}
        assert attachment_id_for("e1", item) == "att-0123456789ab"

    def test_copied_item_id_is_deterministic(self):
        item = {"url": "https://h/a.png"}
        first = attachment_id_for("e1", item)
        assert first == attachment_id_for("e1", {"url": "https://h/a.png", "type": "x"})
        digest = hashlib.sha256(b"e1\0https://h/a.png").hexdigest()[:24]
        assert first == f"att-{digest}"
        assert re.match(ATTACHMENT_ID_RE, first)

    def test_id_depends_on_entry_and_url(self):
        a = attachment_id_for("e1", {"url": "https://h/a.png"})
        assert a != attachment_id_for("e2", {"url": "https://h/a.png"})
        assert a != attachment_id_for("e1", {"url": "https://h/b.png"})

    def test_separator_prevents_concatenation_collisions(self):
        assert attachment_id_for("e1", {"url": "2x"}) != attachment_id_for("e12", {"url": "x"})

    def test_non_fetchable_non_empty_url_still_gets_an_id(self):
        assert attachment_id_for("e1", {"url": "relative/x.png"}) is not None

    @pytest.mark.parametrize("item", [{}, {"url": ""}, {"url": None}, {"url": 7}, {"url": b"x"}])
    def test_empty_missing_or_non_string_url_is_none(self, item):
        assert attachment_id_for("e1", item) is None

    def test_non_dict_item_is_none(self):
        assert attachment_id_for("e1", None) is None


class TestFetchableUrl:
    """Absolute http(s), or a confinable relative path on a file source."""

    @pytest.mark.parametrize("file_source", [False, True])
    def test_absolute_http_urls(self, file_source):
        assert fetchable_url("https://h/a/x.png", file_source=file_source)
        assert fetchable_url("http://h/x.png", file_source=file_source)

    def test_relative_path_not_fetchable_on_http_source(self):
        assert not fetchable_url("/rel/x.png", file_source=False)
        assert not fetchable_url("rel/x.png", file_source=False)

    def test_relative_path_fetchable_on_file_source(self):
        assert fetchable_url("rel/x.png", file_source=True)
        assert fetchable_url("x.png", file_source=True)

    @pytest.mark.parametrize("file_source", [False, True])
    @pytest.mark.parametrize(
        "url",
        [
            "https://h/a/../x",
            "https://h/..",
            "https://h/a/%2e%2e/x",
            "../x.png",
            "a/../x.png",
            "a/..",
            "a\\..\\x.png",
        ],
    )
    def test_dot_dot_segment_never_fetchable(self, url, file_source):
        assert not fetchable_url(url, file_source=file_source)

    @pytest.mark.parametrize("file_source", [False, True])
    @pytest.mark.parametrize(
        "url",
        [
            "",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "ftp://h/x.png",
            "data:image/png;base64,AAAA",
            "//h/x.png",
            "https:///x.png",
            "/etc/passwd",
            "\\\\server\\share\\x.png",
            "C:\\x.png",
            "a\0b.png",
        ],
    )
    def test_never_fetchable(self, url, file_source):
        assert not fetchable_url(url, file_source=file_source)

    def test_non_string_never_fetchable(self):
        assert not fetchable_url(None, file_source=True)  # type: ignore[arg-type]
        assert not fetchable_url(b"x.png", file_source=True)  # type: ignore[arg-type]

    def test_dotted_names_are_not_dot_dot_segments(self):
        assert fetchable_url("https://h/a..b/x.png", file_source=False)
        assert fetchable_url("a/..b/x.png", file_source=True)


class TestAttachmentIdRe:
    """Native ids are 12 hex digits, copied ids 24."""

    @pytest.mark.parametrize("value", ["att-" + "a" * 12, "att-" + "0" * 24])
    def test_valid_ids(self, value):
        assert re.fullmatch(ATTACHMENT_ID_RE, value)

    def test_generated_native_id_matches(self):
        assert re.fullmatch(ATTACHMENT_ID_RE, generate_attachment_id())

    @pytest.mark.parametrize(
        "value",
        [
            "att-[image not sent]",
            "att-" + "x" * 24,
            "att-" + "a" * 11,
            "att-" + "a" * 13,
            "att-" + "a" * 18,
            "att-" + "a" * 25,
            "att-" + "A" * 12,
            "att-",
            "att-" + "a" * 12 + "\n",
            "ATT-" + "a" * 12,
        ],
    )
    def test_invalid_ids(self, value):
        assert not re.fullmatch(ATTACHMENT_ID_RE, value)


# ---------------------------------------------------------------------------
# Native writers: sniff, prepare, one copied row with its rendition
# ---------------------------------------------------------------------------


def _fake_repo(facts: SchemaFacts) -> AsyncMock:
    """A repository double whose ``schema_facts`` answers ``facts``."""
    repo = AsyncMock()
    repo.schema_facts = AsyncMock(return_value=facts)
    return repo


def _png_bytes() -> bytes:
    """A small real PNG."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 10, 10)).save(buf, "PNG")
    return buf.getvalue()


_PREPARED = PreparedPicture(
    mime_type="image/png",
    skip_reason=None,
    rendition_bytes=b"rendition",
    rendition_mime="image/png",
    rendition_w=8,
    rendition_h=6,
    rendition_sha256=hashlib.sha256(b"rendition").hexdigest(),
)


class TestStoreNativeAttachment:
    """``store_native_attachment``: the one path web upload and ``entry_create`` share."""

    async def test_native_picture_is_inserted_copied_with_its_rendition(self, monkeypatch):
        calls: list[bytes] = []

        async def fake_prepare(data, **_kwargs):
            calls.append(data)
            return _PREPARED

        monkeypatch.setattr(prepare_module, "prepare_picture", fake_prepare)
        repo = _fake_repo(SchemaFacts(True, True))
        data = _png_bytes()

        info = await store_native_attachment(
            repo, "e-1", filename="shot.png", declared_mime="image/png", data=data
        )

        assert calls == [data]
        repo.store_attachment.assert_not_called()
        repo.insert_native_attachment.assert_awaited_once()
        args, kwargs = repo.insert_native_attachment.call_args
        assert args[0] == "e-1"
        assert re.fullmatch(ATTACHMENT_ID_RE, args[1])
        assert info["url"] == f"/api/attachments/{args[1]}"
        assert kwargs["data"] == data
        assert kwargs["mime_type"] == "image/png"
        assert kwargs["skip_reason"] is None
        assert kwargs["rendition"] == CopyRendition(
            data=b"rendition",
            mime_type="image/png",
            width=8,
            height=6,
            sha256=_PREPARED.rendition_sha256,
        )
        assert info["filename"] == "shot.png"
        assert info["type"] == "image/png"

    async def test_native_render_unavailable_leaves_a_copied_row_without_rendition(
        self, monkeypatch
    ):
        async def unavailable(*_args, **_kwargs):
            raise RenderUnavailable("worker cannot start")

        monkeypatch.setattr(prepare_module, "prepare_picture", unavailable)
        repo = _fake_repo(SchemaFacts(True, True))
        data = _png_bytes()

        await store_native_attachment(
            repo, "e-1", filename="shot.png", declared_mime=None, data=data
        )

        kwargs = repo.insert_native_attachment.call_args.kwargs
        assert kwargs["data"] == data
        assert kwargs["mime_type"] == "image/png"
        assert kwargs["skip_reason"] is None
        assert kwargs["rendition"] is None

    async def test_native_schema_behind_uses_the_b1_insert_and_skips_prepare(self, monkeypatch):
        async def never(*_args, **_kwargs):
            raise AssertionError("prepare_picture must not run on a schema without copy state")

        monkeypatch.setattr(prepare_module, "prepare_picture", never)
        repo = _fake_repo(SchemaFacts(False, False))

        info = await store_native_attachment(
            repo, "e-1", filename="a.png", declared_mime="image/png", data=b"\x89PNGxx"
        )

        repo.insert_native_attachment.assert_not_called()
        repo.store_attachment.assert_awaited_once()
        kwargs = repo.store_attachment.call_args.kwargs
        assert kwargs == {
            "entry_id": "e-1",
            "attachment_id": info["url"].rsplit("/", 1)[1],
            "filename": "a.png",
            "mime_type": "image/png",
            "data": b"\x89PNGxx",
            "size_bytes": len(b"\x89PNGxx"),
        }
        assert info["type"] == "image/png"

    @pytest.mark.parametrize("mode", ["images", "none", "all"])
    @pytest.mark.parametrize(
        ("data", "mime", "reason"),
        [
            (b"%PDF-1.4\n%fake pdf body\n", "application/pdf", "reserved_format"),
            (b"\x00\x01\x02\x03 binary blob", "application/octet-stream", "not_an_image"),
        ],
    )
    async def test_native_non_image_keeps_data_and_records_its_skip_reason(
        self, monkeypatch, mode, data, mime, reason
    ):
        """``copy_on_ingest`` never applies to a native writer: the bytes are always kept."""
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value", _section({"copy_on_ingest": mode})
        )
        repo = _fake_repo(SchemaFacts(True, True))

        info = await store_native_attachment(
            repo, "e-1", filename="f.bin", declared_mime=None, data=data
        )

        kwargs = repo.insert_native_attachment.call_args.kwargs
        assert kwargs["data"] == data
        assert kwargs["mime_type"] == mime
        assert kwargs["skip_reason"] == reason
        assert kwargs["rendition"] is None
        assert info["type"] == mime

    async def test_native_entry_create_files_go_through_prepare(self, tmp_path, monkeypatch):
        """``process_attachments_for_entry`` (the MCP ``entry_create`` path) prepares pictures."""

        async def fake_prepare(*_args, **_kwargs):
            return _PREPARED

        monkeypatch.setattr(prepare_module, "prepare_picture", fake_prepare)
        img = tmp_path / "plot.png"
        img.write_bytes(_png_bytes())
        repo = _fake_repo(SchemaFacts(True, True))

        infos = await process_attachments_for_entry("e-2", [str(img)], repo)

        assert len(infos) == 1
        repo.store_attachment.assert_not_called()
        kwargs = repo.insert_native_attachment.call_args.kwargs
        assert kwargs["rendition"] is not None
        assert kwargs["data"] == img.read_bytes()

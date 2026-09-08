"""Tests for ARIEL attachment processing module."""

from unittest.mock import AsyncMock

import pytest

from osprey.services.ariel_search import attachments as attachments_module
from osprey.services.ariel_search.attachments import (
    DEFAULT_MAX_ATTACHMENT_MB,
    AttachmentValidationError,
    generate_attachment_id,
    guess_mime_type,
    process_attachments_for_entry,
    read_local_file,
    validate_file_size,
)

_DEFAULT_MAX_BYTES = DEFAULT_MAX_ATTACHMENT_MB * 1024 * 1024


class TestValidateFileSize:
    """Tests for validate_file_size."""

    @pytest.mark.unit
    def test_valid_size(self):
        """Files under the limit pass validation."""
        validate_file_size(1024, "small.txt")

    @pytest.mark.unit
    def test_exact_limit(self):
        """Files at exactly the limit pass validation."""
        validate_file_size(_DEFAULT_MAX_BYTES, "exact.bin")

    @pytest.mark.unit
    def test_exceeds_limit(self):
        """Files over the limit raise AttachmentValidationError."""
        with pytest.raises(AttachmentValidationError, match="exceeds"):
            validate_file_size(_DEFAULT_MAX_BYTES + 1, "big.bin")


class TestGuessMimeType:
    """Tests for guess_mime_type."""

    @pytest.mark.unit
    def test_png(self):
        assert guess_mime_type("photo.png") == "image/png"

    @pytest.mark.unit
    def test_jpeg(self):
        assert guess_mime_type("photo.jpg") == "image/jpeg"

    @pytest.mark.unit
    def test_pdf(self):
        assert guess_mime_type("doc.pdf") == "application/pdf"

    @pytest.mark.unit
    def test_unknown(self):
        result = guess_mime_type("data.xyz123")
        # Unknown extensions return None
        assert result is None


class TestGenerateAttachmentId:
    """Tests for generate_attachment_id."""

    @pytest.mark.unit
    def test_prefix(self):
        aid = generate_attachment_id()
        assert aid.startswith("att-")

    @pytest.mark.unit
    def test_length(self):
        aid = generate_attachment_id()
        # "att-" + 12 hex chars = 16 total
        assert len(aid) == 16

    @pytest.mark.unit
    def test_uniqueness(self):
        ids = {generate_attachment_id() for _ in range(100)}
        assert len(ids) == 100


class TestReadLocalFile:
    """Tests for read_local_file."""

    @pytest.mark.unit
    def test_reads_file(self, tmp_path):
        """Reading a valid file returns data, filename, and mime_type."""
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)

        data, filename, mime_type = read_local_file(str(f))
        assert data == b"\x89PNG" + b"\x00" * 100
        assert filename == "test.png"
        assert mime_type == "image/png"

    @pytest.mark.unit
    def test_file_not_found(self):
        """Nonexistent file raises AttachmentValidationError."""
        with pytest.raises(AttachmentValidationError, match="not found"):
            read_local_file("/nonexistent/path/file.txt")

    @pytest.mark.unit
    def test_directory_rejected(self, tmp_path):
        """Directories are rejected."""
        with pytest.raises(AttachmentValidationError, match="Not a file"):
            read_local_file(str(tmp_path))

    @pytest.mark.unit
    def test_oversized_file(self, tmp_path):
        """Files exceeding the size limit are rejected."""
        f = tmp_path / "huge.bin"
        f.write_bytes(b"\x00" * (_DEFAULT_MAX_BYTES + 1))

        with pytest.raises(AttachmentValidationError, match="exceeds"):
            read_local_file(str(f))


class TestProcessAttachmentsForEntry:
    """Tests for process_attachments_for_entry."""

    @pytest.mark.unit
    async def test_processes_files(self, tmp_path):
        """Processing valid files stores them and returns AttachmentInfo list."""
        f1 = tmp_path / "image.png"
        f1.write_bytes(b"\x89PNG" + b"\x00" * 50)

        f2 = tmp_path / "notes.txt"
        f2.write_bytes(b"Some notes here")

        mock_repo = AsyncMock()

        result = await process_attachments_for_entry(
            entry_id="test-entry-1",
            file_paths=[str(f1), str(f2)],
            repository=mock_repo,
        )

        assert len(result) == 2
        assert mock_repo.store_attachment.call_count == 2

        # Check returned AttachmentInfo
        assert result[0]["filename"] == "image.png"
        assert result[0]["type"] == "image/png"
        assert result[0]["url"].startswith("/api/attachments/att-")

        assert result[1]["filename"] == "notes.txt"
        assert result[1]["type"] == "text/plain"

    @pytest.mark.unit
    async def test_validation_fails_before_storing(self, tmp_path):
        """If any file fails validation, no attachments are stored."""
        good = tmp_path / "good.txt"
        good.write_bytes(b"ok")

        mock_repo = AsyncMock()

        with pytest.raises(AttachmentValidationError, match="not found"):
            await process_attachments_for_entry(
                entry_id="test-entry-2",
                file_paths=[str(good), "/nonexistent/bad.txt"],
                repository=mock_repo,
            )

        # No store calls should have been made
        mock_repo.store_attachment.assert_not_called()


class TestTheAttachmentCapIsAConfigKey:
    """``ariel.attachments.max_file_mb``: default, override, refusal.

    Attachments are BYTEA rows in the same Postgres the logbook lives in, so
    both directions of this number are a site storage decision.
    """

    @pytest.mark.unit
    def test_default_when_no_config_is_primed(self, monkeypatch):
        """A standalone ARIEL reads no config and still has a bound."""
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no config")),
        )

        assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES

    @pytest.mark.unit
    def test_configured_value_is_read_in_megabytes(self, monkeypatch):
        """The key is authored in MB; validation compares bytes."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", lambda *a, **k: 50)

        assert attachments_module.max_attachment_bytes() == 50 * 1024 * 1024
        validate_file_size(40 * 1024 * 1024, "trace.bin")
        with pytest.raises(AttachmentValidationError, match="exceeds"):
            validate_file_size(50 * 1024 * 1024 + 1, "trace.bin")

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [0, -1, True, "10", None])
    def test_an_unusable_cap_falls_back_to_the_default(self, monkeypatch, bad):
        """A nonsense cap keeps the documented bound rather than removing it."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", lambda *a, **k: bad)

        assert attachments_module.max_attachment_bytes() == _DEFAULT_MAX_BYTES

    @pytest.mark.unit
    def test_the_refusal_names_the_configured_limit(self, monkeypatch):
        """The operator is told the number in force, not a framework literal."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", lambda *a, **k: 50)

        with pytest.raises(AttachmentValidationError, match="50 MB limit"):
            validate_file_size(60 * 1024 * 1024, "trace.bin")

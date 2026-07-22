"""Tests for vendor bulk fetch/verify and retry-classification helpers.

Complements ``test_vendor.py`` (which covers is_offline / single-file retry /
URL resolution) with the manifest-walking ``fetch_all`` / ``verify_all`` paths
and the SSL-context + retry-classification edge cases they rely on.
"""

from __future__ import annotations

import http.client
import ssl
import urllib.error
from unittest.mock import patch

from osprey.interfaces import vendor

_FAKE_MANIFEST = {
    "assets": [
        {
            "name": "A",
            "filename": "a.js",
            "url": "https://cdn.example/a.js",
            "sha256": "aaaa",
            "targets": ["static/vendor"],
        }
    ],
    "font_bundles": [
        {
            "base_url": "https://fonts.example/",
            "target": "static/fonts",
            "files": {"f.woff2": "bbbb"},
        }
    ],
}


class TestIsRetryable:
    def test_generic_httpexception_is_retryable(self):
        assert vendor._is_retryable(http.client.BadStatusLine("x")) is True

    def test_oserror_is_retryable(self):
        assert vendor._is_retryable(OSError("boom")) is True

    def test_429_is_retryable(self):
        err = urllib.error.HTTPError("u", 429, "Too Many", {}, None)
        assert vendor._is_retryable(err) is True

    def test_404_is_not_retryable(self):
        err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        assert vendor._is_retryable(err) is False

    def test_cert_error_is_not_retryable(self):
        assert vendor._is_retryable(ssl.SSLCertVerificationError("bad")) is False


class TestSslContext:
    def test_insecure_disables_verification(self):
        ctx = vendor._ssl_context(insecure=True)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_secure_keeps_verification(self):
        ctx = vendor._ssl_context(insecure=False)
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED


class TestFetchAll:
    def test_skips_assets_with_matching_checksum(self, tmp_path):
        (tmp_path / "static/vendor").mkdir(parents=True)
        (tmp_path / "static/vendor/a.js").write_text("x")
        (tmp_path / "static/fonts").mkdir(parents=True)
        (tmp_path / "static/fonts/f.woff2").write_text("y")

        # Both files "match" so everything is skipped.
        with (
            patch.object(vendor, "_load_manifest", return_value=_FAKE_MANIFEST),
            patch.object(
                vendor, "_sha256", side_effect=lambda p: {"a.js": "aaaa"}.get(p.name, "bbbb")
            ),
            patch.object(vendor, "_fetch_one") as fetch_one,
        ):
            downloaded = vendor.fetch_all(base_dir=tmp_path, quiet=True)

        assert downloaded == []
        fetch_one.assert_not_called()

    def test_downloads_missing_asset_and_font(self, tmp_path):
        with (
            patch.object(vendor, "_load_manifest", return_value=_FAKE_MANIFEST),
            patch.object(vendor, "_fetch_one") as fetch_one,
        ):
            downloaded = vendor.fetch_all(base_dir=tmp_path, quiet=True)

        # One asset + one font file, both absent, so both are fetched.
        assert len(downloaded) == 2
        assert fetch_one.call_count == 2
        called_urls = {c.args[0] for c in fetch_one.call_args_list}
        assert "https://cdn.example/a.js" in called_urls
        assert "https://fonts.example/f.woff2" in called_urls

    def test_insecure_defaults_to_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_VENDOR_INSECURE", "1")
        with (
            patch.object(vendor, "_load_manifest", return_value=_FAKE_MANIFEST),
            patch.object(vendor, "_fetch_one") as fetch_one,
        ):
            vendor.fetch_all(base_dir=tmp_path, quiet=True)
        # insecure=True threaded through to each _fetch_one call.
        assert all(c.kwargs.get("insecure") is True for c in fetch_one.call_args_list)


class TestVerifyAll:
    def test_reports_missing_and_mismatch_and_ok(self, tmp_path):
        # asset a.js present + correct; font f.woff2 present but wrong checksum.
        (tmp_path / "static/vendor").mkdir(parents=True)
        (tmp_path / "static/vendor/a.js").write_text("x")
        (tmp_path / "static/fonts").mkdir(parents=True)
        (tmp_path / "static/fonts/f.woff2").write_text("y")

        def fake_sha(path):
            return "aaaa" if path.name == "a.js" else "WRONG"

        with (
            patch.object(vendor, "_load_manifest", return_value=_FAKE_MANIFEST),
            patch.object(vendor, "_sha256", side_effect=fake_sha),
        ):
            ok, problems = vendor.verify_all(base_dir=tmp_path)

        assert any(p.endswith("a.js") for p in ok)
        assert problems == ["CHECKSUM MISMATCH: static/fonts/f.woff2"]

    def test_reports_missing_file(self, tmp_path):
        with patch.object(vendor, "_load_manifest", return_value=_FAKE_MANIFEST):
            ok, problems = vendor.verify_all(base_dir=tmp_path)
        assert ok == []
        assert "MISSING: static/vendor/a.js" in problems
        assert "MISSING: static/fonts/f.woff2" in problems


class TestSha256:
    def test_hashes_file_contents(self, tmp_path):
        import hashlib

        f = tmp_path / "blob.bin"
        payload = b"vendored-bytes" * 1000  # exercise the chunked read loop
        f.write_bytes(payload)
        assert vendor._sha256(f) == hashlib.sha256(payload).hexdigest()

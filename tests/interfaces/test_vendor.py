"""Tests for vendor helpers and offline-mode switching."""

from __future__ import annotations

import hashlib
import re
import ssl
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import osprey.interfaces.vendor as vendor_mod
from osprey.interfaces.vendor import (
    MANIFEST_PATH,
    _fetch_one,
    _load_manifest,
    asset_cdn_url,
    html_has_plotly_bundle,
    html_needs_plotly,
    is_offline,
    vendor_url,
)


@contextmanager
def _scoped_sleep():
    """Mock vendor.py's backoff sleep WITHOUT touching the stdlib.

    `patch("osprey.interfaces.vendor.time.sleep")` would mutate the shared
    `time` module (`vendor_mod.time is time`), so any daemon thread left
    sleeping in a loop by a neighboring test in the same xdist worker
    inflates the mock's call_count by thousands. Rebinding the importing
    module's own `time` name to a stand-in is the actually-scoped spelling.
    """
    stand_in = SimpleNamespace(sleep=MagicMock())
    with patch.object(vendor_mod, "time", stand_in):
        yield stand_in.sleep


class TestIsOffline:
    def test_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            assert is_offline() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_env_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("OSPREY_OFFLINE", val)
        assert is_offline() is True

    @pytest.mark.parametrize("val", ["0", "false", "FALSE", "no", "off"])
    def test_env_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("OSPREY_OFFLINE", val)
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"offline": True},
        ):
            # Env var wins: falsy env overrides config.yml's truthy setting
            assert is_offline() is False

    def test_config_yml_truthy_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"offline": True},
        ):
            assert is_offline() is True

    def test_empty_env_falls_through_to_config(self, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "")
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"offline": True},
        ):
            assert is_offline() is True

    def test_config_load_failure_returns_false(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            side_effect=RuntimeError("boom"),
        ):
            assert is_offline() is False


PAYLOAD = b"console.log('vendored');"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _ok_response():
    """A context-manager stand-in for urlopen's response."""
    resp = MagicMock()
    resp.read.return_value = PAYLOAD
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: False
    return resp


class TestFetchOneRetry:
    """_fetch_one runs at image-build time against a public CDN. A single
    transient network error must not fail the build, but a deterministic
    error must fail immediately rather than burn the retry budget.
    """

    def test_retries_transient_error_then_succeeds(self, tmp_path):
        dest = tmp_path / "highlight.min.js"
        transient = urllib.error.URLError(OSError(101, "Network is unreachable"))
        with patch("osprey.interfaces.vendor.urllib.request.urlopen") as uo:
            uo.side_effect = [transient, _ok_response()]
            with _scoped_sleep() as slept:
                _fetch_one("https://cdn.example/x.js", dest, PAYLOAD_SHA)
        assert dest.read_bytes() == PAYLOAD
        assert uo.call_count == 2
        assert slept.call_count == 1

    def test_gives_up_after_all_attempts(self, tmp_path):
        dest = tmp_path / "x.js"
        transient = urllib.error.URLError(OSError(101, "Network is unreachable"))
        with patch("osprey.interfaces.vendor.urllib.request.urlopen") as uo:
            uo.side_effect = transient
            with _scoped_sleep():
                with pytest.raises(RuntimeError, match="after 3 attempts"):
                    _fetch_one("https://cdn.example/x.js", dest, PAYLOAD_SHA)
        assert uo.call_count == 3
        assert not dest.exists()

    def test_http_5xx_is_retried(self, tmp_path):
        dest = tmp_path / "x.js"
        err = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
        with patch("osprey.interfaces.vendor.urllib.request.urlopen") as uo:
            uo.side_effect = [err, _ok_response()]
            with _scoped_sleep():
                _fetch_one("https://cdn.example/x.js", dest, PAYLOAD_SHA)
        assert uo.call_count == 2

    def test_http_404_is_not_retried(self, tmp_path):
        dest = tmp_path / "x.js"
        err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        with patch("osprey.interfaces.vendor.urllib.request.urlopen") as uo:
            uo.side_effect = err
            with _scoped_sleep() as slept:
                with pytest.raises(RuntimeError, match="Failed to fetch"):
                    _fetch_one("https://cdn.example/x.js", dest, PAYLOAD_SHA)
        assert uo.call_count == 1
        assert slept.call_count == 0

    def test_cert_error_is_not_retried_and_keeps_guidance(self, tmp_path):
        dest = tmp_path / "x.js"
        cert = ssl.SSLCertVerificationError("bad cert")
        with patch("osprey.interfaces.vendor.urllib.request.urlopen") as uo:
            uo.side_effect = urllib.error.URLError(cert)
            with _scoped_sleep() as slept:
                with pytest.raises(RuntimeError, match="TLS cert verification failed"):
                    _fetch_one("https://cdn.example/x.js", dest, PAYLOAD_SHA)
        assert uo.call_count == 1
        assert slept.call_count == 0

    def test_sha_mismatch_is_not_retried_and_removes_file(self, tmp_path):
        dest = tmp_path / "x.js"
        with patch("osprey.interfaces.vendor.urllib.request.urlopen") as uo:
            uo.side_effect = [_ok_response()]
            with _scoped_sleep() as slept:
                with pytest.raises(RuntimeError, match="SHA256 mismatch"):
                    _fetch_one("https://cdn.example/x.js", dest, "deadbeef" * 8)
        assert uo.call_count == 1
        assert slept.call_count == 0
        assert not dest.exists()


class TestAssetCdnUrl:
    def test_known_names(self):
        assert asset_cdn_url("xterm.js").startswith("https://cdn.jsdelivr.net/")
        assert asset_cdn_url("Plotly.js").startswith("https://cdn.plot.ly/")
        assert asset_cdn_url("KaTeX JS").endswith("katex.min.js")

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown vendor asset name"):
            asset_cdn_url("does-not-exist")


class TestVendorUrl:
    def test_offline_returns_local_path(self, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert vendor_url("xterm.js", "/static/vendor/xterm.min.js") == (
            "/static/vendor/xterm.min.js"
        )

    def test_online_returns_cdn_url(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            url = vendor_url("xterm.js", "/static/vendor/xterm.min.js")
        assert url.startswith("https://cdn.jsdelivr.net/")
        assert url.endswith("xterm.min.js")

    def test_online_unknown_name_raises(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            with pytest.raises(KeyError):
                vendor_url("nope", "/static/vendor/nope.js")

    def test_offline_unknown_name_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert vendor_url("nope", "/local/path.js") == "/local/path.js"


class TestWebTerminalRendering:
    """End-to-end: GET the web_terminal index route in both modes and
    assert the rendered HTML contains the expected src values."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import create_app

        ws = tmp_path / "_agent_data"
        ws.mkdir()
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(ws)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                yield c

    def test_cdn_mode_uses_jsdelivr(self, client, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        assert "cdn.jsdelivr.net" in body
        assert 'src="/static/vendor/xterm.min.js"' not in body

    def test_offline_mode_uses_local_paths(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        assert "/static/vendor/xterm.min.js" in body
        assert "cdn.jsdelivr.net" not in body


# --- vendored filename drift ------------------------------------------------

#: Sources that can name a vendored bundle. The vendored payloads themselves are
#: excluded: a minified bundle is one enormous line in which a filename-shaped
#: token is coincidence, and none of it is ours to reword.
_INTERFACES_ROOT = Path(vendor_mod.__file__).parent
_SCAN_SUFFIXES = (".py", ".js", ".html")

#: A bundle filename as it appears in source — basename only, no directory part.
_BUNDLE_TOKEN_RE = re.compile(r"[\w.@-]+\.(?:min\.)?(?:js|css)\b")

#: The version segment of a vendored filename: ``plotly-3.3.1.min.js`` and
#: ``plotly-4.0.0.min.js`` are the same asset at two versions.
_VERSION_SEGMENT_RE = re.compile(r"-\d+(?:\.\d+)*(?=\.)")


def _asset_family(filename: str) -> str:
    """The version-free identity of a vendored filename."""
    return _VERSION_SEGMENT_RE.sub("", filename)


def _scanned_sources() -> list[Path]:
    return [
        path
        for path in sorted(_INTERFACES_ROOT.rglob("*"))
        if path.suffix in _SCAN_SUFFIXES and "vendor" not in path.parts and ".min." not in path.name
    ]


def test_every_vendored_filename_literal_matches_the_manifest() -> None:
    """A filename carrying a version is a copy of the manifest, and copies rot.

    ``vendor_url()`` takes the local path from its caller and the panel JS has
    no manifest access at all, so these literals cannot be re-plumbed away. What
    they can have is a lint: bump an asset in ``vendor_manifest.json`` and every
    literal still naming the old file is named here rather than 404-ing in the
    one deployment that serves vendored assets (offline builds).
    """
    by_family = {_asset_family(a["filename"]): a["filename"] for a in _load_manifest()["assets"]}
    assert len(by_family) == len(_load_manifest()["assets"]), (
        f"two assets in {MANIFEST_PATH.name} share a version-free filename"
    )

    stale: list[str] = []
    for path in _scanned_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for token in _BUNDLE_TOKEN_RE.findall(line):
                expected = by_family.get(_asset_family(token))
                if expected is not None and token != expected:
                    rel = path.relative_to(_INTERFACES_ROOT)
                    stale.append(f"{rel}:{lineno} names {token}, manifest has {expected}")

    assert not stale, (
        "vendored filename literals out of step with vendor_manifest.json:\n  " + "\n  ".join(stale)
    )


# ---------------------------------------------------------------------------
# Plotly bundle detection
# ---------------------------------------------------------------------------

_PLOT_CALL = b'<div id="p"></div><script>Plotly.newPlot("p", [], {});</script>'


def _page(head: bytes = b"", body: bytes = _PLOT_CALL) -> bytes:
    return b"<html><head>" + head + b"</head><body>" + body + b"</body></html>"


def test_html_has_plotly_bundle_false_without_script() -> None:
    assert html_has_plotly_bundle(_page()) is False


def test_html_has_plotly_bundle_detects_cdn_script() -> None:
    html = _page(b'<script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>')
    assert html_has_plotly_bundle(html) is True


def test_html_has_plotly_bundle_detects_other_cdn_version() -> None:
    html = _page(
        b"<script charset='utf-8' src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>"
    )
    assert html_has_plotly_bundle(html) is True


def test_html_has_plotly_bundle_detects_unversioned_cdn() -> None:
    html = _page(
        b'<script type="text/javascript" src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    )
    assert html_has_plotly_bundle(html) is True


def test_html_has_plotly_bundle_detects_vendored_script() -> None:
    html = _page(b'<script src="/static/js/vendor/plotly-3.3.1.min.js"></script>')
    assert html_has_plotly_bundle(html) is True


def test_html_has_plotly_bundle_detects_relative_vendored_script() -> None:
    html = _page(b'<SCRIPT SRC="../vendor/plotly.min.js"></SCRIPT>')
    assert html_has_plotly_bundle(html) is True


def test_html_has_plotly_bundle_detects_inlined_bundle() -> None:
    inline = b"<script>/**\n* plotly.js v3.3.1\n* Copyright 2012-2025, Plotly, Inc.\n*/ !function(){}</script>"
    assert html_has_plotly_bundle(_page(inline)) is True


def test_html_has_plotly_bundle_ignores_non_plotly_script() -> None:
    html = _page(b'<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>')
    assert html_has_plotly_bundle(html) is False


def test_html_has_plotly_bundle_ignores_plotly_word_outside_src() -> None:
    html = _page(
        b'<script src="/app.js" data-note="plotly"></script>', b"<p>plotly.js is great</p>"
    )
    assert html_has_plotly_bundle(html) is False


def test_html_needs_plotly_true_when_call_has_no_bundle() -> None:
    assert html_needs_plotly(_page()) is True


def test_html_needs_plotly_false_with_cdn_bundle() -> None:
    html = _page(b'<script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>')
    assert html_needs_plotly(html) is False


def test_html_needs_plotly_false_with_inlined_bundle() -> None:
    html = _page(b"<script>/** plotly.js v2.35.2 */</script>")
    assert html_needs_plotly(html) is False


def test_html_needs_plotly_false_with_vendored_bundle() -> None:
    html = _page(b'<script src="/static/js/vendor/plotly-3.3.1.min.js"></script>')
    assert html_needs_plotly(html) is False


def test_html_needs_plotly_false_without_plotly_call() -> None:
    assert html_needs_plotly(_page(body=b"<p>No figures here.</p>")) is False


def test_html_needs_plotly_false_for_empty_html() -> None:
    assert html_needs_plotly(b"") is False

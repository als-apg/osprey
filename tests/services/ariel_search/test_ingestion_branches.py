"""Branch coverage for the four ARIEL ingestion adapters.

Companion to ``test_ingestion.py``, which pins the happy paths. This file pins
the branches that only run when a logbook misbehaves: HTTP failures, retries,
unparseable payloads, missing files, and the tolerance fallbacks that let a
malformed entry through instead of stopping the ingest.

Two conventions carry through the file:

* **No network.** Every HTTP path goes through ``patch("aiohttp.ClientSession")``,
  following the precedent already in ``test_ingestion.py``.
* **No real sleeping.** Retry backoff is observed by rebinding the adapter
  module's own ``asyncio`` name to :class:`_RecordingAsyncio`. Rebinding the
  module attribute (rather than patching ``asyncio.sleep`` itself) keeps the
  fake from leaking into every other coroutine in the process.
"""

import json
import logging
import ssl
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import aiohttp
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import (
    AuthenticationRequiredError,
    IngestionError,
)
from osprey.services.ariel_search.ingestion.adapters import als as als_module
from osprey.services.ariel_search.ingestion.adapters import generic as generic_module
from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter
from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter
from osprey.services.ariel_search.ingestion.adapters.jlab import JLabLogbookAdapter
from osprey.services.ariel_search.ingestion.adapters.ornl import ORNLLogbookAdapter
from osprey.services.ariel_search.models import FacilityEntryCreateRequest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _RecordingAsyncio:
    """Scoped stand-in for the ``asyncio`` name inside an adapter module.

    ``sleep`` records its delay and returns immediately, so retry backoff is
    asserted rather than waited out. Every other attribute delegates to the real
    module. Install with ``monkeypatch.setattr(<module>, "asyncio", shim)`` —
    patching ``asyncio.sleep`` directly would be process-global.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)

    def __getattr__(self, name: str) -> Any:
        import asyncio

        return getattr(asyncio, name)


def _http_response(
    *,
    status: int = 200,
    text: str | None = None,
    json_data: Any = None,
    raise_for_status: Exception | None = None,
) -> AsyncMock:
    """Build an aiohttp response double usable as an async context manager."""
    response = AsyncMock()
    response.status = status
    response.raise_for_status = MagicMock(side_effect=raise_for_status)
    response.text = AsyncMock(return_value=text if text is not None else "")
    response.json = AsyncMock(return_value=json_data)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


def _request_mock(outcome: Any) -> MagicMock:
    """Turn a response double, an exception, or a per-attempt list into a call mock."""
    if isinstance(outcome, (list, Exception)):
        return MagicMock(side_effect=outcome)
    return MagicMock(return_value=outcome)


def _http_session(*, get: Any = None, post: Any = None) -> MagicMock:
    """Build an aiohttp session double.

    ``get``/``post`` take a response double, an exception instance (raised on
    every attempt, modelling a persistent connection failure), or a list used as
    a ``side_effect`` to script one outcome per attempt.
    """
    session = MagicMock()
    if get is not None:
        session.get = _request_mock(get)
    if post is not None:
        session.post = _request_mock(post)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@contextmanager
def _patched_session(adapter: Any, session: MagicMock):
    """Route an adapter's HTTP calls to ``session`` and stub out the connector.

    The connector is stubbed because a real ``TCPConnector`` built against a
    mocked session is never closed by anything.
    """
    with (
        patch("aiohttp.ClientSession", return_value=session),
        patch.object(adapter, "_create_connector", return_value=MagicMock()),
    ):
        yield


@pytest.fixture(autouse=True)
def _clear_ariel_env(monkeypatch):
    """Clear the ARIEL_* environment fallbacks read by config/adapter construction.

    ``WriteConfig.from_dict`` and ``IngestionConfig.from_dict`` fold these into
    the config object, so a developer shell that exports them would otherwise
    decide which branch a test takes. Tests that exercise the fallbacks set them
    back explicitly.
    """
    for var in ("ARIEL_SOCKS_PROXY", "ARIEL_WRITE_USER", "ARIEL_WRITE_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


def _config(adapter: str, source_url: str | None, **ingestion: Any) -> ARIELConfig:
    """Build an ARIELConfig with one ingestion block."""
    block: dict[str, Any] = {"adapter": adapter}
    if source_url is not None:
        block["source_url"] = source_url
    block.update(ingestion)
    return ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}, "ingestion": block})


def _als_adapter(
    source_url: str = "https://web7.als.invalid/olog/rpc.php", **ingestion: Any
) -> ALSLogbookAdapter:
    return ALSLogbookAdapter(_config("als_logbook", source_url, **ingestion))


def _als_write_adapter(
    *,
    auth: bool = True,
    **ingestion: Any,
) -> ALSLogbookAdapter:
    """ALS adapter with write enabled, optionally carrying config credentials."""
    write: dict[str, Any] = {
        "enabled": True,
        "write_url": "https://elog.als.invalid/olog/rpc.php",
    }
    if auth:
        write["auth_user"] = "operator"
        write["auth_password"] = "hunter2"
    return _als_adapter(write=write, **ingestion)


def _generic_adapter(source_url: str) -> GenericJSONAdapter:
    return GenericJSONAdapter(_config("generic_json", source_url))


def _jlab_adapter(source_url: str) -> JLabLogbookAdapter:
    return JLabLogbookAdapter(_config("jlab_logbook", source_url))


def _ornl_adapter(source_url: str) -> ORNLLogbookAdapter:
    return ORNLLogbookAdapter(_config("ornl_logbook", source_url))


async def _collect(agen: Any) -> list[Any]:
    return [item async for item in agen]


# ---------------------------------------------------------------------------
# ALS — construction and configuration guards
# ---------------------------------------------------------------------------


class TestALSConfigGuards:
    """Construction guards and environment fallbacks for the ALS adapter."""

    def test_missing_source_url_raises(self):
        """source_url is mandatory: the adapter has nothing to read otherwise."""
        with pytest.raises(IngestionError) as exc_info:
            ALSLogbookAdapter(_config("als_logbook", None))

        assert "source_url is required" in str(exc_info.value)
        assert exc_info.value.source_system == "ALS eLog"

    def test_supports_write_false_when_ingestion_block_absent(self):
        """supports_write is fail-closed when the config carries no ingestion block."""
        adapter = _als_write_adapter()
        assert adapter.supports_write is True

        adapter.config = ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}})
        assert adapter.supports_write is False

    def test_socks_proxy_read_from_environment(self, monkeypatch):
        """ARIEL_SOCKS_PROXY supplies the proxy when the config omits proxy_url."""
        monkeypatch.setenv("ARIEL_SOCKS_PROXY", "socks5://tunnel.invalid:9095")

        adapter = _als_adapter()

        assert adapter.proxy_url == "socks5://tunnel.invalid:9095"

    def test_no_proxy_without_config_or_environment(self):
        """With neither source, proxy_url stays None and a plain TCP connector is used."""
        adapter = _als_adapter()
        assert adapter.proxy_url is None

    def test_explicit_proxy_wins_over_environment(self, monkeypatch):
        """An explicit proxy_url is not overridden by the environment fallback."""
        monkeypatch.setenv("ARIEL_SOCKS_PROXY", "socks5://tunnel.invalid:9095")

        adapter = _als_adapter(proxy_url="socks5://explicit.invalid:1080")

        assert adapter.proxy_url == "socks5://explicit.invalid:1080"


# ---------------------------------------------------------------------------
# ALS — file-backed source
# ---------------------------------------------------------------------------


class TestALSFileSource:
    """JSONL file mode: per-line tolerance and the missing-file guard."""

    @pytest.mark.asyncio
    async def test_http_source_delegates_to_http_reader(self):
        """An http(s) source_url routes fetch_entries to the windowed HTTP reader."""
        adapter = _als_adapter()
        seen: dict[str, Any] = {}

        async def fake_http(since, until, limit):
            seen["args"] = (since, until, limit)
            yield {"entry_id": "from-http"}

        with patch.object(adapter, "_fetch_entries_http", fake_http):
            entries = await _collect(adapter.fetch_entries(limit=7))

        assert [e["entry_id"] for e in entries] == ["from-http"]
        assert seen["args"] == (None, None, 7)

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        """A file source that does not exist fails loudly rather than yielding nothing."""
        adapter = _als_adapter(str(tmp_path / "absent.jsonl"))

        with pytest.raises(IngestionError) as exc_info:
            await _collect(adapter.fetch_entries())

        assert "Source file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_blank_lines_and_empty_entries_are_skipped(self, tmp_path):
        """Blank lines and text-free entries never reach the database."""
        path = tmp_path / "entries.jsonl"
        path.write_text(
            "\n".join(
                [
                    "",
                    json.dumps(
                        {"id": "1", "timestamp": "1704067200", "subject": "", "details": ""}
                    ),
                    "   ",
                    json.dumps(
                        {"id": "2", "timestamp": "1704067200", "subject": "Kept", "details": ""}
                    ),
                ]
            )
            + "\n"
        )
        adapter = _als_adapter(str(path))

        entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["2"]

    @pytest.mark.asyncio
    async def test_since_and_until_bounds_are_exclusive(self, tmp_path):
        """Entries exactly on either bound are dropped (``<=`` since, ``>=`` until)."""
        path = tmp_path / "entries.jsonl"
        stamps = {
            "early": int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
            "middle": int(datetime(2024, 1, 5, tzinfo=UTC).timestamp()),
            "late": int(datetime(2024, 1, 9, tzinfo=UTC).timestamp()),
        }
        path.write_text(
            "\n".join(
                json.dumps({"id": name, "timestamp": str(ts), "subject": name, "details": ""})
                for name, ts in stamps.items()
            )
        )
        adapter = _als_adapter(str(path))

        entries = await _collect(
            adapter.fetch_entries(
                since=datetime(2024, 1, 1, tzinfo=UTC),
                until=datetime(2024, 1, 9, tzinfo=UTC),
            )
        )

        assert [e["entry_id"] for e in entries] == ["middle"]

    @pytest.mark.asyncio
    async def test_limit_stops_reading(self, tmp_path):
        """limit stops the read mid-file instead of converting the whole export."""
        path = tmp_path / "entries.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"id": str(i), "timestamp": "1704067200", "subject": f"e{i}"})
                for i in range(5)
            )
        )
        adapter = _als_adapter(str(path))

        entries = await _collect(adapter.fetch_entries(limit=2))

        assert [e["entry_id"] for e in entries] == ["0", "1"]

    @pytest.mark.asyncio
    async def test_invalid_json_line_is_logged_and_skipped(self, tmp_path, caplog):
        """One corrupt line does not abort the ingest of the rest of the file."""
        path = tmp_path / "entries.jsonl"
        path.write_text(
            "{not json\n"
            + json.dumps({"id": "ok", "timestamp": "1704067200", "subject": "Fine"})
            + "\n"
        )
        adapter = _als_adapter(str(path))

        with caplog.at_level(logging.WARNING, logger="ariel"):
            entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["ok"]
        assert "Invalid JSON at line 1" in caplog.text

    @pytest.mark.asyncio
    async def test_unconvertible_line_is_logged_and_skipped(self, tmp_path, caplog):
        """A syntactically valid line of the wrong shape is skipped, not fatal."""
        path = tmp_path / "entries.jsonl"
        path.write_text(
            "[1, 2, 3]\n"
            + json.dumps({"id": "ok", "timestamp": "1704067200", "subject": "Fine"})
            + "\n"
        )
        adapter = _als_adapter(str(path))

        with caplog.at_level(logging.WARNING, logger="ariel"):
            entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["ok"]
        assert "Failed to convert entry at line 1" in caplog.text


# ---------------------------------------------------------------------------
# ALS — write path (_post_entry)
# ---------------------------------------------------------------------------


class TestALSPostEntry:
    """The olog RPC write: TLS mode, error surfacing, ID parsing, and retries."""

    @pytest.mark.asyncio
    async def test_verify_ssl_true_passes_certificate_verification(self):
        """verify_ssl=True hands aiohttp its default verifying context."""
        adapter = _als_write_adapter(verify_ssl=True)
        session = _http_session(post=_http_response(text="<response><id>7</id></response>"))

        with _patched_session(adapter, session):
            await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert session.post.call_args.kwargs["ssl"] is True

    @pytest.mark.asyncio
    async def test_verify_ssl_false_builds_permissive_context(self):
        """verify_ssl=False (the default) disables hostname and certificate checks.

        Internal olog servers carry self-signed certificates; the default is
        permissive so ingestion works, and this pins exactly how permissive.
        """
        adapter = _als_write_adapter()
        session = _http_session(post=_http_response(text="<response><id>7</id></response>"))

        with _patched_session(adapter, session):
            await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        context = session.post.call_args.kwargs["ssl"]
        assert isinstance(context, ssl.SSLContext)
        assert context.check_hostname is False
        assert context.verify_mode == ssl.CERT_NONE

    @pytest.mark.asyncio
    async def test_http_error_carries_response_body(self, monkeypatch):
        """A >=400 response puts the server's body in the error, and is not retried.

        The olog RPC explains rejections in the body (bad credentials, unknown
        logbook), so discarding it would leave the operator with a bare status code.
        """
        adapter = _als_write_adapter()
        shim = _RecordingAsyncio()
        monkeypatch.setattr(als_module, "asyncio", shim)
        session = _http_session(
            post=_http_response(status=403, text="unknown logbook 'Operations'")
        )

        with _patched_session(adapter, session):
            with pytest.raises(IngestionError) as exc_info:
                await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert "HTTP 403" in str(exc_info.value)
        assert "unknown logbook 'Operations'" in str(exc_info.value)
        assert session.post.call_count == 1
        assert shim.delays == []

    @pytest.mark.asyncio
    async def test_entry_id_from_id_element(self):
        """The nested ``<id>`` element is the primary source of the entry ID."""
        adapter = _als_write_adapter()
        session = _http_session(
            post=_http_response(text="<response><entry><id>4242</id></entry></response>")
        )

        with _patched_session(adapter, session):
            entry_id = await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert entry_id == "4242"

    @pytest.mark.asyncio
    async def test_entry_id_from_root_attribute(self):
        """With no ``<id>`` element, a root ``id`` attribute is the next fallback."""
        adapter = _als_write_adapter()
        session = _http_session(post=_http_response(text='<response id="4243" />'))

        with _patched_session(adapter, session):
            entry_id = await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert entry_id == "4243"

    @pytest.mark.asyncio
    async def test_unparseable_response_falls_back_to_response_text(self):
        """HAZARD: a non-XML success body becomes the entry ID verbatim.

        The write succeeded, so the adapter refuses to fail; the cost is that a
        plain-text or HTML body is stored as the facility entry ID and will not
        match anything on re-ingestion.
        """
        adapter = _als_write_adapter()
        session = _http_session(post=_http_response(text="  saved ok  "))

        with _patched_session(adapter, session):
            entry_id = await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert entry_id == "saved ok"

    @pytest.mark.asyncio
    async def test_empty_id_element_falls_back_to_whole_body(self):
        """HAZARD: an ``<id>`` element with no text yields the raw XML body as the ID.

        ``find(".//id")`` matches but ``.text`` is None, so neither the element
        nor the attribute fallback fires and the whole response becomes the ID.
        """
        adapter = _als_write_adapter()
        body = "<response><id></id></response>"
        session = _http_session(post=_http_response(text=body))

        with _patched_session(adapter, session):
            entry_id = await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert entry_id == body

    @pytest.mark.asyncio
    async def test_blank_response_gets_synthetic_id(self):
        """An empty success body yields a synthetic ``als-<epoch>`` identifier."""
        adapter = _als_write_adapter()
        session = _http_session(post=_http_response(text="   "))

        with _patched_session(adapter, session):
            entry_id = await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert entry_id.startswith("als-")

    @pytest.mark.asyncio
    async def test_transient_failure_retries_with_backoff(self, monkeypatch):
        """A connection error is retried after ``retry_delay * 2**attempt`` seconds."""
        adapter = _als_write_adapter(max_retries=3, retry_delay_seconds=5)
        shim = _RecordingAsyncio()
        monkeypatch.setattr(als_module, "asyncio", shim)
        session = _http_session(
            post=[
                aiohttp.ClientError("connection reset"),
                TimeoutError("read timeout"),
                _http_response(text="<response><id>99</id></response>"),
            ]
        )

        with _patched_session(adapter, session):
            entry_id = await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert entry_id == "99"
        assert shim.delays == [5, 10]

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_names_last_error(self, monkeypatch):
        """After the final attempt the adapter raises, naming the last failure.

        The last attempt does not sleep — the retry budget is spent, so waiting
        would only delay the error.
        """
        adapter = _als_write_adapter(max_retries=2, retry_delay_seconds=5)
        shim = _RecordingAsyncio()
        monkeypatch.setattr(als_module, "asyncio", shim)
        session = _http_session(post=aiohttp.ClientError("connection reset"))

        with _patched_session(adapter, session):
            with pytest.raises(IngestionError) as exc_info:
                await adapter._post_entry("https://elog.als.invalid/olog", "<root/>")

        assert "Max retries (2) exceeded" in str(exc_info.value)
        assert "connection reset" in str(exc_info.value)
        assert session.post.call_count == 2
        assert shim.delays == [5]


class TestALSCreateEntryCredentials:
    """Credential resolution for the write path."""

    @pytest.mark.asyncio
    async def test_environment_credentials_are_folded_in_at_config_load(self, monkeypatch):
        """ARIEL_WRITE_USER/PASSWORD supply credentials when the request omits them."""
        monkeypatch.setenv("ARIEL_WRITE_USER", "env-operator")
        monkeypatch.setenv("ARIEL_WRITE_PASSWORD", "env-secret")
        adapter = _als_write_adapter(auth=False)
        session = _http_session(post=_http_response(text="<response><id>1</id></response>"))

        with _patched_session(adapter, session):
            await adapter.create_entry(
                FacilityEntryCreateRequest(subject="Subject", details="Details")
            )

        payload = session.post.call_args.kwargs["data"]
        assert "<author>env-operator</author>" in payload
        assert "<password>env-secret</password>" in payload

    @pytest.mark.asyncio
    async def test_environment_is_read_at_config_load_not_at_write(self, monkeypatch):
        """HAZARD: credentials exported after the config was built are not picked up.

        ``WriteConfig.from_dict`` snapshots the environment, so a service that
        loaded its config before the credentials were exported keeps refusing to
        publish until it is restarted.
        """
        adapter = _als_write_adapter(auth=False)
        monkeypatch.setenv("ARIEL_WRITE_USER", "late-operator")
        monkeypatch.setenv("ARIEL_WRITE_PASSWORD", "late-secret")

        with pytest.raises(AuthenticationRequiredError):
            await adapter.create_entry(
                FacilityEntryCreateRequest(subject="Subject", details="Details")
            )

    @pytest.mark.asyncio
    async def test_request_credentials_override_config(self):
        """Per-request credentials take precedence over the configured ones."""
        adapter = _als_write_adapter()
        session = _http_session(post=_http_response(text="<response><id>1</id></response>"))

        with _patched_session(adapter, session):
            await adapter.create_entry(
                FacilityEntryCreateRequest(
                    subject="Subject",
                    details="Details",
                    auth_user="request-user",
                    auth_password="request-secret",
                )
            )

        payload = session.post.call_args.kwargs["data"]
        assert "<author>request-user</author>" in payload

    @pytest.mark.asyncio
    async def test_missing_credentials_raise_authentication_required(self):
        """Absent credentials raise the auth-specific error, not a generic one.

        The API layer maps this to HTTP 401 so the operator is prompted, rather
        than the publish silently failing.
        """
        adapter = _als_write_adapter(auth=False)

        with pytest.raises(AuthenticationRequiredError) as exc_info:
            await adapter.create_entry(
                FacilityEntryCreateRequest(subject="Subject", details="Details")
            )

        assert "ARIEL_WRITE_USER" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ALS — windowed HTTP read
# ---------------------------------------------------------------------------


class TestALSFetchEntriesHTTP:
    """Window generation, TLS mode, and per-window error handling."""

    @pytest.mark.asyncio
    async def test_naive_bounds_are_interpreted_as_utc(self):
        """Timezone-naive since/until are treated as UTC before windowing."""
        adapter = _als_adapter()
        session = _http_session(get=_http_response(text="[]"))

        with _patched_session(adapter, session):
            entries = await _collect(
                adapter._fetch_entries_http(since=datetime(2024, 1, 1), until=datetime(2024, 1, 2))
            )

        assert entries == []
        params = session.get.call_args.kwargs["params"]
        assert params["start"] == str(int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()))
        assert params["end"] == str(int(datetime(2024, 1, 2, tzinfo=UTC).timestamp()))

    @pytest.mark.asyncio
    async def test_verify_ssl_true_passes_certificate_verification(self):
        """verify_ssl=True hands aiohttp its default verifying context on reads too."""
        adapter = _als_adapter(verify_ssl=True)
        session = _http_session(get=_http_response(text="[]"))

        with _patched_session(adapter, session):
            await _collect(
                adapter._fetch_entries_http(
                    since=datetime(2024, 1, 1, tzinfo=UTC), until=datetime(2024, 1, 2, tzinfo=UTC)
                )
            )

        assert session.get.call_args.kwargs["ssl"] is True

    @pytest.mark.asyncio
    async def test_ingestion_error_from_a_window_aborts_the_fetch(self):
        """An IngestionError (exhausted retries, 4xx) stops the whole fetch."""
        adapter = _als_adapter()
        session = _http_session(get=_http_response(text="[]"))
        failure = IngestionError("HTTP 404 error from ALS API", source_system="ALS eLog")

        with _patched_session(adapter, session):
            with patch.object(adapter, "_fetch_window_with_retry", side_effect=failure):
                with pytest.raises(IngestionError):
                    await _collect(
                        adapter._fetch_entries_http(
                            since=datetime(2024, 1, 1, tzinfo=UTC),
                            until=datetime(2024, 1, 2, tzinfo=UTC),
                        )
                    )

    @pytest.mark.asyncio
    async def test_unexpected_window_error_skips_only_that_window(self, caplog):
        """An unexpected error is logged and the remaining windows still run."""
        adapter = _als_adapter(chunk_days=1)
        session = _http_session(get=_http_response(text="[]"))
        good = [{"id": "5", "timestamp": "1704067200", "subject": "Second window"}]

        with _patched_session(adapter, session):
            with patch.object(
                adapter,
                "_fetch_window_with_retry",
                side_effect=[RuntimeError("adapter bug"), good],
            ):
                with caplog.at_level(logging.ERROR, logger="ariel"):
                    entries = await _collect(
                        adapter._fetch_entries_http(
                            since=datetime(2024, 1, 1, tzinfo=UTC),
                            until=datetime(2024, 1, 3, tzinfo=UTC),
                        )
                    )

        assert [e["entry_id"] for e in entries] == ["5"]
        assert "Failed to fetch window" in caplog.text

    @pytest.mark.asyncio
    async def test_unconvertible_entry_is_logged_and_skipped(self, caplog):
        """One entry the converter chokes on does not lose the rest of the window."""
        adapter = _als_adapter()
        payload = [
            {"id": "bad", "timestamp": "1704067200", "subject": "Bad"},
            {"id": "good", "timestamp": "1704067200", "subject": "Good"},
        ]
        session = _http_session(get=_http_response(text=json.dumps(payload)))
        real_convert = adapter._convert_entry

        def convert(data):
            if data["id"] == "bad":
                raise ValueError("unconvertible")
            return real_convert(data)

        with _patched_session(adapter, session):
            with patch.object(adapter, "_convert_entry", side_effect=convert):
                with caplog.at_level(logging.WARNING, logger="ariel"):
                    entries = await _collect(
                        adapter._fetch_entries_http(
                            since=datetime(2024, 1, 1, tzinfo=UTC),
                            until=datetime(2024, 1, 2, tzinfo=UTC),
                        )
                    )

        assert [e["entry_id"] for e in entries] == ["good"]
        assert "Failed to convert entry bad" in caplog.text


class TestALSFetchWindowResponses:
    """The olog endpoint answers with several non-JSON shapes; all mean 'empty'."""

    def _adapter(self) -> ALSLogbookAdapter:
        return _als_adapter()

    async def _fetch(self, body: str) -> list[dict[str, Any]]:
        adapter = self._adapter()
        session = _http_session(get=_http_response(text=body))
        return await adapter._fetch_window(
            session, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC), True
        )

    @pytest.mark.asyncio
    async def test_empty_body_is_empty_window(self):
        """An empty body means no entries in the range, not a failure."""
        assert await self._fetch("   ") == []

    @pytest.mark.asyncio
    async def test_html_body_is_empty_window(self):
        """The endpoint returns an HTML page for an empty date range."""
        assert await self._fetch("<html><body>No entries</body></html>") == []

    @pytest.mark.asyncio
    async def test_doctype_body_is_empty_window(self):
        """A leading doctype declaration is recognised as HTML too."""
        assert await self._fetch("<!DOCTYPE html><html></html>") == []

    @pytest.mark.asyncio
    async def test_non_json_body_is_empty_window(self):
        """A body that is neither HTML nor JSON is treated as an empty window."""
        assert await self._fetch("retrieve: no records") == []

    @pytest.mark.asyncio
    async def test_non_list_json_is_empty_window(self, caplog):
        """A JSON object where a list was expected is dropped with a warning."""
        with caplog.at_level(logging.WARNING, logger="ariel"):
            result = await self._fetch(json.dumps({"error": "bad range"}))

        assert result == []
        assert "Unexpected response type" in caplog.text

    @pytest.mark.asyncio
    async def test_json_list_is_returned(self):
        """A JSON array is returned as-is despite the text/html content type."""
        assert await self._fetch(json.dumps([{"id": "1"}])) == [{"id": "1"}]

    @pytest.mark.asyncio
    async def test_server_error_is_retried(self, monkeypatch):
        """A 5xx is retried, unlike a 4xx which fails immediately."""
        adapter = _als_adapter(max_retries=2, retry_delay_seconds=5)
        shim = _RecordingAsyncio()
        monkeypatch.setattr(als_module, "asyncio", shim)
        server_error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=503, message="Service Unavailable"
        )

        with patch.object(adapter, "_fetch_window", side_effect=[server_error, [{"id": "1"}]]):
            result = await adapter._fetch_window_with_retry(
                MagicMock(),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                True,
            )

        assert result == [{"id": "1"}]
        assert shim.delays == [5]


class TestALSConnector:
    """SOCKS proxy connector construction."""

    @pytest.mark.asyncio
    async def test_proxy_connector_built_from_url(self):
        """A configured SOCKS proxy produces an aiohttp-socks connector."""
        from aiohttp_socks import ProxyConnector

        adapter = _als_adapter(proxy_url="socks5://127.0.0.1:9095")

        connector = adapter._create_connector()
        try:
            assert isinstance(connector, ProxyConnector)
        finally:
            await connector.close()

    def test_proxy_without_aiohttp_socks_raises(self, monkeypatch):
        """A proxy without the optional dependency installed fails with an install hint."""
        adapter = _als_adapter(proxy_url="socks5://127.0.0.1:9095")
        monkeypatch.setitem(sys.modules, "aiohttp_socks", None)

        with pytest.raises(IngestionError) as exc_info:
            adapter._create_connector()

        assert "aiohttp-socks is not installed" in str(exc_info.value)


class TestALSConvertEntry:
    """Field-level tolerance in the ALS converter."""

    def test_malformed_timestamp_falls_back_to_now(self):
        """HAZARD: an unparseable timestamp silently becomes ingest time.

        The entry is kept rather than dropped, so a facility export with broken
        timestamps lands as a block of entries all dated at the moment of the
        ingest run — searchable, but chronologically wrong.
        """
        adapter = _als_adapter()

        entry = adapter._convert_entry(
            {"id": "1", "timestamp": "not-an-epoch", "subject": "Subject"}
        )

        assert entry["timestamp"] == entry["created_at"]

    def test_null_timestamp_falls_back_to_now(self):
        """A null timestamp takes the same ingest-time fallback as a malformed one."""
        adapter = _als_adapter()

        entry = adapter._convert_entry({"id": "1", "timestamp": None, "subject": "Subject"})

        assert entry["timestamp"] == entry["created_at"]

    def test_missing_id_yields_empty_entry_id(self):
        """HAZARD: an entry with no id becomes entry_id '' — and '' is a single row.

        ``enhanced_entries`` upserts ``ON CONFLICT (entry_id)``, so every id-less
        entry in an export overwrites the same row: N entries ingest as one, and
        only the last one survives.
        """
        adapter = _als_adapter()

        first = adapter._convert_entry({"timestamp": "1704067200", "subject": "First"})
        second = adapter._convert_entry({"timestamp": "1704067200", "subject": "Second"})

        assert first["entry_id"] == ""
        assert second["entry_id"] == first["entry_id"]

    def test_explicit_metadata_overrides_derived_fields(self):
        """A ``metadata`` dict on the source entry is merged last and wins."""
        adapter = _als_adapter()

        entry = adapter._convert_entry(
            {
                "id": "1",
                "timestamp": "1704067200",
                "subject": "Derived",
                "level": "Info",
                "metadata": {"subject": "Explicit", "extra": "kept"},
            }
        )

        assert entry["metadata"]["subject"] == "Explicit"
        assert entry["metadata"]["extra"] == "kept"
        assert entry["metadata"]["level"] == "Info"

    def test_zero_sentinels_are_normalised_to_absent(self):
        """ALS writes "0" for an unset tag/link; it must not become metadata."""
        adapter = _als_adapter()

        entry = adapter._convert_entry(
            {"id": "1", "timestamp": "1704067200", "subject": "S", "tag": "0", "linkedto": "0"}
        )

        assert "loto_tag" not in entry["metadata"]
        assert "linked_to" not in entry["metadata"]


# ---------------------------------------------------------------------------
# Generic JSON adapter
# ---------------------------------------------------------------------------


@pytest.fixture
def facility_tz(monkeypatch) -> ZoneInfo:
    """Pin the facility timezone the generic converter anchors relative times in.

    ``_convert_entry`` builds its ``now`` from ``get_facility_timezone()``, so an
    unpinned zone would make the resolved wall-clock time depend on whatever
    config the test host happens to load.
    """
    zone = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(generic_module, "get_facility_timezone", lambda: zone)
    return zone


class TestGenericAdapterGuards:
    """Construction and read-loop guards for the generic JSON adapter."""

    def test_missing_source_url_raises(self):
        """source_url is mandatory for the generic adapter too."""
        with pytest.raises(IngestionError) as exc_info:
            GenericJSONAdapter(_config("generic_json", None))

        assert "source_url is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_since_and_until_bounds_are_exclusive(self, tmp_path, facility_tz):
        """Entries on either bound are dropped, matching the other adapters."""
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                {
                    "entries": [
                        {"id": "early", "timestamp": "2024-01-01T00:00:00Z", "title": "e"},
                        {"id": "middle", "timestamp": "2024-01-05T00:00:00Z", "title": "m"},
                        {"id": "late", "timestamp": "2024-01-09T00:00:00Z", "title": "l"},
                    ]
                }
            )
        )
        adapter = _generic_adapter(str(path))

        entries = await _collect(
            adapter.fetch_entries(
                since=datetime(2024, 1, 1, tzinfo=UTC),
                until=datetime(2024, 1, 9, tzinfo=UTC),
            )
        )

        assert [e["entry_id"] for e in entries] == ["middle"]

    @pytest.mark.asyncio
    async def test_limit_stops_reading(self, tmp_path, facility_tz):
        """limit truncates the yielded entries."""
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                {
                    "entries": [
                        {"id": str(i), "timestamp": "2024-01-05T00:00:00Z", "title": f"e{i}"}
                        for i in range(4)
                    ]
                }
            )
        )
        adapter = _generic_adapter(str(path))

        entries = await _collect(adapter.fetch_entries(limit=2))

        assert [e["entry_id"] for e in entries] == ["0", "1"]

    @pytest.mark.asyncio
    async def test_unparseable_timestamp_drops_the_entry(self, tmp_path, facility_tz, caplog):
        """The generic adapter drops an entry with a bad timestamp, unlike ALS/ORNL.

        ``_parse_timestamp`` raises rather than defaulting to now, and
        ``fetch_entries`` turns that into a warning plus a skipped entry — so a
        malformed export loses rows instead of misdating them.
        """
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                {
                    "entries": [
                        {"id": "bad", "timestamp": "yesterday-ish", "title": "Bad"},
                        {"id": "good", "timestamp": "2024-01-05T00:00:00Z", "title": "Good"},
                    ]
                }
            )
        )
        adapter = _generic_adapter(str(path))

        with caplog.at_level(logging.WARNING, logger="ariel"):
            entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["good"]
        assert "Failed to convert entry" in caplog.text


class TestGenericLoadData:
    """Source loading: local file guards and the HTTP branch."""

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        """A missing local file is an ingestion failure, not an empty result."""
        adapter = _generic_adapter(str(tmp_path / "absent.json"))

        with pytest.raises(IngestionError) as exc_info:
            await adapter._load_data()

        assert "Source file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_source_returns_decoded_json(self):
        """An http(s) source is fetched and decoded as JSON."""
        adapter = _generic_adapter("https://logbook.invalid/entries.json")
        session = _http_session(get=_http_response(json_data={"entries": [{"id": "1"}]}))

        with patch("aiohttp.ClientSession", return_value=session):
            data = await adapter._load_data()

        assert data == {"entries": [{"id": "1"}]}

    @pytest.mark.asyncio
    async def test_http_non_200_raises(self):
        """Any non-200 status fails the load with the status in the message."""
        adapter = _generic_adapter("https://logbook.invalid/entries.json")
        session = _http_session(get=_http_response(status=503))

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(IngestionError) as exc_info:
                await adapter._load_data()

        assert "status 503" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_aiohttp_raises_install_hint(self, monkeypatch):
        """aiohttp is optional for file sources; an HTTP source says how to get it."""
        adapter = _generic_adapter("https://logbook.invalid/entries.json")
        monkeypatch.setitem(sys.modules, "aiohttp", None)

        with pytest.raises(IngestionError) as exc_info:
            await adapter._load_data()

        assert "pip install aiohttp" in str(exc_info.value)


class TestGenericAppendEntry:
    """The local-file write path and its failure handling."""

    @pytest.mark.asyncio
    async def test_corrupt_target_file_raises(self, tmp_path):
        """A corrupt target file fails the write instead of silently rewriting it."""
        path = tmp_path / "entries.json"
        path.write_text("{ this is not json")
        adapter = _generic_adapter(str(path))

        with pytest.raises(IngestionError) as exc_info:
            await adapter.create_entry(
                FacilityEntryCreateRequest(subject="Subject", details="Details")
            )

        assert "Failed to append entry" in str(exc_info.value)
        assert path.read_text() == "{ this is not json"

    @pytest.mark.asyncio
    async def test_unreadable_target_raises(self, tmp_path):
        """An OSError reading the target (here: a directory) is wrapped, not leaked."""
        path = tmp_path / "entries.json"
        path.mkdir()
        adapter = _generic_adapter(str(path))

        with pytest.raises(IngestionError) as exc_info:
            await adapter.create_entry(
                FacilityEntryCreateRequest(subject="Subject", details="Details")
            )

        assert "Failed to append entry" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_failed_write_removes_the_temp_file(self, tmp_path, monkeypatch):
        """A failure mid-write cleans up the temp file and leaves the target intact.

        The write is staged through ``<name>.tmp`` and promoted with an atomic
        replace, so a crash must not leave the staging file behind for the next
        run to trip over.
        """
        path = tmp_path / "entries.json"
        path.write_text(json.dumps({"entries": []}))
        adapter = _generic_adapter(str(path))

        def failing_dump(*args, **kwargs):
            raise OSError("no space left on device")

        # Rebind the module's own ``json`` name so the fake cannot escape this module.
        monkeypatch.setattr(
            generic_module,
            "json",
            SimpleNamespace(
                load=json.load, dump=failing_dump, JSONDecodeError=json.JSONDecodeError
            ),
        )

        with pytest.raises(IngestionError):
            await adapter.create_entry(
                FacilityEntryCreateRequest(subject="Subject", details="Details")
            )

        assert not (tmp_path / "entries.tmp").exists()
        assert json.loads(path.read_text()) == {"entries": []}

    @pytest.mark.asyncio
    async def test_entry_without_entries_key_is_seeded(self, tmp_path):
        """A JSON file lacking an ``entries`` key gains one rather than failing."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps({"meta": "no entries key"}))
        adapter = _generic_adapter(str(path))

        entry_id = await adapter.create_entry(
            FacilityEntryCreateRequest(subject="Subject", details="Details")
        )

        data = json.loads(path.read_text())
        assert [e["id"] for e in data["entries"]] == [entry_id]
        assert data["meta"] == "no entries key"


class TestGenericConvertEntry:
    """Metadata passthrough and timestamp resolution."""

    def test_optional_metadata_fields_are_carried(self, facility_tz):
        """Every optional source field lands under its ARIEL metadata key."""
        adapter = _generic_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {
                "id": "1",
                "timestamp": "2024-01-05T00:00:00Z",
                "title": "Title",
                "text": "Body",
                "books": ["Operations"],
                "tags": ["rf"],
                "linked_to": "99",
                "level": "Warning",
                "categories": ["RF Systems"],
                "loto_tag": "LOTO-7",
                "event_time": "2024-01-05T01:00:00Z",
                "segment_area": "Booster",
                "body_format": "markdown",
                "entrymakers": ["operator"],
                "num_comments": 0,
                "needs_attention": False,
                "metadata": {"custom": "value"},
            }
        )

        assert entry["metadata"] == {
            "title": "Title",
            "books": ["Operations"],
            "tags": ["rf"],
            "linked_to": "99",
            "level": "Warning",
            "categories": ["RF Systems"],
            "loto_tag": "LOTO-7",
            "event_time": "2024-01-05T01:00:00Z",
            "segment_area": "Booster",
            "body_format": "markdown",
            "entrymakers": ["operator"],
            "num_comments": 0,
            "needs_attention": False,
            "custom": "value",
        }
        assert entry["raw_text"] == "Title\n\nBody"

    def test_attachments_carry_optional_fields(self, facility_tz):
        """Attachment dicts without a url are skipped; the rest keep their extras."""
        adapter = _generic_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {
                "id": "1",
                "timestamp": "2024-01-05T00:00:00Z",
                "title": "T",
                "attachments": [
                    "not-a-dict",
                    {"caption": "no url"},
                    {
                        "url": "https://logbook.invalid/a.png",
                        "type": "image/png",
                        "filename": "a.png",
                        "thumbnail_url": "https://logbook.invalid/a-thumb.png",
                        "caption": "Orbit",
                    },
                ],
            }
        )

        assert entry["attachments"] == [
            {
                "url": "https://logbook.invalid/a.png",
                "type": "image/png",
                "filename": "a.png",
                "thumbnail_url": "https://logbook.invalid/a-thumb.png",
                "caption": "Orbit",
            }
        ]

    def test_relative_when_resolves_in_facility_zone(self, facility_tz):
        """A relative ``when`` resolves to that wall-clock time in the facility zone.

        Seeded demo data uses this so narrative prose citing a clock time stays
        consistent with the ingested timestamp, whatever day it is ingested.
        """
        adapter = _generic_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {"id": "1", "title": "T", "when": {"days_ago": 2, "time": "13:45:00"}}
        )

        expected_date = (datetime.now(facility_tz) - timedelta(days=2)).date()
        assert entry["timestamp"].tzinfo is facility_tz
        assert entry["timestamp"].date() == expected_date
        assert entry["timestamp"].strftime("%H:%M:%S") == "13:45:00"

    def test_relative_when_defaults_to_midnight(self, facility_tz):
        """``when`` without a ``time`` resolves to midnight facility time."""
        adapter = _generic_adapter("/tmp/entries.json")

        entry = adapter._convert_entry({"id": "1", "title": "T", "when": {"days_ago": 0}})

        assert entry["timestamp"].strftime("%H:%M:%S") == "00:00:00"

    @pytest.mark.parametrize(
        "days_ago",
        [-1, True, "3", 1.5, None],
        ids=["negative", "bool", "string", "float", "none"],
    )
    def test_invalid_days_ago_raises(self, facility_tz, days_ago):
        """``days_ago`` must be a non-negative int; bools are rejected explicitly."""
        adapter = _generic_adapter("/tmp/entries.json")

        with pytest.raises(ValueError, match="non-negative integer"):
            adapter._convert_entry(
                {"id": "1", "title": "T", "when": {"days_ago": days_ago, "time": "01:00:00"}}
            )

    @pytest.mark.parametrize("time_value", ["25:99", 5, None], ids=["malformed", "int", "none"])
    def test_invalid_when_time_raises(self, facility_tz, time_value):
        """``when.time`` must be an 'HH:MM:SS' string."""
        adapter = _generic_adapter("/tmp/entries.json")

        with pytest.raises(ValueError, match="HH:MM:SS"):
            adapter._convert_entry(
                {"id": "1", "title": "T", "when": {"days_ago": 1, "time": time_value}}
            )

    def test_absolute_timestamp_ignores_the_facility_anchor(self, facility_tz):
        """An absolute timestamp is parsed verbatim, not shifted toward ``now``."""
        adapter = _generic_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {"id": "1", "title": "T", "timestamp": "2024-01-05T06:00:00Z"}
        )

        assert entry["timestamp"] == datetime(2024, 1, 5, 6, 0, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1704412800, datetime(2024, 1, 5, tzinfo=UTC)),
            ("1704412800", datetime(2024, 1, 5, tzinfo=UTC)),
            ("2024-01-05T00:00:00Z", datetime(2024, 1, 5, tzinfo=UTC)),
            ("2024-01-05T00:00:00+00:00", datetime(2024, 1, 5, tzinfo=UTC)),
        ],
        ids=["epoch-int", "epoch-string", "iso-z", "iso-offset"],
    )
    def test_parse_timestamp_accepted_formats(self, value, expected):
        """Epoch numbers, epoch strings, and ISO 8601 (Z or offset) all parse."""
        adapter = _generic_adapter("/tmp/entries.json")

        assert adapter._parse_timestamp(value) == expected

    @pytest.mark.parametrize("value", ["yesterday", "", None], ids=["prose", "empty", "none"])
    def test_parse_timestamp_rejects_unparseable(self, value):
        """Unparseable values raise rather than defaulting, so the entry is dropped."""
        adapter = _generic_adapter("/tmp/entries.json")

        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            adapter._parse_timestamp(value)


# ---------------------------------------------------------------------------
# JLab adapter
# ---------------------------------------------------------------------------


class TestJLabAdapter:
    """Payload shapes, filters, and field tolerance for the JLab adapter."""

    def test_missing_source_url_raises(self):
        """source_url is mandatory."""
        with pytest.raises(IngestionError) as exc_info:
            JLabLogbookAdapter(_config("jlab_logbook", None))

        assert "source_url is required" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {"data": {"entries": [{"lognumber": "1", "title": "T"}]}},
            {"entries": [{"lognumber": "1", "title": "T"}]},
            [{"lognumber": "1", "title": "T"}],
        ],
        ids=["nested-data", "top-level-entries", "bare-list"],
    )
    async def test_accepted_payload_shapes(self, tmp_path, payload):
        """The API has shipped three envelope shapes; all three are read."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps(payload))
        adapter = _jlab_adapter(str(path))

        entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["1"]

    @pytest.mark.asyncio
    async def test_unrecognised_payload_yields_nothing(self, tmp_path):
        """A payload of an unknown shape reads as empty rather than raising."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps("just a string"))
        adapter = _jlab_adapter(str(path))

        assert await _collect(adapter.fetch_entries()) == []

    @pytest.mark.asyncio
    async def test_books_filter_excludes_other_logbooks(self, tmp_path):
        """books_filter keeps only entries filed in the configured logbooks."""
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                [
                    {"lognumber": "1", "title": "Ops", "books": ["Operations"]},
                    {"lognumber": "2", "title": "RF", "books": ["RF"]},
                    {"lognumber": "3", "title": "Unfiled"},
                ]
            )
        )
        adapter = _jlab_adapter(str(path))
        adapter.books_filter = ["Operations"]

        entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["1"]

    @pytest.mark.asyncio
    async def test_since_and_until_bounds_are_exclusive(self, tmp_path):
        """Entries on either bound are dropped."""
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                [
                    {"lognumber": "early", "created": {"timestamp": "1704067200"}, "title": "e"},
                    {"lognumber": "middle", "created": {"timestamp": "1704412800"}, "title": "m"},
                    {"lognumber": "late", "created": {"timestamp": "1704758400"}, "title": "l"},
                ]
            )
        )
        adapter = _jlab_adapter(str(path))

        entries = await _collect(
            adapter.fetch_entries(
                since=datetime(2024, 1, 1, tzinfo=UTC),
                until=datetime(2024, 1, 9, tzinfo=UTC),
            )
        )

        assert [e["entry_id"] for e in entries] == ["middle"]

    @pytest.mark.asyncio
    async def test_limit_stops_reading(self, tmp_path):
        """limit truncates the yielded entries."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps([{"lognumber": str(i), "title": f"e{i}"} for i in range(4)]))
        adapter = _jlab_adapter(str(path))

        entries = await _collect(adapter.fetch_entries(limit=2))

        assert [e["entry_id"] for e in entries] == ["0", "1"]

    @pytest.mark.asyncio
    async def test_unconvertible_entry_is_logged_and_skipped(self, tmp_path, caplog):
        """A malformed entry is skipped with a warning; the rest still ingest."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps(["not-a-dict", {"lognumber": "ok", "title": "Fine"}]))
        adapter = _jlab_adapter(str(path))

        with caplog.at_level(logging.WARNING, logger="ariel"):
            entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["ok"]
        assert "Failed to convert entry" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        """A missing local file is an ingestion failure."""
        adapter = _jlab_adapter(str(tmp_path / "absent.json"))

        with pytest.raises(IngestionError) as exc_info:
            await adapter._load_data()

        assert "Source file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_source_returns_decoded_json(self):
        """An http(s) source is fetched and decoded as JSON."""
        adapter = _jlab_adapter("https://logbook.invalid/api/entries")
        session = _http_session(get=_http_response(json_data={"entries": []}))

        with patch("aiohttp.ClientSession", return_value=session):
            assert await adapter._load_data() == {"entries": []}

    @pytest.mark.asyncio
    async def test_http_non_200_raises(self):
        """Any non-200 status fails the load with the status in the message."""
        adapter = _jlab_adapter("https://logbook.invalid/api/entries")
        session = _http_session(get=_http_response(status=500))

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(IngestionError) as exc_info:
                await adapter._load_data()

        assert "status 500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_aiohttp_raises(self, monkeypatch):
        """An HTTP source without aiohttp installed fails with a clear message."""
        adapter = _jlab_adapter("https://logbook.invalid/api/entries")
        monkeypatch.setitem(sys.modules, "aiohttp", None)

        with pytest.raises(IngestionError) as exc_info:
            await adapter._load_data()

        assert "aiohttp is required" in str(exc_info.value)

    def test_malformed_created_timestamp_falls_back_to_now(self):
        """HAZARD: an unparseable created.timestamp silently becomes ingest time."""
        adapter = _jlab_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {"lognumber": "1", "created": {"timestamp": "not-an-epoch"}, "title": "T"}
        )

        assert entry["timestamp"] == entry["created_at"]

    def test_non_dict_created_block_falls_back_to_epoch_zero(self):
        """A ``created`` block of the wrong type reads as epoch 0, not as an error."""
        adapter = _jlab_adapter("/tmp/entries.json")

        entry = adapter._convert_entry({"lognumber": "1", "created": "2024-01-05", "title": "T"})

        assert entry["timestamp"] == datetime(1970, 1, 1, tzinfo=UTC)

    def test_missing_lognumber_and_id_yields_empty_entry_id(self):
        """HAZARD: with neither lognumber nor id the entry_id is '' — one shared row.

        ``enhanced_entries`` upserts ``ON CONFLICT (entry_id)``, so every id-less
        entry collapses onto the same row.
        """
        adapter = _jlab_adapter("/tmp/entries.json")

        assert adapter._convert_entry({"title": "T"})["entry_id"] == ""
        assert adapter._convert_entry({"id": "99", "title": "T"})["entry_id"] == "99"

    def test_title_only_entry_keeps_its_title(self):
        """With no body content the title alone becomes the searchable text."""
        adapter = _jlab_adapter("/tmp/entries.json")

        entry = adapter._convert_entry({"lognumber": "1", "title": "Title only"})

        assert entry["raw_text"] == "Title only"

    def test_optional_metadata_fields_are_carried(self):
        """Body format, entrymakers, comment count, and the attention flag survive."""
        adapter = _jlab_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {
                "lognumber": "1",
                "title": "Title",
                "body": {"content": "Body", "format": "html"},
                "books": ["Operations"],
                "tags": ["rf"],
                "entrymakers": ["operator"],
                "numComments": 0,
                "needsAttention": False,
                "metadata": {"custom": "value"},
            }
        )

        assert entry["metadata"] == {
            "title": "Title",
            "books": ["Operations"],
            "tags": ["rf"],
            "body_format": "html",
            "entrymakers": ["operator"],
            "num_comments": 0,
            "needs_attention": False,
            "custom": "value",
        }

    def test_attachment_transform_skips_unusable_entries(self):
        """Non-dicts and url-less dicts are dropped; thumbnails and captions kept."""
        adapter = _jlab_adapter("/tmp/entries.json")

        result = adapter._transform_attachments(
            [
                "not-a-dict",
                {"type": "image/png"},
                {
                    "url": "https://logbook.invalid/img/a.png",
                    "type": "image/png",
                    "thumbnail": "https://logbook.invalid/img/a-thumb.png",
                    "caption": "Orbit",
                },
                {"url": "bare-name.txt"},
            ]
        )

        assert result == [
            {
                "url": "https://logbook.invalid/img/a.png",
                "type": "image/png",
                "filename": "a.png",
                "thumbnail_url": "https://logbook.invalid/img/a-thumb.png",
                "caption": "Orbit",
            },
            {"url": "bare-name.txt", "type": None, "filename": "bare-name.txt"},
        ]

    def test_thumbnails_can_be_disabled(self):
        """include_thumbnails=False drops the thumbnail URL from the attachment."""
        adapter = _jlab_adapter("/tmp/entries.json")
        adapter.include_thumbnails = False

        result = adapter._transform_attachments(
            [{"url": "https://logbook.invalid/a.png", "thumbnail": "https://x.invalid/t.png"}]
        )

        assert "thumbnail_url" not in result[0]


# ---------------------------------------------------------------------------
# ORNL adapter
# ---------------------------------------------------------------------------


class TestORNLAdapter:
    """Payload shapes, filters, timestamps, and attachment headers for ORNL."""

    def test_missing_source_url_raises(self):
        """source_url is mandatory."""
        with pytest.raises(IngestionError) as exc_info:
            ORNLLogbookAdapter(_config("ornl_logbook", None))

        assert "source_url is required" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [{"entries": [{"ID": "1", "title": "T"}]}, [{"ID": "1", "title": "T"}]],
        ids=["entries-envelope", "bare-list"],
    )
    async def test_accepted_payload_shapes(self, tmp_path, payload):
        """Both the enveloped and bare-list exports are read."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps(payload))
        adapter = _ornl_adapter(str(path))

        entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["1"]

    @pytest.mark.asyncio
    async def test_unrecognised_payload_yields_nothing(self, tmp_path):
        """A payload of an unknown shape reads as empty rather than raising."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps({"unexpected": "shape"}))
        adapter = _ornl_adapter(str(path))

        assert await _collect(adapter.fetch_entries()) == []

    @pytest.mark.asyncio
    async def test_logbooks_filter_excludes_other_logbooks(self, tmp_path):
        """logbooks_filter keeps only entries filed in the configured logbooks."""
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                [
                    {"ID": "1", "title": "Ops", "logbook": "Operations"},
                    {"ID": "2", "title": "RF", "logbook": "RF"},
                    {"ID": "3", "title": "Unfiled"},
                ]
            )
        )
        adapter = _ornl_adapter(str(path))
        adapter.logbooks_filter = ["Operations"]

        entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["1"]

    @pytest.mark.asyncio
    async def test_since_and_until_bounds_are_exclusive(self, tmp_path):
        """Entries on either bound are dropped."""
        path = tmp_path / "entries.json"
        path.write_text(
            json.dumps(
                [
                    {"ID": "early", "entry_time": "2024-01-01T00:00:00Z", "title": "e"},
                    {"ID": "middle", "entry_time": "2024-01-05T00:00:00Z", "title": "m"},
                    {"ID": "late", "entry_time": "2024-01-09T00:00:00Z", "title": "l"},
                ]
            )
        )
        adapter = _ornl_adapter(str(path))

        entries = await _collect(
            adapter.fetch_entries(
                since=datetime(2024, 1, 1, tzinfo=UTC),
                until=datetime(2024, 1, 9, tzinfo=UTC),
            )
        )

        assert [e["entry_id"] for e in entries] == ["middle"]

    @pytest.mark.asyncio
    async def test_limit_stops_reading(self, tmp_path):
        """limit truncates the yielded entries."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps([{"ID": str(i), "title": f"e{i}"} for i in range(4)]))
        adapter = _ornl_adapter(str(path))

        entries = await _collect(adapter.fetch_entries(limit=2))

        assert [e["entry_id"] for e in entries] == ["0", "1"]

    @pytest.mark.asyncio
    async def test_unconvertible_entry_is_logged_and_skipped(self, tmp_path, caplog):
        """A malformed entry is skipped with a warning; the rest still ingest."""
        path = tmp_path / "entries.json"
        path.write_text(json.dumps(["not-a-dict", {"ID": "ok", "title": "Fine"}]))
        adapter = _ornl_adapter(str(path))

        with caplog.at_level(logging.WARNING, logger="ariel"):
            entries = await _collect(adapter.fetch_entries())

        assert [e["entry_id"] for e in entries] == ["ok"]
        assert "Failed to convert entry" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        """A missing local file is an ingestion failure."""
        adapter = _ornl_adapter(str(tmp_path / "absent.json"))

        with pytest.raises(IngestionError) as exc_info:
            await adapter._load_data()

        assert "Source file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_source_returns_decoded_json(self):
        """An http(s) source is fetched and decoded as JSON."""
        adapter = _ornl_adapter("https://logbook.invalid/api/entries")
        session = _http_session(get=_http_response(json_data=[{"ID": "1"}]))

        with patch("aiohttp.ClientSession", return_value=session):
            assert await adapter._load_data() == [{"ID": "1"}]

    @pytest.mark.asyncio
    async def test_http_non_200_raises(self):
        """Any non-200 status fails the load with the status in the message."""
        adapter = _ornl_adapter("https://logbook.invalid/api/entries")
        session = _http_session(get=_http_response(status=404))

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(IngestionError) as exc_info:
                await adapter._load_data()

        assert "status 404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_aiohttp_raises(self, monkeypatch):
        """An HTTP source without aiohttp installed fails with a clear message."""
        adapter = _ornl_adapter("https://logbook.invalid/api/entries")
        monkeypatch.setitem(sys.modules, "aiohttp", None)

        with pytest.raises(IngestionError) as exc_info:
            await adapter._load_data()

        assert "aiohttp is required" in str(exc_info.value)

    def test_missing_entry_time_falls_back_to_now(self):
        """An entry with no entry_time is dated at ingest time."""
        adapter = _ornl_adapter("/tmp/entries.json")

        entry = adapter._convert_entry({"ID": "1", "title": "T"})

        assert entry["timestamp"] == entry["created_at"]

    def test_missing_id_yields_empty_entry_id(self):
        """HAZARD: with neither ID nor id the entry_id is '' — one shared row.

        ``enhanced_entries`` upserts ``ON CONFLICT (entry_id)``, so every id-less
        entry collapses onto the same row.
        """
        adapter = _ornl_adapter("/tmp/entries.json")

        assert adapter._convert_entry({"title": "T"})["entry_id"] == ""
        assert adapter._convert_entry({"id": "99", "title": "T"})["entry_id"] == "99"

    def test_title_only_entry_keeps_its_title(self):
        """With no content the title alone becomes the searchable text."""
        adapter = _ornl_adapter("/tmp/entries.json")

        assert adapter._convert_entry({"ID": "1", "title": "Title only"})["raw_text"] == (
            "Title only"
        )

    def test_optional_metadata_fields_are_carried(self):
        """Area, reference, attachment headers, and explicit metadata all survive."""
        adapter = _ornl_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {
                "ID": "1",
                "title": "Title",
                "content": "Body",
                "entry_time": "2024-01-05T00:00:00Z",
                "event_time": "2024-01-05T01:00:00Z",
                "logbook": "Operations",
                "segment/area": "Booster",
                "reference": "99",
                "attachment_header": ["a.png", "b.png"],
                "metadata": {"custom": "value"},
            }
        )

        assert entry["metadata"] == {
            "title": "Title",
            "books": ["Operations"],
            "segment_area": "Booster",
            "event_time": "2024-01-05T01:00:00+00:00",
            "linked_to": "99",
            "attachment_headers": ["a.png", "b.png"],
            "custom": "value",
        }

    def test_logbook_list_and_single_attachment_header(self):
        """A list logbook is kept as-is; a lone attachment header is wrapped."""
        adapter = _ornl_adapter("/tmp/entries.json")

        entry = adapter._convert_entry(
            {
                "ID": "1",
                "title": "T",
                "logbook": ["Operations", "RF"],
                "attachment_header": "only.png",
            }
        )

        assert entry["metadata"]["books"] == ["Operations", "RF"]
        assert entry["metadata"]["attachment_headers"] == ["only.png"]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1704412800, datetime(2024, 1, 5, tzinfo=UTC)),
            ("1704412800", datetime(2024, 1, 5, tzinfo=UTC)),
            ("2024-01-05T00:00:00Z", datetime(2024, 1, 5, tzinfo=UTC)),
            ("2024-01-05T00:00:00+00:00", datetime(2024, 1, 5, tzinfo=UTC)),
        ],
        ids=["epoch-int", "epoch-string", "iso-z", "iso-offset"],
    )
    def test_parse_timestamp_accepted_formats(self, value, expected):
        """Epoch numbers, epoch strings, and ISO 8601 (Z or offset) all parse."""
        adapter = _ornl_adapter("/tmp/entries.json")

        assert adapter._parse_timestamp(value) == expected

    @pytest.mark.parametrize("value", ["yesterday", None], ids=["prose", "none"])
    def test_parse_timestamp_falls_back_to_now(self, value):
        """HAZARD: ORNL defaults an unparseable timestamp to now instead of raising.

        Unlike the generic adapter — which raises and loses the entry — ORNL keeps
        the entry and misdates it to the moment of the ingest run.
        """
        adapter = _ornl_adapter("/tmp/entries.json")

        before = datetime.now(UTC)
        parsed = adapter._parse_timestamp(value)

        assert before <= parsed <= datetime.now(UTC)

    def test_attachment_flag_without_urls_uses_headers(self):
        """``attachment: "Y"`` with no attachment list yields url-less placeholders.

        ORNL's export names the files but not their locations, so the filename is
        preserved and the URL left empty rather than the attachment being dropped.
        """
        adapter = _ornl_adapter("/tmp/entries.json")

        result = adapter._transform_attachments(
            {"attachment": "Y", "attachment_header": ["a.png", "b.png"]}
        )

        assert result == [
            {"url": "", "filename": "a.png", "type": None},
            {"url": "", "filename": "b.png", "type": None},
        ]

    def test_attachment_flag_with_single_header(self):
        """A lone header string is wrapped into one placeholder attachment."""
        adapter = _ornl_adapter("/tmp/entries.json")

        result = adapter._transform_attachments({"attachment": "Y", "attachment_header": "a.png"})

        assert result == [{"url": "", "filename": "a.png", "type": None}]

    def test_attachment_list_skips_non_dicts(self):
        """A real attachment list is transformed, with non-dict members skipped."""
        adapter = _ornl_adapter("/tmp/entries.json")

        result = adapter._transform_attachments(
            {
                "attachments": [
                    "not-a-dict",
                    {"url": "https://logbook.invalid/a.png", "type": "image/png"},
                ]
            }
        )

        assert result == [
            {"url": "https://logbook.invalid/a.png", "type": "image/png", "filename": None}
        ]

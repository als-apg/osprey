"""Tests for the shared local model server resolution in ``_local_server``."""

import threading
import time
from unittest.mock import patch

import pytest

from osprey.models.providers import _local_server
from osprey.models.providers._local_server import (
    LocalServerUnreachable,
    container_fallback_urls,
    reset_cache,
    resolve_cached,
    resolve_local_server,
)

PROBE = "osprey.models.providers._local_server.probe"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Every test starts with an empty cache and no OLLAMA_HOST override."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    reset_cache()
    yield
    reset_cache()


class TestContainerFallbackUrls:
    """The fallback table: other host classes, in the old order, without duplicates."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            (
                "http://localhost:11434",
                ["http://host.docker.internal:11434", "http://host.containers.internal:11434"],
            ),
            (
                "http://host.containers.internal:11434",
                ["http://host.docker.internal:11434", "http://localhost:11434"],
            ),
            (
                "http://host.docker.internal:11434",
                ["http://host.containers.internal:11434", "http://localhost:11434"],
            ),
            (
                "http://remote.example:11434",
                [
                    "http://localhost:11434",
                    "http://host.docker.internal:11434",
                    "http://host.containers.internal:11434",
                ],
            ),
        ],
    )
    def test_fallback_branches(self, base_url, expected):
        """Each host class yields its exact ordered fallback list."""
        assert container_fallback_urls(base_url, 11434) == expected

    def test_custom_port_is_kept(self):
        """A local URL on 8080 falls back on 8080, never on the default port."""
        result = container_fallback_urls("http://localhost:8080", 11434)
        assert result == [
            "http://host.docker.internal:8080",
            "http://host.containers.internal:8080",
        ]
        assert not any("11434" in url for url in result)

    def test_loopback_ip_is_local(self):
        """127.0.0.1 is a local host and falls back onto both container hosts."""
        assert container_fallback_urls("http://127.0.0.1:11434", 11434) == [
            "http://host.docker.internal:11434",
            "http://host.containers.internal:11434",
        ]

    def test_missing_port_takes_default(self):
        """A local URL without a port falls back on the default port."""
        assert container_fallback_urls("http://localhost", 8080) == [
            "http://host.docker.internal:8080",
            "http://host.containers.internal:8080",
        ]

    def test_scheme_and_path_kept_for_container_classes(self):
        """A container URL keeps its scheme and path on every fallback."""
        assert container_fallback_urls("https://host.docker.internal:9000/v1", 11434) == [
            "https://host.containers.internal:9000/v1",
            "https://localhost:9000/v1",
        ]

    def test_remote_with_path_gets_plain_fallbacks(self):
        """A remote https URL with a path falls back to plain http on the default port."""
        assert container_fallback_urls("https://ollama.site.org/api", 11434) == [
            "http://localhost:11434",
            "http://host.docker.internal:11434",
            "http://host.containers.internal:11434",
        ]

    def test_never_probes(self):
        """Building the table makes no probe."""
        with patch(PROBE) as mock_probe:
            container_fallback_urls("http://localhost:11434", 11434)
        mock_probe.assert_not_called()


class TestResolveLocalServer:
    """The per-call walk: env override, configured URL, fallbacks."""

    def test_primary_answers(self):
        """The configured URL is returned when it answers."""
        with patch(PROBE, return_value=True):
            assert (
                resolve_local_server(
                    "http://localhost:11434",
                    probe_path="/api/tags",
                    env_var="OLLAMA_HOST",
                    default_port=11434,
                )
                == "http://localhost:11434"
            )

    def test_falls_back(self):
        """Primary down, first fallback up -> the fallback is returned."""
        with patch(PROBE, side_effect=[False, True]):
            assert (
                resolve_local_server(
                    "http://localhost:11434",
                    probe_path="/api/tags",
                    env_var="OLLAMA_HOST",
                    default_port=11434,
                )
                == "http://host.docker.internal:11434"
            )

    def test_env_override_first(self, monkeypatch):
        """An answering env override wins before the configured URL is probed."""
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
        with patch(PROBE, return_value=True) as mock_probe:
            result = resolve_local_server(
                "http://localhost:11434",
                probe_path="/api/tags",
                env_var="OLLAMA_HOST",
                default_port=11434,
            )
        assert result == "http://ollama:11434"
        mock_probe.assert_called_once_with("http://ollama:11434", "/api/tags", 2.0)

    def test_nothing_answers_raises_connection_and_runtime_error(self):
        """No answering candidate raises an error that is both ConnectionError and RuntimeError."""
        with patch(PROBE, return_value=False):
            with pytest.raises(LocalServerUnreachable, match="Failed to connect") as info:
                resolve_local_server(
                    "http://localhost:11434",
                    probe_path="/api/tags",
                    env_var="OLLAMA_HOST",
                    default_port=11434,
                )
        assert isinstance(info.value, ConnectionError)
        assert isinstance(info.value, RuntimeError)

    def test_probe_uses_probe_path(self):
        """The probe receives the caller's probe path."""
        with patch(PROBE, return_value=True) as mock_probe:
            resolve_local_server(
                "http://localhost:8080", probe_path="/health", env_var=None, default_port=8080
            )
        mock_probe.assert_called_once_with("http://localhost:8080", "/health", 2.0)


class TestResolveCached:
    """The one resolution cache."""

    def _call(self, **kwargs):
        return resolve_cached(
            "http://localhost:11434",
            probe_path="/api/tags",
            env_var="OLLAMA_HOST",
            default_port=11434,
            **kwargs,
        )

    def test_ten_calls_one_probe(self):
        """Ten calls make one probe."""
        with patch(PROBE, return_value=True) as mock_probe:
            results = [self._call() for _ in range(10)]
        assert set(results) == {"http://localhost:11434"}
        assert mock_probe.call_count == 1

    def test_nothing_answers_returns_configured_uncached(self):
        """With nothing answering the configured URL comes back and is not cached."""
        with patch(PROBE, return_value=False) as mock_probe:
            assert self._call() == "http://localhost:11434"
            first = mock_probe.call_count
            assert self._call() == "http://localhost:11434"
        assert mock_probe.call_count == 2 * first

    def test_refresh_finds_moved_server(self):
        """The cached fallback stops answering and the configured URL starts -> refresh finds it."""
        up = {"http://host.docker.internal:11434"}

        def fake_probe(url, _path, _timeout):
            return url in up

        with patch(PROBE, side_effect=fake_probe):
            assert self._call() == "http://host.docker.internal:11434"
            up.clear()
            up.add("http://localhost:11434")
            assert self._call() == "http://host.docker.internal:11434"
            assert self._call(refresh=True) == "http://localhost:11434"
            assert self._call() == "http://localhost:11434"

    def test_refresh_keeps_answering_cache(self):
        """refresh=True gives a live cached URL one probe and keeps it."""
        with patch(PROBE, return_value=True) as mock_probe:
            self._call()
            assert self._call(refresh=True) == "http://localhost:11434"
        assert mock_probe.call_count == 2

    def test_deadline_stops_the_walk_and_late_walk_still_caches(self):
        """A walk past the deadline returns the configured URL; the late walk still writes the cache."""
        release = threading.Event()

        def slow_probe(url, _path, _timeout):
            if url == "http://localhost:11434":
                return False
            release.wait(5)
            return url == "http://host.docker.internal:11434"

        with patch(PROBE, side_effect=slow_probe):
            started = time.monotonic()
            assert self._call(deadline_s=0.1) == "http://localhost:11434"
            assert time.monotonic() - started < 2
            release.set()
            for _ in range(100):
                if _local_server._cache:
                    break
                time.sleep(0.02)
        with patch(PROBE) as mock_probe:
            assert self._call() == "http://host.docker.internal:11434"
        mock_probe.assert_not_called()

    def test_deadline_met_returns_found(self):
        """A walk that finishes within the deadline returns its answer."""
        with patch(PROBE, return_value=True):
            assert self._call(deadline_s=5) == "http://localhost:11434"

    def test_reset_cache_forgets(self):
        """reset_cache makes the next call walk again."""
        with patch(PROBE, return_value=True) as mock_probe:
            self._call()
            reset_cache()
            self._call()
        assert mock_probe.call_count == 2

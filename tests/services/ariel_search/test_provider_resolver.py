"""Tests for the ARIEL embedding-provider resolver.

``resolve_provider`` turns a module's ``provider`` value into the adapter class,
one instance, the configured base URL and key without touching the network;
``resolve_reachable_base_url`` finds the URL a local server answers on, through
the one local-server cache, for adapters that resolve outside their calls.
"""

from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.models.provider_registry import ProviderRegistry
from osprey.models.providers import _local_server
from osprey.services.ariel_search.enhancement.provider_resolver import (
    ResolvedProvider,
    resolve_provider,
    resolve_reachable_base_url,
)
from osprey.services.ariel_search.exceptions import ModuleConfigError
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

KEY = "ariel.enhancement_modules.text_embedding.provider"
IMAGE_KEY = "ariel.enhancement_modules.image_embedding.provider"

_REAL_GET_PROVIDER = ProviderRegistry.get_provider


def _http_ok(url: str, path: str, timeout: float) -> bool:
    """A real ``GET <url><path>`` answering 200, independent of any patched probe."""
    import requests

    try:
        return requests.get(url.rstrip("/") + path, timeout=timeout).status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _fresh_local_server_cache():
    """Every test starts and ends with an empty local-server cache."""
    _local_server.reset_cache()
    yield
    _local_server.reset_cache()


@pytest.fixture(autouse=True)
def provider_configs(monkeypatch):
    """Route ``api.providers`` entries through a dict the test fills in (empty by default)."""
    entries: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        "osprey.models.config.get_provider_config", lambda name: dict(entries.get(name, {}))
    )
    return entries


def _register(monkeypatch, **classes: type) -> None:
    """Make the registry answer *classes* by name; every other name is looked up for real."""

    def get_provider(self, name):
        return classes[name] if name in classes else _REAL_GET_PROVIDER(self, name)

    monkeypatch.setattr(ProviderRegistry, "get_provider", get_provider)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Stub:
    """A local HTTP server answering 200 on every GET."""

    def __init__(self, port: int = 0) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data": [{"id": "m"}]}')

            def log_message(self, *args):
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stubs():
    started: list[_Stub] = []

    def start(port: int = 0) -> _Stub:
        stub = _Stub(port)
        started.append(stub)
        return stub

    yield start
    for stub in started:
        try:
            stub.stop()
        except Exception:
            pass


class _ProbeRecorder:
    """Patched ``_local_server.probe``: maps container hosts onto local stubs, records calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.docker_target: str | None = None
        self.delay_s = 0.0

    def __call__(self, url: str, path: str, timeout: float) -> bool:
        self.calls.append(url)
        if self.delay_s:
            time.sleep(self.delay_s)
        if "host.docker.internal" in url:
            if self.docker_target is None:
                return False
            return _http_ok(self.docker_target, path, timeout)
        if "host.containers.internal" in url:
            return False
        return _http_ok(url, path, timeout)


@pytest.fixture
def probes(monkeypatch):
    recorder = _ProbeRecorder()
    monkeypatch.setattr(_local_server, "probe", recorder)
    return recorder


def _fallback_class(**kw: Any) -> type:
    """A fake adapter that resolves its reachable URL outside its calls."""
    base = make_fake_embedding_provider(
        name="fakellama",
        default_base_url="http://localhost:8080",
        host_override_env_var="X_HOST",
        resolves_fallback_outside_calls=True,
        **kw,
    )
    return type("FakeLlama", (base,), {"fallback_probe_path": "/v1/models"})


# --------------------------------------------------------------------------
# resolve_provider
# --------------------------------------------------------------------------


class TestResolveProviderPrecedence:
    """Base URL: inline > api.providers entry > adapter default."""

    def test_inline_base_url_wins(self, monkeypatch, provider_configs):
        fake = make_fake_embedding_provider(default_base_url="http://default:1")
        _register(monkeypatch, fake=fake)
        provider_configs["fake"] = {"base_url": "http://entry:1", "api_key": "entry-key"}

        resolved = resolve_provider(
            {"name": "fake", "base_url": "http://inline:1", "api_key": "inline-key"},
            provider_key=KEY,
            default="ollama",
        )

        assert isinstance(resolved, ResolvedProvider)
        assert resolved.cls is fake
        assert type(resolved.instance) is fake
        assert resolved.base_url == "http://inline:1"
        assert resolved.api_key == "inline-key"

    def test_a_provider_name_takes_its_entry(self, monkeypatch, provider_configs):
        fake = make_fake_embedding_provider(default_base_url="http://default:1")
        _register(monkeypatch, fake=fake)
        provider_configs["fake"] = {"base_url": "http://entry:1", "api_key": "entry-key"}

        resolved = resolve_provider("fake", provider_key=KEY, default="ollama")

        assert resolved.base_url == "http://entry:1"
        assert resolved.api_key == "entry-key"

    def test_an_inline_dict_without_base_url_takes_the_entry(self, monkeypatch, provider_configs):
        fake = make_fake_embedding_provider(default_base_url="http://default:1")
        _register(monkeypatch, fake=fake)
        provider_configs["fake"] = {"base_url": "http://entry:1", "api_key": "entry-key"}

        resolved = resolve_provider({"name": "fake"}, provider_key=KEY, default="ollama")

        assert resolved.base_url == "http://entry:1"
        assert resolved.api_key == "entry-key"

    def test_no_inline_and_no_entry_takes_the_adapter_default(self, monkeypatch):
        fake = make_fake_embedding_provider(default_base_url="http://default:1")
        _register(monkeypatch, fake=fake)

        assert (
            resolve_provider("fake", provider_key=KEY, default="ollama").base_url
            == "http://default:1"
        )
        assert (
            resolve_provider({"name": "fake"}, provider_key=KEY, default="ollama").base_url
            == "http://default:1"
        )

    def test_a_missing_config_file_reads_as_an_empty_entry(self, monkeypatch):
        fake = make_fake_embedding_provider(default_base_url="http://default:1")
        _register(monkeypatch, fake=fake)

        def no_file(name):
            raise FileNotFoundError(name)

        monkeypatch.setattr("osprey.models.config.get_provider_config", no_file)

        resolved = resolve_provider("fake", provider_key=KEY, default="ollama")

        assert resolved.base_url == "http://default:1"
        assert resolved.api_key is None

    def test_none_takes_the_default_provider(self):
        from osprey.models.providers.ollama import OllamaProviderAdapter

        resolved = resolve_provider(None, provider_key=KEY, default="ollama")

        assert resolved.cls is OllamaProviderAdapter

    def test_resolution_does_no_network_io(self, monkeypatch):
        def no_network(*args, **kwargs):
            raise AssertionError("resolve_provider touched the network")

        monkeypatch.setattr(_local_server, "probe", no_network)
        monkeypatch.setattr("requests.get", no_network)
        monkeypatch.setattr("requests.post", no_network)

        resolved = resolve_provider("llama-cpp", provider_key=KEY, default="ollama")

        assert resolved.base_url == "http://localhost:8080"


class TestResolveProviderRefusals:
    """Every refusal is a ModuleConfigError naming the module's provider key."""

    def test_an_unknown_provider_names_the_key(self):
        with pytest.raises(ModuleConfigError) as exc:
            resolve_provider("no-such-provider", provider_key=KEY, default="ollama")
        assert exc.value.key == KEY
        assert str(exc.value).startswith(KEY)
        assert "no-such-provider" in str(exc.value)

    def test_a_provider_without_embeddings_is_refused(self, monkeypatch):
        fake = make_fake_embedding_provider()
        monkeypatch.setattr(fake, "supports_embeddings", classmethod(lambda cls: False))
        _register(monkeypatch, chatonly=fake)

        with pytest.raises(ModuleConfigError) as exc:
            resolve_provider("chatonly", provider_key=KEY, default="ollama")
        assert exc.value.key == KEY

    def test_image_embeddings_need_an_image_provider(self):
        with pytest.raises(ModuleConfigError) as exc:
            resolve_provider(
                "ollama", provider_key=IMAGE_KEY, default=None, serves="image_embeddings"
            )
        assert exc.value.key == IMAGE_KEY
        assert "image embeddings" in str(exc.value)

    def test_no_provider_and_no_default_is_refused(self):
        with pytest.raises(ModuleConfigError) as exc:
            resolve_provider(None, provider_key=IMAGE_KEY, default=None)
        assert exc.value.key == IMAGE_KEY

    def test_a_malformed_provider_value_is_refused(self):
        with pytest.raises(ModuleConfigError) as exc:
            resolve_provider(42, provider_key=KEY, default="ollama")  # type: ignore[arg-type]
        assert exc.value.key == KEY

    @pytest.mark.parametrize(
        ("key", "serves"),
        [(KEY, "embeddings"), (IMAGE_KEY, "image_embeddings")],
    )
    def test_a_llama_cpp_v1_base_is_refused(self, key, serves):
        with pytest.raises(ModuleConfigError) as exc:
            resolve_provider(
                {"name": "llama-cpp", "base_url": "http://gpu:8080/v1"},
                provider_key=key,
                default=None,
                serves=serves,
            )
        assert exc.value.key == key
        assert str(exc.value).startswith(key)
        assert isinstance(exc.value.__cause__, ValueError)

    def test_a_refusal_is_still_a_value_error(self):
        with pytest.raises(ValueError):
            resolve_provider("no-such-provider", provider_key=KEY, default="ollama")


# --------------------------------------------------------------------------
# resolve_reachable_base_url
# --------------------------------------------------------------------------


class TestResolveReachableBaseUrl:
    def test_a_class_resolving_inside_calls_gets_its_url_back(self, probes):
        from osprey.models.providers.ollama import OllamaProviderAdapter

        url = "http://localhost:1"
        assert resolve_reachable_base_url(OllamaProviderAdapter, url) == url
        assert probes.calls == []

    def test_a_refusing_localhost_falls_back_to_the_docker_host(self, probes, stubs, monkeypatch):
        monkeypatch.delenv("X_HOST", raising=False)
        stub = stubs()
        port = _free_port()
        probes.docker_target = stub.url
        configured = f"http://127.0.0.1:{port}"

        found = resolve_reachable_base_url(_fallback_class(), configured)

        assert found == f"http://host.docker.internal:{port}"
        assert probes.calls[:2] == [configured, found]

    def test_ten_resolutions_make_one_probe(self, probes, stubs, monkeypatch):
        monkeypatch.delenv("X_HOST", raising=False)
        stub = stubs()
        cls = _fallback_class()

        for _ in range(10):
            assert resolve_reachable_base_url(cls, stub.url) == stub.url

        assert probes.calls == [stub.url]

    def test_a_fallback_walk_is_cached(self, probes, stubs, monkeypatch):
        monkeypatch.delenv("X_HOST", raising=False)
        probes.docker_target = stubs().url
        cls = _fallback_class()
        configured = f"http://127.0.0.1:{_free_port()}"

        first = resolve_reachable_base_url(cls, configured)
        walked = len(probes.calls)
        for _ in range(9):
            assert resolve_reachable_base_url(cls, configured) == first

        assert len(probes.calls) == walked

    def test_the_env_override_is_tried_first(self, probes, stubs, monkeypatch):
        stub = stubs()
        monkeypatch.setenv("X_HOST", stub.url)
        configured = f"http://127.0.0.1:{_free_port()}"

        assert resolve_reachable_base_url(_fallback_class(), configured) == stub.url
        assert probes.calls == [stub.url]

    @pytest.mark.usefixtures("probes")
    def test_nothing_answering_returns_the_configured_url(self, monkeypatch):
        monkeypatch.delenv("X_HOST", raising=False)
        configured = f"http://127.0.0.1:{_free_port()}"

        assert resolve_reachable_base_url(_fallback_class(), configured) == configured

    def test_refresh_finds_a_server_that_moved(self, probes, stubs, monkeypatch):
        monkeypatch.delenv("X_HOST", raising=False)
        fallback = stubs()
        probes.docker_target = fallback.url
        port = _free_port()
        configured = f"http://127.0.0.1:{port}"
        cls = _fallback_class()
        assert resolve_reachable_base_url(cls, configured) == (
            f"http://host.docker.internal:{port}"
        )

        fallback.stop()
        probes.docker_target = None
        stubs(port)

        assert resolve_reachable_base_url(cls, configured, refresh=True) == configured

    def test_a_late_walk_still_writes_the_cache(self, probes, stubs, monkeypatch):
        monkeypatch.delenv("X_HOST", raising=False)
        probes.docker_target = stubs().url
        probes.delay_s = 0.3
        port = _free_port()
        configured = f"http://127.0.0.1:{port}"
        cls = _fallback_class()

        assert resolve_reachable_base_url(cls, configured, deadline_s=0.05) == configured

        deadline = time.monotonic() + 5
        while len(probes.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        walked = len(probes.calls)

        assert resolve_reachable_base_url(cls, configured) == (
            f"http://host.docker.internal:{port}"
        )
        assert len(probes.calls) == walked


# --------------------------------------------------------------------------
# text_embedding through the resolver
# --------------------------------------------------------------------------


def _text_module(provider: Any, models: list[dict[str, Any]] | None = None):
    from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingModule

    module = TextEmbeddingModule()
    module.configure({"provider": provider, "models": models or [{"name": "m", "dimension": 4}]})
    return module


def _tables_exist_conn() -> MagicMock:
    result = MagicMock()
    result.fetchone = AsyncMock(return_value=(True,))
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=result)
    return conn


class TestTextEmbeddingResolution:
    def test_an_inline_dict_without_base_url_takes_the_entry(self, monkeypatch, provider_configs):
        fake = make_fake_embedding_provider(default_base_url="http://default:1")
        _register(monkeypatch, fake=fake)
        provider_configs["fake"] = {"base_url": "http://entry:1", "api_key": "k"}

        module = _text_module({"name": "fake"})

        assert module._base_url() == "http://entry:1"

    def test_an_unknown_provider_is_a_module_config_error(self):
        with pytest.raises(ModuleConfigError) as exc:
            _text_module("no-such-provider")
        assert exc.value.key == KEY

    def test_a_llama_cpp_v1_base_is_refused(self):
        with pytest.raises(ModuleConfigError) as exc:
            _text_module({"name": "llama-cpp", "base_url": "http://gpu:8080/v1"})
        assert exc.value.key == KEY

    @pytest.mark.asyncio
    async def test_a_refusing_localhost_embeds_through_the_fallback(
        self, monkeypatch, probes, stubs
    ):
        monkeypatch.delenv("X_HOST", raising=False)
        probes.docker_target = stubs().url
        cls = _fallback_class()
        _register(monkeypatch, fakellama=cls)
        port = _free_port()
        module = _text_module({"name": "fakellama", "base_url": f"http://127.0.0.1:{port}"})

        await module.enhance({"entry_id": "e1", "raw_text": "beam"}, _tables_exist_conn())
        walked = len(probes.calls)
        await module.enhance({"entry_id": "e2", "raw_text": "orbit"}, _tables_exist_conn())

        fallback = f"http://host.docker.internal:{port}"
        assert [call["base_url"] for call in cls.calls] == [fallback, fallback]
        assert len(probes.calls) == walked

    @pytest.mark.asyncio
    async def test_configure_with_no_server_succeeds_and_health_reports_unreachable(
        self, monkeypatch
    ):
        monkeypatch.delenv("LLAMA_CPP_HOST", raising=False)
        monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: False)

        module = _text_module({"name": "llama-cpp", "base_url": f"http://127.0.0.1:{_free_port()}"})
        verdict = await module.health_check()

        assert verdict.reachable is False
        assert verdict.reason == "unreachable"

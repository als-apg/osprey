"""Tests for the llama-cpp provider adapter: embeddings over an in-process stub server."""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import math
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from osprey.models.provider_registry import get_provider_registry
from osprey.models.providers import _local_server
from osprey.models.providers.base import (
    BaseProvider,
    DegenerateVectorError,
    EmbeddingDimensionError,
)
from osprey.models.providers.llama_cpp import LLAMA_CPP_DEFAULT_MODEL, LlamaCppProviderAdapter
from osprey.services.ariel_search.search import fusion

MODEL = LLAMA_CPP_DEFAULT_MODEL


@dataclass
class StubServer:
    """A running plain-HTTP llama-server stand-in and what it received."""

    url: str
    embed: Callable[[dict], list[float]]
    model_id: str = MODEL
    models_status: int = 200
    seen: list[tuple[str, str, Any]] = field(default_factory=list)

    def methods(self) -> list[str]:
        return [method for method, _, _ in self.seen]


def _vector_of(body: dict) -> list[float]:
    """A 2048-d vector whose first component identifies the input it answers."""
    part = body["input"][0]["content"][0]
    key = part["text"] if part["type"] == "text" else part["image_url"]["url"]
    vector = [1.0] * 2048
    vector[0] = float(len(key))
    return vector


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server_state: dict[str, StubServer] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):  # silence the test output
            pass

        def _reply(self, status: int, payload: Any) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state = server_state["s"]
            state.seen.append(("GET", self.path, None))
            if self.path == "/v1/models":
                self._reply(state.models_status, {"data": [{"id": state.model_id}]})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            state = server_state["s"]
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            state.seen.append(("POST", self.path, body))
            if self.path != "/v1/embeddings":
                self._reply(404, {"error": "not found"})
                return
            self._reply(200, {"data": [{"index": 0, "embedding": state.embed(body)}]})

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state = StubServer(url=f"http://127.0.0.1:{httpd.server_address[1]}", embed=_vector_of)
    server_state["s"] = state
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(autouse=True)
def _no_host_override(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_HOST", raising=False)
    _local_server.reset_cache()
    yield
    _local_server.reset_cache()


@pytest.fixture
def adapter() -> LlamaCppProviderAdapter:
    return LlamaCppProviderAdapter()


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


class TestLlamaCppMetadata:
    def test_declared_metadata(self):
        cls = LlamaCppProviderAdapter
        assert cls.name == "llama-cpp"
        assert cls.description
        assert cls.requires_api_key is False
        assert cls.requires_base_url is True
        assert cls.requires_model_id is True
        assert cls.supports_proxy is False
        assert cls.default_base_url == "http://localhost:8080"
        assert cls.base_url_env_var is None

    def test_provider_facts_in_the_class_body(self):
        body = vars(LlamaCppProviderAdapter)
        assert body["api_key_env_var"] is None
        assert body["api_protocol"] == "openai"
        assert body["supports_interactive_login"] is False
        assert body["supports_images"] is False
        assert body["supports_thinking"] is False

    def test_serves_embeddings_and_image_embeddings_but_no_chat(self):
        assert LlamaCppProviderAdapter.supports_embeddings() is True
        assert LlamaCppProviderAdapter.supports_image_embeddings() is True
        assert LlamaCppProviderAdapter.supports_chat() is False

    def test_registry_resolves_it_as_keyless(self):
        cls = get_provider_registry().get_provider("llama-cpp")
        assert cls is LlamaCppProviderAdapter
        assert cls.requires_api_key is False
        assert get_provider_registry().is_chat("llama-cpp") is False

    def test_the_model_id_is_fixed_once(self):
        assert LLAMA_CPP_DEFAULT_MODEL == "qwen3-vl-embedding-2b"
        assert LlamaCppProviderAdapter.default_embedding_model_id == LLAMA_CPP_DEFAULT_MODEL
        assert LlamaCppProviderAdapter.health_check_embedding_model_id == LLAMA_CPP_DEFAULT_MODEL

    def test_the_three_declared_facts(self):
        assert LlamaCppProviderAdapter.host_override_env_var == "LLAMA_CPP_HOST"
        assert LlamaCppProviderAdapter.resolves_fallback_outside_calls is True
        assert LlamaCppProviderAdapter.truncates_to_dimensions is True
        assert LlamaCppProviderAdapter.fallback_probe_path == "/v1/models"


def _other_builtins() -> list[str]:
    return sorted(n for n in get_provider_registry().list_providers() if n != "llama-cpp")


class TestTheNewFactsDefaultOff:
    def test_base_defaults(self):
        assert BaseProvider.host_override_env_var is None
        assert BaseProvider.resolves_fallback_outside_calls is False
        assert BaseProvider.truncates_to_dimensions is False
        assert BaseProvider.validate_base_url("http://h:8080/v1") is None

    @pytest.mark.parametrize("name", _other_builtins())
    def test_every_other_builtin_keeps_the_defaults(self, name):
        cls = get_provider_registry().get_provider(name)
        assert cls.host_override_env_var is None
        assert cls.resolves_fallback_outside_calls is False
        assert cls.truncates_to_dimensions is False

    @pytest.mark.parametrize("name", _other_builtins())
    def test_every_other_builtin_accepts_a_v1_base_and_a_root(self, name):
        cls = get_provider_registry().get_provider(name)
        cls.validate_base_url("http://h:8080/v1")
        cls.validate_base_url("http://h:8080")


class TestValidateBaseUrl:
    @pytest.mark.parametrize("url", ["http://h:8080/v1", "http://h:8080/v1/"])
    def test_a_v1_base_is_refused(self, url):
        with pytest.raises(ValueError, match=r"server root \(as OLLAMA_HOST\), not …/v1"):
            LlamaCppProviderAdapter.validate_base_url(url)

    @pytest.mark.parametrize("url", ["http://h:8080", "http://h:8080/", None])
    def test_the_root_passes(self, url):
        LlamaCppProviderAdapter.validate_base_url(url)

    def test_execute_refuses_a_v1_base_before_any_request(self, adapter, stub):
        with pytest.raises(ValueError, match="server root"):
            adapter.execute_embedding(["a"], MODEL, base_url=stub.url + "/v1")
        assert stub.seen == []


class TestEmbedding:
    def test_image_request_shape(self, adapter, stub):
        png = b"\x89PNG fake"
        adapter.execute_image_embedding([(png, "image/png")], MODEL, base_url=stub.url)
        [(method, path, body)] = stub.seen
        assert (method, path) == ("POST", "/v1/embeddings")
        encoded = base64.b64encode(png).decode()
        assert body["input"] == [
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                ]
            }
        ]
        assert body["model"] == MODEL

    def test_text_request_shape(self, adapter, stub):
        adapter.execute_embedding(["beam lost"], MODEL, base_url=stub.url)
        [(_, _, body)] = stub.seen
        assert body["input"] == [{"content": [{"type": "text", "text": "beam lost"}]}]

    def test_truncates_to_unit_vectors_of_the_requested_length(self, adapter, stub):
        vectors = adapter.execute_image_embedding(
            [(b"img", "image/png"), "query"], MODEL, base_url=stub.url, dimensions=1024
        )
        assert [len(v) for v in vectors] == [1024, 1024]
        for vector in vectors:
            assert _norm(vector) == pytest.approx(1.0)

    def test_text_embedding_truncates_the_same_way(self, adapter, stub):
        [vector] = adapter.execute_embedding(["q"], MODEL, base_url=stub.url, dimensions=1024)
        assert len(vector) == 1024
        assert _norm(vector) == pytest.approx(1.0)

    def test_without_dimensions_the_full_vector_returns(self, adapter, stub):
        [vector] = adapter.execute_embedding(["q"], MODEL, base_url=stub.url)
        assert len(vector) == 2048

    def test_three_inputs_three_posts_in_input_order(self, adapter, stub):
        texts = ["a", "bbb", "cc"]
        vectors = adapter.execute_embedding(texts, MODEL, base_url=stub.url)
        assert stub.methods() == ["POST", "POST", "POST"]
        sent = [body["input"][0]["content"][0]["text"] for _, _, body in stub.seen]
        assert sent == texts
        assert [v[0] for v in vectors] == [1.0, 3.0, 2.0]

    def test_ten_calls_ten_posts_and_no_get(self, adapter, stub):
        for i in range(10):
            adapter.execute_image_embedding([f"text {i}"], MODEL, base_url=stub.url)
        assert stub.methods() == ["POST"] * 10

    def test_empty_input_sends_nothing(self, adapter, stub):
        assert adapter.execute_embedding([], MODEL, base_url=stub.url) == []
        assert stub.seen == []

    def test_a_zero_vector_is_degenerate(self, adapter, stub):
        stub.embed = lambda _body: [0.0] * 2048
        with pytest.raises(DegenerateVectorError):
            adapter.execute_image_embedding([(b"x", "image/png")], MODEL, base_url=stub.url)

    def test_a_non_finite_vector_is_degenerate(self, adapter, stub):
        stub.embed = lambda _body: [1.0] * 2047 + [1e400]  # JSON Infinity
        with pytest.raises(DegenerateVectorError):
            adapter.execute_embedding(["x"], MODEL, base_url=stub.url, dimensions=2048)

    def test_dimensions_beyond_the_vector_is_a_configuration_fault(self, adapter, stub):
        with pytest.raises(EmbeddingDimensionError):
            adapter.execute_embedding(["x"], MODEL, base_url=stub.url, dimensions=4096)

    def test_both_errors_are_value_errors(self):
        assert issubclass(EmbeddingDimensionError, ValueError)
        assert issubclass(DegenerateVectorError, ValueError)

    def test_an_http_error_is_raised_for_status(self, adapter, stub):
        import requests

        with pytest.raises(requests.HTTPError):
            adapter.execute_embedding(["x"], MODEL, base_url=stub.url + "/missing")


class _FakeResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


class TestTheHostOverride:
    def test_execute_posts_to_the_default_and_never_reads_the_override(self, adapter, monkeypatch):
        import requests

        monkeypatch.setenv("LLAMA_CPP_HOST", "http://h:9000")
        posted: list[str] = []

        def fake_post(url, **_kwargs):
            posted.append(url)
            return _FakeResponse({"data": [{"embedding": [1.0, 0.0]}]})

        def no_get(url, **_kwargs):
            raise AssertionError(f"execute probed {url}")

        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(requests, "get", no_get)

        adapter.execute_image_embedding([(b"x", "image/png")], MODEL, base_url=None)
        assert posted == ["http://localhost:8080/v1/embeddings"]

    def test_health_probes_the_override_first(self, adapter, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_HOST", "http://h:9000")
        probed: list[str] = []

        def fake_probe(url, _path, _timeout):
            probed.append(url)
            return False

        monkeypatch.setattr(_local_server, "probe", fake_probe)
        result = adapter.check_embedding_health(None, None)
        assert probed[0] == "http://h:9000"
        assert probed[1] == "http://localhost:8080"
        assert result.reachable is False
        assert result.reason == "unreachable"


class TestHealth:
    def test_healthy_when_the_server_serves_the_default_model(self, adapter, stub):
        result = adapter.check_embedding_health(None, stub.url)
        assert result.reachable is True
        assert result.reason is None

    def test_a_different_alias_is_reason_model(self, adapter, stub):
        stub.model_id = "some-other-model"
        result = adapter.check_embedding_health(None, stub.url)
        assert result.reachable is False
        assert result.reason == "model"
        assert result.message == (
            f"llama-cpp serves some-other-model, config names {MODEL}: "
            f"start llama-server with --alias {MODEL}"
        )

    def test_a_named_model_is_compared(self, adapter, stub):
        result = adapter.check_embedding_health(None, stub.url, model_id="mine")
        assert result.reason == "model"

    def test_health_never_posts(self, adapter, stub):
        adapter.check_embedding_health(None, stub.url)
        assert set(stub.methods()) == {"GET"}

    def test_a_v1_base_is_a_config_fault(self, adapter):
        result = adapter.check_embedding_health(None, "http://h:8080/v1")
        assert (result.reachable, result.reason) == (False, "config")

    def test_a_request_failure_is_classified_and_redacted(self, adapter, monkeypatch):
        import requests

        monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: True)

        def refuse(_url, **_kwargs):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", refuse)
        result = adapter.check_embedding_health(None, "http://user:secret@h:8080")
        assert result.reachable is False
        assert result.reason == "unreachable"
        assert "secret" not in result.message
        assert "h:8080" in result.message

    def test_an_auth_status_is_reason_auth(self, adapter, stub, monkeypatch):
        monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: True)
        stub.models_status = 401
        result = adapter.check_embedding_health(None, stub.url)
        assert (result.reachable, result.reason) == (False, "auth")


# --- Replay of responses recorded from a real llama-server (b11277) ----------------------

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llama_server"

#: The cosine 'orbit kick near BPM 7' must reach against its own picture after
#: 1024-d truncation: fuse_lanes' floor, so a calibrated floor is checked here.
PROBE_FLOOR = fusion.MIN_SIMILARITY

#: The discriminative probe queries and the picture each must rank first.
#: The RF near-tie (0.405 vs 0.406) is left out: truncation can flip it.
PROBE_NEAREST = {
    "orbit kick near BPM 7": "orbit_kick",
    "horizontal orbit distortion after fill": "orbit_kick",
    "tunnel air temperature drift": "tunnel_temp",
}


def _part_key(part: dict) -> str:
    """sha256 of a content part, serialised canonically (README.md, ``record.part_key``)."""
    canonical = json.dumps(part, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


def _picture(name: str) -> tuple[bytes, str]:
    return (FIXTURES / f"{name}.png").read_bytes(), "image/png"


class _RawResponse:
    """A recorded response body, parsed the way ``requests`` would."""

    def __init__(self, raw: bytes):
        self._raw = raw

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return json.loads(self._raw)


@pytest.fixture
def replay(monkeypatch) -> list[dict]:
    """Serve recorded responses: POSTs keyed on the content part's sha256, GET /v1/models."""
    import requests

    bodies: list[dict] = []

    def fake_post(url, json=None, **_kwargs):
        assert url.endswith("/v1/embeddings")
        bodies.append(json)
        (item,) = json["input"]
        (part,) = item["content"]
        recorded = FIXTURES / "embeddings" / f"{_part_key(part)}.json"
        assert recorded.exists(), f"no recorded response for {part['type']} part"
        return _RawResponse(recorded.read_bytes())

    def fake_get(url, **_kwargs):
        assert url.endswith("/v1/models")
        return _RawResponse((FIXTURES / "models.json").read_bytes())

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: True)
    return bodies


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestReplay:
    def test_replay_the_recorded_files_are_complete(self):
        manifest = _manifest()
        assert set(manifest["pictures"]) == {"orbit_kick", "tunnel_temp"}
        assert set(PROBE_NEAREST) < set(manifest["queries"])
        assert len(manifest["queries"]) == 4
        for key in [*manifest["pictures"].values(), *manifest["queries"].values()]:
            assert (FIXTURES / "embeddings" / f"{key}.json").is_file()

    def test_replay_keys_are_the_sha256_of_the_adapters_content_parts(self):
        from osprey.models.providers.llama_cpp import _content_part

        manifest = _manifest()
        for name, key in manifest["pictures"].items():
            assert _part_key(_content_part(_picture(name))) == key
        for query, key in manifest["queries"].items():
            assert _part_key(_content_part(query)) == key

    def test_replay_recorded_vectors_are_2048_floats(self):
        for path in (FIXTURES / "embeddings").glob("*.json"):
            payload = json.loads(path.read_bytes())
            vector = payload["data"][0]["embedding"]
            assert len(vector) == 2048
            assert all(isinstance(v, float) for v in vector)

    def test_replay_the_adapter_returns_1024_dim_unit_vectors(self, adapter, replay):
        manifest = _manifest()
        inputs = [_picture(name) for name in manifest["pictures"]] + list(manifest["queries"])
        vectors = adapter.execute_image_embedding(
            inputs, MODEL, base_url="http://127.0.0.1:8080", dimensions=1024
        )
        assert len(replay) == len(inputs)
        assert all(body["model"] == MODEL for body in replay)
        for vector in vectors:
            assert len(vector) == 1024
            assert _norm(vector) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.usefixtures("replay")
    def test_replay_the_nearest_picture_matches_the_probe(self, adapter):
        names = list(_manifest()["pictures"])
        pictures = dict(
            zip(
                names,
                adapter.execute_image_embedding(
                    [_picture(name) for name in names],
                    MODEL,
                    base_url="http://127.0.0.1:8080",
                    dimensions=1024,
                ),
                strict=True,
            )
        )
        queries = list(PROBE_NEAREST)
        query_vectors = adapter.execute_embedding(
            queries, MODEL, base_url="http://127.0.0.1:8080", dimensions=1024
        )
        cosines = {
            query: {name: _cosine(vector, pictures[name]) for name in names}
            for query, vector in zip(queries, query_vectors, strict=True)
        }
        for query, expected in PROBE_NEAREST.items():
            assert max(cosines[query], key=cosines[query].get) == expected, cosines[query]
        assert cosines["orbit kick near BPM 7"]["orbit_kick"] >= PROBE_FLOOR

    @pytest.mark.usefixtures("replay")
    def test_replay_models_reports_the_default_alias(self, adapter):
        models = json.loads((FIXTURES / "models.json").read_bytes())
        assert models["data"][0]["id"] == LLAMA_CPP_DEFAULT_MODEL
        result = adapter.check_embedding_health(None, "http://127.0.0.1:8080")
        assert (result.reachable, result.reason) == (True, None)

    def test_replay_readme_names_build_command_and_weights(self):
        readme = (FIXTURES / "README.md").read_text()
        assert "eae11d2" in readme
        assert "b11277" in readme
        assert "build 11277" in readme
        assert f"--alias {LLAMA_CPP_DEFAULT_MODEL}" in readme
        assert "--image-max-tokens 256" in readme
        assert "image_max_pixels:   262144 (custom value)" in readme
        assert re.search(r"model:.*?sha256 `[0-9a-f]{64}`", readme, re.S)
        assert re.search(r"mmproj:.*?sha256 `[0-9a-f]{64}`", readme, re.S)
        hashes = re.findall(r"sha256 `([0-9a-f]{64})`", readme)
        assert len(set(hashes)) >= 2


class TestTheCatalogEntry:
    """The packaged ``llama-cpp`` entry, as a deployment expands and uses it."""

    @staticmethod
    def _entry() -> dict[str, Any]:
        from osprey.profiles.providers import load_provider_catalog

        return dict(load_provider_catalog(None).entries["llama-cpp"])

    @staticmethod
    def _posted_to(adapter, monkeypatch, base_url: str) -> list[str]:
        import requests

        posted: list[str] = []

        def fake_post(url, **_kwargs):
            posted.append(url)
            return _FakeResponse({"data": [{"embedding": [1.0, 0.0]}]})

        monkeypatch.setattr(requests, "post", fake_post)
        adapter.execute_image_embedding([(b"x", "image/png")], MODEL, base_url=base_url)
        return posted

    def test_the_packaged_catalog_loads_with_it(self):
        from osprey.profiles.providers import load_provider_catalog

        assert "llama-cpp" in load_provider_catalog(None).entries

    def test_the_entry_names_the_adapter_default_model(self):
        entry = self._entry()
        assert entry["default_model"] == LLAMA_CPP_DEFAULT_MODEL
        assert entry["models"] == [LLAMA_CPP_DEFAULT_MODEL]
        assert entry["api_key"] == "llama-cpp"

    def test_the_expanded_entry_requests_embeddings_on_the_default_host(self, adapter, monkeypatch):
        from osprey_connectors.config import resolve_env_vars

        base_url = resolve_env_vars(self._entry())["base_url"]
        LlamaCppProviderAdapter.validate_base_url(base_url)
        posted = self._posted_to(adapter, monkeypatch, base_url)
        assert posted == ["http://localhost:8080/v1/embeddings"]

    def test_llama_cpp_host_moves_the_expanded_entry(self, adapter, monkeypatch):
        from osprey_connectors.config import resolve_env_vars

        monkeypatch.setenv("LLAMA_CPP_HOST", "http://h:9000")
        base_url = resolve_env_vars(self._entry())["base_url"]
        assert base_url == "http://h:9000"
        posted = self._posted_to(adapter, monkeypatch, base_url)
        assert posted == ["http://h:9000/v1/embeddings"]

    def test_chat_listings_and_key_tables_omit_it(self):
        from osprey.models.provider_registry import PROVIDER_API_KEYS

        registry = get_provider_registry()
        assert "llama-cpp" not in registry.list_providers(chat_only=True)
        assert "llama-cpp" in registry.list_providers()
        assert "llama-cpp" not in PROVIDER_API_KEYS

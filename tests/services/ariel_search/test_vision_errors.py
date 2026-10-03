"""Tests for :func:`classify_vision_error` against what the llama-cpp adapter raises.

Each failure is produced by the real adapter calling the ``llama_stub`` server,
so the classification is pinned to the exceptions a picture embedding call
actually raises, not to hand-built ones.
"""

from __future__ import annotations

import socket

import pytest
import requests

from osprey.models.providers.base import DegenerateVectorError, EmbeddingDimensionError
from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter
from osprey.services.ariel_search.enhancement import availability
from osprey.services.ariel_search.enhancement.vision_errors import (
    EmptyReplyError,
    classify_vision_error,
    error_signature,
)
from tests.services.ariel_search.llama_stub import MODEL

PICTURE = (b"\x89PNG not really a picture", "image/png")


def _embed(url: str, *, dimensions: int = 1024, timeout: float = 10.0) -> BaseException:
    """The exception one picture embedding call against *url* raises."""
    with pytest.raises(Exception) as caught:
        LlamaCppProviderAdapter().execute_image_embedding(
            [PICTURE], model_id=MODEL, base_url=url, dimensions=dimensions, timeout=timeout
        )
    return caught.value


def _refused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


class TestAgainstLlamaCpp:
    def test_401_is_unavailable_auth(self, llama_stub):
        stub = llama_stub()
        stub.status = 401
        exc = _embed(stub.url)
        assert classify_vision_error(exc) == "unavailable"
        assert availability.unavailable_reason(exc) == "auth"

    def test_404_is_unavailable_model(self, llama_stub):
        stub = llama_stub()
        stub.status = 404
        exc = _embed(stub.url)
        assert classify_vision_error(exc) == "unavailable"
        assert availability.unavailable_reason(exc) == "model"

    def test_400_is_deterministic(self, llama_stub):
        stub = llama_stub()
        stub.embed_status = 400
        exc = _embed(stub.url)
        assert classify_vision_error(exc) == "deterministic"
        assert error_signature(exc) == "HTTPError:400"

    def test_timeout_is_transient(self, llama_stub):
        stub = llama_stub()
        stub.hang = True
        exc = _embed(stub.url, timeout=0.3)
        assert isinstance(exc, requests.Timeout)
        assert classify_vision_error(exc) == "transient"

    def test_connection_refused_is_unavailable_unreachable(self):
        exc = _embed(_refused_url())
        assert classify_vision_error(exc) == "unavailable"
        assert availability.unavailable_reason(exc) == "unreachable"

    def test_malformed_json_is_transient(self, llama_stub):
        stub = llama_stub()
        stub.malformed = True
        exc = _embed(stub.url)
        assert classify_vision_error(exc) == "transient"

    def test_short_vector_is_unavailable_config(self, llama_stub):
        stub = llama_stub()
        stub.short = 16
        exc = _embed(stub.url, dimensions=1024)
        assert isinstance(exc, EmbeddingDimensionError)
        assert classify_vision_error(exc) == "unavailable"
        assert availability.unavailable_reason(exc) == "config"

    def test_zero_vector_is_deterministic(self, llama_stub):
        stub = llama_stub()
        stub.zero_calls = {1}
        exc = _embed(stub.url)
        assert isinstance(exc, DegenerateVectorError)
        assert classify_vision_error(exc) == "deterministic"
        assert error_signature(exc) == "DegenerateVectorError"


class TestClassification:
    def test_requests_timeout_is_transient(self):
        assert classify_vision_error(requests.Timeout("slow")) == "transient"
        assert classify_vision_error(requests.ReadTimeout("slow")) == "transient"

    def test_degenerate_vector_raised_from_another_error_is_deterministic(self):
        try:
            try:
                raise DegenerateVectorError("zero norm")
            except DegenerateVectorError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            assert classify_vision_error(outer) == "deterministic"

    def test_empty_reply_stays_deterministic(self):
        assert classify_vision_error(EmptyReplyError("")) == "deterministic"

    def test_unrecognised_exception_is_transient(self):
        assert classify_vision_error(KeyError("data")) == "transient"
        assert classify_vision_error(ValueError("no embedding")) == "transient"

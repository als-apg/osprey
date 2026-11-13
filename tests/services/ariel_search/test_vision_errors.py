"""Tests for :func:`classify_vision_error` against what the llama-cpp adapter raises.

Each failure is produced by the real adapter calling the ``llama_stub`` server,
so the classification is pinned to the exceptions a picture embedding call
actually raises, not to hand-built ones.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

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


def _refused_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


class TestAgainstLlamaCpp:
    @pytest.mark.parametrize(
        ("status", "reason"),
        [pytest.param(401, "auth", id="401-auth"), pytest.param(404, "model", id="404-model")],
    )
    def test_an_auth_or_model_status_is_unavailable_with_its_reason(
        self, llama_stub, status, reason
    ):
        stub = llama_stub()
        stub.status = status
        exc = _embed(stub.url)
        assert classify_vision_error(exc) == "unavailable"
        assert availability.unavailable_reason(exc) == reason

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

    def test_connection_refused_is_unavailable_unreachable(self, reserve_port):
        exc = _embed(_refused_url(reserve_port()))
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

    def test_classifying_imports_no_library(self):
        """The picture lane classifies its failure on the event loop.

        A cold LiteLLM import there stalls every request the loop serves for
        seconds. Run in a fresh interpreter: in this one, another test has
        usually imported every library already.
        """
        script = textwrap.dedent(
            """
            import sys
            from osprey.services.ariel_search.enhancement import availability
            from osprey.services.ariel_search.enhancement.vision_errors import (
                classify_vision_error,
            )
            from osprey.services.ariel_search.search import image_lane  # noqa: F401

            libraries = ("litellm", "httpx", "requests", "psycopg", "psycopg_pool")
            assert "litellm" not in sys.modules, "importing the lane loaded litellm"
            before = {name for name in libraries if name in sys.modules}
            for exc in (TimeoutError("x"), ConnectionError("x"), ValueError("x")):
                availability.unavailable_reason(exc)
                classify_vision_error(exc)
            loaded = {name for name in libraries if name in sys.modules}
            assert loaded == before, f"classifying imported {sorted(loaded - before)}"
            """
        )
        subprocess.run([sys.executable, "-c", script], check=True, timeout=60)

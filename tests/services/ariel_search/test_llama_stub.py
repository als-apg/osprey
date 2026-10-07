"""The picture path against the shared ``llama_stub`` llama-server stand-in.

One HTTP round trip through the real llama-cpp adapter covers the transport:
one picture and one query, each one POST in the content-parts dialect, each
answered from the recordings of ``tests/fixtures/llama_server`` and returned as
a unit-norm 1024-d vector. Which picture ranks nearest a query is asserted once,
by the replay tests of the recordings, never here.

The rest is what a deployment without a working server sees, with the
repository doubles of the neighbouring tests:

* no server listening: ``status`` reports ``image_embedding`` as
  ``reachable: false, reason: unreachable`` and ``hybrid_search`` answers its
  text ranking with a "Picture search unavailable" diagnostic;
* a server whose ``/v1/models`` names another id: the catch-up driver skips the
  module and leaves every status untouched.

Every assertion is value-agnostic: content with no recording embeds to a
deterministic vector, so the tests hold against re-captured recordings too.
"""

from __future__ import annotations

import json
import math
import socket
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.models.providers import _local_server
from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.image_driver import drive_image_module
from osprey.services.ariel_search.enhancement.image_embedding import module as embed_mod
from osprey.services.ariel_search.search.base import ModuleOutput
from osprey.services.ariel_search.search.qmd import hybrid_search
from tests.services.ariel_search.llama_stub import FIXTURES, MODEL, WIDTH, part_key
from tests.services.ariel_search.test_hybrid_search import (
    LaneRepository,
    StubClient,
    lane_config,
    make_entry,
    make_hit,
)
from tests.services.ariel_search.test_image_embedding import FakeRepo, _module

DIMS = 1024
PICTURE = "orbit_kick"
QUERY = "orbit kick near BPM 7"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()
    monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: True)
    yield
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()


def _nobody_listening() -> str:
    """A 127.0.0.1 URL whose port was free a moment ago and is not listened on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_one_picture_and_one_query_are_unit_norm_1024_d_vectors(self, llama_stub):
        stub = llama_stub()
        picture = ((FIXTURES / f"{PICTURE}.png").read_bytes(), "image/png")

        vectors = LlamaCppProviderAdapter().execute_image_embedding(
            [picture, QUERY], MODEL, base_url=stub.url, dimensions=DIMS, timeout=30.0
        )

        # One POST per input, in the content-parts dialect, naming the served alias.
        assert len(stub.embeddings) == 2
        parts = []
        for request in stub.embeddings:
            assert request["model"] == MODEL
            (item,) = request["input"]
            (part,) = item["content"]
            parts.append(part)
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert parts[1] == {"type": "text", "text": QUERY}

        # Both parts are recorded ones: the stub replayed the server's own answers.
        keys = [part_key(part) for part in parts]
        assert keys == [MANIFEST["pictures"][PICTURE], MANIFEST["queries"][QUERY]]

        assert len(vectors) == 2
        for vector, key in zip(vectors, keys, strict=True):
            assert len(vector) == DIMS
            assert _norm(vector) == pytest.approx(1.0, abs=1e-9)
            recorded = json.loads((FIXTURES / "embeddings" / f"{key}.json").read_text())
            full = recorded["data"][0]["embedding"]
            assert len(full) == WIDTH
            prefix = full[:DIMS]
            scale = _norm(prefix)
            assert vector == pytest.approx([v / scale for v in prefix])


# ---------------------------------------------------------------------------
# No server listening
# ---------------------------------------------------------------------------


class _StatusService:
    """``create_ariel_service`` stand-in: a healthy database behind *repository*."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    async def __aenter__(self) -> _StatusService:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def health_check(self) -> tuple[bool, str]:
        return True, "connected"


def _status_repository() -> MagicMock:
    """A repository double answering every query ``get_status`` makes.

    Its pool is the in-memory one of the ``image_embedding`` tests, so the
    picture table the pre-pass check looks up exists.
    """
    repo = MagicMock()
    repo.pool = FakeRepo().pool
    repo.get_enhancement_stats = AsyncMock(return_value={"total_entries": 0})
    repo.get_embedding_tables = AsyncMock(return_value=[])
    repo.get_image_embedding_tables = AsyncMock(return_value=[])
    repo.get_last_ingestion = AsyncMock(return_value=None)
    repo.schema_facts = AsyncMock(return_value=SchemaFacts(has_v2_fts=True, has_copy_state=True))
    repo.get_attachment_bytes = AsyncMock(return_value=0)
    repo.get_attachment_copy_counts = AsyncMock(return_value=(0, {}))
    return repo


class TestNoServerListening:
    async def test_status_reports_image_embedding_unreachable(self, llama_stub, monkeypatch):
        import osprey.services.ariel_search as ariel_pkg

        llama_stub
        url = _nobody_listening()
        service = _StatusService(_status_repository())

        async def _create(_config: Any) -> _StatusService:
            return service

        monkeypatch.setattr(ariel_pkg, "create_ariel_service", _create)
        monkeypatch.setattr(
            "osprey.imaging.render.probe_render_worker", AsyncMock(return_value=True)
        )
        config = {
            "database": {"uri": "postgresql://localhost/ariel"},
            "search_modules": {"hybrid": {"enabled": True}},
            "enhancement_modules": {
                "image_embedding": {
                    "enabled": True,
                    "provider": {"name": "llama-cpp", "base_url": url},
                    "model": MODEL,
                    "dimensions": DIMS,
                }
            },
        }

        out = await ops.get_status(config)

        assert out["status"] == "healthy", out
        health = out["enhancement_modules"]["image_embedding"]["health"]
        assert health["reachable"] is False
        assert health["reason"] == "unreachable"
        assert out["attachments"]["picture_search_unavailable"] == "unreachable"

    async def test_hybrid_search_answers_text_only(self, llama_stub):
        llama_stub
        config = lane_config(_nobody_listening())
        hits = [make_hit("a", score=0.9), make_hit("b", score=0.5)]

        def repo() -> LaneRepository:
            return LaneRepository([make_entry("a"), make_entry("b")])

        text_only = await hybrid_search(
            "q", repo(), config, client=StubClient(hits), include_images=False
        )
        result = await hybrid_search("q", repo(), config, client=StubClient(hits))

        assert isinstance(result, ModuleOutput)
        assert list(result.entries) == list(text_only)
        unavailable = [
            d for d in result.diagnostics if d.message.startswith("Picture search unavailable")
        ]
        assert len(unavailable) == 1


# ---------------------------------------------------------------------------
# A server serving another model
# ---------------------------------------------------------------------------


class TestAnotherModel:
    async def test_driver_skips_the_module_and_leaves_statuses_untouched(self, llama_stub):
        stub = llama_stub()
        stub.alias = "another-model"
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        repo.add_entry("e2", 2)

        result = await drive_image_module(_module(stub.url), repo, budget=None, stop_event=None)

        assert result.skipped == "unavailable"
        assert result.ended == "model"
        assert result.entries_walked == 0
        assert stub.models_gets >= 1
        assert stub.embeddings == []
        assert repo.status_writes == []
        assert repo.batch_marks == 0
        assert repo.failed == {}
        assert all(e["enhancement_status"] == {} for e in repo.entries.values())
        assert repo.vectors == {}

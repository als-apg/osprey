"""The picture lane of hybrid search against a real pgvector database.

Every test runs on a fresh scratch database of the session server, migrated
with a llama-cpp ``image_embedding`` block so the image table exists. Picture
and query vectors are the recorded PROBE embeddings, replayed through the
llama-cpp adapter by :class:`~tests.services.ariel_search.llama_stub.LlamaStub`
at ``dimensions=1024``; qmd's text ranking is a canned stub client.

The lane sets no version-dependent setting and sizes ``hnsw.ef_search`` to at
least the rows it asks for, so it is pinned on the deployed
``pgvector/pgvector:pg16`` image only; :func:`server` refuses any other major
version instead of passing against a different dev database.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from osprey.models.providers.llama_cpp import LlamaCppProviderAdapter
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.repository import ARIELRepository
from osprey.services.ariel_search.database.vector_literal import vector_literal
from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import ingest_one
from osprey.services.ariel_search.search import image_lane
from osprey.services.ariel_search.search.qmd import hybrid_search
from osprey.services.qmd import QMDSearchResult
from tests.services.ariel_search.llama_stub import FIXTURES, LlamaStub
from tests.services.ariel_search.llama_stub import MODEL as LLAMA_MODEL

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

#: The deployed server's major version (``pgvector/pgvector:pg16``).
DEPLOYED_MAJOR = 16

#: The recorded width the adapter truncates the 2048-float recordings to.
DIMS = 1024

#: The recorded probe query whose nearest picture is ``orbit_kick``.
QUERY = "orbit kick near BPM 7"


def _manifest() -> dict[str, Any]:
    return json.loads((FIXTURES / "manifest.json").read_text())


def _picture(name: str) -> tuple[bytes, str]:
    return (FIXTURES / f"{name}.png").read_bytes(), "image/png"


def _config(uri: str, stub_url: str) -> ARIELConfig:
    """Hybrid on, a llama-cpp picture block at the stub, ingest copies nothing."""
    return ARIELConfig.from_dict(
        {
            "database": {"uri": uri},
            "attachments": {"copy_on_ingest": "none"},
            "search_modules": {"hybrid": {"enabled": True}},
            "enhancement_modules": {
                "image_embedding": {
                    "enabled": True,
                    "provider": {"name": "llama-cpp", "base_url": stub_url},
                    "model": LLAMA_MODEL,
                    "dimensions": DIMS,
                }
            },
        }
    )


class StubClient:
    """A qmd client answering a canned ranking, as the sidecar would report it."""

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def __init__(self, entry_ids: list[str]) -> None:
        self._hits = [
            QMDSearchResult(
                docid=f"#{index:06x}",
                file=f"2024/06/{encode_entry_id(entry_id)}.md",
                collection="ariel",
                title=f"Entry {entry_id}",
                score=1.0 - index / 100,
                line=1,
                snippet="1: text match",
            )
            for index, entry_id in enumerate(entry_ids)
        ]

    def is_available(self) -> bool:
        return True

    def query(self, _collection: str | None, _text: str, **kwargs: Any) -> list[QMDSearchResult]:
        """The canned ranking, cut at ``limit``; the query text is ignored."""
        return list(self._hits[: kwargs.get("limit", len(self._hits))])


@dataclass
class Lane:
    """One scratch database ready for the picture lane."""

    uri: str
    config: ARIELConfig
    repository: ARIELRepository
    vectors: dict[str, list[float]]

    @property
    def table(self) -> str:
        return image_lane.ImageLaneSettings.from_ariel_config(self.config).table

    def entry(self, entry_id: str, *, author: str = "operator", text: str = "text") -> None:
        with psycopg.connect(self.uri, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO enhanced_entries (entry_id, source_system, timestamp, author, raw_text)
                VALUES (%s, 'test', %s, %s, %s)
                """,
                (entry_id, datetime(2024, 6, 1, 12, tzinfo=UTC), author, text),
            )

    def picture(self, attachment_id: str, entry_id: str, vector: list[float]) -> None:
        with psycopg.connect(self.uri, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO attachment_files (
                    attachment_id, entry_id, filename, mime_type, source_url,
                    copy_status, rendition_sha256
                ) VALUES (%s, %s, 'f.png', 'image/png', %s, 'copied', %s)
                """,
                (attachment_id, entry_id, f"https://h.example/{attachment_id}.png", "ab" * 32),
            )
        self.vector(attachment_id, vector)

    def vector(self, attachment_id: str, vector: list[float]) -> None:
        with psycopg.connect(self.uri, autocommit=True) as conn:
            conn.execute(
                f"INSERT INTO {self.table} (attachment_id, embedding, model_ref)"
                " VALUES (%(id)s, %(v)s::vector, %(m)s)",
                {"id": attachment_id, "v": vector_literal(vector), "m": LLAMA_MODEL},
            )

    def vector_ids(self) -> set[str]:
        with psycopg.connect(self.uri) as conn:
            rows = conn.execute(f"SELECT attachment_id FROM {self.table}").fetchall()
        return {row[0] for row in rows}

    async def search(self, text_ranking: list[str], **kwargs: Any) -> list[tuple[Any, ...]]:
        result = await hybrid_search(
            QUERY, self.repository, self.config, client=StubClient(text_ranking), **kwargs
        )
        assert isinstance(result, list), (
            f"picture lane failed: {image_lane.last_unavailable_reason()!r} {result!r}"
        )
        return result


@pytest.fixture
def server(database_url: str, record_property: Callable[[str, Any], None]) -> str:
    """The session server, refused unless it is the deployed major version.

    Records the server's pgvector version in the test output.
    """
    with psycopg.connect(database_url) as conn:
        major = conn.execute("SELECT current_setting('server_version_num')::int / 10000").fetchone()
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    extversion = ext[0] if ext else None
    record_property("pgvector_extversion", extversion)
    print(f"server major={major[0] if major else None} pgvector={extversion}")
    assert major is not None and major[0] == DEPLOYED_MAJOR, (
        f"picture-lane tests are pinned to PostgreSQL {DEPLOYED_MAJOR} "
        f"(pgvector/pgvector:pg16); this server is {major[0] if major else '?'}"
    )
    return database_url


@pytest.fixture
async def lane(
    server: str,  # noqa: ARG001 - requested for its version guard
    scratch_database: str,
    llama_stub: Callable[[], LlamaStub],
) -> AsyncIterator[Lane]:
    """A migrated scratch database, the stub server and the recorded picture vectors."""
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    stub = llama_stub()
    config = _config(scratch_database, stub.url)
    adapter = LlamaCppProviderAdapter()
    names = list(_manifest()["pictures"])
    vectors = dict(
        zip(
            names,
            adapter.execute_image_embedding(
                [_picture(name) for name in names],
                LLAMA_MODEL,
                base_url=stub.url,
                dimensions=DIMS,
            ),
            strict=True,
        )
    )
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
        yield Lane(scratch_database, config, ARIELRepository(pool, config), vectors)
    finally:
        await pool.close()


def _row(result: list[tuple[Any, ...]], entry_id: str) -> dict[str, Any]:
    for entry, _score, _snippets in result:
        if entry["entry_id"] == entry_id:
            return entry
    raise AssertionError(f"{entry_id} not in {[entry['entry_id'] for entry, *_ in result]}")


class TestImageLane:
    async def test_image_only_hit_names_its_nearest_picture(self, lane: Lane) -> None:
        lane.entry("text-1")
        lane.entry("orbit")
        lane.picture("pic-orbit", "orbit", lane.vectors["orbit_kick"])

        result = await lane.search(["text-1"])

        row = _row(result, "orbit")
        assert row["_matched_via"] == ["image"]
        assert row["_matched_attachment_ids"] == ["pic-orbit"]

    async def test_entry_with_two_pictures_names_the_closer_one(self, lane: Lane) -> None:
        lane.entry("two")
        lane.picture("pic-tunnel", "two", lane.vectors["tunnel_temp"])
        lane.picture("pic-orbit", "two", lane.vectors["orbit_kick"])

        result = await lane.search([])

        row = _row(result, "two")
        assert row["_matched_attachment_ids"] == ["pic-orbit"]

    async def test_filters_apply_to_image_hits(self, lane: Lane) -> None:
        lane.entry("kept", author="alice")
        lane.entry("dropped", author="bob")
        lane.picture("pic-kept", "kept", lane.vectors["orbit_kick"])
        lane.picture("pic-dropped", "dropped", lane.vectors["orbit_kick"])

        unfiltered = await lane.search([])
        filtered = await lane.search([], author="alice")

        assert {entry["entry_id"] for entry, *_ in unfiltered} == {"kept", "dropped"}
        assert [entry["entry_id"] for entry, *_ in filtered] == ["kept"]

    async def test_reingest_drops_the_removed_pictures_vector(self, lane: Lane) -> None:
        entry_id = "reingest"
        url = {name: f"https://h.example/files/{name}.png" for name in "abc"}
        ids = {name: attachment_id_for(entry_id, {"url": u}) for name, u in url.items()}
        adapter = _Adapter(lane.config)

        async def ingest(names: str) -> None:
            entry = {
                "entry_id": entry_id,
                "source_system": "test",
                "timestamp": datetime.now(UTC),
                "author": "tester",
                "raw_text": "beam lost",
                "attachments": [
                    {"url": url[n], "type": "image/png", "filename": f"{n}.png"} for n in names
                ],
                "metadata": {},
                "enhancement_status": {},
            }
            await ingest_one(entry, adapter, lane.repository, [], lane.config, None)  # type: ignore[arg-type]

        await ingest("ab")
        lane.vector(ids["a"], lane.vectors["orbit_kick"])
        lane.vector(ids["b"], lane.vectors["tunnel_temp"])
        assert lane.vector_ids() == {ids["a"], ids["b"]}

        await ingest("ac")

        assert lane.vector_ids() == {ids["a"]}

    async def test_picture_match_lifts_text_rank_two_to_first(self, lane: Lane) -> None:
        lane.entry("text-top", text="unrelated top text hit")
        lane.entry("orbit", text="orbit entry")
        lane.entry("tunnel", text="tunnel entry")
        lane.picture("pic-orbit", "orbit", lane.vectors["orbit_kick"])
        lane.picture("pic-tunnel", "tunnel", lane.vectors["tunnel_temp"])

        result = await lane.search(["text-top", "orbit"])

        top = result[0][0]
        assert top["entry_id"] == "orbit"
        assert "image" in top["_matched_via"]
        assert "pic-orbit" in top["_matched_attachment_ids"]


class _Adapter(FacilityAdapter):
    """An http source whose entries the tests hand to ``ingest_one`` directly."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    async def fetch_entries(self, *_args: Any, **_kwargs: Any) -> AsyncIterator:  # type: ignore[override]
        for entry in ():  # never yields
            yield entry

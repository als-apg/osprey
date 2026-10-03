"""Shared fixtures for ARIEL MCP server tool tests.

Provides mock ARIEL service, entry factories, and registry management.

IMPORTANT: FastMCP's @mcp.tool() decorator wraps functions into FunctionTool
objects. To call the original async function in tests, use the `.fn` attribute:
    tool.fn(query="beam loss")
"""

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import reset_ariel_context
from osprey.utils.workspace import reset_config_cache
from tests.mcp_server.conftest import get_tool_fn  # noqa: F401
from tests.services.ariel_search.conftest import (
    install_attachment_fetch_fake,
    reset_image_lane_state,
)
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider


@pytest.fixture(autouse=True)
def attachment_fetch(request, monkeypatch):
    """No ARIEL tool test reaches a real attachment source unless it opts in.

    The same fake as the ARIEL service tests: unopted fetches fail the test at
    teardown; ``real_fetch`` opts out.
    """
    fake = install_attachment_fetch_fake(request, monkeypatch)
    yield fake
    if fake is not None:
        fake.verify()


@pytest.fixture(autouse=True)
def _reset_image_lane():
    """Every ARIEL tool test starts and ends with a closed, reason-free picture lane."""
    reset_image_lane_state()
    yield
    reset_image_lane_state()


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the ARIEL MCP registry singletons between tests."""
    yield
    reset_ariel_context()
    reset_config_cache()


def make_mock_entry(
    entry_id="test-001",
    source_system="Example eLog",
    author="Test User",
    raw_text="Test entry content",
    summary=None,
    timestamp=None,
    logbook=None,
    shift=None,
    tags=None,
    score=None,
    attachments=None,
):
    """Create a mock EnhancedLogbookEntry dict (TypedDict, not MagicMock).

    Returns a plain dict matching the EnhancedLogbookEntry TypedDict shape.
    """
    now = timestamp or datetime(2024, 1, 15, 10, 30, 0)
    entry = {
        "entry_id": entry_id,
        "source_system": source_system,
        "timestamp": now,
        "author": author,
        "raw_text": raw_text,
        "attachments": attachments if attachments is not None else [],
        "metadata": {
            "logbook": logbook,
            "shift": shift,
            "tags": tags or [],
        },
        "created_at": now,
        "updated_at": now,
    }
    if summary is not None:
        entry["summary"] = summary
    if score is not None:
        entry["_score"] = score
    return entry


def attach_fake_attachment_reader(
    service: Any,
    rows: dict[str, list[dict[str, Any]]] | None = None,
    *,
    unmigrated: bool = False,
    error: Exception | None = None,
) -> Any:
    """Give an ``AsyncMock`` service's repository real attachment readers.

    A bare ``AsyncMock`` would answer ``get_attachment_rows`` with a
    ``MagicMock``, which ``serialize_entries`` refuses; ``caption_matches``
    answers ``{}`` (no caption matched) for the same reason. The store reads as
    migrated (``SchemaFacts(has_v2_fts=True, has_copy_state=True)``), so no search
    answered through it reports the schema as behind.

    Args:
        service: The mocked service whose ``repository`` gets the readers.
        rows: The ``{entry_id: [row, ...]}`` mapping the reader answers;
            ``{}`` (no rows for any entry) when omitted.
        unmigrated: Answer None, as a store without copy state does.
        error: Raise this from the reader instead of answering.

    Returns:
        The ``get_attachment_rows`` mock, for call assertions.
    """
    from osprey.services.ariel_search.database.repository import SchemaFacts

    if error is not None:
        reader = AsyncMock(side_effect=error)
    elif unmigrated:
        reader = AsyncMock(return_value=None)
    else:
        reader = AsyncMock(return_value=rows if rows is not None else {})
    service.repository.get_attachment_rows = reader
    service.repository.get_rendition = AsyncMock(return_value=None)
    service.repository.caption_matches = AsyncMock(return_value={})
    service.repository.schema_facts = AsyncMock(
        return_value=SchemaFacts(has_v2_fts=not unmigrated, has_copy_state=not unmigrated)
    )
    return reader


@pytest.fixture
def mock_ariel_service():
    """Create a mock ARIELSearchService with AsyncMock methods."""
    service = AsyncMock()
    service.repository = AsyncMock()
    service.config = MagicMock()
    service.pool = AsyncMock()
    attach_fake_attachment_reader(service)
    return service


# ---------------------------------------------------------------------------
# Keyset harness: a real ARIELSearchService over a fake repository
# ---------------------------------------------------------------------------
#
# ``test_tool_keysets.py`` consumes these fixtures; it never declares fakes of
# its own. A task that adds a repository call extends ``KeysetRepository`` here,
# so the harness file and its goldens stay untouched.

KEYSET_ENTRY_ID = "keyset-001"
KEYSET_SECOND_ENTRY_ID = "keyset-002"
KEYSET_EMBEDDING_MODEL = "nomic-embed-text"
KEYSET_VECTOR = [0.125, 0.25, 0.5, 1.0]

KEYSET_CONFIG: dict[str, Any] = {
    "ariel": {
        "database": {"uri": "postgresql://localhost/test"},
        "entry_url_template": "https://elog.example/entry/{entry_id}",
        "search_modules": {
            "keyword": {"enabled": True},
            "semantic": {"enabled": True, "model": KEYSET_EMBEDDING_MODEL},
            "hybrid": {"enabled": True},
        },
        "enhancement_modules": {
            "text_embedding": {
                "enabled": True,
                "models": [{"name": KEYSET_EMBEDDING_MODEL, "dimension": len(KEYSET_VECTOR)}],
            }
        },
    }
}

_KEYSET_TIME = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

# Long enough to be cut at both the listing and the read text budgets.
_KEYSET_LONG_TEXT = "Beam lost at injection after the kicker timing drifted. " + " ".join(
    f"Step {i}: checked BPM {i} orbit and corrector settings." for i in range(60)
)


def keyset_entries() -> list[dict[str, Any]]:
    """Return fresh copies of the two seeded ``enhanced_entries`` rows.

    The first carries everything a later phase reshapes: long text, a summary,
    metadata tags and two stored attachments (one PNG, one PDF). The second is
    a short entry with no attachments.
    """
    seeded = {
        "entry_id": KEYSET_ENTRY_ID,
        "source_system": "Example eLog",
        "timestamp": _KEYSET_TIME,
        "author": "operator",
        "raw_text": _KEYSET_LONG_TEXT,
        "attachments": [
            {
                "url": "https://elog.example/files/orbit-plot.png",
                "type": "image/png",
                "filename": "orbit-plot.png",
            },
            {
                "url": "https://elog.example/files/report.pdf",
                "type": "application/pdf",
                "filename": "report.pdf",
            },
        ],
        "metadata": {"logbook": "Operations", "shift": "day", "tags": ["injection", "orbit"]},
        "created_at": _KEYSET_TIME,
        "updated_at": _KEYSET_TIME,
        "summary": "Injection beam loss traced to kicker timing drift.",
    }
    second = {
        "entry_id": KEYSET_SECOND_ENTRY_ID,
        "source_system": "Example eLog",
        "timestamp": datetime(2024, 6, 2, 8, 30, tzinfo=UTC),
        "author": "physicist",
        "raw_text": "Kicker timing restored; injection efficiency back to nominal.",
        "attachments": [],
        "metadata": {"logbook": "Operations", "shift": "owl", "tags": []},
        "created_at": datetime(2024, 6, 2, 8, 30, tzinfo=UTC),
        "updated_at": datetime(2024, 6, 2, 8, 30, tzinfo=UTC),
        "summary": "Kicker timing fixed.",
    }
    return [seeded, second]


#: A PNG signature standing in for the stored rendition of the seeded picture.
KEYSET_RENDITION_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def keyset_attachment_rows() -> list[dict[str, Any]]:
    """Return the ``attachment_files`` rows of the seeded entry, without blobs.

    The PNG is copied with a rendition (viewable); the PDF is skipped by the
    ``copy_on_ingest`` mode, as the default ``images`` mode records it.
    """
    from osprey.services.ariel_search.attachments import attachment_id_for

    png, pdf = keyset_entries()[0]["attachments"]
    base = {
        "entry_id": KEYSET_ENTRY_ID,
        "copy_attempts": 1,
        "rendition_mime": None,
        "rendition_w": None,
        "rendition_h": None,
        "rendition_sha256": None,
    }
    return [
        {
            **base,
            "attachment_id": attachment_id_for(KEYSET_ENTRY_ID, png),
            "filename": png["filename"],
            "mime_type": "image/png",
            "size_bytes": 2048,
            "source_url": png["url"],
            "copy_status": "copied",
            "skip_reason": None,
            "rendition_mime": "image/png",
            "rendition_w": 640,
            "rendition_h": 480,
            "rendition_sha256": "0" * 64,
        },
        {
            **base,
            "attachment_id": attachment_id_for(KEYSET_ENTRY_ID, pdf),
            "filename": pdf["filename"],
            "mime_type": "application/pdf",
            "size_bytes": None,
            "source_url": pdf["url"],
            "copy_status": "skipped",
            "skip_reason": "copy_on_ingest_mode",
        },
    ]


class KeysetRepository:
    """In-memory stand-in for ``ARIELRepository`` answering with real rows.

    A plain class rather than a mock, so nothing a tool serializes can be a
    ``Mock`` object stringified by ``json.dumps(default=str)``.
    """

    def __init__(self) -> None:
        self._rows = {row["entry_id"]: row for row in keyset_entries()}
        self.attachment_row_calls: list[list[str]] = []

    def _row(self, entry_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._rows[entry_id])

    async def validate_search_model_table(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def health_check(self) -> tuple[bool, str]:
        return True, "OK"

    async def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        return self._row(entry_id) if entry_id in self._rows else None

    async def get_entries_by_ids(self, entry_ids: list[str]) -> list[dict[str, Any]]:
        wanted = set(entry_ids)
        return [self._row(eid) for eid in sorted(self._rows) if eid in wanted]

    async def search_by_time_range(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [self._row(KEYSET_ENTRY_ID)]

    async def count_entries(self, *args: Any, **kwargs: Any) -> int:
        return 1

    async def keyword_search(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[dict[str, Any], float, list[str]]]:
        return [
            (self._row(KEYSET_ENTRY_ID), 0.75, ["<b>injection</b> after the kicker"]),
            (self._row(KEYSET_SECOND_ENTRY_ID), 0.5, ["<b>injection</b> efficiency"]),
        ]

    async def fuzzy_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def caption_matches(self, *args: Any, **kwargs: Any) -> dict[str, list[str]]:
        return {}

    async def semantic_search(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[dict[str, Any], float]]:
        return [(self._row(KEYSET_ENTRY_ID), 0.875)]

    async def get_embedding_tables(self) -> list[Any]:
        return []

    async def schema_facts(self) -> Any:
        from osprey.services.ariel_search.database.repository import SchemaFacts

        return SchemaFacts(has_v2_fts=True, has_copy_state=True)

    async def get_attachment_rows(self, entry_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        self.attachment_row_calls.append(list(entry_ids))
        if KEYSET_ENTRY_ID not in entry_ids:
            return {}
        return {KEYSET_ENTRY_ID: keyset_attachment_rows()}

    async def get_rendition(self, attachment_id: str) -> dict[str, Any] | None:
        for row in keyset_attachment_rows():
            if row["attachment_id"] == attachment_id and row["rendition_sha256"] is not None:
                return {**row, "rendition_bytes": KEYSET_RENDITION_BYTES}
        return None


class KeysetQMDClient:
    """qmd client faked at its client boundary: both seeded entries, fixed scores."""

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def is_available(self) -> bool:
        return True

    def query(self, collection: str | None, text: str, **kwargs: Any) -> list[Any]:  # noqa: ARG002
        from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
        from osprey.services.qmd.client import QMDSearchResult

        hits = [
            (KEYSET_ENTRY_ID, 0.9, "beam lost at injection"),
            (KEYSET_SECOND_ENTRY_ID, 0.6, "kicker timing restored"),
        ]
        return [
            QMDSearchResult(
                docid=f"#{index:06d}",
                file=f"2024/06/{encode_entry_id(entry_id)}.md",
                collection=collection or "ariel",
                title=f"Entry {entry_id}",
                score=score,
                line=1,
                snippet=f"1: {snippet}",
            )
            for index, (entry_id, score, snippet) in enumerate(hits)
        ]


@dataclass
class KeysetHarness:
    """What the keyset harness hands each test."""

    service: Any
    repository: KeysetRepository
    config: Any


def keyset_config(*, view_enabled: bool = True) -> dict[str, Any]:
    """Return a fresh copy of ``KEYSET_CONFIG``, with the view switch set when off."""
    config = copy.deepcopy(KEYSET_CONFIG)
    if not view_enabled:
        config["ariel"]["attachments"] = {"view": {"enabled": False}}
    return config


@pytest.fixture
def keyset_harness(request, tmp_path, monkeypatch):
    """A real ``ARIELSearchService`` behind the ARIEL MCP tools.

    Writes one ``ariel`` config to ``config.yml`` in ``tmp_path``, initializes
    the MCP context from it, and builds the service from the same dict, so the
    ``capabilities`` tool (context config) and the service see one configuration.
    The service is injected through ``ARIELContext.service``; no pool opens.

    Indirect parametrization with ``{"view_enabled": False}`` writes
    ``ariel.attachments.view.enabled: false`` into that one configuration; the
    default is the key's default (on).
    """
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.search import qmd as qmd_module
    from osprey.services.ariel_search.service import ARIELSearchService

    params = getattr(request, "param", None) or {}
    raw = keyset_config(view_enabled=params.get("view_enabled", True))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(json.dumps(raw))
    initialize_ariel_context()

    config = ARIELConfig.from_dict(copy.deepcopy(raw["ariel"]))
    repository = KeysetRepository()
    pool = MagicMock()
    pool.close = AsyncMock()
    service = ARIELSearchService(config=config, pool=pool, repository=repository)
    service._embedder = make_fake_embedding_provider(vector=KEYSET_VECTOR)()

    monkeypatch.setattr(qmd_module, "_cached_client", KeysetQMDClient())
    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=service),
    ):
        yield KeysetHarness(service=service, repository=repository, config=config)

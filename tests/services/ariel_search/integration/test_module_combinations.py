"""Every combination of the ARIEL picture units is a valid deployment.

The units switch independently: ``attachments.copy_on_ingest`` (``images`` or
``none``), the ``image_caption`` and ``image_embedding`` enhancement modules,
the ``hybrid`` search module and ``ariel.attachments.view.enabled``. One case
per combination (32) runs the whole path on a fresh scratch database: one
ingest of a two-picture entry, one catch-up pass, ``keyword_search``,
``hybrid_search`` (when hybrid is on), ``capabilities``, the listings and
``osprey ariel status --json``, and pins that

* nothing raises;
* ``picture_search`` is ``image_embedding and hybrid``, in ``capabilities``
  and in the status document alike;
* with ``copy_on_ingest: none`` every enabled picture module is complete after
  that one pass;
* a disabled module costs no provider call;
* ``capabilities().attachments.view`` is the view switch; with the view off
  ``tools/list`` has no ``attachment_view``, no listing carries attachment
  keys, and captions and the picture lane still work; with the view on and
  nothing copied, ``attachment_view`` answers ``no_results`` naming
  ``copy_on_ingest_mode``.

Fakes, at their boundaries: qmd's ``_resolve_client`` returns a canned client
whose hits are titled ``Entry <entry_id>``; the attachment fetch fake answers
the two recorded probe pictures; the vision call of ``image_caption`` is a
counting fake; ``image_embedding`` runs the real llama-cpp adapter against the
shared ``llama_stub``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import pytest

from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.attachments import copy as copy_mod
from osprey.services.ariel_search.attachments.fetch import FetchOutcome
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.image_caption import module as caption_mod
from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import ingest_one
from osprey.services.ariel_search.search import qmd as qmd_module
from osprey.services.qmd import QMDSearchResult
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.services.ariel_search.llama_stub import FIXTURES
from tests.services.ariel_search.llama_stub import MODEL as LLAMA_MODEL

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

DIMS = 1024
EMBED = "image_embedding"
CAPTION = "image_caption"
PICTURE_MODULES = (CAPTION, EMBED)
CAPTION_MODEL = "vis-caption-1"
ORIGINS = frozenset({("https", "h.example", 443)})
TS = datetime(2026, 9, 1, tzinfo=UTC)
ENTRY_ID = "combo-1"
RAW_TEXT = "beam lost at 14:02"
PICTURES = ("orbit_kick", "tunnel_temp")
#: The recorded probe query whose nearest picture is ``orbit_kick``.
QUERY = "orbit kick near BPM 7"
REPLY = "Klystron arc trace on the scope.\nVisible text: KLY-3"
#: A word only the caption carries.
CAPTION_WORD = "klystron"
ATTACHMENT_KEYS = {"attachments", "attachment_count", "matched_attachment_ids"}


# --- fakes -------------------------------------------------------------------------


class _Adapter(FacilityAdapter):
    """An http source whose pictures live on ``h.example``."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    def attachment_origins(self) -> frozenset:
        return ORIGINS

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():
            yield entry


class _FakeQMDClient:
    """qmd faked at its client boundary: the seeded entry, titled ``Entry <entry_id>``."""

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def __init__(self) -> None:
        self.queries = 0

    def is_available(self) -> bool:
        return True

    def query(self, collection: str | None, _text: str, **kwargs: Any) -> list[QMDSearchResult]:
        self.queries += 1
        hits = [
            QMDSearchResult(
                docid="#000000",
                file=f"2024/06/{encode_entry_id(ENTRY_ID)}.md",
                collection=collection or "ariel",
                title=f"Entry {ENTRY_ID}",
                score=0.9,
                line=1,
                snippet=f"1: {RAW_TEXT}",
            )
        ]
        return hits[: kwargs.get("limit", len(hits))]


class _FakeVision:
    """The vision call of ``image_caption``: counts calls, answers one reply."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **_kwargs: Any) -> str:
        self.calls += 1
        return REPLY


# --- helpers -----------------------------------------------------------------------


def _url(name: str) -> str:
    return f"https://h.example/files/{name}.png"


def _picture_bytes(url: str) -> bytes:
    name = url.rsplit("/", 1)[-1]
    return (FIXTURES / name).read_bytes()


def _id(name: str) -> str:
    aid = attachment_id_for(ENTRY_ID, {"url": _url(name)})
    assert aid is not None
    return aid


def _config_dict(
    uri: str,
    embed_url: str,
    *,
    copy_on_ingest: str,
    caption: bool,
    embedding: bool,
    hybrid: bool,
    view: bool,
) -> dict[str, Any]:
    """The one raw ``ariel`` block a case uses everywhere."""
    modules: dict[str, Any] = {}
    if embedding:
        modules[EMBED] = {
            "enabled": True,
            "provider": {"name": "llama-cpp", "base_url": embed_url},
            "model": LLAMA_MODEL,
            "dimensions": DIMS,
        }
    else:
        modules[EMBED] = {"enabled": False}
    if caption:
        modules[CAPTION] = {
            "enabled": True,
            "provider": "openai",
            "model": {"model_id": CAPTION_MODEL},
            "timeout_seconds": 60,
        }
    return {
        "database": {"uri": uri},
        "attachments": {"copy_on_ingest": copy_on_ingest, "view": {"enabled": view}},
        "ingestion": {
            "adapter": "generic_json",
            "source_url": "https://h.example/api",
            "watch": {"require_initial_ingest": False},
        },
        "search_modules": {"keyword": {"enabled": True}, "hybrid": {"enabled": hybrid}},
        "enhancement_modules": modules,
    }


def _entry() -> dict[str, Any]:
    return {
        "entry_id": ENTRY_ID,
        "source_system": "test",
        "timestamp": TS,
        "author": "tester",
        "raw_text": RAW_TEXT,
        "attachments": [
            {"url": _url(n), "type": "image/png", "filename": f"{n}.png"} for n in PICTURES
        ],
        "metadata": {},
        "enhancement_status": {},
    }


def _fake_prepare(monkeypatch) -> None:
    """Every fetched picture is its own rendition; no render worker starts."""
    from osprey.services.ariel_search.attachments import prepare as prepare_mod

    async def _prepare(data, **_kwargs):
        return prepare_mod.PreparedPicture(
            mime_type="image/png",
            skip_reason=None,
            rendition_bytes=bytes(data),
            rendition_mime="image/png",
            rendition_w=1,
            rendition_h=1,
            rendition_sha256=hashlib.sha256(data).hexdigest(),
        )

    monkeypatch.setattr(prepare_mod, "prepare_picture", _prepare)


def _enhancement_status(uri: str) -> dict[str, Any]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %s", (ENTRY_ID,)
        ).fetchone()
    assert row is not None
    return row[0] or {}


def _copy_rows(uri: str) -> dict[str, tuple[str, str | None]]:
    with psycopg.connect(uri) as conn:
        rows = conn.execute(
            "SELECT attachment_id, copy_status, skip_reason FROM attachment_files"
            " WHERE entry_id = %s",
            (ENTRY_ID,),
        ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _register_tools() -> None:
    from osprey.mcp_server.ariel.tools import (  # noqa: F401
        attachment,
        browse,
        capabilities,
        entry,
        hybrid_search,
        keyword_search,
    )


def _status_json(config: dict[str, Any]) -> dict[str, Any]:
    """``osprey ariel status --json`` on *config*; stdout is one JSON document."""
    from click.testing import CliRunner

    from osprey.cli.ariel import ariel_group

    with (
        patch("osprey.cli.ariel.get_config_value", return_value=config),
        patch("osprey.imaging.render.probe_render_worker", new=AsyncMock(return_value=True)),
    ):
        result = CliRunner().invoke(ariel_group, ["status", "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _listing_items(document: dict[str, Any]) -> list[dict[str, Any]]:
    items = document.get("entries")
    assert isinstance(items, list), document
    return items


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh availability, offload, MCP and qmd state; a configured caption provider."""
    from osprey.mcp_server.ariel.server_context import reset_ariel_context
    from osprey.utils.workspace import reset_config_cache

    availability.reset_availability()
    _offload.reset_offload_state()
    reset_ariel_context()
    reset_config_cache()
    qmd_module._reset_client_cache()
    monkeypatch.setattr("osprey.models.config.get_provider_config", lambda name: {"api_key": "k"})
    monkeypatch.setattr(
        caption_mod, "probe_models_endpoint", lambda *a, **k: HealthResult(True, "served", None)
    )
    yield
    reset_ariel_context()
    reset_config_cache()
    qmd_module._reset_client_cache()
    availability.reset_availability()
    _offload.reset_offload_state()


# --- the matrix --------------------------------------------------------------------


@pytest.mark.parametrize("view", [False, True], ids=["view-off", "view-on"])
@pytest.mark.parametrize("hybrid", [False, True], ids=["hybrid-off", "hybrid-on"])
@pytest.mark.parametrize("embedding", [False, True], ids=["embed-off", "embed-on"])
@pytest.mark.parametrize("caption", [False, True], ids=["caption-off", "caption-on"])
@pytest.mark.parametrize("copy_on_ingest", ["images", "none"])
async def test_every_module_combination_runs(
    copy_on_ingest: str,
    caption: bool,
    embedding: bool,
    hybrid: bool,
    view: bool,
    scratch_database,
    llama_stub,
    attachment_fetch,
    monkeypatch,
    tmp_path,
):
    from osprey.mcp_server.ariel.server import mcp
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.mcp_server.ariel.tools.attachment import attachment_view
    from osprey.mcp_server.ariel.tools.browse import browse
    from osprey.mcp_server.ariel.tools.capabilities import capabilities
    from osprey.mcp_server.ariel.tools.entry import entries_by_ids
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations
    from osprey.services.ariel_search.service import create_ariel_service

    stub = llama_stub()
    vision = _FakeVision()
    monkeypatch.setattr(caption_mod, "_chat_completion", vision)
    qmd_client = _FakeQMDClient()
    monkeypatch.setattr(qmd_module, "_resolve_client", lambda client: (qmd_client, True))
    copies = copy_on_ingest == "images"
    picture_search = embedding and hybrid
    if copies:
        attachment_fetch.respond(lambda url, *_a, **_k: FetchOutcome(data=_picture_bytes(url)))
        _fake_prepare(monkeypatch)

    raw = _config_dict(
        scratch_database,
        stub.url,
        copy_on_ingest=copy_on_ingest,
        caption=caption,
        embedding=embedding,
        hybrid=hybrid,
        view=view,
    )
    config = ARIELConfig.from_dict(raw)

    # -- ingest one two-picture entry --------------------------------------------
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
        repository = ARIELRepository(pool, config)
        adapter = _Adapter(config)
        await ingest_one(
            _entry(), adapter, repository, [], config, copy_mod.CopyRun(adapter, ORIGINS)
        )
    finally:
        await pool.close()

    rows = _copy_rows(scratch_database)
    assert set(rows) == {_id(n) for n in PICTURES}
    if copies:
        assert {status for status, _ in rows.values()} == {"copied"}
        assert len(attachment_fetch.calls) == len(PICTURES)
    else:
        assert set(rows.values()) == {("skipped", "copy_on_ingest_mode")}
        assert attachment_fetch.calls == []

    # -- one catch-up pass ---------------------------------------------------------
    await ops.run_catchup(raw, budget_s=None, stop_event=None)

    status = _enhancement_status(scratch_database)
    enabled = [m for m, on in ((CAPTION, caption), (EMBED, embedding)) if on]
    if not copies:
        for module in enabled:
            assert status.get(module, {}).get("status") == "complete", (module, status)
    for module in PICTURE_MODULES:
        if module not in enabled:
            assert module not in status, (module, status)

    assert vision.calls == (len(PICTURES) if caption and copies else 0)
    picture_embeds = len(stub.embeddings)
    assert picture_embeds == (len(PICTURES) if embedding and copies else 0)
    if not embedding:
        assert stub.models_gets == 0

    # -- the MCP surface -----------------------------------------------------------
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": raw}))
    initialize_ariel_context()
    _register_tools()

    service = await create_ariel_service(config)
    try:
        with patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=service),
        ):
            names = {tool.name for tool in await mcp.list_tools()}
            assert ("attachment_view" in names) is view

            caps = json.loads(await get_tool_fn(capabilities)())
            assert caps["attachments"]["view"] is view
            assert caps["attachments"]["picture_search"] is picture_search
            assert caps["attachments"]["captions"] is caption

            keyword = json.loads(await get_tool_fn(keyword_search)(query="beam"))
            listings = {"keyword_search": keyword}
            assert [e["entry_id"] for e in _listing_items(keyword)] == [ENTRY_ID]

            captioned = json.loads(await get_tool_fn(keyword_search)(query=CAPTION_WORD))
            listings["keyword_search(caption)"] = captioned
            caption_hits = _listing_items(captioned)
            if caption and copies:
                assert [e["entry_id"] for e in caption_hits] == [ENTRY_ID]
                if view:
                    assert caption_hits[0].get("matched_attachment_ids")
            else:
                assert caption_hits == []

            if hybrid:
                hybrid_doc = json.loads(await get_tool_fn(hybrid_search)(query=QUERY))
                listings["hybrid_search"] = hybrid_doc
                (hit,) = _listing_items(hybrid_doc)
                assert hit["entry_id"] == ENTRY_ID
                # ``matched_via`` is marked by the fusion, which runs only with the
                # picture lane; a text-only hybrid keeps the plain ranking.
                matched_via = hit.get("matched_via", [])
                if picture_search:
                    assert "text" in matched_via
                assert ("image" in matched_via) is (picture_search and copies)
                if embedding:
                    assert len(stub.embeddings) > picture_embeds
            assert qmd_client.queries == (1 if hybrid else 0)

            listings["browse"] = json.loads(await get_tool_fn(browse)())
            listings["entries_by_ids"] = json.loads(
                await get_tool_fn(entries_by_ids)(entry_ids=[ENTRY_ID])
            )

            for tool_name, document in listings.items():
                for item in _listing_items(document):
                    carried = ATTACHMENT_KEYS & set(item)
                    if not view:
                        assert not carried, (tool_name, carried)
                    else:
                        assert item.get("attachment_count") == len(PICTURES), (tool_name, item)

            if view:
                aid = _id(PICTURES[0])
                if copies:
                    result = await get_tool_fn(attachment_view)(attachment_id=aid)
                    kinds = {getattr(part, "type", None) for part in result.content}
                    assert "image" in kinds
                else:
                    with assert_raises_error(error_type="no_results") as ctx:
                        await get_tool_fn(attachment_view)(attachment_id=aid)
                    assert "skip_reason=copy_on_ingest_mode" in ctx["envelope"]["error_message"]
    finally:
        await service.__aexit__(None, None, None)

    # -- osprey ariel status --json ------------------------------------------------
    document = await asyncio.to_thread(_status_json, raw)
    assert document["attachments"]["picture_search"] is picture_search
    assert document["attachments"]["view"] is view
    assert document["attachments"]["copy_on_ingest"] == copy_on_ingest

    # A disabled module never reached its provider, status included.
    if not caption:
        assert vision.calls == 0
    if not embedding:
        assert stub.embeddings == []
        assert stub.models_gets == 0

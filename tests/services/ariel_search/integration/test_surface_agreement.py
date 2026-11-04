"""Every surface reports one enhancement state per entry, on one real store.

A scratch database holds entries in every state of every built-in module --
complete, pending, failed (status, gave up, attempts spent), stale (an older
marker; an older text build key beside a NULL, legacy one) and ``n/a`` -- plus a
facility module the registry adds that has no ``inspect``. One service over that
store answers the REST routes, and the CLI answers ``status --json``. Then:

* ``/api/status`` coverage equals ``status --json`` ``"coverage"``;
* each ``/enhancements`` card state equals the entry's list-row badge state and
  its membership in the problem facet (the ``<module>:<state>`` filter and its
  facet count);
* ``text_embedding`` stale is the same set of entries on the list, in search
  rows, on the cards, in the facet and in the coverage count.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

import httpx
import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.status import (
    SQL_STATES,
    STATES,
    image_embedding_marker,
    qmd_export_marker,
)
from tests.interfaces.ariel._routes_app import routes_app
from tests.services.ariel_search.conftest import skip_or_fail
from tests.services.ariel_search.integration._support import insert_picture, migrated_repository

# xdist_group("docker"): every container-starting test file shares one worker.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

MODEL = "fake-embed"
TABLE = model_to_table_name(MODEL)
CAPTION_MODEL = "vis-b"
SEMANTIC_MODEL = "sp-2"
IMAGE_MODEL = "img-embed"
FACILITY = "facility_tagger"

BUILTINS = (
    "image_caption",
    "semantic_processor",
    "text_embedding",
    "image_embedding",
    "qmd_export",
)
MODULES = (*BUILTINS, FACILITY)

#: Every entry's raw text carries this word, so one keyword search returns them all.
WORD = "surfaceword"

#: The entries that are ``text_embedding`` stale: an older build key on a row.
TEXT_STALE = {"sa-stale", "sa-no-pic"}


class FacilityTagger(BaseEnhancementModule):
    """A facility module: registered beside the built-ins, with no ``inspect``."""

    @property
    def name(self) -> str:
        return FACILITY

    async def enhance(self, entry, conn):  # pragma: no cover - never driven
        raise NotImplementedError


def _rows(markers: dict[str, str]) -> dict[str, tuple[dict[str, Any], bool, Any]]:
    """entry id -> (enhancement_status, has a viewable picture, build key, ``...`` for no row).

    A build key of None is a row whose ``input_key`` is NULL (legacy ``r1``, current).
    """
    current = {
        "image_caption": {"status": "complete", "marker": markers["image_caption"]},
        "semantic_processor": {"status": "complete", "marker": markers["semantic_processor"]},
        "text_embedding": {"status": "complete"},
        "image_embedding": {"status": "complete", "marker": markers["image_embedding"]},
        "qmd_export": {"status": "complete", "marker": markers["qmd_export"]},
        FACILITY: {"status": "complete"},
    }
    return {
        "sa-empty": ({}, False, ...),
        "sa-empty-pic": ({}, True, ...),
        "sa-done": (current, True, "r1"),
        "sa-legacy-key": (current, True, None),
        "sa-stale": (
            {
                "image_caption": {"status": "complete", "marker": "vis-a"},
                "semantic_processor": {"status": "complete", "marker": "sp-1"},
                "text_embedding": {"status": "complete"},
                "image_embedding": {"status": "complete", "marker": "img-old"},
                "qmd_export": {"status": "complete", "marker": "0:old:none"},
                FACILITY: {"status": "pending"},
            },
            True,
            "r2",
        ),
        "sa-failed": (
            {
                "image_caption": {"status": "pending", "gave_up": True},
                "semantic_processor": {"status": "failed", "attempts": 1},
                "text_embedding": {"status": "pending", "attempts": 99},
                "image_embedding": {"status": "failed", "marker": markers["image_embedding"]},
                "qmd_export": {"status": "failed"},
                FACILITY: {"status": "failed", "error": "tagger down"},
            },
            True,
            "r2",
        ),
        "sa-no-pic": (
            {
                "image_caption": {"status": "complete", "marker": markers["image_caption"]},
                "semantic_processor": {"status": "pending"},
                "text_embedding": {"status": "complete"},
                "qmd_export": {"status": "pending"},
            },
            False,
            "r1:e1:0123456789abcdef",
        ),
    }


def _config_dict(uri: str, mirror: str) -> dict[str, Any]:
    return {
        "database": {"uri": uri},
        "search_modules": {"keyword": {"enabled": True}, "semantic": {"enabled": False}},
        "enhancement_modules": {
            "image_caption": {
                "enabled": True,
                "provider": "openai",
                "model": {"model_id": CAPTION_MODEL},
            },
            "semantic_processor": {
                "enabled": True,
                "provider": "openai",
                "model": {"model_id": SEMANTIC_MODEL},
            },
            "text_embedding": {
                "enabled": True,
                "provider": {"name": "fake", "base_url": "http://fake-embeddings.invalid:11434"},
                "models": [{"name": MODEL, "dimension": 4}],
            },
            "image_embedding": {
                "enabled": True,
                "provider": {"name": "llama-cpp", "base_url": "http://127.0.0.1:9"},
                "model": IMAGE_MODEL,
                "dimensions": 4,
            },
            "qmd_export": {"enabled": True, "mirror_path": mirror},
            FACILITY: {"enabled": True},
        },
    }


@pytest.fixture(autouse=True)
def _facility_registry(monkeypatch):
    """The patched registry, plus the facility module after the built-ins."""
    from osprey.registry import get_registry
    from osprey.registry.base import ArielEnhancementModuleRegistration

    registry = get_registry()
    known = {
        name: registry.get_ariel_enhancement_module(name)
        for name in registry.list_ariel_enhancement_modules()
    }
    known[FACILITY] = (
        FacilityTagger,
        ArielEnhancementModuleRegistration(
            name=FACILITY,
            module_path=__name__,
            class_name="FacilityTagger",
            description="Facility tagger",
            execution_order=40,
            label="Facility tags",
        ),
    )
    registry.list_ariel_enhancement_modules.return_value = list(known)
    registry.get_ariel_enhancement_module.side_effect = known.get
    monkeypatch.setattr(
        "osprey.models.config.get_provider_config", lambda name, *a, **k: {"api_key": "k"}
    )


async def _seed(repository, seed_entry_factory, uri: str, rows) -> None:
    for entry_id, (status, picture, key) in rows.items():
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=entry_id,
                raw_text=f"{WORD} {entry_id}",
                enhancement_status=status,
            )
        )
        if picture:
            insert_picture(uri, f"{entry_id}-a", entry_id, copy_status="copied")
        if key is ...:
            continue
        async with repository.pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO {TABLE} (entry_id, input_key) VALUES (%s, %s)", [entry_id, key]
            )


@pytest.fixture
async def store(scratch_database: str, seed_entry_factory, tmp_path) -> AsyncIterator[dict]:
    """The seeded store: its raw config, rows and one service over it."""
    from osprey.services.ariel_search.service import create_ariel_service

    config_dict = _config_dict(scratch_database, str(tmp_path / "mirror"))
    config = ARIELConfig.from_dict(copy.deepcopy(config_dict))
    markers = {
        "image_caption": CAPTION_MODEL,
        "semantic_processor": SEMANTIC_MODEL,
        "image_embedding": image_embedding_marker(config)[0],
        "qmd_export": qmd_export_marker(config),
    }
    assert all(markers.values()), markers
    rows = _rows(markers)
    async with migrated_repository(config) as repository:
        with psycopg.connect(scratch_database) as conn:
            found = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (TABLE,)).fetchone()
        if not found or not found[0]:
            skip_or_fail("this server has no pgvector; the embedding table was not created")
        await _seed(repository, seed_entry_factory, scratch_database, rows)

    service = await create_ariel_service(config)
    try:
        yield {"config_dict": config_dict, "rows": rows, "service": service}
    finally:
        await service.aclose()


async def _get(service, path: str, **params: Any) -> Any:
    transport = httpx.ASGITransport(app=routes_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://ariel.test") as client:
        response = await client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def _post(service, path: str, body: dict[str, Any]) -> Any:
    transport = httpx.ASGITransport(app=routes_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://ariel.test") as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def _list_rows(service, filters: dict[str, Any] | None = None) -> dict[str, dict]:
    params: dict[str, Any] = {"view": "row", "page_size": 100}
    if filters is not None:
        params["filters"] = json.dumps(filters)
    body = await _get(service, "/api/entries", **params)
    assert body["next_cursor"] is None
    return {row["entry_id"]: row["enhancements"] for row in body["entries"]}


async def _cards(service, entry_id: str) -> dict[str, dict]:
    body = await _get(service, f"/api/entries/{entry_id}/enhancements")
    return {card["module"]: card for card in body["modules"]}


async def _problem_facet(service) -> dict[str, int]:
    body = await _get(service, "/api/facets")
    assert body["available"] is True
    (dim,) = [d for d in body["dimensions"] if d["name"] == "enhancement"]
    return {v["value"]: v["count"] for v in dim["values"]}


def _status_json(config_dict: dict[str, Any]) -> dict[str, Any]:
    """``osprey ariel status --json``, run as the operator runs it."""
    from unittest.mock import patch

    from click.testing import CliRunner

    from osprey.cli.ariel import ariel_group

    def _value(key, default=None, *_a, **_k):
        return copy.deepcopy(config_dict) if key == "ariel" else default

    with patch("osprey.cli.ariel.get_config_value", side_effect=_value):
        result = CliRunner().invoke(ariel_group, ["status", "--json"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


async def test_surface_agreement_status_coverage_equals_cli_status_json(store):
    service = store["service"]

    api = (await _get(service, "/api/status"))["enhancement_coverage"]
    cli = (await asyncio.to_thread(_status_json, store["config_dict"]))["coverage"]

    assert api == cli
    assert list(api) == list(MODULES)
    assert api[FACILITY]["label"] == "Facility tags"
    for module, counts in api.items():
        assert sum(counts[s] for s in STATES) == len(store["rows"]), module
    # Every state occurs somewhere, so the equality is not vacuous.
    assert {s for counts in api.values() for s in STATES if counts[s]} == set(STATES)


async def test_surface_agreement_cards_rows_and_problem_facet(store):
    service, rows = store["service"], store["rows"]

    listed = await _list_rows(service)
    assert set(listed) == set(rows)
    members = {
        (module, state): set(await _list_rows(service, {"enhancement": [f"{module}:{state}"]}))
        for module in MODULES
        for state in SQL_STATES
    }
    facet = await _problem_facet(service)
    coverage = (await _get(service, "/api/status"))["enhancement_coverage"]

    tally: dict[str, Counter] = {module: Counter() for module in MODULES}
    for entry_id in rows:
        cards = await _cards(service, entry_id)
        assert list(cards) == list(MODULES), entry_id
        row = listed[entry_id]
        for module in MODULES:
            state = cards[module]["state"]
            assert row[module] == state, (entry_id, module)
            tally[module][state] += 1
            for problem in SQL_STATES:
                assert (entry_id in members[(module, problem)]) == (state == problem), (
                    entry_id,
                    module,
                    problem,
                )
        # The facility module has no inspect: its card carries the stored status only.
        assert cards[FACILITY]["detail"] is None
        assert cards[FACILITY]["label"] == "Facility tags"

    for module in MODULES:
        for problem in SQL_STATES:
            count = len(members[(module, problem)])
            assert facet.get(f"{module}:{problem}", 0) == count, (module, problem)
            assert coverage[module][problem] == count, (module, problem)
        assert {s: coverage[module][s] for s in STATES} == {s: tally[module][s] for s in STATES}, (
            module
        )

    # Each built-in is failed and stale somewhere; the facility module fails but is never stale.
    for module in BUILTINS:
        assert members[(module, "failed")], module
        assert members[(module, "stale")], module
    assert members[(FACILITY, "failed")] == {"sa-failed"}
    assert members[(FACILITY, "stale")] == set()
    # Its failed card names the per-entry rerun, as a built-in's does.
    failed = (await _cards(service, "sa-failed"))[FACILITY]
    assert failed["command"] == f"osprey ariel enhance --module {FACILITY} --entry sa-failed"
    assert failed["error"] == "tagger down"


async def test_surface_agreement_text_embedding_stale_everywhere(store):
    service, rows = store["service"], store["rows"]
    te = "text_embedding"

    listed = {i for i, row in (await _list_rows(service)).items() if row[te] == "stale"}
    found = await _post(
        service,
        "/api/search",
        {"query": WORD, "mode": "keyword", "view": "row", "max_results": 100},
    )
    assert {row["entry_id"] for row in found["entries"]} == set(rows)
    searched = {row["entry_id"] for row in found["entries"] if row["enhancements"][te] == "stale"}
    filtered_search = await _post(
        service,
        "/api/search",
        {
            "query": WORD,
            "mode": "keyword",
            "view": "row",
            "max_results": 100,
            "filters": {"enhancement": [f"{te}:stale"]},
        },
    )
    carded = {i for i in rows if (await _cards(service, i))[te]["state"] == "stale"}
    filtered = set(await _list_rows(service, {"enhancement": [f"{te}:stale"]}))
    facet = (await _problem_facet(service)).get(f"{te}:stale", 0)
    coverage = (await _get(service, "/api/status"))["enhancement_coverage"][te]["stale"]

    # The NULL (legacy) key is current; the failed entry's old key loses to failed.
    assert listed == TEXT_STALE
    assert searched == TEXT_STALE
    assert {row["entry_id"] for row in filtered_search["entries"]} == TEXT_STALE
    assert carded == TEXT_STALE
    assert filtered == TEXT_STALE
    assert facet == len(TEXT_STALE)
    assert coverage == len(TEXT_STALE)

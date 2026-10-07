"""The entry-field descriptor round trip, end to end, with the example adapter.

A facility adapter declares the fields its authors fill in; every surface that
writes an entry reads them from that one declaration. This walks the whole
contract once, in the order an author and an agent meet it:

1. ``GET /api/publish-info`` lists the three declared fields;
2. the options route answers the ``scan`` choices for a day;
3. a web create with a ``scan`` outside those choices is refused naming it;
4. a good web create reaches the adapter with native values;
5. an agent draft carrying ``fields`` loads back through ``GET /api/drafts``;
6. an agent direct write without ``book`` is published once ``fields``
   supplies it.

The web app and the MCP tools are served by one real ``ARIELSearchService``
over the fixture module's dict-backed repository, so the direct write is the
entry the publish reads back. The MCP drafts directory and the web drafts route
point at one temporary directory, so the draft the agent writes is the draft
the web app serves.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.ariel.api.drafts as drafts_mod
import osprey.mcp_server.ariel.tools.entry as entry_mod
from osprey.interfaces.ariel.api import routes
from osprey.interfaces.ariel.api.drafts import draft_router
from osprey.mcp_server.ariel.server import ARIEL_NATIVE_SOURCE_SYSTEM
from osprey.mcp_server.ariel.server_context import (
    initialize_ariel_context,
    reset_ariel_context,
)
from osprey.mcp_server.ariel.tools.entry import entry_create
from osprey.mcp_server.ariel.tools.publish import entry_publish
from osprey.services.ariel_search.search.base import ParameterDescriptor
from osprey.services.ariel_search.service import ARIELSearchService
from osprey_connectors.workspace import reset_config_cache
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    EXAMPLE_SOURCE_SYSTEM,
    DictRepository,
    ExampleEntryFieldsAdapter,
    dict_repository_fixture,
    example_config,
    example_descriptors,
    example_entry_fields_fixture,
)
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

DAY = "2026-10-01"
SCAN_CHOICES = [{"value": "s-17", "label": "Scan 17"}, {"value": "s-18", "label": "Scan 18"}]


@pytest.fixture(name="ariel_mcp_context")
def ariel_mcp_context_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An ARIEL MCP context read from a config in ``tmp_path``, reset afterwards.

    The example adapter replaces ``get_adapter``, so the adapter the config
    names is never built; the ingestion block only has to exist.
    """
    monkeypatch.chdir(tmp_path)
    ariel = {
        "database": {"uri": "postgresql://localhost/test"},
        "ingestion": {"adapter": "generic_json", "source_url": str(tmp_path / "x.json")},
    }
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": ariel}))
    reset_ariel_context()
    reset_config_cache()
    initialize_ariel_context()
    yield
    reset_ariel_context()
    reset_config_cache()


@pytest.fixture(name="drafts_dir")
def drafts_dir_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One drafts directory, written by the MCP tool and served by the web route."""
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    monkeypatch.setattr(entry_mod, "_get_drafts_dir", lambda: drafts_dir)
    monkeypatch.setattr(drafts_mod, "_drafts_dir", lambda: drafts_dir)
    return drafts_dir


@pytest.fixture(name="service")
def service_fixture(
    dict_repository: DictRepository,
    ariel_mcp_context: None,  # noqa: ARG001 - the context must exist before its service is patched
) -> Iterator[ARIELSearchService]:
    """The one service both surfaces use, with every outbound side effect patched out."""
    service = ARIELSearchService(
        config=example_config(), pool=MagicMock(), repository=dict_repository
    )
    with (
        patch(
            "osprey.services.ariel_search.enhancement.qmd_export.mirror_entry_best_effort",
            return_value=True,
        ),
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=service),
        ),
        patch("osprey.mcp_server.ariel.tools.entry._focus_ariel_panel"),
        patch(
            "osprey.mcp_server.ariel.tools.entry.notify_agent_activity_async",
            new=AsyncMock(),
        ),
        patch(
            "osprey.mcp_server.ariel.tools.publish.notify_agent_activity_async",
            new=AsyncMock(),
        ),
    ):
        yield service


@pytest.fixture(name="client")
def client_fixture(service: ARIELSearchService) -> TestClient:
    """The ARIEL web app's routes and drafts route, serving ``service``."""
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(draft_router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    return TestClient(app)


def _without_required_book() -> list[ParameterDescriptor]:
    """The example's fields as a facility declared them before ``book`` was required."""
    return [
        dataclasses.replace(descriptor, required=False) if descriptor.name == "book" else descriptor
        for descriptor in example_descriptors()
    ]


async def test_entry_field_descriptors_round_trip(
    client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    dict_repository: DictRepository,
    drafts_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One declaration drives publish-info, options, web create, agent draft and agent publish."""
    state = example_entry_fields.state
    state.options_table = {DAY: [dict(choice) for choice in SCAN_CHOICES]}

    # 1. publish-info lists the declared fields, in declaration order.
    info = client.get("/api/publish-info")
    assert info.status_code == 200
    info_body = info.json()
    assert info_body["supports_write"] is True
    assert info_body["source_system"] == EXAMPLE_SOURCE_SYSTEM
    by_name = {descriptor["name"]: descriptor for descriptor in info_body["entry_fields"]}
    assert [descriptor["name"] for descriptor in info_body["entry_fields"]] == [
        "book",
        "day",
        "scan",
    ]
    assert by_name["book"]["required"] is True
    assert by_name["day"]["type"] == "date"
    assert by_name["scan"]["depends_on"] == ["day"]
    assert by_name["scan"]["options_endpoint"] == "/entry-fields/scan/options"

    # 2. The options route answers the scan choices for that day.
    options = client.get("/api/entry-fields/scan/options", params={"day": DAY})
    assert options.status_code == 200
    assert options.json() == {"field": "scan", "options": SCAN_CHOICES}
    assert state.options_calls[-1] == ("scan", {"day": DAY})

    # 3. A web create with a scan outside the day's choices is refused naming it.
    refused = client.post(
        "/api/entries",
        json={
            "subject": "Injection tuned",
            "details": "Kicker timing moved by 2 ns.",
            "metadata": {"book": "ops", "day": DAY, "scan": "nope"},
        },
    )
    assert refused.status_code == 422
    refused_body = refused.json()
    assert refused_body["code"] == "invalid_entry_field"
    assert refused_body["field"] == "scan"
    assert state.created == []
    assert dict_repository.entries == {}

    # 4. A good web create reaches the adapter with native values.
    created = client.post(
        "/api/entries",
        json={
            "subject": "Injection tuned",
            "details": "Kicker timing moved by 2 ns.",
            "metadata": {"book": " ops ", "day": DAY, "scan": "s-17"},
        },
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    assert created_body["entry_id"] == "example-1"
    assert created_body["source_system"] == EXAMPLE_SOURCE_SYSTEM
    assert created_body["sync_status"] == "pending_sync"
    assert len(state.created) == 1
    assert state.created[0].metadata == {"book": "ops", "day": DAY, "scan": "s-17"}
    web_copy = dict_repository.entries["example-1"]
    assert web_copy["metadata"]["created_via"] == "ariel-web"
    assert web_copy["metadata"]["book"] == "ops"

    # 5. An agent draft carrying fields loads back through the web drafts route.
    create_tool = get_tool_fn(entry_create)
    draft_result = json.loads(
        await create_tool(
            subject="Orbit drift",
            details="Horizontal orbit drifted 40 um overnight.",
            fields={"day": DAY, "scan": "s-18"},
        )
    )
    draft_id = draft_result["draft_id"]
    assert (drafts_dir / f"{draft_id}.json").exists()
    draft = client.get(f"/api/drafts/{draft_id}")
    assert draft.status_code == 200
    draft_body = draft.json()
    assert draft_body["subject"] == "Orbit drift"
    assert draft_body["fields"] == {"day": DAY, "scan": "s-18"}
    assert "session_metadata" in draft_body["metadata"]
    assert len(state.created) == 1

    # 6. A direct agent write must carry the required book ...
    with assert_raises_error(error_type="validation_error") as refusal:
        await create_tool(
            subject="Orbit drift",
            details="Horizontal orbit drifted 40 um overnight.",
            fields={"day": DAY, "scan": "s-18"},
            draft=False,
        )
    assert refusal["envelope"]["details"]["field"] == "book"
    assert set(dict_repository.entries) == {"example-1"}

    # ... so one written before the facility required it is stored without one,
    with monkeypatch.context() as earlier:
        earlier.setattr(example_entry_fields, "get_entry_field_descriptors", _without_required_book)
        direct = json.loads(
            await create_tool(
                subject="Orbit drift",
                details="Horizontal orbit drifted 40 um overnight.",
                fields={"day": DAY, "scan": "s-18"},
                draft=False,
            )
        )
    native_id = direct["entry_id"]
    assert direct["source_system"] == ARIEL_NATIVE_SOURCE_SYSTEM
    stored: dict[str, Any] = dict_repository.entries[native_id]["metadata"]
    assert "book" not in stored
    assert stored["scan"] == "s-18"
    assert stored["created_via"] == "ariel-mcp"

    # ... and the publish reads it back, refuses it without a book,
    publish_tool = get_tool_fn(entry_publish)
    with assert_raises_error(error_type="validation_error") as missing:
        await publish_tool(entry_id=native_id)
    assert missing["envelope"]["details"]["field"] == "book"
    assert len(state.created) == 1

    # ... and publishes it once fields supplies one, stored values checked live.
    state.options_calls.clear()
    published = json.loads(await publish_tool(entry_id=native_id, fields={"book": "physics"}))
    assert published["entry_id"] == "example-2"
    assert published["source_system"] == EXAMPLE_SOURCE_SYSTEM
    assert published["sync_status"] == "pending_sync"
    assert state.created[-1].subject == "Orbit drift"
    assert state.created[-1].metadata == {"book": "physics", "day": DAY, "scan": "s-18"}
    assert state.options_calls == [("scan", {"day": DAY})]
    published_copy = dict_repository.entries["example-2"]["metadata"]
    assert published_copy["book"] == "physics"
    assert published_copy["created_via"] == "ariel-mcp"
    assert published_copy["session_metadata"] == stored["session_metadata"]

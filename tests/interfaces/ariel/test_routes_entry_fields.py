"""Tests for the ARIEL web routes' handling of facility-declared entry fields.

The app runs a real ``ARIELSearchService`` over an ``AsyncMock`` repository,
with the inline mirror write patched out, so the create paths reach the
example adapter that ``example_entry_fields`` injects and its recordings show
what the routes actually sent.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.services.ariel_search.entry_fields import (
    EntryFieldDeclarationError,
    EntryFieldError,
    EntryFieldOptionsUnavailable,
)
from osprey.services.ariel_search.exceptions import AdapterNotFoundError
from osprey.services.ariel_search.search.base import ParameterDescriptor
from osprey.services.ariel_search.service import ARIELSearchService
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    DictRepository,
    ExampleEntryFieldsAdapter,
    dict_repository_fixture,
    example_config,
    example_descriptors,
    example_entry_fields_fixture,
)


@pytest.fixture(name="service")
def service_fixture() -> Iterator[ARIELSearchService]:
    """A real service over an ``AsyncMock`` repository, mirror write patched out."""
    service = ARIELSearchService(config=example_config(), pool=MagicMock(), repository=AsyncMock())
    with patch(
        "osprey.services.ariel_search.enhancement.qmd_export.mirror_entry_best_effort",
        return_value=True,
    ):
        yield service


@pytest.fixture(name="client")
def client_fixture(service: ARIELSearchService) -> TestClient:
    """A routes-only app serving ``service``."""
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    return TestClient(app)


def _body(response) -> dict:
    return json.loads(response.body)


def test_envelope_invalid_entry_field_is_422_naming_the_field() -> None:
    """A bad submitted value maps to 422 with the field and its message."""
    response = routes._entry_field_error_response(
        EntryFieldError("book", "Book must be one of: ops, physics.")
    )
    assert response.status_code == 422
    assert _body(response) == {
        "detail": "Book must be one of: ops, physics.",
        "code": "invalid_entry_field",
        "field": "book",
    }


def test_envelope_options_unavailable_is_502_with_generic_detail() -> None:
    """An options failure maps to 502 with the generic message and the field."""
    exc = EntryFieldOptionsUnavailable("scan")
    response = routes._entry_field_error_response(exc)
    assert response.status_code == 502
    assert _body(response) == {
        "detail": "The options for entry field 'scan' are unavailable right now.",
        "code": "entry_field_options_unavailable",
        "field": "scan",
    }


def test_envelope_options_unavailable_never_carries_adapter_text() -> None:
    """The adapter's own error text, chained as the cause, stays out of the body."""
    try:
        try:
            raise RuntimeError("secret upstream host db7:5432 refused")
        except RuntimeError as cause:
            raise EntryFieldOptionsUnavailable("scan") from cause
    except EntryFieldOptionsUnavailable as exc:
        response = routes._entry_field_error_response(exc)
    assert b"db7" not in response.body
    assert b"refused" not in response.body


def test_envelope_misdeclared_is_500_without_field() -> None:
    """A broken declaration maps to 500 with its message and no field key."""
    exc = EntryFieldDeclarationError(
        "Example Logbook declares entry field 'tags', a reserved name.",
        adapter="Example Logbook",
        field="tags",
    )
    response = routes._entry_field_error_response(exc)
    assert response.status_code == 500
    assert _body(response) == {
        "detail": "Example Logbook declares entry field 'tags', a reserved name.",
        "code": "entry_fields_misdeclared",
    }


def test_envelope_bodies_are_json_responses() -> None:
    """Every envelope is a JSON response, like the 401 credential prompt."""
    for exc in (
        EntryFieldError("day", "Day must be a date."),
        EntryFieldOptionsUnavailable("scan"),
        EntryFieldDeclarationError("bad", adapter="a", field="f"),
    ):
        response = routes._entry_field_error_response(exc)
        assert response.media_type == "application/json"


def test_envelope_rejects_other_exceptions() -> None:
    """Only the three entry-field exceptions have an envelope."""
    with pytest.raises(TypeError):
        routes._entry_field_error_response(ValueError("plain"))


def test_envelope_app_create_reaches_the_example_adapter(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """The test app's create path reaches the patched adapter, so its recordings count."""
    response = client.post(
        "/api/entries",
        json={"subject": "Beam", "details": "Stable.", "metadata": {"book": "ops"}},
    )
    assert response.status_code == 200, response.text
    assert len(example_entry_fields.state.created) == 1
    assert example_entry_fields.state.created[0].subject == "Beam"


@pytest.mark.usefixtures("example_entry_fields")
def test_publish_info_lists_the_example_entry_fields_in_form_order(client: TestClient) -> None:
    """``entry_fields`` is the checked descriptors' ``to_dict()``, in declaration order."""
    response = client.get("/api/publish-info")
    assert response.status_code == 200, response.text
    expected = [d.to_dict() for d in example_descriptors()]
    expected[2]["options_endpoint"] = "/entry-fields/scan/options"
    assert response.json()["entry_fields"] == expected


@pytest.mark.usefixtures("example_entry_fields")
def test_publish_info_dynamic_select_options_endpoint_is_relative_to_the_api_base(
    client: TestClient,
) -> None:
    """Only the dynamic select carries an options route, written without the ``/api`` prefix."""
    fields = {f["name"]: f for f in client.get("/api/publish-info").json()["entry_fields"]}
    assert fields["scan"]["options_endpoint"] == "/entry-fields/scan/options"
    assert not fields["scan"]["options_endpoint"].startswith("/api")
    assert "options_endpoint" not in fields["book"]
    assert "options_endpoint" not in fields["day"]


def test_publish_info_leaves_the_adapter_descriptors_untouched(
    client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing the options route never mutates the adapter's own descriptor objects."""
    owned = example_descriptors()
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: owned)
    assert client.get("/api/publish-info").status_code == 200
    assert all(d.options_endpoint is None for d in owned)


@pytest.mark.usefixtures("example_entry_fields")
def test_publish_info_keeps_the_write_capability_keys(client: TestClient) -> None:
    """The existing capability keys sit alongside ``entry_fields``."""
    body = client.get("/api/publish-info").json()
    assert body["supports_write"] is True
    assert body["requires_auth"] is False
    assert body["source_system"] == "Example Logbook"


def test_publish_info_without_an_adapter_has_no_entry_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No configured adapter yields an empty ``entry_fields`` list."""

    def _missing(_config):
        raise AdapterNotFoundError("no adapter", adapter_name="none")

    monkeypatch.setattr("osprey.services.ariel_search.ingestion.get_adapter", _missing)
    response = client.get("/api/publish-info")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "supports_write": False,
        "requires_auth": False,
        "source_system": None,
        "entry_fields": [],
    }


def test_publish_info_adapter_declaring_nothing_has_no_entry_fields(
    client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter that declares no entry fields yields an empty list."""
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [])
    response = client.get("/api/publish-info")
    assert response.status_code == 200, response.text
    assert response.json()["entry_fields"] == []


def test_publish_info_broken_declaration_is_the_500_envelope(
    client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misdeclared field returns the 500 ``entry_fields_misdeclared`` envelope."""
    broken = ParameterDescriptor(
        name="tags",
        label="Tags",
        description="Collides with a reserved entry field",
        param_type="text",
        default=None,
    )
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [broken])
    response = client.get("/api/publish-info")
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "entry_fields_misdeclared"
    assert "tags" in body["detail"]
    assert "field" not in body


SCAN_CHOICES = [{"value": "s1", "label": "Scan 1"}, {"value": "s2", "label": "Scan 2"}]


def test_options_route_answers_with_field_and_options(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """A dynamic select's options come back as ``{field, options}`` for its parent's value."""
    example_entry_fields.state.options_table["2026-10-04"] = SCAN_CHOICES
    response = client.get("/api/entry-fields/scan/options", params={"day": "2026-10-04"})
    assert response.status_code == 200, response.text
    assert response.json() == {"field": "scan", "options": SCAN_CHOICES}


@pytest.mark.usefixtures("example_entry_fields")
def test_options_route_path_matches_the_published_options_endpoint(client: TestClient) -> None:
    """The ``options_endpoint`` publish-info writes resolves to this route under ``/api``."""
    response = client.get("/api" + routes.entry_field_options_path("scan"))
    assert response.status_code == 200, response.text
    assert response.json() == {"field": "scan", "options": []}


def test_options_route_forwards_only_coerced_depends_on_values(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """Only ``depends_on`` keys reach the adapter, coerced; other query keys are ignored."""
    example_entry_fields.state.options_table["2026-10-04"] = SCAN_CHOICES
    response = client.get(
        "/api/entry-fields/scan/options",
        params={"day": " 2026-10-04 ", "book": "ops", "junk": "x", "scan": "s1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["options"] == SCAN_CHOICES
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-04"})]


def test_options_route_ignores_a_bad_value_for_a_key_outside_depends_on(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """A key that is declared but not a parent is never read, so its bad value is no error."""
    response = client.get(
        "/api/entry-fields/scan/options", params={"day": "2026-10-04", "book": "nonsense"}
    )
    assert response.status_code == 200, response.text
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-04"})]


def test_options_route_omits_an_absent_or_blank_parent(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """A parent left out or blank is not sent to the adapter at all."""
    assert client.get("/api/entry-fields/scan/options").status_code == 200
    assert client.get("/api/entry-fields/scan/options", params={"day": "  "}).status_code == 200
    assert example_entry_fields.state.options_calls == [("scan", {}), ("scan", {})]


def test_options_route_bad_parent_is_422_naming_the_parent(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """A parent value that does not fit its own declaration is a 422 naming the parent."""
    response = client.get("/api/entry-fields/scan/options", params={"day": "yesterday"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_entry_field"
    assert body["field"] == "day"
    assert example_entry_fields.state.options_calls == []


@pytest.mark.parametrize("name", ["book", "day", "nope"])
@pytest.mark.usefixtures("example_entry_fields")
def test_options_route_is_404_unless_a_declared_dynamic_select(
    client: TestClient, name: str
) -> None:
    """A static field or an undeclared name has no options route."""
    response = client.get(f"/api/entry-fields/{name}/options")
    assert response.status_code == 404
    assert name in response.json()["detail"]


def test_options_route_is_404_without_an_adapter(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no adapter configured no field is declared, so every name is a 404."""

    def _missing(_config):
        raise AdapterNotFoundError("no adapter", adapter_name="none")

    monkeypatch.setattr("osprey.services.ariel_search.ingestion.get_adapter", _missing)
    assert client.get("/api/entry-fields/scan/options").status_code == 404


def test_options_route_adapter_failure_is_the_502_envelope(
    client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """An adapter error becomes the generic 502 envelope, its text kept out of the body."""
    example_entry_fields.state.options_mode = "fail"
    response = client.get("/api/entry-fields/scan/options", params={"day": "2026-10-04"})
    assert response.status_code == 502
    assert response.json() == {
        "detail": "The options for entry field 'scan' are unavailable right now.",
        "code": "entry_field_options_unavailable",
        "field": "scan",
    }
    assert "lookup failed" not in response.text


def test_options_route_adapter_timeout_is_the_502_envelope(
    client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter that never answers is cut off by the server timeout and becomes a 502."""
    monkeypatch.setattr("osprey.services.ariel_search.entry_fields.OPTIONS_TIMEOUT_SECONDS", 0.05)
    example_entry_fields.state.options_mode = "hang"
    response = client.get("/api/entry-fields/scan/options", params={"day": "2026-10-04"})
    assert response.status_code == 502
    assert response.json()["code"] == "entry_field_options_unavailable"
    assert response.json()["field"] == "scan"


def test_options_route_broken_declaration_is_the_500_envelope(
    client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misdeclared field set returns the 500 ``entry_fields_misdeclared`` envelope."""
    broken = ParameterDescriptor(
        name="scan",
        label="Scan",
        description="Depends on an undeclared field",
        param_type="dynamic_select",
        default=None,
        depends_on=("missing",),
    )
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [broken])
    response = client.get("/api/entry-fields/scan/options")
    assert response.status_code == 500
    assert response.json()["code"] == "entry_fields_misdeclared"
    assert example_entry_fields.state.options_calls == []


# --- Create routes: the read-only local fallback stores the declared values ---

_SCAN_CHOICES = [{"value": "s1", "label": "Scan 1"}]


@pytest.fixture(name="local_client")
def local_client_fixture(
    dict_repository: DictRepository, example_entry_fields: ExampleEntryFieldsAdapter
) -> Iterator[TestClient]:
    """A routes-only app over a dict repository, the example adapter read-only."""
    example_entry_fields.state.supports_write = False
    example_entry_fields.state.options_table = {"2026-10-04": _SCAN_CHOICES}
    service = ARIELSearchService(
        config=example_config(), pool=MagicMock(), repository=dict_repository
    )
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    yield TestClient(app)


def _post_json(client: TestClient, metadata: dict, **extra) -> Any:
    return client.post(
        "/api/entries",
        json={"subject": "Beam", "details": "Stable.", "metadata": metadata, **extra},
    )


def _post_upload(client: TestClient, metadata: dict, **extra) -> Any:
    return client.post(
        "/api/entries/upload",
        data={"subject": "Beam", "details": "Stable.", "metadata": json.dumps(metadata), **extra},
    )


_POSTERS = pytest.mark.parametrize("post", [_post_json, _post_upload], ids=["json", "upload"])


def _stored(repository: DictRepository, response) -> dict:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sync_status"] == "local_only"
    assert body["source_system"] == "ARIEL Web"
    return repository.entries[body["entry_id"]]


@_POSTERS
def test_create_local_fallback_stores_the_declared_values(
    post, local_client: TestClient, dict_repository: DictRepository
) -> None:
    """A read-only adapter's local copy keeps each declared value, coerced."""
    response = post(
        local_client,
        {"book": " physics ", "day": "2026-10-04", "scan": "s1"},
        tags=["a"] if post is _post_json else "a",
        logbook="ops-log",
    )
    metadata = _stored(dict_repository, response)["metadata"]
    assert metadata["book"] == "physics"
    assert metadata["day"] == "2026-10-04"
    assert metadata["scan"] == "s1"
    assert metadata["logbook"] == "ops-log"
    assert metadata["tags"] == ["a"]
    assert metadata["created_via"] == "ariel-web"
    assert metadata["sync_status"] == "local_only"
    # ARIEL's own keys come after the declared values, sync_status last.
    assert list(metadata)[-1] == "sync_status"
    assert list(metadata)[:3] == ["book", "day", "scan"]


@_POSTERS
def test_create_local_fallback_date_round_trips_through_the_repository_write(
    post, local_client: TestClient, dict_repository: DictRepository
) -> None:
    """The stored metadata is JSON-native, so a ``date`` survives a JSON store unchanged."""
    response = post(local_client, {"book": "ops", "day": "2026-10-04"})
    metadata = _stored(dict_repository, response)["metadata"]
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["day"] == "2026-10-04"


@_POSTERS
def test_create_local_fallback_does_not_store_client_reserved_keys(
    post, local_client: TestClient, dict_repository: DictRepository
) -> None:
    """A client ``sync_status``/``created_via``/undeclared key never reaches the local copy."""
    response = post(
        local_client,
        {"book": "ops", "sync_status": "synced", "created_via": "forged", "extra": "x"},
    )
    metadata = _stored(dict_repository, response)["metadata"]
    assert metadata["sync_status"] == "local_only"
    assert metadata["created_via"] == "ariel-web"
    assert "extra" not in metadata


@_POSTERS
def test_create_local_fallback_keeps_draft_session_metadata(
    post, local_client: TestClient, dict_repository: DictRepository
) -> None:
    """A draft's ``session_metadata`` stays in ARIEL's local copy."""
    session = {"session_id": "abc", "created_via": "ariel-mcp"}
    response = post(local_client, {"book": "ops", "session_metadata": session})
    metadata = _stored(dict_repository, response)["metadata"]
    assert metadata["session_metadata"] == session


@_POSTERS
def test_create_local_fallback_invalid_value_is_422_and_stores_nothing(
    post, local_client: TestClient, dict_repository: DictRepository
) -> None:
    """A missing required value is refused before any write."""
    response = post(local_client, {"day": "2026-10-04"})
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_entry_field"
    assert response.json()["field"] == "book"
    assert dict_repository.entries == {}


@_POSTERS
def test_create_local_fallback_checks_dynamic_values_live(
    post,
    local_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """A ``dynamic_select`` value is checked against the adapter's live choices."""
    response = post(local_client, {"book": "ops", "day": "2026-10-04", "scan": "nope"})
    assert response.status_code == 422, response.text
    assert response.json()["field"] == "scan"
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-04"})]
    assert dict_repository.entries == {}


@_POSTERS
def test_create_local_fallback_options_failure_is_502_and_stores_nothing(
    post,
    local_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """A live-options failure while validating answers 502 naming the field."""
    example_entry_fields.state.options_mode = "fail"
    response = post(local_client, {"book": "ops", "day": "2026-10-04", "scan": "s1"})
    assert response.status_code == 502, response.text
    assert response.json()["code"] == "entry_field_options_unavailable"
    assert response.json()["field"] == "scan"
    assert dict_repository.entries == {}


@_POSTERS
def test_create_local_fallback_conflicting_logbook_is_422(
    post,
    local_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """A declared ``logbook`` differing from the built-in one is refused."""
    example_entry_fields.state.declare_logbook = True
    response = post(local_client, {"book": "ops", "logbook": "maintenance"}, logbook="control-room")
    assert response.status_code == 422, response.text
    assert response.json()["field"] == "logbook"
    assert dict_repository.entries == {}


@_POSTERS
def test_create_writable_adapter_receives_declared_values_and_local_copy(
    post,
    local_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """On a writable adapter the request carries the declared values only, the copy keeps provenance."""
    example_entry_fields.state.supports_write = True
    example_entry_fields.state.declare_logbook = True
    session = {"session_id": "abc"}
    response = post(
        local_client,
        {"book": "ops", "logbook": "maintenance", "session_metadata": session, "x": 1},
    )
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert request.logbook == "maintenance"
    assert request.metadata == {"book": "ops", "logbook": "maintenance"}
    metadata = dict_repository.entries["example-1"]["metadata"]
    assert metadata["session_metadata"] == session
    assert metadata["created_via"] == "ariel-web"
    assert metadata["sync_status"] == "pending_sync"


# --- POST /entries: declared values are checked before the adapter is called ---


_SHOTS = ParameterDescriptor(
    name="shots",
    label="Shots",
    description="How many shots were taken",
    param_type="int",
    default=None,
    min_value=1,
    max_value=10,
    section="Entry",
)


@pytest.fixture(name="writable_client")
def writable_client_fixture(
    local_client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """The routes app over a dict repository, the example adapter writable and counting shots."""
    example_entry_fields.state.supports_write = True
    monkeypatch.setattr(
        example_entry_fields,
        "get_entry_field_descriptors",
        lambda: [
            *example_descriptors(declare_logbook=example_entry_fields.state.declare_logbook),
            _SHOTS,
        ],
    )
    return local_client


_FR3_CASES = [
    pytest.param({"day": "2026-10-04"}, "book", 0, id="missing-required"),
    pytest.param({"book": "ops", "day": "not-a-date"}, "day", 0, id="wrong-type-date"),
    pytest.param({"book": "ops", "shots": "many"}, "shots", 0, id="wrong-type-int"),
    pytest.param({"book": "archive"}, "book", 0, id="select-outside-options"),
    pytest.param({"book": "ops", "shots": 11}, "shots", 0, id="number-above-max"),
    pytest.param({"book": "ops", "shots": 0}, "shots", 0, id="number-below-min"),
    pytest.param(
        {"book": "ops", "day": "2026-10-04", "scan": "nope"}, "scan", 1, id="dynamic-outside-live"
    ),
]


@pytest.mark.parametrize(("metadata", "field", "options_calls"), _FR3_CASES)
def test_create_json_invalid_value_is_422_with_no_write(
    metadata: dict,
    field: str,
    options_calls: int,
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """Each refused value answers 422 naming the field; neither the adapter nor the store is written."""
    response = _post_json(writable_client, metadata)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "invalid_entry_field"
    assert body["field"] == field
    assert isinstance(body["detail"], str) and body["detail"]
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}
    assert len(example_entry_fields.state.options_calls) == options_calls


def test_create_json_dynamic_check_costs_one_options_call(
    writable_client: TestClient, example_entry_fields: ExampleEntryFieldsAdapter
) -> None:
    """A refused ``dynamic_select`` was checked with exactly one call carrying its parent."""
    response = _post_json(writable_client, {"book": "ops", "day": "2026-10-04", "scan": "nope"})
    assert response.status_code == 422, response.text
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-04"})]


def test_create_json_valid_post_reaches_the_adapter_with_native_values(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """The adapter double receives the declared values coerced to JSON-native types."""
    response = _post_json(
        writable_client,
        {"book": " physics ", "day": "2026-10-04", "scan": "s1", "shots": "3", "x": 1},
        author="alice",
        tags=["a"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["entry_id"] == "example-1"
    assert body["sync_status"] == "pending_sync"
    (request,) = example_entry_fields.state.created
    assert request.metadata == {"book": "physics", "day": "2026-10-04", "scan": "s1", "shots": 3}
    assert type(request.metadata["shots"]) is int
    assert json.loads(json.dumps(request.metadata)) == request.metadata
    assert request.subject == "Beam"
    assert request.details == "Stable."
    assert request.author == "alice"
    assert request.tags == ["a"]
    assert request.logbook is None
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-04"})]
    assert dict_repository.entries["example-1"]["metadata"]["shots"] == 3


def test_create_json_session_metadata_stays_in_the_local_copy(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """A draft's ``session_metadata`` is kept by ARIEL's copy and never sent to the adapter."""
    session = {"session_id": "abc", "created_via": "ariel-mcp"}
    response = _post_json(
        writable_client,
        {"book": "ops", "session_metadata": session, "sync_status": "synced"},
    )
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert request.metadata == {"book": "ops"}
    assert "session_metadata" not in request.metadata
    assert "sync_status" not in request.metadata
    stored = dict_repository.entries["example-1"]["metadata"]
    assert stored["session_metadata"] == session
    assert stored["sync_status"] == "pending_sync"


@pytest.mark.parametrize("builtin", [None, "", "   "], ids=["absent", "empty", "blank"])
def test_create_json_declared_logbook_fills_an_empty_builtin(
    builtin: str | None,
    writable_client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
) -> None:
    """With a declared ``logbook`` and no built-in value, ``request.logbook`` is the declared one."""
    example_entry_fields.state.declare_logbook = True
    extra = {} if builtin is None else {"logbook": builtin}
    response = _post_json(writable_client, {"book": "ops", "logbook": "maintenance"}, **extra)
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert request.logbook == "maintenance"
    assert request.metadata["logbook"] == "maintenance"


def test_create_json_without_declarations_sends_the_unchanged_request(
    local_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter declaring nothing receives exactly the request the route built before entry fields."""
    example_entry_fields.state.supports_write = True
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [])
    response = local_client.post(
        "/api/entries",
        json={
            "subject": "Beam",
            "details": "Stable.",
            "author": "alice",
            "logbook": "ops-log",
            "shift": "day",
            "tags": ["a", "b"],
            "auth_user": "u",
            "auth_password": "p",
            "metadata": {"book": "nope", "session_metadata": {"session_id": "abc"}},
        },
    )
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert dataclasses.asdict(request) == {
        "subject": "Beam",
        "details": "Stable.",
        "author": "alice",
        "logbook": "ops-log",
        "shift": "day",
        "tags": ["a", "b"],
        "attachment_paths": [],
        "metadata": {},
        "auth_user": "u",
        "auth_password": "p",
    }
    assert example_entry_fields.state.options_calls == []
    stored = dict_repository.entries["example-1"]["metadata"]
    assert stored["session_metadata"] == {"session_id": "abc"}
    assert "book" not in stored


def test_create_json_without_declarations_keeps_an_empty_builtin_logbook(
    local_client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing declared an empty built-in ``logbook`` reaches the adapter as sent."""
    example_entry_fields.state.supports_write = True
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [])
    response = _post_json(local_client, {}, logbook="")
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert request.logbook == ""
    assert request.metadata == {}


def test_create_json_broken_declaration_is_the_500_envelope(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misdeclared adapter answers 500 ``entry_fields_misdeclared`` and nothing is written."""
    broken = ParameterDescriptor(
        name="book", label="Book", description="", param_type="select", default=None
    )
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [broken])
    response = _post_json(writable_client, {"book": "ops"})
    assert response.status_code == 500, response.text
    assert response.json()["code"] == "entry_fields_misdeclared"
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}


# --- POST /entries/upload: the same contract, with a file attached ---

_FILE = ("files", ("note.txt", b"abc", "text/plain"))


@pytest.fixture(name="attachment_store")
def attachment_store_fixture(dict_repository: DictRepository) -> AsyncMock:
    """Plain attachment storage on the dict repository; the returned double records each store."""
    from osprey.services.ariel_search.database.repository import SchemaFacts

    dict_repository.schema_facts = AsyncMock(return_value=SchemaFacts(False, False))
    dict_repository.store_attachment = AsyncMock()
    return dict_repository.store_attachment


def _upload_with_file(client: TestClient, metadata: dict | str, **extra) -> Any:
    raw = metadata if isinstance(metadata, str) else json.dumps(metadata)
    return client.post(
        "/api/entries/upload",
        data={"subject": "Beam", "details": "Stable.", "metadata": raw, **extra},
        files=[_FILE],
    )


@pytest.mark.parametrize(("metadata", "field", "options_calls"), _FR3_CASES)
def test_create_upload_invalid_value_is_422_with_no_write(
    metadata: dict,
    field: str,
    options_calls: int,
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
) -> None:
    """Each refused value answers 422 naming the field; no adapter call, entry, or attachment is written."""
    response = _upload_with_file(writable_client, metadata)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "invalid_entry_field"
    assert body["field"] == field
    assert isinstance(body["detail"], str) and body["detail"]
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}
    attachment_store.assert_not_awaited()
    assert len(example_entry_fields.state.options_calls) == options_calls


def test_create_upload_malformed_metadata_form_is_422_on_the_required_field(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
) -> None:
    """A ``metadata`` form string that is not a JSON object carries no values, so a required one is missing."""
    response = _upload_with_file(writable_client, "{not json")
    assert response.status_code == 422, response.text
    assert response.json()["field"] == "book"
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}
    attachment_store.assert_not_awaited()


def test_create_upload_options_failure_is_502_with_no_write(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
) -> None:
    """A live-options failure answers 502 naming the field before anything is written."""
    example_entry_fields.state.options_mode = "fail"
    response = _upload_with_file(
        writable_client, {"book": "ops", "day": "2026-10-04", "scan": "s1"}
    )
    assert response.status_code == 502, response.text
    assert response.json()["code"] == "entry_field_options_unavailable"
    assert response.json()["field"] == "scan"
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}
    attachment_store.assert_not_awaited()


@pytest.mark.parametrize("shots", ["3", 3], ids=["string-form", "native-form"])
def test_create_upload_valid_post_reaches_the_adapter_with_native_values(
    shots: str | int,
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
) -> None:
    """Form values arriving as strings or as JSON natives reach the adapter coerced; the file stays in ARIEL."""
    response = _upload_with_file(
        writable_client,
        {"book": " physics ", "day": "2026-10-04", "scan": "s1", "shots": shots},
        author="alice",
        tags="a, b",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["entry_id"] == "example-1"
    assert body["sync_status"] == "pending_sync"
    assert body["attachment_count"] == 1
    assert "not published to the external logbook" in body["message"]
    (request,) = example_entry_fields.state.created
    assert request.metadata == {"book": "physics", "day": "2026-10-04", "scan": "s1", "shots": 3}
    assert type(request.metadata["shots"]) is int
    assert request.author == "alice"
    assert request.tags == ["a", "b"]
    assert request.attachment_paths == []
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-04"})]
    attachment_store.assert_awaited_once()
    assert attachment_store.await_args.kwargs["entry_id"] == "example-1"
    stored = dict_repository.entries["example-1"]
    assert stored["metadata"]["shots"] == 3
    assert [a["filename"] for a in stored["attachments"]] == ["note.txt"]


def test_create_upload_undeclared_metadata_never_reaches_the_adapter(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
) -> None:
    """Undeclared keys and a draft's ``session_metadata`` stay in ARIEL's copy, out of the adapter request."""
    session = {"session_id": "abc", "created_via": "ariel-mcp"}
    response = _upload_with_file(
        writable_client,
        {"book": "ops", "x": 1, "session_metadata": session, "sync_status": "synced"},
    )
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert request.metadata == {"book": "ops"}
    stored = dict_repository.entries["example-1"]["metadata"]
    assert stored["session_metadata"] == session
    assert stored["sync_status"] == "pending_sync"
    assert "x" not in stored
    attachment_store.assert_awaited_once()


@pytest.mark.parametrize("builtin", [None, ""], ids=["absent", "empty"])
def test_create_upload_declared_logbook_fills_an_empty_builtin(
    builtin: str | None,
    writable_client: TestClient,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
) -> None:
    """With a declared ``logbook`` and no built-in form value, ``request.logbook`` is the declared one."""
    example_entry_fields.state.declare_logbook = True
    extra = {} if builtin is None else {"logbook": builtin}
    response = _upload_with_file(
        writable_client, {"book": "ops", "logbook": "maintenance"}, **extra
    )
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert request.logbook == "maintenance"
    assert request.metadata["logbook"] == "maintenance"
    attachment_store.assert_awaited_once()


def test_create_upload_without_declarations_sends_no_client_metadata(
    local_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter declaring nothing receives the form's built-in fields and no client metadata at all."""
    example_entry_fields.state.supports_write = True
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [])
    response = _upload_with_file(
        local_client,
        {"book": "nope", "session_metadata": {"session_id": "abc"}},
        author="alice",
        logbook="ops-log",
        shift="day",
        tags="a,b",
        auth_user="u",
        auth_password="p",
    )
    assert response.status_code == 200, response.text
    (request,) = example_entry_fields.state.created
    assert dataclasses.asdict(request) == {
        "subject": "Beam",
        "details": "Stable.",
        "author": "alice",
        "logbook": "ops-log",
        "shift": "day",
        "tags": ["a", "b"],
        "attachment_paths": [],
        "metadata": {},
        "auth_user": "u",
        "auth_password": "p",
    }
    assert example_entry_fields.state.options_calls == []
    stored = dict_repository.entries["example-1"]
    assert stored["metadata"]["session_metadata"] == {"session_id": "abc"}
    assert "book" not in stored["metadata"]
    assert response.json()["attachment_count"] == 1
    attachment_store.assert_awaited_once()


def test_create_upload_broken_declaration_is_the_500_envelope(
    writable_client: TestClient,
    dict_repository: DictRepository,
    example_entry_fields: ExampleEntryFieldsAdapter,
    attachment_store: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misdeclared adapter answers 500 ``entry_fields_misdeclared`` and nothing is written."""
    broken = ParameterDescriptor(
        name="book", label="Book", description="", param_type="select", default=None
    )
    monkeypatch.setattr(example_entry_fields, "get_entry_field_descriptors", lambda: [broken])
    response = _upload_with_file(writable_client, {"book": "ops"})
    assert response.status_code == 500, response.text
    assert response.json()["code"] == "entry_fields_misdeclared"
    assert example_entry_fields.state.created == []
    assert dict_repository.entries == {}
    attachment_store.assert_not_awaited()

"""Coverage for `GET /plans/session`: every authored session plan, validated or not.

`GET /plans` is the registry — the set the manager can run — and it deliberately
quarantines a session file with no passing validation record (the load gate in
`plan_loader`). That is right for every consumer that resolves a plan to RUN it,
and wrong for the one consumer that needs to RETIRE plans it did not author: a
cleanup that lists the registry and deletes what it sees never sees a plan whose
validation is still in flight, and that plan outlives the cleanup.

The scan-stack E2E fixture is that consumer. An agent authors a plan, asks for
validation, and its client gives up before the bridge's dry run finishes; the
next attempt's cleanup runs while the dry run is still going, lists the registry,
finds nothing to retire — and the validation then passes, so the next agent finds
a ready-made plan where the floor demanded it author one.

This listing is the counterpart of the write and delete routes, scoped to the
same single directory, and reports what is on disk regardless of validation
state. Mirrors `test_session_plan_delete.py`'s isolation fixture and helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records
from tests.services.bluesky_bridge._session_plans import session_plan_source, write_session_plan


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _author(name: str, *, validated: bool) -> Path:
    """Put a session plan on disk; record a passing validation only when asked.

    The unvalidated case is the one this module exists for: a file the registry
    (`GET /plans`) will not list, which is exactly the state a plan is in while
    its dry run is still running.
    """
    source = session_plan_source(name)
    path = write_session_plan(name, source)
    if validated:
        validation_records.record(hash_plan_body(source))
    plan_loader.reset_facility_plans()
    return path


def _registered_names(client: TestClient) -> set[str]:
    response = client.get("/plans")
    assert response.status_code == 200, response.text
    return {spec["name"] for spec in response.json()}


def _authored(client: TestClient) -> dict[str, bool]:
    """Name → validated flag, as `GET /plans/session` reports it."""
    response = client.get("/plans/session")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list), body
    return {entry["name"]: entry["validated"] for entry in body}


def test_an_unvalidated_session_plan_is_listed_even_though_the_registry_hides_it(
    client: TestClient,
) -> None:
    """The gap the scan-stack cleanup fell through, pinned at the route level.

    The registry quarantining the file is correct and is asserted too — this
    listing exists precisely because that quarantine is not negotiable.
    """
    _author("in_flight", validated=False)

    assert "in_flight" not in _registered_names(client)
    assert _authored(client) == {"in_flight": False}


def test_a_validated_session_plan_is_listed_as_validated(client: TestClient) -> None:
    _author("ready", validated=True)

    assert "ready" in _registered_names(client)
    assert _authored(client) == {"ready": True}


def test_the_listing_covers_only_the_session_directory(client: TestClient) -> None:
    """Shipped plans are registered but never authored: they do not appear here.

    The route is scoped the way the write and delete routes are — one
    directory — so a consumer retiring everything it lists can never be handed
    a shipped/preset/facility name to delete.
    """
    _author("mine", validated=True)

    registered = _registered_names(client)
    assert "mine" in registered
    shipped = registered - {"mine"}
    assert shipped, "the shipped plan tier should register something to contrast against"

    assert set(_authored(client)) == {"mine"}


def test_everything_listed_can_be_retired_without_knowing_a_name(client: TestClient) -> None:
    """The cleanup's actual loop: list, delete each, nothing is left.

    Both validation states are present so the loop is proven over the one the
    registry would have hidden.
    """
    in_flight = _author("in_flight", validated=False)
    ready = _author("ready", validated=True)

    for name in _authored(client):
        response = client.delete(f"/plans/session/{name}")
        assert response.status_code == 200, response.text
        assert response.json()["deleted"] is True

    assert _authored(client) == {}
    assert not in_flight.exists()
    assert not ready.exists()
    assert "ready" not in _registered_names(client)


def test_an_empty_session_directory_lists_nothing(client: TestClient) -> None:
    assert _authored(client) == {}

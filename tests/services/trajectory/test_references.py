"""References: saved from a fresh TbT shot, imported from a file, never across modes or targets.

The routes run on the real gated app with the threading-loop stand-ins of
``test_doors``: a reference saved from the pending shot and selected subtracts
to zero, an import is checked before anything is stored, the readings route
answers exactly the import body, and a reference of the other readout mode is
never selected for the shot on show.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest
import tests.services.trajectory.test_doors as door_tests
from fastapi.testclient import TestClient

from osprey.services.trajectory.app import CLIENT_HEADER, MAX_REFERENCE_IMPORT_BYTES
from osprey.services.trajectory.state import ReferenceImport, ShotResult
from osprey.services.trajectory.store import ForeignReference, TrajectoryStore

# The door tests' deployment, state directory and threading-loop client, shared here.
_deployment = door_tests._deployment
state_dir = door_tests.state_dir
machine = door_tests.machine
loop_client = door_tests.loop_client

ALICE_SECRET = door_tests.ALICE_SECRET
BOB_SECRET = door_tests.BOB_SECRET
X_ROWS = door_tests.X_ROWS
_as = door_tests._as
_code = door_tests._code
_pend = door_tests._pend
_shot = door_tests._shot

pytestmark = pytest.mark.no_auth_seam

PANEL = "/panel/api"


def _holder(client: TestClient) -> dict[str, str]:
    grant = client.post(f"{PANEL}/session", headers=_as(ALICE_SECRET, PANEL)).json()
    return _as(ALICE_SECRET, PANEL, **{CLIENT_HEADER: grant["client_id"]})


def _client_id(headers: dict[str, str]) -> str:
    return headers[CLIENT_HEADER]


def _import_body(name: str = "from-file", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "readings": {
            X_ROWS[0]: {"x": 0.11, "y": -0.11},
            X_ROWS[1]: {"x": None, "y": 0.5},
        },
        "readout": "tbt",
        "select": False,
        "source_target": "standin",
        "source_model": "SR",
    }
    body.update(extra)
    return body


def _stored_shot(state_dir: Path, **update: Any) -> ShotResult:
    shot = _shot("20260101T000000000000Z-shot0002").model_copy(update=update)
    asyncio.run(TrajectoryStore(state_dir).save_shot(shot))
    return shot


# -- save from a shot ----------------------------------------------------------


def test_a_saved_and_selected_reference_subtracts_the_shown_shot_to_zero(
    loop_client: TestClient,
) -> None:
    holder = _holder(loop_client)
    _pend(loop_client, _client_id(holder))
    shot = _shot()

    saved = loop_client.post(f"{PANEL}/references", headers=holder, json={"name": "golden"})
    assert saved.status_code == 200, saved.text
    selected = loop_client.post(
        f"{PANEL}/references/select", headers=holder, json={"name": "golden"}
    )
    assert selected.status_code == 200, selected.text

    reference = loop_client.get(f"{PANEL}/state", headers=holder).json()["reference"]
    assert reference["name"] == "golden"
    assert reference["readout"] == "tbt"
    readings = reference["readings"]
    for bpm, x_row, y_row in zip(shot.bpm_order, shot.x, shot.y, strict=True):
        assert shot.x[x_row] - readings[bpm]["x"] == pytest.approx(0.0)
        assert shot.y[y_row] - readings[bpm]["y"] == pytest.approx(0.0)


def test_a_saved_reference_carries_its_mode_and_provenance(loop_client: TestClient) -> None:
    holder = _holder(loop_client)
    _pend(loop_client, _client_id(holder))
    assert loop_client.post(f"{PANEL}/references", headers=holder, json={"name": "g"}).is_success

    stored = loop_client.get(f"{PANEL}/references/g/readings", headers=holder)
    assert stored.status_code == 200, stored.text
    body = stored.json()
    assert body["readout"] == "tbt"
    assert body["source_model"] == "SR"
    assert body["source_target"] is None  # the app's store is the state directory's own tree
    assert set(body["readings"]) == set(X_ROWS)


def test_a_shot_that_is_not_fresh_is_refused_as_a_reference(
    loop_client: TestClient, state_dir: Path
) -> None:
    shot = _stored_shot(state_dir, fresh=False)
    response = loop_client.post(
        f"{PANEL}/references",
        headers=_as(ALICE_SECRET, PANEL),
        json={"name": "stale", "shot_id": shot.shot_id},
    )
    assert response.status_code == 409
    assert _code(response) == "shot_not_fresh"
    assert not (state_dir / "references" / "SR" / "stale.json").exists()


def test_a_pseudo_orbit_shot_is_refused_as_a_reference(
    loop_client: TestClient, state_dir: Path
) -> None:
    shot = _stored_shot(state_dir, readout="pseudo_orbit", n_turns=8)
    response = loop_client.post(
        f"{PANEL}/references",
        headers=_as(ALICE_SECRET, PANEL),
        json={"name": "orbit", "shot_id": shot.shot_id},
    )
    assert response.status_code == 409
    assert _code(response) == "readout_mismatch"


def test_the_load_route_answers_the_match_without_the_readings(loop_client: TestClient) -> None:
    holder = _holder(loop_client)
    _pend(loop_client, _client_id(holder))
    loop_client.post(f"{PANEL}/references", headers=holder, json={"name": "golden"})
    loaded = loop_client.get(f"{PANEL}/references/golden", headers=holder).json()
    assert loaded["readout"] == "tbt"
    assert "readings" not in loaded


# -- import and the readings route ---------------------------------------------


def test_the_readings_route_answers_exactly_the_import_body(loop_client: TestClient) -> None:
    headers = _as(ALICE_SECRET, PANEL)
    body = _import_body()
    imported = loop_client.post(f"{PANEL}/references/import", headers=headers, json=body)
    assert imported.status_code == 200, imported.text

    answered = loop_client.get(f"{PANEL}/references/from-file/readings", headers=headers)
    assert answered.status_code == 200, answered.text
    assert answered.json() == body

    again = loop_client.post(f"{PANEL}/references/import", headers=headers, json=answered.json())
    assert again.status_code == 200, again.text
    assert again.json()["sha256"] == imported.json()["sha256"]
    ReferenceImport.model_validate(answered.json())


def test_an_import_naming_a_bpm_the_tool_does_not_read_is_refused(
    loop_client: TestClient, state_dir: Path
) -> None:
    body = _import_body(readings={"SR:BPM99:X": {"x": 0.0, "y": 0.0}})
    response = loop_client.post(
        f"{PANEL}/references/import", headers=_as(ALICE_SECRET, PANEL), json=body
    )
    assert response.status_code == 400
    assert _code(response) == "unknown_bpm"
    assert "SR:BPM99:X" in response.json()["detail"]["detail"]
    assert not (state_dir / "references").exists()


@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN"])
def test_an_import_with_a_non_finite_reading_is_refused(
    loop_client: TestClient, value: str
) -> None:
    text = json.dumps(_import_body()).replace("0.11", value, 1)
    response = loop_client.post(
        f"{PANEL}/references/import",
        headers={**_as(ALICE_SECRET, PANEL), "content-type": "application/json"},
        content=text,
    )
    assert response.status_code == 400
    assert _code(response) == "reading_not_finite"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (_import_body(name="bad name!"), 400, "invalid_name"),
        (_import_body(name="current"), 400, "invalid_name"),
        (_import_body(readout="fast"), 400, "bad_request"),
        (_import_body(readings={}), 400, "bad_request"),
        ({**_import_body(), "extra": 1}, 400, "bad_request"),
    ],
)
def test_a_malformed_import_is_refused(
    loop_client: TestClient, body: dict[str, Any], status: int, code: str
) -> None:
    response = loop_client.post(
        f"{PANEL}/references/import", headers=_as(ALICE_SECRET, PANEL), json=body
    )
    assert response.status_code == status, response.text
    assert _code(response) == code


def test_an_import_over_the_size_cap_is_refused(loop_client: TestClient) -> None:
    text = json.dumps(_import_body(padding="x" * MAX_REFERENCE_IMPORT_BYTES))
    response = loop_client.post(
        f"{PANEL}/references/import",
        headers={**_as(ALICE_SECRET, PANEL), "content-type": "application/json"},
        content=text,
    )
    assert response.status_code == 413
    assert _code(response) == "reference_too_large"


def test_an_import_that_selects_makes_it_current_for_the_holder(loop_client: TestClient) -> None:
    holder = _holder(loop_client)
    _pend(loop_client, _client_id(holder))
    response = loop_client.post(
        f"{PANEL}/references/import", headers=holder, json=_import_body(select=True)
    )
    assert response.status_code == 200, response.text
    assert response.json()["reference"]["name"] == "from-file"
    assert response.json()["step"]["reference"] == "from-file"
    listed = loop_client.get(f"{PANEL}/references", headers=holder).json()
    assert listed["current"] == "from-file"


def test_an_import_that_selects_needs_the_holder_and_stores_nothing_without(
    loop_client: TestClient, state_dir: Path
) -> None:
    holder = _holder(loop_client)
    _pend(loop_client, _client_id(holder))
    other = _as(BOB_SECRET, PANEL, **{CLIENT_HEADER: "not-the-holder"})
    response = loop_client.post(
        f"{PANEL}/references/import", headers=other, json=_import_body(select=True)
    )
    assert response.status_code in {403, 409}, response.text
    assert not (state_dir / "references" / "SR" / "from-file.json").exists()


# -- never across modes --------------------------------------------------------


def test_the_other_mode_cannot_be_selected_for_a_tbt_shot(loop_client: TestClient) -> None:
    holder = _holder(loop_client)
    _pend(loop_client, _client_id(holder))
    orbit = _import_body(name="orbit", readout="pseudo_orbit")
    stored = loop_client.post(f"{PANEL}/references/import", headers=holder, json=orbit)
    assert stored.status_code == 200, stored.text

    selected = loop_client.post(
        f"{PANEL}/references/select", headers=holder, json={"name": "orbit"}
    )
    assert selected.status_code == 409
    assert _code(selected) == "readout_mismatch"

    imported = loop_client.post(
        f"{PANEL}/references/import",
        headers=holder,
        json=_import_body(name="orbit2", readout="pseudo_orbit", select=True),
    )
    assert imported.status_code == 409
    assert _code(imported) == "readout_mismatch"
    listed = loop_client.get(f"{PANEL}/references", headers=holder).json()
    assert listed["current"] is None
    assert "orbit2" not in listed["names"]


def test_a_pseudo_orbit_reference_is_selected_for_a_pseudo_orbit_shot(
    loop_client: TestClient, state_dir: Path
) -> None:
    _stored_shot(state_dir, readout="pseudo_orbit", n_turns=8)
    holder = _holder(loop_client)
    orbit = _import_body(name="orbit", readout="pseudo_orbit")
    assert loop_client.post(f"{PANEL}/references/import", headers=holder, json=orbit).is_success
    selected = loop_client.post(
        f"{PANEL}/references/select", headers=holder, json={"name": "orbit"}
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["reference"]["readout"] == "pseudo_orbit"


# -- never across targets ------------------------------------------------------


def _foreign(state_dir: Path) -> None:
    store = TrajectoryStore(state_dir / "elsewhere", target="standin")
    asyncio.run(store.save_reference("SR", "theirs", {X_ROWS[0]: (0.0, 0.0)}))
    source = state_dir / "elsewhere" / "standin" / "references" / "SR" / "theirs.json"
    destination = state_dir / "references" / "SR" / "theirs.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


@pytest.mark.parametrize("path", ["references/theirs", "references/theirs/readings"])
def test_a_stored_file_of_another_target_is_refused_by_name(
    loop_client: TestClient, state_dir: Path, path: str
) -> None:
    _foreign(state_dir)
    response = loop_client.get(f"{PANEL}/{path}", headers=_as(ALICE_SECRET, PANEL))
    assert response.status_code == 409
    assert _code(response) == "reference_foreign"


def test_selecting_a_file_of_another_target_is_refused(
    loop_client: TestClient, state_dir: Path
) -> None:
    _foreign(state_dir)
    holder = _holder(loop_client)
    response = loop_client.post(
        f"{PANEL}/references/select", headers=holder, json={"name": "theirs"}
    )
    assert response.status_code == 409
    assert _code(response) == "reference_foreign"


# -- the store -----------------------------------------------------------------


async def test_the_store_answers_a_reference_file_as_its_import_body(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, target="va")
    await store.save_reference(
        "SR",
        "gold",
        {"A": (0.1, math.nan), "B": (None, 0.2)},
        readout="pseudo_orbit",
        source_target="standin",
        source_model="SR2",
    )
    stored = await store.reference_file("SR", "gold")
    assert stored.model_dump(mode="json") == {
        "name": "gold",
        "readings": {"A": {"x": 0.1, "y": None}, "B": {"x": None, "y": 0.2}},
        "readout": "pseudo_orbit",
        "select": False,
        "source_target": "standin",
        "source_model": "SR2",
    }
    loaded = await store.load_reference("SR", "gold", ["A", "C"])
    assert loaded.load.readout == "pseudo_orbit"
    assert loaded.load.readings is not None
    assert loaded.load.readings["A"].x == 0.1
    assert loaded.load.readings["C"].x is None


async def test_the_store_refuses_a_file_naming_another_target(tmp_path: Path) -> None:
    theirs = TrajectoryStore(tmp_path, target="standin")
    await theirs.save_reference("SR", "gold", {"A": (0.0, 0.0)})
    ours = TrajectoryStore(tmp_path, target="va")
    path = tmp_path / "va" / "references" / "SR" / "gold.json"
    path.parent.mkdir(parents=True)
    path.write_bytes((tmp_path / "standin" / "references" / "SR" / "gold.json").read_bytes())
    with pytest.raises(ForeignReference):
        await ours.load_reference("SR", "gold", ["A"])
    with pytest.raises(ForeignReference):
        await ours.reference_file("SR", "gold")

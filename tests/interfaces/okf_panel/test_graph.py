"""The Knowledge panel's Graph view backend.

The payload is read from the Turtle the graph view writer produces, so these
tests write that file with the writer itself: a change to what the writer
emits shows up here as a change to what the panel draws.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.facility.records import FacilityRecords
from osprey.facility.validate import FACILITY_HEADER
from osprey.facility.views.graph import graph_text
from osprey.interfaces.okf_panel.app import create_app
from osprey.interfaces.okf_panel.graph import (
    NODE_KINDS,
    attach_pages,
    page_references,
    read_graph,
)

BUNDLE = Path(__file__).parent / "fixtures" / "bundle"


def _doc() -> dict[str, Any]:
    return {
        "schema": FACILITY_HEADER,
        "identity": {"code": "fx"},
        "places": [
            {"id": "M", "level": "machine", "description": "The machine."},
            {"id": "M/S1", "level": "sector"},
        ],
        "devices": [
            {"id": "M/BPM1", "class": "BeamPositionMonitor", "place": "M/S1"},
            {"id": "M/Q1", "class": "Quadrupole", "place": "M/S1"},
        ],
        "channels": [
            {
                "id": "M:BPM1:X",
                "on": {"device": "M/BPM1"},
                "signal": "position_x_readback",
                "description": "Horizontal position.",
            },
            {
                "id": "M:Q1:SP",
                "role": "setpoint",
                "pair": "M:Q1:RB",
                "on": {"device": "M/Q1"},
                "signal": "current_setpoint",
            },
            {"id": "M:Q1:RB", "on": {"device": "M/Q1"}, "signal": "current_readback"},
            {"id": "M:TUNE", "on": {"place": "M"}},
        ],
    }


@pytest.fixture
def ttl(tmp_path: Path) -> Path:
    path = tmp_path / "facility.ttl"
    path.write_text(graph_text(FacilityRecords(_doc())), encoding="utf-8")
    return path


def _by_label(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["label"]: node for node in payload["nodes"]}


def _relations(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    nodes = payload["nodes"]
    return {(nodes[a]["label"], relation, nodes[b]["label"]) for a, b, relation in payload["edges"]}


def test_every_record_is_a_node_of_its_kind(ttl):
    nodes = _by_label(read_graph(ttl))

    assert {node["kind"] for node in nodes.values()} == set(NODE_KINDS)
    assert nodes["M/S1"]["kind"] == "place"
    assert nodes["M/BPM1"]["kind"] == "device"
    assert nodes["M/BPM1"]["type"] == "BeamPositionMonitor"
    assert nodes["M:BPM1:X"]["kind"] == "channel"
    assert nodes["M:BPM1:X"]["props"]["description"] == "Horizontal position."
    assert nodes["position_x_readback"]["kind"] == "signal"
    assert nodes["Quadrupole"]["kind"] == "class"


def test_no_vocabulary_term_is_a_node(ttl):
    labels = set(_by_label(read_graph(ttl)))

    assert not {"hasBinding", "fullPv", "deviceId"} & labels


def test_record_nodes_carry_their_facility_file_id(ttl):
    nodes = _by_label(read_graph(ttl))

    assert nodes["M:Q1:SP"]["record"] == "M:Q1:SP"
    assert nodes["M/Q1"]["record"] == "M/Q1"
    assert nodes["M/S1"]["record"] == "M/S1"
    assert "record" not in nodes["Quadrupole"]


def test_edges_follow_the_graph_and_name_their_relation(ttl):
    relations = _relations(read_graph(ttl))

    assert ("M/BPM1", "hasBinding", "M:BPM1:X") in relations
    assert ("M", "hasBinding", "M:TUNE") in relations
    assert ("M:BPM1:X", "readsSignal", "position_x_readback") in relations
    assert ("M:Q1:SP", "writesSignal", "current_setpoint") in relations
    assert ("M/Q1", "type", "Quadrupole") in relations


def test_a_device_is_located_in_its_place_and_a_place_is_part_of_its_parent(ttl):
    payload = read_graph(ttl)
    relations = _relations(payload)

    assert ("M/BPM1", "locatedIn", "M/S1") in relations
    assert ("M/Q1", "locatedIn", "M/S1") in relations
    assert ("M/S1", "partOf", "M") in relations
    assert not any(relation == "inPlace" for _, relation, _ in relations)
    assert "derived" not in payload


def test_a_classless_device_is_a_device_node_and_its_place_keeps_its_own(tmp_path):
    doc = _doc()
    doc["devices"].append({"id": "M/BOX", "place": "M/S1"})
    path = tmp_path / "facility.ttl"
    path.write_text(graph_text(FacilityRecords(doc)), encoding="utf-8")

    payload = read_graph(path)
    nodes = payload["nodes"]
    box = [i for i, node in enumerate(nodes) if node["label"] == "M/BOX"]
    places = [i for i, node in enumerate(nodes) if node["label"] == "M/S1"]

    assert len(box) == 1
    assert nodes[box[0]]["kind"] == "device"
    assert nodes[box[0]]["type"] is None
    assert nodes[box[0]]["record"] == "M/BOX"
    assert len(places) == 1
    assert nodes[places[0]]["kind"] == "place"
    located = [edge for edge in payload["edges"] if edge[0] == box[0] and edge[2] == "locatedIn"]
    assert located == [[box[0], places[0], "locatedIn"]]


def test_the_payload_is_the_same_on_every_read(ttl):
    assert read_graph(ttl) == read_graph(ttl)


def test_pages_are_attached_to_the_nodes_they_name(ttl, tmp_path):
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "devices").mkdir(parents=True)
    (bundle_dir / "devices" / "bpm.md").write_text(
        "---\ntype: device\ntitle: BPM\ndescription: A monitor.\n---\n\n"
        "See [the monitor](facility:M/BPM1) and [X](facility:M:BPM1:X).\n",
        encoding="utf-8",
    )
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    records = FacilityRecords(_doc())
    pages = page_references(OKFBundle(bundle_dir), records)
    nodes = _by_label(attach_pages(read_graph(ttl), pages))

    assert nodes["M/BPM1"]["pages"] == ["devices/bpm"]
    assert nodes["M:BPM1:X"]["pages"] == ["devices/bpm"]
    assert "pages" not in nodes["M/Q1"]


# ---------------------------------------------------------------------------
# The panel serving it
# ---------------------------------------------------------------------------


def test_api_graph_serves_the_payload(ttl):
    client = TestClient(create_app(str(BUNDLE), graph_path=str(ttl)))

    response = client.get("/api/graph")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == read_graph(ttl)


def test_the_page_states_that_a_graph_is_served(ttl):
    with_graph = TestClient(create_app(str(BUNDLE), graph_path=str(ttl))).get("/").text
    without = TestClient(create_app(str(BUNDLE))).get("/").text

    assert 'data-graph="true"' in with_graph
    assert 'data-graph="false"' in without


def test_no_graph_configured_is_a_404_and_says_why():
    client = TestClient(create_app(str(BUNDLE)))

    response = client.get("/api/graph")

    assert response.status_code == 404
    assert response.json() == {"error": "services.graphdb.ttl_path not configured"}


def test_an_unreadable_graph_degrades_to_no_graph_view(tmp_path, caplog):
    broken = tmp_path / "facility.ttl"
    broken.write_text("this is not turtle <<<", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        client = TestClient(create_app(str(BUNDLE), graph_path=str(broken)))

    assert client.get("/api/graph").status_code == 404
    assert client.get("/api/concepts").status_code == 200
    assert "cannot be read" in caplog.text


def test_a_missing_graph_file_degrades_to_no_graph_view(tmp_path):
    client = TestClient(create_app(str(BUNDLE), graph_path=str(tmp_path / "absent.ttl")))

    assert client.get("/api/graph").status_code == 404


def test_the_graph_is_served_without_a_bundle(ttl):
    client = TestClient(create_app(bundle_path=None, graph_path=str(ttl)))

    assert client.get("/api/graph").status_code == 200


def test_the_configured_graph_is_read_relative_to_the_render(monkeypatch, tmp_path):
    import osprey_connectors.workspace as workspace

    render = tmp_path / "render"
    (render / "data" / "graph").mkdir(parents=True)
    (render / "data" / "graph" / "facility.ttl").write_text(
        graph_text(FacilityRecords(_doc())), encoding="utf-8"
    )
    (render / "facility.json").write_text(json.dumps(_doc()), encoding="utf-8")
    (render / "config.yml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(render / "config.yml"))
    monkeypatch.setattr(
        workspace,
        "load_osprey_config",
        lambda: {
            "config_dir": str(render),
            "services": {"graphdb": {"ttl_path": "./data/graph/facility.ttl"}},
        },
    )

    client = TestClient(create_app(str(BUNDLE)))

    labels = {node["label"] for node in client.get("/api/graph").json()["nodes"]}
    assert "M/BPM1" in labels

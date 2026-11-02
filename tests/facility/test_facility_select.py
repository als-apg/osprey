"""The facility questions the e2e lanes ask, answered on the shipped trees.

Every proof asserts a relation to the facility file or the view it renders,
never a figure copied out of them: an edit to a shipped tree moves the answer
and the proof with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.facility import FACILITY_FILE
from osprey.facility.build import build_facility
from osprey.facility.render import render_facility_outputs
from osprey_connectors.simulation.view import SimulatorView
from tests.e2e._facility_select import (
    bounded_corrector_of,
    bounded_setpoint,
    facility_view,
    graph_probe_device,
    lane_model,
    monitor_addresses,
    monitors_in_plane,
    seeded_readback,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITIES = REPO_ROOT / "src/osprey/templates/facilities"
CONTROL_ASSISTANT = FACILITIES / "example"
HELLO_WORLD = FACILITIES / "hello_world"


@pytest.fixture(scope="module")
def example(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SimulatorView, dict]]:
    """The example tree rendered once: its simulator view and its facility file."""
    root = tmp_path_factory.mktemp("example-render")
    view = facility_view(CONTROL_ASSISTANT, root)
    document = json.loads((root / FACILITY_FILE).read_text(encoding="utf-8"))
    yield view, document


@pytest.mark.parametrize("tree", [HELLO_WORLD, CONTROL_ASSISTANT], ids=lambda tree: tree.name)
def test_each_shipped_facility_answers_the_e2e_lanes_channel_questions(
    tmp_path: Path, tree: Path
) -> None:
    render_facility_outputs(tmp_path, build_facility(tree, project_name=tree.name), {}, tree)
    view = SimulatorView.of_render(tmp_path)

    reading = seeded_readback(view)
    assert reading in view.channels()
    assert view.seed(reading) is not None

    setpoint, value = bounded_setpoint(view)
    channel = view.channel(setpoint)
    assert channel.role == "setpoint"
    assert channel.writable
    assert channel.value_range is not None
    low, high = channel.value_range
    assert low <= value <= high


def test_facility_view_serves_the_trees_channels(tmp_path: Path) -> None:
    view = facility_view(HELLO_WORLD, tmp_path)
    document = build_facility(HELLO_WORLD, project_name=HELLO_WORLD.name)
    assert set(view.channels()) == {channel["id"] for channel in document["channels"]}


def test_facility_view_creates_the_root_it_renders_into(tmp_path: Path) -> None:
    view = facility_view(HELLO_WORLD, tmp_path / "probe_view")
    document = build_facility(HELLO_WORLD, project_name=HELLO_WORLD.name)
    assert set(view.channels()) == {channel["id"] for channel in document["channels"]}


def test_lane_model_is_the_bounded_correctors_model(example: tuple[SimulatorView, dict]) -> None:
    view, _ = example
    found = bounded_corrector_of(view)
    assert found is not None
    assert lane_model(view) == found[2]


def test_lane_model_refuses_a_view_without_a_bounded_corrector(tmp_path: Path) -> None:
    view = facility_view(HELLO_WORLD, tmp_path)
    assert bounded_corrector_of(view) is None
    with pytest.raises(AssertionError, match="no corrector of the served view"):
        lane_model(view)


def test_monitors_in_plane_are_monitors_of_that_plane_in_address_order(
    example: tuple[SimulatorView, dict],
) -> None:
    view, _ = example
    model = lane_model(view)
    picked = monitors_in_plane(view, "x", 3, model=model)
    assert len(picked) == 3
    assert list(picked) == sorted(monitor_addresses(view, plane="x", model=model))[:3]
    for address in picked:
        binding = view.binding(address)
        assert binding is not None
        assert binding.role == "monitor"
        assert binding.plane == "x"


def test_monitors_in_plane_scopes_to_the_model(example: tuple[SimulatorView, dict]) -> None:
    view, _ = example
    model = lane_model(view)
    picked = monitors_in_plane(view, "x", 3, model=model)
    assert all(view.binding(address).model == model for address in picked)  # type: ignore[union-attr]
    unscoped = sorted(monitor_addresses(view, plane="x"))[:3]
    assert list(picked) != unscoped, "the example's unscoped run is another model's"


def test_monitors_in_plane_refuses_a_short_view(example: tuple[SimulatorView, dict]) -> None:
    view, _ = example
    model = lane_model(view)
    available = len(monitor_addresses(view, plane="y", model=model))
    with pytest.raises(AssertionError, match=f"serves {available} monitor"):
        monitors_in_plane(view, "y", available + 1, model=model)


def _on(document: dict[str, Any], device_id: str) -> list[dict[str, Any]]:
    return [c for c in document["channels"] if (c.get("on") or {}).get("device") == device_id]


def _qualifies(document: dict[str, Any], device: dict[str, Any], model: str) -> bool:
    positions = [d.get("s") for d in document["devices"] if d.get("model") == model]
    channels = _on(document, device["id"])
    return (
        device.get("model") == model
        and "s" in device
        and "place" in device
        and positions.count(device["s"]) == 1
        and len(channels) >= 3
        and sum(c.get("role") == "setpoint" for c in channels) == 1
    )


def test_graph_probe_device_on_the_example_is_drawn_from_the_lane_model(
    example: tuple[SimulatorView, dict],
) -> None:
    view, document = example
    model = lane_model(view)
    device = graph_probe_device(document, model=model)
    record = next(d for d in document["devices"] if d["id"] == device.device_id)

    assert device.model == model
    assert _qualifies(document, record, model)

    in_model = [d for d in document["devices"] if d.get("model") == model and "s" in d]
    per_class = {d["class"]: sum(o["class"] == d["class"] for o in in_model) for d in in_model}
    qualifying_classes = {d["class"] for d in in_model if _qualifies(document, d, model)}
    assert all(per_class[device.klass] <= per_class[c] for c in qualifying_classes)

    same_class = sorted(d["s"] for d in in_model if d["class"] == device.klass)
    qualifying_of_class = [
        d["s"] for d in in_model if d["class"] == device.klass and _qualifies(document, d, model)
    ]
    assert device.s == min(qualifying_of_class)

    assert device.class_count_in_model == len(same_class)
    assert device.class_count_in_facility == sum(
        d.get("class") == device.klass for d in document["devices"]
    )
    assert device.class_span == (same_class[0], same_class[-1])
    gaps = [b - a for a, b in zip(same_class, same_class[1:], strict=False)]
    assert device.class_spacing == (min(gaps), max(gaps))

    assert device.previous is not None
    assert device.next is not None
    previous_s, next_s = device.previous[1], device.next[1]
    assert previous_s < device.s < next_s
    assert not [d for d in in_model if previous_s < d["s"] < device.s]
    assert not [d for d in in_model if device.s < d["s"] < next_s]

    channels = _on(document, device.device_id)
    assert device.bindings == tuple(sorted(c["id"] for c in channels))
    assert [c["id"] for c in channels if c.get("role") == "setpoint"] == [device.setpoint]
    assert view.channel(device.setpoint).role == "setpoint"


def _device(device_id: str, klass: str, s: float) -> dict[str, Any]:
    return {
        "id": device_id,
        "class": klass,
        "names": [device_id.split("/")[-1]],
        "label": device_id,
        "model": "M",
        "place": "TOP/S1",
        "s": s,
        "ordinalInPlace": 1,
        "ordinalInModel": 1,
    }


def _channels(device_id: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{device_id}:SP", "role": "setpoint", "on": {"device": device_id}},
        {"id": f"{device_id}:RB", "role": "readback", "on": {"device": device_id}},
        {"id": f"{device_id}:ST", "role": "readback", "on": {"device": device_id}},
    ]


def _document(devices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "identity": {"code": "t"},
        "places": [{"id": "TOP", "names": ["top"]}],
        "devices": devices,
        "channels": [c for d in devices for c in _channels(d["id"])],
    }


def test_graph_probe_device_takes_the_smallest_class() -> None:
    document = _document(
        [
            _device("M/A1", "Common", 1.0),
            _device("M/A2", "Common", 2.0),
            _device("M/B2", "Rare", 4.0),
            _device("M/B1", "Rare", 3.0),
            _device("M/A3", "Common", 5.0),
        ]
    )
    device = graph_probe_device(document, model="M")
    assert device.device_id == "M/B1"
    assert device.class_count_in_model == 2
    assert device.previous == ("A2", 2.0, "Common")
    assert device.next == ("B2", 4.0, "Rare")


def test_graph_probe_device_skips_shared_positions() -> None:
    document = _document(
        [
            _device("M/A1", "Common", 1.0),
            _device("M/A2", "Common", 2.0),
            _device("M/A3", "Common", 3.0),
            _device("M/B1", "Rare", 4.0),
            _device("M/C1", "Common", 4.0),
        ]
    )
    device = graph_probe_device(document, model="M")
    assert device.device_id == "M/A1"


def test_graph_probe_device_refuses_a_document_without_one() -> None:
    lone = _device("M/A1", "Common", 1.0)
    document = _document([lone])
    document["channels"] = document["channels"][:2]
    with pytest.raises(AssertionError, match="no device of model 'M'"):
        graph_probe_device(document, model="M")

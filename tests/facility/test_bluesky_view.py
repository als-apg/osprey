"""The Bluesky devices view: the facility's channels as the worker's device file.

``data/bluesky_devices.yml`` opens with ``schema: osprey.facility.bluesky_devices/1``
and holds one settable per setpoint (its readback the pair when the pair is
another channel) and one readable per readback; a channel whose role is
``none`` is no device, and a device's name is its address. A render carries
the view when it runs a Bluesky lane.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from osprey.facility.views.bluesky import (
    BLUESKY_DEVICES_FILE,
    BLUESKY_DEVICES_SCHEMA,
    bluesky_configured,
    bluesky_document,
    write_bluesky_view,
)
from osprey.services.bluesky_bridge.devices._specs_from_file import (
    specs_from_file,
    validate_device_document,
)
from tests.facility.test_cf_view_parity import fingerprint_rows, load_golden

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject


def _channel(id_: str, role: str, pair: str | None = None) -> dict[str, Any]:
    channel: dict[str, Any] = {"id": id_, "role": role}
    if pair is not None:
        channel["pair"] = pair
    return channel


def _inputs(channels: list[dict[str, Any]], services: dict[str, Any] | None = None) -> Any:
    from osprey.facility.views import ViewInputs

    return ViewInputs(
        doc={"channels": channels},  # type: ignore[arg-type]
        rendered_config={"services": services or {}},
        facility_dir=Path("."),
        served=[],
    )


# --- the document ------------------------------------------------------------------


def test_a_setpoint_is_a_settable_reading_its_pair_back() -> None:
    document = bluesky_document(
        {"channels": [_channel("A:SP", "setpoint", "A:RB"), _channel("A:RB", "readback")]}
    )

    assert document == {
        "settables": [{"name": "A:SP", "setpoint": "A:SP", "readback": "A:RB"}],
        "readables": [{"name": "A:RB", "pv": "A:RB"}],
    }


def test_a_setpoint_paired_with_itself_names_no_readback() -> None:
    document = bluesky_document({"channels": [_channel("A:SP", "setpoint", "A:SP")]})

    assert document["settables"] == [{"name": "A:SP", "setpoint": "A:SP"}]


def test_an_unpaired_readback_is_a_readable() -> None:
    document = bluesky_document({"channels": [_channel("T:01", "readback")]})

    assert document == {"settables": [], "readables": [{"name": "T:01", "pv": "T:01"}]}


def test_a_channel_with_role_none_is_no_device() -> None:
    document = bluesky_document({"channels": [_channel("S:01", "none")]})

    assert document == {"settables": [], "readables": []}


def test_no_channel_is_an_empty_document() -> None:
    assert bluesky_document({"channels": []}) == {"settables": [], "readables": []}


# --- the file ----------------------------------------------------------------------


def test_the_file_opens_with_its_schema_and_loads_in_full(tmp_path: Path) -> None:
    inputs = _inputs([_channel("A:SP", "setpoint", "A:RB"), _channel("A:RB", "readback")])

    written = write_bluesky_view(tmp_path, inputs)

    target = tmp_path / BLUESKY_DEVICES_FILE
    assert written == [target]
    text = target.read_text(encoding="utf-8")
    assert text.split("\n", 1)[0] == f"schema: {BLUESKY_DEVICES_SCHEMA}"
    assert "data/facility/" in text
    assert "osprey build" in text
    loaded = yaml.safe_load(text)
    assert validate_device_document(loaded) == []
    settables, readables = specs_from_file(target)
    assert [(s.name, s.setpoint_pv, s.readback_pv) for s in settables] == [("A:SP", "A:SP", "A:RB")]
    assert [(r.name, r.read_pv) for r in readables] == [("A:RB", "A:RB")]


def test_the_file_is_world_readable(tmp_path: Path) -> None:
    write_bluesky_view(tmp_path, _inputs([_channel("T:01", "readback")]))

    assert (tmp_path / BLUESKY_DEVICES_FILE).stat().st_mode & 0o044 == 0o044


def test_the_file_is_the_same_bytes_on_every_write(tmp_path: Path) -> None:
    inputs = _inputs([_channel("A:SP", "setpoint", "A:RB"), _channel("A:RB", "readback")])
    write_bluesky_view(tmp_path / "one", inputs)
    write_bluesky_view(tmp_path / "two", inputs)

    assert (tmp_path / "one" / BLUESKY_DEVICES_FILE).read_bytes() == (
        tmp_path / "two" / BLUESKY_DEVICES_FILE
    ).read_bytes()


# --- when a render carries it --------------------------------------------------------


@pytest.mark.parametrize("lane", ["bluesky", "bluesky_va", "bluesky_live", "bluesky_standin"])
def test_a_render_running_a_bluesky_lane_carries_the_view(lane: str) -> None:
    assert bluesky_configured(_inputs([], {lane: {"path": "./services/bluesky"}})) is True


@pytest.mark.parametrize("services", [{}, {"openobserve": {"path": "x"}}, {"bluesky": None}])
def test_a_render_without_a_bluesky_lane_does_not(services: dict[str, Any]) -> None:
    assert bluesky_configured(_inputs([], services)) is False


def test_the_view_is_registered() -> None:
    from osprey.facility.views import VIEWS

    (view,) = [view for view in VIEWS if view.name == "bluesky"]
    assert (view.path, view.reason) == (".", "services.bluesky")


def test_a_build_names_the_omitted_view_once_over_all_its_renders(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osprey.facility import views
    from osprey.facility.render import render_facility_outputs

    (view,) = [view for view in views.VIEWS if view.name == "bluesky"]
    monkeypatch.setattr(views, "VIEWS", (view,))
    reported: set[str] = set()
    for name in ("deployment", "compose"):
        render_dir = tmp_path / name
        render_dir.mkdir()
        render_facility_outputs(
            render_dir,
            {"channels": []},
            {"services": {}},
            tmp_path / "facility",
            omitted_reported=reported,
        )

    err = capsys.readouterr().err
    assert err.count("view bluesky not written: services.bluesky") == 1


# --- the demo ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_demo_view_holds_every_setpoint_and_readback(
    built_control_assistant: BuiltProject,
) -> None:
    rows = fingerprint_rows() + load_golden("demo_fingerprint_additions.json")["rows"]
    setpoints = sorted(row["address"] for row in rows if row["role"] == "setpoint")
    readbacks = sorted(row["address"] for row in rows if row["role"] == "readback")
    assert len(setpoints) == 396

    settables, readables = specs_from_file(
        built_control_assistant.build_dir / "data" / BLUESKY_DEVICES_FILE
    )

    assert sorted(spec.name for spec in settables) == setpoints
    assert sorted(spec.name for spec in readables) == readbacks
    assert all(spec.name == spec.setpoint_pv for spec in settables)
    assert all(spec.name == spec.read_pv for spec in readables)

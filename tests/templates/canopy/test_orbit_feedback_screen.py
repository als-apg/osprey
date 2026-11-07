"""The demo template's storage-ring orbit-feedback screen.

``data/canopy/library/screens/sr/orbit-feedback.screen.json`` ships with the app
template and is copied into a deployment's screen library. It shows the stored
current, the orbit RMS per plane, the BPM orbit and corrector readbacks, and the
one corrector setpoint. Its PV set is what the viewer subscribes when the screen
opens, read with CANOPY's own walk (``canopy_mcp.screens.expanded_pvs``), so a
change that adds, drops or misspells a channel shows up here as a count or name
failure.

Every colour the screen paints is a role: an OSPREY custom role with light and
dark fallbacks for neutral colours, a scheme role only where it carries alarm
meaning, and never a bare hex value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from canopy_mcp import screens
from canopy_mcp.check.lint import served_roles
from canopy_mcp.check.schema import SPEC_UNAVAILABLE, check_schema, schema_path
from canopy_mcp.check.semantic import check_semantic
from canopy_mcp.store import DraftStore

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = ROOT / "src" / "osprey" / "templates" / "apps" / "demo" / "data" / "canopy" / "library"
EXAMPLE_CHANNELS = (
    ROOT / "src" / "osprey" / "templates" / "facilities" / "example" / "records" / "channels.yaml"
)
SCREEN = LIBRARY / "screens" / "sr" / "orbit-feedback.screen.json"
DRAFT_ID = "sr/orbit-feedback"

# 144 BPM positions + 72 HCM + 72 VCM readbacks + 2 DCCT + 1 corrector setpoint.
EXPECTED_PV_COUNT = 291

# The storage-ring overview this screen was cut from weighed 88 KB.
SIZE_BUDGET = 88_000

# The colour props each IR widget type on this screen reads: the pinned runtime
# registry's ``color`` props of the widget the type maps to.
COLOUR_PROPS = {
    "absolute-canvas": ("backgroundColor",),
    "text-label": ("foregroundColor", "backgroundColor"),
    "pv-label": ("foregroundColor", "backgroundColor"),
    "pv-text-input": ("foregroundColor", "backgroundColor"),
    "pv-led": ("onColor", "offColor"),
    "group-box": (),
    "pv-array-plot": (),
}

# Scheme roles allowed only where they carry alarm meaning; this screen has none.
ALARM_ROLES: frozenset[str] = frozenset()

CURVE_ROLES = {"X": "Trace_X", "HCM": "Trace_X", "Y": "Trace_Y", "VCM": "Trace_Y"}

MACRO = re.compile(r"\$\{|\$\(")
HEX = re.compile(r"^#[0-9a-fA-F]{3,8}$")


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(SCREEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pvs(doc: dict) -> list[str]:
    return screens.expanded_pvs(doc, DraftStore(LIBRARY), DRAFT_ID)


@pytest.fixture(scope="module")
def channels() -> dict:
    """The example facility's channel records, by name."""
    records = yaml.safe_load(EXAMPLE_CHANNELS.read_text(encoding="utf-8"))
    return {record["id"]: record for record in records}


@pytest.fixture(scope="module")
def scheme_roles() -> set[str]:
    roles = served_roles()
    assert roles, "the pinned CANOPY serves no colour scheme"
    return set(roles)


def _nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _nodes(child)


def _is_role(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("$role"), str)
        and isinstance(value.get("fallback"), str)
        and isinstance(value.get("fallbackDark"), str)
    )


def _roles(value: object):
    """Every ColorRole anywhere under ``value``."""
    if isinstance(value, dict):
        if "$role" in value:
            yield value
            return
        for item in value.values():
            yield from _roles(item)
    elif isinstance(value, list):
        for item in value:
            yield from _roles(item)


def _bare_hex(value: object, path: str = ""):
    """Every hex colour string that is not the fallback of a ColorRole."""
    if isinstance(value, dict):
        if "$role" in value:
            return
        for key, item in value.items():
            yield from _bare_hex(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _bare_hex(item, f"{path}[{index}]")
    elif isinstance(value, str) and HEX.match(value):
        yield path


@pytest.fixture
def spec_dir(tmp_path: Path) -> Path:
    """A spec staging directory holding the IR 1.1 schema the converter validates against.

    CANOPY ships no schema of its own; it reads one from a staged spec. The
    staged copy is pydmconverter's packaged ``screen-ir.schema.json``, put where
    CANOPY's staging layout expects it.
    """
    from importlib.resources import files

    source = files("pydmconverter.ir").joinpath("data", "screen-ir.schema.json")
    target = schema_path(tmp_path / "spec", "1.1")
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    return tmp_path / "spec"


def test_the_screen_validates_against_the_screen_ir_schema(doc: dict, spec_dir: Path) -> None:
    assert doc["schemaVersion"] == "1.1"
    assert check_schema(doc, spec_dir) == []
    assert doc["id"] == DRAFT_ID
    assert doc["metadata"]["title"] == "Orbit feedback"


def test_without_a_staged_spec_the_schema_check_says_so(doc: dict, tmp_path: Path) -> None:
    for spec in (None, tmp_path / "absent"):
        findings = check_schema(doc, spec)
        assert [(f.code, f.severity) for f in findings] == [(SPEC_UNAVAILABLE, "warning")]


def test_the_semantic_check_finds_only_the_inert_setpoint(doc: dict) -> None:
    findings = check_semantic(doc)
    assert [f for f in findings if f.severity == "error"] == []
    assert {f.code for f in findings} <= {"write_not_relayed"}


def test_the_screen_binds_the_stated_pv_count(pvs: list[str]) -> None:
    assert len(pvs) == EXPECTED_PV_COUNT


def test_every_bound_name_is_a_records_channel(pvs: list[str], channels: dict) -> None:
    assert [name for name in pvs if name not in channels] == []


def test_no_bound_name_carries_an_unresolved_macro(doc: dict, pvs: list[str]) -> None:
    assert [name for name in pvs if MACRO.search(name)] == []
    nodes = screens.expand(doc, DraftStore(LIBRARY), DRAFT_ID)
    assert [n["nodeId"] for n in nodes if n["unresolvedMacro"]] == []


def test_the_rf_box_and_the_magnet_matrix_are_gone(pvs: list[str]) -> None:
    assert [name for name in pvs if ":RF:" in name] == []
    families = {name.split(":")[2] for name in pvs if name.startswith("SR:MAG:")}
    assert families == {"HCM", "VCM"}


def test_orbit_plots_cover_every_bpm_in_micrometres(doc: dict, channels: dict) -> None:
    plots = [n for n in _nodes(doc["root"]) if n["type"] == "pv-array-plot"]
    by_curve = {n["props"]["curves"][0]["name"]: n["props"] for n in plots}
    assert sorted(by_curve) == ["HCM", "VCM", "X", "Y"]
    for plane in ("X", "Y"):
        props = by_curve[plane]
        expected = sorted(
            n for n in channels if n.startswith("SR:DIAG:BPM:") and n.endswith(f":POSITION:{plane}")
        )
        assert len(expected) == 72
        assert props["curves"][0]["y_channels"] == expected
        assert props["mode"] == "line"
        assert props["scale"] == 1e6
        assert props["yAxes"][0]["label"] == "µm"
    for family in ("HCM", "VCM"):
        props = by_curve[family]
        expected = sorted(
            n for n in channels if n.startswith(f"SR:MAG:{family}:") and n.endswith(":CURRENT:RB")
        )
        assert len(expected) == 72
        assert props["mode"] == "bar"
        assert props["curves"][0]["y_channels"] == expected


def test_the_rms_pair_per_plane_is_shown_and_not_counted(doc: dict, pvs: list[str]) -> None:
    bound = [
        n["props"]["pv"]
        for n in _nodes(doc["root"])
        if isinstance(n["props"].get("pv"), str) and n["props"]["pv"].startswith("formula://")
    ]
    assert sorted(bound) == ["formula://orbit_rms_x", "formula://orbit_rms_y"]
    assert [f["name"] for f in doc["formulas"]] == ["orbit_rms_x", "orbit_rms_y"]
    for formula, plane in zip(doc["formulas"], ("x", "y"), strict=True):
        assert formula["kind"] == "scalar"
        bindings = formula["bindings"]
        assert sorted(bindings) == [f"{plane}{i:02d}" for i in range(1, 73)]
        assert all(name.endswith(f":POSITION:{plane.upper()}") for name in bindings.values())
    assert [name for name in pvs if name.startswith("formula://")] == []


def test_one_setpoint_input_on_hcm_01(doc: dict) -> None:
    inputs = [n for n in _nodes(doc["root"]) if n["type"] == "pv-text-input"]
    assert [n["props"]["pv"] for n in inputs] == ["SR:MAG:HCM:01:CURRENT:SP"]


def test_readouts_show_units_and_alarm_state(doc: dict) -> None:
    labels = [
        n
        for n in _nodes(doc["root"])
        if n["type"] == "pv-label" and not n["props"]["pv"].startswith("formula://")
    ]
    assert [n["props"]["pv"] for n in labels] == ["SR:DIAG:DCCT:01:CURRENT:RB"]
    for node in labels:
        assert node["props"]["showUnits"] is True, node["id"]
        assert node["props"]["alarmSensitive"] is True, node["id"]
        assert node["props"]["alarmBorder"] is True, node["id"]


def test_the_screen_is_about_one_view_tall(doc: dict) -> None:
    root = doc["root"]
    width, height = root["geometry"]["width"], root["geometry"]["height"]
    assert height <= 1100
    assert doc["metadata"]["size"] == {"width": width, "height": height}
    assert (root["props"]["width"], root["props"]["height"]) == (width, height)
    for node in _nodes(root):
        if node is root:
            continue
        g = node["geometry"]
        assert g["x"] >= 0 and g["y"] >= 0, node["id"]
        assert g["x"] + g["width"] <= width, node["id"]
        assert g["y"] + g["height"] <= height, node["id"]


def test_every_colour_prop_is_present_as_a_role(doc: dict) -> None:
    for node in _nodes(doc["root"]):
        assert node["type"] in COLOUR_PROPS, node["type"]
        for prop in COLOUR_PROPS[node["type"]]:
            assert _is_role(node["props"].get(prop)), (node["id"], prop)
        if node["type"] == "pv-array-plot":
            for curve in node["props"]["curves"]:
                assert _is_role(curve.get("color")), (node["id"], curve["name"])


def test_curves_carry_the_trace_role_of_their_plane(doc: dict) -> None:
    plots = [n for n in _nodes(doc["root"]) if n["type"] == "pv-array-plot"]
    assert {
        curve["name"]: curve["color"]["$role"] for n in plots for curve in n["props"]["curves"]
    } == CURVE_ROLES


def test_no_scheme_role_outside_alarm_uses(doc: dict, scheme_roles: set[str]) -> None:
    used = {role["$role"] for role in _roles(doc)}
    assert used
    assert used & scheme_roles <= ALARM_ROLES


def test_every_role_has_a_light_and_a_dark_fallback(doc: dict) -> None:
    roles = list(_roles(doc))
    assert roles
    for role in roles:
        assert _is_role(role), role
        assert HEX.match(role["fallback"]) and HEX.match(role["fallbackDark"]), role
    by_name: dict[str, set[tuple[str, str]]] = {}
    for role in roles:
        by_name.setdefault(role["$role"], set()).add((role["fallback"], role["fallbackDark"]))
    assert {name: pairs for name, pairs in by_name.items() if len(pairs) > 1} == {}


def test_no_literal_hex_outside_a_role(doc: dict) -> None:
    assert list(_bare_hex(doc)) == []


def test_the_screen_is_smaller_than_the_overview_it_was_cut_from() -> None:
    assert SCREEN.stat().st_size < SIZE_BUDGET


def test_the_file_is_serialised_canonically(doc: dict) -> None:
    text = SCREEN.read_text(encoding="utf-8")
    assert text == json.dumps(doc, indent=2, ensure_ascii=False) + "\n"

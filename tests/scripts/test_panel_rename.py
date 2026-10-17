"""The panel rename-map script: structural edits, then binding sites.

``scripts/facility_demo/panel_rename.py`` edits the GTLView/BTSView CANOPY
panels (``_panel_edits``) and rewrites them from legacy names to the demo
grammar, driven by the element table, and checks that a built
``facility.json`` holds every element-table name. These tests run it on a
placeholder library and table under ``fixtures/panels``, shaped like the real
panels (same files, same widget ids at every edit site) but with ``XX:``
placeholder names.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "facility_demo" / "panel_rename.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "panels"
TABLE = FIXTURES / "elements.tsv"
FACILITY = FIXTURES / "facility.json"

_SPEC = importlib.util.spec_from_file_location("panel_rename", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
panel_rename = importlib.util.module_from_spec(_SPEC)
# import-time required because the rows below feed parametrized cases, and the module
# under test is a script loaded by path: its dataclasses resolve their annotations
# through sys.modules, so it is registered before it executes.
sys.modules.setdefault("panel_rename", panel_rename)
_SPEC.loader.exec_module(panel_rename)

import _panel_edits  # noqa: E402  (on sys.path once the script is loaded)
from _elements import load_elements  # noqa: E402

GTL = "gtb/GTLView.screen.json"
BTS = "bts/BTSView.screen.json"
PANELS = (GTL, BTS)
EMBEDDED = tuple(
    f"{section}/embedded/{name}.screen.json"
    for section in ("gtb", "bts")
    for name in ("_tv", "_KnobDevice", "_bpm")
)
#: Legacy name patterns no macro value may keep.
LEGACY_PATTERNS = ("irm:", "EG___", "LTB___", "BTS___", "ALS:")

ROWS = load_elements(TABLE)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A writable copy of the fixture library."""
    target = tmp_path / "library"
    shutil.copytree(FIXTURES / "library", target)
    return target


def _load(library: Path, relative: str) -> dict[str, Any]:
    return json.loads((library / "screens" / relative).read_text(encoding="utf-8"))


def _nodes(document: dict[str, Any]) -> list[dict[str, Any]]:
    found, stack = [], [document["root"]]
    while stack:
        node = stack.pop()
        found.append(node)
        stack.extend(node.get("children", []))
    return found


def _node(document: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in _nodes(document):
        if node["id"] == node_id:
            return node
    raise KeyError(node_id)


def _screens(library: Path) -> dict[str, dict[str, Any]]:
    """Every panel screen (both panels and their sub-screens), by path under screens/."""
    return {rel: _load(library, rel) for rel in PANELS + EMBEDDED}


def _snapshot(library: Path) -> dict[str, bytes]:
    return {str(p.relative_to(library)): p.read_bytes() for p in sorted(library.rglob("*.json"))}


def _edit(library: Path, relative: str, node_id: str, **props: Any) -> None:
    path = library / "screens" / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    _node(document, node_id)["props"].update(props)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# --- apply: binding sites ----------------------------------------------------------


def test_apply_rewrites_every_binding_site(library: Path) -> None:
    changed = panel_rename.apply(library, ROWS)

    assert {p.relative_to(library).as_posix() for p in changed} == {
        "screens/gtb/GTLView.screen.json",
        "screens/gtb/embedded/_tv.screen.json",
        "screens/gtb/embedded/_KnobDevice.screen.json",
        "screens/bts/BTSView.screen.json",
        "screens/bts/embedded/_tv.screen.json",
        "screens/bts/embedded/_KnobDevice.screen.json",
        "screens/common/KnobPanel.screen.json",
    }
    main = _load(library, GTL)
    assert main["formulas"][0]["bindings"] == {"0": "BTS:DIAG:PADDLES:01:COUNT:INSERTED"}
    assert _node(main, "w-005")["props"]["macros"] == {
        "CNTL_PV": "LTB:MAG:QUAD:01:CURRENT:SP",
        "COLOR_CODE": "3",
        "COLOR_PV": "LTB:MAG:QUAD:01:STATUS:ON",
        "DESC": "XX:Q1:Set",
        "MON_PV": "LTB:MAG:QUAD:01:CURRENT:RB",
    }
    bpm = _node(main, "w-007")["props"]["macros"]
    assert bpm == {
        "BPM": "XX:BPM1",
        "BPM_X": "LTB:DIAG:BPM:01:POSITION:X",
        "BPM_Y": "LTB:DIAG:BPM:01:POSITION:Y",
    }
    tv = _node(main, "w-008")["props"]["macros"]
    assert tv["paddle"] == "LTB:DIAG:TV:01:PADDLE:SP"
    assert tv["paddle_rb"] == "LTB:DIAG:TV:01:PADDLE:RB"
    assert tv["light"] == "LTB:DIAG:TV:01:LAMP:RB"
    label = _node(main, "w-010")
    assert label["props"]["pv"] == "LTB:MAG:QUAD:01:CURRENT:RB"
    assert label["rules"][0]["pvs"][0]["name"] == "GTL:GUN:EGUN:01:HV_ON:RB"
    assert _node(main, "w-011")["props"]["pv"] == "GTL:GUN:EGUN:01:HV_ON:RB"
    actions = _node(main, "w-012")["props"]["actions"]
    assert [a.get("pvName") for a in actions] == [
        "GTL:GUN:EGUN:01:HV_ON:SP",
        "LTB:MAG:QUAD:02:CURRENT:SP",
    ]
    curve = _node(main, "w-013")["props"]["curves"][0]
    assert (curve["x_channel"], curve["y_channel"]) == (
        "LTB:DIAG:BPM:01:POSITION:X",
        "LTB:DIAG:BPM:01:POSITION:Y",
    )

    bts = _load(library, BTS)
    assert _node(bts, "w-002")["props"]["macros"]["CNTL_PV"] == "BTS:MAG:QUAD:01:CURRENT:SP"
    assert _node(bts, "w-003")["props"]["macros"]["paddle"] == "BTS:DIAG:TV:01:PADDLE:SP"
    assert _node(bts, "w-003")["props"]["macros"]["paddle_rb"] == "BTS:DIAG:TV:01:PADDLE:RB"


def test_a_legacy_name_with_sp_and_rb_rows_resolves_by_site(library: Path) -> None:
    panel_rename.apply(library, ROWS)
    main = _load(library, GTL)

    knob = _node(main, "w-409")["props"]["macros"]
    assert knob["CNTL_PV"] == "LTB:MAG:QUAD:02:CURRENT:SP"
    assert knob["MON_PV"] == "LTB:MAG:QUAD:02:CURRENT:RB"
    toggle = _node(main, "w-009")["props"]
    assert toggle["pv"] == "LTB:DIAG:TV:01:PADDLE:SP"
    assert toggle["readbackPV"] == "LTB:DIAG:TV:01:PADDLE:RB"
    assert _node(main, "w-012")["props"]["actions"][1]["pvName"] == "LTB:MAG:QUAD:02:CURRENT:SP"


def test_free_text_local_channels_and_macro_references_are_untouched(library: Path) -> None:
    before = _load(library, GTL)
    panel_rename.apply(library, ROWS)
    main = _load(library, GTL)

    assert _node(main, "w-002")["props"]["pv"] == "loc://tv1?type=float&init=0"
    assert _node(main, "w-003")["props"]["pv"] == "fox://f-001"
    assert _node(main, "w-004")["props"]["text"] == "XX:Q1:Set"
    assert _node(main, "w-008")["props"]["macros"]["TV"] == "XX:TV1:Paddle"
    assert _node(main, "w-008")["props"]["macros"]["loctv"] == "loc://tv1<VDouble>(0)"
    assert _node(main, "w-011")["meta"] == {"comment": "reads XX:Gun:HvOn"}
    assert _node(main, "w-012")["props"]["label"] == "XX:Gun:HvCmd ON/OFF"
    assert _node(main, "w-013")["props"]["curves"][0]["name"] == "XX:BPM1:X"
    assert _node(main, "w-014")["props"]["pv"] == "${S}Q1:Set"
    assert main["macros"] == [m for m in before["macros"] if m["name"] not in ("P", "R")]
    for section in ("gtb", "bts"):
        tv = _load(library, f"{section}/embedded/_tv.screen.json")
        loc = _node(tv, "w-004")
        assert loc["meta"]["comment"].endswith("$(paddle_in), $(light_on)]")
        assert loc["props"] == {"pv": "${loctv}", "readbackPV": "${loctv}", "buttonType": "toggle"}


def test_only_binding_values_change_after_the_structural_edits(library: Path) -> None:
    """Masking the binding sites, the re-map changes nothing the edits left."""
    edited = {}
    for rel, document in _screens(library).items():
        edits = _panel_edits.Edits(panel_rename.RenameMap(ROWS).paddle_readback)
        edits.edit(rel, document)
        assert edits.errors == []
        edited[rel] = document
    panel_rename.apply(library, ROWS)
    after = _screens(library)

    def masked(rel: str, document: dict[str, Any]) -> dict[str, Any]:
        for site in list(panel_rename.binding_sites(Path(rel), document)):
            site.holder[site.key] = "<binding>"
        return document

    assert edited.keys() == after.keys()
    for rel in edited:
        assert masked(rel, edited[rel]) == masked(rel, after[rel]), rel


def test_screens_outside_the_panel_directories_are_not_read(library: Path) -> None:
    other = library / "screens" / "common" / "Other.screen.json"
    before = other.read_bytes()

    panel_rename.apply(library, ROWS)

    assert other.read_bytes() == before


# --- apply: structural edits -------------------------------------------------------


@pytest.mark.parametrize("section", ["gtb", "bts"])
def test_both_tv_copies_split_the_paddle_and_light_the_lamp(library: Path, section: str) -> None:
    panel_rename.apply(library, ROWS)

    tv = _load(library, f"{section}/embedded/_tv.screen.json")
    assert [m["name"] for m in tv["macros"]] == ["TV", "light", "loctv", "paddle", "paddle_rb"]
    paddle = _node(tv, "w-005")
    assert paddle["type"] == "pv-button"
    assert (paddle["props"]["pv"], paddle["props"]["readbackPV"]) == ("${paddle}", "${paddle_rb}")
    assert paddle["props"]["buttonType"] == "toggle"
    lamp = _node(tv, "w-006")
    assert lamp["type"] == "pv-led"
    assert lamp["props"]["pv"] == "${light}"
    assert "readbackPV" not in lamp["props"]
    assert lamp["geometry"] == {"x": 0, "y": 0, "width": 10, "height": 10}


def test_every_write_targets_a_setpoint_and_every_toggle_reads_its_readback(
    library: Path,
) -> None:
    panel_rename.apply(library, ROWS)
    channels = {c["id"] for c in json.loads(FACILITY.read_text(encoding="utf-8"))["channels"]}

    def is_facility(value: str) -> bool:
        return not value.startswith(("loc://", "fox://")) and "${" not in value

    writes: list[str] = []
    toggles: list[tuple[str, str]] = []
    for document in _screens(library).values():
        for node in _nodes(document):
            props = node.get("props", {})
            if node["type"] in panel_rename.WRITABLE_TYPES and is_facility(props.get("pv", "")):
                writes.append(props["pv"])
                if props.get("buttonType") == "toggle":
                    toggles.append((props["pv"], props["readbackPV"]))
            for action in props.get("actions", []):
                if action["type"] == "write_pv":
                    assert is_facility(action["pvName"]), action
                    writes.append(action["pvName"])
            macros = props.get("macros", {})
            if "CNTL_PV" in macros:
                writes.append(macros["CNTL_PV"])
                assert macros["MON_PV"] == macros["CNTL_PV"].removesuffix(":SP") + ":RB"
            if "paddle" in macros:
                toggles.append((macros["paddle"], macros["paddle_rb"]))
                assert macros["light"].endswith(":LAMP:RB")

    assert writes and toggles
    for target in writes + [sp for sp, _ in toggles]:
        assert target.endswith(":SP") and target in channels, target
    for sp, rb in toggles:
        assert rb == sp.removesuffix(":SP") + ":RB" and rb in channels, (sp, rb)


def test_no_dead_or_legacy_macro_remains(library: Path) -> None:
    def macro_values(screens: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
        values = []
        for document in screens.values():
            values += [(m["name"], m["default"]) for m in document["macros"]]
            for node in _nodes(document):
                values += list(node.get("props", {}).get("macros", {}).items())
        return values

    before = macro_values(_screens(library))
    assert any(value.startswith(LEGACY_PATTERNS) for _, value in before)
    assert {"light_on", "paddle_in", "P", "R", "GAIN", "MIN", "MAX"} <= {n for n, _ in before}

    panel_rename.apply(library, ROWS)
    after = macro_values(_screens(library))

    assert not {"light_on", "paddle_in", "P", "R", "GAIN", "MIN", "MAX"} & {n for n, _ in after}
    for name, value in after:
        assert not any(pattern in value for pattern in LEGACY_PATTERNS), (name, value)


def test_cut_widgets_and_their_references_are_gone(library: Path) -> None:
    panel_rename.apply(library, ROWS)

    for rel, cuts in _panel_edits.CUT_NODES.items():
        ids = {node["id"] for node in _nodes(_load(library, rel))}
        assert not ids & cuts, rel
    for rel, document in _screens(library).items():
        text = json.dumps({k: v for k, v in document.items() if k != "root"})
        text += json.dumps([n.get("props") for n in _nodes(document)])
        text += json.dumps([n.get("rules") for n in _nodes(document)])
        for name in ("P", "R", "GAIN", "MIN", "MAX", "light_on", "paddle_in"):
            assert "${" + name + "}" not in text and "$(" + name + ")" not in text, (rel, name)
    assert not (library / "screens" / "common" / "KnobPanel.screen.json").exists()


@pytest.mark.parametrize(
    ("panel", "target", "label", "geometry"),
    [
        # GTLView's KnobPanel button sat far from the TV banks: the link takes the
        # free slot under the BTS TV PADDLES bank (D5), at these coordinates.
        (GTL, "../bts/BTSView.screen.json", "Open BTSView", (1960, 490, 130, 24)),
        # BTSView's link takes the slot of its cut "Open KnobPanel" button (w-195).
        (BTS, "../gtb/GTLView.screen.json", "Open GTLView", (910, 830, 120, 55)),
    ],
)
def test_each_panel_has_one_document_relative_cross_link(
    library: Path, panel: str, target: str, label: str, geometry: tuple[int, ...]
) -> None:
    panel_rename.apply(library, ROWS)
    document = _load(library, panel)

    opens = [
        (node, action)
        for node in _nodes(document)
        for action in node.get("props", {}).get("actions", [])
        if action["type"] == "open_display"
    ]
    assert len(opens) == 1
    node, action = opens[0]
    assert action == {"type": "open_display", "file": target, "target": "window"}
    assert node["type"] == "regular-button" and node["props"]["label"] == label
    g = node["geometry"]
    assert (g["x"], g["y"], g["width"], g["height"]) == geometry
    assert (library / "screens" / panel).parent.joinpath(target).resolve().is_file()
    assert node["id"] not in _panel_edits.CUT_NODES[panel]


def test_the_four_source_bugs_are_corrected(library: Path) -> None:
    panel_rename.apply(library, ROWS)
    gtl, bts = _load(library, GTL), _load(library, BTS)

    # The LTB Q5 glyph read Q4.2's supply status; it reads Q5's own.
    assert _node(gtl, "w-300")["rules"][0]["pvs"][0]["name"] == "LTB:MAG:QUAD:05:STATUS:ON"
    assert _node(gtl, "w-297")["rules"][0]["pvs"][0]["name"] == "LTB:MAG:QUAD:02:STATUS:ON"
    # The LTB Q4.1 knob read its setpoint as its readback.
    q41 = _node(gtl, "w-408")["props"]["macros"]
    assert (q41["CNTL_PV"], q41["MON_PV"]) == (
        "LTB:MAG:QUAD:04:CURRENT:SP",
        "LTB:MAG:QUAD:04:CURRENT:RB",
    )
    # The BTS BPM5 plot was titled BPM1.
    assert _node(bts, "w-157")["props"]["macros"]["BPM"] == "BTS BPM5"
    # The BTS Q6.1 knob wrote its readback.
    q61 = _node(bts, "w-184")["props"]["macros"]
    assert (q61["CNTL_PV"], q61["MON_PV"]) == (
        "BTS:MAG:QUAD:06:CURRENT:SP",
        "BTS:MAG:QUAD:06:CURRENT:RB",
    )
    # Both outer Q1.1 glyphs show Q1's status: correct in the source, left alone.
    for node_id in ("w-276", "w-279"):
        assert _node(gtl, node_id)["rules"][0]["pvs"][0]["name"] == "LTB:MAG:QUAD:01:STATUS:ON"


def test_the_widget_and_action_types_lose_only_execute(library: Path) -> None:
    def types(screens: dict[str, dict[str, Any]]) -> set[str]:
        found = set()
        for document in screens.values():
            for node in _nodes(document):
                found.add(node["type"])
                found |= {f"action:{a['type']}" for a in node.get("props", {}).get("actions", [])}
        return found

    before = types(_screens(library))
    panel_rename.apply(library, ROWS)

    assert "action:execute" in before
    assert types(_screens(library)) == before - {"action:execute"}


def test_paddle_readback_pairs_a_demo_paddle_and_keeps_a_legacy_one() -> None:
    mapping = panel_rename.RenameMap(ROWS)
    assert mapping.paddle_readback("XX:TV1:Paddle") == "XX:TV1:Paddle"
    assert mapping.paddle_readback("LTB:DIAG:TV:01:PADDLE:SP") == "LTB:DIAG:TV:01:PADDLE:RB"


def test_a_panel_already_re_mapped_without_the_split_still_gets_it(library: Path) -> None:
    for rel in PANELS:
        document = _load(library, rel)
        for site in list(panel_rename.binding_sites(Path(rel), document)):
            if panel_rename.RenameMap(ROWS).needs_rename(site.value):
                site.holder[site.key] = panel_rename.RenameMap(ROWS).resolve(
                    site.value, site.selector
                )
        (library / "screens" / rel).write_text(json.dumps(document, indent=2) + "\n")

    panel_rename.apply(library, ROWS)

    tv = _node(_load(library, GTL), "w-008")["props"]["macros"]
    assert tv["paddle_rb"] == "LTB:DIAG:TV:01:PADDLE:RB"


# --- apply: failures and idempotence -----------------------------------------------


def test_apply_is_idempotent(library: Path) -> None:
    panel_rename.apply(library, ROWS)
    once = _snapshot(library)

    assert panel_rename.apply(library, ROWS) == []
    assert _snapshot(library) == once


def test_an_unknown_legacy_name_fails_lists_every_site_and_writes_nothing(library: Path) -> None:
    _edit(library, GTL, "w-010", pv="XX:Nope:1")
    _edit(library, "bts/embedded/_tv.screen.json", "w-005", pv="XX:Nope:2")
    before = _snapshot(library)

    with pytest.raises(panel_rename.RenameError) as raised:
        panel_rename.apply(library, ROWS)

    message = str(raised.value)
    assert "'XX:Nope:1' is not in the element table" in message
    assert "'XX:Nope:2' is not in the element table" in message
    assert "screens/gtb/GTLView.screen.json: root.children[7].props.pv" in message
    assert _snapshot(library) == before


def test_a_panel_without_an_edit_site_fails_and_writes_nothing(library: Path) -> None:
    path = library / "screens" / BTS
    document = json.loads(path.read_text(encoding="utf-8"))
    document["root"]["children"] = [c for c in document["root"]["children"] if c["id"] != "w-195"]
    _node(document, "w-157")["props"]["macros"]["BPM"] = "BTS BPM9"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    before = _snapshot(library)

    with pytest.raises(panel_rename.RenameError) as raised:
        panel_rename.apply(library, ROWS)

    message = str(raised.value)
    assert "no w-195 to take the slot of 'Open GTLView'" in message
    assert "w-157 macros.BPM is 'BTS BPM9'" in message
    assert _snapshot(library) == before


def test_a_multi_row_name_at_a_site_outside_the_role_table_fails(library: Path) -> None:
    _edit(library, GTL, "w-010", pv="XX:Q2:Both")

    with pytest.raises(panel_rename.RenameError, match="not in the role table"):
        panel_rename.apply(library, ROWS)


def test_a_site_whose_role_the_name_lacks_fails(library: Path) -> None:
    path = library / "screens" / GTL
    document = json.loads(path.read_text(encoding="utf-8"))
    _node(document, "w-005")["props"]["macros"]["COLOR_PV"] = "XX:Q1:Mon"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(panel_rename.RenameError, match="no row for site status"):
        panel_rename.apply(library, ROWS)


@pytest.mark.parametrize(
    ("legacy", "selector", "expected"),
    [
        ("XX:TV1:Paddle", panel_rename.MACRO_SITES["paddle"], "LTB:DIAG:TV:01:PADDLE:SP"),
        ("XX:TV1:Paddle", panel_rename.MACRO_SITES["paddle_rb"], "LTB:DIAG:TV:01:PADDLE:RB"),
        ("XX:TV1:Paddle", panel_rename.READBACK, "LTB:DIAG:TV:01:PADDLE:RB"),
        ("XX:TV1:Paddle", panel_rename.SETPOINT, "LTB:DIAG:TV:01:PADDLE:SP"),
        ("XX:TV1:Light", panel_rename.MACRO_SITES["light"], "LTB:DIAG:TV:01:LAMP:RB"),
        ("XX:Q1:On", panel_rename.MACRO_SITES["COLOR_PV"], "LTB:MAG:QUAD:01:STATUS:ON"),
        ("XX:Q1:On", None, "LTB:MAG:QUAD:01:STATUS:ON"),
    ],
)
def test_resolve_picks_the_row_of_the_site(legacy: str, selector: Any, expected: str) -> None:
    assert panel_rename.RenameMap(ROWS).resolve(legacy, selector) == expected


def test_the_site_table_names_every_macro_the_task_binds() -> None:
    assert set(panel_rename.MACRO_SITES) == {
        "CNTL_PV",
        "MON_PV",
        "COLOR_PV",
        "BPM_X",
        "BPM_Y",
        "paddle",
        "paddle_rb",
        "light",
    }


# --- the CLI -----------------------------------------------------------------------


def test_cli_apply_exits_zero_then_non_zero_on_an_unknown_name(
    library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert panel_rename.main(["--table", str(TABLE), "apply", "--library", str(library)]) == 0
    out = capsys.readouterr().out
    assert "re-mapped 6 screen(s)" in out
    assert "deleted screens/common/KnobPanel.screen.json" in out

    _edit(library, GTL, "w-010", pv="XX:Nope:1")
    assert panel_rename.main(["--table", str(TABLE), "apply", "--library", str(library)]) == 1
    assert "XX:Nope:1" in capsys.readouterr().err


def test_check_exits_zero_when_the_facility_holds_every_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert panel_rename.main(["--table", str(TABLE), "check", "--facility", str(FACILITY)]) == 0
    assert "all 32 element-table names present" in capsys.readouterr().out


def test_check_lists_the_names_missing_from_the_facility(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = json.loads(FACILITY.read_text(encoding="utf-8"))
    removed = {"LTB:DIAG:BPM:01:SUM:RB", "BTS:DIAG:TV:01:LAMP:RB"}
    document["channels"] = [c for c in document["channels"] if c["id"] not in removed]
    facility = tmp_path / "facility.json"
    facility.write_text(json.dumps(document), encoding="utf-8")

    assert panel_rename.main(["--table", str(TABLE), "check", "--facility", str(facility)]) == 1
    err = capsys.readouterr().err
    assert "2 element-table name(s) missing" in err
    for name in removed:
        assert f"  {name}\n" in err


def test_check_reads_the_checked_in_element_table_by_default(tmp_path: Path) -> None:
    names = list(dict.fromkeys(row.demo_name for row in load_elements()))
    facility = tmp_path / "facility.json"
    facility.write_text(json.dumps({"channels": [{"id": n} for n in names]}), encoding="utf-8")
    assert panel_rename.main(["check", "--facility", str(facility)]) == 0

    facility.write_text(json.dumps({"channels": [{"id": n} for n in names[1:]]}), encoding="utf-8")
    assert panel_rename.main(["check", "--facility", str(facility)]) == 1

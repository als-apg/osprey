"""The device card's looks: the drawn frame, its chain, controls and grid; the calibration block."""

from __future__ import annotations

import dataclasses
import json

import pytest

from osprey.facility.panels import emit, layout
from osprey.facility.panels.describe import device_card
from osprey.facility.panels.view import Section, ValueRow
from tests.facility.panels._views import nodes, stub_render


def _by_type(doc: dict, kind: str) -> list[dict]:
    return [n for n in nodes(doc) if n["type"] == kind]


def _text(doc: dict, text: str) -> dict:
    (node,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == text]
    return node


def _columns(doc: dict) -> list[dict]:
    """The nodes of the two columns: those above the calibration block, if any."""
    found = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "CALIBRATION"]
    end = found[0]["geometry"]["y"] if found else float("inf")
    return [n for n in nodes(doc) if n["geometry"]["y"] < end]


def _fields(doc: dict) -> dict[str, dict]:
    return {n["props"]["pv"]: n for n in _columns(doc) if n["type"].startswith("pv-")}


ROLES_HEADER = (
    "class,role,signal,scope,shown_as,unit,scale,press_value,label,basis,source,revision,locator\n"
)


def _roles_graph(tmp_path, roles: str, labels: str = "", metrics: str = ""):
    """The fixture with a channel_roles.csv (and state words, card key numbers)."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(ROLES_HEADER + roles)
    if labels:
        (tables / "state_labels.csv").write_text(
            "channel,value,label,source,revision,locator\n" + labels
        )
    if metrics:
        with (tables / "metrics.csv").open("a") as handle:
            handle.write(metrics)
    return load_graph(write_fixture_facility(tmp_path / "f.json"), tables)


def _reads(doc: dict, node: dict) -> set[str]:
    """The channels a node shows: its own, its rules', their formulas' bindings."""
    found = {node["props"].get("pv")}
    formulas = {"formula://" + f["name"]: f for f in doc.get("formulas", ())}
    for rule in node.get("rules", ()):
        found |= {pv["name"] for pv in rule.get("pvs", ())}
        found |= set(formulas.get(rule.get("formula"), {}).get("bindings", {}).values())
    return found - {None}


def _heads(doc: dict) -> list[str]:
    """The numbered section heads, in page order."""
    texts = sorted(_by_type(doc, "text-label"), key=lambda n: n["geometry"]["y"])
    return [n["props"]["text"] for n in texts if " · " in n["props"]["text"][1:5]]


def test_a_drawn_card_has_its_title_line_numbered_sections_and_no_signal_table(panel_graph):
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    name = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "XR01:QF1"]
    assert any(n["props"]["fontSize"] == layout.CARD_NAME_SIZE for n in name)
    assert (
        _text(doc, "Quadrupole · default recipe")["props"]["foregroundColor"] == layout.MUTED_TEXT
    )
    assert _heads(doc) == [
        "1 · RELATIONS",
        "2 · CALIBRATION",
        "3 · OPERATE",
        "4 · ALL CHANNELS (2)",  # every channel, those drawn above too
    ]  # where and sources: one muted line each, no section
    texts = {n["props"]["text"] for n in _by_type(doc, "text-label")}
    assert not texts & {"SIGNAL", "SETPOINT", "READBACK", "SIGNALS", "SETTINGS"}
    bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in nodes(doc))
    assert doc["metadata"]["size"]["height"] == bottom + layout.CARD_END


def test_the_relations_are_blocks_with_arrows_the_device_in_focus(panel_graph) -> None:
    from osprey.facility.panels import symbols

    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    arrows = [n for n in _by_type(doc, "line") if n["props"].get("arrowEnd")]
    assert len(arrows) >= 2
    (focus,) = [
        n for n in _by_type(doc, "rectangle")
        if n["props"].get("lineColor") == symbols.FOCUS
    ]  # fmt: skip
    assert focus["props"]["fillColor"] == symbols.FOCUS_TINT
    feeder = [n for n in nodes(doc) if n["props"].get("tooltip") == "open XR01:PS1"]
    assert feeder and feeder[0]["props"]["actions"]
    # no network or controller is stated: a dashed block says so
    dashed = [n for n in _by_type(doc, "polygon") if n["props"].get("lineStyle") == "dashed"]
    assert len(dashed) >= 2
    assert _text(doc, "Network: not stated")


def _apply_buttons(doc: dict) -> list[dict]:
    return [n for n in _by_type(doc, "pv-button") if n["props"]["label"] == "Apply"]


def test_a_setpoint_is_typed_checked_and_applied(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    (field,) = _by_type(doc, "pv-text-input")
    local = field["props"]["pv"]
    assert local.startswith("loc://") and local.endswith('<VString>("")')  # nothing entered
    (apply,) = _apply_buttons(doc)
    assert apply["props"]["pv"] == "XR01-QF1:Current-SP"
    where, hidden, value = apply["rules"]
    assert value["targetProperty"] == "pressValue"
    assert (hidden["targetProperty"], hidden["default"], apply["props"]["invisible"]) == (
        "invisible",
        True,
        True,
    )  # out of the page, and so of the keyboard's reach, until a number is entered
    assert apply["props"]["tooltip"] == "writes the new value to XR01-QF1:Current-SP"
    (formula,) = [f for f in doc["formulas"] if "formula://" + f["name"] == value["formula"]]
    assert formula["bindings"] == {"v": local}
    # in place of the button until a number is entered
    hint = _text(doc, "type a value first")
    assert hint["geometry"]["x"] == where["conditions"][0]["value"]
    assert hint["geometry"]["x"] > field["geometry"]["x"]
    now = [n for n in _by_type(doc, "pv-label") if n["props"]["pv"] == "XR01-QF1:Current-SP"]
    assert now and now[0]["geometry"]["x"] < field["geometry"]["x"]  # the live value first
    for kind in ("ACTION", "CHOICE", "ENTRY", "READ ONLY"):  # the legend
        assert [n for n in _by_type(doc, "text-label") if n["props"]["text"] == kind]


def test_written_apply_shows_what_it_writes_from_the_same_local(panel_graph, tmp_path) -> None:
    """On the written card the line beside Apply that says what it will write and the
    button's press value come from one formula over the field's own local: when the
    field's text stops being a number, both keep the last number."""
    import json

    from osprey.facility.panels import layout
    from osprey.facility.panels.emit import write

    out = tmp_path / "out"
    write(panel_graph, out)
    (card,) = out.glob("card/*xr01-qf1.screen.json")
    doc = json.loads(card.read_text())
    (field,) = _by_type(doc, "pv-text-input")
    (apply,) = _apply_buttons(doc)
    (press,) = [r for r in apply["rules"] if r["targetProperty"] == "pressValue"]
    (shown,) = [
        n
        for n in _by_type(doc, "text-label")
        if any(r["targetProperty"] == "text" for r in n.get("rules", ()))
        and n["props"].get("tooltip") == f"what Apply writes to {apply['props']['pv']}"
    ]
    (rule,) = shown["rules"]
    assert rule["formula"] == press["formula"]  # one value: shown is written
    assert rule["default"] == press["default"] == shown["props"]["text"] == layout.NOTHING_ENTERED
    (formula,) = [f for f in doc["formulas"] if "formula://" + f["name"] == press["formula"]]
    assert formula["bindings"] == {"v": field["props"]["pv"]}
    assert _text(doc, "will write")["geometry"]["x"] < shown["geometry"]["x"]


def test_a_press_whose_value_no_source_states_is_left_out_and_counted(tmp_path) -> None:
    from osprey.facility.panels.describe import PressCount, press_counts

    roles = (  # "Bypass" is no state word of the channel: what it writes is not stated
        "PowerSupply,on-command,*:On-Cmd,device,action,,,,Bypass,stated,S,1,l\n"
        "PowerSupply,on,*:On-Sts,device,word,,,,,stated,S,1,l\n"
    )
    graph = _roles_graph(tmp_path, roles, "XR01-PS1:On-Cmd,1,On,S,1,l\n")
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert _by_type(doc, "pv-button") == []
    grid = [n for n in _by_type(doc, "rectangle") if n["props"].get("tooltip") == "XR01-PS1:On-Cmd"]
    assert grid  # read only, in the grid
    counted = press_counts(graph, ["XR01:PS1", "XR01:QF1"])
    assert (counted.left_out, counted.left_out_cards) == (1, 1)
    assert isinstance(counted, PressCount)


def test_an_inferred_press_value_is_said_so_wherever_its_value_is_named(tmp_path) -> None:
    from osprey.facility.panels.describe import INFERRED_NOTE, press_counts

    roles = (  # the database runs a sequence on any write: 1 is inferred
        "PowerSupply,reset,*:On-Cmd,device,action,,,1,Reset,inferred,S,1,l\n"
        "PowerSupply,on,*:On-Sts,device,word,,,,,stated,S,1,l\n"
    )
    graph = _roles_graph(tmp_path, roles, "XR01-PS1:On-Cmd,1,On,S,1,l\n")
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    (button,) = _by_type(doc, "pv-button")
    assert button["props"]["pressValue"] == 1
    assert button["props"]["tooltip"] == f"writes XR01-PS1:On-Cmd = 1 {INFERRED_NOTE}"
    assert _text(doc, f"writes XR01-PS1:On-Cmd = 1 {INFERRED_NOTE}; once per press")
    counted = press_counts(graph, ["XR01:PS1"])
    assert (counted.stated, counted.inferred) == (0, 1)


def test_a_command_with_no_role_is_an_action_named_by_its_word(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:PS1"), panel_graph)
    (button,) = _by_type(doc, "pv-button")
    assert (button["props"]["label"], button["props"]["pressValue"]) == ("On", 1)
    assert button["props"]["backgroundColor"]["$role"] == "Control_Action"
    # no state of the channel names 1: the value is inferred, and the card says so
    assert _text(
        doc,
        "writes XR01-PS1:On-Cmd = 1 (value inferred, not stated in the database); once per press",
    )


def test_every_channel_not_drawn_above_is_in_the_grid_with_its_full_name_on_hover(
    panel_graph,
) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:IP1"), panel_graph)
    for channel in ("XR01-IP1:Pressure-RB", "XR01-IP1:State-Sts"):
        assert [n for n in _by_type(doc, "rectangle") if n["props"].get("tooltip") == channel]
        assert [n for n in nodes(doc) if n["type"] != "polygon" and channel in _reads(doc, n)]
    # the state channel has a health rule but no stated word for its values: its
    # light is the neutral outline, as on a tile or in a chain block (one rule)
    assert not layout.meaning_stated(panel_graph, "XR01-IP1:State-Sts")
    assert not [n for n in _by_type(doc, "polygon") if n.get("rules")]
    assert [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == "no rule"]


def test_tiles_and_controls_come_from_the_channel_roles(tmp_path) -> None:
    roles = (
        "Pump,pressure,*:Pressure-RB,device,value,Torr,,,,stated,S,1,l\n"
        "Pump,state,*:State-Sts,device,lamp,,,,,stated,S,1,l\n"
        "PowerSupply,on-command,*:On-Cmd,device,choice,,,1,On,stated,S,1,l\n"
        "PowerSupply,on-command,*:On-Cmd,device,choice,,,,Off,stated,S,1,l\n"
        "PowerSupply,on,*:On-Sts,device,word,,,,,stated,S,1,l\n"
    )
    labels = (
        "XR01-IP1:State-Sts,0,Off,S,1,l\nXR01-IP1:State-Sts,1,Running,S,1,l\n"
        "XR01-PS1:On-Cmd,0,Off,S,1,l\nXR01-PS1:On-Cmd,1,On,S,1,l\n"
    )
    graph = _roles_graph(tmp_path, roles, labels)
    pump = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert _text(pump, "PRESSURE") and _text(pump, "STATE")  # one tile per role
    word = [n for n in _by_type(pump, "text-label") if n.get("rules") and n["props"].get("tooltip") == "XR01-IP1:State-Sts"][0]  # fmt: skip
    conditions = word["rules"][0]["conditions"]
    assert [c["value"] for c in conditions] == ["Off", "Running"]
    supply = stub_render(device_card(graph, "XR01:PS1"), graph)
    segments = _by_type(supply, "pv-button")
    assert [(b["props"]["label"], b["props"]["pressValue"]) for b in segments] == [
        ("On", 1),
        ("Off", 0),  # its value from the channel's own state words
    ]
    shade = segments[0]["rules"][0]
    assert shade["targetProperty"] == "backgroundColor"
    assert shade["pvs"] == [{"name": "XR01-PS1:On-Sts", "trigger": True}]  # the status role


def test_card_key_numbers_name_their_roles_and_a_missing_one_is_not_served(tmp_path) -> None:
    roles = "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
    metrics = (
        "card,,1,Vacuum,value,pressure,Pump,Torr,,XR-METRICS-001\n"
        "card,,2,Supply,word,supply-status,Pump,,,XR-METRICS-001\n"
    )
    graph = _roles_graph(tmp_path, roles, metrics=metrics)
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert _text(doc, "VACUUM") and _text(doc, "SUPPLY")
    assert _text(doc, "not served yet")
    assert [n for n in _by_type(doc, "pv-label") if n["props"]["pv"] == "XR01-IP1:Pressure-RB"]


def test_card_key_numbers_over_every_channel_of_a_role_count_max_and_sum(tmp_path) -> None:
    """count_true reads "k of N", max and sum one formula over all members; while any
    member has no value the figure is a dash and a line says so."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def outlets(document) -> None:
        for n in (1, 2, 3):
            for name in ("On-Sts", "Cur-RB"):
                document["channels"].append(
                    {"id": f"XR01-PS1:Out{n}{name}", "on": {"device": "XR01:PS1"}}
                )

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "PowerSupply,on,*:Out[0-9]On-Sts,device,lamp,,,,,stated,S,1,l\n"
        + "PowerSupply,load,*:Out[0-9]Cur-RB,device,value,A,,,,stated,S,1,l\n"
    )
    with (tables / "metrics.csv").open("a") as handle:
        handle.write(
            "card,,1,Outlets on,count_true,on,PowerSupply,,,S\n"
            "card,,2,Largest load,max,load,PowerSupply,A,,S\n"
            "card,,3,Total load,sum,load,PowerSupply,A,,S\n"
        )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", outlets), tables)
    view = device_card(graph, "XR01:PS1")
    tiles = [i for s in view.sections for i in s.items if type(i).__name__ == "KeyTile"]
    assert [(t.shown_as, len(t.channels)) for t in tiles] == [
        ("count_true", 3),
        ("max", 3),
        ("sum", 3),
    ]
    doc = stub_render(view, graph)
    formulas = {"formula://" + f["name"]: f for f in doc["formulas"]}
    figures = [
        n
        for n in _by_type(doc, "text-label")
        if any(
            r.get("formula") in formulas and r["name"].endswith("figure")
            for r in n.get("rules", ())
        )
    ]
    assert len(figures) == 3
    for node in figures:
        (rule,) = node["rules"]
        assert node["props"]["text"] == rule["default"] == layout.NO_VALUE  # no partial figure
        assert len(formulas[rule["formula"]]["bindings"]) == 3  # every member
    count = formulas[figures[0]["rules"][0]["formula"]]["expression"]
    assert count.count("!= 0") == 3
    assert "+" in formulas[figures[2]["rules"][0]["formula"]]["expression"]
    assert "max((" in formulas[figures[1]["rules"][0]["formula"]]["expression"]
    assert _text(doc, "of 3")
    notes = [
        n for n in _by_type(doc, "text-label") if n["props"]["text"] == "one of 3 has no value"
    ]
    assert len(notes) == 2  # the count's slot is its row of lamps, which needs no note
    for note, figure in zip(notes, figures[1:], strict=True):
        (rule,) = note["rules"]
        (pv,) = rule["pvs"]
        # its own short formula over the figure's members: the shaped figure inlined
        # into a condition would be too long for the engine
        assert pv["trigger"] and pv["name"] != figure["rules"][0]["formula"]
        assert (
            formulas[pv["name"]]["bindings"] == formulas[figure["rules"][0]["formula"]]["bindings"]
        )
        assert rule["conditions"][0]["value"] == ""  # cleared once every member reports
    assert emit._long_formulas(doc, "card") == []


def _over_all_card(
    tmp_path, metrics: str = "", words: str = "", n: int = 3, two_state: bool = True
):
    """A supply with *n* outlets (a lamp each, ``bool`` when *two_state*) and *n*
    loads (a value each); *metrics* card rows and *words* state-label rows as
    given."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def outlets(document) -> None:
        for i in range(1, n + 1):
            on = {"id": f"XR01-PS1:Out{i}On-Sts", "on": {"device": "XR01:PS1"}}
            if two_state:
                on["value_type"] = "bool"
            document["channels"].append(on)
            document["channels"].append(
                {"id": f"XR01-PS1:Out{i}Cur-RB", "on": {"device": "XR01:PS1"}}
            )

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "PowerSupply,outlet,*:Out*On-Sts,device,lamp,,,,,stated,S,1,l\n"
        + "PowerSupply,load,*:Out*Cur-RB,device,value,A,,,,stated,S,1,l\n"
    )
    if metrics.startswith("view,"):  # a whole file, its own header
        (tables / "metrics.csv").write_text(metrics)
    elif metrics:
        with (tables / "metrics.csv").open("a") as handle:
            handle.write(metrics)
    if words:
        labels = tables / "state_labels.csv"
        if not labels.exists():
            labels.write_text("channel,value,label,source,revision,locator\n")
        with labels.open("a") as handle:
            handle.write(words)
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", outlets), tables)
    return graph, device_card(graph, "XR01:PS1")


def _figures(doc) -> list[dict]:
    return [
        n
        for n in _by_type(doc, "text-label")
        if any(r["name"].endswith(" figure") for r in n.get("rules", ()))
    ]


def test_several_own_channels_of_a_role_get_a_default_figure_titled_by_the_role(tmp_path):
    from osprey.facility.panels.view import KeyTile

    _graph, view = _over_all_card(tmp_path)
    tiles = {t.title: t for s in view.sections for t in s.items if isinstance(t, KeyTile)}
    assert tiles["Outlet on"].shown_as == "count_true"  # a lamp: how many are true
    assert tiles["Outlet on"].names == ()  # a count names no holder
    assert tiles["Highest load"].shown_as == "max"  # a value: the highest
    assert tiles["Highest load"].unit == "A"
    assert tiles["Highest load"].names == ("Out1Cur-RB", "Out2Cur-RB", "Out3Cur-RB")


def test_a_count_s_title_takes_the_word_stated_for_the_true_state(tmp_path):
    from osprey.facility.panels.view import KeyTile

    words = "".join(f"XR01-PS1:Out{n}On-Sts,1,live,S,1,l\n" for n in (1, 2, 3))
    _graph, view = _over_all_card(tmp_path, words=words)
    titles = [t.title for s in view.sections for t in s.items if isinstance(t, KeyTile)]
    assert "Outlet live" in titles


def test_a_stated_card_row_wins_over_the_default_figure(tmp_path):
    from osprey.facility.panels.view import KeyTile

    _graph, view = _over_all_card(
        tmp_path, metrics="card,,1,Live,count_true,outlet,PowerSupply,,,S\n"
    )
    tiles = [t for s in view.sections for t in s.items if isinstance(t, KeyTile)]
    assert [t.title for t in tiles] == ["Live"]  # the stated row only, no default beside it


def test_a_count_reads_k_of_n_on_the_figure_s_baseline_and_the_line_keeps_a_margin(tmp_path):
    graph, view = _over_all_card(tmp_path)
    doc = stub_render(view, graph)
    figure, highest = _figures(doc)
    g = figure["geometry"]
    (of,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "of 3"]
    o = of["geometry"]
    # "of 3" follows the figure's widest count, a small gap after it, not at the tile's end
    assert o["x"] == pytest.approx(g["x"] + g["width"] + layout.ROW_GAP)
    assert g["width"] == layout.text_width("0", layout.KEY_VALUE_SIZE)
    figure_base = g["y"] + g["height"] / 2 + layout.baseline_drop(layout.KEY_VALUE_SIZE)
    of_base = o["y"] + o["height"] / 2 + layout.baseline_drop(of["props"]["fontSize"])
    assert of_base == pytest.approx(figure_base, abs=0.5)  # one reading, on one baseline
    # the line under a figure ends a margin above the tile's border
    h = highest["geometry"]
    (note,) = [
        n for n in _by_type(doc, "text-label")
        if n["props"]["text"] == "one of 3 has no value" and n["geometry"]["x"] == h["x"]
    ]  # fmt: skip
    tile = next(
        n for n in _by_type(doc, "rectangle")
        if n["geometry"]["x"] < h["x"] < n["geometry"]["x"] + n["geometry"]["width"]
        and n["geometry"]["height"] == layout.KEY_TILE_HEIGHT
    )  # fmt: skip
    bottom = note["geometry"]["y"] + note["geometry"]["height"]
    assert bottom + layout.KEY_NOTE_MARGIN == tile["geometry"]["y"] + layout.KEY_TILE_HEIGHT


def _holds(test: str, values: list[float]) -> bool:
    """A rule's condition evaluated as the formula engine would, over *values*: its
    ``max`` and ``min`` are numpy's, so one missing value (NaN) makes the extreme no
    number."""
    import numpy as np

    expression = test
    for i, v in enumerate(values):
        expression = expression.replace(f"{{{i}}}", repr(v))
    # the expression is one the layout just wrote, evaluated with nothing but max
    names = {
        "__builtins__": {},
        "max": lambda t: float(np.max(t)),
        "min": lambda t: float(np.min(t)),
        "nan": float("nan"),
    }
    return bool(eval(expression, names))


def _slot(doc, figure) -> list[dict]:
    g = figure["geometry"]
    return [
        n for n in _by_type(doc, "text-label")
        if n["geometry"]["x"] == g["x"] and n["geometry"]["y"] == g["y"] + g["height"]
    ]  # fmt: skip


def _shown(lines: list[dict], values: list[float]) -> list[str]:
    """The texts of the hidden *lines* whose rule shows them for *values*."""
    return [
        line["props"]["text"]
        for line in lines
        for rule in line["rules"]
        if rule["targetProperty"] == "invisible"
        and _holds(rule["conditions"][0]["expression"], values)
    ]


def test_the_highest_s_line_names_the_first_member_that_holds_it(tmp_path):
    graph, view = _over_all_card(tmp_path)
    doc = stub_render(view, graph)
    _count, highest = _figures(doc)
    lines = [n for n in _slot(doc, highest) if n["props"].get("invisible")]
    assert len(lines) == 3  # one per member, in the one slot of the line, all hidden
    assert _shown(lines, [1.0, 5.0, 2.0]) == ["highest of 3 · Out2Cur-RB"]
    assert _shown(lines, [7.0, 7.0, 2.0]) == ["highest of 3 · Out1Cur-RB"]  # a tie: the first
    assert _shown(lines, [1.0, float("nan"), 2.0]) == []  # no value: the note says so
    # what the figure is over is its tooltip, member by member
    tooltip = highest["props"]["tooltip"].splitlines()
    assert tooltip == ["max over 3 channels", "Out1Cur-RB", "Out2Cur-RB", "Out3Cur-RB"]


def test_the_lowest_is_drawn_like_the_highest(tmp_path):
    from osprey.facility.panels.view import KeyTile

    graph, view = _over_all_card(tmp_path, metrics="card,,1,Least load,min,load,PowerSupply,A,,S\n")
    (tile,) = [t for s in view.sections for t in s.items if isinstance(t, KeyTile)]
    assert tile.shown_as == "min" and len(tile.names) == 3
    doc = stub_render(view, graph)
    (figure,) = _figures(doc)
    formula = next(f for f in doc["formulas"] if "min((" in f["expression"])
    assert len(formula["bindings"]) == 3
    lines = [n for n in _slot(doc, figure) if n["props"].get("invisible")]
    assert _shown(lines, [4.0, 1.0, 1.0]) == ["lowest of 3 · Out2Cur-RB"]


def test_past_the_holder_bound_the_line_says_only_of_how_many(tmp_path):
    from osprey.facility.panels.describe import HOLDER_MAX, figure_notes

    n = HOLDER_MAX + 1
    # a stated precision keeps the figure's own formula short enough for this many
    row = (
        "view,level,order,title,measure,signal,class,unit,caption,source,precision\n"
        "card,,1,Load peak,max,load,PowerSupply,A,,S,1\n"
    )
    graph, view = _over_all_card(tmp_path, metrics=row, n=n)
    doc = stub_render(view, graph)
    (highest,) = _figures(doc)
    slot = _slot(doc, highest)
    assert not [line for line in slot if line["props"].get("invisible")]  # no name lines
    (note,) = slot
    (rule,) = note["rules"]
    assert rule["conditions"][0]["value"] == f"highest of {n}"  # with values
    assert rule["default"] == note["props"]["text"] == f"one of {n} has no value"
    _skipped, unnamed = figure_notes(graph, ["XR01:PS1"])
    assert unnamed == {"PowerSupply: Load peak": 1}


def test_a_count_s_slot_is_a_row_of_lamps_one_per_member(tmp_path):
    graph, view = _over_all_card(tmp_path)
    doc = stub_render(view, graph)
    count, _highest = _figures(doc)
    g = count["geometry"]
    below = g["y"] + g["height"]
    lamps = [
        n for n in nodes(doc)
        if n["geometry"]["y"] >= below - layout.ROW and n["geometry"]["y"] < below + layout.SMALL_LINE
        and n["geometry"]["width"] == layout.MOSAIC_LIGHT and n["type"] != "text-label"
        and g["x"] <= n["geometry"]["x"] < g["x"] + 300
    ]  # fmt: skip
    assert len(lamps) == 3
    xs = sorted(n["geometry"]["x"] for n in lamps)
    assert xs == [g["x"] + i * (layout.MOSAIC_LIGHT + layout.MOSAIC_GAP) for i in range(3)]
    for lamp in lamps:  # inside the slot, which keeps its margin above the border
        top = lamp["geometry"]["y"]
        assert below <= top and top + layout.MOSAIC_LIGHT <= below + layout.SMALL_LINE
    assert not [n for n in _slot(doc, count) if "has no value" in n["props"]["text"]]


def test_a_count_whose_lamps_do_not_fit_keeps_the_note(tmp_path):
    n = 30  # more lamps than a tile's width holds at the mosaic's size
    graph, view = _over_all_card(tmp_path, n=n)
    doc = stub_render(view, graph)
    count = _figures(doc)[0]
    (note,) = [x for x in _slot(doc, count) if "has no value" in x["props"]["text"]]
    assert note["props"]["text"] == f"one of {n} has no value"


def test_no_figure_s_rules_compile_past_the_engine_s_limit(tmp_path):
    """Significant digits shape a figure into a long formula; the slot's rule reads
    a short one, so at the holder bound nothing is too long. A count of more
    members than the slot's rule can say keeps its figure and drops the note."""
    from osprey.facility.panels.describe import HOLDER_MAX

    for n, row in (
        (HOLDER_MAX, "card,,1,Load peak,max,load,PowerSupply,A,,S,\n"),
        (100, "card,,1,Outlets on,count_true,outlet,PowerSupply,,,S,\n"),
    ):
        head = "view,level,order,title,measure,signal,class,unit,caption,source,precision\n"
        graph, view = _over_all_card(tmp_path / str(n), metrics=head + row, n=n)
        doc = stub_render(view, graph)
        assert emit._long_formulas(doc, "card") == []
        (figure,) = _figures(doc)
        assert figure["rules"][0]["formula"] in {f"formula://{f['name']}" for f in doc["formulas"]}
    assert not [x for x in _slot(doc, figure) if "has no value" in x["props"]["text"]]


def test_a_count_of_a_role_not_two_state_with_no_word_gets_no_default_tile(tmp_path):
    from osprey.facility.panels.describe import figure_notes
    from osprey.facility.panels.view import KeyTile

    graph, view = _over_all_card(tmp_path, two_state=False)
    titles = [t.title for s in view.sections for t in s.items if isinstance(t, KeyTile)]
    assert not any(title.startswith("Outlet") for title in titles)  # no word guessed
    skipped, _unnamed = figure_notes(graph, ["XR01:PS1"])
    assert skipped == {"PowerSupply.outlet": 1}


def test_a_row_named_by_its_raw_channel_shows_the_last_part_and_the_name_muted(
    panel_graph,
) -> None:
    from osprey.facility.panels.view import Target, View

    raw = ValueRow("XR01-QF1:Current-RB", "XR01-QF1:Current-RB", None, None, False)
    named = ValueRow("current readback", "XR01-QF1:Current-RB", None, None, False)
    view = View(
        Target("card", "device", "XR01:QF1"), "XR01:QF1", (),
        (Section("signals", (raw, named)),),
    )  # fmt: skip
    doc = stub_render(view, panel_graph)
    short = _text(doc, "Current-RB")
    full = _text(doc, "XR01-QF1:Current-RB")
    assert full["props"]["fontSize"] == layout.SMALL_SIZE
    assert full["props"]["foregroundColor"] == layout.MUTED_TEXT
    assert full["geometry"]["y"] >= short["geometry"]["y"] + layout.ROW
    assert _text(doc, "current readback")["geometry"]["y"] >= full["geometry"]["y"] + layout.ROW_GAP


def test_a_value_field_is_framed_once_by_the_viewer(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    for field in _fields(doc).values():
        if field["type"] in ("pv-label", "pv-text-input"):
            assert field["props"]["alarmSensitiveBorder"] is False
            assert "alarmBorder" not in field["props"]


def _calibration(doc: dict) -> list[dict]:
    """The calibration section's nodes: under its head, above the next section's."""
    heads = sorted(
        (n for n in _by_type(doc, "text-label") if " · " in n["props"]["text"][:5]),
        key=lambda n: n["geometry"]["y"],
    )
    (head,) = [n for n in heads if n["props"]["text"].endswith("· CALIBRATION")]
    later = [n["geometry"]["y"] for n in heads if n["geometry"]["y"] > head["geometry"]["y"]]
    end = min(later, default=float("inf")) - layout.PANEL_PAD
    return [n for n in nodes(doc) if head["geometry"]["y"] < n["geometry"]["y"] < end]


def test_the_calibration_block_is_full_width_under_the_columns_and_read_only(panel_graph):
    from osprey.facility.panels import symbols

    doc = stub_render(device_card(panel_graph, "XR01:QF1"), panel_graph)
    block = _calibration(doc)
    texts = {n["props"]["text"]: n for n in block if n["type"] == "text-label"}
    for head in ("PARAMETER", "VALUE IN FORCE", "BASIS", "DATE", "SOURCE", "HISTORY"):
        assert head in texts
    assert texts["HISTORY"]["geometry"]["x"] > layout.MARGIN + layout.STRIP_WIDTH / 2
    assert not [n for n in block if n["type"] in ("pv-text-input", "pv-button")]
    (live,) = [n for n in block if n["type"] == "pv-label"]
    assert live["props"]["pv"] == "XR01-QF1:Current-SP"  # held in EPICS: shown live
    chips = {
        n["props"]["text"]: n for n in block
        if n["type"] == "text-label" and n["props"]["text"] in (
            "bench · stretched wire", "beam wanted", "EPICS"
        )
    }  # fmt: skip
    assert chips["bench · stretched wire"]["props"]["foregroundColor"] == symbols.BASIS_BENCH
    boxes = [n for n in block if n["type"] == "rectangle" and n["props"].get("lineStyle")]
    dashed = [n for n in boxes if n["props"]["lineStyle"] == "dashed"]
    assert dashed and all(n["props"]["lineColor"] == symbols.MUTED_TEXT for n in dashed)
    contested = stub_render(device_card(panel_graph, "XR02:QF1"), panel_graph)
    (chip,) = [
        n for n in _calibration(contested)
        if n["type"] == "text-label" and n["props"]["text"] == "two sources disagree"
    ]  # fmt: skip
    assert chip["props"]["foregroundColor"] == symbols.DISAGREE


def test_the_basis_colours_stay_clear_of_the_state_colours_and_read_on_every_ground():
    from osprey.facility.panels import symbols
    from tests.facility.panels.test_symbols import GROUNDS, _resolved, contrast

    for basis, colour in symbols.BASIS_ROLES.items():
        for theme in ("light", "dark"):
            shade = _resolved(colour, theme)
            if basis != "contested":
                assert symbols.near_state(shade) is None, (basis, theme)
            for ground in GROUNDS:
                assert contrast(shade, _resolved(ground, theme)) >= 4.5, (basis, theme)


_LONG_TITLE = "Integrated gradient at the nominal current against the design lattice (x; y)"
_LONG_CONDITIONS = (
    "serial=XQF-01; current=73.33 A; cycle=5 times to 92.6 A, then to 73.33 A; "
    "ramp rate 5; hold 10 s; report state=Released; probe=coil B; temperature=22.5 C; "
    "operator log entry 17 of the second shift, with the rotating coil swapped once"
)


def test_a_parameter_title_wraps_and_is_never_cut_and_long_conditions_wrap(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    kinds = (tables / "calibration_kinds.csv").read_text()
    (tables / "calibration_kinds.csv").write_text(
        kinds.replace("Integrated gradient,", f'"{_LONG_TITLE}",', 1)
    )
    claims = (tables / "calibrations.csv").read_text()
    (tables / "calibrations.csv").write_text(
        claims.replace("at 73.33 A,bench", f'"{_LONG_CONDITIONS}",bench', 1)
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    doc = stub_render(device_card(graph, "XR01:QF1"), graph)
    block = _calibration(doc)
    texts = [n for n in block if n["type"] == "text-label"]
    words = " ".join(n["props"]["text"] for n in texts)
    assert _LONG_TITLE in " ".join(
        n["props"]["text"] for n in texts if n["props"]["fontSize"] == layout.TEXT_SIZE
    )  # every word, in order, over lines of their own
    assert "…" not in "".join(
        n["props"]["text"] for n in texts if n["props"]["fontSize"] == layout.TEXT_SIZE
    )
    lines = [
        n for n in texts
        if n["props"]["fontSize"] == layout.SMALL_SIZE
        and len(n["props"]["text"]) > 20
        and n["props"]["text"].rstrip("…") in _LONG_CONDITIONS
    ]  # fmt: skip
    assert len(lines) == 2 and lines[-1]["props"]["text"].endswith("…")
    # the cut line shows the whole conditions on hover, from a box over it
    (hover,) = [n for n in block if n["props"].get("tooltip") == _LONG_CONDITIONS]
    assert hover["type"] == "rectangle" and hover["geometry"] == lines[-1]["geometry"]
    assert "serial=XQF-01" in words
    value = next(n for n in texts if n["props"]["text"].startswith("-4.9198"))
    assert all(n["geometry"]["x"] == value["geometry"]["x"] for n in lines)  # under the value


def _ported(document) -> None:
    """The pump read out on port CH2 of its chassis, which states one channel per port."""
    for claim in document["relations"]:
        if claim["id"] == "xr01-ip1-readout":
            claim["object"]["port"] = "CH2"
    for n in (1, 2):
        document["channels"].append({"id": f"XR01-CH2:IP{n}:P", "on": {"device": "XR01:CH2"}})


def test_a_walked_role_takes_the_channel_of_its_port_and_a_contested_one_none(tmp_path):
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.model import substitute
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    assert substitute("*:IP{n}:P", "CH2") == "*:IP2:P"
    assert substitute("*:{port}:P", "A[1]") == "*:A[[]1]:P"  # the port matched literally
    assert substitute("*:IP{n}:P", "CH") is None and substitute("*:{port}", None) is None
    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "Pump,pressure,*:IP{n}:P,read_out_by,value,,,,,stated,S,1,l\n"
        + "Pump,current,*:IP?:P,read_out_by,value,,,,,stated,S,1,l\n"  # two match: contested
        + "Pump,state,*:State-Sts,device,word,,,,,stated,S,1,l\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _ported), tables)
    hits = graph.roles("XR01:IP1")
    assert [(h.role.role, h.channel, h.owner, h.port) for h in hits] == [
        ("pressure", "XR01-CH2:IP2:P", "XR01:CH2", "CH2"),
        ("state", "XR01-IP1:State-Sts", "XR01:IP1", None),
        ("vacuum_readback", "XR01-IP1:Pressure-RB", "XR01:IP1", None),  # its signal, by default
    ]
    from osprey.facility.panels.view import ChannelGrid

    view = device_card(graph, "XR01:IP1")
    grid = next(i for s in view.sections for i in s.items if isinstance(i, ChannelGrid))
    # its own, then the chassis's channels beside the one its port names; not IP1's
    assert [c for _s, c in grid.rows] == [
        "XR01-IP1:Pressure-RB",
        "XR01-IP1:State-Sts",
        "XR01-CH2:IP2:P",
    ]
    from osprey.facility.panels.describe import channel_list

    listed = {i.channel for s in channel_list(graph, "XR01:IP1").sections for i in s.items}
    assert listed == {c for _s, c in grid.rows}  # the full list holds what the grid does


def test_a_recipe_names_the_card_fills_its_sentence_and_sets_the_grid_columns(tmp_path):
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import CardHead, ChannelGrid
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,recipe,description,channels_cols,source\n"
        "Pump,,pump,Pump on {located_in}; read out by {read_out_by}; fed by {powered_by},2,S\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    view = device_card(graph, "XR01:IP1")
    head = next(i for s in view.sections for i in s.items if isinstance(i, CardHead))
    assert head.kind == "Pump · pump recipe" and head.default is False
    assert head.description == "Pump on XR/XR01; read out by XR01:CH2; fed by not stated"
    grid = next(i for s in view.sections for i in s.items if isinstance(i, ChannelGrid))
    assert grid.columns == 2
    doc = stub_render(view, graph)
    assert _text(doc, head.description)
    other = device_card(graph, "XR01:QF1")
    assert next(i for s in other.sections for i in s.items if isinstance(i, CardHead)).default


def test_where_and_sources_are_one_muted_line_each_a_contested_claim_says_so(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR02:CH5"), panel_graph)
    (head,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "Where:"]
    y = head["geometry"]["y"]
    line = sorted(
        (n for n in _by_type(doc, "text-label") if n["geometry"]["y"] == y),
        key=lambda n: n["geometry"]["x"],
    )
    text = "".join(n["props"]["text"] for n in line)
    assert text == "Where:XR/XR01/RK1 units 40, 41 or XR/XR02/RK2 unit 5 · contested"
    (sources,) = [
        n for n in _by_type(doc, "text-label") if n["props"]["text"].startswith("Sources:")
    ]
    for n in (*line, sources):
        assert n["props"]["foregroundColor"] == layout.MUTED_TEXT
    assert sources["geometry"]["y"] == y + layout.SMALL_LINE + layout.ROW_GAP
    opens = [
        n
        for n in _by_type(doc, "rectangle")
        if n["props"].get("actions") and n["geometry"]["y"] == y
    ]
    assert len(opens) == 2  # each place opens its rack view
    assert not [n for n in _by_type(doc, "text-label") if n["props"]["text"].endswith("· WHERE")]


def test_a_recipe_for_some_devices_of_a_class_wins_and_keys_its_own_tiles(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\n"
        "Pump,XR,*:IP1,first-pump,,,S\n"
        "Pump,,,pump,,,S\n"
    )
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
    )
    with (tables / "metrics.csv").open("a") as handle:
        handle.write("card,first-pump,1,Vacuum,value,pressure,Pump,,,S\n")
        handle.write("card,pump,1,Pressure here,value,pressure,Pump,,,S\n")
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    assert graph.recipe("XR01:IP1").recipe == "first-pump"  # the first row that takes it
    assert graph.recipe("XR01:IP2").recipe == "pump"
    first = stub_render(device_card(graph, "XR01:IP1"), graph)
    other = stub_render(device_card(graph, "XR01:IP2"), graph)
    assert _text(first, "VACUUM") and not [
        n for n in _by_type(first, "text-label") if n["props"]["text"] == "PRESSURE HERE"
    ]
    assert _text(other, "PRESSURE HERE")


def _chained(tmp_path, change=None):
    """A pump recipe whose chain walks its read-out (a chassis, its temperature),
    its power (none stated) and what it is read out by in turn."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nPump,,,pump,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "pump,1,chain,Signal chain,,signal,,,the chain's caption,S\n"
    )
    (tables / "chains.csv").write_text(
        "recipe,chain,step,from_step,relation,direction,to,many,port_from,title,roles,source\n"
        "pump,signal,1,0,read_out_by,out,ElectronicsChassis:*:CH*,,object_port,Read-out,temperature,S\n"
        "pump,signal,2,0,powered_by,out,device,,,Supply,,S\n"
        "pump,signal,3,1,read_out_by,out,,,,Further,,S\n"
    )
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "ElectronicsChassis,temperature,*:Temp-RB,device,value,,,,,stated,S,1,l\n"
        + "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
    )
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _chain_of(view):
    from osprey.facility.panels.view import Chain

    return next(i for s in view.sections for i in s.items if isinstance(i, Chain))


def test_the_device_s_block_shows_no_role_several_of_its_channels_play(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    def statuses(document) -> None:
        _ported(document)
        for n in (1, 2, 3):
            document["channels"].append(
                {"id": f"XR01-IP1:OutStatus{n}", "on": {"device": "XR01:IP1"}}
            )

    _chained(tmp_path)
    tables = tmp_path / "panels"
    with (tables / "channel_roles.csv").open("a") as handle:
        handle.write("Pump,out-status,*:OutStatus[0-9],device,word,,,,,stated,S,1,l\n")
    graph = load_graph(write_fixture_facility(tmp_path / "g.json", statuses), tables)
    device = _chain_of(device_card(graph, "XR01:IP1")).blocks[0]
    assert [r.channel for r in device.rows] == [
        "XR01-IP1:Pressure-RB",
        "XR01-IP1:State-Sts",  # its signal, a role by default
    ]  # one per connector: not here


def test_a_recipe_chain_walks_its_hops_and_says_where_one_is_not_stated(tmp_path):
    from osprey.facility.panels.describe import chain_states

    graph = _chained(tmp_path, _ported)
    view = device_card(graph, "XR01:IP1")
    assert "Signal chain" in [s.title for s in view.sections]
    assert "relations" not in [s.title for s in view.sections]  # the recipe's chain instead
    device, readout, supply, further = _chain_of(view).blocks
    assert device.focus and [r.channel for r in device.rows] == [
        "XR01-IP1:Pressure-RB",
        "XR01-IP1:State-Sts",  # no row names it: its signal is its role
    ]
    assert device.arrow == "CH2"  # the read-out's object port, on the arrow into it
    assert readout.title == "Read-out  XR01:CH2" and readout.link.target.id == "XR01:CH2"
    assert [(r.label, r.channel) for r in readout.rows] == [("temperature", "XR01-CH2:Temp-RB")]
    assert supply.dashed and supply.title == "Supply: not stated"
    assert further.dashed  # the read-out is read out by nothing in turn
    assert _chain_of(view).foot == "the chain's caption"
    states = chain_states(graph, ["XR01:IP1", "XR01:IP2"])
    assert states["pump/signal"] == {"Supply: not stated": 1, "Read-out: not stated": 1}
    doc = stub_render(view, graph)
    assert _text(doc, "1 · SIGNAL CHAIN")
    assert _text(doc, "CH2")  # the arrow's word


def test_a_hop_with_two_answers_stacks_them_contested(tmp_path):
    def two_readouts(document) -> None:
        _ported(document)
        document["relations"].append(
            {
                "id": "xr01-ip1-readout-2",
                "type": "read_out_by",
                "subject": {"device": "XR01:IP1"},
                "object": {"device": "XR01:CH3"},
            }
        )

    graph = _chained(tmp_path, two_readouts)
    _device, readout, *_rest = _chain_of(device_card(graph, "XR01:IP1")).blocks
    assert readout.standing == "contested"
    assert [b.title for b in (readout, *readout.stack)] == [
        "Read-out  XR01:CH2",
        "Read-out  XR01:CH3",
    ]
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert len([n for n in _by_type(doc, "text-label") if n["props"]["text"] == "contested"]) == 2


def test_a_claim_contested_over_something_else_is_still_one_target(tmp_path):
    def contested_claim(document) -> None:
        _ported(document)
        for claim in document["relations"]:
            if claim["id"] == "xr01-ip1-readout":
                claim["standing"] = "contested"  # contested over another thing, not here

    graph = _chained(tmp_path, contested_claim)
    _device, readout, *_rest = _chain_of(device_card(graph, "XR01:IP1")).blocks
    assert readout.standing is None and not readout.stack
    assert readout.title == "Read-out  XR01:CH2"


def test_a_hop_from_a_many_step_walks_from_each_and_unites_the_targets(tmp_path):
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.describe import chain_states
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nPowerSupply,,,supply,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,chain,Loads,,loads,,,,S\n"
    )
    (tables / "chains.csv").write_text(
        "recipe,chain,step,from_step,relation,direction,to,many,port_from,title,roles,source\n"
        "supply,loads,1,0,powered_by,in,Quadrupole,1,feeds,Magnet,,S\n"
        "supply,loads,2,1,powered_by,out,device,,,Fed by,,S\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    _device, magnets, fed = _chain_of(device_card(graph, "XR01:PS1")).blocks
    assert len((magnets, *magnets.stack)) == 2
    # both magnets name the one supply: one distinct target, found, not contested
    assert fed.title == "Fed by  XR01:PS1" and fed.standing is None and not fed.stack
    assert chain_states(graph, ["XR01:PS1"])["supply/loads"] == {"found": 1}


def test_a_hop_to_many_draws_every_thing_it_reaches_in_one_column(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nPowerSupply,,,supply,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,chain,Loads,,loads,,,,S\n"
    )
    (tables / "chains.csv").write_text(
        "recipe,chain,step,from_step,relation,direction,to,many,port_from,title,roles,source\n"
        "supply,loads,1,0,powered_by,in,Quadrupole,yes,feeds,Magnet,,S\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
    _device, magnets = _chain_of(device_card(graph, "XR01:PS1")).blocks
    assert [b.title for b in (magnets, *magnets.stack)] == ["Magnet  XR01:QD1", "Magnet  XR01:QF1"]
    assert magnets.standing is None and _device.arrow == "feeds"  # a word of its own
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    tops = sorted(
        n["geometry"]["y"]
        for n in _by_type(doc, "text-label")
        if n["props"]["text"].startswith("Magnet")
    )
    lefts = {
        n["geometry"]["x"]
        for n in _by_type(doc, "text-label")
        if n["props"]["text"].startswith("Magnet")
    }
    assert len(tops) == 2 and tops[1] > tops[0] and len(lefts) == 1  # one column


def test_a_word_with_no_stated_meaning_keeps_inside_its_block(tmp_path):
    graph = _chained(tmp_path, _ported)
    roles = graph._tables.channel_roles  # the read-out's temperature, shown as a word
    graph._roles_of["ElectronicsChassis"] = [
        dataclasses.replace(r, shown_as="word") for r in roles if r.cls == "ElectronicsChassis"
    ]
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    (box,) = [
        n for n in _by_type(doc, "rectangle")
        if n["props"].get("lineWidth") == layout.BLOCK_LINE
        and any(
            m["props"]["text"] == "temperature" and abs(m["geometry"]["y"] - n["geometry"]["y"]) < 40
            for m in _by_type(doc, "text-label")
        )
    ]  # fmt: skip
    right = box["geometry"]["x"] + box["geometry"]["width"]
    inside = [
        n for n in nodes(doc)
        if box["geometry"]["y"] <= n["geometry"]["y"] < box["geometry"]["y"] + box["geometry"]["height"]
        and box["geometry"]["x"] <= n["geometry"]["x"] < right
    ]  # fmt: skip
    assert inside and all(n["geometry"]["x"] + n["geometry"]["width"] <= right for n in inside)


def _stripped(tmp_path, *, place_pose: bool = False, change=None):
    """A pump recipe whose first section is a strip of pumps and quadrupoles."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nPump,,,pump,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "pump,1,strip,Where it sits,,Pump;Quadrupole,3,pressure,the strip's caption,S\n"
    )
    (tables / "symbols.csv").write_text(
        "class,devices,symbol,lane,source\nPump,,pump,below,S\nQuadrupole,*:QF*,quadrupole,beam,S\n"
    )
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
    )
    if place_pose:  # the place has a position of its own
        with (tables / "positions.csv").open("a") as handle:
            handle.write(
                "XR/XR01,XR-RING,0,0,0,,,,1.5,0.0,0.0,,,stated,XR-LAYOUT,A,released,sheet 9\n"
            )
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def test_a_strip_draws_the_members_round_the_device_to_scale(tmp_path) -> None:
    from osprey.facility.panels.view import CardStrip

    graph = _stripped(tmp_path)
    view = device_card(graph, "XR01:IP1")
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    assert strip.window == (-1.75, 4.25) and strip.bracket is None
    assert [(m.thing, m.symbol, m.lane, m.focus) for m in strip.members] == [
        ("XR01:IP1", "pump", "below", True),
        ("XR01:QF1", "quadrupole", "beam", False),  # QD1 matches no symbols row: not drawn
    ]
    assert strip.members[0].channel == "XR01-IP1:Pressure-RB"
    doc = stub_render(view, graph)
    # the quadrupole by its record's label; the pump, with none, by its id's last part
    assert [m.label for m in strip.members] == ["IP1", "F1"]
    assert _text(doc, "IP1 · here")["props"]["foregroundColor"]["$role"] == "Focus"
    assert _text(doc, "2 m") and _text(doc, "-1 m")  # a tick every metre
    opens = [n for n in _by_type(doc, "rectangle") if n["props"].get("tooltip") == "XR01:QF1"]
    assert opens and opens[0]["props"]["actions"]


def test_a_strip_window_reads_on_across_a_rings_start(tmp_path) -> None:
    from osprey.facility.panels.view import CardStrip
    from tests.facility.panels.test_describe import ring_model

    graph = _stripped(tmp_path, change=ring_model(12.0))
    view = device_card(graph, "XR01:IP1")  # at 1.25; XR02:QF1 at 11.0 is 2.25 m before it
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    assert [(m.thing, m.s_m) for m in strip.members] == [
        ("XR02:QF1", pytest.approx(-1.0)),
        ("XR01:IP1", 1.25),
        ("XR01:QF1", 1.25),
    ]
    line = _stripped(tmp_path / "line")  # no ring length stated: nothing read across
    view = device_card(line, "XR01:IP1")
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    assert "XR02:QF1" not in [m.thing for m in strip.members]


def test_close_strip_members_keep_their_names_and_values_apart(tmp_path) -> None:
    from osprey.facility.panels.view import CardStrip, StripMember

    graph = _stripped(tmp_path)
    view = device_card(graph, "XR01:IP1")
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    channel = strip.members[0].channel
    crowd = tuple(
        StripMember(f"XR01:M{i}", f"MEMBER{i}", 0.1 * i, "pump", lane, channel=channel)
        for i, lane in enumerate(("below", "beam", "below", "beam", "above", "above"))
    )
    crowded = dataclasses.replace(strip, members=crowd)
    sections = tuple(
        dataclasses.replace(s, items=tuple(crowded if i is strip else i for i in s.items))
        for s in view.sections
    )
    doc = stub_render(dataclasses.replace(view, sections=sections), graph)
    boxes = [
        n["geometry"]
        for n in nodes(doc)
        if n["type"] == "text-label"
        and (str(n["props"].get("text", "")).startswith("MEMBER") or channel in json.dumps(n))
    ]
    assert len(boxes) >= 2 * len(crowd)

    def apart(a, b) -> bool:
        return (
            a["x"] + a["width"] <= b["x"] + 0.01
            or b["x"] + b["width"] <= a["x"] + 0.01
            or a["y"] + a["height"] <= b["y"] + 0.01
            or b["y"] + b["height"] <= a["y"] + 0.01
        )

    clashes = [(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :] if not apart(a, b)]
    assert clashes == []


def test_a_device_placed_only_by_its_place_is_a_dashed_bracket_and_one_placed_by_none_says_so(
    tmp_path,
) -> None:
    from osprey.facility.panels.describe import STRIP_BY_PLACE, STRIP_NOWHERE, strip_state
    from osprey.facility.panels.view import CardHead, CardStrip

    graph = _stripped(tmp_path, place_pose=True)
    view = device_card(graph, "XR01:IP2")
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    # the place's stretch: its own point (1.5) and its devices' positions (1.25 to 2.0)
    assert strip.bracket == (1.25, 2.0, "in XR/XR01, place in it not stated")
    assert strip.window == (1.625 - 3, 1.625 + 3)
    assert [m.thing for m in strip.members if m.focus] == ["XR01:IP2"]
    doc = stub_render(view, graph)
    assert _text(doc, "in XR/XR01, place in it not stated")
    assert strip_state(graph, graph.thing("XR01:IP2", "device")) == STRIP_BY_PLACE

    def unplaced(document) -> None:
        for device in document["devices"]:
            if device["id"] == "XR01:IP2":
                device.pop("place", None)

    nowhere = _stripped(tmp_path / "b", change=unplaced)
    view = device_card(nowhere, "XR01:IP2")
    assert not [i for s in view.sections for i in s.items if isinstance(i, CardStrip)]
    head = next(i for s in view.sections for i in s.items if isinstance(i, CardHead))
    assert head.nowhere
    assert _text(stub_render(view, nowhere), "where it sits is not stated")
    assert strip_state(nowhere, nowhere.thing("XR01:IP2", "device")) == STRIP_NOWHERE


def test_strip_states_count_the_devices_of_a_recipe_by_how_they_are_placed(tmp_path):
    from osprey.facility.panels.describe import strip_states

    graph = _stripped(tmp_path, place_pose=True)
    assert strip_states(graph, ["XR01:IP1", "XR01:IP2", "XR01:QF1"]) == {
        "pump": {"placed": 1, "by its place": 1}
    }


def test_a_lamp_whose_meaning_is_not_stated_is_the_neutral_light(tmp_path) -> None:
    roles = "Pump,state,*:State-Sts,device,lamp,,,,,stated,S,1,l\n"
    bare = _roles_graph(tmp_path / "a", roles)
    assert bare.rule("XR01-IP1:State-Sts")  # a health rule exists, its words do not
    tile_lights = [
        n
        for n in _by_type(stub_render(device_card(bare, "XR01:IP1"), bare), "polygon")
        if n["props"].get("tooltip") == layout._NO_RULE
    ]
    assert tile_lights
    labels = "XR01-IP1:State-Sts,0,Off,S,1,l\nXR01-IP1:State-Sts,1,Running,S,1,l\n"
    worded = _roles_graph(tmp_path / "b", roles, labels)
    doc = stub_render(device_card(worded, "XR01:IP1"), worded)
    assert not [n for n in _by_type(doc, "polygon") if n["props"].get("tooltip") == layout._NO_RULE]


def test_driver_internal_records_are_never_offered_nor_listed_only_counted(tmp_path) -> None:
    roles = "PowerSupply,internal,*:On-Cmd,device,value,,,,,stated,S,1,l\n"
    graph = _roles_graph(tmp_path, roles)
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert _by_type(doc, "pv-button") == []  # its only command is internal
    assert _text(doc, "no operator actions stated")
    assert _text(doc, "1 driver-internal records not shown")
    assert not [n for n in nodes(doc) if "XR01-PS1:On-Cmd" in _reads(doc, n)]
    assert [n for n in nodes(doc) if "XR01-PS1:On-Sts" in _reads(doc, n)]


def test_written_screens_offer_no_press_nor_entry_on_an_internal_record(tmp_path) -> None:
    """A device whose bo command record is internal: on the written card no button
    and no entry acts on it, through no section and no fallback; its channels view
    lists it, read only, and the card links there."""
    import json

    from osprey.facility.panels.emit import write

    roles = "PowerSupply,internal,*:On-Cmd,device,value,,,,,stated,S,1,l\n"
    graph = _roles_graph(tmp_path, roles)
    assert graph.record_type("XR01-PS1:On-Cmd").record_type == "bo"
    out = tmp_path / "out"
    write(graph, out)
    (card,) = out.glob("card/*xr01-ps1.screen.json")
    (listing,) = out.glob("channels/*xr01-ps1.screen.json")
    writing = {"pv-button", "pv-text-input", "pv-slider", "pv-combo"}

    def touching(doc: dict) -> list[dict]:
        return [n for n in nodes(doc) if "XR01-PS1:On-Cmd" in _reads(doc, n)]

    doc = json.loads(card.read_text())
    assert not [n for n in touching(doc) if n["type"] in writing]
    assert not [
        n
        for n in nodes(doc)
        if n["type"] in writing and n["props"].get("pv", "").startswith("XR01-PS1:On-Cmd")
    ]
    assert _text(doc, "1 driver-internal records not shown")
    assert [n for n in nodes(doc) if n["props"].get("text", "").startswith("in the full list")]
    shown = json.loads(listing.read_text())
    found = touching(shown)
    assert found and {n["type"] for n in found} <= {
        "pv-label",
        "rectangle",
        "polygon",
        "text-label",
    }


def test_a_device_over_the_cap_shows_counts_and_opens_its_channels_view(tmp_path, monkeypatch):
    import importlib

    describing = importlib.import_module("osprey.facility.panels.describe")
    from osprey.facility.panels.describe import describe, targets
    from osprey.facility.panels.view import ChannelCount, Target, ValueRow

    monkeypatch.setattr(describing, "GRID_CAP", 1)
    roles = (
        "Quadrupole,current,*:Current-RB,device,value,,,,,stated,S,1,l\n"
        "PowerSupply,internal,*:On-Cmd,device,value,,,,,stated,S,1,l\n"
    )
    graph = _roles_graph(tmp_path, roles)
    found = {t for t in targets(graph) if t.view == "channels"}
    assert Target("channels", "device", "XR01:QF1") in found
    assert Target("channels", "device", "XR01:PS1") in found  # its internal record is read there
    view = device_card(graph, "XR01:QF1")
    (count,) = [i for s in view.sections for i in s.items if isinstance(i, ChannelCount)]
    assert count.total == 2 and count.parts[0] == "2 on Quadrupole XR01:QF1"
    doc = stub_render(view, graph)
    assert _text(doc, "open the full list ›")
    listing = describe(graph, Target("channels", "device", "XR01:QF1"))
    titles = [s.title for s in listing.sections]
    assert titles[0] == "Current"  # by role first, then by record type
    assert all(isinstance(i, ValueRow) for s in listing.sections for i in s.items)
    assert {i.channel for s in listing.sections for i in s.items} == {
        "XR01-QF1:Current-RB",
        "XR01-QF1:Current-SP",
    }


def test_a_choice_of_one_stated_segment_is_an_action_with_the_status_beside_it(tmp_path):
    roles = (
        "PowerSupply,bypass,*:On-Cmd,device,choice,,,1,Bypass,stated,S,1,l\n"
        "PowerSupply,on,*:On-Sts,device,word,,,,,stated,S,1,l\n"
    )
    labels = "XR01-PS1:On-Sts,0,Normal,S,1,l\nXR01-PS1:On-Sts,1,Bypassed,S,1,l\n"
    graph = _roles_graph(tmp_path / "a", roles, labels)
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    (button,) = _by_type(doc, "pv-button")
    assert (button["props"]["label"], button["props"]["pressValue"]) == ("Bypass", 1)
    assert button["props"]["backgroundColor"]["$role"] == "Control_Action"
    tags = [
        n["props"]["text"]
        for n in _by_type(doc, "text-label")
        if n["props"]["text"] in ("ACTION", "CHOICE")
    ]
    assert tags == ["ACTION", "CHOICE", "ACTION"]  # the legend's two, then the group's
    beside = [
        n for n in nodes(doc)
        if n.get("rules") and n["props"].get("tooltip") == "XR01-PS1:On-Sts"
        and n["geometry"]["y"] >= button["geometry"]["y"]
        and n["geometry"]["x"] > button["geometry"]["x"]
    ]  # fmt: skip
    assert beside  # the live word, right of the button
    both = labels + "XR01-PS1:On-Cmd,0,Normal,S,1,l\nXR01-PS1:On-Cmd,1,Bypass,S,1,l\n"
    graph = _roles_graph(tmp_path / "b", roles, both)
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert [
        (b["props"]["label"], b["props"]["pressValue"]) for b in _by_type(doc, "pv-button")
    ] == [
        ("Normal", 0),
        ("Bypass", 1),
    ]  # the off value stated: a choice of both


def test_press_groups_stand_side_by_side_and_entries_take_the_width(tmp_path) -> None:
    roles = (
        "PowerSupply,reset,*:On-Cmd,device,action,,,1,Reset,stated,S,1,l\n"
        "PowerSupply,start,*:On-Cmd,device,action,,,1,Start,stated,S,1,l\n"
    )
    graph = _roles_graph(tmp_path, roles)
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    reset, start = sorted(_by_type(doc, "pv-button"), key=lambda n: n["geometry"]["x"])
    assert reset["geometry"]["y"] == start["geometry"]["y"]
    third = (layout.STRIP_WIDTH - 2 * layout.COLUMN_GAP) / 3 + layout.COLUMN_GAP
    assert abs(start["geometry"]["x"] - reset["geometry"]["x"] - third) < 0.1


FRONTS_HEADER = (
    "recipe,face,group,order,connector,kind,count,channel,far_relation,far_direction,"
    "far_port,label,source\n"
)


def _fronted(tmp_path, change=None, fronts: str = "", roles: str = ""):
    """A supply recipe with a front whose outputs are the loads powered by it on a port."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\n"
        "PowerSupply,,,supply,,,S\nInfrastructure,,XR02:PP*,panel,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,front,Outputs,,front,,state;cmd,the front's caption,S\n"
        "panel,1,front,Sockets,,front,,,,S\n"
    )
    (tables / "fronts.csv").write_text(
        FRONTS_HEADER
        + (fronts or "supply,front,outputs,1,Out{n},power,3,,powered_by,in,{n},Out {n},S\n")
    )
    (tables / "channel_roles.csv").write_text(ROLES_HEADER + roles)
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _ports(*pairs):
    """Give the supply's loads object ports: (claim id, port)."""

    def change(document) -> None:
        for claim in document["relations"]:
            for claim_id, port in pairs:
                if claim["id"] == claim_id:
                    claim["object"]["port"] = port

    return change


def _front_of(view):
    from osprey.facility.panels.view import Front

    return next(i for s in view.sections for i in s.items if isinstance(i, Front))


def test_a_front_numbers_its_connectors_and_names_what_is_on_each(tmp_path):
    graph = _fronted(tmp_path, _ports(("xr01-qf1-power", "1"), ("xr01-qd1-power", "2")))
    front = _front_of(device_card(graph, "XR01:PS1"))
    (group,) = front.groups
    assert [c.label for c in group.connectors] == ["Out 1", "Out 2", "Out 3"]
    one, two, three = group.connectors
    assert [e.thing for e in one.sides[0]] == ["XR01:QF1"]
    assert one.sides[0][0].link.target.id == "XR01:QF1"  # opens its card
    assert [e.thing for e in two.sides[0]] == ["XR01:QD1"] and three.sides == ((),)
    assert front.caption == "the front's caption"
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    for word in ("Out 1", "Out 2", "Out 3", "—"):  # nothing stated on output 3
        assert [n for n in _by_type(doc, "text-label") if n["props"]["text"] == word], word
    assert [n for n in nodes(doc) if n["props"].get("tooltip") == "open XR01:QF1"]


def test_two_things_on_one_connector_are_both_drawn_contested(tmp_path):
    from osprey.facility.panels.describe import FRONT_CONTESTED, front_state

    graph = _fronted(tmp_path, _ports(("xr01-qf1-power", "1"), ("xr01-qd1-power", "1")))
    front = _front_of(device_card(graph, "XR01:PS1"))
    one = front.groups[0].connectors[0]
    assert [e.thing for e in one.sides[0]] == ["XR01:QD1", "XR01:QF1"]
    assert front_state(one.sides) == FRONT_CONTESTED
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert _text(doc, "contested")


def test_a_range_is_numbered_by_its_served_channels_in_number_order(tmp_path):
    from osprey.facility.panels.describe import front_numbers

    def outlets(document) -> None:
        for n in (10, 2, 1):
            document["channels"].append(
                {"id": f"XR01-PS1:Out{n}-Sts", "on": {"device": "XR01:PS1"}}
            )

    graph = _fronted(
        tmp_path,
        outlets,
        "supply,front,outputs,1,Out{n},socket,,*:Out{n}-Sts,powered_by,in,{n},,S\n",
    )
    (row,) = graph.fronts("supply")
    assert front_numbers(graph, "XR01:PS1", row) == ["1", "2", "10"]  # never as text
    connectors = _front_of(device_card(graph, "XR01:PS1")).groups[0].connectors
    assert [c.lamp for c in connectors] == [
        "XR01-PS1:Out1-Sts",
        "XR01-PS1:Out2-Sts",
        "XR01-PS1:Out10-Sts",
    ]


def test_clients_on_no_stated_port_are_drawn_apart_by_class(tmp_path):
    graph = _fronted(
        tmp_path, None, "supply,front,outputs,1,Out{n},power,,,powered_by,in,{n},,S\n"
    )  # no port is stated: no connector, every load on no stated port
    front = _front_of(device_card(graph, "XR01:PS1"))
    (group,) = front.groups
    assert group.connectors == () and group.empty == "port count and assignment not stated"
    assert [(cls, [e.thing for e, _rows in things]) for cls, things in front.loose] == [
        ("Quadrupole", ["XR01:QD1", "XR01:QF1"])
    ]
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert _text(doc, "Quadrupole (2)")


def test_a_socket_of_its_own_shows_both_sides_and_a_range_with_no_socket_is_left_out(tmp_path):
    def sockets(document) -> None:
        document["devices"] += [
            {"id": "XR02:PP1", "class": "Infrastructure"},
            {"id": "XR02:PP1:A1", "class": "Infrastructure"},
        ]
        document["relations"] += [
            {"id": "pp-out", "type": "networked_to", "subject": {"device": "XR02:PP1:A1"},
             "object": {"device": "XR01:CH1"}},
            {"id": "pp-in", "type": "networked_to", "subject": {"device": "XR01:CH2"},
             "object": {"device": "XR02:PP1:A1"}},
        ]  # fmt: skip

    graph = _fronted(
        tmp_path, sockets,
        "panel,front,row A,1,A{n},socket,,,networked_to,socket,,A{n},S\n"
        "panel,front,row B,2,B{n},socket,,,networked_to,socket,,B{n},S\n",
    )  # fmt: skip
    front = _front_of(device_card(graph, "XR02:PP1"))
    (group,) = front.groups  # row B has no socket: no part of this panel
    (socket,) = group.connectors
    assert [[e.thing for e in side] for side in socket.sides] == [["XR01:CH1"], ["XR01:CH2"]]


def test_a_connector_shows_its_roles_and_offers_only_stated_presses(tmp_path):
    def outlets(document) -> None:
        for n in (1, 2):
            for name in ("Sts", "Cmd"):
                record = {"id": f"XR01-PS1:Out{n}-{name}", "on": {"device": "XR01:PS1"}}
                if name == "Cmd":  # writable: the fallback must not offer it again
                    record["role"] = "setpoint"
                document["channels"].append(record)

    roles = (
        "PowerSupply,state,*:Out[0-9]-Sts,device,lamp,,,,,stated,S,1,l\n"
        "PowerSupply,cmd,*:Out[0-9]-Cmd,device,choice,,,1,On,stated,S,1,l\n"
        "PowerSupply,cmd,*:Out[0-9]-Cmd,device,choice,,,3,Cycle,inferred,S,1,l\n"
        "PowerSupply,cmd,*:Out[0-9]-Cmd,device,choice,,,,Off,stated,S,1,l\n"  # no value
    )
    graph = _fronted(
        tmp_path, outlets,
        "supply,front,outputs,1,Out{n},socket,,*:Out{n}-Sts,powered_by,in,{n},,S\n", roles,
    )  # fmt: skip
    view = device_card(graph, "XR01:PS1")
    front = _front_of(view)
    one, two = front.groups[0].connectors
    assert [(c.label, c.channel, c.press_value, c.inferred) for c in two.controls] == [
        ("On", "XR01-PS1:Out2-Cmd", 1.0, False),
        ("Cycle", "XR01-PS1:Out2-Cmd", 3.0, True),
    ]  # Off states no value: no press
    assert front.writes == (
        "each button writes its own connector's channel (XR01-PS1:Out1-Cmd … XR01-PS1:Out2-Cmd) "
        "once: On = 1, Cycle = 3 (value inferred, not stated in the database)"
    )
    operate = [i for s in view.sections for i in s.items if type(i).__name__ == "Operate"]
    assert all(
        c.channel not in {"XR01-PS1:Out1-Cmd", "XR01-PS1:Out2-Cmd"}
        for o in operate for g in o.groups for c in g.controls
    )  # drawn on the front, not twice  # fmt: skip
    doc = stub_render(view, graph)
    presses = [n for n in _by_type(doc, "pv-button") if n["props"]["pv"].startswith("XR01-PS1:Out")]
    assert sorted(n["props"]["pressValue"] for n in presses) == [1, 1, 3, 3]


FRONTS_ROLES_HEADER = FRONTS_HEADER.rstrip("\n") + ",roles\n"

_GROUP_ROLES = (
    "PowerSupply,state,*:Out[0-9]-Sts,device,lamp,,,,,stated,S,1,l\n"
    "PowerSupply,current,*:Ph[0-9]-Cur,device,value,,,,,stated,S,1,l\n"
)


def _two_groups(tmp_path, phases: bool, roles_column: bool = True):
    """A supply whose outlets 1-2 and phases 1-2 share numbers: two groups of one front."""

    def channels(document) -> None:
        names = ["Out1-Sts", "Out2-Sts"] + (["Ph1-Cur", "Ph2-Cur"] if phases else [])
        for name in names:
            document["channels"].append({"id": f"XR01-PS1:{name}", "on": {"device": "XR01:PS1"}})

    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    _fronted(tmp_path, channels, roles=_GROUP_ROLES)  # its tables, then this front's rows
    tables = tmp_path / "panels"
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,front,Outputs,,front,,state;current,,S\n"
    )
    if roles_column:
        rows = (
            "supply,front,outlets,1,Out{n},socket,,*:Out{n}-Sts,powered_by,in,{n},,S,state\n"
            "supply,front,phases,2,Ph{n},power,,,,,,,S,current\n"  # nothing on it is looked for
        )
        (tables / "fronts.csv").write_text(FRONTS_ROLES_HEADER + rows)
    else:
        rows = (
            "supply,front,outlets,1,Out{n},socket,,*:Out{n}-Sts,powered_by,in,{n},,S\n"
            "supply,front,phases,2,Ph{n},socket,2,,powered_by,in,{n},,S\n"
        )
        (tables / "fronts.csv").write_text(FRONTS_HEADER + rows)
    return load_graph(write_fixture_facility(tmp_path / "g.json", channels), tables)


def _carried(front) -> dict[str, list[str]]:
    return {c.label: sorted(r.channel for r in c.rows) for g in front.groups for c in g.connectors}


def test_two_groups_sharing_numbers_take_only_the_roles_their_rows_list(tmp_path):
    graph = _two_groups(tmp_path, phases=True)
    front = _front_of(device_card(graph, "XR01:PS1"))
    assert [g.title for g in front.groups] == ["outlets", "phases"]
    assert _carried(front) == {
        "Out1": [],  # its state is its lamp
        "Out2": [],
        "Ph1": ["XR01-PS1:Ph1-Cur"],  # numbered from its role's channels; no lamp
        "Ph2": ["XR01-PS1:Ph2-Cur"],
    }
    assert [c.lamp for c in front.groups[0].connectors] == [
        "XR01-PS1:Out1-Sts",
        "XR01-PS1:Out2-Sts",
    ]
    assert all(c.lamp is None for c in front.groups[1].connectors)


def test_without_the_roles_column_every_section_role_binds_by_number_as_before(tmp_path):
    graph = _two_groups(tmp_path / "old", phases=True, roles_column=False)
    carried = _carried(_front_of(device_card(graph, "XR01:PS1")))
    assert carried["Ph1"] == ["XR01-PS1:Out1-Sts", "XR01-PS1:Ph1-Cur"]  # the old reading


def test_a_group_numbered_from_a_role_the_device_does_not_serve_is_left_out(tmp_path):
    graph = _two_groups(tmp_path, phases=False)
    front = _front_of(device_card(graph, "XR01:PS1"))
    assert [g.title for g in front.groups] == ["outlets"]  # no phase channel: no connector


def test_a_literal_connector_takes_its_listed_role_s_channel_carrying_its_name(tmp_path):
    def channels(document) -> None:
        for name in ("PSUA-Sts", "PSUB-Sts"):
            document["channels"].append({"id": f"XR01-PS1:{name}", "on": {"device": "XR01:PS1"}})

    roles = "PowerSupply,state,*:PSU?-Sts,device,lamp,,,,,stated,S,1,l\n"
    _fronted(tmp_path, channels, roles=roles)
    tables = tmp_path / "panels"
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,front,Outputs,,front,,state,,S\n"
    )
    (tables / "fronts.csv").write_text(
        FRONTS_ROLES_HEADER
        + "supply,rear,supplies,1,PSUA,power,,,powered_by,in,,PSU A,S,state\n"
        + "supply,rear,supplies,2,PSUB,power,,,powered_by,in,,PSU B,S,state\n"
    )
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "g.json", channels), tables)
    assert _carried(_front_of(device_card(graph, "XR01:PS1"))) == {
        "PSU A": ["XR01-PS1:PSUA-Sts"],
        "PSU B": ["XR01-PS1:PSUB-Sts"],
    }


def _named_outlets(document) -> None:
    for n in range(1, 9):
        document["channels"].append(
            {"id": f"XR01-PS1:Out{n}-Name", "on": {"device": "XR01:PS1"}, "role": "setpoint"}
        )


_NAME_ROLE = "PowerSupply,name,*:Out[0-9]-Name,device,entry,,,,Name,stated,S,1,l\n"


def test_many_entries_of_one_role_are_a_grid_whose_line_names_no_channel(tmp_path):
    from osprey.facility.panels import layout

    graph = _fronted(tmp_path, _named_outlets, roles=_NAME_ROLE)  # name is no front role
    view = device_card(graph, "XR01:PS1")
    operate = [i for s in view.sections for i in s.items if type(i).__name__ == "Operate"]
    (group,) = [g for o in operate for g in o.groups if g.kind == "entry"]
    assert len(group.controls) == 8
    assert group.writes.endswith("Apply writes each entry's own channel")
    doc = stub_render(view, graph)
    assert _text(doc, "NAME")  # the role's title over the grid
    fields = _by_type(doc, "pv-text-input")
    assert len({f["geometry"]["x"] for f in fields}) == layout.ENTRY_COLUMNS  # in columns
    (hover,) = [
        n
        for n in _by_type(doc, "rectangle")
        if "XR01-PS1:Out8-Name, " in n["props"].get("tooltip", "") + ", "
        and "XR01-PS1:Out1-Name" in n["props"].get("tooltip", "")
    ]
    assert "XR01-PS1:Out1-Name" in hover["props"]["tooltip"]


def test_a_connector_s_typed_value_is_set_under_the_front_named_by_its_connector(tmp_path):
    graph = _fronted(
        tmp_path, _named_outlets,
        "supply,front,outputs,1,Out{n},socket,,*:Out{n}-Name,powered_by,in,{n},Out {n},S\n",
        _NAME_ROLE,
    )  # fmt: skip
    (tmp_path / "panels" / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,front,Outputs,,front,,name,,S\n"
    )
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    graph = load_graph(
        write_fixture_facility(tmp_path / "g.json", _named_outlets), tmp_path / "panels"
    )
    view = device_card(graph, "XR01:PS1")
    front = _front_of(view)
    (group,) = front.entries
    assert [c.label for c in group.controls][:2] == ["Out 1 · Name", "Out 2 · Name"]
    operate = [i for s in view.sections for i in s.items if type(i).__name__ == "Operate"]
    assert not [
        c for o in operate for g in o.groups for c in g.controls if c.channel.endswith("-Name")
    ]
    doc = stub_render(view, graph)
    assert len(_by_type(doc, "pv-text-input")) == 8  # each once


def test_tables_without_fronts_still_build(panel_graph):
    assert panel_graph.fronts("supply") == []


def _sectioned(tmp_path, sections: str, roles: str = "", change=None):
    """A pump recipe with the given card sections."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nPump,,,pump,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n" + sections
    )
    (tables / "channel_roles.csv").write_text(ROLES_HEADER + roles)
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _item(view, kind):
    return next(i for s in view.sections for i in s.items if type(i).__name__ == kind)


def test_a_ring_strip_draws_its_group_in_number_order_when_a_position_is_missing(tmp_path):
    graph = _sectioned(
        tmp_path,
        "pump,1,ring-strip,The pumps here,,same place,,pressure,,S\n",
        "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n",
    )
    ring = _item(device_card(graph, "XR01:IP1"), "RingStrip")
    assert [m.thing for m in ring.members] == ["XR01:IP1", "XR01:IP2"]
    first, second = ring.members
    assert first.focus and first.link is None and second.link.target.id == "XR01:IP2"
    assert [r.channel for r in first.rows] == ["XR01-IP1:Pressure-RB"]
    if ring.ordered == "number":
        assert ring.caption.startswith("in number order: positions not stated")
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert [n for n in nodes(doc) if n["props"].get("tooltip") == "open XR01:IP2"]


def test_a_ring_strip_of_placed_members_runs_by_place_under_a_count_of_ok_lamps(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    def permit(document) -> None:
        document["channels"].append({"id": "XR01-IP2:Permit-Sts", "on": {"device": "XR01:IP2"}})
        document["channels"].append({"id": "XR01-IP1:Permit-Sts", "on": {"device": "XR01:IP1"}})

    _sectioned(
        tmp_path,
        "pump,1,ring-strip,The pumps here,,same place,,state,,S\n",
        "Pump,state,*:Permit-Sts,device,lamp,,,,,stated,S,1,l\n",
    )
    tables = tmp_path / "panels"
    with (tables / "health.csv").open("a") as handle:
        for pump in ("XR01-IP1", "XR01-IP2"):
            handle.write(f"{pump}:Permit-Sts,1,normal,screen,stated,s,1,l\n")
            handle.write(f"{pump}:Permit-Sts,0,bad,screen,stated,s,1,l\n")
    graph = load_graph(write_fixture_facility(tmp_path / "g.json", permit), tables)
    ring = _item(device_card(graph, "XR01:IP1"), "RingStrip")
    assert ring.places == ()  # one place: no rows by place
    assert ring.counted == "giving state now"
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert _text(doc, "giving state now:")
    (count,) = [
        n
        for n in _by_type(doc, "text-label")
        if any(r["name"] == "giving state now count" for r in n.get("rules", ()))
    ]
    (formula,) = [
        f for f in doc["formulas"] if "formula://" + f["name"] == count["rules"][0]["formula"]
    ]
    assert set(formula["bindings"].values()) == {"XR01-IP1:Permit-Sts", "XR01-IP2:Permit-Sts"}
    assert "1 if" in formula["expression"]  # ok counts one, by the lamp's own arms
    assert _text(doc, "of 2")


def test_ring_members_group_at_the_finest_level_with_few_places(panel_graph) -> None:
    from osprey.facility.panels.describe import _ring_places

    members = [panel_graph.thing(d, "device") for d in ("XR01:QF1", "XR01:QD1", "XR02:QF1")]
    held = _ring_places(panel_graph, members)
    assert len(set(held.values())) == 2  # one group per place, not one per member
    assert held["XR01:QF1"] == held["XR01:QD1"] != held["XR02:QF1"]


def test_a_ring_strip_of_a_relation_takes_the_devices_it_names(tmp_path):
    graph = _sectioned(tmp_path, "pump,1,ring-strip,Read out by,,read_out_by out,,,,S\n")
    ring = _item(device_card(graph, "XR01:IP1"), "RingStrip")
    assert [m.thing for m in ring.members] == ["XR01:CH2"]
    assert ring.members[0].rows == ()  # plays no role: an empty ring


def _inferred(*claims: str, rule: str | None = "joined by the document's name"):
    """Mark *claims* inferred, with *rule* (none when ``None``)."""

    def change(document) -> None:
        for claim in document["relations"]:
            if claim["id"] in claims:
                claim["basis"] = "inferred"
                if rule is not None:
                    claim["rule"] = rule

    return change


def _both(*changes):
    def change(document) -> None:
        for one in changes:
            one(document)

    return change


def test_an_inferred_relation_says_so_on_a_hop_a_connector_and_a_member(tmp_path):
    from osprey.facility.panels.describe import RULE_NOT_STATED, inferred_rule

    # a hop
    graph = _chained(tmp_path / "a", _both(_ported, _inferred("xr01-ip1-readout")))
    view = device_card(graph, "XR01:IP1")
    _device, readout, *_rest = _chain_of(view).blocks
    assert readout.inferred == "joined by the document's name"
    doc = stub_render(view, graph)
    assert _text(doc, "inferred")
    assert [n for n in nodes(doc) if n["props"].get("tooltip") == "joined by the document's name"]
    # a connector
    graph = _fronted(
        tmp_path / "b",
        _both(_ports(("xr01-qf1-power", "1")), _inferred("xr01-qf1-power", rule=None)),
    )
    (end,) = _front_of(device_card(graph, "XR01:PS1")).groups[0].connectors[0].sides[0]
    assert end.inferred == RULE_NOT_STATED
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert [n for n in nodes(doc) if "inferred" in n["props"].get("text", "")]
    # a member
    graph = _sectioned(
        tmp_path / "c",
        "pump,1,ring-strip,Read out by,,read_out_by out,,,,S\n",
        change=_inferred("xr01-ip1-readout"),
    )
    ring = _item(device_card(graph, "XR01:IP1"), "RingStrip")
    (member,) = ring.members  # every member inferred: said once, in the caption
    assert member.inferred is None
    assert ring.caption.endswith("all 1 inferred: joined by the document's name")
    # a stated one says nothing
    assert inferred_rule(graph.neighbours("XR01:QF1", "device").get("powered_by", ())) is None


def test_in_a_mixed_ring_strip_only_the_inferred_member_says_so(tmp_path):
    def second(document) -> None:
        document["relations"].append(
            {
                "id": "xr01-ip1-readout-2",
                "type": "read_out_by",
                "subject": {"device": "XR01:IP1"},
                "object": {"device": "XR01:CH1"},
                "standing": "accepted",
            }
        )

    graph = _sectioned(
        tmp_path,
        "pump,1,ring-strip,Read out by,,read_out_by out,,,,S\n",
        change=_both(second, _inferred("xr01-ip1-readout")),
    )
    ring = _item(device_card(graph, "XR01:IP1"), "RingStrip")
    marked = {m.thing: m.inferred for m in ring.members}
    assert marked == {"XR01:CH1": None, "XR01:CH2": "joined by the document's name"}
    assert "inferred" not in (ring.caption or "")


def test_when_most_members_are_inferred_only_the_stated_one_is_marked(tmp_path):
    def more(document) -> None:
        for other, claim in (("XR01:CH1", "readout-1"), ("XR01:CH3", "readout-3")):
            document["relations"].append(
                {
                    "id": claim,
                    "type": "read_out_by",
                    "subject": {"device": "XR01:IP1"},
                    "object": {"device": other},
                    "standing": "accepted",
                }
            )

    graph = _sectioned(
        tmp_path,
        "pump,1,ring-strip,Read out by,,read_out_by out,,,,S\n",
        change=_both(more, _inferred("xr01-ip1-readout", "readout-3")),
    )
    ring = _item(device_card(graph, "XR01:IP1"), "RingStrip")
    assert {m.thing: (m.inferred, m.stated) for m in ring.members} == {
        "XR01:CH1": (None, True),
        "XR01:CH2": (None, False),
        "XR01:CH3": (None, False),
    }
    assert ring.caption.endswith("inferred unless marked stated: joined by the document's name")
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert _text(doc, "stated")


def test_a_group_wrapping_onto_a_second_line_names_it_continued(tmp_path):
    graph = _fronted(
        tmp_path, None, "supply,front,outputs,1,Out{n},power,40,,powered_by,in,{n},Out {n},S\n"
    )
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    titles = [n["props"]["text"] for n in _by_type(doc, "text-label")]
    assert "outputs" in titles and "outputs (continued)" in titles


def test_inside_a_place_a_member_shows_its_short_name_and_two_alike_keep_what_differs():
    from osprey.facility.panels.describe import _short_labels
    from osprey.facility.panels.model import Thing

    members = [Thing(i, "device", None, None) for i in ("XR05A:G1", "XR05A:P1", "XR05B:P1")]
    assert _short_labels(members) == {"XR05A:G1": "G1", "XR05A:P1": "A:P1", "XR05B:P1": "B:P1"}


def test_number_order_compares_numbers_as_numbers() -> None:
    from osprey.facility.panels.describe import _number_key

    names = ["XR03:T10", "XR03:T9", "XR03:T30"]
    assert sorted(names, key=_number_key) == ["XR03:T9", "XR03:T10", "XR03:T30"]


def test_a_ladder_marks_the_set_points_in_order_and_names_those_not_served(tmp_path):
    def limits(document) -> None:
        for name in ("Lo-SP", "Hi-SP"):
            document["channels"].append({"id": f"XR01-IP1:{name}", "on": {"device": "XR01:IP1"}})

    graph = _sectioned(
        tmp_path,
        "pump,1,ladder,Its limits,,,,pressure;threshold-lolo;threshold-lo;threshold-hi,the caption,S\n",
        "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
        "Pump,threshold-lo,*:Lo-SP,device,value,,,,,stated,S,1,l\n"
        "Pump,threshold-hi,*:Hi-SP,device,value,,,,,stated,S,1,l\n",
        limits,
    )
    view = device_card(graph, "XR01:IP1")
    ladder = _item(view, "LimitLadder")
    assert ladder.value == "XR01-IP1:Pressure-RB"
    assert ladder.marks == (("Lo", "XR01-IP1:Lo-SP"), ("Hi", "XR01-IP1:Hi-SP"))
    assert ladder.left_out == ("Lolo",)
    doc = stub_render(view, graph)
    (dot,) = [n for n in _by_type(doc, "ellipse") if n.get("rules")]
    x_rule = next(r for r in dot["rules"] if r["targetProperty"] == "x")
    (formula,) = [f for f in doc["formulas"] if "formula://" + f["name"] == x_rule["formula"]]
    assert set(formula["bindings"].values()) == {
        "XR01-IP1:Pressure-RB", "XR01-IP1:Lo-SP", "XR01-IP1:Hi-SP",
    }  # fmt: skip
    assert dot["props"]["fill"] is False  # hollow until the value is known
    assert _text(doc, "set points not served, left out: Lolo")


def test_entries_of_one_role_are_each_named_by_their_channel(tmp_path):
    from osprey.facility.panels.describe import _operate
    from osprey.facility.panels.model import RoleHit
    from osprey.facility.panels.tables import ChannelRole

    role = ChannelRole(
        "PowerSupply", "set-point", "*", "device", "entry", None, None, None, None, "stated",
        "S", "1", "l",
    )  # fmt: skip
    graph = _fronted(tmp_path)
    thing = graph.thing("XR01:PS1", "device")
    hits = [RoleHit(role, f"XR01:PS1:{name}", "XR01:PS1") for name in ("Current-SP", "Voltage-SP")]
    operate, _used, _left = _operate(graph, thing, hits, [])
    (group,) = operate.groups  # one word ("Set point") for two channels would say nothing
    assert [c.label for c in group.controls] == ["Current-SP", "Voltage-SP"]


def _switch(tmp_path, signal: str | None = "*:TSW1-Sts"):
    """A thermal switch XR01:QF1:TSW1, part of the quadrupole XR01:QF1, whose drawing
    holds one switch row reading *signal*."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import add_part, copy_tables, write_fixture_facility

    def part(document) -> None:
        add_part(document, "XR01:QF1:TSW1", "XR01:QF1", "XR01:QF1:TSW1-Sts")
        document["devices"][-1]["class"] = "Switch"
        document["relations"].append(
            {"id": "tsw-mps", "type": "interlocked_by", "subject": {"device": "XR01:QF1:TSW1"},
             "object": {"device": "XR01:CH3"}}
        )  # fmt: skip

    tables = copy_tables(tmp_path / "panels")
    lines = (tables / "pictures.csv").read_text().splitlines()
    header = lines[0].split(",") + ["w_mm", "h_mm", "angle_deg", "signal", "under"]
    rows = []
    for line in lines[1:]:
        cells = dict(zip(header, line.split(","), strict=False))
        if cells.get("kind") == "thermal_switch":
            cells["signal"] = signal or ""
        rows.append(",".join(cells.get(h, "") for h in header))
    (tables / "pictures.csv").write_text("\n".join([",".join(header), *rows]) + "\n")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nSwitch,,,switch,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "switch,1,section,On the magnet,part_of,,,,,S\n"
        "switch,2,chain,Interlock,,interlock,,,,S\n"
    )
    (tables / "chains.csv").write_text(
        "recipe,chain,step,from_step,relation,direction,to,many,port_from,title,roles,source\n"
        "switch,interlock,1,0,interlocked_by,out,device,,,MPS,,S\n"
        "switch,interlock,2,1,powered_by,out,device,,,MPS power,,S\n"
    )
    return load_graph(write_fixture_facility(tmp_path / "f.json", part), tables)


def test_a_part_is_drawn_in_focus_on_its_owner_with_what_is_and_is_not_stated(tmp_path):
    graph = _switch(tmp_path)
    view = device_card(graph, "XR01:QF1:TSW1")
    pictures = [i for s in view.sections for i in s.items if type(i).__name__ == "Picture"]
    assert pictures and pictures[0].device_class == "Quadrupole"
    assert pictures[0].focus == ("thermal switch 1",)
    facts = _item(view, "PartFacts")
    assert [f.label for f in facts.stated] == ["protected by: XR01:CH3", "part of: XR01:QF1"]
    assert facts.not_stated == ("MPS power: not stated",)
    doc = stub_render(view, graph)
    assert [n for n in _by_type(doc, "ellipse") if n["props"].get("tooltip") == "thermal switch 1"]
    texts = [n["props"]["text"] for n in _by_type(doc, "text-label")]
    assert "NOT STATED" in texts and "MPS power: not stated" in texts


def test_a_part_whose_place_on_the_drawing_is_not_stated_says_so(tmp_path):
    graph = _switch(tmp_path, signal=None)
    view = device_card(graph, "XR01:QF1:TSW1")
    facts = _item(view, "PartFacts")
    assert "where it sits on XR01:QF1's drawing" in facts.not_stated


def _curved(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\nQuadrupole,,,magnet,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "magnet,1,curve,Excitation curve,,excitation,,,,S\n"
    )
    return load_graph(write_fixture_facility(tmp_path / "f.json"), tables)


def test_a_curve_draws_the_points_of_the_claim_in_force(tmp_path):
    graph = _curved(tmp_path)
    view = device_card(graph, "XR01:QF1")
    curve = _item(view, "CurvePlot")
    assert curve.points == ((0.0, -0.02376), (73.33, -4.9198))
    assert (curve.x_unit, curve.y_unit, curve.source) == ("A", "T", "XR-MAG-TEST A.2")
    assert curve.note.startswith("operating point not drawn")
    doc = stub_render(view, graph)
    assert (
        len([n for n in _by_type(doc, "ellipse") if n["props"].get("tooltip") == "73.33, -4.9198"])
        == 1
    )
    assert _text(doc, "2 points from XR-MAG-TEST A.2")


def test_a_device_with_no_curve_says_none_is_stated(tmp_path):
    graph = _curved(tmp_path)
    curve = _item(device_card(graph, "XR01:QD1"), "CurvePlot")
    assert curve.points == () and curve.note == "no excitation curve stated for this device"


CARDS_HEADER = (
    "action,class,under,devices,attribute,when,owner_relation,owner_direction,owner_to,"
    "reason,source\n"
)


def _carded(tmp_path, rows: str | None, attributes: dict | None = None):
    """The fixture with two parts in no place of their own: ``XR01:CARD1`` (part of the
    racked ``XR01:CH1``, with a rule channel and *attributes*) and ``XR01:CARD2``
    (only cabled to it, no channel); *rows* is the ``cards.csv`` body (``None``: no
    file)."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import (
        PART_RULE,
        add_part,
        copy_tables,
        write_fixture_facility,
    )

    def change(document) -> None:
        add_part(document, "XR01:CARD1", "XR01:CH1", "XR01-CARD1:Fault-Sts")
        add_part(document, "XR01:CARD2", "XR01:CH1")
        document["relations"][-1]["subject"]["device"] = "XR01:CARD2"
        document["relations"][-1]["type"] = "cabled_to"  # not part of anything
        for device in document["devices"]:
            if device["id"] == "XR01:CARD1":
                device["attributes"] = attributes or {}

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a") as handle:
        handle.write(PART_RULE)
    (tables / "cards.csv").write_text(CARDS_HEADER + (rows or ""))
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def test_by_default_a_device_gets_a_card_only_with_a_channel_or_a_drawing(tmp_path):
    from osprey.facility.panels.describe import (
        _index,
        _ref,
        card_counts,
        card_rule_counts,
        drawn_on,
        targets,
    )
    from osprey.facility.panels.view import Fact, Target

    graph = _carded(tmp_path, None)
    found = targets(graph)
    assert Target("card", "device", "XR01:CARD1") in found  # a channel of its own
    assert Target("card", "device", "XR01:CH1") in found  # its part's channel walks to it
    # a pose and relations alone are no reason for a card
    for bare in ("XR01:CARD2", "XR02:SHIELD1", "XR02:BLOCK1"):
        assert Target("card", "device", bare) not in found
    assert graph.card_rule("XR01:CARD2") is None and card_rule_counts(graph) == []
    cardless, ownerless, by_default = card_counts(graph)
    assert cardless == ownerless == by_default >= 3
    # no screen, no index entry, and a related card names it without a link
    assert "XR01:CARD2" not in {thing for thing, _target in drawn_on(graph)}
    assert isinstance(_ref(graph, _index(graph), graph.thing("XR01:CARD2", "device")), Fact)


def test_a_drop_row_with_an_owner_draws_the_device_on_its_owners_card(tmp_path):
    import json

    from osprey.facility.panels.describe import _index, _ref, card_rule_counts, targets
    from osprey.facility.panels.emit import write
    from osprey.facility.panels.view import Target

    graph = _carded(tmp_path, "drop,ElectronicsModule,,,,,part_of,out,device,in its chassis,S\n")
    assert Target("card", "device", "XR01:CARD1") not in targets(graph)
    assert [n for _row, n in card_rule_counts(graph)] == [2]  # CARD2 too, with no owner
    out = tmp_path / "out"
    write(graph, out)
    index = json.loads((out / "index.json").read_text())
    (entry,) = index["drawn_on"]
    assert entry["thing"] == "XR01:CARD1" and entry["file"].endswith("card-xr01-ch1.screen.json")
    link = _ref(graph, _index(graph), graph.thing("XR01:CARD1", "device"))
    assert link.target == Target("card", "device", "XR01:CH1") and link.label == "XR01:CARD1"


def test_a_keep_row_gives_a_card_the_default_would_not(tmp_path):
    from osprey.facility.panels.describe import card_counts, targets
    from osprey.facility.panels.view import Target

    bare = card_counts(_carded(tmp_path / "a", None))
    graph = _carded(tmp_path / "b", "keep,*,,XR01:CARD2,,,,,,a passive part worth a card,S\n")
    assert Target("card", "device", "XR01:CARD2") in targets(graph)
    assert card_counts(graph) == (bare[0] - 1, bare[1] - 1, bare[2] - 1)


def test_a_row_with_an_attribute_selects_only_the_devices_that_carry_it(tmp_path):
    from osprey.facility.panels.describe import card_rule_counts, targets
    from osprey.facility.panels.view import Target

    graph = _carded(
        tmp_path,
        "drop,ElectronicsModule,,,kind=carrier,,,,,a carrier card,S\n"
        "keep,ElectronicsModule,,,,,,,,the others keep theirs,S\n",
        {"kind": "carrier", "slots": [1, 2]},
    )
    found = targets(graph)
    assert Target("card", "device", "XR01:CARD1") not in found
    assert Target("card", "device", "XR01:CARD2") in found  # same class, no such attribute
    (first, n_first), (_second, n_second) = card_rule_counts(graph)
    assert first.attribute == ("kind", "carrier") and (n_first, n_second) == (1, 1)
    assert ("slots", "1") in graph.thing("XR01:CARD1", "device").attributes


def test_a_row_under_a_place_selects_a_part_by_its_wholes_place(tmp_path):
    from osprey.facility.panels.describe import card_rule_counts

    graph = _carded(tmp_path, "drop,*,XR,XR01:CARD*,,,,,,parts in the ring,S\n")
    assert graph.card_rule("XR01:CARD1") is not None  # part of a racked chassis in XR
    assert graph.card_rule("XR01:CARD2") is None  # in no place
    assert [n for _row, n in card_rule_counts(graph)] == [1]


def test_when_conditions_select_by_channel_and_by_drawing(tmp_path):
    graph = _carded(tmp_path, "keep,*,,XR01:*,,no-channel;no-drawing,,,,bare devices,S\n")
    kept = {t.id for t in graph.things() if t.kind == "device" and graph.card_rule(t.id)}
    assert "XR01:CARD2" in kept and "XR01:CARD1" not in kept
    assert all(not graph.card_channels(d) and not graph.card_drawing(d) for d in kept)


@pytest.mark.parametrize(
    ("row", "said"),
    [
        ("drop,ElectronicsModule,,,,,part_of,,,r,S\n", "together"),
        ("drop,ElectronicsModule,,,,,,,device,r,S\n", "owner_to needs owner_relation"),
        ("keep,ElectronicsModule,,,,,part_of,out,,r,S\n", "a keep row names no owner"),
        ("hide,ElectronicsModule,,,,,,,,r,S\n", "action is 'hide'"),
        (",ElectronicsModule,,,,,,,,r,S\n", "action"),
        ("drop,ElectronicsModule,,,,no-place,,,,r,S\n", "when is 'no-place'"),
        ("drop,ElectronicsModule,,,,no-channel;no-channel,,,,r,S\n", "when is"),
        ("drop,ElectronicsModule,,,carrier,,,,,r,S\n", "attribute is 'carrier', not name=value"),
    ],
)
def test_a_cards_row_that_cannot_be_read_says_why(tmp_path, row, said):
    from osprey.facility.panels.tables import CARDS, PanelTableError
    from tests.facility.panels._fixture import copy_tables, read_fixture_tables

    tables = copy_tables(tmp_path / "panels")
    (tables / CARDS).write_text(CARDS_HEADER + row)
    with pytest.raises(PanelTableError, match=said):
        read_fixture_tables(tables)


def test_choice_segments_that_share_a_word_carry_their_value() -> None:
    from osprey.facility.panels.describe import _distinct
    from osprey.facility.panels.view import Control

    both = [Control("Reboot", "X:REBOOT", 0.0), Control("Reboot", "X:REBOOT", 1.0)]
    assert [c.label for c in _distinct(both)] == ["Reboot = 0", "Reboot = 1"]
    apart = [Control("Off", "X:C", 0.0), Control("On", "X:C", 1.0)]
    assert _distinct(apart) == apart


def test_a_segment_carrying_its_value_is_not_given_it_twice_in_the_writes_line() -> None:
    from osprey.facility.panels.describe import _distinct, _writes
    from osprey.facility.panels.view import Control

    both = _distinct([Control("Reboot", "X:REBOOT", 0.0), Control("Reboot", "X:REBOOT", 1.0)])
    assert (
        _writes("choice", both, "X") == "writes REBOOT: Reboot = 0, Reboot = 1; stays until changed"
    )


def test_a_front_row_with_no_far_relation_is_a_reading_with_no_far_end(tmp_path):
    from osprey.facility.panels.describe import FRONT_READING, front_state, front_states

    graph = _two_groups(tmp_path, phases=True)  # its phases row names no far relation
    front = _front_of(device_card(graph, "XR01:PS1"))
    phases = front.groups[1].connectors
    assert [c.sides for c in phases] == [(), ()]
    assert {front_state(c.sides) for c in phases} == {FRONT_READING}
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    dashes = [
        n
        for n in _by_type(doc, "text-label")
        if n["props"]["text"] == layout._NOTHING and not n.get("rules")  # a figure's is ruled
    ]
    assert len(dashes) == 2  # the two outlets with nothing stated, not the phases
    (states,) = front_states(graph, ["XR01:PS1"]).values()
    assert states[FRONT_READING] == 2
    assert states["devices: none"] == 1  # the outlets have nothing on them; readings aside


def test_a_device_whose_front_is_readings_only_is_counted_so(tmp_path):
    from osprey.facility.panels.describe import FRONT_READING, front_states

    graph = _literal(tmp_path, ("PSUA-Sts",))  # no row names a far relation
    (states,) = front_states(graph, ["XR01:PS1"]).values()
    assert states[FRONT_READING] == 3  # PSU A, Out 1, Out 2
    assert states["devices: readings only"] == 1
    assert states["devices: none"] == 0


def _literal(tmp_path, served: tuple[str, ...]):
    def channels(document) -> None:
        for name in served:
            document["channels"].append({"id": f"XR01-PS1:{name}", "on": {"device": "XR01:PS1"}})

    roles = "PowerSupply,state,*:PSU?-Sts,device,lamp,,,,,stated,S,1,l\n"
    _fronted(tmp_path, channels, roles=roles)
    tables = tmp_path / "panels"
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,front,Outputs,,front,,state,,S\n"
    )
    (tables / "fronts.csv").write_text(
        FRONTS_ROLES_HEADER
        + "supply,rear,supplies,1,PSUA,power,,,,,,PSU A,S,state\n"
        + "supply,rear,supplies,2,PSUB,power,,,,,,PSU B,S,state\n"
        + "supply,front,outputs,3,Out{n},power,2,,,,,Out {n},S,\n"
    )
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    return load_graph(write_fixture_facility(tmp_path / "g.json", channels), tables)


def test_a_literal_connector_nothing_of_the_device_binds_to_is_left_out(tmp_path):
    graph = _literal(tmp_path, ("PSUA-Sts",))
    front = _front_of(device_card(graph, "XR01:PS1"))
    supplies = next(g for g in front.groups if g.title == "supplies")
    assert [c.label for c in supplies.connectors] == ["PSU A"]  # PSU B: no channel binds


def test_a_group_all_of_whose_literal_connectors_are_left_out_is_dropped(tmp_path):
    graph = _literal(tmp_path, ())
    front = _front_of(device_card(graph, "XR01:PS1"))
    assert [g.title for g in front.groups] == ["outputs"]


def test_a_reading_group_is_a_table_with_its_rows_labelled_at_the_left(tmp_path):
    graph = _two_groups(tmp_path, phases=True)  # outlets on sockets, phases readings
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    sockets = [
        n
        for n in _by_type(doc, "rectangle")
        if n["geometry"]["width"] == layout.SOCKET_SIZE == n["geometry"]["height"]
    ]
    assert len(sockets) == 2  # the two outlets; a reading has no socket
    heads = [_text(doc, "Ph1"), _text(doc, "Ph2")]  # the phases' names as column heads
    (label,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "current"]
    assert label["geometry"]["x"] < min(h["geometry"]["x"] for h in heads)  # at the left
    cells = [n for n in nodes(doc) if "current: XR01-PS1:Ph" in n["props"].get("tooltip", "")]
    assert len({c["geometry"]["y"] for c in cells}) == 1  # one row across the columns


def test_a_front_s_values_under_its_sockets_are_one_band_labelled_with_role_and_unit(tmp_path):
    def currents(document) -> None:
        for n in (1, 2, 3):
            document["channels"].append(
                {"id": f"XR01-PS1:Out{n}-Cur", "on": {"device": "XR01:PS1"}}
            )

    roles = "PowerSupply,current,*:Out[0-9]-Cur,device,value,A,,,,stated,S,1,l\n"
    _fronted(tmp_path, currents, roles=roles)
    (tmp_path / "panels" / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "supply,1,front,Outputs,,front,,current,,S\n"
    )
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    graph = load_graph(write_fixture_facility(tmp_path / "g.json", currents), tmp_path / "panels")
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    (label,) = [n for n in _by_type(doc, "text-label") if n["props"]["text"] == "current (A)"]
    cells = [n for n in nodes(doc) if "current: XR01-PS1:Out" in n["props"].get("tooltip", "")]
    assert len(cells) == 3 and {c["geometry"]["y"] for c in cells} == {cells[0]["geometry"]["y"]}
    assert label["geometry"]["x"] < min(c["geometry"]["x"] for c in cells)


def test_a_front_row_apart_no_draws_only_clients_on_a_stated_port(tmp_path):
    _fronted(tmp_path, _ports(("xr01-qf1-power", "1")))  # its tables; then this row
    (tmp_path / "panels" / "fronts.csv").write_text(
        FRONTS_HEADER.rstrip("\n")
        + ",roles,apart\n"
        + "supply,front,outputs,1,Out{n},power,,,powered_by,in,{n},,S,,no\n"
    )
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import write_fixture_facility

    graph = load_graph(
        write_fixture_facility(tmp_path / "g.json", _ports(("xr01-qf1-power", "1"))),
        tmp_path / "panels",
    )
    front = _front_of(device_card(graph, "XR01:PS1"))
    (group,) = front.groups
    assert [e.thing for c in group.connectors for e in c.sides[0]] == ["XR01:QF1"]
    assert front.loose == ()  # XR01:QD1 is on no stated port: not drawn apart
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert not [n for n in _by_type(doc, "text-label") if "on no stated port" in n["props"]["text"]]


def _joined(tmp_path, inferred: bool):
    """Two pumps each read out by a chassis point, one join stated and the other
    inferred when *inferred*; the pump's open command is the point's channel."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def change(document) -> None:
        for n in (1, 2):
            document["channels"].append(
                {"id": f"XR01-CH2:P{n}-Open", "on": {"device": "XR01:CH2"}, "role": "setpoint"}
            )
        document["relations"].append(
            {
                "id": "xr01-ip2-readout",
                "type": "read_out_by",
                "subject": {"device": "XR01:IP2"},
                "object": {"device": "XR01:CH2", "port": "P2"},
                "standing": "accepted",
            }
        )
        for claim in document["relations"]:
            if claim["id"] == "xr01-ip1-readout":
                claim["object"]["port"] = "P1"
            if claim["id"] == "xr01-ip2-readout" and inferred:
                claim["basis"] = "inferred"
                claim["rule"] = "joined by the point's number"

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,open,*:{port}-Open,read_out_by,action,,,1,Open,stated,S,1,l\n"
    )
    return load_graph(write_fixture_facility(tmp_path / "f.json", change), tables)


def _writers(doc: dict) -> set[str]:
    """Every channel a widget of the screen can write."""
    kinds = ("pv-button", "pv-text-input", "pv-slider", "pv-combo")
    return {
        str(n["props"].get("pv") or n["props"].get("pvName") or "")
        for n in nodes(doc)
        if n["type"] in kinds
    }


def test_a_command_reached_through_an_inferred_join_is_never_offered(tmp_path):
    from osprey.facility.panels.describe import JOIN_WITHHELD, press_counts

    graph = _joined(tmp_path, inferred=True)
    stated = stub_render(device_card(graph, "XR01:IP1"), graph)
    assert any("XR01-CH2:P1-Open" in w for w in _writers(stated))  # its button
    withheld = stub_render(device_card(graph, "XR01:IP2"), graph)
    assert not any("XR01-CH2:P2-Open" in w for w in _writers(withheld))  # no writer at all
    (line,) = [n for n in _by_type(withheld, "text-label") if n["props"]["text"] == JOIN_WITHHELD]
    assert [
        n for n in nodes(withheld) if n["props"].get("tooltip") == "joined by the point's number"
    ]
    assert "XR01-CH2:P2-Open" in json.dumps(withheld)  # shown read only where it stood
    counts = press_counts(graph, ["XR01:IP1", "XR01:IP2"])
    assert (counts.stated, counts.withheld) == (1, 1)


def test_the_same_join_stated_offers_the_command(tmp_path):
    graph = _joined(tmp_path, inferred=False)
    doc = stub_render(device_card(graph, "XR01:IP2"), graph)
    assert any("XR01-CH2:P2-Open" in w for w in _writers(doc))


def test_a_band_of_rows_alike_in_place_is_labelled_with_the_words_they_end_in():
    from osprey.facility.panels.layout import _band_label, _bands
    from osprey.facility.panels.view import BlockRow, Connector

    def rows(*labels):
        return tuple(BlockRow(label, f"X:{label}", shown_as="value") for label in labels)

    columns = [
        (0.0, Connector("L1", "power", rows=rows("phase voltage", "phase current"))),
        (1.0, Connector("C1", "power", rows=rows("circuit voltage", "circuit current"))),
    ]
    first, second = _bands(columns)
    assert [r.label for _x, r in first] == ["phase voltage", "circuit voltage"]
    assert _band_label([r.label for _x, r in first]) == "voltage"
    assert _band_label([r.label for _x, r in second]) == "current"
    assert _band_label(["state"]) == "state"
    assert _band_label(["open", "shut"]) == "open / shut"
    unlike = [
        (0.0, Connector("L1", "power", rows=rows("phase voltage", "phase frequency"))),
        (1.0, Connector("C1", "power", rows=rows("circuit voltage", "circuit breaker"))),
    ]
    assert len(_bands(unlike)) == 4  # a place whose labels differ: by label, not by place


def test_a_text_set_record_is_never_an_entry_on_a_card(tmp_path):
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.describe import channel_list
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def named(document) -> None:
        document["channels"].append(
            {"id": "XR01-QF1:Name-SP", "on": {"device": "XR01:QF1"}, "role": "setpoint"}
        )

    tables = copy_tables(tmp_path / "panels")
    with (tables / "record_types.csv").open("a") as rows:
        rows.write(
            "XR01-QF1:Name-SP,stringout,text,stated,xr-ioc,0a1b2c3,magnetApp/Db/magnet.db:30\n"
        )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", named), tables)
    doc = stub_render(device_card(graph, "XR01:QF1"), graph)
    inputs = [n for n in nodes(doc) if n["type"] in ("pv-text-input", "pv-button")]
    assert any(n["props"].get("pv") == "XR01-QF1:Current-SP" for n in nodes(doc))  # still there
    assert not [n for n in inputs if "Name-SP" in str(n["props"].get("pv", ""))]
    assert "XR01-QF1:Name-SP" in json.dumps(stub_render(channel_list(graph, "XR01:QF1"), graph))


def test_a_device_whose_presses_all_stand_on_its_front_has_no_operate_block(tmp_path):
    def outlets(document) -> None:
        for n in (1, 2):
            document["channels"].append(
                {"id": f"XR01-PS1:Out{n}-Cmd", "on": {"device": "XR01:PS1"}, "role": "setpoint"}
            )

    roles = (
        "PowerSupply,internal,*:On-Cmd,device,value,,,,,stated,S,1,l\n"  # its one other press
        "PowerSupply,cmd,*:Out[0-9]-Cmd,device,choice,,,1,On,stated,S,1,l\n"
    )
    graph = _fronted(
        tmp_path, outlets,
        "supply,front,outputs,1,Out{n},socket,,*:Out{n}-Cmd,powered_by,in,{n},,S\n", roles,
    )  # fmt: skip
    view = device_card(graph, "XR01:PS1")
    assert [c.channel for g in _front_of(view).groups for k in g.connectors for c in k.controls]
    assert "operate" not in [s.title for s in view.sections]  # no "no operator actions"
    doc = stub_render(view, graph)
    assert "no operator actions stated" not in [n["props"].get("text") for n in nodes(doc)]


def test_presses_that_share_a_word_are_each_named_by_what_tells_their_channel_apart(tmp_path):
    from osprey.facility.panels.describe import _operate
    from osprey.facility.panels.model import RoleHit
    from osprey.facility.panels.tables import ChannelRole

    role = ChannelRole(
        "PowerSupply", "enable", "*", "device", "action", None, None, 1.0, "Enable", "stated",
        "S", "1", "l",
    )  # fmt: skip
    graph = _fronted(tmp_path)
    thing = graph.thing("XR01:PS1", "device")
    hits = [RoleHit(role, f"XR01:PS1:{name}:ENABLE", "XR01:PS1") for name in ("CH1", "CH2")]
    operate, _used, _left = _operate(graph, thing, hits, [])
    (group,) = operate.groups  # "Enable" twice would not say which is which
    assert [c.label for c in group.controls] == ["Enable CH1", "Enable CH2"]


def test_what_tells_names_apart_drops_the_segments_they_share():
    from osprey.facility.panels.describe import _apart

    assert _apart(["CH1:ENABLE", "CH2:ENABLE"]) == ["CH1", "CH2"]
    assert _apart(["R1:CH1_Flt", "R1:CH2_Flt"]) == ["CH1", "CH2"]
    assert _apart(["Current-SP", "Voltage-SP"]) == ["Current-SP", "Voltage-SP"]  # a suffix stays
    assert _apart(["A", "A"]) == ["A", "A"]  # nothing tells them apart: whole
    assert _apart(["ONE"]) == ["ONE"]


@pytest.mark.parametrize("under", [False, True])
def test_a_key_tile_s_note_goes_under_its_number_when_beside_it_would_cut_the_number(
    tmp_path, monkeypatch, under
):
    if under:  # a number that needs the tile's whole width leaves no room beside it
        monkeypatch.setattr(layout, "VALUE_WIDTH", 2 * layout.VALUE_WIDTH)
    roles = "Pump,state,*:State-Sts,device,word,,,,,stated,S,1,l\n"
    graph = _roles_graph(tmp_path, roles)  # no state word stated for it
    doc = stub_render(device_card(graph, "XR01:IP1"), graph)
    (note,) = [
        n for n in _by_type(doc, "text-label")
        if n["props"]["text"] == layout._MEANING_NOT_STATED and n["geometry"]["y"] < 200
    ]  # fmt: skip
    (number,) = [
        n for n in _by_type(doc, "pv-label")
        if n["props"]["pv"] == "XR01-IP1:State-Sts" and n["props"]["fontSize"] == layout.KEY_VALUE_SIZE
    ]  # fmt: skip
    n, t = number["geometry"], note["geometry"]
    if under:
        assert t["y"] >= n["y"] + n["height"]
    else:
        assert t["x"] >= n["x"] + n["width"] and t["y"] == n["y"]
        assert n["width"] >= layout.VALUE_WIDTH  # the number keeps its room


def test_a_chain_block_is_wide_enough_for_a_value_with_its_unit(panel_graph):
    from osprey.facility.panels.view import BlockRow, CardHead, Chain, ChainBlock, Target, View

    def width(unit: str | None) -> float:
        row = BlockRow("current", "XR01-QF1:Current-SP", shown_as="value", unit=unit)
        block = ChainBlock("magnet", None, (row,), focus=True)
        view = View(
            Target("card", "device", "XR01:QF1"), "XR01:QF1", (),
            (Section("head", (CardHead("Quadrupole · default recipe", None),)),
             Section("circuit", (Chain((block,)),))),
        )  # fmt: skip
        doc = stub_render(view, panel_graph)
        (box,) = [
            n for n in nodes(doc)
            if n["type"] == "rectangle" and n["props"].get("lineColor") == layout.symbols.FOCUS
        ]  # fmt: skip
        (field,) = [n for n in nodes(doc) if n["type"] == "pv-label"]
        return box["geometry"]["width"], field["geometry"]["width"]

    long_unit = "1/Meter^2"
    _box, field = width(long_unit)
    # the field holds the number and its unit whole
    assert field >= layout.text_width(f"-0.000 {long_unit}", layout.BLOCK_TEXT_SIZE)
    assert width(None)[0] == layout.BLOCK_LEAST  # the facility's unit, A, fits the least


def _family_supply_graph(tmp_path):
    """The fixture with one family supply whose two channels have the ring's first
    quadrupoles as endpoints."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def supply(document) -> None:
        document["devices"].append({"id": "XR:FAMILY-PS", "class": "FamilySupply"})
        members = ["XR01:QF1", "XR02:QF1"]
        for name, role in (
            ("XR:FAMILY-PS:Setpoint", "setpoint"),
            ("XR:FAMILY-PS:Reset", "setpoint"),
        ):
            document["channels"].append(
                {"id": name, "on": {"device": "XR:FAMILY-PS"}, "endpoint_of": members, "role": role}
            )

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "FamilySupply,set-point,*:Setpoint,device,entry,A,,,set point,stated,S,,\n"
        + "FamilySupply,reset,*:Reset,device,action,,,1,Reset,stated,S,,\n"
    )
    return load_graph(write_fixture_facility(tmp_path / "f.json", supply), tables)


def test_a_member_card_shows_its_shared_supply_read_only_and_links_to_it(tmp_path) -> None:
    from osprey.facility.panels.view import Link, Target, ValueRow

    graph = _family_supply_graph(tmp_path)
    view = device_card(graph, "XR01:QF1")
    (section,) = [s for s in view.sections if s.title == "shared supply XR:FAMILY-PS"]
    link, row = section.items
    assert link == Link("XR:FAMILY-PS", Target("card", "device", "XR:FAMILY-PS"))
    assert isinstance(row, ValueRow)
    assert (row.label, row.channel, row.writable) == (
        "set point (shared)",
        "XR:FAMILY-PS:Setpoint",
        False,
    )
    # the press stays on the supply's own card; no member writes the family's channel
    doc = stub_render(view, graph)
    assert not [
        n
        for n in nodes(doc)
        if n["type"] in ("pv-text-input", "pv-button") and "FAMILY-PS" in json.dumps(n)
    ]
    texts = [n["props"].get("text") for n in nodes(doc)]
    assert "set point (shared)" in texts


@pytest.mark.parametrize("write_under", [(), ("XR/XR01",)])
def test_a_shared_supply_on_a_member_card_links_to_no_unwritten_channels_list(
    tmp_path, write_under
) -> None:
    """A channel drawn under its supply is shown, not hidden: the member card offers
    the full list only when the list is a screen, in a whole write or a scoped one."""
    from osprey.facility.panels.emit import Draw, check, write
    from osprey.facility.panels.view import ChannelGrid

    graph = _family_supply_graph(tmp_path)
    view = device_card(graph, "XR01:QF1")
    (grid,) = [i for s in view.sections for i in s.items if isinstance(i, ChannelGrid)]
    assert (grid.hidden, grid.full) == (0, None)
    out = tmp_path / "out"
    draw = Draw(write_under=write_under)
    write(graph, out, draw=draw)
    assert not list(out.glob("channels/channels-XR01*"))
    problems = check(graph, out, draw=draw)
    assert not [p for p in problems if p.failing], problems


def _interlocked(document) -> None:
    """Both XR01 pumps interlocked by the chassis XR01:CH3 on inputs IN1 and IN2."""
    for n, pump in enumerate(("XR01:IP1", "XR01:IP2"), start=1):
        document["relations"].append(
            {
                "id": f"{pump.lower()}-interlock",
                "type": "interlocked_by",
                "subject": {"device": pump},
                "object": {"device": "XR01:CH3", "port": f"IN{n}"},
                "standing": "accepted",
            }
        )
        document["channels"].append({"id": f"XR01-CH3:In{n}-Sts", "on": {"device": "XR01:CH3"}})


def test_a_role_walks_any_relation_type_from_a_device_to_a_device(tmp_path):
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.tables import walked_relations
    from osprey.facility.panels.view import Chain
    from tests.facility.panels._fixture import (
        copy_tables,
        fixture_relation_types,
        write_fixture_facility,
    )

    walked = walked_relations(fixture_relation_types())
    assert {"read_out_by", "controlled_by", "interlocked_by"} <= set(walked)
    assert "located_in" not in walked  # its object is a place
    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,interlock,*:In{n}-Sts,interlocked_by,lamp,,,,,stated,S,1,l\n"
    )
    (tables / "card_recipes.csv").write_text(
        "class,under,devices,recipe,description,channels_cols,source\n"
        "ElectronicsChassis,,XR01:CH3,interlocks,,,S\n"
    )
    (tables / "card_sections.csv").write_text(
        "recipe,order,kind,title,focus,members,window_m,roles,caption,source\n"
        "interlocks,1,chain,Inputs,,inputs,,,,S\n"
    )
    (tables / "chains.csv").write_text(
        "recipe,chain,step,from_step,relation,direction,to,many,port_from,title,roles,source\n"
        "interlocks,inputs,1,0,interlocked_by,in,Pump,yes,,Interlocks,interlock,S\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _interlocked), tables)
    # the device's card shows its own point by the stated relation
    hits = [(h.role.role, h.channel, h.owner, h.port) for h in graph.roles("XR01:IP1")]
    assert hits[0] == ("interlock", "XR01-CH3:In1-Sts", "XR01:CH3", "IN1")
    assert all(h[2] == "XR01:IP1" for h in hits[1:])  # then its own signals, by default
    assert graph.card_channels("XR01:IP2")
    # the interlock's card carries the point of each device it interlocks
    chain = next(
        i for s in device_card(graph, "XR01:CH3").sections for i in s.items if isinstance(i, Chain)
    )
    _own, pumps = chain.blocks
    points = {b.title: [r.channel for r in b.rows] for b in (pumps, *pumps.stack)}
    assert points == {
        "Interlocks  XR01:IP1": ["XR01-CH3:In1-Sts"],
        "Interlocks  XR01:IP2": ["XR01-CH3:In2-Sts"],
    }


def test_a_role_rows_stated_label_names_its_tile_value_and_entries(tmp_path) -> None:
    """A stated label is the word on a lamp's and a value's tile, over the entries'
    group and in front of what tells two entries of one role apart."""
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import ControlGroup, KeyTile
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def sizes(document) -> None:
        for name in ("XR01-IP1:Size_A", "XR01-IP1:Size_B"):
            document["channels"].append(
                {"id": name, "on": {"device": "XR01:IP1"}, "role": "setpoint"}
            )

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "Pump,pump-state,*:State-Sts,device,lamp,,,,Pump OK,stated,S,,\n"
        + "Pump,pressure,*:Pressure-RB,device,value,,,,Vacuum,stated,S,,\n"
        + "Pump,pump-size,*:Size_*,device,entry,,,,Pump size,stated,S,,\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", sizes), tables)
    view = device_card(graph, "XR01:IP1")
    items = [i for s in view.sections for i in s.items]
    tiles = {t.channel: t.title for t in items if isinstance(t, KeyTile)}
    assert tiles == {"XR01-IP1:State-Sts": "Pump OK", "XR01-IP1:Pressure-RB": "Vacuum"}
    (group,) = [
        g for o in items if hasattr(o, "groups") for g in o.groups
        if isinstance(g, ControlGroup) and g.kind == "entry"
    ]  # fmt: skip
    assert group.title == "Pump size"
    assert [c.label for c in group.controls] == ["Pump size A", "Pump size B"]


def test_a_channels_signal_never_turns_its_stated_words_into_a_typed_entry(tmp_path) -> None:
    """A writable channel that only states a signal plays it as a role, and the card
    still offers it as the choice of its stated words, a plain setpoint as an entry."""
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import Operate
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def modes(document) -> None:
        for name, signal in (("XR01-IP1:Mode-Sel", "mode"), ("XR01-IP1:Size-SP", "size")):
            document["channels"].append(
                {"id": name, "on": {"device": "XR01:IP1"}, "role": "setpoint", "signal": signal}
            )

    tables = copy_tables(tmp_path / "panels")
    with (tables / "record_types.csv").open("a") as handle:
        handle.write("XR01-IP1:Mode-Sel,mbbo,setpoint,stated,xr-ioc,0a1b2c3,vacApp/Db/pump.db:9\n")
        handle.write("XR01-IP1:Size-SP,ao,setpoint,stated,xr-ioc,0a1b2c3,vacApp/Db/pump.db:12\n")
    (tables / "state_labels.csv").write_text(
        "channel,value,label,source,revision,locator\n"
        "XR01-IP1:Mode-Sel,0,Off,S,,\nXR01-IP1:Mode-Sel,1,On,S,,\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", modes), tables)
    assert {h.channel: h.role.role for h in graph.roles("XR01:IP1")}["XR01-IP1:Mode-Sel"] == "mode"
    (operate,) = [i for s in device_card(graph, "XR01:IP1").sections for i in s.items
                  if isinstance(i, Operate)]  # fmt: skip
    kinds = {c.channel: g.kind for g in operate.groups for c in g.controls}
    assert kinds["XR01-IP1:Mode-Sel"] == "choice"
    assert kinds["XR01-IP1:Size-SP"] == "entry"


def test_a_state_its_health_rows_rule_is_a_lamp_by_default(tmp_path) -> None:
    """With no role row, a state channel's signal plays its role as a lamp when
    health rows say what its states mean, else as a word; a row overrides."""
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def shown(tables) -> str:
        graph = load_graph(write_fixture_facility(tmp_path / "f.json"), tables)
        (hit,) = [h for h in graph.roles("XR01:IP1") if h.channel == "XR01-IP1:State-Sts"]
        return hit.role.shown_as

    ruled = copy_tables(tmp_path / "ruled")
    assert shown(ruled) == "lamp"  # the fixture's health rows rule it
    unruled = copy_tables(tmp_path / "unruled")
    header = (unruled / "health.csv").read_text().splitlines()[0]
    (unruled / "health.csv").write_text(header + "\n")
    assert shown(unruled) == "word"
    (ruled / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,status,*:State-Sts,device,word,,,,,stated,S,,\n"
    )
    assert shown(ruled) == "word"  # a row overrides


def test_a_nameplate_signal_says_what_a_device_is_and_is_no_key_tile(tmp_path) -> None:
    """A role of a signal the facility marks ``nameplate`` (it names the unit, as a
    bootloader version does) is never a key tile; unmarked, the same channel is one;
    a row that gives the channel another role word overrides the mark."""
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import KeyTile
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    firmware = "XR01-IP1:Bootloader-RB"

    def facility(marked: bool):
        def change(document) -> None:
            signal = {"name": "bootloader_version_readback", "description": "Bootloader version."}
            if marked:
                signal["nameplate"] = True
            document.setdefault("signals", []).append(signal)
            document["channels"].append(
                {"id": firmware, "on": {"device": "XR01:IP1"}, "role": "readback",
                 "signal": "bootloader_version_readback"}
            )  # fmt: skip

        return write_fixture_facility(tmp_path / f"f-{marked}.json", change)

    def tiles(graph) -> set[str]:
        view = device_card(graph, "XR01:IP1")
        return {i.channel for s in view.sections for i in s.items if isinstance(i, KeyTile)}

    tables = copy_tables(tmp_path / "panels")
    marked = load_graph(facility(True), tables)
    assert "bootloader_version_readback" in marked.identity
    assert firmware not in tiles(marked)
    assert firmware in tiles(load_graph(facility(False), tables))
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,bootloader,*:Bootloader-RB,device,value,,,,,stated,S,,\n"
    )
    assert firmware in tiles(load_graph(facility(True), tables))  # a row's word decides


def test_the_vocabularys_nameplate_signals_say_what_a_device_is_with_no_table(tmp_path) -> None:
    """The shipped vocabulary marks its model, serial number and firmware version
    signals nameplate: a channel stating one is never a key tile, with no row."""
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import KeyTile
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    signals = ("model_readback", "serial_number_readback", "firmware_version_readback")

    def change(document) -> None:
        for n, signal in enumerate(signals):
            document["channels"].append(
                {"id": f"XR01-IP1:Plate{n}-RB", "on": {"device": "XR01:IP1"},
                 "role": "readback", "signal": signal}
            )  # fmt: skip

    graph = load_graph(
        write_fixture_facility(tmp_path / "f.json", change), copy_tables(tmp_path / "p")
    )
    assert set(signals) <= graph.identity
    view = device_card(graph, "XR01:IP1")
    tiles = {i.channel for s in view.sections for i in s.items if isinstance(i, KeyTile)}
    assert not {c for c in tiles if ":Plate" in c}
    from osprey.facility.panels.view import Chain

    (chain,) = [i for s in view.sections for i in s.items if isinstance(i, Chain)]
    (own,) = [b for b in chain.blocks if b.focus]
    # in the device's own block, each by the quantity its signal names
    assert [r.label for r in own.rows] == ["model", "serial number", "firmware version"]


def test_a_signal_s_title_names_its_channel_and_a_row_label_wins(tmp_path) -> None:
    """A channel whose signal states a ``title`` (the vocabulary's alarm-limit word,
    or one the facility gives its own signal) is called by it; a row's ``label``
    still decides."""
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import KeyTile
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    limit, trip = "XR01-IP1:TempHi-RB", "XR01-IP1:Trip-RB"

    def change(document) -> None:
        document.setdefault("signals", []).append(
            {"name": "trip_level_readback", "description": "Trip level.", "title": "Trip"}
        )
        for channel, signal in (
            (limit, "temperature_hi_limit_readback"),
            (trip, "trip_level_readback"),
        ):
            document["channels"].append(
                {"id": channel, "on": {"device": "XR01:IP1"}, "role": "readback",
                 "signal": signal}
            )  # fmt: skip

    def titles(graph) -> dict[str, str]:
        view = device_card(graph, "XR01:IP1")
        tiles = [i for s in view.sections for i in s.items if isinstance(i, KeyTile)]
        return {t.channel: t.title for t in tiles if t.channel in (limit, trip)}

    tables = copy_tables(tmp_path / "panels")
    facility = write_fixture_facility(tmp_path / "f.json", change)
    graph = load_graph(facility, tables)
    assert graph.titles["temperature_hi_limit_readback"] == "Hi"  # the vocabulary's word
    assert titles(graph) == {limit: "Hi", trip: "Trip"}
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "Pump,temperature_hi_limit_readback,*:TempHi-RB,device,value,,,,High limit,stated,S,,\n"
    )
    assert titles(load_graph(facility, tables))[limit] == "High limit"


def test_the_vocabulary_titles_its_general_signals_with_the_operator_s_word() -> None:
    from osprey.facility.validate import vocabulary

    titles = {row["name"]: row.get("title") for row in vocabulary()["signal_roles"]}
    assert {name: titles[name] for name in (
        "current_readback", "current_setpoint", "voltage_readback", "ramp_rate_setpoint",
        "power_on_command", "temperature_lolo_limit_readback", "temperature_hihi_limit_setpoint",
    )} == {
        "current_readback": "Current", "current_setpoint": "Setpoint",
        "voltage_readback": "Voltage", "ramp_rate_setpoint": "Ramp rate",
        "power_on_command": "On command", "temperature_lolo_limit_readback": "Lolo",
        "temperature_hihi_limit_setpoint": "Hihi",
    }  # fmt: skip


def _default_tiles(graph, device: str) -> list[tuple[str, str | None]]:
    from osprey.facility.panels.view import KeyTile

    view = device_card(graph, device)
    return [(i.title, i.channel) for s in view.sections for i in s.items if isinstance(i, KeyTile)]


def test_a_default_tile_counts_only_the_device_s_own_channels_of_its_role(tmp_path):
    from osprey.facility.panels import load_graph
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
        + "Pump,pressure,*:IP{n}:P,read_out_by,value,,,,,stated,S,1,l\n"  # the chassis's
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", _ported), tables)
    owners = {h.owner for h in graph.roles("XR01:IP1") if h.role.role == "pressure"}
    assert owners == {"XR01:IP1", "XR01:CH2"}  # the role reaches a neighbour's channel too
    # the device's one own channel of the role is its figure; the chassis's is not
    assert ("Pressure", "XR01-IP1:Pressure-RB") in _default_tiles(graph, "XR01:IP1")


def test_a_device_with_two_own_channels_of_a_role_gets_one_figure_over_both(tmp_path):
    from osprey.facility.panels import load_graph
    from osprey.facility.panels.view import KeyTile
    from tests.facility.panels._fixture import copy_tables, write_fixture_facility

    def second(document) -> None:
        document["channels"].append({"id": "XR01-IP1:Pressure2-RB", "on": {"device": "XR01:IP1"}})

    tables = copy_tables(tmp_path / "panels")
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER
        + "Pump,pressure,*:Pressure*-RB,device,value,,,,,stated,S,1,l\n"
        + "Pump,state,*:State-Sts,device,word,,,,,stated,S,1,l\n"
    )
    graph = load_graph(write_fixture_facility(tmp_path / "f.json", second), tables)
    own = [h.channel for h in graph.roles("XR01:IP1") if h.role.role == "pressure"]
    assert sorted(own) == ["XR01-IP1:Pressure-RB", "XR01-IP1:Pressure2-RB"]
    view = device_card(graph, "XR01:IP1")
    tiles = [i for s in view.sections for i in s.items if isinstance(i, KeyTile)]
    (over,) = [t for t in tiles if t.channels]  # two are one figure: the highest
    assert over.shown_as == "max"
    assert sorted(over.channels) == own
    assert over.title == "Highest pressure"  # the role's word, after the generic word
    assert over.names == ("Pressure-RB", "Pressure2-RB")  # each short of the device's id
    assert ("State", "XR01-IP1:State-Sts") in _default_tiles(graph, "XR01:IP1")

"""The device card's looks: the drawn frame, its chain, controls and grid; the calibration block."""

from __future__ import annotations

import dataclasses

from osprey.facility.panels import layout
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
    where, value = apply["rules"]
    assert value["targetProperty"] == "pressValue"
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


def test_a_press_whose_value_no_source_states_is_left_out_and_counted(tmp_path) -> None:
    from osprey.facility.panels.describe import unstated_presses

    roles = (  # "Bypass" is no state word of the channel: what it writes is not stated
        "PowerSupply,on-command,*:On-Cmd,device,action,,,,Bypass,stated,S,1,l\n"
        "PowerSupply,on,*:On-Sts,device,word,,,,,stated,S,1,l\n"
    )
    graph = _roles_graph(tmp_path, roles, "XR01-PS1:On-Cmd,1,On,S,1,l\n")
    doc = stub_render(device_card(graph, "XR01:PS1"), graph)
    assert _by_type(doc, "pv-button") == []
    grid = [n for n in _by_type(doc, "rectangle") if n["props"].get("tooltip") == "XR01-PS1:On-Cmd"]
    assert grid  # read only, in the grid
    assert unstated_presses(graph, ["XR01:PS1", "XR01:QF1"]) == (1, 1)


def test_a_command_with_no_role_is_an_action_named_by_its_word(panel_graph) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:PS1"), panel_graph)
    (button,) = _by_type(doc, "pv-button")
    assert (button["props"]["label"], button["props"]["pressValue"]) == ("On", 1)
    assert button["props"]["backgroundColor"]["$role"] == "Control_Action"
    assert _text(doc, "each writes its channel = 1 once (XR01-PS1:On-Cmd)")


def test_every_channel_not_drawn_above_is_in_the_grid_with_its_full_name_on_hover(
    panel_graph,
) -> None:
    doc = stub_render(device_card(panel_graph, "XR01:IP1"), panel_graph)
    for channel in ("XR01-IP1:Pressure-RB", "XR01-IP1:State-Sts"):
        assert [n for n in _by_type(doc, "rectangle") if n["props"].get("tooltip") == channel]
        assert [n for n in nodes(doc) if n["type"] != "polygon" and channel in _reads(doc, n)]
    lights = [n for n in _by_type(doc, "polygon") if n.get("rules")]
    assert lights  # the state channel's rule lights it in the grid


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
    (hover,) = [n for n in block if n["props"].get("tooltip") == _LONG_CONDITIONS]
    assert hover["type"] == "rectangle"
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


def test_a_recipe_chain_walks_its_hops_and_says_where_one_is_not_stated(tmp_path):
    from osprey.facility.panels.describe import chain_states

    graph = _chained(tmp_path, _ported)
    view = device_card(graph, "XR01:IP1")
    assert "Signal chain" in [s.title for s in view.sections]
    assert "relations" not in [s.title for s in view.sections]  # the recipe's chain instead
    device, readout, supply, further = _chain_of(view).blocks
    assert device.focus and [r.channel for r in device.rows] == ["XR01-IP1:Pressure-RB"]
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


def _stripped(tmp_path, *, raft: bool = False):
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
        "class,devices,symbol,lane,source\nPump,,pump,below,S\nQuadrupole,*:QF*,quad,beam,S\n"
    )
    (tables / "channel_roles.csv").write_text(
        ROLES_HEADER + "Pump,pressure,*:Pressure-RB,device,value,,,,,stated,S,1,l\n"
    )
    if raft:  # the place is placed; the second pump only by it
        with (tables / "positions.csv").open("a") as handle:
            handle.write(
                "XR/XR01,XR-RING,0,0,0,,,,1.5,0.0,0.0,,,stated,XR-LAYOUT,A,released,sheet 9\n"
            )
    return load_graph(write_fixture_facility(tmp_path / "f.json"), tables)


def test_a_strip_draws_the_members_round_the_device_to_scale(tmp_path) -> None:
    from osprey.facility.panels.view import CardStrip

    graph = _stripped(tmp_path)
    view = device_card(graph, "XR01:IP1")
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    assert strip.window == (-1.75, 4.25) and strip.bracket is None
    assert [(m.thing, m.symbol, m.lane, m.focus) for m in strip.members] == [
        ("XR01:IP1", "pump", "below", True),
        ("XR01:QF1", "quad", "beam", False),  # QD1 matches no symbols row: not drawn
    ]
    assert strip.members[0].channel == "XR01-IP1:Pressure-RB"
    doc = stub_render(view, graph)
    assert _text(doc, "XR01:IP1 · here")["props"]["foregroundColor"]["$role"] == "Focus"
    assert _text(doc, "2 m") and _text(doc, "-1 m")  # a tick every metre
    opens = [n for n in _by_type(doc, "rectangle") if n["props"].get("tooltip") == "XR01:QF1"]
    assert opens and opens[0]["props"]["actions"]


def test_a_device_placed_only_by_its_place_is_a_dashed_bracket_and_one_placed_by_none_says_so(
    tmp_path,
) -> None:
    from osprey.facility.panels.describe import STRIP_NOWHERE, STRIP_ON_RAFT, strip_state
    from osprey.facility.panels.view import CardHead, CardStrip

    graph = _stripped(tmp_path, raft=True)
    view = device_card(graph, "XR01:IP2")
    strip = next(i for s in view.sections for i in s.items if isinstance(i, CardStrip))
    assert strip.bracket == (0.0, 3.0, "on XR/XR01, place on the raft not stated")
    assert [m.thing for m in strip.members if m.focus] == ["XR01:IP2"]
    doc = stub_render(view, graph)
    assert _text(doc, "on XR/XR01, place on the raft not stated")
    assert strip_state(graph, graph.thing("XR01:IP2", "device")) == STRIP_ON_RAFT
    nowhere = _stripped(tmp_path / "b")
    view = device_card(nowhere, "XR01:IP2")
    assert not [i for s in view.sections for i in s.items if isinstance(i, CardStrip)]
    head = next(i for s in view.sections for i in s.items if isinstance(i, CardHead))
    assert head.nowhere
    assert _text(stub_render(view, nowhere), "where it sits is not stated")
    assert strip_state(nowhere, nowhere.thing("XR01:IP2", "device")) == STRIP_NOWHERE


def test_strip_states_count_the_devices_of_a_recipe_by_how_they_are_placed(tmp_path):
    from osprey.facility.panels.describe import strip_states

    graph = _stripped(tmp_path, raft=True)
    assert strip_states(graph, ["XR01:IP1", "XR01:IP2", "XR01:QF1"]) == {
        "pump": {"placed": 1, "on a raft": 1}
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
    assert Target("channels", "device", "XR01:PS1") not in found  # one listed, one internal
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

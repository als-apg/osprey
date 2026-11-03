"""The control-assistant template's storage-ring sector screens, ``sr/sector-NN``.

``scripts/facility_demo/facility_overview.py`` generates one screen per storage-ring
sector from the example facility's records, beside ``facility/overview`` whose sector
areas open them. Each draws its sector unrolled: the straight's own devices, the
magnets in beam order, a corrector pair and a BPM at each BPM position, and the
sector's peripherals where the records place them (ion pumps, gauges, gate valves,
insertion devices, front-end shutters, absorber thermocouples, scrapers, the tunnel
environment, harmonic cavities, radiation monitors), every element with its device
name as its ``tooltip``. Under the arc the orbit band draws each BPM's X and Y reading
as a bar below it. Panels bind each device's status and readbacks. Each sector opens
its vacuum detail ``sr/_vacuum-NN``, the orbit feedback ``sr/orbit-feedback`` and,
where its straight has absorber thermocouples, its absorber detail
``sr/_absorbers-NN``. The booster screen carries the
same names on its lattice.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from canopy_mcp import screens
from canopy_mcp.check.schema import check_schema, schema_path
from canopy_mcp.check.semantic import check_semantic
from canopy_mcp.store import DraftStore

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = (
    ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "canopy"
    / "library"
)
RECORDS = ROOT / "src" / "osprey" / "templates" / "facilities" / "example" / "records"
EXAMPLE_CHANNELS = RECORDS / "channels.yaml"
SECTORS = range(1, 13)

GENERATOR = ROOT / "scripts" / "facility_demo" / "facility_overview.py"

# Budgets per sector screen: the file and the bound names. The file budget allows
# the peripherals' icons and rules (a sector draws about three times the elements of
# its lattice alone); the name budget is the operator screen's channel cap.
SIZE_BUDGET = 512_000
NAME_BUDGET = 200
MAGNET_FAMILIES = ("DIPOLE", "QF", "QD", "QFA", "SF", "SD", "SHF", "SHD")
# What each family binds on the sector screen, by channel tail.
SECTOR_SIGNALS = {
    "SR:VAC:ION-PUMP": ("PRESSURE:RB", "CURRENT:RB", "STATUS:ON", "STATUS:FAULT"),
    "SR:VAC:GAUGE": ("PRESSURE:RB", "STATUS:VALID", "STATUS:TRIP", "RESET:SP"),
    "SR:VAC:VALVE": (
        "POSITION:OPEN",
        "POSITION:CLOSED",
        "STATUS:MOVING",
        "STATUS:FAULT",
        "OPEN:SP",
    ),
    "SR:ID": ("GAP:RB", "STATUS:GRANT", "STATUS:MOVING", "FF:ENABLE", "OFFSET:RB"),
    "SR:FE:SHUTTER": ("OPEN:RB",),
    "SR:VAC:TC": ("TEMPERATURE:RB",),
    "SR:DIAG:SCRAPER": ("POSITION:RB", "STATUS:MOVING", "STATUS:LIMIT", "STATUS:HOMED"),
    "SR:ENV:AIR": ("TEMPERATURE:RB",),
    "SR:ENV:AHU": ("VALVE:RB",),
    "SR:RF:HARMONIC": (
        "TUNER:RB",
        "TEMPERATURE:RB",
        "FLOW:RB",
        "POWER:RB",
        "STATUS:TEMP_OK",
        "STATUS:LOOP",
    ),
    "SR:DIAG:GAMMA": ("STATUS:ALARM", "DOSE_RATE:INST"),
    "SR:DIAG:NEUTRON": ("STATUS:ALARM", "DOSE_RATE:INST"),
}
# What the vacuum detail binds, by family.
VACUUM_DETAIL_SIGNALS = {
    "SR:VAC:ION-PUMP": (
        "PRESSURE:RB",
        "CURRENT:RB",
        "VOLTAGE:RB",
        "VOLTAGE:SP",
        "STATUS:ON",
        "STATUS:FAULT",
    ),
    "SR:VAC:GAUGE": ("PRESSURE:RB", "STATUS:ON", "STATUS:VALID", "STATUS:TRIP", "RESET:SP"),
    "SR:VAC:VALVE": (
        "POSITION:OPEN",
        "POSITION:CLOSED",
        "STATUS:MOVING",
        "STATUS:FAULT",
        "STATUS:READY",
        "CONTROL:OPEN",
        "CONTROL:CLOSE",
        "OPEN:SP",
    ),
}
# The injection straight's screen: the BTS line's last, inside straight 1.
INJECTION_SCREEN = "BTS:DIAG:TV:07"


def _path(screen_id: str) -> Path:
    return LIBRARY / "screens" / f"{screen_id}.screen.json"


def _nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _nodes(child)


@pytest.fixture(scope="module")
def gen() -> Iterator[ModuleType]:
    """The sector-screen generator, loaded by path: ``scripts/`` is not a package.

    It is registered in ``sys.modules`` before exec so its dataclasses resolve their
    module, and unregistered when the module's tests are done.
    """
    spec = importlib.util.spec_from_file_location("facility_overview_sectors", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture(scope="module")
def channels() -> dict:
    records = yaml.safe_load(EXAMPLE_CHANNELS.read_text(encoding="utf-8"))
    return {record["id"]: record for record in records}


@pytest.fixture(scope="module")
def devices() -> dict:
    records = yaml.safe_load((RECORDS / "devices.yaml").read_text(encoding="utf-8"))
    return {record["id"]: record for record in records}


@pytest.fixture(scope="module")
def docs() -> dict[int, dict]:
    return {n: json.loads(_path(f"sr/sector-{n:02d}").read_text("utf-8")) for n in SECTORS}


@pytest.fixture
def spec_dir(tmp_path: Path) -> Path:
    from importlib.resources import files

    source = files("pydmconverter.ir").joinpath("data", "screen-ir.schema.json")
    target = schema_path(tmp_path / "spec", "1.1")
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    return tmp_path / "spec"


def _sector_devices(channels: dict, prefix: str, n: int) -> list[int]:
    numbers = sorted(
        {int(m.group(1)) for c in channels if (m := re.match(rf"^{re.escape(prefix)}:(\d+):", c))}
    )
    per = len(numbers) // 12
    return numbers[(n - 1) * per : n * per]


def _device(channels: dict, prefix: str, number: int) -> str:
    head = f"{prefix}:{number:02d}:"
    return next(r["on"]["device"] for name, r in sorted(channels.items()) if name.startswith(head))


def _placed(channels: dict, devices: dict, prefix: str, n: int) -> dict[str, str]:
    """``{name: device}`` of every ``<prefix>:<name>:...`` device the records place in ``n``."""
    out = {}
    for address, record in channels.items():
        m = re.match(rf"^{re.escape(prefix)}:([^:]+):", address)
        if m and devices[record["on"]["device"]].get("place") == f"SR/SECT{n}":
            out[m.group(1)] = record["on"]["device"]
    return out


def _bound(doc: dict) -> set[str]:
    bound = {node["props"].get("pv") for node in _nodes(doc["root"])}
    bound |= {v for f in doc["formulas"] for v in f["bindings"].values()}
    for node in _nodes(doc["root"]):
        for rule in node.get("rules", []):
            bound |= {p["name"] for p in rule.get("pvs", [])}
        for action in node["props"].get("actions", []):
            bound.add(action.get("pvName"))
    return {name for name in bound if isinstance(name, str) and not name.startswith("formula://")}


def _expected_peripherals(channels: dict, devices: dict, n: int, signals: dict) -> set[str]:
    expected = set()
    for prefix, tails in signals.items():
        for name in _placed(channels, devices, prefix, n):
            expected |= {
                f"{prefix}:{name}:{tail}" for tail in tails if f"{prefix}:{name}:{tail}" in channels
            }
    return expected


def test_one_screen_per_sector_and_one_detail_per_family_where_needed(
    channels: dict, devices: dict
) -> None:
    held = sorted(p.name for p in (LIBRARY / "screens" / "sr").glob("*.screen.json"))
    expected = ["orbit-feedback.screen.json"]
    for n in SECTORS:
        expected += [f"sector-{n:02d}.screen.json", f"_vacuum-{n:02d}.screen.json"]
        if _placed(channels, devices, "SR:VAC:TC", n):
            expected.append(f"_absorbers-{n:02d}.screen.json")
    assert held == sorted(expected)


def test_each_sector_screen_validates_and_passes_the_semantic_check(
    docs: dict[int, dict], spec_dir: Path
) -> None:
    """The one finding allowed is CANOPY's note that a write button is inert in OSPREY."""
    for n, doc in docs.items():
        assert doc["id"] == f"sr/sector-{n:02d}"
        assert doc["metadata"]["title"] == f"SR sector {n:02d}"
        assert check_schema(doc, spec_dir) == [], n
        buttons = {node["id"] for node in _nodes(doc["root"]) if node["type"] == "regular-button"}
        findings = check_semantic(doc)
        assert {f.node_id for f in findings} == buttons, n
        assert {(f.code, f.severity) for f in findings} == {("write_not_relayed", "warning")}, n
        assert doc["macros"] == []


def test_each_sector_binds_its_records_within_budget(
    docs: dict[int, dict], channels: dict, devices: dict
) -> None:
    """A sector binds exactly its lattice, its straight's own devices and its peripherals."""
    store = DraftStore(LIBRARY)
    for n, doc in docs.items():
        assert _path(doc["id"]).stat().st_size <= SIZE_BUDGET, n
        pvs = set(screens.expanded_pvs(doc, store, doc["id"])) | _bound(doc)
        assert [p for p in pvs if p not in channels and not p.startswith("loc://")] == [], n
        for formula in doc["formulas"]:
            assert all(name in channels for name in formula["bindings"].values()), n
        names = {p for p in pvs if not p.startswith("loc://")}
        assert len(names) <= NAME_BUDGET, (n, len(names))
        expected = _expected_peripherals(channels, devices, n, SECTOR_SIGNALS)
        assert expected <= set(names), (n, sorted(expected - set(names))[:5])
        peripheral = {
            p for p in names if any(p.startswith(f"{prefix}:") for prefix in SECTOR_SIGNALS)
        }
        assert peripheral == expected, (n, sorted(peripheral - expected)[:5])


def test_each_sector_binds_its_own_share_of_every_family(
    docs: dict[int, dict], channels: dict
) -> None:
    for n, doc in docs.items():
        bound = _bound(doc)
        for family in (*MAGNET_FAMILIES, "HCM", "VCM"):
            prefix = f"SR:MAG:{family}"
            mine = _sector_devices(channels, prefix, n)
            shown = {int(name.split(":")[3]) for name in bound if name.startswith(f"{prefix}:")}
            assert shown == set(mine), (n, family)
        bpms = _sector_devices(channels, "SR:DIAG:BPM", n)
        for b in bpms:
            for signal in ("STATUS:VALID", "POSITION:X", "POSITION:Y"):
                assert f"SR:DIAG:BPM:{b:02d}:{signal}" in bound, (n, b, signal)
        assert f"SR:VAC:GAUGE:SR{n:02d}:PRESSURE:RB" in bound
        assert f"SR:VAC:VALVE:{n:02d}:POSITION:OPEN" in bound


def test_every_element_carries_its_records_device_name(
    docs: dict[int, dict], channels: dict, devices: dict, gen: ModuleType
) -> None:
    """The strip's symbols, dots and icons name exactly the sector's devices as tooltips."""
    for n, doc in docs.items():
        tips = {
            node["props"][gen.TOOLTIP]
            for node in _nodes(doc["root"])
            if gen.TOOLTIP in node["props"]
        }
        expected = set()
        for family in MAGNET_FAMILIES:
            expected |= {
                _device(channels, f"SR:MAG:{family}", d)
                for d in _sector_devices(channels, f"SR:MAG:{family}", n)
            }
        for prefix in ("SR:MAG:HCM", "SR:MAG:VCM", "SR:DIAG:BPM"):
            expected |= {_device(channels, prefix, d) for d in _sector_devices(channels, prefix, n)}
        expected |= {1: {"SR/SEPTUM01", "SR/SEPTUM02", "BTS/TV07"}}.get(n, set())
        expected |= {2: {"SR/CAVITY01", "SR/CAVITY02", "SR/KLYSTRON01", "SR/KLYSTRON02"}}.get(
            n, set()
        )
        for prefix in SECTOR_SIGNALS:
            expected |= set(_placed(channels, devices, prefix, n).values())
        assert tips == expected, (n, sorted(tips ^ expected)[:6])


def test_the_sector_magnets_follow_the_overview_beam_order(
    docs: dict[int, dict], gen: ModuleType
) -> None:
    """The strip draws the sector's magnets left to right in the overview's cell order."""
    order = gen.sequence({"DIPOLE": 3, **dict.fromkeys(MAGNET_FAMILIES[1:], 2)})
    for n, doc in docs.items():
        bars: dict[str, float] = {}
        for node in _nodes(doc["root"]):
            tip = node["props"].get(gen.TOOLTIP, "")
            family = tip.split("/")[-1].rstrip("0123456789")
            if node["type"] == "rectangle" and family in MAGNET_FAMILIES and tip.startswith("SR/"):
                centre = node["geometry"]["x"] + node["geometry"]["width"] / 2
                bars[tip] = round(centre)
        families = [t.split("/")[-1].rstrip("0123456789") for t in sorted(bars, key=bars.get)]
        assert families == order, n


def test_the_superbends_draw_in_their_cryostat(
    docs: dict[int, dict], devices: dict, gen: ModuleType
) -> None:
    """A superbend dipole draws its bar inside an outline; no other dipole has one."""
    superbends = {d for d, record in devices.items() if "superbend" in record.get("names", [])}
    assert len(superbends) == 3
    for n, doc in docs.items():
        outlined = {
            node["props"][gen.TOOLTIP]
            for node in _nodes(doc["root"])
            if node["type"] == "rectangle"
            and node["props"].get("fill") is False
            and node["props"].get(gen.TOOLTIP, "").startswith("SR/DIPOLE")
        }
        assert outlined <= superbends, n
        assert len(outlined) == (1 if n in (4, 8, 12) else 0), n


def test_the_only_writes_are_valve_and_gauge_buttons_and_links_open_details_and_orbit(
    docs: dict[int, dict], channels: dict, devices: dict
) -> None:
    for n, doc in docs.items():
        writes, links = [], []
        for node in _nodes(doc["root"]):
            for action in node["props"].get("actions", []):
                if action["type"] == "write_pv":
                    assert node["type"] == "regular-button", (n, node["id"])
                    writes.append((action["pvName"], action["value"]))
                elif action["type"] == "open_display":
                    links.append(action["file"])
                else:
                    raise AssertionError((n, action))
            assert node["type"] not in ("pv-button", "pv-text-input", "pv-slider"), n
        valves = _placed(channels, devices, "SR:VAC:VALVE", n)
        gauges = _placed(channels, devices, "SR:VAC:GAUGE", n)
        expected = [(f"SR:VAC:VALVE:{v}:OPEN:SP", value) for v in valves for value in "10"]
        expected += [(f"SR:VAC:GAUGE:{g}:RESET:SP", "1") for g in gauges]
        assert sorted(writes) == sorted(expected), n
        detail = [f"../sr/_vacuum-{n:02d}.screen.json", "../sr/orbit-feedback.screen.json"]
        if _placed(channels, devices, "SR:VAC:TC", n):
            detail.append(f"../sr/_absorbers-{n:02d}.screen.json")
        assert sorted(links) == sorted(detail), n


def test_formula_readouts_round_and_drawing_formulas_follow_their_device(
    docs: dict[int, dict],
) -> None:
    """A readout formula rounds itself; a drawing formula moves an icon with its readback.

    The runtime shows a formula's value with every digit, and the formula language has
    no ``round``: BPM position rounds to 0.1 µm, corrector current to 0.01 A, pressure
    to 0.01 nTorr and the hottest absorber to 0.1 degC, each by
    ``floor(x * 10**k + 0.5) / 10**k``. An insertion device's rows follow its gap (and
    an elliptically polarising one's upper row its offset); a scraper blade's depth
    follows its position; an orbit bar follows its BPM's reading in its plane.
    """
    readouts = {"_um": 1, "_a": 2, "_ntorr": 2, "_c": 1}
    orbit = ("POSITION:X", "POSITION:Y")
    drawings = {
        "_y": ("GAP:RB", "POSITION:RB", *orbit),
        "_h": ("POSITION:RB", *orbit),
        "_x": ("OFFSET:RB",),
    }
    for n, doc in docs.items():
        for formula in doc["formulas"]:
            name, expression = formula["name"], formula["expression"]
            k = next((k for suffix, k in readouts.items() if name.endswith(suffix)), None)
            if k is not None:
                assert expression.startswith("floor(("), (n, name)
                assert expression.endswith(f") * {10**k} + 0.5) / {10**k}"), (n, name)
                continue
            tails = next(t for suffix, t in drawings.items() if name.endswith(suffix))
            [channel] = formula["bindings"].values()
            assert channel.endswith(tails), (n, name, channel)
        names = [f["name"] for f in doc["formulas"]]
        assert sum(name.endswith("_a") for name in names) == 2 * 6, n


def test_the_orbit_band_draws_each_bpm_reading_below_its_bpm(
    docs: dict[int, dict], channels: dict, gen: ModuleType
) -> None:
    """One X bar and one Y bar per BPM, centred under its BPM, grown and tinted by its reading.

    A bar's top and height follow the reading on one full scale; its fill is grey while
    the BPM is invalid and otherwise the reading's magnitude band.
    """
    for n, doc in docs.items():
        # The strip's BPM LEDs, the ones naming their device; the panel's carry no name.
        leds = {
            node["props"]["pv"].split(":")[3]: node["geometry"]
            for node in _nodes(doc["root"])
            if node["type"] == "pv-led"
            and gen.TOOLTIP in node["props"]
            and re.fullmatch(r"SR:DIAG:BPM:\d+:STATUS:VALID", node["props"]["pv"])
        }
        formulas = {f"formula://{f['name']}": f for f in doc["formulas"]}
        bars: dict[tuple[str, str], dict] = {}
        for node in _nodes(doc["root"]):
            rules = {r["targetProperty"]: r for r in node.get("rules", [])}
            if node["type"] != "rectangle" or "height" not in rules:
                continue
            [channel] = formulas[rules["height"]["formula"]]["bindings"].values()
            match = re.fullmatch(r"SR:DIAG:BPM:(\d+):POSITION:([XY])", channel)
            if match is None:
                continue
            bpm, plane = match.groups()
            assert (bpm, plane) not in bars, (n, bpm, plane)
            bars[bpm, plane] = node
            [top_channel] = formulas[rules["y"]["formula"]]["bindings"].values()
            assert top_channel == channel, (n, node["id"])
            tint = rules["fillColor"]
            assert [p["name"] for p in tint["pvs"]] == [
                f"SR:DIAG:BPM:{bpm}:STATUS:VALID",
                channel,
            ], (n, node["id"])
            led = leds[bpm]
            centre = node["geometry"]["x"] + node["geometry"]["width"] / 2
            assert centre == pytest.approx(led["x"] + led["width"] / 2), (n, bpm, plane)
            assert node["props"][gen.TOOLTIP] == _device(channels, "SR:DIAG:BPM", int(bpm))
        assert sorted(bars) == sorted((b, p) for b in leds for p in "XY"), n


def test_rule_driven_geometry_targets_a_formula(docs: dict[int, dict]) -> None:
    for n, doc in docs.items():
        formulas = {f"formula://{f['name']}" for f in doc["formulas"]}
        for node in _nodes(doc["root"]):
            for rule in node.get("rules", []):
                if rule["targetProperty"] in ("x", "y", "height"):
                    assert rule["formula"] in formulas, (n, node["id"])


def test_pressures_read_in_ntorr_through_their_formulas(
    docs: dict[int, dict], channels: dict
) -> None:
    for n, doc in docs.items():
        fields = [node["props"]["pv"] for node in _nodes(doc["root"]) if node["type"] == "pv-label"]
        assert not any(re.fullmatch(r"SR:MAG:[HV]CM:\d+:CURRENT:RB", pv) for pv in fields), n
        assert not any(pv.endswith(":PRESSURE:RB") for pv in fields), n
        texts = [node["props"].get("text") for node in _nodes(doc["root"])]
        assert "P nTorr" in texts, n
        assert not any("hover" in (text or "").lower() for text in texts), n
        for formula in doc["formulas"]:
            if formula["name"].endswith("_ntorr"):
                [channel] = formula["bindings"].values()
                assert channels[channel]["unit"] == "Torr", channel
                assert "* 1e9" in formula["expression"], formula["name"]


def test_the_injection_straight_shows_its_screen_and_its_septa_interlocks(
    docs: dict[int, dict], channels: dict, gen: ModuleType
) -> None:
    for n, doc in docs.items():
        embeds = [node for node in _nodes(doc["root"]) if node["type"] == "embedded-display"]
        leds = [node for node in _nodes(doc["root"]) if node["type"] == "pv-byte-led"]
        if n != 1:
            assert embeds == [] and leds == [], n
            continue
        [tv] = embeds
        assert tv["props"]["file"] == "../common/embedded/_tv.screen.json"
        assert tv["props"]["macros"]["paddle"] == f"{INJECTION_SCREEN}:PADDLE:SP"
        words = Counter(node["props"]["pv"] for node in leds)
        assert words == {
            f"SR:MAG:SEPTUM:0{k}:INTERLOCK:WORD": len(gen.INTERLOCK_BIT_TEXT) for k in (1, 2)
        }
        assert all(channels[w]["value_type"] == "int" for w in words)
        assert sorted(node["props"]["shift"] for node in leds) == sorted(
            list(range(len(gen.INTERLOCK_BIT_TEXT))) * 2
        )


def test_the_detail_screens_bind_every_signal_of_their_family(
    channels: dict, devices: dict, spec_dir: Path
) -> None:
    store = DraftStore(LIBRARY)
    for n in SECTORS:
        doc = json.loads(_path(f"sr/_vacuum-{n:02d}").read_text("utf-8"))
        assert doc["metadata"]["title"] == f"SR sector {n:02d} · vacuum"
        assert check_schema(doc, spec_dir) == [], n
        pvs = set(screens.expanded_pvs(doc, store, doc["id"])) | _bound(doc)
        assert pvs == _expected_peripherals(channels, devices, n, VACUUM_DETAIL_SIGNALS), n
        tcs = _placed(channels, devices, "SR:VAC:TC", n)
        if not tcs:
            continue
        doc = json.loads(_path(f"sr/_absorbers-{n:02d}").read_text("utf-8"))
        assert check_schema(doc, spec_dir) == [], n
        assert check_semantic(doc) == [], n
        pvs = set(screens.expanded_pvs(doc, store, doc["id"]))
        assert pvs == {
            f"SR:VAC:TC:{name}:{tail}"
            for name in tcs
            for tail in ("TEMPERATURE:RB", "LIMIT:RB", "STATUS:OVERTEMP")
        }, n


def test_the_peripheral_readouts_state_precision_and_units_on_their_records(
    channels: dict,
) -> None:
    """The runtime formats a readout from its record, so each float shown states both."""
    for prefix, tails in SECTOR_SIGNALS.items():
        for address, record in channels.items():
            if not address.startswith(f"{prefix}:") or not address.endswith(tails):
                continue
            if record.get("value_type", "float") != "float" or address.endswith(
                ("PRESSURE:RB", "DOSE_RATE:INST")
            ):
                continue
            assert "unit" in record and "precision" in record, address


def test_the_straight_two_cavities_read_in_mv_with_two_decimals(channels: dict) -> None:
    for number in (1, 2):
        record = channels[f"SR:RF:CAVITY:{number:02d}:VOLTAGE:RB"]
        assert (record["unit"], record["precision"]) == ("MV", 2)


def test_the_booster_lattice_names_its_devices(channels: dict, gen: ModuleType) -> None:
    doc = json.loads((LIBRARY / "screens" / "br" / "overview.screen.json").read_text("utf-8"))
    tips = sorted(
        node["props"][gen.TOOLTIP] for node in _nodes(doc["root"]) if gen.TOOLTIP in node["props"]
    )
    expected = set()
    for name, record in channels.items():
        if re.match(r"^BR:(MAG:(DIPOLE|QF|QD):\d+|DIAG:BPM:\d+):", name):
            expected.add(record["on"]["device"])
    assert tips == sorted(expected)

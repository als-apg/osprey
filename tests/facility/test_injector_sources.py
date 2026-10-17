"""The example facility's injector records, merged from the element table.

The generator's ``injector`` mode merges the storage ring's injection septa SEK
and SEN into ``records/`` and ``seeds.yaml``. On a copy of the committed tree
with the septa removed it writes the committed records back, every other record
byte-equal; run twice on the committed tree it changes nothing. It writes no
limits: the demo ``limits.yaml`` holds only its teaching records. The septa
are texture rows: no model wires them, every setpoint and on-status carries a
seed, and the SEK current pulse is a ``pulse`` waveform driven by the SEK
setpoint and enabled by its on-status.
The rule table gives the septa their readback noise and setpoint tolerance, and
the committed seeds are exactly what ``reseed`` writes.

The same mode merges the gun-to-linac line (GTL) and the linac (LN) as texture
stand-ins: every GTL/LN row of the element table is a channel, in places GTL
and LN, with a group per family. Each setpoint pairs with its readback, every
setpoint, on-status and unpaired readback carries its nominal, a paired
readback follows its setpoint, and no stand-in channel is wired or limited.

It merges the linac-to-booster line (LTB) and grows the booster-to-storage line
(BTS) the same way: every LTB/BTS row of the element table is a channel, place
LTB spans the LTB deck, and every LTB/BTS family has a group. The 72 frozen
BTS channels, their 24 devices and BTS's system groups keep their text; the
"all paddles out" count and the storage ring's first screen sit in BTS. A
channel the LTB or BTS model wires carries no seed nominal: its operating
point is the deck's.

It adds the booster's panel rows as texture rows of place BR: the injection
kicker (a ``Kicker``), the extraction septa SEN and SEK (unwired, no pulse), the
five screens, and the energy ramps' gain/offset stand-in setpoints, with a group
per new family. With them every element-table row is a channel. The booster's
older records keep their text, and no booster channel has a limits record.
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from osprey.facility.sources import read_yaml
from tests.facility.test_cf_view_parity import fingerprint_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts/facility_demo/generate.py"
RESEED = REPO_ROOT / "scripts/demo_seeds/reseed.py"
EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"

SEK = "SR:MAG:SEPTUM:01"
SEN = "SR:MAG:SEPTUM:02"

#: The sections the stand-ins fill.
STANDIN_SECTIONS = ("GTL", "LN")

#: The transfer lines the table fills or grows.
TRANSFER_SECTIONS = ("LTB", "BTS")

#: The 72 frozen BTS channels, from the demo fingerprint.
FROZEN_BTS = frozenset(
    row["address"] for row in fingerprint_rows() if row["address"].startswith("BTS:")
)

#: The 24 BTS devices that carry them.
FROZEN_BTS_DEVICES = frozenset(
    f"BTS/{family}{number:02d}" for family in ("HCM", "VCM", "QF", "BPM") for number in range(1, 7)
)

#: The BTS records that predate the table and that no generator re-emits.
FROZEN_BTS_RECORDS = FROZEN_BTS | FROZEN_BTS_DEVICES | {"BTS/DIAG", "BTS/MAG"}

#: The records that predate the table and that the generator re-emits in place:
#: the stripped tree keeps a stub of each, so the record keeps its position.
REWRITTEN_IN_PLACE = {"BTS"}

#: The machine places the injector emits, in beam order.
INJECTOR_PLACES = ("GTL", "LN", "LTB", "BTS")

#: The files the ``injector`` mode merges into, relative to the tree.
RECORDS = [
    "records/places.yaml",
    "records/devices.yaml",
    "records/channels.yaml",
    "records/groups.yaml",
]
MERGED = [*RECORDS, "seeds.yaml"]


def _is_septum(record_id: str) -> bool:
    return record_id.startswith(("SR:MAG:SEPTUM:", "SR/SEPTUM"))


def _is_standin(record_id: str) -> bool:
    return record_id.split(":", 1)[0].split("/", 1)[0] in STANDIN_SECTIONS


def _is_transfer(record_id: str) -> bool:
    """An LTB or BTS record the generator emits: any but the frozen BTS ones."""
    section = record_id.split(":", 1)[0].split("/", 1)[0]
    return section in TRANSFER_SECTIONS and record_id not in FROZEN_BTS_RECORDS


#: The booster families the table adds; the booster's other families predate it.
BOOSTER_FAMILIES = ("KICKER", "SEPTUM", "TV", "DIPOLE-RAMP", "QF-RAMP", "QD-RAMP")


def _is_booster(record_id: str) -> bool:
    """A booster record the generator emits: a channel, device or group of a new family."""
    if record_id.startswith("BR:"):
        return record_id.split(":")[2] in BOOSTER_FAMILIES
    if record_id.startswith("BR/"):
        return record_id[3:].rstrip("0123456789") in BOOSTER_FAMILIES
    return False


def _is_generated(record_id: str) -> bool:
    return (
        _is_septum(record_id)
        or _is_booster(record_id)
        or _is_standin(record_id)
        or _is_transfer(record_id)
    )


def _chunks(text: str) -> list[str]:
    """A top-level YAML list's items, each as its exact text."""
    chunks: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("- ") or not chunks:
            chunks.append(line)
        else:
            chunks[-1] += line
    return chunks


def _chunk_id(chunk: str) -> str:
    (record,) = read_yaml(chunk)
    return str(record["id"])


def _without_generated(text: str) -> list[str]:
    return [chunk for chunk in _chunks(text) if not _is_generated(_chunk_id(chunk))]


def _stripped(text: str) -> str:
    """``text`` without its generated records; one rewritten in place leaves a stub."""
    kept = []
    for chunk in _chunks(text):
        key = _chunk_id(chunk)
        if key in REWRITTEN_IN_PLACE:
            kept.append(f"- id: {key}\n  description: stub\n")
        elif not _is_generated(key):
            kept.append(chunk)
    return "".join(kept)


def _run_injector(tree: Path) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR), "injector", str(tree)],
        check=True,
        capture_output=True,
    )


@cache
def _committed(rel: str) -> Any:
    return read_yaml((EXAMPLE / rel).read_text(encoding="utf-8"))


def _channels() -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in _committed("records/channels.yaml")}


def _devices() -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in _committed("records/devices.yaml")}


def _reseed() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reseed_under_test", RESEED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stripped(tmp_path: Path) -> Path:
    """A copy of the committed example tree with every generated record removed.

    A record the generator rewrites in place is left as a stub instead.
    """
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    for rel in RECORDS:
        path = tree / rel
        path.write_text(_stripped(path.read_text(encoding="utf-8")), "utf-8")
    seeds = (tree / "seeds.yaml").read_text(encoding="utf-8")
    kept = [chunk for chunk in _top_level_keys(seeds) if not _is_generated(chunk.split(":\n")[0])]
    (tree / "seeds.yaml").write_text("".join(kept), "utf-8")
    return tree


def _top_level_keys(text: str) -> list[str]:
    """A top-level YAML mapping's entries, each as its exact text."""
    chunks: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.startswith(" ") or not chunks:
            chunks.append(line)
        else:
            chunks[-1] += line
    return chunks


# --- the generator -----------------------------------------------------------------


def test_the_injector_mode_writes_the_committed_records(stripped: Path) -> None:
    _run_injector(stripped)
    for rel in RECORDS:
        assert (stripped / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel


def test_the_injector_mode_writes_the_committed_seed_keys(stripped: Path) -> None:
    _run_injector(stripped)
    written = read_yaml((stripped / "seeds.yaml").read_text(encoding="utf-8"))
    for address, seed in _committed("seeds.yaml").items():
        owned = {key: seed[key] for key in ("nominal", "pulse", "gate") if key in seed}
        if _is_generated(address) and owned:
            assert written[address] == owned, address


def test_the_injector_mode_leaves_every_other_record_byte_equal(stripped: Path) -> None:
    before = {rel: (stripped / rel).read_text(encoding="utf-8") for rel in RECORDS}
    _run_injector(stripped)
    for rel in RECORDS:
        after = (stripped / rel).read_text(encoding="utf-8")
        assert _without_generated(after) == _without_generated(before[rel]), rel


def test_the_injector_mode_leaves_the_limits_untouched(stripped: Path) -> None:
    before = (stripped / "limits.yaml").read_bytes()
    _run_injector(stripped)
    assert (stripped / "limits.yaml").read_bytes() == before


def test_the_injector_mode_is_idempotent(tmp_path: Path) -> None:
    tree = tmp_path / "example"
    shutil.copytree(EXAMPLE, tree)
    _run_injector(tree)
    _run_injector(tree)
    for rel in [*MERGED, "limits.yaml"]:
        assert (tree / rel).read_bytes() == (EXAMPLE / rel).read_bytes(), rel


# --- the records -------------------------------------------------------------------


def test_the_septa_are_septum_devices_in_the_storage_ring() -> None:
    devices = {key: value for key, value in _devices().items() if _is_septum(key)}
    assert sorted(devices) == ["SR/SEPTUM01", "SR/SEPTUM02"]
    assert {device["class"] for device in devices.values()} == {"Septum"}
    assert devices["SR/SEPTUM01"]["label"] == "SEK"
    assert devices["SR/SEPTUM02"]["label"] == "SEN"
    for device in devices.values():
        assert device["place"].split("/")[0] == "SR"
        assert device["description"]


def test_each_septum_has_its_setpoint_readback_and_on_status() -> None:
    channels = _channels()
    for prefix, device in ((SEK, "SR/SEPTUM01"), (SEN, "SR/SEPTUM02")):
        sp, rb, on = (
            channels[f"{prefix}:{tail}"] for tail in ("CURRENT:SP", "CURRENT:RB", "STATUS:ON")
        )
        assert sp["role"] == "setpoint"
        assert sp["pair"] == rb["id"]
        assert sp["signal"] == "current_setpoint" and sp["unit"] == "A"
        assert rb["signal"] == "current_readback" and rb["unit"] == "A"
        assert on["signal"] == "power_on_status" and on["value_type"] == "bool"
        for channel in (sp, rb, on):
            assert channel["on"] == {"device": device}


def test_only_sek_carries_the_sampled_current_pulse() -> None:
    waveforms = {
        address: channel
        for address, channel in _channels().items()
        if address.startswith("SR:MAG:SEPTUM:") and channel.get("value_type") == "waveform"
    }
    assert list(waveforms) == [f"{SEK}:CURRENT:WAVEFORM"]
    (channel,) = waveforms.values()
    assert channel["shape"] == [500]
    assert channel["signal"] == "pulse_waveform"
    assert channel["unit"] == "A"
    assert "role" not in channel


def test_every_septum_channel_has_a_description() -> None:
    for address, channel in _channels().items():
        if address.startswith("SR:MAG:SEPTUM:"):
            assert channel.get("description"), address
    assert "SEK" in _channels()[f"{SEK}:CURRENT:SP"]["description"]
    assert "SEN" in _channels()[f"{SEN}:CURRENT:SP"]["description"]
    for tail in ("CURRENT:SP", "STATUS:ON"):
        assert "sampled current pulse" in _channels()[f"{SEK}:{tail}"]["description"]
        assert "sampled current pulse" not in _channels()[f"{SEN}:{tail}"]["description"]


def test_no_model_wires_a_septum() -> None:
    for model in _committed("models.yaml"):
        for record in model.get("wiring", []):
            assert not str(record["address"]).startswith("SR:MAG:SEPTUM:"), model["name"]


# --- the seeds ---------------------------------------------------------------------


def test_every_septum_setpoint_and_on_status_carries_its_nominal() -> None:
    seeds = _committed("seeds.yaml")
    assert seeds[f"{SEK}:CURRENT:SP"]["nominal"] == 3000.0
    assert seeds[f"{SEN}:CURRENT:SP"]["nominal"] == 800.0
    assert seeds[f"{SEK}:STATUS:ON"]["nominal"] == "TRUE"
    assert seeds[f"{SEN}:STATUS:ON"]["nominal"] == "TRUE"


def test_the_sek_waveform_pulses_from_its_setpoint_and_on_status() -> None:
    seeds = _committed("seeds.yaml")
    assert seeds[f"{SEK}:CURRENT:WAVEFORM"] == {
        "pulse": {"driver": f"{SEK}:CURRENT:SP", "enable": f"{SEK}:STATUS:ON"}
    }


def test_no_storage_ring_channel_carries_a_gate() -> None:
    gated = [
        address
        for address, seed in _committed("seeds.yaml").items()
        if address.startswith("SR:") and "gate" in seed
    ]
    assert gated == []


def test_the_seeds_are_what_reseed_writes() -> None:
    assert (EXAMPLE / "seeds.yaml").read_text(encoding="utf-8") == _reseed().render()


def _rule_tolerance(address: str) -> float:
    """The tolerance the rule table states for the setpoint ``address``, in its unit."""
    reseed = _reseed()
    channel = _channels()[address]
    device = channel["on"]["device"]
    row = reseed.match(
        reseed.tolerance_rules(reseed.load_rules()),
        cls=_devices()[device]["class"],
        signal=channel["signal"],
        device=device,
    )
    assert row is not None, address
    return reseed.convert(row["tolerance"], channel["unit"])


def test_every_septum_setpoint_carries_its_rule_tolerance() -> None:
    for prefix in (SEK, SEN):
        address = f"{prefix}:CURRENT:SP"
        assert _channels()[address]["tolerance"] == {"absolute": _rule_tolerance(address)}
        assert _rule_tolerance(address) == 1.0


def test_every_septum_rule_row_names_a_source() -> None:
    rows = [row for row in _reseed().load_rules() if row["class"] == "Septum"]
    assert {row["signal"] for row in rows} == {"current_setpoint", "current_readback"}
    assert all(row["source"] for row in rows)


def test_each_septum_readback_moves_by_its_rule_noise() -> None:
    seeds = _committed("seeds.yaml")
    for prefix in (SEK, SEN):
        assert seeds[f"{prefix}:CURRENT:RB"]["noise"] == {"absolute": 0.1}


# --- the limits and the group ------------------------------------------------------


def test_no_septum_channel_has_a_limits_record() -> None:
    addresses = [record["address"] for record in _committed("limits.yaml")["records"]]
    assert [address for address in addresses if address.startswith((SEK, SEN))] == []


def test_the_septum_group_lists_sek_and_sen() -> None:
    groups = {record["id"]: record for record in _committed("records/groups.yaml")}
    group = groups["SR/SEPTUM"]
    assert group["members"] == ["SR/SEPTUM01", "SR/SEPTUM02"]
    assert "SEK" in group["description"] and "SEN" in group["description"]


def test_the_middle_layer_view_keys_both_septa_under_the_septum_family() -> None:
    from osprey.facility.views.channel_finder import middle_layer_families

    doc = {
        "devices": _committed("records/devices.yaml"),
        "groups": _committed("records/groups.yaml"),
    }
    assert middle_layer_families(doc)["Septum"] == [("BR", "SEPTUM"), ("SR", "SEPTUM")]


# --- the build ---------------------------------------------------------------------


def test_the_example_tree_validates() -> None:
    from osprey.facility.validate import validate

    out = io.StringIO()
    assert validate(EXAMPLE, project_name="example", file=out) == 0, out.getvalue()
    assert "seed-missing" not in out.getvalue()
    assert "seed-invalid" not in out.getvalue()


# --- the gun-to-linac line and the linac -------------------------------------------

SOURCES = REPO_ROOT / "scripts/facility_demo"
EXCERPTS = Path(__file__).resolve().parent / "fixtures/mml"

#: Each lattice file and its checked-in excerpt.
EXCERPT_OF = {"GTB/gtblattice.m": "gtblattice_excerpt.m"}


def _facility_demo(name: str) -> ModuleType:
    """A ``scripts/facility_demo`` module, imported by the bare name the generator uses."""
    if str(SOURCES) not in sys.path:
        sys.path.insert(0, str(SOURCES))
    return importlib.import_module(name)


def _table_addresses() -> set[str]:
    elements = _facility_demo("_elements")
    return {row.demo_name for row in elements.load_elements() if row.section in STANDIN_SECTIONS}


def _standin_channels() -> dict[str, dict[str, Any]]:
    return {address: record for address, record in _channels().items() if _is_standin(address)}


def test_every_gtl_and_ln_table_row_is_a_channel() -> None:
    addresses = _table_addresses()
    assert addresses
    assert set(_standin_channels()) == addresses


def test_places_gtl_and_ln_exist_and_hold_their_devices() -> None:
    places = {record["id"]: record for record in _committed("records/places.yaml")}
    for section in STANDIN_SECTIONS:
        assert places[section]["level"] == "machine"
        assert places[section]["description"]
    devices = {key: value for key, value in _devices().items() if _is_standin(key)}
    assert devices
    for key, device in devices.items():
        assert device["place"] == key.split("/")[0], key
        assert device["description"]


def test_every_gtl_and_ln_family_has_a_group_of_its_devices() -> None:
    groups = {record["id"]: record for record in _committed("records/groups.yaml")}
    families: dict[str, list[str]] = {}
    for key in sorted(device for device in _devices() if _is_standin(device)):
        families.setdefault(key.rstrip("0123456789"), []).append(key)
    assert {"GTL/EGUN", "GTL/SHB", "GTL/TV", "LN/ACC", "LN/BPM", "LN/QF"} <= set(families)
    for family, members in families.items():
        group = groups[family]
        assert group["members"] == members, family
        assert group["description"] and "\n" not in group["description"], family


def test_every_channel_names_a_device_of_its_section() -> None:
    devices = _devices()
    for address, channel in _standin_channels().items():
        device = channel["on"]["device"]
        assert device in devices, address
        assert device.split("/")[0] == address.split(":")[0], address
        assert channel["description"], address


def test_each_gtl_and_ln_setpoint_pairs_with_its_readback() -> None:
    channels = _channels()
    setpoints = {a: c for a, c in _standin_channels().items() if c.get("role") == "setpoint"}
    assert setpoints
    for address, channel in setpoints.items():
        assert address.endswith(":SP"), address
        assert channel["pair"] == f"{address.rsplit(':', 1)[0]}:RB"
        readback = channels[channel["pair"]]
        assert "role" not in readback, address
        assert readback["signal"] != channel["signal"], address
        assert readback.get("value_type", "float") == channel.get("value_type", "float")


def test_the_gun_on_offs_and_the_paddles_are_writable_bool_setpoints() -> None:
    channels = _channels()
    gun = "GTL:GUN:EGUN:01"
    bools = [f"{gun}:{field}:SP" for field in ("HV_ON", "HEATER_ON", "AUX_ON")]
    paddles = [
        address
        for address, channel in _standin_channels().items()
        if ":TV:" in address and channel.get("role") == "setpoint"
    ]
    assert len(paddles) == 4, paddles
    limited = {record["address"] for record in _committed("limits.yaml")["records"]}
    for address in [*bools, *paddles]:
        channel = channels[address]
        assert channel["role"] == "setpoint" and channel["value_type"] == "bool", address
        assert address not in limited, address


def test_no_gtl_or_ln_channel_has_a_limits_record() -> None:
    addresses = [record["address"] for record in _committed("limits.yaml")["records"]]
    assert [address for address in addresses if _is_standin(address)] == []


def test_every_gtl_and_ln_setpoint_carries_its_nominal() -> None:
    seeds = _committed("seeds.yaml")
    elements = _facility_demo("_elements")
    nominal = {
        row.demo_name: row.nominal_value()
        for row in elements.load_elements()
        if row.section in STANDIN_SECTIONS and row.nominal_value() is not None
    }
    for address, channel in _standin_channels().items():
        if channel.get("role") == "setpoint":
            assert address in nominal, address
            assert seeds[address]["nominal"] == nominal[address], address


def test_a_paired_gtl_or_ln_readback_follows_its_setpoint() -> None:
    seeds = _committed("seeds.yaml")
    for channel in _standin_channels().values():
        if channel.get("role") == "setpoint":
            assert "nominal" not in seeds.get(channel["pair"], {}), channel["pair"]


def test_the_transmitted_charge_stands_in_with_a_nominal() -> None:
    seeds = _committed("seeds.yaml")
    charges = [a for a in _standin_channels() if ":ICT:" in a]
    assert {a.split(":")[0] for a in charges} == set(STANDIN_SECTIONS)
    for address in charges:
        assert seeds[address]["nominal"] > 0, address


def test_no_model_wires_a_gtl_or_ln_channel() -> None:
    for model in _committed("models.yaml"):
        for record in model.get("wiring", []):
            assert not _is_standin(str(record["address"])), model["name"]


def _mml_section(section: str):
    mml = _facility_demo("_mml")
    original = mml.LATTICES[section][0]
    source = mml.MML_ROOT / original if mml.MML_ROOT is not None else None
    path = source if source is not None and source.is_file() else EXCERPTS / EXCERPT_OF[original]
    return mml.parse(path.read_text(encoding="latin-1"), section)


@pytest.mark.parametrize("section", STANDIN_SECTIONS)
def test_the_table_lists_correctors_and_monitors_in_parsed_beam_order(section: str) -> None:
    standin = _facility_demo("_standin")
    families = {"HCM", "VCM", "BPM"}
    table: list[str] = []
    for row in standin.rows():
        device = standin.device_id(row)
        family = device.split("/")[1].rstrip("0123456789")
        if row.section == section and family in families and device not in table:
            table.append(device)
    parsed = [
        f"{section}/{element.family}{element.occurrence:02d}"
        for element in _mml_section(section).elements
        if element.family in families
    ]
    assert table == parsed


def test_the_injector_mode_writes_the_gtl_and_ln_seed_nominals(stripped: Path) -> None:
    _run_injector(stripped)
    written = read_yaml((stripped / "seeds.yaml").read_text(encoding="utf-8"))
    for address in _standin_channels():
        committed = _committed("seeds.yaml").get(address, {})
        if "nominal" in committed:
            assert written[address]["nominal"] == committed["nominal"], address


def test_the_nominal_scenario_says_the_linac_energy_is_150_mev() -> None:
    text = (EXAMPLE / "scenarios/nominal.yaml").read_text(encoding="utf-8")
    assert "100 MeV" not in text
    assert "150 MeV" in text


# --- the linac-to-booster line and the booster-to-storage line ----------------------


def _transfer_rows() -> list[Any]:
    elements = _facility_demo("_elements")
    return [row for row in elements.load_elements() if row.section in TRANSFER_SECTIONS]


def _groups() -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in _committed("records/groups.yaml")}


def test_every_ltb_and_bts_table_row_is_a_channel() -> None:
    addresses = {row.demo_name for row in _transfer_rows()}
    assert {address.split(":")[0] for address in addresses} == set(TRANSFER_SECTIONS)
    channels = {a for a in _channels() if a.split(":")[0] in TRANSFER_SECTIONS}
    assert channels == addresses


def test_the_72_frozen_bts_channels_are_table_rows_with_their_frozen_text() -> None:
    assert len(FROZEN_BTS) == 72
    channels = _channels()
    frozen = {row["address"]: row for row in fingerprint_rows() if row["address"] in FROZEN_BTS}
    table = {row.demo_name for row in _transfer_rows()}
    for address, row in frozen.items():
        assert address in table, address
        channel = channels[address]
        assert channel["description"].startswith(row["description"]), address
        assert channel.get("role", "readback") == row["role"], address
        assert channel.get("value_type", "float") == row["value_type"], address
        if ":POSITION:" in address:
            assert channel["unit"] == "mm", address


def test_the_frozen_bts_records_are_not_emitted() -> None:
    transfer = _facility_demo("_transfer")
    emitted = {record["id"] for record in transfer.channels()}
    emitted |= {record["id"] for record in transfer.devices()}
    emitted |= {record["id"] for record in transfer.places()}
    emitted |= {record["id"] for record in transfer.groups()}
    emitted |= {record["id"] for record in _facility_demo("_injector").line_groups()}
    assert emitted & FROZEN_BTS_RECORDS == set()
    assert FROZEN_BTS <= set(_channels())
    assert FROZEN_BTS_DEVICES <= set(_devices())


def test_place_ltb_spans_its_deck_and_holds_its_devices() -> None:
    places = {record["id"]: record for record in _committed("records/places.yaml")}
    ltb = places["LTB"]
    assert ltb["level"] == "machine"
    assert ltb["description"] and ltb["names"]
    assert ltb["span"] == {"model": "LTB", "from_marker": "START"}
    order = list(places)
    assert order.index("GTL") < order.index("LN") < order.index("LTB")
    for key, device in _devices().items():
        if key.split("/")[0] in TRANSFER_SECTIONS:
            assert device["place"] == key.split("/")[0], key


def test_every_ltb_and_bts_family_has_a_group_of_its_devices() -> None:
    groups = _groups()
    families: dict[str, list[str]] = {}
    for key in sorted(_devices()):
        if key.split("/")[0] in TRANSFER_SECTIONS:
            families.setdefault(key.rstrip("0123456789"), []).append(key)
    expected = {
        *(f"LTB/{f}" for f in ("HCM", "VCM", "QF", "QD", "DIPOLE", "BPM", "TV", "ICT")),
        *(f"BTS/{f}" for f in ("HCM", "VCM", "QF", "QD", "DIPOLE", "BPM", "TV", "PADDLES", "ICT")),
    }
    assert set(families) == expected
    for family, members in families.items():
        group = groups[family]
        assert group["members"] == members, family
        assert group["description"] and "\n" not in group["description"], family


def test_the_bts_corrector_groups_grow_to_nine_and_keep_their_description() -> None:
    groups = _groups()
    for family, description in (
        ("HCM", "Horizontal Correctors: Horizontal steering magnets in the transfer line."),
        ("VCM", "Vertical Correctors: Vertical steering magnets in the transfer line."),
    ):
        group = groups[f"BTS/{family}"]
        assert group["members"] == [f"BTS/{family}{n:02d}" for n in range(1, 10)]
        assert group["description"] == description


def test_bts_correctors_07_to_09_exist_with_panel_numbering() -> None:
    channels = _channels()
    for family in ("HCM", "VCM"):
        for number in ("07", "08", "09"):
            prefix = f"BTS:MAG:{family}:{number}"
            sp = channels[f"{prefix}:CURRENT:SP"]
            assert sp["role"] == "setpoint" and sp["pair"] == f"{prefix}:CURRENT:RB"
            assert f"{prefix}:STATUS:ON" in channels
            assert f"{prefix}:STATUS:READY" not in channels
            assert sp["on"] == {"device": f"BTS/{family}{number}"}


def test_bts_corrector_06_follows_its_explicit_table_row() -> None:
    elements = _facility_demo("_elements")
    rows = {row.demo_name: row for row in elements.load_elements()}
    for family in ("HCM", "VCM"):
        prefix = f"BTS:MAG:{family}:06"
        sp = rows[f"{prefix}:CURRENT:SP"]
        assert sp.legacy_pv is None and sp.deck_element == f"BTS:{family}#6"
        assert f"{prefix}:CURRENT:SP" in FROZEN_BTS
        assert f"{prefix}:STATUS:ON" in _channels()
        assert sp.wired
    assert rows["BTS:MAG:HCM:06:CURRENT:SP"].nominal_value() == 1.18
    assert rows["BTS:MAG:VCM:06:CURRENT:SP"].nominal_value() == 0.984
    seeds = _committed("seeds.yaml")
    for family in ("HCM", "VCM"):
        for tail in ("SP", "RB"):
            address = f"BTS:MAG:{family}:06:CURRENT:{tail}"
            assert "nominal" not in seeds.get(address, {}), address


def test_all_paddles_out_is_an_int_with_nominal_zero_in_bts() -> None:
    address = "BTS:DIAG:PADDLES:01:COUNT:INSERTED"
    channel = _channels()[address]
    assert channel["value_type"] == "int"
    assert "role" not in channel
    assert channel["on"]["device"].startswith("BTS/")
    assert _devices()[channel["on"]["device"]]["place"] == "BTS"
    seed = _committed("seeds.yaml")[address]
    assert seed["nominal"] == 0
    assert seed["gate"]["gated"] == 1


def test_the_storage_ring_first_screen_sits_in_bts() -> None:
    elements = _facility_demo("_elements")
    screen = [row for row in elements.load_elements() if row.legacy_pv == "SR01C___TV1____BC17"]
    assert {row.demo_name for row in screen} == {
        "BTS:DIAG:TV:07:PADDLE:SP",
        "BTS:DIAG:TV:07:PADDLE:RB",
    }
    channels = _channels()
    for tail in ("PADDLE:SP", "PADDLE:RB", "LAMP:RB"):
        assert channels[f"BTS:DIAG:TV:07:{tail}"]["on"] == {"device": "BTS/TV07"}
    assert _devices()["BTS/TV07"]["place"] == "BTS"
    assert not [a for a in _channels() if a.startswith("SR:DIAG:TV:")]


def test_every_ltb_and_bts_bpm_reads_millimetres_and_has_a_sum() -> None:
    channels = _channels()
    bpms = {
        a.rsplit(":", 2)[0]
        for a in channels
        if a.split(":")[0] in TRANSFER_SECTIONS and ":BPM:" in a
    }
    assert bpms
    for prefix in bpms:
        for plane in ("X", "Y"):
            assert channels[f"{prefix}:POSITION:{plane}"]["unit"] == "mm", prefix
        assert f"{prefix}:SUM:RB" in channels, prefix


def test_each_new_ltb_and_bts_setpoint_pairs_and_carries_its_nominal_unless_wired() -> None:
    channels = _channels()
    seeds = _committed("seeds.yaml")
    nominal = {row.demo_name: row.nominal_value() for row in _transfer_rows()}
    wired = {row.demo_name for row in _transfer_rows() if row.wired}
    setpoints = {
        a: c for a, c in channels.items() if _is_transfer(a) and c.get("role") == "setpoint"
    }
    assert setpoints
    for address, channel in setpoints.items():
        assert channel["pair"] == f"{address.rsplit(':', 1)[0]}:RB", address
        assert "role" not in channels[channel["pair"]], address
        if address in wired:
            assert "nominal" not in seeds.get(address, {}), address
        else:
            assert seeds[address]["nominal"] == nominal[address], address
        assert "nominal" not in seeds.get(channel["pair"], {}), channel["pair"]


def test_no_ltb_or_bts_channel_has_a_limits_record() -> None:
    addresses = [record["address"] for record in _committed("limits.yaml")["records"]]
    assert [a for a in addresses if a.split(":")[0] in TRANSFER_SECTIONS] == []


def test_the_wired_frozen_bts_seeds_carry_no_nominal_and_keep_their_motion() -> None:
    seeds = _committed("seeds.yaml")
    wired = {row.demo_name for row in _transfer_rows() if row.wired}
    frozen_wired = sorted(FROZEN_BTS & wired)
    assert len(frozen_wired) == 48
    for address in frozen_wired:
        assert "nominal" not in seeds.get(address, {}), address
        if address.endswith(":CURRENT:RB"):
            assert "noise" in seeds[address], address


# --- the booster -------------------------------------------------------------------

BOOSTER_KICKER = "BR:MAG:KICKER:01"


def _booster_channels() -> dict[str, dict[str, Any]]:
    return {address: record for address, record in _channels().items() if _is_booster(address)}


def test_every_element_table_row_is_a_channel() -> None:
    elements = _facility_demo("_elements")
    addresses = {row.demo_name for row in elements.load_elements()}
    assert len(addresses) == 459
    assert addresses <= set(_channels())


def test_every_booster_table_row_is_a_channel_of_a_new_family() -> None:
    elements = _facility_demo("_elements")
    addresses = {row.demo_name for row in elements.load_elements() if row.section == "BR"}
    assert len(addresses) == 39
    assert set(_booster_channels()) == addresses


def test_the_booster_kicker_is_a_kicker_with_its_voltage_pair() -> None:
    device = _devices()["BR/KICKER01"]
    assert device["class"] == "Kicker"
    assert device["place"] == "BR"
    channels = _channels()
    sp, rb, on = (
        channels[f"{BOOSTER_KICKER}:{tail}"] for tail in ("VOLTAGE:SP", "VOLTAGE:RB", "STATUS:ON")
    )
    assert sp["role"] == "setpoint" and sp["pair"] == rb["id"]
    assert sp["signal"] == "voltage_setpoint" and sp["unit"] == "kV"
    assert rb["signal"] == "voltage_readback" and "role" not in rb
    assert on["signal"] == "power_on_status" and on["value_type"] == "bool"
    assert _committed("seeds.yaml")[f"{BOOSTER_KICKER}:VOLTAGE:SP"]["nominal"] == 25.0


@pytest.mark.parametrize(
    ("family", "device_class"),
    [("DIPOLE-RAMP", "Dipole"), ("QF-RAMP", "Quadrupole"), ("QD-RAMP", "Quadrupole")],
)
def test_the_ramp_gain_and_offset_are_writable_stand_ins_in_the_booster(
    family: str, device_class: str
) -> None:
    device = _devices()[f"BR/{family}01"]
    assert device["class"] == device_class and device["place"] == "BR"
    channels = _channels()
    seeds = _committed("seeds.yaml")
    for field, signal, nominal in (("GAIN", "ramp_gain", 1.0), ("OFFSET", "ramp_offset", 0.0)):
        sp = channels[f"BR:MAG:{family}:01:{field}:SP"]
        rb = channels[f"BR:MAG:{family}:01:{field}:RB"]
        assert sp["role"] == "setpoint" and sp["pair"] == rb["id"]
        assert sp["signal"] == f"{signal}_setpoint"
        assert rb["signal"] == f"{signal}_readback"
        assert sp["on"] == rb["on"] == {"device": f"BR/{family}01"}
        assert seeds[sp["id"]]["nominal"] == nominal
        assert "nominal" not in seeds.get(rb["id"], {})
    gain = channels[f"BR:MAG:{family}:01:GAIN:SP"]
    assert "unit" not in gain and "tolerance" not in gain
    assert channels[f"BR:MAG:{family}:01:OFFSET:SP"]["unit"] == "A"


def test_the_booster_septa_are_sen_and_sek_unwired_and_unpulsed() -> None:
    devices = _devices()
    assert devices["BR/SEPTUM01"]["label"] == "SEN"
    assert devices["BR/SEPTUM02"]["label"] == "SEK"
    assert {devices[key]["class"] for key in ("BR/SEPTUM01", "BR/SEPTUM02")} == {"Septum"}
    seeds = _committed("seeds.yaml")
    assert seeds["BR:MAG:SEPTUM:01:CURRENT:SP"]["nominal"] == 800.0
    assert seeds["BR:MAG:SEPTUM:02:CURRENT:SP"]["nominal"] == 3000.0
    septum = [address for address in _booster_channels() if address.startswith("BR:MAG:SEPTUM:")]
    assert len(septum) == 6
    for address in septum:
        assert _channels()[address].get("value_type") != "waveform", address
        assert "pulse" not in seeds.get(address, {}), address
    for model in _committed("models.yaml"):
        for record in model.get("wiring", []):
            assert not str(record["address"]).startswith("BR:MAG:SEPTUM:"), model["name"]


def test_the_booster_screens_are_writable_paddles_with_lamps() -> None:
    channels = _channels()
    for number in range(1, 6):
        prefix = f"BR:DIAG:TV:{number:02d}"
        sp = channels[f"{prefix}:PADDLE:SP"]
        assert sp["role"] == "setpoint" and sp["value_type"] == "bool"
        assert sp["pair"] == f"{prefix}:PADDLE:RB"
        assert channels[f"{prefix}:LAMP:RB"]["signal"] == "lamp_status"
        assert _devices()[f"BR/TV{number:02d}"]["class"] == "ScreenDiagnostic"


def test_every_booster_channel_has_a_description_and_only_its_lamps_carry_a_gate() -> None:
    seeds = _committed("seeds.yaml")
    for address, channel in _booster_channels().items():
        assert channel["description"].startswith("Booster ring "), address
        if channel.get("signal") == "lamp_status":
            paddle = address.removesuffix("LAMP:RB") + "PADDLE:RB"
            assert seeds[address]["gate"] == {"inputs": {paddle: "TRUE"}, "gated": "TRUE"}
        else:
            assert "gate" not in seeds.get(address, {}), address


def test_each_new_booster_family_has_a_one_line_group() -> None:
    groups = {record["id"]: record for record in _committed("records/groups.yaml")}
    devices = _devices()
    for family in BOOSTER_FAMILIES:
        group = groups[f"BR/{family}"]
        assert group["description"] and "\n" not in group["description"], family
        members = sorted(key for key in devices if key.rstrip("0123456789") == f"BR/{family}")
        assert group["members"] == members, family
    assert len(groups["BR/TV"]["members"]) == 5


def test_no_booster_channel_has_a_limits_record() -> None:
    addresses = [record["address"] for record in _committed("limits.yaml")["records"]]
    assert [address for address in addresses if address.startswith("BR:")] == []


# --- the hand-authored descriptions ------------------------------------------------


def _groups_by_id() -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in _committed("records/groups.yaml")}


def test_the_injector_places_carry_their_hand_authored_text_in_beam_order() -> None:
    injector = _facility_demo("_injector")
    places = {record["id"]: record for record in _committed("records/places.yaml")}
    assert tuple(injector.PLACE_TEXT) == INJECTOR_PLACES
    for key, (description, names) in injector.PLACE_TEXT.items():
        place = places[key]
        assert place["level"] == "machine", key
        assert place["description"] == description, key
        assert description.startswith(f"{description.split(' (')[0]} ({key}): "), key
        assert "\n" not in description, key
        assert place["names"] == list(names) and len(names) >= 3, key
        assert key in names, key
    assert [record["id"] for record in injector.places()] == list(INJECTOR_PLACES)


def test_the_bts_place_keeps_its_span_and_describes_the_grown_line() -> None:
    bts = {record["id"]: record for record in _committed("records/places.yaml")}["BTS"]
    assert bts["span"] == {"model": "BTS", "from_marker": "START"}
    assert bts["description"].startswith("Booster-to-Storage Transfer Line (BTS): ")
    for word in ("2.0 GeV", "Bending magnets", "screens", "paddle summary", "current transformer"):
        assert word in bts["description"], word
    assert "booster-to-storage transfer line" in bts["names"]


def test_every_family_group_the_table_adds_carries_its_hand_authored_text() -> None:
    injector = _facility_demo("_injector")
    groups = _groups_by_id()
    emitted = {
        record["id"]
        for emit in (injector.booster_groups, injector.line_groups)
        for record in emit()
    }
    frozen = set(_facility_demo("_transfer").FROZEN_GROUPS)
    assert set(injector.GROUP_TEXT) == emitted - frozen
    for key, (description, names) in injector.GROUP_TEXT.items():
        group = groups[key]
        assert group["description"] == description, key
        assert group["names"] == list(names) and names, key
        assert "\n" not in description, key
        family = key.split("/", 1)[1]
        assert f"({family}): " in description, key


def test_the_bts_groups_that_predate_the_table_keep_their_text() -> None:
    transfer = _facility_demo("_transfer")
    groups = _groups_by_id()
    for key, (description, names) in transfer.FROZEN_GROUPS.items():
        assert groups[key]["description"] == description, key
        assert groups[key]["names"] == list(names), key

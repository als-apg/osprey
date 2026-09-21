"""The synthetic MML fixtures carry the shapes the loader, census and hazard tests rely on.

These checks read the committed files under ``tests/fixtures/mml/`` with a small,
test-local walker that follows the MML shape rules from the proposal directly,
deliberately independent of ``osprey.services.mml``: a fixture that silently lost
its broadcast row or its zero-channel family would otherwise let a loader or
census test pass for the wrong reason.

The ``mapping.yaml`` committed beside each export is read the same way, so the
answers a reviewer writes stay pinned next to the shapes that raise them.
"""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import warnings
from pathlib import Path
from typing import Any

import at
import numpy as np
import pytest
import yaml
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

from tests.templates.mml_export_contract import (
    EXPORT_BLOCK_KEYS,
    EXPORTER_VERSION,
    VA_CALIBRATION_KEYS,
    VA_ENERGY_TABLE_KEYS,
    VA_FAMILY_KEYS,
    VA_LATTICE_KEYS,
    VA_LINEAR_KEYS,
    VA_MONITOR_KEYS,
    VA_NOMINAL_KEYS,
    VA_READOUT_KEYS,
    VA_RESPONSE_BLOCK_KEYS,
    VA_RESPONSE_SIDE_KEYS,
    VA_SETPOINT_KEYS,
    VA_TABLE_KEYS,
    VA_VOCABULARIES,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"

#: Keys whose value is a channel list.
CHANNEL_KEYS = ("ChannelNames", "TangoNames")

#: Real facility names that must never appear in an invented export.
REAL_NAMES = re.compile(
    r"\b(ALS|ALS-?U|LBNL?|Berkeley|Elettra|Trieste|SIRIUS|LNLS|NSLS-?(II|2)?|SPEAR3?|SSRL|"
    r"SLAC|ESRF|SOLEIL|Diamond|BESSY|PETRA|DESY|APS|Argonne|CLS|MAX ?IV|SLS|PSI|"
    r"Australian Synchrotron|CHESS|Spring-?8)\b",
    re.IGNORECASE,
)

ALL_FIXTURES = ("tango", "dualkey", "casedup", "wrapped", "dialect", "paired", "mat")


# ---------------------------------------------------------------------------
# Loading: every fixture becomes {system: {family: body}}.
# ---------------------------------------------------------------------------


def _decode_mat(value: Any) -> Any:
    """Decode a ``loadmat(squeeze_me=True, struct_as_record=False)`` value to plain Python."""
    if isinstance(value, mat_struct):
        return {name: _decode_mat(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_decode_mat(item) for item in value.tolist()]
        if value.dtype.kind == "U":
            return "" if value.size == 0 else [str(row) for row in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_mat(path: Path) -> dict[str, Any]:
    """Return the decoded ``AO`` and ``AD`` variables of a MAT-file."""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {name: _decode_mat(raw[name]) for name in ("AO", "AD") if name in raw}


def _read_json(path: Path) -> Any:
    """Parse JSON the way the loader must: bare ``NaN``/``Infinity`` accepted."""
    return json.loads(path.read_text(encoding="utf-8"))


def _systems(name: str) -> dict[str, dict[str, Any]]:
    """Return the fixture's AO content keyed by system token."""
    root = FIXTURES / name
    if name == "dialect":
        data = _read_json(root / "export.json")
        return {key: value for key, value in data.items() if not key.startswith("_")}
    if name == "wrapped":
        return {"INJ": _read_json(root / "export.json")["ao"]}
    if name == "paired":
        ad = _read_json(root / "quokka.ring.ad.json")
        return {ad["SubMachine"]: _read_json(root / "quokka.ring.ao.json")}
    if name == "mat":
        variables = _load_mat(root / "quokka_booster.mat")
        return {variables["AD"]["SubMachine"]: variables["AO"]}
    token = {"tango": "RING", "dualkey": "STOR", "casedup": "MAIN"}[name]
    return {token: _read_json(root / "export.json")}


# ---------------------------------------------------------------------------
# Shape walker.
# ---------------------------------------------------------------------------


def _is_family(body: object) -> bool:
    """A family is a dict with a channel-keyed sub-dict or a ``DeviceList``."""
    if not isinstance(body, dict):
        return False
    setup = body.get("setup") or body.get("_setup") or {}
    if "DeviceList" in body or "DeviceList" in setup:
        return True
    return any(
        isinstance(sub, dict) and any(key in sub for key in CHANNEL_KEYS) for sub in body.values()
    )


def _families(system: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the families of one system, skipping underscore metadata keys."""
    return {
        name: body for name, body in system.items() if not name.startswith("_") and _is_family(body)
    }


def _fields(family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the field dicts of a family; ``setup`` is never a field."""
    return {
        name: body
        for name, body in family.items()
        if isinstance(body, dict) and not name.startswith("_") and name != "setup"
    }


def _array(family: dict[str, Any], name: str) -> Any:
    """Read a family array from the family level, else from ``setup``/``_setup``."""
    if name in family:
        return family[name]
    setup = family.get("setup") or family.get("_setup") or {}
    return setup.get(name)


def _slots(value: Any) -> list[Any]:
    """Return a channel value as a list of slots; a bare string is one slot."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    return list(value) if isinstance(value, list) else []


def _is_blank(slot: Any) -> bool:
    return slot is None or (isinstance(slot, str) and not slot.strip())


def _n_devices(family: dict[str, Any]) -> int:
    devices = _array(family, "DeviceList")
    if isinstance(devices, list) and devices and all(isinstance(row, list) for row in devices):
        return len(devices)
    if isinstance(devices, list) and len(devices) == 2:
        return 1
    return max(
        (len(_slots(field.get(key))) for field in _fields(family).values() for key in CHANNEL_KEYS),
        default=0,
    )


def _census(name: str) -> dict[str, Any]:
    """Count the five cases every fixture must carry."""
    zero_channel: list[str] = []
    broadcasts: list[str] = []
    blanks: list[str] = []
    owners: dict[str, set[tuple[str, str, int]]] = {}
    hwunits_shapes: set[str] = set()
    for system, body in _systems(name).items():
        for family_name, family in _families(body).items():
            n_devices = _n_devices(family)
            channels = 0
            for field_name, field in _fields(family).items():
                hwunits = field.get("HWUnits")
                if hwunits == []:
                    hwunits_shapes.add("empty")
                elif isinstance(hwunits, str) and hwunits:
                    hwunits_shapes.add("string")
                elif isinstance(hwunits, list):
                    hwunits_shapes.add("per-device")
                for key in CHANNEL_KEYS:
                    slots = _slots(field.get(key))
                    where = f"{system}:{family_name}:{field_name}:{key}"
                    channels += sum(not _is_blank(slot) for slot in slots)
                    if any(_is_blank(slot) for slot in slots):
                        blanks.append(where)
                    if len(slots) == 1 and n_devices > 1:
                        broadcasts.append(where)
                        continue
                    for index, slot in enumerate(slots, start=1):
                        if not _is_blank(slot):
                            owners.setdefault(slot.strip(), set()).add((system, family_name, index))
            if channels == 0:
                zero_channel.append(f"{system}:{family_name}")
    shared = {pv: devices for pv, devices in owners.items() if len(devices) > 1}
    return {
        "zero_channel": zero_channel,
        "broadcasts": broadcasts,
        "blanks": blanks,
        "shared": shared,
        "hwunits_shapes": hwunits_shapes,
    }


def _walk(node: Any, path: tuple[str, ...] = ()):
    """Yield ``(path, value)`` for every value in a nested dict/list tree."""
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, (*path, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, (*path, str(index)))


# ---------------------------------------------------------------------------
# Judgment answers, read straight out of the committed mapping documents.
# ---------------------------------------------------------------------------

#: Every form a judgment answer may take, as the contract words them.
ANSWER_FORMS = frozenset({"drop", "device", "keep", "keep_all", "field:", "owner"})

#: The fixtures whose export raises no judgment, so their mapping carries no block.
FIXTURES_WITHOUT_JUDGMENTS = ("tango", "dualkey")


def _mapping(name: str) -> dict[str, Any]:
    """Return a fixture's committed mapping document in file order.

    ``yaml.safe_load`` keeps this module independent of the product parser and
    leaves the document's own key order intact, so the block order of the file
    is what the contract sees.
    """
    return yaml.safe_load((FIXTURES / name / "mapping.yaml").read_text(encoding="utf-8"))


def _judgment_answers(document: dict[str, Any]) -> list[Any]:
    """Return every answer written in one mapping's ``judgments`` block."""
    answers: list[Any] = []
    for family in (document.get("judgments") or {}).values():
        for signals in (family.get("rows_beyond_devices") or {}).values():
            answers.extend(signals.values())
        answers.extend((family.get("unbound_devices") or {}).values())
        if "shared_pvs" in family:
            shared = family["shared_pvs"]
            answers.extend(shared.values() if isinstance(shared, dict) else [shared])
    return answers


def _answer_form(answer: Any) -> str:
    """Name the contract form of one answer; an unknown answer names itself."""
    if isinstance(answer, str) and answer in ANSWER_FORMS:
        return answer
    if isinstance(answer, dict) and set(answer) == {"field"}:
        return "field:"
    if isinstance(answer, int) and not isinstance(answer, bool):
        return "owner"
    return repr(answer)


def _answer_forms(name: str) -> set[str]:
    """Return the answer forms one fixture's mapping uses."""
    return {_answer_form(answer) for answer in _judgment_answers(_mapping(name))}


def _top_level_keys(name: str) -> list[str]:
    """Return one mapping's top-level keys in the order the file writes them."""
    return list(_mapping(name))


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_FIXTURES)
class TestEveryFixture:
    """Each fixture exercises the census and hazard paths on its own input form."""

    def test_it_has_families(self, name):
        """Every system of the fixture yields at least one family under the shape rule."""
        systems = _systems(name)
        assert systems
        for system, body in systems.items():
            assert _families(body), f"{name}: system {system} has no families"

    def test_it_has_a_zero_channel_family(self, name):
        """A family with devices but no channel is present, so it is reported, never dropped."""
        assert _census(name)["zero_channel"]

    def test_it_has_a_broadcast_row(self, name):
        """A one-slot channel list sits on a family with several devices."""
        assert _census(name)["broadcasts"]

    def test_it_has_a_blank_slot(self, name):
        """A channel list holds an empty or whitespace slot."""
        assert _census(name)["blanks"]

    def test_it_has_a_pv_shared_by_two_devices(self, name):
        """One PV is bound by two distinct devices outside any broadcast row."""
        assert _census(name)["shared"]

    def test_hwunits_come_in_all_three_shapes(self, name):
        """``HWUnits`` appears as ``[]``, as a plain string and as a per-device list."""
        assert _census(name)["hwunits_shapes"] == {"empty", "string", "per-device"}

    def test_it_uses_invented_names_only(self, name):
        """No real facility name appears anywhere in the export content."""
        root = FIXTURES / name
        if name == "mat":
            text = json.dumps(_load_mat(root / "quokka_booster.mat"))
            text += (root / "build_mat.py").read_text(encoding="utf-8")
        else:
            text = "".join(p.read_text(encoding="utf-8") for p in sorted(root.glob("*.json")))
        assert REAL_NAMES.search(text) is None

    def test_the_readme_names_it(self, name):
        """The fixture README has a row for this fixture."""
        readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
        assert f"`{name}/`" in readme


class TestFlatForms:
    """The flat fixtures carry no system token of their own."""

    @pytest.mark.parametrize("name", ["tango", "dualkey", "casedup", "wrapped"])
    def test_flat_exports_need_a_system_flag(self, name):
        """Top-level entries are families (or metadata), with no ``_export.submachine``."""
        data = _read_json(FIXTURES / name / "export.json")
        if name == "wrapped":
            assert list(data) == ["ao"]
            data = data["ao"]
        entries = {key: value for key, value in data.items() if not key.startswith("_")}
        assert entries and all(_is_family(value) for value in entries.values())
        assert "submachine" not in data.get("_export", {})


class TestJudgmentContract:
    """The synthetic mappings together answer every judgment a reviewer can be asked."""

    def test_the_answers_span_every_form_and_add_none_of_their_own(self):
        """Across the synthetic mappings each answer form is used, and nothing else is."""
        forms = set().union(*(_answer_forms(name) for name in ALL_FIXTURES))

        assert forms == set(ANSWER_FORMS)

    def test_a_row_beyond_the_devices_becomes_a_field_of_its_own(self):
        """Some ``{field: <name>}`` answer names a non-empty field."""
        minted = [
            answer
            for name in ALL_FIXTURES
            for answer in _judgment_answers(_mapping(name))
            if _answer_form(answer) == "field:"
        ]

        assert minted
        assert all(isinstance(a["field"], str) and a["field"].strip() for a in minted)

    def test_a_supply_group_names_one_owning_device(self):
        """Some ``shared_pvs`` answer is an owner map of 1-based ordinals, not ``keep_all``."""
        maps = [
            family["shared_pvs"]
            for name in ALL_FIXTURES
            for family in (_mapping(name).get("judgments") or {}).values()
            if isinstance(family.get("shared_pvs"), dict)
        ]

        assert maps
        for owners in maps:
            assert owners
            assert all(isinstance(group, int) and group >= 1 for group in owners)
            assert all(owner == "keep_all" or owner >= 1 for owner in owners.values())

    @pytest.mark.parametrize("name", ALL_FIXTURES)
    def test_the_judgments_block_closes_the_mapping(self, name):
        """A mapping that carries judgments writes them as its last top-level key."""
        keys = _top_level_keys(name)

        if "judgments" in keys:
            assert keys[-1] == "judgments"

    @pytest.mark.parametrize("name", FIXTURES_WITHOUT_JUDGMENTS)
    def test_a_mapping_with_nothing_to_answer_carries_no_block(self, name):
        """An export that raises no judgment leaves the block out of the mapping."""
        assert "judgments" not in _top_level_keys(name)


class TestTango:
    """The Tango fixture binds through ``TangoNames`` alone."""

    def test_every_channel_key_is_tango(self):
        """No field carries ``ChannelNames``; at least one carries ``TangoNames``."""
        fields = [
            field
            for family in _families(_systems("tango")["RING"]).values()
            for field in _fields(family).values()
        ]
        assert not any("ChannelNames" in field for field in fields)
        assert any("TangoNames" in field for field in fields)


class TestDualKey:
    """The dual-key fixture stages both channel keys on one field."""

    def test_a_field_carries_both_keys_slot_for_slot(self):
        """Some field has non-empty ``ChannelNames`` and ``TangoNames`` of equal length."""
        pairs = [
            (_slots(field["ChannelNames"]), _slots(field["TangoNames"]))
            for family in _families(_systems("dualkey")["STOR"]).values()
            for field in _fields(family).values()
            if all(key in field for key in CHANNEL_KEYS)
        ]
        assert any(ca and tango and len(ca) == len(tango) for ca, tango in pairs)


class TestCaseDuplicates:
    """The case-duplicate fixture holds ``BPMx`` and ``bpmx`` in one system."""

    def test_both_spellings_are_families_of_one_system(self):
        """Both families exist and fold to the same lowercase token."""
        families = _families(_systems("casedup")["MAIN"])
        assert {"BPMx", "bpmx"} <= set(families)
        assert "BPMx".lower() == "bpmx".lower()


class TestDialect:
    """The system-keyed dialect excerpt carries every value spelling the normaliser folds."""

    RAW = FIXTURES / "dialect" / "export.json"

    def test_it_is_at_most_thirty_lines(self):
        """The excerpt stays short enough to read in one screen."""
        assert len(self.RAW.read_text(encoding="utf-8").splitlines()) <= 30

    def test_it_is_system_keyed(self):
        """Every top-level non-metadata value is a system holding families, not a family."""
        data = _read_json(self.RAW)
        systems = {key: value for key, value in data.items() if not key.startswith("_")}
        assert len(systems) >= 2
        for body in systems.values():
            assert not _is_family(body)
            assert _families(body)

    def test_quoted_non_finite_strings(self):
        """Non-finite values appear as the quoted strings ``"inf"`` and ``"-inf"``."""
        text = self.RAW.read_text(encoding="utf-8")
        assert '"inf"' in text
        assert '"-inf"' in text

    def test_bare_nan_and_infinity_tokens(self):
        """The bare JSON tokens ``NaN`` and ``Infinity`` are present and parse."""
        seen: set[str] = set()

        def record(token: str) -> float:
            seen.add(token)
            return float("nan")

        json.loads(self.RAW.read_text(encoding="utf-8"), parse_constant=record)
        assert {"NaN", "Infinity"} <= seen

    def test_function_handle_spellings(self):
        """``*Fcn: 1``, a bare-string handle and a ``function_handle`` record under the typo key."""
        values = list(_walk(_read_json(self.RAW)))
        fcn = [(path[-1], value) for path, value in values if path and path[-1].endswith("Fcn")]
        assert any(value == 1 and not isinstance(value, bool) for _, value in fcn)
        assert any(isinstance(value, str) and value for _, value in fcn)
        typo = [value for path, value in values if path and path[-1] == "HW2PhysicSDcn"]
        assert typo and isinstance(typo[0], dict) and "function_handle" in typo[0]

    def test_whitespace_blanks(self):
        """A whitespace-only string (not merely empty) appears."""
        assert any(
            isinstance(value, str) and value and not value.strip()
            for _, value in _walk(_read_json(self.RAW))
        )

    def test_family_arrays_come_from_setup_and_from_the_family(self):
        """At least one family keeps its arrays under ``setup`` and one at family level."""
        families = [
            family for body in _systems("dialect").values() for family in _families(body).values()
        ]
        assert any("DeviceList" in family.get("setup", {}) for family in families)
        assert any("DeviceList" in family for family in families)


class TestPaired:
    """The paired fixture resolves its system from the sibling AD file."""

    def test_one_stem_with_ao_and_ad(self):
        """Exactly one ``<stem>.ao.json`` and its ``<stem>.ad.json`` sit side by side."""
        root = FIXTURES / "paired"
        ao = sorted(p.name for p in root.glob("*.ao.json"))
        ad = sorted(p.name for p in root.glob("*.ad.json"))
        assert len(ao) == 1
        assert ad == [ao[0].replace(".ao.json", ".ad.json")]

    def test_only_the_ad_names_the_sub_machine(self):
        """``AD.SubMachine`` is set and the AO ``_export`` block names no sub-machine."""
        root = FIXTURES / "paired"
        ad = _read_json(root / "quokka.ring.ad.json")
        ao = _read_json(root / "quokka.ring.ao.json")
        assert ad["SubMachine"]
        assert ad["Machine"]
        assert "submachine" not in ao["_export"]


class TestMat:
    """The ``.mat`` fixture is a v7 MAT-file that ``build_mat.py`` reproduces."""

    PATH = FIXTURES / "mat" / "quokka_booster.mat"

    def test_header_is_v7_not_v73(self):
        """Header bytes 124-125 carry version 0x0100 (v7 and older), never HDF5's 0x0200."""
        header = self.PATH.read_bytes()[:128]
        assert header[126:128] in (b"IM", b"MI")
        version = int.from_bytes(header[124:126], "little" if header[126:128] == b"IM" else "big")
        assert version == 0x0100

    def test_it_holds_ao_and_ad(self):
        """Both variables load, and ``AD.SubMachine`` names the system."""
        variables = _load_mat(self.PATH)
        assert set(variables) == {"AO", "AD"}
        assert variables["AD"]["SubMachine"] == "BOOSTER"

    def test_build_mat_reproduces_the_committed_content(self, tmp_path):
        """Re-running ``build_mat.py`` yields the same decoded content as the committed file."""
        module = runpy.run_path(str(FIXTURES / "mat" / "build_mat.py"))
        rebuilt = module["build"](tmp_path / "rebuilt.mat")
        assert _load_mat(rebuilt) == _load_mat(self.PATH)


# ---------------------------------------------------------------------------
# The synthetic 2.0 export.
# ---------------------------------------------------------------------------

SYNTHETIC = FIXTURES / "synthetic"

#: Every file ``synthetic/build.py`` commits.
SYNTHETIC_FILES = (
    "quokka.sr.lattice.mat",
    "mismatched.lattice.mat",
    "quokka.sr.ao.json",
    "quokka.sr.ad.json",
    "quokka.sr.va.json",
    "quokka.sr.response.json",
)

#: The closed vocabularies as sets, for comparing against what the fixture spends.
SYNTHETIC_VOCABULARIES = {name: set(words) for name, words in VA_VOCABULARIES.items()}


def _synthetic(name: str) -> Any:
    """Return one committed JSON file of the synthetic export."""
    return _read_json(SYNTHETIC / name)


def _synthetic_va() -> dict[str, Any]:
    """Return the synthetic ``va.json``."""
    return _synthetic("quokka.sr.va.json")


def _synthetic_calibrations(va: dict[str, Any]) -> list[dict[str, Any]]:
    """Every calibration and inverse the document holds, in no particular order."""
    found = []
    for block in va["families"].values():
        for field in ("Setpoint", "Monitor"):
            body = block.get(field)
            if not body:
                continue
            found.append(body["calibration"])
            if "monitor_inverse" in body:
                found.append(body["monitor_inverse"])
    return found


def _synthetic_ring(name: str, keep_all: bool):
    """Load one of the synthetic MAT-files back as pyAT reads ``THERING``."""
    return at.load_mat(str(SYNTHETIC / name), use="THERING", keep_all=keep_all)


def _synthetic_documents() -> dict[str, Any]:
    """Every JSON document of the synthetic export, by file name."""
    return {name: _synthetic(name) for name in SYNTHETIC_FILES if name.endswith(".json")}


def _assert_conversion(conversion: dict[str, Any], where: str, anchored: bool) -> None:
    """One calibration or inverse, against the frozen key set a conversion is written in."""
    keys = set(conversion)
    assert keys <= set(VA_CALIBRATION_KEYS), where
    assert {"kind", "grid_source", "fcn"} <= keys, where
    # The anchor names the point a hardware grid is laid about, which only a
    # forward calibration lays; a sampled inverse is handed its grid.
    assert ("anchor" in keys) == anchored, where
    linear = conversion["kind"] == "linear"
    carried = set(VA_LINEAR_KEYS) if linear else set(VA_TABLE_KEYS)
    withheld = set(VA_TABLE_KEYS) if linear else set(VA_LINEAR_KEYS)
    assert carried <= keys, where
    assert not withheld & keys, where


class TestSyntheticExport:
    """The synthetic 2.0 export carries every shape of the frozen key set."""

    def test_the_synthetic_fixture_regenerates_byte_identically(self, tmp_path):
        """Re-running ``build.py`` writes the committed bytes, the MAT-files included."""
        module = runpy.run_path(str(SYNTHETIC / "build.py"))
        for rebuilt in module["build"](tmp_path):
            committed = SYNTHETIC / rebuilt.name
            assert committed.read_bytes() == rebuilt.read_bytes(), rebuilt.name

    def test_the_synthetic_lattice_pairs_with_its_own_fingerprint(self):
        """The saved ring answers the four facts ``va.json`` records for it."""
        fingerprint = _synthetic_va()["lattice"]
        ring = _synthetic_ring("quokka.sr.lattice.mat", keep_all=True)
        names = [element.FamName for element in ring]

        assert fingerprint["elements"] == len(ring)
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        assert fingerprint["famname_sha256"] == digest
        assert fingerprint["energy_gev"] > 0
        # One parameter element, at the head of the ring, one-based.
        assert fingerprint["ringparam_indices"] == 1

    def test_the_mismatched_synthetic_lattice_fails_the_fingerprint(self):
        """The counter-example matches on every fact but the digest."""
        fingerprint = _synthetic_va()["lattice"]
        ring = _synthetic_ring("mismatched.lattice.mat", keep_all=True)
        names = [element.FamName for element in ring]

        assert fingerprint["elements"] == len(ring)
        assert fingerprint["energy_gev"] == ring.energy / 1e9
        # The parameter element sits where the fingerprint says it does: it
        # carries the ring's own name, one-based, at the head.
        assert ring[fingerprint["ringparam_indices"] - 1].FamName == ring.name
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        assert fingerprint["famname_sha256"] != digest

    def test_the_synthetic_ring_boots(self):
        """The ring the calibrations were sampled over closes, four-dimensionally and in six."""
        elements = _synthetic_va()["lattice"]["elements"]
        ring = _synthetic_ring("quokka.sr.lattice.mat", keep_all=False)
        ring.disable_6d()
        # Without the parameter element the ring is one shorter than the
        # fingerprint counts, which is the bookkeeping the fingerprint states.
        assert len(ring) == elements - 1

        with warnings.catch_warnings():
            # pyAT hands back its starting guess with a warning when the solve
            # does not converge, so a warning here is a ring that does not close.
            warnings.simplefilter("error", at.AtWarning)
            orbit = at.find_orbit4(ring)[0]
            tunes = ring.get_tune()
            six = at.find_orbit6(ring.enable_6d(copy=True))[0]

        assert np.isfinite(orbit).all()
        assert np.isfinite(tunes).all()
        assert all(0.0 < tune < 1.0 for tune in tunes)
        assert np.isfinite(six).all()

    def test_the_synthetic_va_blocks_stay_inside_the_frozen_key_set(self):
        """Every family block carries frozen keys only, and the required ones."""
        for family, block in _synthetic_va()["families"].items():
            assert set(block) <= set(VA_FAMILY_KEYS), family
            if set(block) == {"refused"}:
                continue
            assert {"device_list", "fields", "energy_candidate"} <= set(block), family

    def test_the_synthetic_export_names_the_version_a_reader_recognises_it_by(self):
        """Every file of the export carries the same provenance block and the 2.0 token."""
        for name, document in _synthetic_documents().items():
            block = document["_export"]
            assert set(block) == set(EXPORT_BLOCK_KEYS), name
            assert block["exporter"] == EXPORTER_VERSION, name

    def test_the_synthetic_va_document_walks_the_frozen_key_set_to_its_leaves(self):
        """Every block under a family — calibration, nominal, energy table — is frozen too."""
        va = _synthetic_va()
        assert set(va) == {"_export", "lattice", "families"}
        assert set(va["lattice"]) == set(VA_LATTICE_KEYS)

        spent: dict[str, set[str]] = {
            "Setpoint": set(),
            "Monitor": set(),
            "calibration": set(),
            "monitor_inverse": set(),
            "readout": set(),
        }
        for family, block in va["families"].items():
            assert set(block) <= set(VA_FAMILY_KEYS), family
            for field, record in block.get("nominals", {}).items():
                assert set(record) == set(VA_NOMINAL_KEYS), f"{family}.{field}"
            for field, frozen in (("Setpoint", VA_SETPOINT_KEYS), ("Monitor", VA_MONITOR_KEYS)):
                body = block.get(field)
                if not body:
                    continue
                assert set(body) <= set(frozen), f"{family}.{field}"
                assert "calibration" in body, f"{family}.{field}"
                spent[field] |= set(body)
                readout = body.get("readout")
                if readout:
                    assert set(readout) <= set(VA_READOUT_KEYS), f"{family}.{field}.readout"
                    spent["readout"] |= set(readout)
                for role in ("calibration", "monitor_inverse"):
                    conversion = body.get(role)
                    if not conversion:
                        continue
                    _assert_conversion(
                        conversion, f"{family}.{field}.{role}", anchored=role == "calibration"
                    )
                    spent[role].add(conversion["kind"])
            table = block.get("energy_table")
            if table:
                assert set(table) == set(VA_ENERGY_TABLE_KEYS), family

        # The fixture is the reader's worked example, so it spends the whole of
        # each key set rather than a corner of it.
        assert spent["Setpoint"] == set(VA_SETPOINT_KEYS)
        assert spent["Monitor"] == set(VA_MONITOR_KEYS)
        assert spent["readout"] == set(VA_READOUT_KEYS)
        assert spent["calibration"] == spent["monitor_inverse"] == SYNTHETIC_VOCABULARIES["kind"]

    def test_the_synthetic_response_document_stays_inside_the_frozen_key_set(self):
        """The response document's blocks and both their sides carry frozen keys only."""
        response = _synthetic("quokka.sr.response.json")
        assert set(response) == {"_export", "file", "blocks"}
        for index, block in enumerate(response["blocks"]):
            assert set(block) == set(VA_RESPONSE_BLOCK_KEYS), index
            for side in ("monitor", "actuator"):
                assert set(block[side]) == set(VA_RESPONSE_SIDE_KEYS), f"{index}.{side}"

    def test_the_synthetic_vocabularies_are_closed_and_every_word_is_used(self):
        """Each closed vocabulary is honoured, and the fixture spends all of it."""
        va = _synthetic_va()
        seen = {name: set() for name in SYNTHETIC_VOCABULARIES}
        for calibration in _synthetic_calibrations(va):
            for name in ("kind", "grid_source", "anchor"):
                if name in calibration:
                    seen[name].add(calibration[name])
        for block in va["families"].values():
            if "energy_scaling" in block.get("Setpoint", {}):
                seen["energy_scaling"].add(block["Setpoint"]["energy_scaling"])
        assert seen == SYNTHETIC_VOCABULARIES

    def test_the_synthetic_sliced_kick_carries_one_nan_slice(self):
        """One corrector device is made of one element where the others are made of two."""
        rows = _synthetic("quokka.sr.ao.json")["HC"]["AT"]["ATIndex"]
        finite = [sum(1 for entry in row if entry != "NaN") for row in rows]
        assert sorted(finite) == [1, 2, 2, 2]
        assert sum(row.count("NaN") for row in rows) == 1

    def test_the_synthetic_inverse_is_not_the_calibration_read_backwards(self):
        """A family's sampled inverse differs from the inverse of its own calibration."""
        monitor = _synthetic_va()["families"]["QF"]["Monitor"]
        gain = monitor["calibration"]["gain"][0]
        inverse = monitor["monitor_inverse"]["gain"][0]
        assert monitor["monitor_inverse"]["kind"] == "linear"
        assert inverse != pytest.approx(1.0 / gain, rel=1e-3)

    def test_the_synthetic_table_calibration_stops_where_its_numbers_do(self):
        """A conversion with an end writes ``"NaN"`` past it and records the span."""
        calibration = _synthetic_va()["families"]["BEND"]["Setpoint"]["calibration"]
        assert calibration["kind"] == "table"
        for grid, values, span in zip(
            calibration["grid"], calibration["values"], calibration["finite_span"], strict=True
        ):
            assert "NaN" in values
            assert span[0] == grid[0]
            assert span[1] < grid[-1]

    def test_the_synthetic_energy_tables_come_both_flat_and_not(self):
        """One energy candidate's table moves with its current and one does not."""
        families = _synthetic_va()["families"]
        candidates = {
            name: block for name, block in families.items() if block.get("energy_candidate")
        }
        flat = {
            name
            for name, block in candidates.items()
            if len(set(block["energy_table"]["values"])) == 1
        }
        assert flat and set(candidates) - flat
        for block in candidates.values():
            assert block["energy_table"]["energy_at_nominal"] > 0

    def test_the_synthetic_export_carries_both_shapes_of_refusal(self):
        """One block is a refusal alone; another keeps its facts beside one."""
        families = _synthetic_va()["families"]
        alone = [name for name, block in families.items() if set(block) == {"refused"}]
        beside = [
            name
            for name, block in families.items()
            if "refused" in block and set(block) != {"refused"}
        ]
        assert alone and beside
        assert any("nominals" in families[name] for name in beside)

    def test_the_synthetic_nominal_outside_its_range_stretches_the_band(self):
        """A family whose nominal sits outside its own band keeps that band, widened.

        ``Range`` is the band the facility runs the family in, and it stays
        the band the conversion is sampled over: a nominal beyond it stretches
        the sampling far enough to reach the nominal rather than throwing the
        band away for a symmetric guess around it.
        """
        va = _synthetic_va()
        band = _synthetic("quokka.sr.ao.json")["HC"]["Setpoint"]["Range"]
        nominal = va["families"]["HC"]["nominals"]["Setpoint"]["values"]
        assert max(abs(value) for value in nominal) > band[1]
        assert va["families"]["HC"]["Setpoint"]["calibration"]["grid_source"] == "range"

    def test_the_synthetic_escape_hatch_carries_both_spellings(self):
        """The escape-hatch family names a replacement write path and a parameter group."""
        block = _synthetic("quokka.sr.ao.json")["IDGAP"]["AT"]
        assert "$fn" in block["SpecialFunctionSet"]
        assert isinstance(block["ATParameterGroup"], str)

    def test_the_synthetic_export_pairs_families_on_one_element(self):
        """Two families bind the same lattice elements, so one element serves both."""
        ao = _synthetic("quokka.sr.ao.json")
        bound = {}
        for family, body in ao.items():
            index = body.get("AT", {}).get("ATIndex") if isinstance(body, dict) else None
            if isinstance(index, list) and index and not isinstance(index[0], list):
                bound[family] = tuple(index)
        pairs = [
            (one, other)
            for one in bound
            for other in bound
            if one < other and bound[one] == bound[other]
        ]
        assert pairs

    def test_the_synthetic_one_device_family_is_written_flat(self):
        """A one-device family's rows are flat arrays and its one index a bare number."""
        block = _synthetic_va()["families"]["RF"]
        assert block["device_list"] == [1, 1]
        nominals = block["nominals"]["Setpoint"]
        assert isinstance(nominals["values"], float)
        assert isinstance(nominals["at_index"], (int, float))

    def test_the_synthetic_response_matrix_lines_up_with_its_device_lists(self):
        """Every block's matrix is one row per monitor and one column per corrector."""
        response = _synthetic("quokka.sr.response.json")
        assert response["file"] == ""
        assert response["blocks"]
        for block in response["blocks"]:
            monitors = len(block["monitor"]["device_list"])
            actuators = len(block["actuator"]["device_list"])
            assert len(block["data"]) == monitors
            assert all(len(row) == actuators for row in block["data"])
            assert len(block["monitor"]["status"]) == monitors
            assert len(block["actuator"]["status"]) == actuators
            model = {block["monitor"]["mode"], block["actuator"]["mode"]} & {"Simulator", "Model"}
            assert block["origin"] == ("model" if model else "measured")

    def test_the_synthetic_response_matrix_marks_the_device_it_lost(self):
        """A device the file does not hold is flagged down and its row is no number."""
        blocks = _synthetic("quokka.sr.response.json")["blocks"]
        dropped = [
            (block, index)
            for block in blocks
            for index, flag in enumerate(block["monitor"]["status"])
            if flag == 0
        ]
        assert dropped
        for block, index in dropped:
            assert all(entry == "NaN" for entry in block["data"][index])
            assert block["monitor"]["data"][index] == "NaN"

    def test_the_synthetic_export_uses_invented_names_only(self):
        """No real facility name appears in the fixture, its builder or its README."""
        text = "".join(
            (SYNTHETIC / name).read_text(encoding="utf-8")
            for name in SYNTHETIC_FILES
            if name.endswith(".json")
        )
        text += (SYNTHETIC / "build.py").read_text(encoding="utf-8")
        text += (SYNTHETIC / "README.md").read_text(encoding="utf-8")
        assert REAL_NAMES.search(text) is None

    def test_the_synthetic_readme_names_every_committed_file(self):
        """The fixture README has a row for each file the builder writes."""
        readme = (SYNTHETIC / "README.md").read_text(encoding="utf-8")
        for name in SYNTHETIC_FILES:
            assert name in readme

"""The synthetic MML fixtures carry the shapes the loader, census and hazard tests rely on.

These checks read the committed files under ``tests/fixtures/mml/`` with a small,
test-local walker that follows the MML shape rules from the proposal directly,
deliberately independent of ``osprey.services.mml``: a fixture that silently lost
its broadcast row or its zero-channel family would otherwise let a loader or
census test pass for the wrong reason.
"""

from __future__ import annotations

import json
import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

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

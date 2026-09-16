"""Tests for the channel-database emitter ``build_channel_db``.

The emitted dict is the depth-3 ``middle_layer.json`` the channel finder reads.
Every case writes it through ``json.dumps(..., sort_keys=False)`` and loads it
with ``MiddleLayerDatabase``, so what is pinned is what the loader sees: system
order from ``section_order``, sorted families and fields, zero-channel families
omitted, the ``setup`` block, the per-field key allowlist, blank and broadcast
slots, prose under ``_description`` and the top-level ``_provenance`` string.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase
from osprey.services.mml.emit.channel_db import FIELD_METADATA_KEYS, build_channel_db
from osprey.services.mml.emit.context import EmitContext, build_context
from osprey.services.mml.mapping.schema import Facility, Family, Field, Mapping, System
from osprey.services.mml.systems import EXPORTS_KEY, IMPORT_ORDER_KEY

#: Every key a field dict may carry, per the FR4 allowlist.
ALLOWED_FIELD_KEYS = frozenset(
    {
        "ChannelNames",
        "TangoNames",
        "Description",
        "DataType",
        "Mode",
        "Units",
        "HWUnits",
        "PhysicsUnits",
        "MemberOf",
        "Range",
        "Tolerance",
    }
)


def _ctx(tmp_path: Path, ao: dict) -> EmitContext:
    ao_path = tmp_path / "ao.json"
    mapping_path = tmp_path / "mapping.yaml"
    ao_path.write_bytes(json.dumps(ao).encode())
    mapping_path.write_bytes(b"facility:\n  token: quokka\n")
    return build_context(ao_path, mapping_path, ao)


def _family(
    raw: str,
    *,
    channels: int = 1,
    rename: str | None = None,
    description: str | None = None,
    fields: dict[str, str | None] | None = None,
) -> Family:
    return Family(
        raw=raw,
        rename=rename,
        branch=None,
        class_="BPM" if channels else None,
        aliases=(),
        description=description if description is not None else f"{raw} family",
        provenance="human",
        channels=channels,
        fields={
            name: Field(description=text, provenance="human")
            for name, text in (fields or {}).items()
        },
    )


def _mapping(
    systems: dict[str, tuple[str, str | None]],
    section_order: tuple[str, ...],
    families: list[Family],
) -> Mapping:
    return Mapping(
        facility=Facility(token="quokka", title=None, description=None, provenance="human"),
        systems={
            raw: System(raw=raw, name=name, description=text, provenance="human")
            for raw, (name, text) in systems.items()
        },
        section_order=section_order,
        families={family.raw: family for family in families},
    )


def _load(tmp_path: Path, db: dict) -> MiddleLayerDatabase:
    path = tmp_path / "middle_layer.json"
    path.write_text(json.dumps(db, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False))
    return MiddleLayerDatabase(str(path))


def _bpm_body() -> dict:
    return {
        "DeviceList": [[1, 1], [1, 2]],
        "CommonNames": ["BPM1", "BPM2"],
        "Monitor": {
            "ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"],
            "Units": "Hardware",
            "HWUnits": "mm",
        },
    }


class TestSystemOrder:
    def test_systems_follow_section_order_not_ao_order(self, tmp_path: Path) -> None:
        ao = {
            EXPORTS_KEY: {},
            IMPORT_ORDER_KEY: ["BOOSTER", "RING", "GTL"],
            "BOOSTER": {"BPMx": _bpm_body()},
            "GTL": {"BPMx": _bpm_body()},
            "RING": {"BPMx": _bpm_body()},
        }
        mapping = _mapping(
            {
                "BOOSTER": ("BR", "The booster"),
                "GTL": ("GTL", "Gun to linac"),
                "RING": ("SR", "The storage ring"),
            },
            ("SR", "BR", "GTL"),
            [_family("BPMx", fields={"Monitor": "Horizontal position"})],
        )

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        assert [s["name"] for s in loaded.list_systems()] == ["SR", "BR", "GTL"]
        assert [s["description"] for s in loaded.list_systems()] == [
            "The storage ring",
            "The booster",
            "Gun to linac",
        ]
        assert loaded.get_statistics()["systems"] == 3
        assert [key for key in db if not key.startswith("_")] == ["SR", "BR", "GTL"]

    def test_order_entry_without_ao_system_is_refused(self, tmp_path: Path) -> None:
        ao = {"RING": {"BPMx": _bpm_body()}}
        mapping = _mapping(
            {"RING": ("SR", None), "BOOSTER": ("BR", None)},
            ("SR", "BR"),
            [_family("BPMx")],
        )
        with pytest.raises(ValueError, match="BR"):
            build_channel_db(ao, mapping, _ctx(tmp_path, ao))

    def test_ao_system_left_out_of_order_is_refused(self, tmp_path: Path) -> None:
        ao = {"RING": {"BPMx": _bpm_body()}, "BOOSTER": {"BPMx": _bpm_body()}}
        mapping = _mapping(
            {"RING": ("SR", None), "BOOSTER": ("BR", None)},
            ("SR",),
            [_family("BPMx")],
        )
        with pytest.raises(ValueError, match="BOOSTER"):
            build_channel_db(ao, mapping, _ctx(tmp_path, ao))


class TestFamilies:
    def test_families_are_sorted_by_mapped_name_and_fields_sorted(self, tmp_path: Path) -> None:
        body = _bpm_body()
        body["Setpoint"] = {"ChannelNames": ["SR:C1:SP", "SR:C2:SP"]}
        ao = {
            "RING": {
                "QF": {"Monitor": {"ChannelNames": ["SR:QF"]}},
                "BPMx": body,
                "HCM": {"Monitor": {"ChannelNames": ["SR:HCM"]}},
            }
        }
        mapping = _mapping(
            {"RING": ("SR", "Ring")},
            ("SR",),
            [
                _family("QF"),
                _family("BPMx", rename="BPM_X"),
                _family("HCM", rename="Corrector_H"),
            ],
        )

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        assert [f["name"] for f in loaded.list_families("SR")] == ["BPM_X", "Corrector_H", "QF"]
        field_keys = [k for k in db["SR"]["BPM_X"] if not k.startswith("_") and k != "setup"]
        assert field_keys == ["Monitor", "Setpoint"]

    def test_zero_channel_mapping_family_is_omitted(self, tmp_path: Path) -> None:
        ao = {
            "RING": {
                "BPMx": _bpm_body(),
                "DCCT": {"Monitor": {"ChannelNames": ["SR:DCCT"]}},
            }
        }
        mapping = _mapping(
            {"RING": ("SR", "Ring")},
            ("SR",),
            [_family("BPMx"), _family("DCCT", channels=0)],
        )

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        assert [f["name"] for f in loaded.list_families("SR")] == ["BPMx"]
        assert loaded.get_statistics()["families"] == 1

    def test_family_without_bound_channel_in_this_system_is_omitted(self, tmp_path: Path) -> None:
        ao = {
            "RING": {"BPMx": _bpm_body(), "Blank": {"Monitor": {"ChannelNames": [None, " "]}}},
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx"), _family("Blank")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert "Blank" not in db["SR"]

    def test_family_missing_from_mapping_is_refused(self, tmp_path: Path) -> None:
        ao = {"RING": {"BPMx": _bpm_body()}}
        mapping = _mapping({"RING": ("SR", None)}, ("SR",), [])
        with pytest.raises(ValueError, match="BPMx"):
            build_channel_db(ao, mapping, _ctx(tmp_path, ao))

    def test_underscore_ao_keys_are_skipped(self, tmp_path: Path) -> None:
        ao = {
            EXPORTS_KEY: {"RING": {"exporter": "1.2"}},
            IMPORT_ORDER_KEY: ["RING"],
            "RING": {"_meta": {"x": 1}, "BPMx": _bpm_body()},
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert set(db) == {"SR", "_provenance"}
        assert set(db["SR"]) == {"_description", "BPMx"}

    def test_family_and_system_prose_under_description(self, tmp_path: Path) -> None:
        ao = {"RING": {"BPMx": _bpm_body()}}
        mapping = _mapping(
            {"RING": ("SR", "The storage ring")},
            ("SR",),
            [_family("BPMx", description="Horizontal beam position monitors")],
        )

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        assert db["SR"]["_description"] == "The storage ring"
        assert loaded.list_families("SR") == [
            {"name": "BPMx", "description": "Horizontal beam position monitors"}
        ]


class TestSetup:
    def test_device_list_carries_common_names_status_position(self, tmp_path: Path) -> None:
        body = _bpm_body()
        body["setup"] = {"Status": [1, 0], "Position": [1.5, 3.0], "ElementList": [1, 2]}
        ao = {"RING": {"BPMx": body}}
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert db["SR"]["BPMx"]["setup"] == {
            "DeviceList": [[1, 1], [1, 2]],
            "CommonNames": ["BPM1", "BPM2"],
            "Status": [1, 0],
            "Position": [1.5, 3.0],
        }
        info = _load(tmp_path, db).inspect_fields("SR", "BPMx")
        assert info["setup"]["type"] == "metadata"

    def test_flat_device_pair_is_normalised_to_n_by_2(self, tmp_path: Path) -> None:
        ao = {"RING": {"DCCT": {"DeviceList": [1, 1], "Monitor": {"ChannelNames": ["SR:DCCT"]}}}}
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("DCCT")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert db["SR"]["DCCT"]["setup"] == {"DeviceList": [[1, 1]]}

    def test_no_device_list_keeps_only_common_names_and_status(self, tmp_path: Path) -> None:
        ao = {
            "RING": {
                "BPMx": {
                    "CommonNames": ["BPM1", None],
                    "Status": [1, 1],
                    "Position": [1.0, 2.0],
                    "Monitor": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]},
                }
            }
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert db["SR"]["BPMx"]["setup"] == {"CommonNames": ["BPM1", ""], "Status": [1, 1]}

    def test_an_unaligned_array_is_left_out_of_setup(self, tmp_path: Path) -> None:
        """A one-slot CommonNames on a two-device family names no device at all."""
        ao = {
            "RING": {
                "QF": {
                    "DeviceList": [[1, 1], [2, 1]],
                    "CommonNames": ["qf"],
                    "Monitor": {"ChannelNames": ["SR:QF1:RB", "SR:QF2:RB"]},
                }
            }
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("QF")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert db["SR"]["QF"]["setup"] == {"DeviceList": [[1, 1], [2, 1]]}

    def test_setup_is_placed_after_description_and_before_fields(self, tmp_path: Path) -> None:
        ao = {"RING": {"BPMx": _bpm_body()}}
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert list(db["SR"]["BPMx"]) == ["_description", "setup", "Monitor"]


class TestFields:
    def test_field_keys_are_within_the_allowlist(self, tmp_path: Path) -> None:
        body = _bpm_body()
        body["Monitor"].update(
            {
                "DataType": "Scalar",
                "Mode": "Online",
                "PhysicsUnits": "m",
                "MemberOf": ["BPM", "Monitor"],
                "Range": [-10, 10],
                "Tolerance": 0.1,
                "HW2PhysicsParams": 0.001,
                "HW2PhysicsFcn": {"$fn": "mm2m", "file": None},
                "SpecialFunctionGet": "getx",
                "_description": "imported prose",
                "Description": "imported field prose",
            }
        )
        ao = {"RING": {"BPMx": body}}
        mapping = _mapping(
            {"RING": ("SR", "Ring")},
            ("SR",),
            [_family("BPMx", fields={"Monitor": "Horizontal position readback"})],
        )

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        monitor = db["SR"]["BPMx"]["Monitor"]
        assert set(monitor) <= ALLOWED_FIELD_KEYS
        assert list(monitor) == [
            "ChannelNames",
            "Description",
            "DataType",
            "Mode",
            "Units",
            "HWUnits",
            "PhysicsUnits",
            "MemberOf",
            "Range",
            "Tolerance",
        ]
        assert monitor["Description"] == "Horizontal position readback"
        details = loaded.inspect_fields("SR", "BPMx", "Monitor")
        assert set(details) <= ALLOWED_FIELD_KEYS
        assert all(entry["type"] != "dict (subfield)" for entry in details.values())
        top = loaded.inspect_fields("SR", "BPMx")
        assert top["Monitor"] == {
            "type": "ChannelNames",
            "description": "Horizontal position readback",
        }
        channel = loaded.get_channel("SR:BPM1:X")
        assert channel["HWUnits"] == "mm"
        assert channel["Description"] == "Horizontal position readback"

    def test_metadata_keys_constant_matches_loader_allowlist(self) -> None:
        assert FIELD_METADATA_KEYS == (
            "DataType",
            "Mode",
            "Units",
            "HWUnits",
            "PhysicsUnits",
            "MemberOf",
            "Range",
            "Tolerance",
        )

    def test_only_present_channel_keys_are_written(self, tmp_path: Path) -> None:
        ao = {
            "RING": {
                "BPMx": {
                    "Monitor": {"TangoNames": ["sr/bpm/1/x"]},
                    "Both": {"ChannelNames": ["SR:A"], "TangoNames": ["sr/a/1"]},
                }
            }
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        assert set(db["SR"]["BPMx"]["Monitor"]) == {"TangoNames"}
        assert set(db["SR"]["BPMx"]["Both"]) == {"ChannelNames", "TangoNames"}
        assert loaded.get_channel("sr/bpm/1/x")["protocol"] == "tango"

    def test_non_field_subdicts_are_dropped(self, tmp_path: Path) -> None:
        body = _bpm_body()
        body["pyAT"] = {"ATIndex": [1, 2]}
        body["Setpoint"] = {"X": {"ChannelNames": ["SR:SP:X"]}}
        ao = {"RING": {"BPMx": body}}
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert list(db["SR"]["BPMx"]) == ["_description", "setup", "Monitor"]

    def test_blank_slots_become_empty_strings(self, tmp_path: Path) -> None:
        ao = {
            "RING": {
                "BPMx": {
                    "DeviceList": [[1, 1], [1, 2], [1, 3]],
                    "Monitor": {"ChannelNames": ["SR:BPM1:X", None, "SR:BPM3:X"]},
                }
            }
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        loaded = _load(tmp_path, db)

        assert db["SR"]["BPMx"]["Monitor"]["ChannelNames"] == ["SR:BPM1:X", "", "SR:BPM3:X"]
        assert loaded.get_statistics()["total_channels"] == 2

    def test_broadcast_row_is_expanded_to_n_devices(self, tmp_path: Path) -> None:
        ao = {
            "RING": {
                "QF": {
                    "DeviceList": [[1, 1], [1, 2], [2, 1]],
                    "Setpoint": {"ChannelNames": ["SR:QF:SP"]},
                }
            }
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("QF")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert db["SR"]["QF"]["Setpoint"]["ChannelNames"] == ["SR:QF:SP"] * 3

    def test_short_list_is_kept_as_exported(self, tmp_path: Path) -> None:
        ao = {
            "RING": {
                "BPMx": {
                    "DeviceList": [[1, 1], [1, 2], [1, 3]],
                    "Monitor": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]},
                }
            }
        }
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))

        assert db["SR"]["BPMx"]["Monitor"]["ChannelNames"] == ["SR:BPM1:X", "SR:BPM2:X"]

    def test_input_ao_is_not_mutated(self, tmp_path: Path) -> None:
        body = _bpm_body()
        body["Monitor"]["MemberOf"] = ["BPM"]
        ao = {"RING": {"BPMx": body}}
        snapshot = json.loads(json.dumps(ao))
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])

        db = build_channel_db(ao, mapping, _ctx(tmp_path, ao))
        db["SR"]["BPMx"]["Monitor"]["MemberOf"].append("changed")
        db["SR"]["BPMx"]["setup"]["DeviceList"][0][0] = 99

        assert ao == snapshot


class TestProvenance:
    def test_top_level_provenance_is_the_context_string(self, tmp_path: Path) -> None:
        ao = {EXPORTS_KEY: {"RING": {"exporter": "3.1"}}, "RING": {"BPMx": _bpm_body()}}
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("BPMx")])
        ctx = _ctx(tmp_path, ao)

        db = build_channel_db(ao, mapping, ctx)
        loaded = _load(tmp_path, db)

        assert db["_provenance"] == ctx.provenance_string
        assert isinstance(db["_provenance"], str)
        assert loaded.get_statistics()["systems"] == 1
        assert [s["name"] for s in loaded.list_systems()] == ["SR"]

    def test_serialisation_is_deterministic(self, tmp_path: Path) -> None:
        ao = {"RING": {"BPMx": _bpm_body(), "QF": {"Monitor": {"ChannelNames": ["SR:QF"]}}}}
        mapping = _mapping({"RING": ("SR", "Ring")}, ("SR",), [_family("QF"), _family("BPMx")])
        ctx = _ctx(tmp_path, ao)

        first = json.dumps(build_channel_db(ao, mapping, ctx), sort_keys=False)
        second = json.dumps(build_channel_db(ao, mapping, ctx), sort_keys=False)

        assert first == second

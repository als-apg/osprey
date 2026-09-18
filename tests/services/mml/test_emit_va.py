"""Tests for the virtual-accelerator emitters in ``emit/va.py``.

``emit_machine`` and ``emit_state_channels`` write the two documents a served
deployment starts from, ``emit_lattice`` saves the deck, ``emit_bindings`` says
what each coupled address does to it, and ``emit_channel_limits`` says what a
write to each address may do. So every case here reads the emitted text back
through the readers that consume it in production -- ``parse_machine`` (the
simulation engine's own parser), ``load_machine_json_channels``, the
machine-state loader's key scan, ``load_bindings`` and
``LimitsValidator._load_limits_database`` -- rather than through assertions on
a dict the emitter happened to build.

What is pinned: every hardware nominal reaches its addresses, a family the
model does not drive is marked ``nominal_seed_only``, a physics-units or
non-finite nominal is refused rather than seeded, the provenance stamp is the
first key of both documents (the pre-flight reads it there), the machine-state
list holds monitor-only single-device families and nothing else, and both
documents are byte-stable across re-emits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.mml.emit.context import LATTICE_ARTIFACT, EmitContext, build_context
from osprey.services.mml.emit.va import (
    PROVENANCE_KEY,
    SEED_ONLY_KEY,
    emit_bindings,
    emit_channel_limits,
    emit_lattice,
    emit_machine,
    emit_state_channels,
)
from osprey.services.mml.family import FamilyView
from osprey.services.mml.judgments import judged_family_views, judged_va_block
from osprey.services.mml.mapping.schema import (
    Direction,
    Facility,
    Family,
    FamilyJudgments,
    Field,
    Mapping,
    System,
    VAFamily,
    VirtualAccelerator,
)
from osprey.services.mml.va.elements import ElementBinding, ElementSlice
from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey_connectors.control_system.limits_validator import LimitsValidator
from osprey_connectors.simulation.machine import parse_machine

SYSTEM = "RING"


def _ctx(tmp_path: Path) -> EmitContext:
    ao_path = tmp_path / "ao.json"
    mapping_path = tmp_path / "mapping.yaml"
    ao_path.write_bytes(b'{"RING": {}}')
    mapping_path.write_bytes(b"facility:\n  token: quokka\n")
    return build_context(ao_path, mapping_path, {})


def _family(
    raw: str,
    *,
    fields: dict[str, str | None] | None = None,
    description: str | None = None,
    rename: str | None = None,
) -> Family:
    return Family(
        raw=raw,
        rename=rename,
        branch=None,
        class_="BPM",
        aliases=(),
        description=description,
        provenance="human",
        channels=1,
        fields={
            name: Field(description=text, provenance="human")
            for name, text in (fields or {}).items()
        },
    )


def _mapping(
    families: list[Family],
    directions: dict[str, str | None] | None = None,
    *,
    judgments: dict[str, FamilyJudgments] | None = None,
    verdicts: dict[str, VAFamily] | None = None,
    title: str | None = "Quokka storage ring",
) -> Mapping:
    return Mapping(
        facility=Facility(token="quokka", title=title, description=None, provenance="human"),
        systems={SYSTEM: System(raw=SYSTEM, name="SR", description=None, provenance="human")},
        section_order=("SR",),
        families={family.raw: family for family in families},
        directions={
            key: Direction(direction=value, provenance="rule", override=False)
            for key, value in (directions or {}).items()
        },
        judgments=judgments or {},
        virtual_accelerator=VirtualAccelerator(system=SYSTEM, families=verdicts or {}),
    )


def _view(raw: str, body: dict) -> FamilyView:
    return FamilyView(SYSTEM, raw, body)


def _nominals(field: str, values, **extra) -> dict:
    nominal = {
        "values": values,
        "units": "Hardware",
        "at_type": "K",
        "at_index": [1],
        "synthetic": 0,
    }
    nominal.update(extra)
    return {"nominals": {field: nominal}}


def _quad_body() -> dict:
    return {
        "DeviceList": [[1, 1], [1, 2]],
        "Setpoint": {
            "ChannelNames": ["SR:QF:1:SP", "SR:QF:2:SP"],
            "HWUnits": "A",
            "MemberOf": ["Magnet", "Setpoint"],
        },
        "Monitor": {"ChannelNames": ["SR:QF:1:RB", "SR:QF:2:RB"], "HWUnits": "A"},
    }


def _dcct_body() -> dict:
    return {
        "DeviceList": [1, 1],
        "Monitor": {"ChannelNames": ["SR:DCCT:CURRENT"], "HWUnits": "mA"},
    }


def _coupled() -> VAFamily:
    return VAFamily(verdict="couple", kind="strength", element_field="PolynomB[1]")


def _machine(tmp_path: Path, *, views, judged_va, verdicts=None, mapping=None):
    """Emit machine.json and return (document, seeds, text)."""
    mapping = mapping or _mapping([_family("QF", fields={"Setpoint": "Quadrupole current"})])
    text, seeds = emit_machine(verdicts or {}, views, judged_va, mapping, _ctx(tmp_path))
    return json.loads(text), seeds, text


class TestMachineSeedsEveryNominal:
    def test_machine_json_seeds_one_channel_per_device(self, tmp_path):
        document, seeds, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, -2.5])},
            verdicts={(SYSTEM, "QF"): _coupled()},
        )
        assert document["channels"]["SR:QF:1:SP"]["value"] == 1.5
        assert document["channels"]["SR:QF:2:SP"]["value"] == -2.5
        assert [seed.address for seed in seeds] == ["SR:QF:1:SP", "SR:QF:2:SP"]

    def test_machine_json_carries_the_hardware_units_and_prose(self, tmp_path):
        document, _, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
            verdicts={(SYSTEM, "QF"): _coupled()},
        )
        entry = document["channels"]["SR:QF:1:SP"]
        assert entry["units"] == "A"
        assert entry["description"] == "Quadrupole current"

    def test_machine_json_broadcasts_a_scalar_nominal_to_every_device(self, tmp_path):
        document, _, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", 7.0)},
            verdicts={(SYSTEM, "QF"): _coupled()},
        )
        values = [entry["value"] for entry in document["channels"].values()]
        assert values == [7.0, 7.0]

    def test_machine_json_skips_a_device_whose_slot_names_no_channel(self, tmp_path):
        body = _quad_body()
        body["Setpoint"]["ChannelNames"] = ["SR:QF:1:SP", ""]
        document, seeds, _ = _machine(
            tmp_path,
            views=[_view("QF", body)],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
            verdicts={(SYSTEM, "QF"): _coupled()},
        )
        assert list(document["channels"]) == ["SR:QF:1:SP"]
        assert [seed.device for seed in seeds] == [1]

    def test_machine_json_seeds_both_channel_keys_of_one_device(self, tmp_path):
        body = _dcct_body()
        body["Monitor"]["TangoNames"] = ["sr/dcct/1/current"]
        document, _, _ = _machine(
            tmp_path,
            views=[_view("DCCT", body)],
            judged_va={(SYSTEM, "DCCT"): _nominals("Monitor", 300.0)},
            mapping=_mapping([_family("DCCT", fields={"Monitor": None})]),
        )
        assert sorted(document["channels"]) == ["SR:DCCT:CURRENT", "sr/dcct/1/current"]


class TestMachineMarksWhatTheModelDoesNotDrive:
    def test_machine_json_marks_a_latched_family_seed_only(self, tmp_path):
        document, seeds, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5], at_type="SEXT")},
            verdicts={(SYSTEM, "QF"): VAFamily(verdict="latch", reason="unanswered slot")},
        )
        assert document["channels"]["SR:QF:1:SP"][SEED_ONLY_KEY] is True
        assert {seed.seed_only for seed in seeds} == {True}
        assert {seed.at_type for seed in seeds} == {"SEXT"}

    def test_machine_json_marks_an_unjudged_family_seed_only(self, tmp_path):
        document, _, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
            verdicts={},
        )
        assert document["channels"]["SR:QF:1:SP"][SEED_ONLY_KEY] is True

    def test_machine_json_leaves_a_coupled_family_unmarked(self, tmp_path):
        document, _, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
            verdicts={(SYSTEM, "QF"): _coupled()},
        )
        assert SEED_ONLY_KEY not in document["channels"]["SR:QF:1:SP"]


class TestMachineRefusals:
    def test_machine_json_refuses_a_physics_units_nominal(self, tmp_path):
        document, seeds, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5], units="Physics")},
        )
        assert document["channels"] == {}
        assert len(seeds) == 1
        assert "not hardware" in seeds[0].refused

    def test_machine_json_refuses_a_non_finite_nominal(self, tmp_path):
        document, seeds, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", ["NaN", 2.5])},
        )
        assert list(document["channels"]) == ["SR:QF:2:SP"]
        assert [seed.refused for seed in seeds] == ["non-finite nominal 'NaN'", None]

    def test_machine_json_refuses_rows_that_do_not_match_the_judged_devices(self, tmp_path):
        with pytest.raises(ValueError, match="no longer line up"):
            _machine(
                tmp_path,
                views=[_view("QF", _quad_body())],
                judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5, 3.5])},
            )

    def test_machine_json_refuses_a_block_judged_for_other_devices(self, tmp_path):
        block = _nominals("Setpoint", [1.5, 2.5])
        block["device_list"] = [[1, 1], [1, 2], [1, 3]]
        with pytest.raises(ValueError, match="device"):
            _machine(tmp_path, views=[_view("QF", _quad_body())], judged_va={(SYSTEM, "QF"): block})

    def test_machine_json_refuses_a_second_value_on_one_address(self, tmp_path):
        body = _quad_body()
        body["Setpoint"]["ChannelNames"] = ["SR:QF:BOTH:SP"]
        document, seeds, _ = _machine(
            tmp_path,
            views=[_view("QF", body)],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
        )
        assert document["channels"]["SR:QF:BOTH:SP"]["value"] == 1.5
        assert [seed.refused for seed in seeds][0] is None
        assert "already seeded with 1.5" in seeds[1].refused

    def test_machine_json_skips_a_family_with_no_virtual_accelerator_block(self, tmp_path):
        document, seeds, _ = _machine(tmp_path, views=[_view("QF", _quad_body())], judged_va={})
        assert document["channels"] == {}
        assert seeds == ()


class TestMachineDocumentShape:
    def test_machine_json_stamps_the_provenance_first(self, tmp_path):
        ctx = _ctx(tmp_path)
        text, _ = emit_machine(
            {},
            [_view("QF", _quad_body())],
            {(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
            _mapping([_family("QF", fields={"Setpoint": None})]),
            ctx,
        )
        document = json.loads(text)
        assert next(iter(document)) == PROVENANCE_KEY
        assert document[PROVENANCE_KEY] == ctx.provenance_string

    def test_machine_json_loads_through_the_readers_that_consume_it(self, tmp_path):
        from osprey.services.virtual_accelerator.manifest.loaders import (
            load_machine_json_channels,
        )

        _, _, text = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
        )
        path = tmp_path / "machine.json"
        path.write_text(text)
        assert load_machine_json_channels(path)["SR:QF:1:SP"]["value"] == 1.5
        parsed = parse_machine(json.loads(text), path)
        assert parsed.channels["SR:QF:1:SP"].value == 1.5
        assert parsed.channels["SR:QF:1:SP"].units == "A"
        assert parsed.name == "Quokka storage ring"

    def test_machine_json_is_byte_stable_and_sorted(self, tmp_path):
        views = [_view("QF", _quad_body()), _view("DCCT", _dcct_body())]
        judged = {
            (SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5]),
            (SYSTEM, "DCCT"): _nominals("Monitor", 300.0),
        }
        mapping = _mapping([_family("QF"), _family("DCCT")])
        first, _, _ = _machine(tmp_path, views=views, judged_va=judged, mapping=mapping)
        _, _, again = _machine(
            tmp_path, views=list(reversed(views)), judged_va=judged, mapping=mapping
        )
        assert list(first["channels"]) == sorted(first["channels"])
        assert json.dumps(first) == json.dumps(json.loads(again))

    def test_machine_json_is_named_after_the_facility(self, tmp_path):
        document, _, _ = _machine(
            tmp_path,
            views=[_view("QF", _quad_body())],
            judged_va={(SYSTEM, "QF"): _nominals("Setpoint", [1.5, 2.5])},
            mapping=_mapping([_family("QF")], title=None),
        )
        assert document["name"] == "quokka"


class TestMachineReadsJudgedRows:
    def test_machine_json_seeds_the_devices_a_judgment_left(self, tmp_path):
        body = {
            "DeviceList": [[1, 1], [1, 2], [1, 3]],
            "Setpoint": {"ChannelNames": ["SR:QF:1:SP", "SR:QF:2:SP"], "HWUnits": "A"},
        }
        mapping = _mapping(
            [_family("QF", fields={"Setpoint": None})],
            judgments={"QF": FamilyJudgments(unbound_devices={3: "drop"})},
            verdicts={},
        )
        va_json = {
            SYSTEM: {
                "families": {
                    "QF": {
                        "device_list": [[1, 1], [1, 2], [1, 3]],
                        "nominals": {
                            "Setpoint": {
                                "values": [1.5, 2.5, 3.5],
                                "units": "Hardware",
                                "at_type": "K",
                                "at_index": [1, 2, 3],
                                "synthetic": 0,
                            }
                        },
                    }
                }
            }
        }
        views = list(judged_family_views(SYSTEM, {"QF": body}, mapping))
        judged = {
            (SYSTEM, "QF"): judged_va_block(
                SYSTEM, "QF", va_json, mapping, devices=views[0].n_devices
            )
        }
        document, seeds, _ = _machine(tmp_path, views=views, judged_va=judged, mapping=mapping)
        assert [entry["value"] for entry in document["channels"].values()] == [1.5, 2.5]
        assert [seed.device for seed in seeds] == [1, 2]


class TestStateChannelsListGlobalMonitors:
    def _emit(self, tmp_path, views, mapping) -> dict:
        return json.loads(emit_state_channels(views, mapping, _ctx(tmp_path)))

    def test_state_channels_list_a_monitor_only_single_device_family(self, tmp_path):
        mapping = _mapping(
            [_family("DCCT", fields={"Monitor": None}, description="Beam current")],
            {"DCCT.Monitor": "read"},
        )
        document = self._emit(tmp_path, [_view("DCCT", _dcct_body())], mapping)
        assert document["SR:DCCT:CURRENT"] == {"label": "Beam current", "group": "sr"}

    def test_state_channels_leave_out_a_per_device_monitor_family(self, tmp_path):
        body = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["SR:BPM:1:X", "SR:BPM:2:X"]},
        }
        mapping = _mapping([_family("BPMx", fields={"Monitor": None})], {"BPMx.Monitor": "read"})
        document = self._emit(tmp_path, [_view("BPMx", body)], mapping)
        assert [key for key in document if not key.startswith("_")] == []

    def test_state_channels_leave_out_a_family_with_a_written_field(self, tmp_path):
        body = {
            "DeviceList": [1, 1],
            "Setpoint": {"ChannelNames": ["SR:RF:1:SP"]},
            "Monitor": {"ChannelNames": ["SR:RF:1:RB"]},
        }
        mapping = _mapping(
            [_family("RF", fields={"Setpoint": None, "Monitor": None})],
            {"RF.Setpoint": "write", "RF.Monitor": "read"},
        )
        document = self._emit(tmp_path, [_view("RF", body)], mapping)
        assert [key for key in document if not key.startswith("_")] == []

    def test_state_channels_leave_out_a_family_no_direction_states(self, tmp_path):
        mapping = _mapping([_family("DCCT", fields={"Monitor": None})], {"DCCT.Monitor": None})
        document = self._emit(tmp_path, [_view("DCCT", _dcct_body())], mapping)
        assert [key for key in document if not key.startswith("_")] == []

    def test_state_channels_name_the_field_when_a_family_reads_several(self, tmp_path):
        body = {
            "DeviceList": [1, 1],
            "Monitor": {"ChannelNames": ["SR:TUNE:X"]},
            "Lifetime": {"ChannelNames": ["SR:TUNE:LIFE"]},
        }
        mapping = _mapping(
            [_family("TUNE", fields={"Monitor": None}, description="Measured tune")],
            {"TUNE.Monitor": "read", "TUNE.Lifetime": "read"},
        )
        document = self._emit(tmp_path, [_view("TUNE", body)], mapping)
        assert document["SR:TUNE:X"]["label"] == "Measured tune (Monitor)"
        assert document["SR:TUNE:LIFE"]["label"] == "Measured tune (Lifetime)"

    def test_state_channels_label_a_family_with_no_prose_by_its_mapped_token(self, tmp_path):
        mapping = _mapping(
            [_family("DCCT", fields={"Monitor": None}, rename="BeamCurrent")],
            {"DCCT.Monitor": "read"},
        )
        document = self._emit(tmp_path, [_view("DCCT", _dcct_body())], mapping)
        assert document["SR:DCCT:CURRENT"]["label"] == "BeamCurrent"


class TestStateChannelsDocumentShape:
    def test_state_channels_stamp_the_provenance_first(self, tmp_path):
        ctx = _ctx(tmp_path)
        mapping = _mapping([_family("DCCT", fields={"Monitor": None})], {"DCCT.Monitor": "read"})
        document = json.loads(emit_state_channels([_view("DCCT", _dcct_body())], mapping, ctx))
        assert next(iter(document)) == PROVENANCE_KEY
        assert document[PROVENANCE_KEY] == ctx.provenance_string

    def test_state_channels_are_read_by_the_loader_that_scans_them(self, tmp_path):
        from osprey.services.virtual_accelerator.manifest.loaders import (
            load_machine_state_candidate_addresses,
        )
        from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

        mapping = _mapping([_family("DCCT", fields={"Monitor": None})], {"DCCT.Monitor": "read"})
        text = emit_state_channels([_view("DCCT", _dcct_body())], mapping, _ctx(tmp_path))
        (tmp_path / "machine_state_channels.json").write_text(text)
        paths = ManifestPaths(tmp_path)
        assert load_machine_state_candidate_addresses(paths) == ["SR:DCCT:CURRENT"]

    def test_state_channels_are_byte_stable_and_sorted(self, tmp_path):
        views = [
            _view("DCCT", _dcct_body()),
            _view("TUNE", {"Monitor": {"ChannelNames": "SR:TUNE:X"}}),
        ]
        mapping = _mapping(
            [_family("DCCT", fields={"Monitor": None}), _family("TUNE", fields={"Monitor": None})],
            {"DCCT.Monitor": "read", "TUNE.Monitor": "read"},
        )
        first = emit_state_channels(views, mapping, _ctx(tmp_path))
        again = emit_state_channels(list(reversed(views)), mapping, _ctx(tmp_path))
        assert first == again
        addresses = [key for key in json.loads(first) if not key.startswith("_")]
        assert addresses == sorted(addresses)


SYNTHETIC = Path(__file__).resolve().parents[2] / "fixtures" / "mml" / "synthetic"


def _synthetic(suffix: str) -> dict:
    """Return one committed file of the synthetic 2.0 export."""
    return json.loads((SYNTHETIC / f"quokka.sr.{suffix}.json").read_text())


class TestMachineOnTheCommittedExport:
    """The emitter against the only committed 2.0 export, shapes and all.

    The fixture states each of the cases the rules turn on: a per-device
    hardware nominal (QF), a scalar one on a one-device family (RF), a
    monitor-only family's reading (DCCT), a nominal MML could only answer in
    physics units (SEPTUM) and one it could not answer at all (IDGAP, ``NaN``
    per device).
    """

    @pytest.fixture
    def emitted(self, tmp_path):
        from osprey.services.mml.normalize import normalize_family

        ao = _synthetic("ao")
        views = [
            _view(raw, normalize_family(body))
            for raw, body in ao.items()
            if not raw.startswith("_") and isinstance(body, dict)
        ]
        judged_va = {(SYSTEM, raw): block for raw, block in _synthetic("va")["families"].items()}
        text, seeds = emit_machine(
            {(SYSTEM, "QF"): _coupled()}, views, judged_va, _mapping([]), _ctx(tmp_path)
        )
        return json.loads(text), seeds, text

    def test_machine_json_seeds_the_exports_hardware_nominals(self, emitted):
        document, _, _ = emitted
        channels = document["channels"]
        assert channels["QK:QF:1:CUR:SP"] == {"value": 120.0, "units": "Ampere"}
        assert channels["QK:RF:1:CUR:SP"]["value"] == 516.883548276
        assert channels["QK:DCCT:1:CUR:RB"] == {
            "value": 300.0,
            "units": "mA",
            SEED_ONLY_KEY: True,
        }

    def test_machine_json_seeds_only_the_field_the_nominal_was_read_through(self, emitted):
        document, _, _ = emitted
        assert "QK:QF:1:CUR:RB" not in document["channels"]
        assert "QK:BPMx:1:CUR:RB" in document["channels"]

    def test_machine_json_refuses_the_exports_physics_and_non_finite_nominals(self, emitted):
        document, seeds, _ = emitted
        refused = {(seed.family, seed.refused) for seed in seeds if seed.refused}
        assert ("SEPTUM", "nominal read in Physics units, not hardware") in refused
        assert ("IDGAP", "non-finite nominal 'NaN'") in refused
        assert "QK:SEPTUM:1:CUR:RB" not in document["channels"]
        assert "QK:IDGAP:1:CUR:SP" not in document["channels"]

    def test_machine_json_of_the_export_names_only_exported_channels(self, emitted):
        document, _, _ = emitted
        exported = {
            name
            for body in _synthetic("ao").values()
            if isinstance(body, dict)
            for field in body.values()
            if isinstance(field, dict)
            for name in _listed(field.get("ChannelNames"))
        }
        assert set(document["channels"]) <= exported

    def test_machine_json_of_the_export_parses_as_a_machine_file(self, emitted, tmp_path):
        _, _, text = emitted
        path = tmp_path / "machine.json"
        path.write_text(text)
        parsed = parse_machine(json.loads(text), path)
        assert parsed.channels["QK:QF:1:CUR:SP"].value == 120.0


def _listed(value) -> list[str]:
    """Return a channel-name entry as the list of names it states."""
    if isinstance(value, str):
        return [value]
    return [name for name in value if isinstance(name, str)] if isinstance(value, list) else []


# --- lattice.json (task 2.10) ------------------------------------------------
#
# ``emit_lattice`` writes the deck the bindings document binds against, so the
# cases below read the emitted file back through the consumers that serve it --
# ``at.load_lattice`` and ``build_ring`` -- rather than through assertions on
# the dict pyAT happened to render. Imports stay local to these helpers so the
# block is self-contained.


def _deck():
    """The committed synthetic export's saved ring, as the exporter sampled it."""
    from osprey.services.mml.loaders.mat import load_lattice

    return load_lattice(SYNTHETIC / "quokka.sr.lattice.mat")


def _addressed():
    """The same ring after the addressing pass every virtual-accelerator emit runs."""
    from osprey.services.mml.normalize import normalize_family
    from osprey.services.mml.va.elements import address_elements
    from osprey.services.mml.va.verdicts import propose

    ring = _deck()
    views = {
        raw: _view(raw, normalize_family(body))
        for raw, body in _synthetic("ao").items()
        if not raw.startswith("_") and isinstance(body, dict)
    }
    block = _synthetic("va")
    return address_elements(block, ring, propose(block, ring, views))


def _saved_elements(ring):
    """The elements pyAT saves: the deck minus its ``RingParam`` markers.

    Their content is saved as the lattice properties instead, so the saved
    document is one element shorter than the ring the ``.mat`` was read into.
    """
    import at

    return [
        element
        for element in ring
        if not (isinstance(element, at.Marker) and getattr(element, "tag", None) == "RingParam")
    ]


def _emit_lattice(tmp_path, ring=None):
    """Emit one lattice into a served tree, returning ``(path, text, ctx)``."""
    ctx = _ctx(tmp_path)
    path = tmp_path / "data" / "simulation" / "lattice.json"
    text = emit_lattice(_deck() if ring is None else ring, path, ctx)
    return path, text, ctx


def _bindings_for(addressing, digest):
    """A minimal bindings document over one addressed device of the deck."""
    from osprey.services.virtual_accelerator.bindings import (
        Binding,
        BindingsDocument,
        Linear,
        Slice,
    )

    device = addressing.bindings["QF"][0]
    return BindingsDocument(
        system="sr",
        energy_gev=2.0,
        lattice_sha256=digest,
        bindings=(
            Binding(
                kind="strength",
                family="QF",
                setpoint_address="QK:QF:1:CUR:SP",
                readback_address="QK:QF:1:CUR:RB",
                readback="identity",
                element=device.element,
                attribute="PolynomB",
                index=1,
                slices=(Slice(element=device.element, weight=1.0),),
                owner=device.owner,
                calibration=Linear(gain=0.01, offset=0.0),
                monitor_inverse=None,
                nominal=120.0,
                energy_scaling="brho",
                energy_table=None,
            ),
        ),
    )


class TestLatticeJsonIsTheDeckWithoutItsPyATVersion:
    """The emitted file is pyAT's own document, minus the one churning key."""

    def test_the_emitted_lattice_is_pyats_document_shape(self, tmp_path):
        deck = _deck()
        _, text, _ = _emit_lattice(tmp_path)
        document = json.loads(text)
        assert document["atjson"] == 1
        assert isinstance(document["properties"], dict)
        assert len(document["elements"]) == len(_saved_elements(deck)) == len(deck) - 1

    def test_the_at_version_key_is_stripped_from_the_emitted_lattice(self, tmp_path):
        _, text, _ = _emit_lattice(tmp_path)
        assert "at_version" not in json.loads(text)

    def test_the_pyat_version_is_recorded_in_the_emit_context_instead(self, tmp_path):
        import at

        _, _, ctx = _emit_lattice(tmp_path)
        assert ctx.pyat_version == ".".join(at.__version__.split(".")[:3])

    def test_the_lattice_file_holds_exactly_the_returned_text(self, tmp_path):
        path, text, _ = _emit_lattice(tmp_path)
        assert text.endswith("}\n")
        assert path.read_text(encoding="utf-8") == text


class TestLatticeJsonIsByteStableAndRecorded:
    """A re-emit of an unchanged deck leaves the file alone, digest and all."""

    def test_a_second_emit_of_the_same_deck_writes_the_same_bytes(self, tmp_path):
        path, text, _ = _emit_lattice(tmp_path)
        stamp = path.stat().st_mtime_ns
        second = emit_lattice(_deck(), path, _ctx(tmp_path))
        assert second == text
        assert path.read_bytes() == text.encode("utf-8")
        assert path.stat().st_mtime_ns == stamp

    def test_rendering_without_writing_records_the_same_digest_and_no_file(self, tmp_path):
        # The digest is taken over the text, so a caller that must know every
        # refusal before it touches the tree can render first and write later.
        path = tmp_path / "data" / "simulation" / "lattice.json"
        ctx = _ctx(tmp_path)

        text = emit_lattice(_deck(), path, ctx, write=False)
        assert not path.exists()

        _, written, written_ctx = _emit_lattice(tmp_path)

        assert text == written
        assert ctx.lattice_sha256 == written_ctx.lattice_sha256
        assert ctx.pyat_version == written_ctx.pyat_version

    def test_the_context_records_the_sha256_of_the_written_text(self, tmp_path):
        from osprey.services.mml.canonical import sha256_of

        path, _, ctx = _emit_lattice(tmp_path)
        assert ctx.lattice_sha256 == sha256_of(path)

    def test_emitting_the_lattice_leaves_the_exported_mat_untouched(self, tmp_path):
        from osprey.services.mml.canonical import sha256_of

        source = SYNTHETIC / "quokka.sr.lattice.mat"
        before = sha256_of(source)
        _emit_lattice(tmp_path)
        assert sha256_of(source) == before


class TestLatticeJsonLoadsThroughItsRealConsumer:
    """The served virtual accelerator reads back the file this lane writes."""

    def test_at_load_lattice_round_trips_the_renamed_deck(self, tmp_path):
        import at

        addressing = _addressed()
        path, _, _ = _emit_lattice(tmp_path, addressing.ring)
        loaded = at.load_lattice(path)
        saved = _saved_elements(addressing.ring)
        assert [element.FamName for element in loaded] == [element.FamName for element in saved]
        assert [element.Length for element in loaded] == [element.Length for element in saved]

    def test_the_ring_parameters_survive_the_round_trip_as_lattice_properties(self, tmp_path):
        import at

        deck = _deck()
        path, _, _ = _emit_lattice(tmp_path, deck)
        loaded = at.load_lattice(path)
        assert loaded.energy == deck.energy
        assert loaded.periodicity == deck.periodicity
        assert loaded.name == deck.name

    def test_build_ring_serves_the_emitted_lattice_by_its_recorded_digest(self, tmp_path):
        from osprey.services.virtual_accelerator.bindings import dump_bindings
        from osprey.services.virtual_accelerator.lattice import build_ring
        from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

        addressing = _addressed()
        ctx = _ctx(tmp_path)
        data_root = tmp_path / "data"
        emit_lattice(addressing.ring, data_root / "simulation" / "lattice.json", ctx)
        document = _bindings_for(addressing, ctx.lattice_sha256)
        (data_root / "simulation" / "va_bindings.json").write_text(
            dump_bindings(document), encoding="utf-8"
        )

        ring = build_ring(ManifestPaths(data_root=data_root))

        bound = document.bindings[0].slices[0].element
        assert [element.FamName for element in ring].count(bound) == 1


# -- va_bindings.json ---------------------------------------------------------
#
# What is pinned below: the document loads through the reader the served
# virtual accelerator uses rather than through assertions on a dict; the
# readback of a coupled setpoint collapses to the written value only when the
# facility's own inverse returns it; a split kick is shared over its pieces
# while a strength is written whole to each; the energy knob binds one address
# and no element; and a per-device row that no longer counts the family's
# devices is refused rather than read at the wrong device.


#: Stand-in for the lattice text the lane writes just before the bindings; only
#: its digest reaches the document.
LATTICE_TEXT = '{"elements": "the deck this run wrote"}'


def _bindings_ctx(tmp_path: Path, *, lattice: bool = True) -> EmitContext:
    """An emit context that has already written a lattice, as the lane does."""
    ctx = _ctx(tmp_path)
    if lattice:
        ctx.record_artifact(LATTICE_ARTIFACT, LATTICE_TEXT, writer="pyat 0.8.0")
    return ctx


def _linear(gain, offset) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _element(family: str, kind: str, device, attribute, index, *names: str) -> ElementBinding:
    """One device's element row, as ``address_elements`` hands it over."""
    return ElementBinding(
        family=family,
        kind=kind,
        device=device,
        attribute=attribute,
        index=index,
        slices=tuple(
            ElementSlice(element=name, position=slot, slot=slot + 1, owner=family)
            for slot, name in enumerate(names)
        ),
    )


def _quad_block(*, inverse=None, monitor: bool = True) -> dict:
    block: dict = {
        "device_list": [[1, 1], [1, 2]],
        "nominals": {
            "Setpoint": {
                "values": [1.5, 2.5],
                "units": "Hardware",
                "at_type": "K",
                "at_index": [1, 2],
                "synthetic": 0,
            }
        },
        "Setpoint": {
            "calibration": _linear([0.01, 0.01], [0.0, 0.0]),
            "energy_scaling": "brho",
        },
    }
    if monitor:
        block["Monitor"] = {
            "calibration": _linear([0.01, 0.01], [0.0, 0.0]),
            "monitor_inverse": _linear(inverse or [100.0, 100.0], [0.0, 0.0]),
        }
    return block


def _quad_elements() -> dict:
    return {
        "QF": (
            _element("QF", "strength", (1, 1), "PolynomB", 1, "QF_1_1"),
            _element("QF", "strength", (1, 2), "PolynomB", 1, "QF_1_2"),
        )
    }


def _bpm_body() -> dict:
    return {
        "DeviceList": [[1, 1], [1, 2]],
        "Monitor": {"ChannelNames": ["SR:BPM:1:X", "SR:BPM:2:X"], "HWUnits": "mm"},
    }


def _bpm_block(*, inverse: bool = True) -> dict:
    block: dict = {
        "device_list": [[1, 1], [1, 2]],
        "nominals": {
            "Monitor": {
                "values": [0.1, -0.2],
                "units": "Hardware",
                "at_type": "BPMx",
                "at_index": [3, 4],
                "synthetic": 0,
            }
        },
        "Monitor": {"calibration": _linear([0.001, 0.001], [0.0, 0.0])},
    }
    if inverse:
        block["Monitor"]["monitor_inverse"] = _linear([1000.0, 1000.0], [0.0, 0.0])
    return block


def _bpm_elements() -> dict:
    return {
        "BPMx": (
            _element("BPMx", "monitor", (1, 1), "x", None, "BPMx_1_1"),
            _element("BPMx", "monitor", (1, 2), "x", None, "BPMx_1_2"),
        )
    }


def _monitor_verdict() -> VAFamily:
    return VAFamily(verdict="couple", kind="monitor", element_field="x", nominal_source="Monitor")


def _kick_body() -> dict:
    return {"DeviceList": [1, 1], "Setpoint": {"ChannelNames": ["SR:HC:1:SP"], "HWUnits": "A"}}


def _kick_block() -> dict:
    return {
        "device_list": [1, 1],
        "nominals": {
            "Setpoint": {
                "values": 1.5,
                "units": "Hardware",
                "at_type": "HCM",
                "at_index": [[9, 10]],
                "synthetic": 0,
            }
        },
        "Setpoint": {"calibration": _linear(0.0001, 0.0), "energy_scaling": "brho"},
    }


def _energy_body() -> dict:
    return {
        "DeviceList": [[1, 1], [1, 2]],
        "Setpoint": {"ChannelNames": ["SR:BEND:1:SP", "SR:BEND:2:SP"], "HWUnits": "A"},
        "Monitor": {"ChannelNames": ["SR:BEND:1:RB", "SR:BEND:2:RB"], "HWUnits": "A"},
    }


def _energy_block(*, table: bool = True) -> dict:
    block: dict = {
        "device_list": [[1, 1], [1, 2]],
        "nominals": {
            "Setpoint": {
                "values": [420.0, 420.0],
                "units": "Hardware",
                "at_type": "BEND",
                "at_index": [5, 6],
                "synthetic": 0,
            }
        },
        "Setpoint": {
            "calibration": _linear([0.01, 0.01], [0.0, 0.0]),
            "energy_scaling": "brho",
        },
        "energy_candidate": 1,
    }
    if table:
        block["energy_table"] = {
            "device_row": [1, 2],
            "grid": [0.0, 210.0, 420.0, 630.0],
            "values": [0.0, 1.0, 2.0, "NaN"],
            "finite_span": [0.0, 420.0],
            "I_nom": 420.0,
            "energy_at_nominal": 2.0,
        }
    return block


def _energy_elements() -> dict:
    return {
        "BEND": (
            _element("BEND", "energy", (1, 1), None, None, "BEND_1_1"),
            _element("BEND", "energy", (1, 2), None, None, "BEND_1_2"),
        )
    }


def _energy_verdict() -> VAFamily:
    return VAFamily(verdict="couple", kind="energy", calibration="table", nominal_source="Setpoint")


def _bindings(
    tmp_path: Path,
    *,
    views=None,
    judged_va=None,
    verdicts=None,
    elements=None,
    energy_gev: float = 2.0,
    ctx: EmitContext | None = None,
):
    """Emit va_bindings.json over the quadrupole case, or whatever is passed."""
    text = emit_bindings(
        {(SYSTEM, "QF"): _coupled()} if verdicts is None else verdicts,
        [_view("QF", _quad_body())] if views is None else views,
        _quad_elements() if elements is None else elements,
        {(SYSTEM, "QF"): _quad_block()} if judged_va is None else judged_va,
        ctx or _bindings_ctx(tmp_path),
        system=SYSTEM,
        energy_gev=energy_gev,
    )
    path = tmp_path / "va_bindings.json"
    path.write_text(text)
    return text, load_bindings(path)


def _bindings_from_export(tmp_path: Path, directory: Path, stem: str, system: str):
    """Run the whole VA lane over one export on disk and read back what it wrote."""
    from osprey.services.mml.loaders.mat import load_lattice
    from osprey.services.mml.normalize import normalize_family
    from osprey.services.mml.va.elements import address_elements
    from osprey.services.mml.va.verdicts import propose

    ao = json.loads((directory / f"{stem}.ao.json").read_text())
    va = json.loads((directory / f"{stem}.va.json").read_text())
    ring = load_lattice(directory / f"{stem}.lattice.mat")
    views = {
        raw: FamilyView(system, raw, normalize_family(body))
        for raw, body in ao.items()
        if not raw.startswith("_") and isinstance(body, dict)
    }
    verdicts = propose(va, ring, views)
    text = emit_bindings(
        {(system, name): verdict for name, verdict in verdicts.items()},
        list(views.values()),
        dict(address_elements(va, ring, verdicts).bindings),
        {(system, name): block for name, block in va["families"].items()},
        _bindings_ctx(tmp_path),
        system=system,
        energy_gev=va["lattice"]["energy_gev"],
    )
    path = tmp_path / "va_bindings.json"
    path.write_text(text)
    return load_bindings(path), text


class TestBindingsDocumentShape:
    def test_bindings_load_through_the_reader_the_service_boots_from(self, tmp_path):
        _, document = _bindings(tmp_path)
        assert document.system == SYSTEM
        assert document.energy_gev == 2.0
        first = document.bindings[0]
        assert (first.kind, first.family) == ("strength", "QF")
        assert first.setpoint_address == "SR:QF:1:SP"
        assert (first.element, first.attribute, first.index) == ("QF_1_1", "PolynomB", 1)
        assert first.owner == "QF"
        assert first.nominal == 1.5

    def test_bindings_stamp_the_provenance_first(self, tmp_path):
        ctx = _bindings_ctx(tmp_path)
        text, document = _bindings(tmp_path, ctx=ctx)
        assert next(iter(json.loads(text))) == PROVENANCE_KEY
        assert document.provenance == ctx.provenance_string

    def test_bindings_name_the_lattice_this_run_wrote(self, tmp_path):
        ctx = _bindings_ctx(tmp_path)
        _, document = _bindings(tmp_path, ctx=ctx)
        assert document.lattice_sha256 == ctx.lattice_sha256

    def test_bindings_refuse_to_be_written_before_the_lattice(self, tmp_path):
        with pytest.raises(ValueError, match="before the bindings"):
            _bindings(tmp_path, ctx=_bindings_ctx(tmp_path, lattice=False))

    def test_bindings_are_byte_stable_across_re_emits(self, tmp_path):
        views = [_view("QF", _quad_body()), _view("BPMx", _bpm_body())]
        verdicts = {(SYSTEM, "QF"): _coupled(), (SYSTEM, "BPMx"): _monitor_verdict()}
        judged = {(SYSTEM, "QF"): _quad_block(), (SYSTEM, "BPMx"): _bpm_block()}
        elements = {**_quad_elements(), **_bpm_elements()}
        first, _ = _bindings(
            tmp_path, views=views, verdicts=verdicts, judged_va=judged, elements=elements
        )
        again, _ = _bindings(
            tmp_path,
            views=list(reversed(views)),
            verdicts=dict(reversed(list(verdicts.items()))),
            judged_va=judged,
            elements=elements,
        )
        assert first == again

    def test_bindings_leave_out_a_family_the_verdict_latched(self, tmp_path):
        _, document = _bindings(
            tmp_path,
            verdicts={(SYSTEM, "QF"): VAFamily(verdict="latch", reason="no lattice element")},
        )
        assert document.bindings == ()


class TestBindingsReadbackCollapse:
    def test_bindings_serve_the_written_value_when_the_inverse_returns_it(self, tmp_path):
        _, document = _bindings(tmp_path)
        first = document.bindings[0]
        assert first.readback == "identity"
        assert first.readback_address == "SR:QF:1:RB"
        assert first.monitor_inverse is None

    def test_bindings_serve_the_inverse_when_its_slope_does_not_return_it(self, tmp_path):
        _, document = _bindings(
            tmp_path, judged_va={(SYSTEM, "QF"): _quad_block(inverse=[95.0, 95.0])}
        )
        first = document.bindings[0]
        assert first.readback == "inverse"
        assert first.monitor_inverse.gain == 95.0

    def test_bindings_serve_the_inverse_when_its_offset_does_not_return_it(self, tmp_path):
        block = _quad_block()
        block["Monitor"]["monitor_inverse"] = _linear([100.0, 100.0], [0.5, 0.5])
        _, document = _bindings(tmp_path, judged_va={(SYSTEM, "QF"): block})
        assert document.bindings[0].readback == "inverse"

    def test_bindings_share_one_address_when_setpoint_and_monitor_name_it(self, tmp_path):
        body = _quad_body()
        body["Monitor"]["ChannelNames"] = list(body["Setpoint"]["ChannelNames"])
        _, document = _bindings(tmp_path, views=[_view("QF", body)])
        first = document.bindings[0]
        assert first.readback == "same_as_setpoint"
        assert first.readback_address is None
        assert first.monitor_inverse is None

    def test_bindings_share_one_address_when_a_family_states_no_monitor(self, tmp_path):
        body = _quad_body()
        body.pop("Monitor")
        _, document = _bindings(
            tmp_path,
            views=[_view("QF", body)],
            judged_va={(SYSTEM, "QF"): _quad_block(monitor=False)},
        )
        assert document.bindings[0].readback == "same_as_setpoint"

    def test_bindings_serve_a_reading_through_its_inverse(self, tmp_path):
        _, document = _bindings(
            tmp_path,
            views=[_view("BPMx", _bpm_body())],
            verdicts={(SYSTEM, "BPMx"): _monitor_verdict()},
            judged_va={(SYSTEM, "BPMx"): _bpm_block()},
            elements=_bpm_elements(),
        )
        first = document.bindings[0]
        assert (first.kind, first.readback) == ("monitor", "inverse")
        assert first.setpoint_address == "SR:BPM:1:X"
        assert first.readback_address is None
        assert first.monitor_inverse.gain == 1000.0
        assert first.energy_scaling == "none"

    def test_bindings_refuse_a_reading_with_no_exported_inverse(self, tmp_path):
        with pytest.raises(ValueError, match="only way back to hardware"):
            _bindings(
                tmp_path,
                views=[_view("BPMx", _bpm_body())],
                verdicts={(SYSTEM, "BPMx"): _monitor_verdict()},
                judged_va={(SYSTEM, "BPMx"): _bpm_block(inverse=False)},
                elements=_bpm_elements(),
            )

    def test_bindings_refuse_a_second_address_with_no_exported_inverse(self, tmp_path):
        block = _quad_block()
        block["Monitor"].pop("monitor_inverse")
        with pytest.raises(ValueError, match="never inverted to stand in"):
            _bindings(tmp_path, judged_va={(SYSTEM, "QF"): block})


class TestBindingsSliceWeights:
    def _kick(self, tmp_path, *names: str, kind: str = "kick"):
        return _bindings(
            tmp_path,
            views=[_view("HC", _kick_body())],
            verdicts={
                (SYSTEM, "HC"): VAFamily(verdict="couple", kind=kind, element_field="KickAngle[0]")
            },
            judged_va={(SYSTEM, "HC"): _kick_block()},
            elements={"HC": (_element("HC", kind, (1, 1), "KickAngle", 0, *names),)},
        )

    def test_bindings_divide_a_split_kick_over_its_pieces(self, tmp_path):
        _, document = self._kick(tmp_path, "HC_1_1_1", "HC_1_1_2")
        first = document.bindings[0]
        assert [piece.weight for piece in first.slices] == [0.5, 0.5]
        assert [piece.element for piece in first.slices] == ["HC_1_1_1", "HC_1_1_2"]
        assert first.element == "HC_1_1_1"

    def test_bindings_write_one_whole_piece_when_a_device_is_not_split(self, tmp_path):
        _, document = self._kick(tmp_path, "HC_1_1")
        assert [piece.weight for piece in document.bindings[0].slices] == [1.0]

    def test_bindings_keep_the_rigidity_word_a_kick_carries(self, tmp_path):
        _, document = self._kick(tmp_path, "HC_1_1")
        assert document.bindings[0].energy_scaling == "brho"


class TestBindingsEnergyKnob:
    def _knob(self, tmp_path, **extra):
        return _bindings(
            tmp_path,
            views=[_view("BEND", _energy_body())],
            verdicts={(SYSTEM, "BEND"): _energy_verdict()},
            judged_va={(SYSTEM, "BEND"): _energy_block(**extra)},
            elements=_energy_elements(),
        )

    def test_bindings_bind_the_energy_knob_to_one_address_and_no_element(self, tmp_path):
        _, document = self._knob(tmp_path)
        assert len(document.bindings) == 1
        knob = document.bindings[0]
        assert knob.kind == "energy"
        assert knob.setpoint_address == "SR:BEND:2:SP"
        assert (knob.element, knob.attribute, knob.index, knob.owner) == (None, None, None, None)
        assert knob.slices == ()
        assert knob.calibration is None

    def test_bindings_leave_the_energy_knob_unscaled(self, tmp_path):
        _, document = self._knob(tmp_path)
        assert document.bindings[0].energy_scaling == "none"

    def test_bindings_read_the_energy_knob_back_on_its_own_monitor(self, tmp_path):
        _, document = self._knob(tmp_path)
        knob = document.bindings[0]
        assert knob.readback == "identity"
        assert knob.readback_address == "SR:BEND:2:RB"

    def test_bindings_carry_the_energy_ramp_over_its_sampled_span(self, tmp_path):
        _, document = self._knob(tmp_path)
        table = document.bindings[0].energy_table
        assert table.grid == (0.0, 210.0, 420.0)
        assert table.values == (0.0, 1.0, 2.0)

    def test_bindings_refuse_an_energy_knob_with_no_ramp(self, tmp_path):
        with pytest.raises(ValueError, match="energy_table"):
            self._knob(tmp_path, table=False)


class TestBindingsReadJudgedRows:
    def test_bindings_refuse_a_calibration_stating_other_devices(self, tmp_path):
        block = _quad_block()
        block["Setpoint"]["calibration"] = _linear([0.01, 0.01, 0.01], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="no longer line up"):
            _bindings(tmp_path, judged_va={(SYSTEM, "QF"): block})

    def test_bindings_refuse_a_nominal_stating_other_devices(self, tmp_path):
        block = _quad_block()
        block["nominals"]["Setpoint"]["values"] = [1.5, 2.5, 3.5]
        with pytest.raises(ValueError, match="no longer line up"):
            _bindings(tmp_path, judged_va={(SYSTEM, "QF"): block})

    def test_bindings_refuse_a_block_judged_for_other_devices(self, tmp_path):
        block = _quad_block()
        block["device_list"] = [[1, 1], [1, 2], [1, 3]]
        with pytest.raises(ValueError, match="device"):
            _bindings(tmp_path, judged_va={(SYSTEM, "QF"): block})

    def test_bindings_skip_a_device_whose_slot_names_no_channel(self, tmp_path):
        body = _quad_body()
        body["Setpoint"]["ChannelNames"] = ["SR:QF:1:SP", ""]
        _, document = _bindings(tmp_path, views=[_view("QF", body)])
        assert [binding.setpoint_address for binding in document.bindings] == ["SR:QF:1:SP"]

    def test_bindings_skip_a_device_that_drives_no_element(self, tmp_path):
        elements = {"QF": (_element("QF", "strength", (1, 2), "PolynomB", 1, "QF_1_2"),)}
        _, document = _bindings(tmp_path, elements=elements)
        assert [binding.element for binding in document.bindings] == ["QF_1_2"]
        assert document.bindings[0].nominal == 2.5

    def test_bindings_refuse_a_coupled_family_with_no_nominal(self, tmp_path):
        block = _quad_block()
        block["nominals"]["Setpoint"]["units"] = "Physics"
        with pytest.raises(ValueError, match="no hardware nominal"):
            _bindings(tmp_path, judged_va={(SYSTEM, "QF"): block})


class TestBindingsOnTheCommittedExport:
    """The lane end to end over the only committed 2.0 export.

    The synthetic ring states each case the rules turn on: a matching inverse
    (HC) beside one that differs (QF), a setpoint-only family (SQ) whose
    element another family owns, a split kick with a missing middle piece, the
    energy knob and the cavity.
    """

    @pytest.fixture
    def exported(self, tmp_path):
        return _bindings_from_export(tmp_path, SYNTHETIC, "quokka.sr", SYSTEM)

    def test_bindings_of_the_export_bind_every_coupled_family(self, exported):
        document, _ = exported
        assert {binding.family for binding in document.bindings} == {
            "QF",
            "QD",
            "SF",
            "SQ",
            "HC",
            "VC",
            "BPMx",
            "BPMy",
            "BEND",
            "RF",
        }

    def test_bindings_of_the_export_collapse_only_a_matching_inverse(self, exported):
        document, _ = exported
        bound = {binding.setpoint_address: binding for binding in document.bindings}
        assert bound["QK:HC:1:CUR:SP"].readback == "identity"
        assert bound["QK:QF:1:CUR:SP"].readback == "inverse"
        assert bound["QK:SQ:1:CUR:SP"].readback == "same_as_setpoint"

    def test_bindings_of_the_export_divide_a_split_kick(self, exported):
        document, _ = exported
        bound = {binding.setpoint_address: binding for binding in document.bindings}
        whole = bound["QK:HC:1:CUR:SP"].slices
        assert [(piece.element, piece.weight) for piece in whole] == [
            ("HC_1_1_1", 0.5),
            ("HC_1_1_2", 0.5),
        ]
        missing = bound["QK:HC:4:CUR:SP"].slices
        assert [(piece.element, piece.weight) for piece in missing] == [("HC_4_1_1", 1.0)]

    def test_bindings_of_the_export_name_the_owner_of_a_shared_element(self, exported):
        document, _ = exported
        bound = {binding.setpoint_address: binding for binding in document.bindings}
        skew = bound["QK:SQ:1:CUR:SP"]
        assert (skew.element, skew.owner) == ("SF_1_1", "SF")
        assert (skew.attribute, skew.index) == ("PolynomA", 1)

    def test_bindings_of_the_export_bind_one_energy_knob(self, exported):
        document, _ = exported
        knobs = [binding for binding in document.bindings if binding.kind == "energy"]
        assert len(knobs) == 1
        assert knobs[0].setpoint_address == "QK:BEND:1:CUR:SP"
        assert knobs[0].energy_scaling == "none"

    def test_bindings_of_the_export_cut_a_table_to_its_sampled_span(self, exported):
        document, _ = exported
        knob = next(binding for binding in document.bindings if binding.kind == "energy")
        assert knob.energy_table.grid[-1] == 487.5
        assert len(knob.energy_table.grid) == 27

    def test_bindings_of_the_export_bind_the_cavity_by_class(self, exported):
        document, _ = exported
        cavity = next(binding for binding in document.bindings if binding.kind == "rf")
        assert (cavity.attribute, cavity.index) == ("Frequency", None)
        assert [piece.weight for piece in cavity.slices] == [1.0]

    def test_bindings_of_the_export_keep_the_rigidity_word_where_it_applies(self, exported):
        document, _ = exported
        scalings = {
            binding.family: binding.energy_scaling
            for binding in document.bindings
            if binding.setpoint_address.endswith(":1:CUR:SP")
            or binding.setpoint_address.endswith(":1:CUR:RB")
        }
        assert scalings["QF"] == "brho"
        assert scalings["BPMx"] == "none"

    def test_bindings_of_the_export_name_only_exported_channels(self, exported):
        document, _ = exported
        exported_names = {
            name
            for body in _synthetic("ao").values()
            if isinstance(body, dict)
            for field in body.values()
            if isinstance(field, dict)
            for name in _listed(field.get("ChannelNames"))
        }
        bound = {binding.setpoint_address for binding in document.bindings}
        bound |= {
            binding.readback_address
            for binding in document.bindings
            if binding.readback_address is not None
        }
        assert bound <= exported_names


@pytest.mark.parametrize(
    ("tree", "family", "readback"),
    [("nsls2", "SQ", "identity"), ("spear3", "RF", "same_as_setpoint")],
)
def test_bindings_on_a_real_export_collapse_as_the_facility_sampled_it(
    tmp_path, tree, family, readback
):
    """The two committed real trees, once either carries a 2.0 export.

    They hold 1.0 files today, so the lane discovers there is nothing to run
    against and skips itself; the day a ``va.json`` lands beside them it runs
    without being edited.
    """
    directory = SYNTHETIC.parent / tree
    exports = sorted(directory.glob("*.va.json"))
    if not exports:
        pytest.skip(
            f"{tree} holds no 2.0 export; re-export it with mml_export 2.0 to run this lane"
        )
    stem = exports[0].name[: -len(".va.json")]
    document, _ = _bindings_from_export(tmp_path, directory, stem, stem.split(".")[1])
    assert {binding.readback for binding in document.bindings if binding.family == family} == {
        readback
    }


# --- channel_limits.json -----------------------------------------------------
#
# The write-safety database is the one file this lane shares with a facility,
# so every case here reads the emitted text back through
# ``LimitsValidator._load_limits_database`` -- the loader that fails a whole
# file on one key it does not know -- rather than through assertions on a dict
# the emitter happened to build.


def _banded_quad_body(band: object = (0, 200)) -> dict:
    """The quadrupole, with the operating band the export states for it."""
    body = _quad_body()
    if band is not None:
        body["Setpoint"]["Range"] = list(band)
    return body


def _limits(
    tmp_path: Path,
    *,
    existing: dict | None = None,
    views=None,
    judged_va=None,
    verdicts=None,
    elements=None,
    channel_addresses=(),
    body: dict | None = None,
):
    """Emit channel_limits.json over the quadrupole case, or whatever is passed."""
    views = [_view("QF", body or _banded_quad_body())] if views is None else views
    _, document = _bindings(
        tmp_path, views=views, judged_va=judged_va, verdicts=verdicts, elements=elements
    )
    text, bands = emit_channel_limits(
        existing,
        document.bindings,
        channel_addresses,
        _ctx(tmp_path),
        views=views,
        system=SYSTEM,
    )
    return json.loads(text), bands, text


def _validated(tmp_path: Path, text: str):
    """Read one emitted document back through the write path's own loader."""
    path = tmp_path / "channel_limits.json"
    path.write_text(text)
    return LimitsValidator._load_limits_database(str(path))


def _facility_file(**entries: dict) -> dict:
    """A limits database a facility authored before the virtual accelerator existed."""
    document: dict = {
        "_comment": "the facility's own write-safety database",
        "defaults": {"writable": False},
    }
    document.update(entries)
    return document


def _limits_from_export(
    tmp_path: Path, directory: Path, stem: str, system: str, *, existing: dict | None = None
):
    """Run the whole VA lane over one export on disk and load what it banded."""
    from osprey.services.mml.loaders.mat import load_lattice
    from osprey.services.mml.normalize import normalize_family
    from osprey.services.mml.va.elements import address_elements
    from osprey.services.mml.va.verdicts import propose

    ao = json.loads((directory / f"{stem}.ao.json").read_text())
    va = json.loads((directory / f"{stem}.va.json").read_text())
    ring = load_lattice(directory / f"{stem}.lattice.mat")
    views = {
        raw: FamilyView(system, raw, normalize_family(body))
        for raw, body in ao.items()
        if not raw.startswith("_") and isinstance(body, dict)
    }
    verdicts = propose(va, ring, views)
    ctx = _bindings_ctx(tmp_path)
    bindings_path = tmp_path / "va_bindings.json"
    bindings_path.write_text(
        emit_bindings(
            {(system, name): verdict for name, verdict in verdicts.items()},
            list(views.values()),
            dict(address_elements(va, ring, verdicts).bindings),
            {(system, name): block for name, block in va["families"].items()},
            ctx,
            system=system,
            energy_gev=va["lattice"]["energy_gev"],
        )
    )
    addresses = set()
    for view in views.values():
        for field in view.fields.values():
            for key in field.keys:
                for slot in field.slots(key):
                    if isinstance(slot, str) and slot.strip():
                        addresses.add(slot.strip())
    text, bands = emit_channel_limits(
        existing,
        load_bindings(bindings_path).bindings,
        sorted(addresses),
        ctx,
        views=list(views.values()),
        system=system,
    )
    return _validated(tmp_path, text)[0], bands, text


class TestChannelLimitsOnATreeItCreates:
    def test_channel_limits_band_a_coupled_setpoint_from_its_range(self, tmp_path):
        document, bands, _ = _limits(tmp_path)
        entry = document["SR:QF:1:SP"]
        assert (entry["min_value"], entry["max_value"]) == (0, 200)
        assert entry["writable"] is True
        assert entry[PROVENANCE_KEY]
        assert [band.address for band in bands] == ["SR:QF:1:SP", "SR:QF:2:SP"]

    def test_channel_limits_leave_every_other_address_unwritable(self, tmp_path):
        document, _, _ = _limits(tmp_path, channel_addresses=["SR:DCCT:CURRENT"])
        assert document["SR:QF:1:RB"]["writable"] is False
        assert document["SR:DCCT:CURRENT"]["writable"] is False
        assert "min_value" not in document["SR:QF:1:RB"]

    def test_channel_limits_state_one_entry_per_address_the_channel_database_carries(
        self, tmp_path
    ):
        document, _, _ = _limits(tmp_path, channel_addresses=["SR:DCCT:CURRENT", "SR:QF:1:SP", ""])
        assert sorted(key for key in document if not key.startswith("_")) == [
            "SR:DCCT:CURRENT",
            "SR:QF:1:RB",
            "SR:QF:1:SP",
            "SR:QF:2:RB",
            "SR:QF:2:SP",
        ]

    def test_channel_limits_widen_a_band_the_nominal_sits_outside(self, tmp_path):
        document, bands, _ = _limits(tmp_path, body=_banded_quad_body((0, 1)))
        assert document["SR:QF:1:SP"]["max_value"] == 1.5
        assert document["SR:QF:2:SP"]["max_value"] == 2.5
        assert [band.widened for band in bands] == [True, True]
        assert [band.nominal for band in bands] == [1.5, 2.5]

    def test_channel_limits_skip_an_infinite_range_bound(self, tmp_path):
        document, _, _ = _limits(tmp_path, body=_banded_quad_body(("-Inf", 5)))
        entry = document["SR:QF:1:SP"]
        assert "min_value" not in entry
        assert entry["max_value"] == 5

    def test_channel_limits_keep_a_per_device_range_row_on_its_own_device(self, tmp_path):
        document, _, _ = _limits(tmp_path, body=_banded_quad_body([[0, 100], [0, 200]]))
        assert document["SR:QF:1:SP"]["max_value"] == 100
        assert document["SR:QF:2:SP"]["max_value"] == 200

    def test_channel_limits_band_nothing_when_the_family_states_no_range(self, tmp_path):
        document, bands, _ = _limits(tmp_path, body=_banded_quad_body(None))
        entry = document["SR:QF:1:SP"]
        assert entry["writable"] is True
        assert "min_value" not in entry and "max_value" not in entry
        assert [band.min_value for band in bands] == [None, None]

    def test_channel_limits_leave_a_monitor_family_read_only(self, tmp_path):
        views = [_view("BPMx", _bpm_body())]
        document, bands, _ = _limits(
            tmp_path,
            views=views,
            verdicts={(SYSTEM, "BPMx"): _monitor_verdict()},
            judged_va={(SYSTEM, "BPMx"): _bpm_block()},
            elements=_bpm_elements(),
        )
        assert document["SR:BPM:1:X"]["writable"] is False
        assert bands == ()

    def test_channel_limits_load_through_the_write_safety_validator(self, tmp_path):
        _, _, text = _limits(tmp_path, channel_addresses=["SR:DCCT:CURRENT"])
        limits, _raw = _validated(tmp_path, text)
        band = limits["SR:QF:1:SP"]
        assert (band.min_value, band.max_value, band.writable) == (0, 200, True)
        assert limits["SR:DCCT:CURRENT"].writable is False

    def test_channel_limits_are_byte_identical_on_a_second_emit(self, tmp_path):
        _, _, first = _limits(tmp_path, channel_addresses=["SR:DCCT:CURRENT"])
        _, _, second = _limits(tmp_path, channel_addresses=["SR:DCCT:CURRENT"])
        assert first == second


class TestChannelLimitsMergeIntoAFacilityFile:
    def test_channel_limits_merge_leaves_a_foreign_entry_byte_identical(self, tmp_path):
        foreign = {"min_value": -5, "max_value": 5, "confirm": True, "_note": "signed off"}
        document, _, _ = _limits(
            tmp_path,
            existing=_facility_file(**{"FAC:PS:1:SP": dict(foreign)}),
            channel_addresses=["SR:DCCT:CURRENT"],
        )
        assert json.dumps(document["FAC:PS:1:SP"]) == json.dumps(foreign)

    def test_channel_limits_merge_keeps_the_defaults_block_and_the_file_prose(self, tmp_path):
        existing = _facility_file()
        document, _, _ = _limits(tmp_path, existing=existing)
        assert document["defaults"] == {"writable": False}
        assert document["_comment"] == existing["_comment"]
        assert list(document)[:2] == ["_comment", "defaults"]

    def test_channel_limits_merge_states_no_address_the_lane_does_not_own(self, tmp_path):
        document, _, _ = _limits(
            tmp_path, existing=_facility_file(), channel_addresses=["SR:DCCT:CURRENT"]
        )
        assert "SR:DCCT:CURRENT" not in document
        assert "SR:QF:1:SP" in document
        assert "SR:QF:1:RB" in document

    def test_channel_limits_merge_rewrites_only_the_band_of_a_stamped_entry(self, tmp_path):
        stamped = {
            PROVENANCE_KEY: "an older run",
            "writable": True,
            "confirm": True,
            "max_step": 0.5,
            "min_value": -1,
            "max_value": 1,
        }
        document, bands, _ = _limits(
            tmp_path, existing=_facility_file(**{"SR:QF:1:SP": dict(stamped)})
        )
        entry = document["SR:QF:1:SP"]
        assert (entry["min_value"], entry["max_value"]) == (0, 200)
        assert (entry["confirm"], entry["max_step"]) == (True, 0.5)
        assert entry[PROVENANCE_KEY] != "an older run"
        assert [band.refused for band in bands] == [None, None]

    def test_channel_limits_merge_refuses_a_hand_edited_band(self, tmp_path):
        hand = {"min_value": -1, "max_value": 1, "writable": True}
        document, bands, _ = _limits(
            tmp_path, existing=_facility_file(**{"SR:QF:1:SP": dict(hand)})
        )
        assert json.dumps(document["SR:QF:1:SP"]) == json.dumps(hand)
        refused = [band for band in bands if band.refused]
        assert [band.address for band in refused] == ["SR:QF:1:SP"]
        assert PROVENANCE_KEY in refused[0].refused

    def test_channel_limits_merge_accepts_a_hand_written_band_that_agrees(self, tmp_path):
        agrees = {"min_value": 0, "max_value": 200}
        document, bands, _ = _limits(
            tmp_path, existing=_facility_file(**{"SR:QF:1:SP": dict(agrees)})
        )
        assert json.dumps(document["SR:QF:1:SP"]) == json.dumps(agrees)
        assert [band.refused for band in bands] == [None, None]

    def test_channel_limits_merge_loads_through_the_write_safety_validator(self, tmp_path):
        _, _, text = _limits(
            tmp_path, existing=_facility_file(**{"FAC:PS:1:SP": {"confirm": True}})
        )
        limits, raw = _validated(tmp_path, text)
        assert raw["defaults"] == {"writable": False}
        assert limits["FAC:PS:1:SP"].writable is False
        assert limits["SR:QF:1:SP"].writable is True

    def test_channel_limits_merge_is_byte_identical_on_a_second_emit(self, tmp_path):
        existing = _facility_file(**{"FAC:PS:1:SP": {"confirm": True}})
        _, _, first = _limits(tmp_path, existing=existing)
        _, _, second = _limits(tmp_path, existing=json.loads(first))
        assert first == second


class TestChannelLimitsOnTheCommittedExport:
    """The lane end to end over the only committed 2.0 export."""

    def test_channel_limits_of_the_export_load_through_the_write_safety_validator(self, tmp_path):
        limits, _bands, _text = _limits_from_export(tmp_path, SYNTHETIC, "quokka.sr", SYSTEM)
        quad = limits["QK:QF:1:CUR:SP"]
        assert (quad.min_value, quad.max_value, quad.writable) == (0, 200, True)
        assert limits["QK:BPMx:1:CUR:RB"].writable is False
        assert limits["QK:DCCT:1:CUR:RB"].writable is False

    def test_channel_limits_of_the_export_widen_a_band_around_its_nominal(self, tmp_path):
        _limits_db, bands, _text = _limits_from_export(tmp_path, SYNTHETIC, "quokka.sr", SYSTEM)
        widened = {band.address: band for band in bands if band.widened}
        assert widened["QK:HC:1:CUR:SP"].max_value == 1.5
        assert widened["QK:HC:1:CUR:SP"].nominal == 1.5


@pytest.mark.parametrize("tree", ["nsls2", "spear3"])
def test_channel_limits_on_a_real_export_load_through_the_write_safety_validator(tmp_path, tree):
    """The two committed real trees, once either carries a 2.0 export.

    They hold 1.0 files today, so the lane discovers there is nothing to run
    against and skips itself; the day a ``va.json`` lands beside them it runs
    without being edited.
    """
    directory = SYNTHETIC.parent / tree
    exports = sorted(directory.glob("*.va.json"))
    if not exports:
        pytest.skip(
            f"{tree} holds no 2.0 export; re-export it with mml_export 2.0 to run this lane"
        )
    stem = exports[0].name[: -len(".va.json")]
    limits, bands, _text = _limits_from_export(tmp_path, directory, stem, stem.split(".")[1])
    assert bands
    assert all(limits[band.address].writable for band in bands if band.refused is None)

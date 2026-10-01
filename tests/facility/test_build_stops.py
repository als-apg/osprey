"""Every stop of the facility build: one synthetic tree, one line, exit 1.

``STOP_SENTENCES`` lists each sentence of the error table once with its kind.
Each case is a minimal tree that breaks exactly that rule. It replaces the
``data/facility`` tree of a control-assistant repo, beside any profile edit the
case makes, and runs through ``osprey build --skip-deps`` and through
``osprey facility validate``; both must print the case's line byte for byte on
stderr and exit 1. A persona render is checked by the build alone, so a case
that breaks one runs through ``osprey build --skip-deps`` only.
"""

from __future__ import annotations

import copy
import itertools
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.facility.build import build_facility
from osprey.facility.errors import KINDS
from tests._builds import init_project, run_build
from tests.facility._synthetic_trees import (
    BPM,
    QUAD,
    READING,
    SETTING,
    deck_tree,
    model,
    plain_tree,
    sr_deck,
    write_tree,
)

PROJECT = "demo"

Tree = dict[str, Any]
Edit = Callable[[Tree], None]


# --- tree edits ------------------------------------------------------------------------


def _plain(*edits: Edit) -> Callable[[], Tree]:
    return _from(plain_tree, edits)


def _deck(*edits: Edit) -> Callable[[], Tree]:
    return _from(deck_tree, edits)


def _from(base: Callable[[], Tree], edits: tuple[Edit, ...]) -> Callable[[], Tree]:
    def make() -> Tree:
        tree = base()
        for edit in edits:
            edit(tree)
        return tree

    return make


def put(rel: str, data: Any) -> Edit:
    """Write a whole file."""

    def edit(tree: Tree) -> None:
        tree[rel] = copy.deepcopy(data)

    return edit


def append(rel: str, *rows: Any) -> Edit:
    """Append rows to a list file, creating it when absent."""

    def edit(tree: Tree) -> None:
        tree.setdefault(rel, []).extend(copy.deepcopy(list(rows)))

    return edit


def update(rel: str, index: int, **slots: Any) -> Edit:
    """Set slots on one row of a list file."""

    def edit(tree: Tree) -> None:
        tree[rel][index].update(copy.deepcopy(slots))

    return edit


def drop_slot(rel: str, index: int, slot: str) -> Edit:
    """Remove one slot from one row of a list file."""

    def edit(tree: Tree) -> None:
        del tree[rel][index][slot]

    return edit


def on_model(name: str, change: Callable[[dict[str, Any]], None]) -> Edit:
    """Change one model record of ``models.yaml``."""

    def edit(tree: Tree) -> None:
        change(model(tree, name))

    return edit


def wiring(name: str, index: int, **slots: Any) -> Edit:
    """Set slots on one wiring record of a model."""

    def change(record: dict[str, Any]) -> None:
        record["wiring"][index].update(copy.deepcopy(slots))

    return on_model(name, change)


def wire(name: str, *records: dict[str, Any]) -> Edit:
    """Append wiring records to a model."""

    def change(record: dict[str, Any]) -> None:
        record["wiring"].extend(copy.deepcopy(list(records)))

    return on_model(name, change)


def fixes(*entries: dict[str, Any]) -> Edit:
    return put("fixes.yaml", {"schema": "osprey.facility.fixes/1", "fixes": list(entries)})


def limits(*records: dict[str, Any]) -> Edit:
    return put("limits.yaml", {"records": list(records)})


def seeds(**by_address: dict[str, Any]) -> Edit:
    return put("seeds.yaml", by_address)


def scenario(name: str, body: dict[str, Any]) -> Edit:
    return put(f"scenarios/{name}.yaml", body)


def _set_slice(where: str, element: str) -> Edit:
    """Point one wired element of model SR at ``element``."""
    if where == "element":
        return wiring("SR", 1, element=element)
    if where == "first slice":
        return wiring("SR", 0, slices=[{"element": element}, {"element": "QFB"}])
    if where == "later slice":
        return wiring(
            "SR", 0, slices=[{"element": "QFA"}, {"element": "QFB"}, {"element": element}]
        )
    return wiring("SR", 2, element=element)


#: The files of a second layer, ``mml``.
MML_CHANNELS = "imported/mml/channels.yaml"
MML_MODELS = "imported/mml/models.yaml"
#: An unwired setpoint beside the plain tree's pair.
EXTRA_SP = {"id": "Q2:SP", "role": "setpoint", "on": {"device": "SR/Q1"}}
#: A model without a deck, wiring the plain tree's setpoint.
NO_DECK = {"name": "optics", "engine": "pyat", "wiring": [{"address": "Q1:SP", "element": "Q1"}]}


#: Where the mml layer keeps model SR's deck once SR is imported.
MML_DECK = "imported/mml/decks/SR.json"


def imported_sr(tree: Tree) -> None:
    """Move model SR and its deck into the mml layer."""
    record = model(tree, "SR")
    tree["models.yaml"].remove(record)
    record["deck"] = MML_DECK
    tree.setdefault(MML_MODELS, []).append(record)
    tree[MML_DECK] = tree.pop("decks/sr.json")


def imported_wiring(index: int, **slots: Any) -> Edit:
    """Set slots on one wiring record of the imported model SR."""

    def edit(tree: Tree) -> None:
        model(tree, "SR", MML_MODELS)["wiring"][index].update(copy.deepcopy(slots))

    return edit


def _cavity(at: Any) -> Any:
    cavity = at.RFCavity("RFC", 0.0, 1e6, 5e8, 300, 3e9)
    cavity.PassMethod = "IdentityPass"
    return cavity


# --- the table -------------------------------------------------------------------------

#: Each sentence of the error table once: (case id, kind, sentence).
STOP_SENTENCES: tuple[tuple[str, str, str], ...] = (
    ("source_invalid__pydantic_failure", "source-invalid", "a pydantic failure"),
    ("source_invalid__unknown_key", "source-invalid", "an unknown key such as `kind:`"),
    ("source_invalid__on_both_kinds", "source-invalid", "an `on` naming both kinds"),
    (
        "source_invalid__list_row_device_and_place",
        "source-invalid",
        "a list row with both `device` and `place`",
    ),
    ("source_invalid__identity_code", "source-invalid", "identity `code` not PN_LOCAL"),
    ("source_invalid__model_name", "source-invalid", "a model name not PN_LOCAL"),
    (
        "source_invalid__model_names_casefold",
        "source-invalid",
        "two model names equal case-insensitively",
    ),
    ("source_invalid__computed_slot", "source-invalid", "a layer writes a computed slot"),
    (
        "layer_conflict__map_slot",
        "layer-conflict",
        "unequal layer values, a map slot compared whole",
    ),
    ("layer_conflict__on", "layer-conflict", "differing `on` across layers"),
    ("layer_duplicate__id_twice", "layer-duplicate", "an id twice in one layer"),
    ("fix_missing__no_target", "fix-missing", "a fix naming no record"),
    ("fix_stale__was", "fix-stale", "a `set` whose `was` no longer holds"),
    ("fix_duplicate__two_fixes", "fix-duplicate", "two fixes on one record"),
    ("fix_computed__set", "fix-computed", "a `set` of a computed slot"),
    ("fix_authored__authored_only", "fix-authored", "a fix of an authored-only record"),
    ("fix_referenced__device", "fix-referenced", "a drop of a device still referenced"),
    ("fix_referenced__model", "fix-referenced", "a drop of a model still referenced"),
    ("fix_referenced__group", "fix-referenced", "a drop of a group still referenced"),
    ("reference_missing__on_device", "reference-missing", "channel `on.device`"),
    ("reference_missing__on_place", "reference-missing", "channel `on.place`"),
    ("reference_missing__pair", "reference-missing", "channel `pair`"),
    ("reference_missing__endpoint_of", "reference-missing", "channel `endpoint_of`"),
    ("reference_missing__linear_input", "reference-missing", "`linear` input"),
    ("reference_missing__group_member", "reference-missing", "group `members`"),
    ("reference_missing__wiring_address", "reference-missing", "wiring address"),
    ("reference_missing__slice_device", "reference-missing", "`slices[].device`"),
    ("reference_missing__span_model", "reference-missing", "span model"),
    ("reference_missing__device_place", "reference-missing", "device place"),
    ("reference_missing__place_parent", "reference-missing", "place parent path"),
    ("reference_missing__seed_address", "reference-missing", "seeds.yaml address"),
    ("reference_missing__limit_address", "reference-missing", "limits.yaml record address"),
    ("reference_missing__override", "reference-missing", "scenario `overrides` address"),
    ("reference_missing__archiver", "reference-missing", "scenario `archiver` channel"),
    ("reference_missing__fault_model", "reference-missing", "scenario `faults` model key"),
    ("reference_missing__fault_channel", "reference-missing", "scenario `faults` channel"),
    (
        "reference_missing__measurement_model",
        "reference-missing",
        "a measurement file naming a missing model",
    ),
    (
        "reference_missing__measurement_group",
        "reference-missing",
        "a measurement file naming a missing group",
    ),
    (
        "reference_missing__measurement_instrument",
        "reference-missing",
        "a measurement file naming a missing instrument",
    ),
    (
        "class_unknown__device_class",
        "class-unknown",
        "device class in neither vocabulary nor classes.yaml",
    ),
    ("class_unknown__parent", "class-unknown", "a `parent` that is neither"),
    ("pair_invalid__pair_on_readback", "pair-invalid", "a `pair` on a readback channel"),
    ("pair_invalid__pair_on_none", "pair-invalid", "a `pair` on a `none` channel"),
    ("pair_invalid__type_mismatch", "pair-invalid", "pair type mismatch"),
    ("pair_invalid__element_and_slices", "pair-invalid", "`element` and `slices` both present"),
    ("pair_invalid__weight_zero", "pair-invalid", "a slice weight zero"),
    ("pair_invalid__weight_non_finite", "pair-invalid", "a slice weight non-finite"),
    (
        "pair_invalid__endpoint_unnamed",
        "pair-invalid",
        "a wired `endpoint_of` device named by no slice",
    ),
    ("value_invalid__nominal_float", "value-invalid", "coercion refusal: float nominal"),
    ("value_invalid__nominal_bool", "value-invalid", "coercion refusal: bool nominal label"),
    ("value_invalid__nominal_enum", "value-invalid", "coercion refusal: enum nominal index"),
    ("value_invalid__nominal_string", "value-invalid", "coercion refusal: string nominal"),
    ("value_invalid__nominal_waveform", "value-invalid", "coercion refusal: waveform nominal"),
    ("value_invalid__override", "value-invalid", "coercion refusal of a scenario override"),
    ("value_invalid__fault", "value-invalid", "coercion refusal of a fault value"),
    ("value_invalid__options_presence", "value-invalid", "`options` presence wrong"),
    ("value_invalid__shape_presence", "value-invalid", "`shape` presence wrong"),
    ("value_invalid__motion_non_float", "value-invalid", "motion on non-float"),
    ("value_invalid__enum_bounds", "value-invalid", "enum bounds"),
    ("value_invalid__linear_wired", "value-invalid", "`linear` inputs wired"),
    ("value_invalid__linear_cyclic", "value-invalid", "`linear` inputs cyclic"),
    ("value_invalid__linear_nominal", "value-invalid", "`linear` with `nominal`"),
    ("value_invalid__linear_non_float", "value-invalid", "`linear` on a non-float"),
    ("value_invalid__drift_period", "value-invalid", "`drift.period_s` <= 0"),
    (
        "value_invalid__limit_bounds_non_numeric",
        "value-invalid",
        "limit `min/max/max_step` on a non-numeric channel",
    ),
    ("value_invalid__clamp_order", "value-invalid", "`clamp` lo > hi"),
    ("value_invalid__clamp_non_float", "value-invalid", "`clamp` on a non-float"),
    ("value_invalid__stuck_non_setpoint", "value-invalid", "`stuck` on a non-setpoint"),
    ("seed_invalid__nominal_out_of_band", "seed-invalid", "nominal out of band"),
    ("seed_invalid__nominal_wired", "seed-invalid", "nominal on a wired channel"),
    ("seed_invalid__motion_on_setpoint", "seed-invalid", "noise/drift on a setpoint"),
    ("seed_invalid__paired_seed", "seed-invalid", "paired seed disagrees"),
    ("seed_invalid__int_nominal", "seed-invalid", "non-integral nominal on an int channel"),
    ("seed_invalid__int_override", "seed-invalid", "non-integral override on an int channel"),
    ("limit_invalid__writable", "limit-invalid", "writable on a non-setpoint"),
    ("limit_invalid__int_bound", "limit-invalid", "non-integral bound on an int channel"),
    (
        "place_conflict__imported_place",
        "place-conflict",
        "an imported place contradicting its span",
    ),
    ("span_invalid__overlap", "span-invalid", "overlapping spans"),
    ("span_invalid__marker_absent", "span-invalid", "a marker absent"),
    ("span_invalid__marker_not_unique", "span-invalid", "a marker not unique"),
    ("span_invalid__no_deck", "span-invalid", "no deck"),
    ("wiring_conflict__device_two_models", "wiring-conflict", "a device wired in two models"),
    ("wiring_conflict__address_twice", "wiring-conflict", "an address wired twice"),
    (
        "wiring_conflict__address_twice_on_device",
        "wiring-conflict",
        "an address on a device wired twice",
    ),
    (
        "wiring_conflict__repeated_element",
        "wiring-conflict",
        "a wired `element` repeated in its deck",
    ),
    (
        "wiring_conflict__repeated_first_slice",
        "wiring-conflict",
        "a wired first slice repeated in its deck",
    ),
    (
        "wiring_conflict__repeated_later_slice",
        "wiring-conflict",
        "a wired later slice repeated in its deck",
    ),
    (
        "wiring_conflict__repeated_readback",
        "wiring-conflict",
        "a wired readback element repeated in its deck",
    ),
    ("engine_missing__no_plugin", "engine-missing", "no plug-in"),
    ("engine_invalid__single_pass_twiss", "engine-invalid", "`single_pass` without `twiss_in`"),
    ("engine_invalid__twiss_length", "engine-invalid", "wrong `twiss_in` length"),
    (
        "engine_invalid__frozen_cavity",
        "engine-invalid",
        "periodic deck with an `RFCavity` whose `longt_motion` is False",
    ),
    ("engine_invalid__repeated_monitors", "engine-invalid", "repeated monitor names"),
    (
        "engine_invalid__imported_frozen_cavity",
        "engine-invalid",
        "an imported periodic deck with an `RFCavity` whose `longt_motion` is False",
    ),
    (
        "engine_invalid__imported_repeated_monitors",
        "engine-invalid",
        "repeated monitor names in an imported deck",
    ),
    ("engine_invalid__missing_element", "engine-invalid", "`locate` of an unknown `element`"),
    (
        "engine_invalid__imported_missing_element",
        "engine-invalid",
        "`locate` of an unknown `element` in an imported deck",
    ),
    (
        "engine_invalid__missing_first_slice",
        "engine-invalid",
        "`locate` of an unknown first slice",
    ),
    (
        "engine_invalid__missing_later_slice",
        "engine-invalid",
        "`locate` of an unknown later slice",
    ),
    (
        "engine_invalid__missing_readback",
        "engine-invalid",
        "`locate` of an unknown readback element",
    ),
    (
        "engine_invalid__table_without_inverse",
        "engine-invalid",
        "a calibration table without `inverse` on a setpoint",
    ),
    ("engine_invalid__linear_gain_zero", "engine-invalid", "a linear gain of 0 on a setpoint"),
    ("model_conflict__texture", "model-conflict", "a layer declares a model named texture"),
    (
        "model_conflict__status_address",
        "model-conflict",
        "a facility channel equals a status address",
    ),
    ("profile_invalid__mirrored_facility_file", "profile-invalid", "`project/facility.json`"),
    (
        "profile_invalid__mirrored_simulator_view",
        "profile-invalid",
        "`project/data/simulator/x.json`",
    ),
    (
        "profile_invalid__unknown_served_model",
        "profile-invalid",
        "`simulation.models` names a model the facility file lacks",
    ),
    (
        "profile_invalid__persona_served_models",
        "profile-invalid",
        "a persona on a VA target sets `simulation.models`",
    ),
)

#: Each case: the tree that breaks the rule, and the one line it stops with.
CASES: dict[str, tuple[Callable[[], Tree], str]] = {
    "source_invalid__pydantic_failure": (
        _plain(append("records/channels.yaml", {"id": "T", "value_type": "complex"})),
        (
            "facility: source-invalid: path channels.3.value_type — records/channels.yaml: input "
            "should be 'float', 'int', 'bool', 'enum', 'string' or 'waveform'; fix: correct "
            "`value_type` in records/channels.yaml"
        ),
    ),
    "source_invalid__unknown_key": (
        _plain(update("records/devices.yaml", 0, kind="magnet")),
        (
            "facility: source-invalid: device SR/Q1 — unknown key `kind`; fix: remove `kind` from "
            "records/devices.yaml"
        ),
    ),
    "source_invalid__on_both_kinds": (
        _plain(
            append("records/places.yaml", {"id": "SR"}),
            update("records/channels.yaml", 2, on={"device": "SR/BPM1", "place": "SR"}),
        ),
        (
            "facility: source-invalid: channel BPM1:X — `on` names both a device and a place; "
            "fix: keep one of `device` and `place` in records/channels.yaml"
        ),
    ),
    "source_invalid__list_row_device_and_place": (
        _plain(
            append("records/places.yaml", {"id": "SR"}),
            append(
                "imported/list/channels.yaml",
                {"id": "BPM1:Y", "on": {"device": "SR/BPM1", "place": "SR"}},
            ),
        ),
        (
            "facility: source-invalid: channel BPM1:Y — `on` names both a device and a place; "
            "fix: keep one of `device` and `place` in imported/list/channels.yaml"
        ),
    ),
    "source_invalid__identity_code": (
        _plain(put("identity.yaml", {"code": "9lives"})),
        (
            "facility: source-invalid: path identity.yaml.code — code '9lives' does not match "
            "[A-Za-z_][A-Za-z0-9_]*; fix: rename the code"
        ),
    ),
    "source_invalid__model_name": (
        _plain(put("models.yaml", [{**NO_DECK, "name": "optics-1"}])),
        (
            "facility: source-invalid: model optics-1 — the name does not match "
            "[A-Za-z_][A-Za-z0-9_]*; fix: rename the model"
        ),
    ),
    "source_invalid__model_names_casefold": (
        _plain(
            put("models.yaml", [NO_DECK]),
            put(MML_MODELS, [{"name": "Optics", "engine": "pyat"}]),
        ),
        (
            "facility: source-invalid: model Optics — model names Optics, optics differ only in "
            "case; fix: use one spelling"
        ),
    ),
    "source_invalid__computed_slot": (
        _plain(update("records/devices.yaml", 0, s=1.0)),
        (
            "facility: source-invalid: device SR/Q1 — `s` is computed by the build; fix: remove "
            "`s` from records/devices.yaml"
        ),
    ),
    "layer_conflict__map_slot": (
        _plain(
            put("models.yaml", [{**NO_DECK, "settings": {"pyat": {"solve": "single_pass"}}}]),
            put(
                MML_MODELS,
                [{"name": "optics", "settings": {"pyat": {"twiss_in": {"beta": [1.0, 1.0]}}}}],
            ),
        ),
        (
            "facility: layer-conflict: model optics — `settings` differs: authored={pyat: {solve: "
            "single_pass}}; mml={pyat: {twiss_in: {beta: [1.0, 1.0]}}}; fix: add a `set` fix for "
            "`settings` to fixes.yaml"
        ),
    ),
    "layer_conflict__on": (
        _plain(
            append("records/places.yaml", {"id": "SR"}),
            put(MML_CHANNELS, [{"id": "BPM1:X", "on": {"place": "SR"}}]),
        ),
        (
            "facility: layer-conflict: channel BPM1:X — `on` differs: authored={device: SR/BPM1}; "
            "mml={place: SR}; fix: add a `set` fix for `on` to fixes.yaml"
        ),
    ),
    "layer_duplicate__id_twice": (
        _plain(append("records/devices.yaml", {"id": "SR/Q1", "class": QUAD})),
        (
            "facility: layer-duplicate: device SR/Q1 — layer authored states it twice "
            "(records/devices.yaml, records/devices.yaml); fix: keep one record per id in layer "
            "authored"
        ),
    ),
    "fix_missing__no_target": (
        _plain(fixes({"op": "drop", "kind": "device", "id": "SR/Q9", "why": "gone"})),
        (
            "facility: fix-missing: device SR/Q9 — no device SR/Q9 exists; fix: remove the fix "
            "from fixes.yaml"
        ),
    ),
    "fix_stale__was": (
        _plain(
            put(MML_CHANNELS, [{"id": "BPM1:X", "unit": "mm"}]),
            fixes(
                {
                    "op": "set",
                    "kind": "channel",
                    "id": "BPM1:X",
                    "fields": {"unit": "um"},
                    "was": {"unit": {"mml": "m"}},
                    "why": "The export rounds the unit.",
                }
            ),
        ),
        (
            "facility: fix-stale: channel BPM1:X — `was` does not match the layers; fix: replace "
            "the fix with {fields: {unit: um}, id: 'BPM1:X', kind: channel, op: set, was: {unit: "
            "{mml: mm}}, why: The export rounds the unit.}"
        ),
    ),
    "fix_duplicate__two_fixes": (
        _plain(
            put(MML_CHANNELS, [{"id": "BPM1:X", "unit": "mm"}]),
            fixes(
                {"op": "drop", "kind": "channel", "id": "BPM1:X", "why": "a"},
                {
                    "op": "set",
                    "kind": "channel",
                    "id": "BPM1:X",
                    "fields": {"unit": "um"},
                    "was": {"unit": {"mml": "mm"}},
                    "why": "b",
                },
            ),
        ),
        (
            "facility: fix-duplicate: channel BPM1:X — fixes.yaml has `drop` and `set` for it; "
            "fix: keep one fix per record in fixes.yaml"
        ),
    ),
    "fix_computed__set": (
        _plain(
            put("imported/mml/devices.yaml", [{"id": "SR/Q1", "class": QUAD}]),
            fixes({"op": "set", "kind": "device", "id": "SR/Q1", "fields": {"s": 1.0}, "why": "x"}),
        ),
        (
            "facility: fix-computed: device SR/Q1 — `set` names `s`, which the build computes; "
            "fix: remove `s` from the fix"
        ),
    ),
    "fix_authored__authored_only": (
        _plain(
            fixes(
                {
                    "op": "set",
                    "kind": "device",
                    "id": "SR/BPM1",
                    "fields": {"description": "the first BPM"},
                    "why": "x",
                }
            ),
        ),
        (
            "facility: fix-authored: device SR/BPM1 — `set` targets a record only authored "
            "sources state; fix: edit data/facility/records/devices.yaml"
        ),
    ),
    "fix_referenced__device": (
        _plain(
            put("imported/mml/devices.yaml", [{"id": "SR/BPM1", "class": BPM}]),
            fixes({"op": "drop", "kind": "device", "id": "SR/BPM1", "why": "x"}),
        ),
        (
            "facility: fix-referenced: device SR/BPM1 — `drop` leaves 1 referrer(s): channel "
            "BPM1:X (on); fix: drop or re-point each referrer by a fix"
        ),
    ),
    "fix_referenced__model": (
        _plain(
            put(MML_MODELS, [{"name": "optics", "engine": "pyat", "wiring": []}]),
            put(
                "records/places.yaml",
                [{"id": "SR", "span": {"model": "optics", "from_marker": "M"}}],
            ),
            fixes({"op": "drop", "kind": "model", "id": "optics", "why": "x"}),
        ),
        (
            "facility: fix-referenced: model optics — `drop` leaves 1 referrer(s): place SR "
            "(span); fix: drop or re-point each referrer by a fix"
        ),
    ),
    "fix_referenced__group": (
        _plain(
            put(MML_MODELS, [{"name": "optics", "engine": "pyat", "wiring": []}]),
            put("imported/mml/groups.yaml", [{"id": "SR/BPMS", "members": ["SR/BPM1"]}]),
            put("measurement/optics.yaml", {"kinds": ["orm"], "groups": {"bpm": "SR/BPMS"}}),
            fixes({"op": "drop", "kind": "group", "id": "SR/BPMS", "why": "x"}),
        ),
        (
            "facility: fix-referenced: group SR/BPMS — `drop` leaves 1 referrer(s): "
            "measurement/optics.yaml (groups); fix: drop or re-point each referrer by a fix"
        ),
    ),
    "reference_missing__on_device": (
        _plain(update("records/channels.yaml", 2, on={"device": "SR/BPM9"})),
        (
            "facility: reference-missing: channel BPM1:X — records/channels.yaml `on.device` "
            "names device SR/BPM9, which does not exist; fix: add device SR/BPM9 or correct "
            "`on.device`"
        ),
    ),
    "reference_missing__on_place": (
        _plain(update("records/channels.yaml", 2, on={"place": "SR"})),
        (
            "facility: reference-missing: channel BPM1:X — records/channels.yaml `on.place` names "
            "place SR, which does not exist; fix: add place SR or correct `on.place`"
        ),
    ),
    "reference_missing__pair": (
        _plain(
            append("records/channels.yaml", {"id": "Q2:SP", "role": "setpoint", "pair": "Q2:RB"})
        ),
        (
            "facility: reference-missing: channel Q2:SP — records/channels.yaml `pair` names "
            "channel Q2:RB, which does not exist; fix: add channel Q2:RB or correct `pair`"
        ),
    ),
    "reference_missing__endpoint_of": (
        _plain(update("records/channels.yaml", 2, endpoint_of=["SR/Q9"])),
        (
            "facility: reference-missing: channel BPM1:X — records/channels.yaml `endpoint_of` "
            "names device SR/Q9, which does not exist; fix: add device SR/Q9 or correct "
            "`endpoint_of`"
        ),
    ),
    "reference_missing__linear_input": (
        _plain(seeds(**{"BPM1:X": {"linear": {"Q9:SP": 1.0}}})),
        (
            "facility: reference-missing: channel BPM1:X — seeds.yaml `simulation.linear` names "
            "channel Q9:SP, which does not exist; fix: add channel Q9:SP or correct "
            "`simulation.linear`"
        ),
    ),
    "reference_missing__group_member": (
        _plain(put("records/groups.yaml", [{"id": "SR/QUADS", "members": ["SR/Q9"]}])),
        (
            "facility: reference-missing: group SR/QUADS — records/groups.yaml `members` names "
            "device SR/Q9, which does not exist; fix: add device SR/Q9 or correct `members`"
        ),
    ),
    "reference_missing__wiring_address": (
        _plain(
            put("models.yaml", [{**NO_DECK, "wiring": [{"address": "Q9:SP", "element": "Q9"}]}])
        ),
        (
            "facility: reference-missing: wiring optics/Q9:SP — models.yaml `address` names "
            "channel Q9:SP, which does not exist; fix: add channel Q9:SP or correct `address`"
        ),
    ),
    "reference_missing__slice_device": (
        _plain(
            put(
                "models.yaml",
                [
                    {
                        **NO_DECK,
                        "wiring": [
                            {
                                "address": "Q1:SP",
                                "slices": [{"element": "Q1A", "device": "SR/Q9"}],
                            }
                        ],
                    }
                ],
            )
        ),
        (
            "facility: reference-missing: wiring optics/Q1:SP — models.yaml `slices.device` names "
            "device SR/Q9, which does not exist; fix: add device SR/Q9 or correct `slices.device`"
        ),
    ),
    "reference_missing__span_model": (
        _plain(
            put(
                "records/places.yaml",
                [{"id": "SR", "span": {"model": "optics", "from_marker": "M"}}],
            )
        ),
        (
            "facility: reference-missing: place SR — records/places.yaml `span.model` names model "
            "optics, which does not exist; fix: add model optics or correct `span.model`"
        ),
    ),
    "reference_missing__device_place": (
        _plain(update("records/devices.yaml", 0, place="SR")),
        (
            "facility: reference-missing: device SR/Q1 — records/devices.yaml `place` names place "
            "SR, which does not exist; fix: add place SR or correct `place`"
        ),
    ),
    "reference_missing__place_parent": (
        _plain(put("records/places.yaml", [{"id": "SR/S01"}])),
        (
            "facility: reference-missing: place SR/S01 — records/places.yaml `id` names place SR, "
            "which does not exist; fix: add place SR or correct `id`"
        ),
    ),
    "reference_missing__seed_address": (
        _plain(seeds(**{"Q9:RB": {"nominal": 1.0}})),
        (
            "facility: reference-missing: seed Q9:RB — seeds.yaml `address` names channel Q9:RB, "
            "which does not exist; fix: add channel Q9:RB or correct `address`"
        ),
    ),
    "reference_missing__limit_address": (
        _plain(limits({"address": "Q9:SP", "min_value": 0.0, "max_value": 1.0})),
        (
            "facility: reference-missing: limit Q9:SP — limits.yaml `records.address` names "
            "channel Q9:SP, which does not exist; fix: add channel Q9:SP or correct "
            "`records.address`"
        ),
    ),
    "reference_missing__override": (
        _plain(scenario("warm", {"overrides": {"Q9:SP": 1.0}})),
        (
            "facility: reference-missing: scenario warm — scenarios/warm.yaml `overrides` names "
            "channel Q9:SP, which does not exist; fix: add channel Q9:SP or correct `overrides`"
        ),
    ),
    "reference_missing__archiver": (
        _plain(scenario("warm", {"archiver": [{"channel": "Q9:RB"}]})),
        (
            "facility: reference-missing: scenario warm — scenarios/warm.yaml `archiver.channel` "
            "names channel Q9:RB, which does not exist; fix: add channel Q9:RB or correct "
            "`archiver.channel`"
        ),
    ),
    "reference_missing__fault_model": (
        _plain(scenario("warm", {"faults": {"optics": {"BPM1:X": 1.0}}})),
        (
            "facility: reference-missing: scenario warm — scenarios/warm.yaml `faults` names "
            "model optics, which does not exist; fix: add model optics or correct `faults`"
        ),
    ),
    "reference_missing__fault_channel": (
        _plain(
            put("models.yaml", [NO_DECK]), scenario("warm", {"faults": {"optics": {"Q9:RB": 1.0}}})
        ),
        (
            "facility: reference-missing: scenario warm — scenarios/warm.yaml `faults.optics` "
            "names channel Q9:RB, which does not exist; fix: add channel Q9:RB or correct "
            "`faults.optics`"
        ),
    ),
    "reference_missing__measurement_model": (
        _plain(put("measurement/optics.yaml", {"kinds": ["orm"]})),
        (
            "facility: reference-missing: measurement optics — measurement/optics.yaml `file "
            "name` names model optics, which does not exist; fix: add model optics or correct "
            "`file name`"
        ),
    ),
    "reference_missing__measurement_group": (
        _plain(
            put("models.yaml", [NO_DECK]),
            put("measurement/optics.yaml", {"kinds": ["orm"], "groups": {"bpm": "SR/BPMS"}}),
        ),
        (
            "facility: reference-missing: measurement optics — measurement/optics.yaml "
            "`groups.bpm` names group SR/BPMS, which does not exist; fix: add group SR/BPMS or "
            "correct `groups.bpm`"
        ),
    ),
    "reference_missing__measurement_instrument": (
        _plain(
            put("models.yaml", [NO_DECK]),
            put("measurement/optics.yaml", {"kinds": ["orm"], "instruments": {"rf": "RF:FREQ"}}),
        ),
        (
            "facility: reference-missing: measurement optics — measurement/optics.yaml "
            "`instruments.rf` names channel RF:FREQ, which does not exist; fix: add channel "
            "RF:FREQ or correct `instruments.rf`"
        ),
    ),
    "class_unknown__device_class": (
        _plain(update("records/devices.yaml", 0, **{"class": "Warp"})),
        (
            "facility: class-unknown: device SR/Q1 — class Warp is in neither the vocabulary nor "
            "classes.yaml; fix: use a vocabulary class or add it to classes.yaml"
        ),
    ),
    "class_unknown__parent": (
        _plain(put("classes.yaml", [{"class": "Septum2", "parent": "Warp"}])),
        (
            "facility: class-unknown: class Septum2 — classes.yaml `parent` Warp is neither a "
            "vocabulary class nor an earlier facility-added class; fix: name a vocabulary class "
            "or a class listed earlier in classes.yaml"
        ),
    ),
    "pair_invalid__pair_on_readback": (
        _plain(update("records/channels.yaml", 2, pair="Q1:RB")),
        (
            "facility: pair-invalid: channel BPM1:X — a `pair` on a readback channel; fix: remove "
            "`pair`; only a setpoint names its readback"
        ),
    ),
    "pair_invalid__pair_on_none": (
        _plain(append("records/channels.yaml", {"id": "T", "role": "none", "pair": "Q1:RB"})),
        (
            "facility: pair-invalid: channel T — a `pair` on a none channel; fix: remove `pair`; "
            "only a setpoint names its readback"
        ),
    ),
    "pair_invalid__type_mismatch": (
        _plain(update("records/channels.yaml", 1, value_type="int")),
        (
            "facility: pair-invalid: channel Q1:SP — setpoint and pair Q1:RB differ in "
            "`value_type`; fix: give Q1:SP and Q1:RB the same `value_type`"
        ),
    ),
    "pair_invalid__element_and_slices": (
        _plain(
            put(
                "models.yaml",
                [
                    {
                        **NO_DECK,
                        "wiring": [
                            {"address": "Q1:SP", "element": "Q1", "slices": [{"element": "Q1"}]}
                        ],
                    }
                ],
            )
        ),
        (
            "facility: pair-invalid: wiring optics/Q1:SP — states both `element` and `slices`; "
            "fix: keep one of `element` and `slices`"
        ),
    ),
    "pair_invalid__weight_zero": (
        _plain(
            put(
                "models.yaml",
                [
                    {
                        **NO_DECK,
                        "wiring": [
                            {"address": "Q1:SP", "slices": [{"element": "Q1", "weight": 0}]}
                        ],
                    }
                ],
            )
        ),
        (
            "facility: pair-invalid: wiring optics/Q1:SP — slice weight 0 is zero or not finite; "
            "fix: give every slice a finite, non-zero weight"
        ),
    ),
    "pair_invalid__weight_non_finite": (
        _plain(
            put(
                "models.yaml",
                [
                    {
                        **NO_DECK,
                        "wiring": [
                            {
                                "address": "Q1:SP",
                                "slices": [{"element": "Q1", "weight": float("inf")}],
                            }
                        ],
                    }
                ],
            )
        ),
        (
            "facility: pair-invalid: wiring optics/Q1:SP — slice weight inf is zero or not "
            "finite; fix: give every slice a finite, non-zero weight"
        ),
    ),
    "pair_invalid__endpoint_unnamed": (
        _plain(
            update("records/channels.yaml", 0, endpoint_of=["SR/BPM1"]),
            put("models.yaml", [NO_DECK]),
        ),
        (
            "facility: pair-invalid: wiring optics/Q1:SP — `endpoint_of` device SR/BPM1 is named "
            "by no slice; fix: name each `endpoint_of` device in a slice, or remove it from "
            "`endpoint_of`"
        ),
    ),
    "value_invalid__nominal_float": (
        _plain(seeds(**{"BPM1:X": {"nominal": "high"}})),
        (
            "facility: value-invalid: channel BPM1:X — `nominal` of channel BPM1:X: value 'high' "
            "is not a valid float: expected a finite int or float; fix: write a float value for "
            "`nominal`"
        ),
    ),
    "value_invalid__nominal_bool": (
        _plain(
            append(
                "records/channels.yaml", {"id": "T", "value_type": "bool", "options": ["OFF", "ON"]}
            ),
            seeds(T={"nominal": "MAYBE"}),
        ),
        (
            "facility: value-invalid: channel T — `nominal` of channel T: value 'MAYBE' is not a "
            "valid bool: label not in ['OFF', 'ON']; fix: write a bool value for `nominal`"
        ),
    ),
    "value_invalid__nominal_enum": (
        _plain(
            append(
                "records/channels.yaml",
                {"id": "T", "value_type": "enum", "options": ["A", "B", "C"]},
            ),
            seeds(T={"nominal": 5}),
        ),
        (
            "facility: value-invalid: channel T — `nominal` of channel T: value 5 is not a valid "
            "enum: index outside 0..2; fix: write a enum value for `nominal`"
        ),
    ),
    "value_invalid__nominal_string": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "string"}),
            seeds(T={"nominal": 3}),
        ),
        (
            "facility: value-invalid: channel T — `nominal` of channel T: value 3 is not a valid "
            "string: expected a str; fix: write a string value for `nominal`"
        ),
    ),
    "value_invalid__nominal_waveform": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "waveform", "shape": [3]}),
            seeds(T={"nominal": [1.0, 2.0]}),
        ),
        (
            "facility: value-invalid: channel T — `nominal` of channel T: value [1.0, 2.0] is not "
            "a valid waveform: expected 3 values for shape [3]; fix: write a waveform value for "
            "`nominal`"
        ),
    ),
    "value_invalid__override": (
        _plain(scenario("warm", {"overrides": {"BPM1:X": "high"}})),
        (
            "facility: value-invalid: scenario warm — `overrides.BPM1:X` of channel BPM1:X: value "
            "'high' is not a valid float: expected a finite int or float; fix: write a float "
            "value for `overrides.BPM1:X`"
        ),
    ),
    "value_invalid__fault": (
        _plain(
            put("models.yaml", [NO_DECK]),
            scenario("warm", {"faults": {"optics": {"BPM1:X": "high"}}}),
        ),
        (
            "facility: value-invalid: scenario warm — `faults.optics.BPM1:X` of channel BPM1:X: "
            "value 'high' is not a valid float: expected a finite int or float; fix: write a "
            "float value for `faults.optics.BPM1:X`"
        ),
    ),
    "value_invalid__options_presence": (
        _plain(update("records/channels.yaml", 2, options=["A", "B"])),
        (
            "facility: value-invalid: channel BPM1:X — a float channel carries `options`; fix: "
            "state `options` only on bool and enum channels and `shape` only on waveforms"
        ),
    ),
    "value_invalid__shape_presence": (
        _plain(append("records/channels.yaml", {"id": "T", "value_type": "waveform"})),
        (
            "facility: value-invalid: channel T — a waveform channel needs `shape`, a list of "
            "positive ints; fix: state `options` only on bool and enum channels and `shape` only "
            "on waveforms"
        ),
    ),
    "value_invalid__motion_non_float": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "int"}),
            seeds(T={"noise": 1.0}),
        ),
        (
            "facility: value-invalid: channel T — a int channel carries `noise`; motion, `clamp` "
            "and `linear` apply to float channels only; fix: remove `noise`"
        ),
    ),
    "value_invalid__enum_bounds": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "enum", "options": ["ONLY"]})
        ),
        (
            "facility: value-invalid: channel T — an enum channel has fewer than 2 `options`; "
            "fix: state `options` only on bool and enum channels and `shape` only on waveforms"
        ),
    ),
    "value_invalid__linear_wired": (
        _plain(put("models.yaml", [NO_DECK]), seeds(**{"BPM1:X": {"linear": {"Q1:SP": 1.0}}})),
        (
            "facility: value-invalid: channel BPM1:X — `linear` input Q1:SP is wired; fix: take "
            "`linear` inputs from unwired float channels and drop `nominal`"
        ),
    ),
    "value_invalid__linear_cyclic": (
        _plain(
            append("records/channels.yaml", {"id": "T"}),
            seeds(**{"BPM1:X": {"linear": {"T": 1.0}}, "T": {"linear": {"BPM1:X": 1.0}}}),
        ),
        (
            "facility: value-invalid: channel BPM1:X — `linear` inputs form a cycle: BPM1:X -> T "
            "-> BPM1:X; fix: break the cycle"
        ),
    ),
    "value_invalid__linear_nominal": (
        _plain(seeds(**{"BPM1:X": {"nominal": 1.0, "linear": {"Q1:RB": 1.0}}})),
        (
            "facility: value-invalid: channel BPM1:X — carries `nominal` beside `linear`; fix: "
            "take `linear` inputs from unwired float channels and drop `nominal`"
        ),
    ),
    "value_invalid__linear_non_float": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "int"}),
            seeds(T={"linear": {"BPM1:X": 1.0}}),
        ),
        (
            "facility: value-invalid: channel T — a int channel carries `linear`; motion, `clamp` "
            "and `linear` apply to float channels only; fix: remove `linear`"
        ),
    ),
    "value_invalid__drift_period": (
        _plain(seeds(**{"BPM1:X": {"drift": {"amplitude": 1.0, "period_s": 0}}})),
        (
            "facility: value-invalid: channel BPM1:X — `drift.period_s` is 0, not above 0; fix: "
            "set `drift.period_s` above 0"
        ),
    ),
    "value_invalid__limit_bounds_non_numeric": (
        _plain(
            append(
                "records/channels.yaml", {"id": "T", "role": "setpoint", "value_type": "string"}
            ),
            limits({"address": "T", "max_step": 1.0}),
        ),
        (
            "facility: value-invalid: limit T — `max_step` on a string channel; fix: remove "
            "`max_step`; a string channel carries `writable` and `confirm` only"
        ),
    ),
    "value_invalid__clamp_order": (
        _plain(seeds(**{"BPM1:X": {"clamp": [2.0, 1.0]}})),
        (
            "facility: value-invalid: channel BPM1:X — `clamp` low 2.0 is above high 1.0; fix: "
            "write `clamp` as [low, high] with low <= high; either side may be null"
        ),
    ),
    "value_invalid__clamp_non_float": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "int"}),
            seeds(T={"clamp": [0, 1]}),
        ),
        (
            "facility: value-invalid: channel T — a int channel carries `clamp`; motion, `clamp` "
            "and `linear` apply to float channels only; fix: remove `clamp`"
        ),
    ),
    "value_invalid__stuck_non_setpoint": (
        _plain(
            put("models.yaml", [NO_DECK]),
            scenario("warm", {"faults": {"optics": {"BPM1:X": "stuck"}}}),
        ),
        (
            "facility: value-invalid: scenario warm — `faults.optics.BPM1:X` is `stuck` on a "
            "readback channel; fix: fault a setpoint with `stuck`, or write a value"
        ),
    ),
    "seed_invalid__nominal_out_of_band": (
        _plain(
            append("records/channels.yaml", EXTRA_SP),
            seeds(**{"Q2:SP": {"nominal": 5.0}}),
            limits({"address": "Q2:SP", "min_value": 0.0, "max_value": 1.0}),
        ),
        (
            "facility: seed-invalid: channel Q2:SP — nominal 5 lies above `max_value` 1; fix: "
            "move the operating point inside [min_value, max_value], or widen the limits record"
        ),
    ),
    "seed_invalid__nominal_wired": (
        _plain(put("models.yaml", [NO_DECK]), seeds(**{"Q1:SP": {"nominal": 1.0}})),
        (
            "facility: seed-invalid: channel Q1:SP — a `nominal` on a channel model optics wires; "
            "fix: remove `nominal`; the operating point comes from the deck"
        ),
    ),
    "seed_invalid__motion_on_setpoint": (
        _plain(seeds(**{"Q1:SP": {"noise": 0.1}})),
        (
            "facility: seed-invalid: channel Q1:SP — a setpoint carries `noise`; fix: remove "
            "`noise`; a setpoint holds the value written to it"
        ),
    ),
    "seed_invalid__paired_seed": (
        _plain(seeds(**{"Q1:SP": {"nominal": 1.0}, "Q1:RB": {"nominal": 2.0}})),
        (
            "facility: seed-invalid: channel Q1:RB — `nominal` 2.0 differs from its setpoint "
            "Q1:SP's 1.0; fix: remove `nominal` from Q1:RB; a paired readback starts at its "
            "setpoint's value"
        ),
    ),
    "seed_invalid__int_nominal": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "int"}),
            seeds(T={"nominal": 1.5}),
        ),
        (
            "facility: seed-invalid: channel T — `nominal` of int channel T is 1.5, not integral; "
            "fix: write an integral `nominal`"
        ),
    ),
    "seed_invalid__int_override": (
        _plain(
            append("records/channels.yaml", {"id": "T", "value_type": "int"}),
            scenario("warm", {"overrides": {"T": 1.5}}),
        ),
        (
            "facility: seed-invalid: scenario warm — `overrides.T` of int channel T is 1.5, not "
            "integral; fix: write an integral `overrides.T`"
        ),
    ),
    "limit_invalid__writable": (
        _plain(limits({"address": "Q1:RB", "writable": True})),
        (
            "facility: limit-invalid: limit Q1:RB — `writable: true` on a readback channel; fix: "
            "remove `writable`; only a setpoint is writable"
        ),
    ),
    "limit_invalid__int_bound": (
        _plain(
            append("records/channels.yaml", {"id": "T", "role": "setpoint", "value_type": "int"}),
            limits({"address": "T", "min_value": 0, "max_value": 2.5}),
        ),
        (
            "facility: limit-invalid: limit T — `max_value` 2.5 on an int channel is not "
            "integral; fix: write integral bounds"
        ),
    ),
    "place_conflict__imported_place": (
        _deck(
            append(
                "records/places.yaml",
                {"id": "SR/A", "span": {"model": "SR", "from_marker": "M1", "to_marker": "M2"}},
                {"id": "SR/B", "span": {"model": "SR", "from_marker": "M2", "to_marker": "M1"}},
            ),
            put("imported/mml/devices.yaml", [{"id": "SR/QD", "place": "SR/B"}]),
        ),
        (
            "facility: place-conflict: device SR/QD — layer mml states place SR/B, but the span "
            "of place SR/A holds the device at s 1.25 in model SR; fix: drop `place` from the "
            "layer, or add a fix `set` of place SR/B"
        ),
    ),
    "span_invalid__overlap": (
        _deck(
            append(
                "records/places.yaml", {"id": "SR2", "span": {"model": "SR", "from_marker": "M2"}}
            )
        ),
        (
            "facility: span-invalid: place SR2 — span overlaps place SR's span in model SR; fix: "
            "make the spans of one level disjoint"
        ),
    ),
    "span_invalid__marker_absent": (
        _deck(update("records/places.yaml", 0, span={"model": "SR", "from_marker": "NOPE"})),
        (
            "facility: span-invalid: place SR — `span.from_marker`: element NOPE is not in the "
            "deck of model SR; fix: name a marker that appears once in the deck"
        ),
    ),
    "span_invalid__marker_not_unique": (
        _deck(update("records/places.yaml", 0, span={"model": "SR", "from_marker": "D"})),
        (
            "facility: span-invalid: place SR — `span.from_marker`: element D appears 3 times in "
            "the deck of model SR; fix: name a marker that appears once in the deck"
        ),
    ),
    "span_invalid__no_deck": (
        _deck(
            append("models.yaml", {"name": "BOOST", "engine": "pyat", "wiring": []}),
            append(
                "records/places.yaml", {"id": "BR", "span": {"model": "BOOST", "from_marker": "M0"}}
            ),
        ),
        (
            "facility: span-invalid: place BR — `span.model` BOOST has no deck to place the span "
            "in; fix: name a model with a deck, or place the devices by hand"
        ),
    ),
    "wiring_conflict__device_two_models": (
        _deck(
            append(
                "records/channels.yaml",
                {"id": "QD2:SP", "role": "setpoint", "on": {"device": "SR/QD"}},
            ),
            wire("LINE", {"address": "QD2:SP", "element": "Q1", "engine": SETTING}),
        ),
        (
            "facility: wiring-conflict: device SR/QD — models LINE, SR each wire an element of "
            "the device; fix: wire the device's elements in one model"
        ),
    ),
    "wiring_conflict__address_twice": (
        _deck(
            append("records/channels.yaml", {"id": "TUNE:X"}),
            wire("SR", {"address": "TUNE:X", "element": "QD", "engine": READING}),
            wire("LINE", {"address": "TUNE:X", "element": "Q1", "engine": READING}),
        ),
        (
            "facility: wiring-conflict: channel TUNE:X — models LINE, SR each wire the address; "
            "fix: wire the address in one model"
        ),
    ),
    "wiring_conflict__address_twice_on_device": (
        _deck(
            append(
                "records/channels.yaml",
                {"id": "QD2:SP", "role": "setpoint", "on": {"device": "SR/QD"}},
            ),
            wire("SR", {"address": "QD2:SP", "element": "QD", "engine": SETTING}),
            wire("LINE", {"address": "QD2:SP", "element": "Q1", "engine": SETTING}),
        ),
        (
            "facility: wiring-conflict: channel QD2:SP — models LINE, SR each wire the address; "
            "fix: wire the address in one model"
        ),
    ),
    "wiring_conflict__repeated_element": (
        _deck(_set_slice("element", "D")),
        (
            "facility: wiring-conflict: wiring SR/QD:SP — element D appears 3 times in the deck "
            "of model SR; fix: give the element a unique name in the deck"
        ),
    ),
    "wiring_conflict__repeated_first_slice": (
        _deck(_set_slice("first slice", "D")),
        (
            "facility: wiring-conflict: wiring SR/QF:SP — element D appears 3 times in the deck "
            "of model SR; fix: give the element a unique name in the deck"
        ),
    ),
    "wiring_conflict__repeated_later_slice": (
        _deck(_set_slice("later slice", "D")),
        (
            "facility: wiring-conflict: wiring SR/QF:SP — element D appears 3 times in the deck "
            "of model SR; fix: give the element a unique name in the deck"
        ),
    ),
    "wiring_conflict__repeated_readback": (
        _deck(_set_slice("readback", "D")),
        (
            "facility: wiring-conflict: wiring SR/BPM1:X — element D appears 3 times in the deck "
            "of model SR; fix: give the element a unique name in the deck"
        ),
    ),
    "engine_missing__no_plugin": (
        _deck(
            on_model(
                "LINE", lambda record: record.update(engine="warpcore", settings={"warpcore": {}})
            )
        ),
        (
            "facility: engine-missing: model LINE — engine warpcore is not registered under "
            "osprey.simulation.engines; fix: install the engine's package, or name an engine the "
            "environment registers"
        ),
    ),
    "engine_invalid__single_pass_twiss": (
        _deck(
            on_model(
                "LINE", lambda record: record.update(settings={"pyat": {"solve": "single_pass"}})
            )
        ),
        (
            "facility: engine-invalid: model LINE — settings key pyat.twiss_in is required when "
            "solve is single_pass; fix: state twiss_in with beta and alpha"
        ),
    ),
    "engine_invalid__twiss_length": (
        _deck(
            on_model(
                "LINE",
                lambda record: record.update(
                    settings={
                        "pyat": {
                            "solve": "single_pass",
                            "twiss_in": {"beta": [5.0, 3.0, 1.0], "alpha": [0.0, 0.0]},
                        }
                    }
                ),
            )
        ),
        (
            "facility: engine-invalid: model LINE — settings key pyat.twiss_in.beta must be 2 "
            "finite numbers; fix: give twiss_in.beta 2 numbers"
        ),
    ),
    "engine_invalid__frozen_cavity": (
        _deck(put("decks/sr.json", sr_deck(_cavity))),
        (
            "facility: engine-invalid: model SR — cavity RFC has no longitudinal motion in a "
            "periodic deck; fix: give the cavity a longitudinal pass method such as RFCavityPass"
        ),
    ),
    "engine_invalid__repeated_monitors": (
        _deck(put("decks/sr.json", sr_deck(lambda at: at.Monitor("BPM1")))),
        (
            "facility: engine-invalid: model SR — monitor names repeat in the deck: BPM1; fix: "
            "give every monitor a unique name in the deck"
        ),
    ),
    "engine_invalid__imported_frozen_cavity": (
        _deck(imported_sr, put(MML_DECK, sr_deck(_cavity))),
        (
            "facility: engine-invalid: model SR — cavity RFC has no longitudinal motion in a "
            "periodic deck; fix: give the cavity a longitudinal pass method such as RFCavityPass"
        ),
    ),
    "engine_invalid__imported_repeated_monitors": (
        _deck(imported_sr, put(MML_DECK, sr_deck(lambda at: at.Monitor("BPM1")))),
        (
            "facility: engine-invalid: model SR — monitor names repeat in the deck: BPM1; fix: "
            "give every monitor a unique name in the deck"
        ),
    ),
    "engine_invalid__missing_element": (
        _deck(_set_slice("element", "GHOST")),
        (
            "facility: engine-invalid: wiring SR/QD:SP — element GHOST is not in the deck of "
            "model SR; fix: name an element the deck holds"
        ),
    ),
    "engine_invalid__imported_missing_element": (
        _deck(imported_sr, imported_wiring(1, element="GHOST")),
        (
            "facility: engine-invalid: wiring SR/QD:SP — element GHOST is not in the deck of "
            "model SR; fix: name an element the deck holds"
        ),
    ),
    "engine_invalid__missing_first_slice": (
        _deck(_set_slice("first slice", "GHOST")),
        (
            "facility: engine-invalid: wiring SR/QF:SP — element GHOST is not in the deck of "
            "model SR; fix: name an element the deck holds"
        ),
    ),
    "engine_invalid__missing_later_slice": (
        _deck(_set_slice("later slice", "GHOST")),
        (
            "facility: engine-invalid: wiring SR/QF:SP — element GHOST is not in the deck of "
            "model SR; fix: name an element the deck holds"
        ),
    ),
    "engine_invalid__missing_readback": (
        _deck(_set_slice("readback", "GHOST")),
        (
            "facility: engine-invalid: wiring SR/BPM1:X — element GHOST is not in the deck of "
            "model SR; fix: name an element the deck holds"
        ),
    ),
    "engine_invalid__table_without_inverse": (
        _deck(
            wiring(
                "SR",
                1,
                calibration={"curve": {"table": {"grid": [0.0, 1.0], "values": [0.0, 2.0]}}},
            )
        ),
        (
            "facility: engine-invalid: wiring SR/QD:SP — a table calibration has no inverse to "
            "derive the start value; fix: add calibration.inverse"
        ),
    ),
    "engine_invalid__linear_gain_zero": (
        _deck(wiring("SR", 1, calibration={"curve": {"linear": {"gain": 0.0, "offset": 0.0}}})),
        (
            "facility: engine-invalid: wiring SR/QD:SP — the linear calibration's gain is 0, so "
            "it has no inverse; fix: give the calibration a non-zero gain"
        ),
    ),
    "model_conflict__texture": (
        _plain(put("models.yaml", [{"name": "texture", "engine": "texture", "wiring": []}])),
        (
            "facility: model-conflict: model texture — a source declares model texture, which the "
            "build provides; fix: remove model texture; every channel no model wires is texture's"
        ),
    ),
    "model_conflict__status_address": (
        _plain(
            put("models.yaml", [NO_DECK]),
            append("records/channels.yaml", {"id": "demo:SIM:optics:STATUS"}),
        ),
        (
            "facility: model-conflict: channel demo:SIM:optics:STATUS — the address is model "
            "optics's status channel; fix: rename the channel; the simulator serves the status "
            "address itself"
        ),
    ),
    "profile_invalid__mirrored_facility_file": (
        _plain(),
        (
            "facility: profile-invalid: path project/facility.json — the project/ mirror writes "
            "facility.json, which the build writes from data/facility/; fix: remove "
            "project/facility.json and author the facility in data/facility/"
        ),
    ),
    "profile_invalid__mirrored_simulator_view": (
        _plain(),
        (
            "facility: profile-invalid: path project/data/simulator/x.json — the project/ "
            "mirror writes data/simulator/x.json, which the build writes from data/facility/; "
            "fix: remove project/data/simulator/x.json and author the facility in data/facility/"
        ),
    ),
    "profile_invalid__unknown_served_model": (
        _deck(),
        (
            "facility: profile-invalid: path simulation.models — `NOPE` is not a model in the "
            "facility file; its models are `LINE`, `SR`, `texture`; fix: name only models the "
            "facility file holds in `simulation.models`"
        ),
    ),
    "profile_invalid__persona_served_models": (
        _plain(),
        (
            "facility: profile-invalid: path personas/reader.yml — a persona on the `va` target "
            "sets `simulation.models`, and that target serves the deployment's list; fix: remove "
            "`simulation.models` from personas/reader.yml"
        ),
    ),
}

#: The files a case puts in the profile's ``project/`` mirror, beside a clean tree.
MIRRORED: dict[str, tuple[str, ...]] = {
    "profile_invalid__mirrored_facility_file": ("facility.json",),
    "profile_invalid__mirrored_simulator_view": ("data/simulator/x.json",),
}


def _serve_unknown_model(repo: Path) -> None:
    profile = repo / "profile.yml"
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    data.setdefault("config", {})["simulation.models"] = ["NOPE"]
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _persona_serves_models(repo: Path) -> None:
    persona = repo / "personas" / "reader.yml"
    persona.parent.mkdir(parents=True, exist_ok=True)
    persona.write_text("name: reader\nconfig:\n  simulation.models: [texture]\n", encoding="utf-8")


#: The profile edit a case makes beside a clean tree.
PROFILE_EDITS: dict[str, Callable[[Path], None]] = {
    "profile_invalid__unknown_served_model": _serve_unknown_model,
    "profile_invalid__persona_served_models": _persona_serves_models,
}

#: Cases only ``osprey build`` stops on: validate checks the main profile render, and
#: persona renders are checked by the build.
BUILD_ONLY: frozenset[str] = frozenset({"profile_invalid__persona_served_models"})

IDS = list(CASES)


def _kind_of(line: str) -> str:
    return line.split(": ", 2)[1]


def _write(tmp_path: Path, case: str) -> Path:
    make, _line = CASES[case]
    tree = make()
    if any(rel.startswith(("decks/", "imported/mml/decks/")) for rel in tree):
        pytest.importorskip("at")
    return write_tree(tmp_path / "facility", tree)


# --- every stop ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant repo named for the project, never edited."""
    return init_project(tmp_path_factory.mktemp("ca"), "control-assistant", PROJECT)


def _repo(initialised: Path, tmp_path: Path, case: str) -> Path:
    """A copy of the initialised repo whose ``data/facility`` is the case's tree.

    The case's ``project/`` mirror files and profile edit are applied beside it.
    """
    repo = tmp_path / PROJECT
    shutil.copytree(initialised, repo, symlinks=True)
    shutil.rmtree(repo / "data" / "facility")
    _write(repo / "data", case)
    for rel in MIRRORED.get(case, ()):
        mirrored = repo / "project" / rel
        mirrored.parent.mkdir(parents=True, exist_ok=True)
        mirrored.write_text("{}\n", encoding="utf-8")
    edit = PROFILE_EDITS.get(case)
    if edit is not None:
        edit(repo)
    return repo


@pytest.mark.slow
@pytest.mark.parametrize("case", IDS)
def test_build_stops_on_the_line(initialised: Path, tmp_path: Path, case: str) -> None:
    repo = _repo(initialised, tmp_path, case)

    result = run_build(repo)

    assert result.exit_code == 1, result.output
    assert result.stderr == CASES[case][1] + "\n"


@pytest.mark.slow
@pytest.mark.parametrize("case", [case for case in IDS if case not in BUILD_ONLY])
def test_validate_prints_the_one_line(initialised: Path, tmp_path: Path, case: str) -> None:
    repo = _repo(initialised, tmp_path, case)

    result = CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])

    assert result.exit_code == 1, result.output
    assert (result.stdout, result.stderr) == ("", CASES[case][1] + "\n")


# --- the table and the cases agree -------------------------------------------------


def test_the_case_ids_are_the_stop_sentences() -> None:
    assert IDS == [case for case, _kind, _sentence in STOP_SENTENCES]


def test_each_case_stops_with_its_sentence_kind() -> None:
    kinds = {case: kind for case, kind, _sentence in STOP_SENTENCES}
    assert {case: _kind_of(line) for case, (_make, line) in CASES.items()} == kinds


def test_every_kind_has_a_case() -> None:
    covered = {kind for _case, kind, _sentence in STOP_SENTENCES}
    assert [kind for kind in KINDS if kind not in covered] == []


def test_each_sentence_is_listed_once() -> None:
    sentences = [(kind, sentence) for _case, kind, sentence in STOP_SENTENCES]
    assert len(sentences) == len(set(sentences))


# --- what builds ---------------------------------------------------------------------

#: Fixes of every op, on records a second layer states.
PERMUTED_FIXES: tuple[dict[str, Any], ...] = (
    {
        "op": "set",
        "kind": "channel",
        "id": "BPM1:X",
        "fields": {"unit": "um"},
        "was": {"unit": {"mml": "mm"}},
        "why": "The export rounds the unit.",
    },
    {"op": "add", "kind": "device", "id": "SR/Q9", "record": {"class": QUAD}, "why": "a"},
    {"op": "drop", "kind": "device", "id": "SR/SPARE", "why": "b"},
    {"op": "drop", "kind": "channel", "id": "SPARE:X", "why": "c"},
)


def test_permuted_fixes_build_a_byte_equal_file(tmp_path: Path) -> None:
    documents = set()
    for index, order in enumerate(itertools.permutations(PERMUTED_FIXES)):
        tree = _plain(
            put(MML_CHANNELS, [{"id": "BPM1:X", "unit": "mm"}, {"id": "SPARE:X"}]),
            put("imported/mml/devices.yaml", [{"id": "SR/SPARE", "class": BPM}]),
            fixes(*order),
        )()
        root = write_tree(tmp_path / str(index) / "facility", tree)
        documents.add(json.dumps(build_facility(root, project_name=PROJECT)))
    assert len(documents) == 1


def test_two_layers_stating_and_defaulting_a_slot_build(tmp_path: Path) -> None:
    tree = _plain(
        put(MML_CHANNELS, [{"id": "BPM1:X", "role": "readback", "value_type": "float"}]),
        put(
            "models.yaml",
            [{**NO_DECK, "wiring": [{"address": "Q1:SP", "slices": [{"element": "Q1"}]}]}],
        ),
        put(
            MML_MODELS,
            [
                {
                    "name": "optics",
                    "wiring": [{"address": "Q1:SP", "slices": [{"element": "Q1", "weight": 1}]}],
                }
            ],
        ),
    )()
    document = build_facility(write_tree(tmp_path / "facility", tree), project_name=PROJECT)
    channel = next(c for c in document["channels"] if c["id"] == "BPM1:X")
    assert (channel["role"], channel["value_type"]) == ("readback", "float")
    (record,) = next(m for m in document["models"] if m["name"] == "optics")["wiring"]
    assert [(piece["element"], piece["weight"]) for piece in record["slices"]] == [("Q1", 1)]


def test_limits_records_without_a_defaults_block_build(tmp_path: Path) -> None:
    tree = _plain(limits({"address": "Q1:SP", "min_value": 0.0, "max_value": 1.0}))()
    document = build_facility(write_tree(tmp_path / "facility", tree), project_name=PROJECT)
    assert document["limits"] == {
        "records": [{"address": "Q1:SP", "min_value": 0.0, "max_value": 1.0}]
    }


def test_a_tree_without_limits_builds(tmp_path: Path) -> None:
    root = write_tree(tmp_path / "facility", plain_tree())
    assert not (root / "limits.yaml").exists()
    assert "limits" not in build_facility(root, project_name=PROJECT)

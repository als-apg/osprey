"""Tests for the declared channel contract (`plan_fields`).

The declaring half -- what an author writes -- is pinned on three properties:

1. **Serialization** -- the declared role reaches ``model_json_schema()``, both
   for a top-level field and for a field on a nested model (where it lands
   inside the nested ``$defs`` entry).
2. **Additivity** -- the role only *adds* the ``x-channel-role`` key. Every
   other constraint, title, description and presentation hint an author writes
   on the field survives untouched, which is why the panel's schema-driven form
   generator keeps rendering the same controls it rendered before the role was
   declared.
3. **Introspectability** -- the role is readable from ``model_fields`` as a
   plain dict, the seam the role-walking helpers use.

The reading half -- what consumers do with it -- is pinned on the behaviors
they each depend on: `channel_roles` sees through nesting and stops at a cycle;
`collect_channels` matches the *nearest enclosing* field name and matches it
exactly; `scan_metadata` emits the standard run-metadata keys and is the only
place in the module they are spelled; and `resolve_column` prefers a channel's
own column over the prefixed spelling and answers ``None`` rather than raising.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from osprey.services.bluesky_bridge import plan_fields
from osprey.services.bluesky_bridge.app import _BRIDGE_ONLY_MODULES
from osprey.services.bluesky_bridge.plan_fields import (
    CHANNEL_ROLE_KEY,
    MOVABLE_ROLE,
    READABLE_ROLE,
    MovableChannel,
    MovableChannels,
    OptionalMovableChannel,
    OptionalReadableChannel,
    ReadableChannel,
    ReadableChannels,
    channel_roles,
    collect_channels,
    declared_channels,
    resolve_column,
    scan_metadata,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_FORM_JS = _REPO_ROOT / "src/osprey/interfaces/bluesky_panels/panels/bluesky/schema-form.js"


class _Axis(BaseModel):
    """A nested model holding one movable channel -- the `GridAxis` shape."""

    setpoint: MovableChannel
    start: float


class _Params(BaseModel):
    """A plan-shaped model exercising every role type at once."""

    correctors: MovableChannels = Field(
        ...,
        min_length=1,
        title="Correctors",
        description="Channels driven one at a time.",
        json_schema_extra={"x-widget": "channel-list"},
    )
    monitors: ReadableChannels = Field(..., min_length=1, description="Channels recorded.")
    reference: ReadableChannel = Field(..., description="One recorded channel.")
    axes: list[_Axis] = Field(..., min_length=1)


class _PlainAxis(BaseModel):
    """`_Axis` with the role annotations removed -- the additivity baseline."""

    setpoint: str
    start: float


class _PlainParams(BaseModel):
    """`_Params` with the role annotations removed -- the additivity baseline."""

    correctors: list[str] = Field(
        ...,
        min_length=1,
        title="Correctors",
        description="Channels driven one at a time.",
        json_schema_extra={"x-widget": "channel-list"},
    )
    monitors: list[str] = Field(..., min_length=1, description="Channels recorded.")
    reference: str = Field(..., description="One recorded channel.")
    axes: list[_PlainAxis] = Field(..., min_length=1)


def _strip_roles(node: Any) -> Any:
    """Recursively drop every ``x-channel-role`` key from a JSON schema."""
    if isinstance(node, dict):
        return {k: _strip_roles(v) for k, v in node.items() if k != CHANNEL_ROLE_KEY}
    if isinstance(node, list):
        return [_strip_roles(v) for v in node]
    return node


def _rename_defs(schema: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """Rewrite a schema's ``$defs`` name (and its ``$ref``s) so two structurally
    identical models can be compared without their class names getting in the way.
    """
    return json.loads(json.dumps(schema).replace(old, new))


# --- serialization ----------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "role"),
    [
        ("correctors", MOVABLE_ROLE),
        ("monitors", READABLE_ROLE),
        ("reference", READABLE_ROLE),
    ],
)
def test_top_level_field_serializes_its_role(field_name: str, role: str) -> None:
    """A role-typed field on the model itself carries its role in the schema."""
    prop = _Params.model_json_schema()["properties"][field_name]
    assert prop[CHANNEL_ROLE_KEY] == role


def test_nested_model_field_serializes_its_role_under_defs() -> None:
    """A role-typed field on a *nested* model carries its role inside that
    model's ``$defs`` entry -- the placement `GridAxis.setpoint` produces, and
    the reason consumers must resolve ``$ref``s to find roles.
    """
    schema = _Params.model_json_schema()

    assert schema["properties"]["axes"]["items"] == {"$ref": "#/$defs/_Axis"}
    assert schema["$defs"]["_Axis"]["properties"]["setpoint"][CHANNEL_ROLE_KEY] == MOVABLE_ROLE
    # The role belongs to the annotated field, not to the model or its siblings.
    assert CHANNEL_ROLE_KEY not in schema["$defs"]["_Axis"]
    assert CHANNEL_ROLE_KEY not in schema["$defs"]["_Axis"]["properties"]["start"]


def test_role_types_still_validate_as_their_underlying_types() -> None:
    """The role declares intent and nothing else: the list/str validation, and
    any constraint the author added, behave exactly as they would untyped.
    """
    params = _Params(
        correctors=["CORR:1"],
        monitors=["BPM:1"],
        reference="BPM:REF",
        axes=[{"setpoint": "CORR:2", "start": 0.0}],
    )
    assert params.correctors == ["CORR:1"]
    assert params.axes[0].setpoint == "CORR:2"

    with pytest.raises(ValidationError):
        _Params(
            correctors=[],  # min_length=1 still applies
            monitors=["BPM:1"],
            reference="BPM:REF",
            axes=[{"setpoint": "CORR:2", "start": 0.0}],
        )
    with pytest.raises(ValidationError):
        _Params(
            correctors="CORR:1",  # type: ignore[arg-type]  # still a list, not a str
            monitors=["BPM:1"],
            reference="BPM:REF",
            axes=[{"setpoint": "CORR:2", "start": 0.0}],
        )


# --- additivity -------------------------------------------------------------


def test_role_is_the_only_difference_from_an_unannotated_model() -> None:
    """Declaring roles adds ``x-channel-role`` keys and changes nothing else.

    Everything the panel's form generator dispatches on -- ``type``, ``items``,
    ``enum``, ``$ref``, ``properties``, ``x-widget`` -- is byte-identical to the
    same model written without roles, so the generator renders the same controls.
    """
    declared = _rename_defs(_Params.model_json_schema(), "_Axis", "Axis")
    plain = _rename_defs(_PlainParams.model_json_schema(), "_PlainAxis", "Axis")
    # The two models' own titles and docstrings differ by construction; every
    # per-field entry below them is what this test compares.
    for schema in (declared, plain):
        schema.pop("title", None)
        schema.pop("description", None)
        schema["$defs"]["Axis"].pop("title", None)
        schema["$defs"]["Axis"].pop("description", None)

    assert _strip_roles(declared) == plain
    # Guard against a vacuous pass: the declared schema really did carry roles.
    assert declared != plain


def test_field_level_extras_survive_alongside_the_role() -> None:
    """An author's own ``json_schema_extra`` merges with the role rather than
    replacing it. Channel-list fields carry an ``x-widget`` presentation hint
    (the shipped ``orm`` parameters do), so a pydantic release that clobbered
    instead of merging would silently erase their declared roles -- this is the
    tripwire for that.
    """
    prop = _Params.model_json_schema()["properties"]["correctors"]
    assert prop[CHANNEL_ROLE_KEY] == MOVABLE_ROLE
    assert prop["x-widget"] == "channel-list"
    assert prop["title"] == "Correctors"
    assert prop["minItems"] == 1


def test_role_type_is_not_mutated_by_a_field_that_adds_extras() -> None:
    """Reusing a role type must not carry one model's extras into the next.

    `_Params.correctors` adds an ``x-widget`` hint to `MovableChannels`; a model
    declared afterwards with the same type must come out clean.
    """
    assert _Params.model_json_schema()["properties"]["correctors"]["x-widget"] == "channel-list"

    class _Later(BaseModel):
        channels: MovableChannels

    prop = _Later.model_json_schema()["properties"]["channels"]
    assert prop[CHANNEL_ROLE_KEY] == MOVABLE_ROLE
    assert "x-widget" not in prop


# --- introspectability ------------------------------------------------------


def test_role_is_readable_from_model_fields() -> None:
    """The role is a plain dict on the field info -- the seam the role-walking
    helpers read, so they never have to serialize a schema to learn a role.
    """
    extras = {name: field.json_schema_extra for name, field in _Params.model_fields.items()}
    assert extras["correctors"] == {CHANNEL_ROLE_KEY: MOVABLE_ROLE, "x-widget": "channel-list"}
    assert extras["monitors"] == {CHANNEL_ROLE_KEY: READABLE_ROLE}
    assert extras["reference"] == {CHANNEL_ROLE_KEY: READABLE_ROLE}
    assert extras["axes"] is None
    assert _Axis.model_fields["setpoint"].json_schema_extra == {CHANNEL_ROLE_KEY: MOVABLE_ROLE}


# --- channel_roles: reading the declaration ---------------------------------


def test_channel_roles_reports_every_declared_field_with_a_readable_path() -> None:
    """Declaration order, depth-first, with nested paths spelled legibly enough
    to name in an operator-facing message.
    """
    assert channel_roles(_Params) == [
        ("correctors", MOVABLE_ROLE),
        ("monitors", READABLE_ROLE),
        ("reference", READABLE_ROLE),
        ("axes[].setpoint", MOVABLE_ROLE),
    ]


def test_channel_roles_is_empty_for_a_model_that_declares_nothing() -> None:
    """The signal the load gate reads: a plan claiming to write while declaring
    no movable channel comes back with nothing to point at.
    """
    assert channel_roles(_PlainParams) == []


def test_channel_roles_ignores_an_unrecognized_role_value() -> None:
    """A misspelled role is no declaration at all.

    Failing this way costs the plan its declared channels (and a `writes: true`
    plan its load gate), which is loud; the alternative -- treating an unknown
    string as a role -- would let a typo through as if it were a contract.
    """

    class _Typo(BaseModel):
        channels: list[str] = Field(..., json_schema_extra={CHANNEL_ROLE_KEY: "moveable"})

    assert channel_roles(_Typo) == []


def test_channel_roles_reaches_roles_through_optional_and_mapping_fields() -> None:
    """Nested models are found however they are held, and the path says whether
    a container was crossed to reach them.
    """

    class _Group(BaseModel):
        probe: ReadableChannel

    class _Held(BaseModel):
        one: _Group | None = None
        many: dict[str, _Group] = {}

    assert channel_roles(_Held) == [
        ("one.probe", READABLE_ROLE),
        ("many[].probe", READABLE_ROLE),
    ]


def test_optional_singles_declare_their_role_and_skip_an_unset_value() -> None:
    """The optional singles carry the same role a required single carries, and
    a ``None`` value simply contributes no channel -- the shape a plan with an
    optional guard channel (set it and the run is gated, leave it unset and it
    is not) depends on.
    """

    class _Guarded(BaseModel):
        gate: OptionalReadableChannel = Field(default=None, title="Gate")
        trim: OptionalMovableChannel = Field(default=None, title="Trim")

    assert channel_roles(_Guarded) == [
        ("gate", READABLE_ROLE),
        ("trim", MOVABLE_ROLE),
    ]
    schema = _Guarded.model_json_schema()
    assert schema["properties"]["gate"][CHANNEL_ROLE_KEY] == READABLE_ROLE
    assert schema["properties"]["gate"]["title"] == "Gate"
    assert collect_channels(_Guarded, {"gate": "SR:DCCT", "trim": None}, READABLE_ROLE) == [
        "SR:DCCT"
    ]
    assert collect_channels(_Guarded, {"gate": None, "trim": None}, MOVABLE_ROLE) == []
    assert collect_channels(_Guarded, {}, READABLE_ROLE) == []


def test_a_union_spelled_at_the_field_site_is_no_declaration() -> None:
    """Why the optional singles exist as their own types.

    Pydantic lifts an ``Annotated`` declaration into the field's metadata only
    when it is the outermost annotation, so ``ReadableChannel | None`` reads
    back as *no role at all* -- silently, which for a channel field means no
    mock, no pre-enqueue check and no device at run time. This pin is the
    tripwire: if pydantic ever starts lifting union-inner metadata, the
    optional types stop being load-bearing and this test says so.
    """

    class _UnionSpelling(BaseModel):
        gate: ReadableChannel | None = Field(default=None)

    assert channel_roles(_UnionSpelling) == []


def test_channel_roles_terminates_on_a_self_referential_model() -> None:
    """A model that can contain itself must not walk forever."""

    class _Node(BaseModel):
        channel: MovableChannel
        children: list[_Node] = []

    assert channel_roles(_Node) == [("channel", MOVABLE_ROLE)]


# --- collect_channels: reading one plan's parameters -------------------------


_PARAMS_ARGS = {
    "correctors": ["COR1", "COR2"],
    "monitors": ["BPM1"],
    "reference": "BPM:REF",
    "axes": [
        {"setpoint": "COR3", "start": 0.0},
        {"setpoint": "COR4", "start": 1.0},
    ],
}


def test_collect_channels_gathers_one_role_across_nesting_levels() -> None:
    """Top-level and nested declarations answer the same call, so no consumer
    has to know a plan's parameter shape.
    """
    assert collect_channels(_Params, _PARAMS_ARGS, MOVABLE_ROLE) == [
        "COR1",
        "COR2",
        "COR3",
        "COR4",
    ]
    assert collect_channels(_Params, _PARAMS_ARGS, READABLE_ROLE) == ["BPM1", "BPM:REF"]


def test_collect_channels_accepts_a_validated_model_as_well_as_a_dict() -> None:
    """The enqueue path holds a dict and a plan holds its validated parameters;
    both are the same question.
    """
    params = _Params.model_validate(_PARAMS_ARGS)
    assert collect_channels(_Params, params, MOVABLE_ROLE) == collect_channels(
        _Params, _PARAMS_ARGS, MOVABLE_ROLE
    )


def test_collect_channels_only_takes_the_nearest_enclosing_field() -> None:
    """A sibling value under a role-declared field's *parent* is not a channel.

    Given ``axes: [{"setpoint": "COR3", "mode": "fast"}]`` only ``setpoint`` is
    declared, so ``"fast"`` must not be collected: a consumer that swept up
    every string under ``axes`` would refuse a perfectly good enqueue over a
    mode string, and nothing the caller does to the channel name fixes that.
    """
    args = {**_PARAMS_ARGS, "axes": [{"setpoint": "COR3", "start": 0.0, "mode": "fast"}]}
    collected = collect_channels(_Params, args, MOVABLE_ROLE)
    assert collected == ["COR1", "COR2", "COR3"]
    assert "fast" not in collected


def test_collect_channels_matches_field_names_exactly() -> None:
    """Exact key equality, so a near-miss key contributes nothing.

    This is what replaced the old singular/plural guess: ``corrector`` is not
    ``correctors``, and a plan that wants both declared must declare both.
    """
    args = {
        "corrector": "COR9",
        "correctors": ["COR1"],
        "monitors": [],
        "reference": "",
        "axes": [],
    }
    assert collect_channels(_Params, args, MOVABLE_ROLE) == ["COR1"]


def test_collect_channels_walks_the_container_shapes_params_arrive_in() -> None:
    """Lists, tuples and sets all carry the enclosing field name through
    unchanged -- enqueued parameters have been round-tripped through JSON and
    through pydantic, and both spellings reach this function.
    """
    args = {"correctors": ("COR1", ["COR2"]), "monitors": [], "reference": "", "axes": []}
    assert collect_channels(_Params, args, MOVABLE_ROLE) == ["COR1", "COR2"]


def test_collect_channels_deduplicates_keeping_first_appearance() -> None:
    """A channel named twice is one channel, and the order plans declare their
    channels in is the order runs report them.
    """
    args = {**_PARAMS_ARGS, "axes": [{"setpoint": "COR1", "start": 0.0}]}
    assert collect_channels(_Params, args, MOVABLE_ROLE) == ["COR1", "COR2"]


def test_collect_channels_returns_nothing_rather_than_raising_on_absent_fields() -> None:
    """Callers use this to *describe* an enqueue; a description must never be
    the thing that refuses one.
    """
    assert collect_channels(_Params, {}, MOVABLE_ROLE) == []
    assert collect_channels(_PlainParams, _PARAMS_ARGS, MOVABLE_ROLE) == []


# --- declared_channels: the worker-side bound --------------------------------


def test_declared_channels_is_every_role_movable_first() -> None:
    """The union both worker sides bound their device mapping by.

    Movable first because that is the order the roles are declared in, and the
    bound is the concatenation of the two role reads -- nothing a plan supplies
    under a declared field is left out of the devices it may resolve.
    """
    assert declared_channels(_Params, _PARAMS_ARGS) == [
        "COR1",
        "COR2",
        "COR3",
        "COR4",
        "BPM1",
        "BPM:REF",
    ]


def test_declared_channels_reports_a_dual_role_channel_once() -> None:
    """A channel the plan both drives and records is one device, not two.

    The device mapping is keyed by name, so a duplicate could never have
    changed the bound -- but a caller reading this list to *describe* the bound
    would otherwise report a channel twice.
    """
    args = {**_PARAMS_ARGS, "monitors": ["COR1", "BPM1"]}
    assert declared_channels(_Params, args) == ["COR1", "COR2", "COR3", "COR4", "BPM1", "BPM:REF"]


def test_declared_channels_bounds_an_undeclaring_model_to_nothing() -> None:
    """A plan file with no role-declaring ``PARAMS`` claims no devices at all --
    the direction this bound has to fail.
    """
    assert declared_channels(_PlainParams, _PARAMS_ARGS) == []
    assert declared_channels(_Params, {}) == []


# --- scan_metadata: the one translation --------------------------------------


def test_scan_metadata_emits_the_standard_run_metadata_keys() -> None:
    """The exact dict a plan hands its run decorator.

    Spelled out literally here on purpose: this is the contract with analysis
    and data-access tooling outside OSPREY, so a change to it is a change other
    people's code sees, and it should cost a deliberate test edit.
    """
    assert scan_metadata(movable=["COR1"], readable=["BPM1", "BPM2"], points=21) == {
        "motors": ["COR1"],
        "detectors": ["BPM1", "BPM2"],
        "num_points": 21,
    }


def test_scan_metadata_accepts_any_sequence_and_returns_plain_lists() -> None:
    """The result is handed to a document serializer, so it holds plain types."""
    md = scan_metadata(movable=("COR1",), readable=("BPM1",), points=3)
    assert md["motors"] == ["COR1"]
    assert md["detectors"] == ["BPM1"]
    assert all(isinstance(value, list) for value in (md["motors"], md["detectors"]))
    assert isinstance(md["num_points"], int)


def test_scan_metadata_omits_interval_and_dimensionality_hints() -> None:
    """Neither is derivable truthfully for a plan whose points are several
    sweeps rather than one traversal, and an absent hint degrades better than a
    fabricated one.
    """
    md = scan_metadata(movable=["COR1"], readable=["BPM1"], points=21)
    assert set(md) == {"motors", "detectors", "num_points"}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"movable": [], "readable": ["BPM1"], "points": 3}, "movable"),
        ({"movable": ["COR1"], "readable": [], "points": 3}, "readable"),
        ({"movable": ["COR1"], "readable": ["BPM1"], "points": 0}, "point"),
    ],
)
def test_scan_metadata_rejects_a_run_it_cannot_describe(
    kwargs: dict[str, Any], expected: str
) -> None:
    """Moving nothing, reading nothing, or producing no points is an authoring
    mistake; the message names the gap in the vocabulary the author writes in.
    """
    with pytest.raises(ValueError, match=expected):
        scan_metadata(**kwargs)


def test_run_metadata_keys_are_spelled_only_inside_scan_metadata() -> None:
    """The vocabulary rule, enforced rather than asserted in prose.

    Every other line of this module -- docstrings included -- speaks capability
    language, so the run's own key names exist in exactly one function and a
    consumer cannot copy them out of a docstring by mistake.
    """
    source = Path(plan_fields.__file__).read_text(encoding="utf-8")
    before, marker, rest = source.partition("def scan_metadata(")
    body, next_def, after = rest.partition("\ndef ")
    assert marker and next_def, "scan_metadata is no longer a top-level function in this module"

    for key in ("motors", "detectors", "num_points"):
        assert key in body, f"{key!r} is no longer emitted by scan_metadata"
        assert key not in before, f"{key!r} appears in this module before scan_metadata"
        assert key not in after, f"{key!r} appears in this module after scan_metadata"


# --- resolve_column: finding a channel's readings ----------------------------


def test_resolve_column_prefers_an_exact_match_over_a_prefixed_one() -> None:
    """Both spellings can be present at once; the bare name is the channel's own
    column and wins wherever it sits in the row.
    """
    assert resolve_column("motor1", ["motor1-readback", "motor1", "det-value"]) == "motor1"


def test_resolve_column_falls_back_to_the_prefixed_spelling() -> None:
    """The other spelling a device produces: one child signal per column."""
    assert (
        resolve_column("motor1", ["seq_num", "motor1-readback", "det-value"]) == "motor1-readback"
    )
    assert resolve_column("det", ["det-value", "det-other"]) == "det-value"


def test_resolve_column_does_not_match_a_channel_whose_name_merely_starts_the_same() -> None:
    """The separator is part of the rule: ``COR10`` is not ``COR1``'s column."""
    assert resolve_column("COR1", ["COR10", "COR10-readback"]) is None
    assert resolve_column("COR1", ["COR10-readback", "COR1-readback"]) == "COR1-readback"


def test_resolve_column_returns_none_when_the_channel_has_no_column() -> None:
    """A channel that was never read is "no reading", not an error -- every
    caller draws a gap rather than failing the figure.
    """
    assert resolve_column("motor1", []) is None
    assert resolve_column("motor1", ["det-value"]) is None


def test_resolve_column_accepts_a_data_row_directly() -> None:
    """Callers hold rows, not column lists; iterating a row yields its keys."""
    row = {"seq_num": 1, "motor1-readback": 0.5, "det-value": 7}
    assert resolve_column("motor1", row) == "motor1-readback"


# --- panel tolerance --------------------------------------------------------


def test_form_generator_never_reads_the_role_key() -> None:
    """The panel's schema-driven form generator ignores the role extra.

    It dispatches on a fixed set of schema keys and never mentions
    ``x-channel-role``, so the declared role is inert there -- combined with the
    additivity test above, that is what makes adding roles to a shipped plan's
    parameters a no-op for the operator's form.
    """
    source = _SCHEMA_FORM_JS.read_text(encoding="utf-8")
    assert "x-widget" in source, (
        f"{_SCHEMA_FORM_JS} is not the schema-driven form generator this test "
        "means to check -- it no longer reads any schema extras"
    )
    assert CHANNEL_ROLE_KEY not in source


# --- import cleanliness -----------------------------------------------------


def test_importing_plan_fields_stays_import_clean() -> None:
    """`plan_fields` is pydantic-only: every consumer on the bridge's
    import-clean side imports it, so a stray bluesky/ophyd/tiled import here
    would breach that boundary for all of them.

    Runs in a fresh subprocess for the same reason the app's gate does: the dev
    venv has all four packages installed and sibling tests import them, so an
    in-process check would answer test-ordering, not what this module imports.
    """
    assert _BRIDGE_ONLY_MODULES, "the import-clean module set is empty -- nothing is being checked"

    code = (
        "import osprey.services.bluesky_bridge.plan_fields, sys; "
        f"leaked = sorted(set({sorted(_BRIDGE_ONLY_MODULES)!r}) & set(sys.modules)); "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        "importing osprey.services.bluesky_bridge.plan_fields leaked a top-level "
        f"import of a bridge-only module (child exit {result.returncode}):\n{result.stderr}"
    )

"""The partition of a model's variables, and the verbs of the model surface.

A model variable is *served* when the channel manifest serves its name as an
address, and *model-only* otherwise. The rule is by name alone: whether a
variable is writable decides what a client may do with it on its side of the
partition, never which side it is on. These tests pin that rule, the
``info`` / ``get`` / ``diff`` / ``status`` verbs built on it, and the ``set`` /
``reset`` verbs that write model-only variables behind a token, against the
real serving database, never a mock of it.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, StrVariable, Variable

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    RECORD_TYPE_ANALOG,
)
from osprey.services.virtual_accelerator.serving import model_surface
from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError
from osprey.services.virtual_accelerator.serving.model_stub import NullModel
from osprey.services.virtual_accelerator.serving.model_surface import (
    SURFACE_MODEL_ONLY,
    SURFACE_SERVED,
    ModelSurface,
    VariablePartition,
    partition_variables,
)
from osprey.services.virtual_accelerator.serving.pvdb import (
    ServingRecords,
    build_serving_pvdb,
)
from osprey.services.virtual_accelerator.serving.write_path import (
    RUNNER_CONFIG_POLICY,
    STUCK_SETPOINTS_VARIABLE,
    SetpointRoutedModel,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 90

RING = "ZZMS"

# Served addresses: a writable magnet setpoint and a read-only BPM reading.
MAG_SP = f"{RING}:MAG:HCM:01:CURRENT:SP"
MAG_RB = f"{RING}:MAG:HCM:01:CURRENT:RB"
BPM_X = f"{RING}:DIAG:BPM:01:POSITION:X"

# Names the model declares and the manifest does not serve.
STUCK = "stuck_setpoints"
TUNE_X = "tune_x"


def _channel(address: str, *, subfield: str, system: str, family: str, field: str) -> dict:
    """One synthetic manifest channel, in the shape ``build_manifest()`` emits."""
    return {
        "address": address,
        "ring": RING,
        "system": system,
        "family": family,
        "device": "01",
        "field": field,
        "subfield": subfield,
        "partition": PARTITION_PYAT_COUPLED,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    }


CHANNELS = [
    _channel(MAG_SP, subfield="SP", system="MAG", family="HCM", field="CURRENT"),
    _channel(MAG_RB, subfield="RB", system="MAG", family="HCM", field="CURRENT"),
    _channel(BPM_X, subfield="X", system="DIAG", family="BPM", field="POSITION"),
]


def _scalar(name: str, *, read_only: bool) -> ScalarVariable:
    return ScalarVariable(
        name=name,
        default_value=0.0,
        default_validation_config="none",
        read_only=read_only,
    )


class DeclaringModel(LUMEModel):
    """A model that declares exactly the variables it is given, in that order."""

    def __init__(self, variables: list[Variable]) -> None:
        self._vars: dict[str, Variable] = {v.name: v for v in variables}

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._vars

    def _get(self, names: list[str]) -> dict[str, Any]:  # noqa: ARG002 - the model surface fixes this signature  # pragma: no cover - unused
        raise AssertionError("partitioning must never read a value")

    def _set(self, values: dict[str, Any]) -> None:  # noqa: ARG002 - the model surface fixes this signature  # pragma: no cover - unused
        raise AssertionError("partitioning must never write a value")

    def reset(self) -> None:  # pragma: no cover - unused
        raise AssertionError("partitioning must never reset the model")


@pytest.fixture()
def records() -> ServingRecords:
    """The real serving database -- never a mock of it."""
    return build_serving_pvdb(CHANNELS)


def _recording_model() -> DeclaringModel:
    """Two served addresses and one name the manifest does not serve."""
    return DeclaringModel(
        [
            _scalar(MAG_SP, read_only=False),
            _scalar(BPM_X, read_only=True),
            StrVariable(name=STUCK, default_value=""),
        ]
    )


class TestPartitionOfNullModel:
    def test_a_model_with_no_variables_partitions_into_two_empty_sides(
        self, records: ServingRecords
    ) -> None:
        """The manifest still serves its addresses; none of them is a model
        variable, so neither side has anything in it."""
        partition = partition_variables(NullModel(), records)

        assert partition.served == {}
        assert partition.model_only == {}

    def test_empty_records_leave_both_sides_empty(self) -> None:
        partition = partition_variables(NullModel(), ServingRecords())

        assert partition == VariablePartition(served={}, model_only={})


class TestPartitionByName:
    def test_served_addresses_and_an_extra_name_split_exactly(
        self, records: ServingRecords
    ) -> None:
        model = _recording_model()

        partition = partition_variables(model, records)

        assert set(partition.served) == {MAG_SP, BPM_X}
        assert set(partition.model_only) == {STUCK}

    def test_each_side_carries_the_models_own_variable_objects(
        self, records: ServingRecords
    ) -> None:
        """The partition classifies; it never copies or rebuilds a variable,
        so a verb reporting ``value_range`` or ``unit`` reports the model's."""
        model = _recording_model()
        declared = model.supported_variables

        partition = partition_variables(model, records)

        for name, variable in {**partition.served, **partition.model_only}.items():
            assert variable is declared[name]

    def test_every_variable_lands_on_exactly_one_side(self, records: ServingRecords) -> None:
        model = _recording_model()

        partition = partition_variables(model, records)

        assert set(partition.served).isdisjoint(partition.model_only)
        assert set(partition.served) | set(partition.model_only) == set(model.supported_variables)

    def test_a_served_address_the_model_does_not_declare_is_on_neither_side(
        self, records: ServingRecords
    ) -> None:
        """The partition is of the model's variables: a Channel Access-only
        address (a magnet's ``:RB``) belongs to no side."""
        assert MAG_RB in records.all

        partition = partition_variables(_recording_model(), records)

        assert MAG_RB not in partition.served
        assert MAG_RB not in partition.model_only

    def test_read_only_never_decides_the_side(self, records: ServingRecords) -> None:
        """All four combinations of served/unserved and writable/read-only:
        each lands on the side its name alone selects."""
        model = DeclaringModel(
            [
                _scalar(MAG_SP, read_only=False),  # served, writable
                _scalar(BPM_X, read_only=True),  # served, read-only
                _scalar(TUNE_X, read_only=True),  # model-only, read-only
                StrVariable(name=STUCK, default_value=""),  # model-only, writable
            ]
        )

        partition = partition_variables(model, records)

        assert set(partition.served) == {MAG_SP, BPM_X}
        assert set(partition.model_only) == {TUNE_X, STUCK}
        assert {v.read_only for v in partition.served.values()} == {True, False}
        assert {v.read_only for v in partition.model_only.values()} == {True, False}

    def test_each_side_keeps_the_models_declaration_order(self, records: ServingRecords) -> None:
        """A roster listed from either side is stable across boots."""
        model = DeclaringModel(
            [
                StrVariable(name=STUCK, default_value=""),
                _scalar(BPM_X, read_only=True),
                _scalar(TUNE_X, read_only=True),
                _scalar(MAG_SP, read_only=False),
            ]
        )

        partition = partition_variables(model, records)

        assert list(partition.served) == [BPM_X, MAG_SP]
        assert list(partition.model_only) == [STUCK, TUNE_X]


class TestPartitionShape:
    def test_the_partition_is_frozen(self, records: ServingRecords) -> None:
        partition = partition_variables(_recording_model(), records)

        with pytest.raises(dataclasses.FrozenInstanceError):
            partition.served = {}  # type: ignore[misc]

    def test_the_partitions_dicts_are_its_own(self, records: ServingRecords) -> None:
        """Editing a side never edits the model's declared namespace."""
        model = _recording_model()
        partition = partition_variables(model, records)

        partition.model_only.clear()

        assert STUCK in model.supported_variables

    def test_the_surface_names_info_reports(self) -> None:
        assert SURFACE_SERVED == "served"
        assert SURFACE_MODEL_ONLY == "model-only"


# ---------------------------------------------------------------------------
# The read verbs
# ---------------------------------------------------------------------------

BACKEND = "pyat"
LATTICE_SOURCE = "example-ring (built in)"
INSTANCE = "va-test"
ENDPOINT = "localhost:5075"

# What the model holds -- the un-faulted truth -- per declared name.
TRUTH = {MAG_SP: 1.5, BPM_X: 0.25, TUNE_X: 0.31, STUCK: ""}

# What the driver serves for each served address: a fault has pulled both
# away from the model's truth.
SERVED = {MAG_SP: 1.625, BPM_X: 0.5}


class HoldingModel(DeclaringModel):
    """A model that holds a value per declared name and records every read."""

    def __init__(self, variables: list[Variable], values: dict[str, Any]) -> None:
        super().__init__(variables)
        self.values = dict(values)
        self.reads: list[list[str]] = []

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.reads.append(list(names))
        return {name: self.values[name] for name in names}


def _holding_model() -> HoldingModel:
    """Both served addresses, a read-only model-only reading and a writable
    model-only name, each with a value and the first two with units/ranges."""
    return HoldingModel(
        [
            ScalarVariable(
                name=MAG_SP,
                default_value=0.0,
                value_range=(-10.0, 10.0),
                unit="A",
                default_validation_config="none",
            ),
            ScalarVariable(
                name=BPM_X,
                default_value=0.0,
                unit="mm",
                read_only=True,
                default_validation_config="none",
            ),
            _scalar(TUNE_X, read_only=True),
            StrVariable(name=STUCK, default_value=""),
        ],
        TRUTH,
    )


class FakeClock:
    """A clock the test advances by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _surface(model: LUMEModel, records: ServingRecords, **overrides: Any) -> ModelSurface:
    kwargs: dict[str, Any] = {
        "backend_name": BACKEND,
        "lattice_source": LATTICE_SOURCE,
        "instance": INSTANCE,
        "endpoint": ENDPOINT,
        "clock": FakeClock(),
    }
    kwargs.update(overrides)
    return ModelSurface(model, partition_variables(model, records), records, **kwargs)


def _wrapped_null_model() -> SetpointRoutedModel:
    """What the serving runner serves when no physics backend is loaded."""
    return SetpointRoutedModel(NullModel(), on_setpoint=None, routed=frozenset())


def _json_plain(result: Any) -> bool:
    """A result survives a JSON round trip unchanged: no tuple, no numpy."""
    return json.loads(json.dumps(result)) == result


class TestInfo:
    def test_info_over_the_wrapped_null_model_lists_exactly_the_stuck_set(
        self, records: ServingRecords
    ) -> None:
        """With no backend, the only variable is the one the wrapper owns --
        and the manifest serves no address by that name."""
        info = _surface(_wrapped_null_model(), records).info()

        assert [v["name"] for v in info["variables"]] == [STUCK]
        assert info["variables"][0]["surface"] == SURFACE_MODEL_ONLY

    def test_info_reports_the_backend_and_lattice_source(self, records: ServingRecords) -> None:
        info = _surface(_holding_model(), records).info()

        assert info["backend"] == BACKEND
        assert info["lattice_source"] == LATTICE_SOURCE

    def test_info_reports_the_surface_of_each_variable_by_side(
        self, records: ServingRecords
    ) -> None:
        info = _surface(_holding_model(), records).info()

        surfaces = {v["name"]: v["surface"] for v in info["variables"]}
        assert surfaces == {
            MAG_SP: SURFACE_SERVED,
            BPM_X: SURFACE_SERVED,
            TUNE_X: SURFACE_MODEL_ONLY,
            STUCK: SURFACE_MODEL_ONLY,
        }

    def test_info_reports_read_only_independently_of_the_side(
        self, records: ServingRecords
    ) -> None:
        info = _surface(_holding_model(), records).info()

        read_only = {v["name"]: v["read_only"] for v in info["variables"]}
        assert read_only == {MAG_SP: False, BPM_X: True, TUNE_X: True, STUCK: False}

    def test_info_reports_the_models_unit_and_value_range(self, records: ServingRecords) -> None:
        """A variable kind without a unit or a range reports ``None`` for it."""
        info = _surface(_holding_model(), records).info()

        by_name = {v["name"]: v for v in info["variables"]}
        assert by_name[MAG_SP]["unit"] == "A"
        assert by_name[MAG_SP]["value_range"] == [-10.0, 10.0]
        assert by_name[BPM_X]["unit"] == "mm"
        assert by_name[BPM_X]["value_range"] is None
        assert by_name[STUCK]["unit"] is None
        assert by_name[STUCK]["value_range"] is None

    def test_info_entries_carry_exactly_the_documented_fields(
        self, records: ServingRecords
    ) -> None:
        info = _surface(_holding_model(), records).info()

        assert set(info) == {"backend", "lattice_source", "variables"}
        for entry in info["variables"]:
            assert set(entry) == {"name", "unit", "value_range", "read_only", "surface"}

    def test_info_lists_served_then_model_only_each_in_declaration_order(
        self, records: ServingRecords
    ) -> None:
        info = _surface(_holding_model(), records).info()

        assert [v["name"] for v in info["variables"]] == [MAG_SP, BPM_X, TUNE_X, STUCK]

    def test_info_never_reads_a_value(self, records: ServingRecords) -> None:
        model = _holding_model()

        _surface(model, records).info()

        assert model.reads == []

    def test_info_is_plain_json(self, records: ServingRecords) -> None:
        assert _json_plain(_surface(_holding_model(), records).info())


class TestGet:
    def test_get_returns_what_the_model_holds_on_either_side(self, records: ServingRecords) -> None:
        """A served address answers the model's truth, not the served value."""
        surface = _surface(_holding_model(), records)

        assert surface.get([MAG_SP, TUNE_X]) == {MAG_SP: TRUTH[MAG_SP], TUNE_X: TRUTH[TUNE_X]}

    def test_get_of_one_name_still_answers_a_dict(self, records: ServingRecords) -> None:
        assert _surface(_holding_model(), records).get((BPM_X,)) == {BPM_X: TRUTH[BPM_X]}

    def test_get_of_an_unknown_name_refuses_the_whole_call_without_reading(
        self, records: ServingRecords
    ) -> None:
        model = _holding_model()
        surface = _surface(model, records)

        with pytest.raises(ModelRpcError, match="no_such_variable"):
            surface.get([MAG_SP, "no_such_variable"])

        assert model.reads == []

    def test_get_refusal_names_every_unknown_name_sorted(self, records: ServingRecords) -> None:
        surface = _surface(_holding_model(), records)

        with pytest.raises(ModelRpcError) as refused:
            surface.get(["zz_unknown", MAG_SP, "aa_unknown"])

        message = str(refused.value)
        assert message.index("aa_unknown") < message.index("zz_unknown")
        assert MAG_SP not in message

    def test_get_refuses_a_served_address_the_model_does_not_declare(
        self, records: ServingRecords
    ) -> None:
        """A Channel Access-only address is not a model variable to ``get``."""
        model = _holding_model()

        with pytest.raises(ModelRpcError, match=MAG_RB):
            _surface(model, records).get([MAG_RB])

        assert model.reads == []

    def test_get_reads_the_model_once_per_call(self, records: ServingRecords) -> None:
        model = _holding_model()

        _surface(model, records).get([MAG_SP, BPM_X, STUCK])

        assert model.reads == [[MAG_SP, BPM_X, STUCK]]

    def test_get_of_the_stuck_set_over_the_wrapped_null_model(
        self, records: ServingRecords
    ) -> None:
        assert _surface(_wrapped_null_model(), records).get([STUCK]) == {STUCK: ""}


class TestDiff:
    def test_diff_pairs_the_served_value_with_the_models_truth(
        self, records: ServingRecords
    ) -> None:
        surface = _surface(_holding_model(), records)

        assert surface.diff(SERVED.__getitem__) == {
            MAG_SP: {"served": SERVED[MAG_SP], "truth": TRUTH[MAG_SP]},
            BPM_X: {"served": SERVED[BPM_X], "truth": TRUTH[BPM_X]},
        }

    def test_diff_asks_the_driver_only_for_served_names(self, records: ServingRecords) -> None:
        asked: list[str] = []

        def get_param(name: str) -> Any:
            asked.append(name)
            return SERVED[name]

        _surface(_holding_model(), records).diff(get_param)

        assert sorted(asked) == sorted([MAG_SP, BPM_X])

    def test_diff_reads_the_truth_in_one_batch(self, records: ServingRecords) -> None:
        model = _holding_model()

        _surface(model, records).diff(SERVED.__getitem__)

        assert model.reads == [[MAG_SP, BPM_X]]

    def test_diff_over_the_wrapped_null_model_is_empty(self, records: ServingRecords) -> None:
        """Nothing is served, so there is nothing to compare and nothing to ask."""

        def get_param(name: str) -> Any:  # pragma: no cover - must not be called
            raise AssertionError(f"asked the driver for {name!r}")

        assert _surface(_wrapped_null_model(), records).diff(get_param) == {}

    def test_diff_is_plain_json(self, records: ServingRecords) -> None:
        assert _json_plain(_surface(_holding_model(), records).diff(SERVED.__getitem__))


class TestStatus:
    def test_status_before_the_runner_has_recorded_anything(self, records: ServingRecords) -> None:
        status = _surface(_holding_model(), records).status()

        assert status == {
            "backend": BACKEND,
            "lattice_source": LATTICE_SOURCE,
            "instance": INSTANCE,
            "endpoint": ENDPOINT,
            "update_rate": RUNNER_CONFIG_POLICY["update_rate"],
            "last_cycle_ms": None,
            "queue_depth": 0,
            "uptime_s": 0.0,
            "last_refused_write": None,
        }

    def test_status_uptime_follows_the_injected_clock(self, records: ServingRecords) -> None:
        clock = FakeClock(start=50.0)
        surface = _surface(_holding_model(), records, clock=clock)

        clock.now = 62.5

        assert surface.status()["uptime_s"] == 12.5

    def test_status_reports_what_the_runner_recorded_latest_first(
        self, records: ServingRecords
    ) -> None:
        surface = _surface(_holding_model(), records)

        surface.record_cycle(3.0)
        surface.record_cycle(12.5)
        surface.record_queue_depth(4)
        surface.record_queue_depth(2)
        surface.record_refusal("first refusal")
        surface.record_refusal(f"{MAG_SP}: value out of range")
        status = surface.status()

        assert status["last_cycle_ms"] == 12.5
        assert status["queue_depth"] == 2
        assert status["last_refused_write"] == f"{MAG_SP}: value out of range"

    def test_status_reports_an_update_rate_the_runner_passes(self, records: ServingRecords) -> None:
        surface = _surface(_holding_model(), records, update_rate=5.0)

        assert surface.status()["update_rate"] == 5.0

    def test_status_never_reads_the_model(self, records: ServingRecords) -> None:
        model = _holding_model()

        _surface(model, records).status()

        assert model.reads == []

    def test_status_is_plain_json(self, records: ServingRecords) -> None:
        surface = _surface(_holding_model(), records)
        surface.record_cycle(1)
        surface.record_queue_depth(1)
        surface.record_refusal("refused")

        assert _json_plain(surface.status())


# ---------------------------------------------------------------------------
# The write verbs
# ---------------------------------------------------------------------------

TOKEN = "model-write-token-for-tests"

# Model-only writables with a boot seed and a range lume enforces.
CAL = "HCM01.cal_factor"
OFFSET = "BPM01.offset_x"
CAL_SEED = 1.0
OFFSET_SEED = 0.0


class WritableModel(HoldingModel):
    """A holding model that applies and records every write reaching ``_set``.

    ``LUMEModel.set`` validates names and values before it calls ``_set``, so
    an empty :attr:`sets` means nothing was written. ``reset`` is inherited
    from :class:`DeclaringModel` and fails the test if anything calls it.
    """

    def __init__(self, variables: list[Variable], values: dict[str, Any]) -> None:
        super().__init__(variables, values)
        self.sets: list[dict[str, Any]] = []

    def _set(self, values: dict[str, Any]) -> None:
        self.sets.append(dict(values))
        self.values.update(values)


def _fault(name: str, seed: float, value_range: tuple[float, float]) -> ScalarVariable:
    return ScalarVariable(
        name=name,
        default_value=seed,
        value_range=value_range,
        default_validation_config="error",
    )


def _fault_variables() -> list[Variable]:
    """Both served addresses, a read-only model-only reading and two faults."""
    return [
        ScalarVariable(
            name=MAG_SP,
            default_value=0.0,
            value_range=(-10.0, 10.0),
            default_validation_config="error",
        ),
        _scalar(BPM_X, read_only=True),
        _scalar(TUNE_X, read_only=True),
        _fault(CAL, CAL_SEED, (0.5, 1.5)),
        _fault(OFFSET, OFFSET_SEED, (-1e-3, 1e-3)),
    ]


# Every value starts at its seed, except the served setpoint and the
# read-only reading, whose values differ from their declared defaults.
BOOT_VALUES = {MAG_SP: 1.5, BPM_X: 0.25, TUNE_X: 0.31, CAL: CAL_SEED, OFFSET: OFFSET_SEED}


def _writable_model() -> WritableModel:
    """The fault model plus a stuck set it declares itself, unwrapped."""
    return WritableModel(
        [*_fault_variables(), StrVariable(name=STUCK, default_value="")],
        {**BOOT_VALUES, STUCK: ""},
    )


class RefreshRecorder:
    """Stands in for the physics bridge's ``refresh``."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, changed: Any) -> None:
        self.calls.append(list(changed))


def _writer(
    model: LUMEModel,
    records: ServingRecords,
    refresh: RefreshRecorder | None = None,
    **overrides: Any,
) -> ModelSurface:
    """A surface that accepts :data:`TOKEN` and refreshes through ``refresh``."""
    overrides.setdefault("model_write_token", TOKEN)
    if refresh is not None:
        overrides["refresh"] = refresh
    return _surface(model, records, **overrides)


REFUSED_SETS = [
    pytest.param(None, TOKEN, {CAL: 1.1}, "model writes are disabled", id="writes-disabled"),
    pytest.param("", "", {CAL: 1.1}, "model writes are disabled", id="empty-token-disables"),
    pytest.param(TOKEN, "", {CAL: 1.1}, "token", id="missing-token"),
    pytest.param(TOKEN, None, {CAL: 1.1}, "token", id="no-token-at-all"),
    pytest.param(TOKEN, "wrong-guess", {CAL: 1.1}, "token", id="wrong-token"),
    pytest.param(TOKEN, TOKEN, {MAG_SP: 2.0}, MAG_SP, id="served-setpoint"),
    pytest.param(TOKEN, TOKEN, {BPM_X: 0.1}, BPM_X, id="served-reading"),
    pytest.param(TOKEN, TOKEN, {MAG_RB: 1.0}, MAG_RB, id="served-address-the-model-lacks"),
    pytest.param(TOKEN, TOKEN, {TUNE_X: 0.3}, "read-only", id="read-only"),
    pytest.param(TOKEN, TOKEN, {"no_such": 1.0}, "not a model variable", id="unknown-name"),
    pytest.param(TOKEN, TOKEN, {CAL: math.nan}, "not a finite value", id="nan"),
    pytest.param(TOKEN, TOKEN, {CAL: math.inf}, "not a finite value", id="inf"),
    pytest.param(TOKEN, TOKEN, {CAL: -math.inf}, "not a finite value", id="minus-inf"),
    pytest.param(TOKEN, TOKEN, {CAL: 2.0}, "out of valid range", id="out-of-range"),
    pytest.param(
        TOKEN, TOKEN, {CAL: 1.1, OFFSET: 5.0}, "out of valid range", id="one-bad-value-in-a-batch"
    ),
    pytest.param(TOKEN, TOKEN, {CAL: 1.1, MAG_SP: 2.0}, MAG_SP, id="one-served-name-in-a-batch"),
]


class TestSetRefusals:
    @pytest.mark.parametrize(("configured", "presented", "values", "reason"), REFUSED_SETS)
    def test_a_refused_write_leaves_the_model_unwritten(
        self,
        records: ServingRecords,
        configured: str | None,
        presented: str | None,
        values: dict[str, float],
        reason: str,
    ) -> None:
        model = _writable_model()
        refresh = RefreshRecorder()
        surface = _writer(model, records, refresh, model_write_token=configured)

        with pytest.raises(ModelRpcError, match=re.escape(reason)):
            surface.set(values, presented)

        assert model.sets == []
        assert refresh.calls == []
        assert model.values == {**BOOT_VALUES, STUCK: ""}

    @pytest.mark.parametrize(("configured", "presented", "values", "reason"), REFUSED_SETS)
    def test_every_refusal_is_recorded_for_status(
        self,
        records: ServingRecords,
        configured: str | None,
        presented: str | None,
        values: dict[str, float],
        reason: str,
    ) -> None:
        surface = _writer(_writable_model(), records, model_write_token=configured)

        with pytest.raises(ModelRpcError) as refused:
            surface.set(values, presented)

        assert reason in str(refused.value)
        assert surface.status()["last_refused_write"] == str(refused.value)

    def test_the_token_is_checked_before_any_name(self, records: ServingRecords) -> None:
        surface = _writer(_writable_model(), records, model_write_token=None)

        with pytest.raises(ModelRpcError, match="^model writes are disabled$"):
            surface.set({MAG_SP: 1.0, "no_such": 1.0}, TOKEN)

    @pytest.mark.parametrize(
        ("values", "reason", "offender", "spared"),
        [
            pytest.param({TUNE_X: 0.3, MAG_SP: 1.0}, "served", MAG_SP, TUNE_X, id="served-first"),
            pytest.param(
                {"no_such": 1.0, TUNE_X: 0.3}, "read-only", TUNE_X, "no_such", id="then-read-only"
            ),
            pytest.param(
                {CAL: math.nan, "no_such": 1.0},
                "not a model variable",
                "no_such",
                CAL,
                id="then-unknown",
            ),
            pytest.param(
                {OFFSET: 5.0, CAL: math.inf}, "not a finite value", CAL, OFFSET, id="then-finite"
            ),
        ],
    )
    def test_refusals_come_in_their_documented_order(
        self,
        records: ServingRecords,
        values: dict[str, float],
        reason: str,
        offender: str,
        spared: str,
    ) -> None:
        """Served, then read-only, then unknown, then non-finite, then range:
        the earliest check a batch fails is the reason given, and it names
        only its own offenders."""
        surface = _writer(_writable_model(), records)

        with pytest.raises(ModelRpcError) as refused:
            surface.set(values, TOKEN)

        message = str(refused.value)
        assert reason in message
        assert offender in message
        assert spared not in message

    def test_a_refusal_names_every_offender_sorted(self, records: ServingRecords) -> None:
        surface = _writer(_writable_model(), records)

        with pytest.raises(ModelRpcError) as refused:
            surface.set({"zz_unknown": 1.0, CAL: 1.1, "aa_unknown": 1.0}, TOKEN)

        message = str(refused.value)
        assert message.index("aa_unknown") < message.index("zz_unknown")
        assert CAL not in message

    def test_the_range_refusal_is_lumes_own_text(self, records: ServingRecords) -> None:
        surface = _writer(_writable_model(), records)

        with pytest.raises(ModelRpcError) as refused:
            surface.set({CAL: 2.0}, TOKEN)

        assert str(refused.value).startswith(f"Validation failed for variable '{CAL}'")

    def test_a_refusal_never_echoes_either_token(self, records: ServingRecords) -> None:
        surface = _writer(_writable_model(), records)

        with pytest.raises(ModelRpcError) as refused:
            surface.set({CAL: 1.1}, "wrong-guess")

        assert "wrong-guess" not in str(refused.value)
        assert TOKEN not in str(refused.value)
        assert TOKEN not in str(surface.status())

    def test_the_token_is_compared_with_compare_digest(
        self, records: ServingRecords, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The comparison is ``hmac.compare_digest``'s, not ``==``: a
        compare_digest that says no refuses even the right token."""
        compared: list[tuple[Any, Any]] = []

        def refuse_everything(a: Any, b: Any) -> bool:
            compared.append((a, b))
            return False

        monkeypatch.setattr(model_surface.hmac, "compare_digest", refuse_everything)
        model = _writable_model()
        surface = _writer(model, records)

        with pytest.raises(ModelRpcError, match="token"):
            surface.set({CAL: 1.1}, TOKEN)

        assert len(compared) == 1
        assert model.sets == []

    def test_compare_digest_alone_decides_the_token(
        self, records: ServingRecords, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(model_surface.hmac, "compare_digest", lambda a, b: True)
        model = _writable_model()

        written = _writer(model, records).set({CAL: 1.1}, "wrong-guess")

        assert written == [CAL]

    def test_a_non_ascii_token_is_refused_not_raised(self, records: ServingRecords) -> None:
        """``compare_digest`` rejects non-ASCII text; a client typing one gets
        the token refusal, never a ``TypeError``."""
        surface = _writer(_writable_model(), records)

        with pytest.raises(ModelRpcError, match="token"):
            surface.set({CAL: 1.1}, "töken")

    def test_a_number_for_the_stuck_set_is_refused_by_lume(self, records: ServingRecords) -> None:
        """Over the wrapped null model the stuck set is the only variable, and
        it holds text: a number is refused by the variable's own check."""
        surface = _writer(_wrapped_null_model(), records)

        with pytest.raises(ModelRpcError, match=STUCK):
            surface.set({STUCK: 1.0}, TOKEN)


class TestSet:
    def test_a_correct_token_writes_once_and_refreshes_once(self, records: ServingRecords) -> None:
        model = _writable_model()
        refresh = RefreshRecorder()

        written = _writer(model, records, refresh).set({CAL: 1.1, OFFSET: 1e-4}, TOKEN)

        assert written == [CAL, OFFSET]
        assert model.sets == [{CAL: 1.1, OFFSET: 1e-4}]
        assert refresh.calls == [[CAL, OFFSET]]
        assert model.values[CAL] == 1.1
        assert model.values[OFFSET] == 1e-4

    def test_refresh_runs_after_the_model_holds_the_write(self, records: ServingRecords) -> None:
        model = _writable_model()
        seen: list[float] = []

        _writer(model, records, refresh=lambda changed: seen.append(model.values[CAL])).set(
            {CAL: 1.25}, TOKEN
        )

        assert seen == [1.25]

    def test_a_write_without_a_refresh_hook_still_writes(self, records: ServingRecords) -> None:
        """A backend with nothing derived from its variables -- the null
        model -- passes no ``refresh``; the default does nothing."""
        model = _writable_model()

        assert _writer(model, records).set({CAL: 0.9}, TOKEN) == [CAL]
        assert model.sets == [{CAL: 0.9}]

    def test_an_accepted_write_records_no_refusal(self, records: ServingRecords) -> None:
        surface = _writer(_writable_model(), records)

        surface.set({CAL: 1.1}, TOKEN)

        assert surface.status()["last_refused_write"] is None

    def test_an_empty_write_touches_nothing(self, records: ServingRecords) -> None:
        """An empty batch is the model's cue to re-solve; an empty write
        must not send one."""
        model = _writable_model()
        refresh = RefreshRecorder()

        assert _writer(model, records, refresh).set({}, TOKEN) == []
        assert model.sets == []
        assert refresh.calls == []

    def test_an_empty_write_still_needs_the_token(self, records: ServingRecords) -> None:
        with pytest.raises(ModelRpcError, match="token"):
            _writer(_writable_model(), records).set({}, "wrong-guess")

    def test_the_stuck_set_takes_text_through_the_wrapper(self, records: ServingRecords) -> None:
        """Only numbers are checked for finiteness; text goes to the variable's
        own check, and the wrapper takes a served setpoint as stuck."""
        changes: list[frozenset[str]] = []
        model = SetpointRoutedModel(
            NullModel(),
            on_setpoint=None,
            routed=frozenset(),
            known_setpoints=frozenset({MAG_SP}),
            on_stuck_change=changes.append,
        )
        refresh = RefreshRecorder()

        written = _writer(model, records, refresh).set({STUCK: MAG_SP}, TOKEN)

        assert written == [STUCK]
        assert changes == [frozenset({MAG_SP})]
        assert refresh.calls == [[STUCK]]

    def test_the_written_names_are_plain_json(self, records: ServingRecords) -> None:
        assert _json_plain(_writer(_writable_model(), records).set({CAL: 1.1}, TOKEN))


def _drift(model: WritableModel) -> None:
    """Move both faults off their seeds without going through the surface."""
    model.values.update({CAL: 1.2, OFFSET: 2e-4})


class TestReset:
    def test_reset_restores_every_drifted_seed_and_names_it(self, records: ServingRecords) -> None:
        model = _writable_model()
        refresh = RefreshRecorder()
        surface = _writer(model, records, refresh)
        _drift(model)

        reset = surface.reset(TOKEN)

        assert reset == [CAL, OFFSET]
        assert model.values[CAL] == CAL_SEED
        assert model.values[OFFSET] == OFFSET_SEED
        assert model.sets == [{CAL: CAL_SEED, OFFSET: OFFSET_SEED}]
        assert refresh.calls == [[CAL, OFFSET]]

    def test_reset_leaves_a_variable_at_its_seed_alone(self, records: ServingRecords) -> None:
        model = _writable_model()
        surface = _writer(model, records)
        model.values[CAL] = 0.8

        assert surface.reset(TOKEN) == [CAL]
        assert model.sets == [{CAL: CAL_SEED}]

    def test_reset_reads_the_current_values_once(self, records: ServingRecords) -> None:
        model = _writable_model()
        surface = _writer(model, records)
        _drift(model)

        surface.reset(TOKEN)

        assert len(model.reads) == 1
        assert set(model.reads[0]) == {CAL, OFFSET, STUCK}

    def test_reset_with_nothing_drifted_is_an_empty_list(self, records: ServingRecords) -> None:
        """Nothing to restore is not an error, and moves nothing."""
        model = _writable_model()
        refresh = RefreshRecorder()
        surface = _writer(model, records, refresh)

        assert surface.reset(TOKEN) == []
        assert model.sets == []
        assert refresh.calls == []
        assert surface.status()["last_refused_write"] is None

    def test_reset_never_moves_a_served_setpoint(self, records: ServingRecords) -> None:
        """The served setpoint holds a value that is not its declared default;
        a reset of the faults leaves it where the control system put it."""
        model = _writable_model()
        surface = _writer(model, records)
        _drift(model)

        reset = surface.reset(TOKEN)

        assert MAG_SP not in reset
        assert model.values[MAG_SP] == BOOT_VALUES[MAG_SP]
        assert all(MAG_SP not in batch for batch in model.sets)

    def test_reset_never_writes_a_read_only_variable(self, records: ServingRecords) -> None:
        model = _writable_model()
        surface = _writer(model, records)
        _drift(model)

        reset = surface.reset(TOKEN)

        assert TUNE_X not in reset
        assert BPM_X not in reset

    def test_reset_restores_the_seed_declared_at_construction(
        self, records: ServingRecords
    ) -> None:
        model = _writable_model()
        surface = _writer(model, records)
        model.supported_variables[CAL].default_value = 1.3
        model.values[CAL] = 1.3

        assert surface.reset(TOKEN) == [CAL]
        assert model.values[CAL] == CAL_SEED

    def test_a_writable_with_no_declared_default_is_never_reset(
        self, records: ServingRecords
    ) -> None:
        """With no boot seed there is nothing to restore it to."""
        unseeded = ScalarVariable(name="HCM01.cal_offset", default_validation_config="none")
        model = WritableModel([_fault(CAL, CAL_SEED, (0.5, 1.5)), unseeded], {CAL: 0.9})
        surface = _writer(model, records)
        model.values["HCM01.cal_offset"] = 0.5

        assert surface.reset(TOKEN) == [CAL]
        assert model.reads == [[CAL]]

    @pytest.mark.parametrize(
        ("configured", "presented", "reason"),
        [
            pytest.param(None, TOKEN, "model writes are disabled", id="writes-disabled"),
            pytest.param("", "", "model writes are disabled", id="empty-token-disables"),
            pytest.param(TOKEN, "", "token", id="missing-token"),
            pytest.param(TOKEN, "wrong-guess", "token", id="wrong-token"),
        ],
    )
    def test_a_refused_reset_leaves_the_model_unwritten(
        self,
        records: ServingRecords,
        configured: str | None,
        presented: str,
        reason: str,
    ) -> None:
        model = _writable_model()
        refresh = RefreshRecorder()
        surface = _writer(model, records, refresh, model_write_token=configured)
        _drift(model)

        with pytest.raises(ModelRpcError, match=re.escape(reason)) as refused:
            surface.reset(presented)

        assert model.sets == []
        assert model.reads == []
        assert refresh.calls == []
        assert surface.status()["last_refused_write"] == str(refused.value)

    def test_reset_over_the_wrapper_restores_faults_then_the_stuck_set(
        self, records: ServingRecords
    ) -> None:
        """Through the serving wrapper: the faults are written back first, the
        boot stuck set last, and neither model's ``reset`` is ever called --
        the wrapped model's would fail this test, and the wrapper's cascades
        to it."""
        events: list[tuple[str, Any]] = []
        inner = WritableModel(_fault_variables(), dict(BOOT_VALUES))
        inner_set = inner._set

        def recording_set(values: dict[str, Any]) -> None:
            events.append(("set", dict(values)))
            inner_set(values)

        inner._set = recording_set  # type: ignore[method-assign]
        model = SetpointRoutedModel(
            inner,
            on_setpoint=None,
            routed=frozenset(),
            known_setpoints=frozenset({MAG_SP}),
            on_stuck_change=lambda stuck: events.append(("stuck", stuck)),
        )
        refresh = RefreshRecorder()
        surface = _writer(model, records, refresh)
        surface.set({STUCK: MAG_SP}, TOKEN)
        _drift(inner)
        events.clear()
        refresh.calls.clear()

        reset = surface.reset(TOKEN)

        assert reset == [CAL, OFFSET, STUCK_SETPOINTS_VARIABLE]
        assert events == [
            ("set", {CAL: CAL_SEED, OFFSET: OFFSET_SEED}),
            ("stuck", frozenset()),
        ]
        assert model.get([STUCK])[STUCK] == ""
        assert refresh.calls == [[CAL, OFFSET, STUCK]]

    def test_reset_over_the_wrapped_null_model_restores_the_boot_stuck_set(
        self, records: ServingRecords
    ) -> None:
        other = f"{RING}:MAG:HCM:02:CURRENT:SP"
        model = SetpointRoutedModel(
            NullModel(),
            on_setpoint=None,
            routed=frozenset(),
            stuck_setpoints=frozenset({MAG_SP}),
            known_setpoints=frozenset({MAG_SP, other}),
        )
        surface = _writer(model, records)
        surface.set({STUCK: other}, TOKEN)

        assert surface.reset(TOKEN) == [STUCK]
        assert model.get([STUCK])[STUCK] == MAG_SP

    def test_reset_over_the_wrapped_null_model_with_nothing_drifted(
        self, records: ServingRecords
    ) -> None:
        assert _writer(_wrapped_null_model(), records).reset(TOKEN) == []

    def test_the_reset_names_are_plain_json(self, records: ServingRecords) -> None:
        model = _writable_model()
        surface = _writer(model, records)
        _drift(model)

        assert _json_plain(surface.reset(TOKEN))


def test_the_partition_module_imports_no_server_library() -> None:
    """The partition decides what a verb may touch, and the verbs run on the
    model alone; both must stay importable and testable in process, like the
    write path, with neither Channel Access server nor serving runner behind
    them."""
    tree = ast.parse(Path(model_surface.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert roots.isdisjoint({"pcaspy", "p4p", "lume_pva_apg"})
    assert not any(name.endswith("serving.runner") for name in imported)


def test_the_partition_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_model_surface.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS

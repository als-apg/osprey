"""A run's limits posture is its target's, and a command write is a write the run lock gates.

An :class:`ActionContext` says whether limits checking is off for the run's
target, and that is never the caller's to say: the lock derives it from the
target's limits posture (complete, and ``enabled: false``) and refuses a
context that disagrees.

``command_write`` refuses the whole batch, before anything is written, unless
this process holds the run lock with a context whose owner is the owner the
write path reads, the connector drives the context's target, the context's
``limits_off`` is the target's, the connector's limits validator is not the
fail-safe one (and it carries one unless limits are off), and every address is
a command of the rendered facility definition and passes the validator in the
validator's own mode. Then it makes one ``write_multiple_channels`` call with
``confirm=False``. Every call files one audit record naming the commands that
fired. The validator here is the product's own; the connector records what it
was asked to write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime
from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    ENV_CONTROL_IDENTITY,
    SURFACE_COMMAND_WRITE,
    ActionContext,
    OspreyWriteRefused,
    command_write,
    lock,
    target_limits_off,
)
from osprey_connectors import posture_store
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system._limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey_connectors.workspace import reset_config_cache
from tests._guarded_run import LIVE_BASELINE, reset_run_state

#: Commands of the facility definition: setpoints whose signal ends in ``command``.
ARM = "SYN:BPM:ARM"
REQUEST = "SYN:INJ:REQ"
CONFIRMING = "SYN:INJ:CONFIRMING"
#: A setpoint of the facility definition that is not a command.
CORRECTOR = "SYN:COR:SP"
#: A readback whose signal ends in ``command``: a readback is never a command.
ECHO = "SYN:INJ:ECHO"
#: A channel the facility definition does not name at all.
STRANGER = "SYN:NOT:THERE"

#: The rendered facility definition's channels.
FACILITY_CHANNELS: list[dict[str, Any]] = [
    {"id": ARM, "role": "setpoint", "signal": "reset_command"},
    {"id": REQUEST, "role": "setpoint", "signal": "power_on_command"},
    {"id": CONFIRMING, "role": "setpoint", "signal": "lamp_command"},
    {"id": CORRECTOR, "role": "setpoint", "signal": "ramp_rate_setpoint"},
    {"id": ECHO, "signal": "valve_open_command"},
]

ALICE_LIVE = ActionContext(target="live", owner="alice", approved_target="live")
ALICE_LIVE_LIMITS_OFF = ActionContext(
    target="live", owner="alice", approved_target="live", limits_off=True
)

#: Limits blocks by what they say, as ``control_system.limits_checking``.
LIMITS_OFF: dict[str, Any] = {"enabled": False}
STRICT: dict[str, Any] = {"enabled": True, "mode": "exclusive", "on_violation": "error"}
OPTIONAL: dict[str, Any] = {"enabled": True, "mode": "optional", "on_violation": "error"}
UNSTATED: dict[str, Any] = {"mode": "exclusive"}


def _deploy(root: Path, limits_checking: Any = None) -> None:
    """Write ``build/config.yml`` on a live baseline with *limits_checking*, none when ``None``."""
    section: dict[str, Any] = dict(LIVE_BASELINE)
    if limits_checking is not None:
        section["limits_checking"] = limits_checking
    config = {"control_system": section, "approval": {"enabled": False}}
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()


def _facility(root: Path, channels: list[dict[str, Any]]) -> None:
    """Render ``build/facility.json`` holding *channels*."""
    document = {"identity": {"code": "SYN", "name": "Synthetic"}, "channels": channels}
    (root / "build" / "facility.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host repo on a live baseline with no limits block, stamped ``live``, owner alice.

    The run state is fresh: nothing is held.
    """
    reset_run_state(monkeypatch)
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    _deploy(root)
    _facility(root, FACILITY_CHANNELS)
    monkeypatch.chdir(root)
    for name in ("OSPREY_CONFIG", "CONFIG_FILE", "OSPREY_EXECUTION_MODE", ENV_CONTROL_IDENTITY):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    yield root
    reset_config_cache()


# --------------------------------------------------------------------------
# limits_off is the target's
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limits_checking", "off"),
    [
        (LIMITS_OFF, True),
        (STRICT, False),
        (OPTIONAL, False),
        (UNSTATED, False),
        (None, False),
        ("not a block", False),
    ],
    ids=["enabled-false", "strict", "optional", "unstated-enabled", "no-block", "unreadable"],
)
def test_limits_are_off_only_for_a_complete_posture_that_says_enabled_false(
    repo: Path, limits_checking: Any, off: bool
) -> None:
    _deploy(repo, limits_checking)

    assert target_limits_off("live") is off


def test_a_context_saying_limits_are_off_holds_the_lock_on_a_limits_off_target(
    repo: Path,
) -> None:
    _deploy(repo, LIMITS_OFF)

    with lock(None, context=ALICE_LIVE_LIMITS_OFF):
        pass


@pytest.mark.parametrize(
    "limits_checking",
    [STRICT, OPTIONAL, UNSTATED, None],
    ids=["strict", "optional", "unstated-enabled", "no-block"],
)
def test_a_caller_built_context_claiming_limits_off_is_refused(
    repo: Path, limits_checking: Any
) -> None:
    _deploy(repo, limits_checking)

    with pytest.raises(RuntimeError, match="limits checking is off"):
        with lock(None, context=ALICE_LIVE_LIMITS_OFF):
            pytest.fail("a run whose context claims limits are off never starts")


def test_a_context_saying_limits_are_on_is_refused_on_a_limits_off_target(repo: Path) -> None:
    _deploy(repo, LIMITS_OFF)

    with pytest.raises(RuntimeError, match="limits checking is on"):
        with lock(None, context=ALICE_LIVE):
            pytest.fail("a run whose context disagrees with its target never starts")


# --------------------------------------------------------------------------
# command_write
# --------------------------------------------------------------------------


def _validator(failsafe_reason: str | None = None) -> LimitsValidator:
    """The product's validator over ARM, REQUEST, CONFIRMING and CORRECTOR, exclusive."""
    if failsafe_reason is not None:
        return LimitsValidator({}, {}, {}, failsafe_reason=failsafe_reason)
    bands = {ARM: (0, 1), REQUEST: (0, 1000), CONFIRMING: (0, 1), CORRECTOR: (-5, 5)}
    limits = {a: ChannelLimitsConfig(a, lo, hi, writable=True) for a, (lo, hi) in bands.items()}
    raw = {a: {"min_value": lo, "max_value": hi} for a, (lo, hi) in bands.items()}
    return LimitsValidator(limits, {"mode": "exclusive", "on_violation": "error"}, raw)


def _mode_validator(mode: str, records: dict[str, dict[str, Any]] | None = None) -> LimitsValidator:
    """The product's validator in *mode* over *records* (none: an empty database)."""
    records = records or {}
    limits = {
        address: ChannelLimitsConfig(
            address, record.get("min_value"), record.get("max_value"), writable=True
        )
        for address, record in records.items()
    }
    return LimitsValidator(limits, {"mode": mode, "on_violation": "error"}, dict(records))


class _Connector:
    """Records every batch it is asked to write; each result is the outcome set for its address."""

    def __init__(self, target: str | None = "live", validator: Any = "default") -> None:
        self._control_target = target
        self._limits_validator = _validator() if validator == "default" else validator
        self.outcomes: dict[str, WriteOutcome] = {}
        self.calls: list[tuple[list[tuple[str, Any]], dict[str, Any]]] = []

    def current_value_reader(self) -> None:
        return None

    async def write_multiple_channels(
        self, operations: list[tuple[str, Any]], **kwargs: Any
    ) -> list[ChannelWriteResult]:
        self.calls.append((list(operations), kwargs))
        results = []
        for address, value in operations:
            outcome = self.outcomes.get(address, WriteOutcome.UNREQUESTED)
            message = None if outcome is WriteOutcome.UNREQUESTED else f"{outcome} here"
            results.append(
                ChannelWriteResult(
                    channel_address=address,
                    value_written=value,
                    outcome=outcome,
                    refusal_reason="WRITES_DISABLED" if outcome is WriteOutcome.REFUSED else None,
                    error_message=message,
                )
            )
        return results


def _command_write_records(recorded_entries) -> list[dict]:
    """The audit record bodies filed on the command-write surface, oldest first."""
    return [
        entry.body
        for entry in recorded_entries("audit")
        if entry.body.get("surface") == SURFACE_COMMAND_WRITE
    ]


def _fired_in(record: dict[str, Any]) -> list[str]:
    """The ``commands_fired`` an audit record of a command write names."""
    key, _, listed = record["detail"].partition("=")
    assert key == "commands_fired"
    return listed.split(",") if listed else []


async def _refused(
    connector: _Connector, addresses: list[str], value: Any = 1, context: Any = ALICE_LIVE
) -> OspreyWriteRefused:
    """Fire under *context*'s lock; return the refusal, having checked nothing was written."""
    with lock(None, context=context), pytest.raises(OspreyWriteRefused) as caught:
        await command_write(connector, addresses, value)
    assert connector.calls == [], "a refused batch writes nothing"
    return caught.value


async def test_a_batch_of_commands_goes_out_in_one_call_with_confirm_false() -> None:
    connector = _Connector()

    with lock(None, context=ALICE_LIVE):
        fired = await command_write(connector, [ARM, REQUEST], 1)

    assert fired == [ARM, REQUEST]
    assert connector.calls == [([(ARM, 1), (REQUEST, 1)], {"confirm": False})]


async def test_a_fired_batch_files_an_audit_record_naming_the_commands_that_fired(
    recorded_entries,
) -> None:
    with lock(None, context=ALICE_LIVE):
        await command_write(_Connector(), [ARM, REQUEST], 1)

    [record] = _command_write_records(recorded_entries)
    assert record["decision"] == "allowed"
    assert record["subject"] == "live"
    assert record["on_behalf_of"] == "alice"
    assert _fired_in(record) == [ARM, REQUEST]


async def test_a_command_write_without_a_held_lock_is_refused_and_audited(
    recorded_entries,
) -> None:
    connector = _Connector()

    with pytest.raises(OspreyWriteRefused, match="run lock with a context"):
        await command_write(connector, [ARM], 1)
    assert connector.calls == []
    [record] = _command_write_records(recorded_entries)
    assert record["decision"] == "refused"
    assert _fired_in(record) == []


async def test_a_command_write_under_a_lock_without_a_context_is_refused() -> None:
    connector = _Connector()

    with lock(None), pytest.raises(OspreyWriteRefused, match="run lock with a context"):
        await command_write(connector, [ARM], 1)
    assert connector.calls == []


async def test_a_command_write_for_another_owner_than_the_runs_is_refused(
    monkeypatch: pytest.MonkeyPatch, recorded_entries
) -> None:
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "bob")

    assert "'bob'" in (await _refused(_Connector(), [ARM])).reason
    [record] = _command_write_records(recorded_entries)
    assert record["decision"] == "refused"
    assert _fired_in(record) == []


async def test_a_command_write_while_the_owner_bound_for_the_plan_differs_is_refused() -> None:
    with posture_store.bind_owner({posture_store.RESERVED_OWNER_KWARG: "bob"}):
        assert "'bob'" in (await _refused(_Connector(), [ARM])).reason


async def test_a_command_write_in_a_run_that_belongs_to_nobody_is_refused() -> None:
    nobody = ActionContext(target="live", owner=None, approved_target="live")

    assert "nobody" in (await _refused(_Connector(), [ARM], context=nobody)).reason


async def test_a_command_write_when_the_write_path_names_no_owner_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR)
    monkeypatch.setattr(posture_store, "current_owner", lambda: posture_store.NO_OWNER)

    assert "nobody" in (await _refused(_Connector(), [ARM])).reason


async def test_a_connector_driving_another_target_than_the_runs_is_refused() -> None:
    assert "drives va" in (await _refused(_Connector(target="va"), [ARM])).reason


async def test_a_connector_naming_no_target_is_refused() -> None:
    assert "names no control target" in (await _refused(_Connector(target=None), [ARM])).reason


async def test_a_connector_on_the_baseline_stand_in_drives_the_baseline_target() -> None:
    connector = _Connector(target="baseline")

    with lock(None, context=ALICE_LIVE):
        assert await command_write(connector, [ARM], 1) == [ARM]
    assert len(connector.calls) == 1


async def test_an_empty_batch_is_refused() -> None:
    assert "no command" in (await _refused(_Connector(), [])).reason


async def test_a_batch_naming_a_command_twice_is_refused_whole(recorded_entries) -> None:
    # A command fires once per write: a repeated address would leave the audit
    # record unable to say which occurrence went out when one is refused.
    refusal = await _refused(_Connector(), [ARM, REQUEST, ARM])

    assert refusal.channel_address == ARM
    assert "once" in refusal.reason
    [record] = _command_write_records(recorded_entries)
    assert record["decision"] == "refused"
    assert _fired_in(record) == []


@pytest.mark.parametrize(
    "address", [CORRECTOR, ECHO, STRANGER], ids=["setpoint", "readback", "unnamed"]
)
async def test_a_batch_naming_a_channel_the_facility_does_not_mark_a_command_is_refused_whole(
    address: str,
) -> None:
    refusal = await _refused(_Connector(), [ARM, address])

    assert refusal.channel_address == address
    assert "facility definition" in refusal.reason


async def test_command_ness_is_the_facility_definitions_and_never_the_callers() -> None:
    import inspect

    assert list(inspect.signature(command_write).parameters) == ["connector", "addresses", "value"]


async def test_no_rendered_facility_definition_means_no_command_fires(repo: Path) -> None:
    (repo / "build" / "facility.json").unlink()

    assert "facility definition" in (await _refused(_Connector(), [ARM])).reason


async def test_an_unreadable_facility_definition_means_no_command_fires(repo: Path) -> None:
    (repo / "build" / "facility.json").write_text("{not json", encoding="utf-8")

    assert "facility definition" in (await _refused(_Connector(), [ARM])).reason


async def test_a_connector_with_no_limits_validator_is_refused_while_limits_are_on() -> None:
    reason = (await _refused(_Connector(validator=None), [ARM])).reason

    assert "enforcing limits validator" in reason


@pytest.mark.parametrize(
    "limits_checking", [UNSTATED, OPTIONAL], ids=["unstated-enabled", "optional"]
)
async def test_a_validator_less_command_write_is_refused_unless_limits_are_explicitly_off(
    repo: Path, limits_checking: dict[str, Any]
) -> None:
    _deploy(repo, limits_checking)

    reason = (await _refused(_Connector(validator=None), [ARM])).reason

    assert "enforcing limits validator" in reason


async def test_a_connector_with_no_validator_fires_on_a_limits_off_target(repo: Path) -> None:
    _deploy(repo, LIMITS_OFF)
    connector = _Connector(validator=None)

    with lock(None, context=ALICE_LIVE_LIMITS_OFF):
        assert await command_write(connector, [ARM], 1) == [ARM]
    assert connector.calls == [([(ARM, 1)], {"confirm": False})]


async def test_a_held_context_whose_limits_off_the_target_no_longer_says_is_refused(
    repo: Path,
) -> None:
    _deploy(repo, LIMITS_OFF)
    connector = _Connector(validator=None)

    with lock(None, context=ALICE_LIVE_LIMITS_OFF):
        _deploy(repo, STRICT)
        with pytest.raises(OspreyWriteRefused, match="limits posture"):
            await command_write(connector, [ARM], 1)
    assert connector.calls == []


async def test_the_failsafe_validator_refuses_even_on_a_limits_off_target(repo: Path) -> None:
    _deploy(repo, LIMITS_OFF)
    connector = _Connector(validator=_validator(failsafe_reason="the limits file is unreadable"))

    refusal = await _refused(connector, [ARM], context=ALICE_LIVE_LIMITS_OFF)

    assert "the limits file is unreadable" in refusal.reason


async def test_limits_off_never_admits_a_channel_that_is_not_a_command(repo: Path) -> None:
    _deploy(repo, LIMITS_OFF)

    refusal = await _refused(
        _Connector(validator=None), [ARM, CORRECTOR], context=ALICE_LIVE_LIMITS_OFF
    )

    assert refusal.channel_address == CORRECTOR


async def test_a_strict_target_applies_each_fired_commands_limits_record(repo: Path) -> None:
    _deploy(repo, STRICT)
    connector = _Connector()

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelLimitsViolationError) as caught:
        # 5 is inside REQUEST's band and outside ARM's.
        await command_write(connector, [REQUEST, ARM], 5)

    assert caught.value.channel_address == ARM
    assert connector.calls == [], "the command before it is not written either"


async def test_exclusive_mode_refuses_a_command_with_no_limits_record_whole() -> None:
    connector = _Connector(
        validator=_mode_validator("exclusive", {ARM: {"min_value": 0, "max_value": 1}})
    )

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelLimitsViolationError) as caught:
        await command_write(connector, [ARM, REQUEST], 1)

    assert caught.value.channel_address == REQUEST
    assert connector.calls == []


async def test_optional_mode_writes_a_command_with_no_limits_record_unchecked() -> None:
    connector = _Connector(validator=_mode_validator("optional"))

    with lock(None, context=ALICE_LIVE):
        assert await command_write(connector, [ARM, REQUEST], 7) == [ARM, REQUEST]
    assert connector.calls == [([(ARM, 7), (REQUEST, 7)], {"confirm": False})]


async def test_optional_mode_applies_a_listed_commands_bounds() -> None:
    connector = _Connector(
        validator=_mode_validator("optional", {ARM: {"min_value": 0, "max_value": 1}})
    )

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelLimitsViolationError) as caught:
        await command_write(connector, [REQUEST, ARM], 5)

    assert caught.value.channel_address == ARM
    assert connector.calls == []


async def test_a_failed_command_raises_naming_the_commands_that_fired(
    recorded_entries,
) -> None:
    connector = _Connector()
    connector.outcomes[REQUEST] = WriteOutcome.FAILED

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelWriteFailedError) as caught:
        await command_write(connector, [ARM, REQUEST], 1)

    assert caught.value.channel_address == REQUEST
    assert caught.value.commands_fired == [ARM, REQUEST]
    [record] = _command_write_records(recorded_entries)
    assert record["decision"] == "refused"
    assert _fired_in(record) == [ARM, REQUEST]


async def test_a_refused_command_is_not_named_among_those_that_fired(
    recorded_entries,
) -> None:
    connector = _Connector()
    connector.outcomes[ARM] = WriteOutcome.REFUSED

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelWriteBlockedError) as caught:
        await command_write(connector, [ARM, REQUEST], 1)

    assert caught.value.channel_address == ARM
    assert caught.value.commands_fired == [REQUEST]
    [record] = _command_write_records(recorded_entries)
    assert _fired_in(record) == [REQUEST]


async def test_a_batch_whose_write_raises_is_audited_as_attempted_with_its_outcome_unknown(
    recorded_entries,
) -> None:
    class _Raising(_Connector):
        async def write_multiple_channels(
            self, operations: list[tuple[str, Any]], **kwargs: Any
        ) -> list[ChannelWriteResult]:
            self.calls.append((list(operations), kwargs))
            raise ConnectionError("lost the link after the first command went out")

    connector = _Raising()

    with lock(None, context=ALICE_LIVE), pytest.raises(ConnectionError):
        await command_write(connector, [ARM, REQUEST], 1)

    assert connector.calls == [([(ARM, 1), (REQUEST, 1)], {"confirm": False})]
    [record] = _command_write_records(recorded_entries)
    assert record["decision"] == "refused"
    assert record["reason"] == "outcome_unknown"
    key, _, listed = record["detail"].partition("=")
    assert key == "commands_attempted", "a raise after admission never reads as nothing fired"
    assert listed.split(",") == [ARM, REQUEST]


async def test_a_refusal_raised_partway_names_every_command_not_refused_as_fired(
    recorded_entries,
) -> None:
    class _RefusesTheSecond(_Connector):
        async def write_multiple_channels(
            self, operations: list[tuple[str, Any]], **kwargs: Any
        ) -> list[ChannelWriteResult]:
            self.calls.append((list(operations), kwargs))
            raise ChannelWriteBlockedError(REQUEST, "WRITES_DISABLED", "refused partway")

    connector = _RefusesTheSecond()

    with (
        osprey.runtime.record_writes() as records,
        lock(None, context=ALICE_LIVE),
        pytest.raises(ChannelWriteBlockedError) as caught,
    ):
        await command_write(connector, [ARM, REQUEST], 1)

    assert caught.value.commands_fired == [ARM], "the first command may have gone out"
    assert records == [(ARM, "attempt"), (REQUEST, "attempt"), (REQUEST, "refused")]
    [record] = _command_write_records(recorded_entries)
    assert record["reason"] == "outcome_unknown"


async def test_the_phases_of_a_fired_batch_reach_every_write_recorder() -> None:
    connector = _Connector()
    connector.outcomes[ARM] = WriteOutcome.REFUSED

    with (
        osprey.runtime.record_writes() as records,
        lock(None, context=ALICE_LIVE),
        pytest.raises(ChannelWriteBlockedError),
    ):
        await command_write(connector, [ARM, REQUEST], 1)

    assert records == [
        (ARM, "attempt"),
        (REQUEST, "attempt"),
        (ARM, "refused"),
        (REQUEST, "sent"),
    ]


@pytest.mark.parametrize("raises", [False, True], ids=["results", "raised-partway"])
async def test_a_concurrent_write_in_the_process_never_counts_as_a_fired_command(
    raises: bool, recorded_entries
) -> None:
    class _AnotherWriterLandsMeanwhile(_Connector):
        async def write_multiple_channels(
            self, operations: list[tuple[str, Any]], **kwargs: Any
        ) -> list[ChannelWriteResult]:
            # Another writer in this process reports its own phases mid-call.
            osprey.runtime._notify_write(CORRECTOR, "attempt")
            osprey.runtime._notify_write(CORRECTOR, "sent")
            if raises:
                self.calls.append((list(operations), kwargs))
                raise ChannelWriteBlockedError(REQUEST, "WRITES_DISABLED", "refused partway")
            return await super().write_multiple_channels(operations, **kwargs)

    connector = _AnotherWriterLandsMeanwhile()
    connector.outcomes[REQUEST] = WriteOutcome.REFUSED

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelWriteBlockedError) as caught:
        await command_write(connector, [ARM, REQUEST], 1)

    assert caught.value.commands_fired == [ARM]
    [record] = _command_write_records(recorded_entries)
    key, _, listed = record["detail"].partition("=")
    assert CORRECTOR not in listed.split(","), f"{key} names another writer's channel"


async def test_a_fired_batch_returns_the_commands_that_fired() -> None:
    connector = _Connector()

    with lock(None, context=ALICE_LIVE):
        assert await command_write(connector, [ARM, REQUEST], 1) == [ARM, REQUEST]


async def test_a_command_write_never_reaches_the_sandbox_connector_pin_or_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sandbox_connector() -> None:
        raise AssertionError("the sandbox's connector was asked for")

    def no_pin() -> None:
        raise AssertionError("the sandbox's target pin was checked")

    class NoValidator:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("the sandbox's validator was read")

    monkeypatch.setattr(osprey.runtime, "_get_connector", no_sandbox_connector)
    monkeypatch.setattr(osprey.runtime, "_assert_target_pin", no_pin)
    monkeypatch.setattr(osprey.runtime, "_limits_validator", NoValidator())

    with lock(None, context=ALICE_LIVE):
        assert await command_write(_Connector(), [ARM], 1) == [ARM]


def test_command_write_is_public_on_the_guarded_run_module() -> None:
    import osprey.runtime.guarded_run as guarded_run

    assert {"command_write", "SURFACE_COMMAND_WRITE", "target_limits_off"} <= set(
        guarded_run.__all__
    )

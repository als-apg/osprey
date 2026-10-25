"""A command write: one batch of commands a run that holds the lock with a context fires.

``command_write`` refuses the whole batch, before anything is written, unless
this process holds the run lock with a context whose owner is the owner the
write path reads, the connector drives the context's target, its limits
validator is not the fail-safe one (and it carries one unless the context says
limits checking is off), and every address is a command of the caller's and
passes the validator in the validator's own mode. No limits record need make
an address a command (q-0001/1, q-0002). Then it makes one
``write_multiple_channels`` call with ``confirm=False``. The validator here is
the product's own; the connector records what it was asked to write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime
from osprey.facility.views.limits import limits_document
from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    ENV_CONTROL_IDENTITY,
    ActionContext,
    OspreyWriteRefused,
    command_write,
    lock,
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
from tests._guarded_run import reset_run_state, write_config

ARM = "SR:BPM:ARM"
REQUEST = "SR:INJ:REQ"
#: A channel with limits the caller does not fire as a command.
CORRECTOR = "SCOR:H:SP"
#: A channel the caller lists as a command whose limits record confirms.
CONFIRMING = "SR:INJ:CONFIRMING"
COMMANDS = frozenset({ARM, REQUEST, CONFIRMING})

ALICE_LIVE = ActionContext(target="live", owner="alice", approved_target="live")


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host repo on a live baseline, stamped ``live``, whose write path reads owner alice.

    The run state is fresh: nothing is held.
    """
    reset_run_state(monkeypatch)
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(ENV_CONTROL_IDENTITY, raising=False)
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    return root


def _validator(failsafe_reason: str | None = None) -> LimitsValidator:
    """The product's validator over ARM, REQUEST (confirm: false), CONFIRMING and CORRECTOR."""
    limits = {
        ARM: ChannelLimitsConfig(ARM, 0, 1, writable=True),
        REQUEST: ChannelLimitsConfig(REQUEST, 0, 1000, writable=True),
        CONFIRMING: ChannelLimitsConfig(CONFIRMING, 0, 1, writable=True),
        CORRECTOR: ChannelLimitsConfig(CORRECTOR, -5, 5, writable=True),
    }
    raw = {
        ARM: {"min_value": 0, "max_value": 1, "confirm": False},
        REQUEST: {"min_value": 0, "max_value": 1000, "confirm": False},
        CONFIRMING: {"min_value": 0, "max_value": 1},
        CORRECTOR: {"min_value": -5, "max_value": 5, "confirm": False},
    }
    if failsafe_reason is not None:
        return LimitsValidator({}, {}, {}, failsafe_reason=failsafe_reason)
    return LimitsValidator(limits, {"mode": "exclusive", "on_violation": "error"}, raw)


class _Connector:
    """Records every batch it is asked to write; each result is the outcome set for its address."""

    def __init__(self, target: str | None = "live", validator: Any = "default") -> None:
        self._control_target = target
        self._limits_validator = _validator() if validator == "default" else validator
        self.outcomes: dict[str, WriteOutcome] = {}
        self.calls: list[tuple[list[tuple[str, Any]], dict[str, Any]]] = []

    def _resolve_confirm(self, address: str) -> bool:
        if self._limits_validator is None:
            return True
        return bool(self._limits_validator.resolve_confirm(address))

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


async def _refused(connector: _Connector, addresses: list[str], value: Any = 1) -> str:
    """Fire under alice's live lock; return the refusal, having checked nothing was written."""
    with lock(None, context=ALICE_LIVE), pytest.raises(OspreyWriteRefused) as caught:
        await command_write(connector, addresses, value, commands=COMMANDS)
    assert connector.calls == [], "a refused batch writes nothing"
    return caught.value.reason


async def test_a_batch_of_commands_goes_out_in_one_call_with_confirm_false() -> None:
    # Flipped per q-0001/1: a command holds no setting, so it is fired unconfirmed.
    connector = _Connector()

    with lock(None, context=ALICE_LIVE):
        fired = await command_write(connector, [ARM, REQUEST], 1, commands=COMMANDS)

    assert fired == [ARM, REQUEST]
    assert connector.calls == [([(ARM, 1), (REQUEST, 1)], {"confirm": False})]


async def test_a_command_write_without_a_held_lock_is_refused() -> None:
    connector = _Connector()

    with pytest.raises(OspreyWriteRefused, match="run lock with a context"):
        await command_write(connector, [ARM], 1, commands=COMMANDS)
    assert connector.calls == []


async def test_a_command_write_under_a_lock_without_a_context_is_refused() -> None:
    connector = _Connector()

    with lock(None), pytest.raises(OspreyWriteRefused, match="run lock with a context"):
        await command_write(connector, [ARM], 1, commands=COMMANDS)
    assert connector.calls == []


async def test_a_command_write_for_another_owner_than_the_runs_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "bob")

    assert "'bob'" in await _refused(_Connector(), [ARM])


async def test_a_command_write_while_the_owner_bound_for_the_plan_differs_is_refused() -> None:
    with posture_store.bind_owner({posture_store.RESERVED_OWNER_KWARG: "bob"}):
        assert "'bob'" in await _refused(_Connector(), [ARM])


async def test_a_command_write_in_a_run_that_belongs_to_nobody_is_refused() -> None:
    connector = _Connector()
    nobody = ActionContext(target="live", owner=None, approved_target="live")

    with lock(None, context=nobody), pytest.raises(OspreyWriteRefused, match="nobody"):
        await command_write(connector, [ARM], 1, commands=COMMANDS)
    assert connector.calls == []


async def test_a_command_write_when_the_write_path_names_no_owner_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR)
    monkeypatch.setattr(posture_store, "current_owner", lambda: posture_store.NO_OWNER)

    assert "nobody" in await _refused(_Connector(), [ARM])


async def test_a_connector_with_no_limits_validator_is_refused() -> None:
    assert "enforcing limits validator" in await _refused(_Connector(validator=None), [ARM])


async def test_a_connector_with_the_failsafe_validator_is_refused() -> None:
    connector = _Connector(validator=_validator(failsafe_reason="the limits file is unreadable"))

    assert "enforcing limits validator" in await _refused(connector, [ARM])


async def test_a_batch_naming_one_address_that_is_not_a_command_is_refused_whole() -> None:
    connector = _Connector()

    with lock(None, context=ALICE_LIVE), pytest.raises(OspreyWriteRefused) as caught:
        await command_write(connector, [ARM, CORRECTOR], 1, commands=COMMANDS)

    assert caught.value.channel_address == CORRECTOR
    assert connector.calls == [], "the command before it is not written either"


async def test_a_command_whose_limits_record_confirms_is_still_fired_unconfirmed() -> None:
    # Flipped per q-0001/1: command-ness is the caller's, never a limits record's.
    connector = _Connector()

    with lock(None, context=ALICE_LIVE):
        fired = await command_write(connector, [ARM, CONFIRMING], 1, commands=COMMANDS)

    assert fired == [ARM, CONFIRMING]
    assert connector.calls == [([(ARM, 1), (CONFIRMING, 1)], {"confirm": False})]


async def test_a_batch_whose_value_one_command_s_limits_refuse_is_refused_whole() -> None:
    connector = _Connector()

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelLimitsViolationError):
        # 5 is inside REQUEST's band and outside ARM's.
        await command_write(connector, [REQUEST, ARM], 5, commands=COMMANDS)
    assert connector.calls == []


async def test_a_connector_driving_another_target_than_the_runs_is_refused() -> None:
    assert "drives va" in await _refused(_Connector(target="va"), [ARM])


async def test_a_connector_naming_no_target_is_refused() -> None:
    assert "names no control target" in await _refused(_Connector(target=None), [ARM])


async def test_a_connector_on_the_baseline_stand_in_drives_the_baseline_target() -> None:
    connector = _Connector(target="baseline")

    with lock(None, context=ALICE_LIVE):
        assert await command_write(connector, [ARM], 1, commands=COMMANDS) == [ARM]
    assert len(connector.calls) == 1


async def test_an_empty_batch_is_refused() -> None:
    assert "no command" in await _refused(_Connector(), [])


async def test_a_failed_command_raises_and_names_the_commands_that_fired() -> None:
    connector = _Connector()
    connector.outcomes[REQUEST] = WriteOutcome.FAILED

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelWriteFailedError) as caught:
        await command_write(connector, [ARM, REQUEST], 1, commands=COMMANDS)

    assert caught.value.channel_address == REQUEST
    assert caught.value.commands_fired == [ARM, REQUEST]
    assert len(connector.calls) == 1


async def test_a_refused_command_raises_and_is_not_named_among_those_that_fired() -> None:
    connector = _Connector()
    connector.outcomes[ARM] = WriteOutcome.REFUSED

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelWriteBlockedError) as caught:
        await command_write(connector, [ARM, REQUEST], 1, commands=COMMANDS)

    assert caught.value.channel_address == ARM
    assert caught.value.commands_fired == [REQUEST]


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
    connector = _Connector()

    with lock(None, context=ALICE_LIVE):
        assert await command_write(connector, [ARM], 1, commands=COMMANDS) == [ARM]


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


async def test_optional_mode_writes_a_command_with_no_limits_record_unchecked() -> None:
    connector = _Connector(validator=_mode_validator("optional"))

    with lock(None, context=ALICE_LIVE):
        fired = await command_write(connector, [ARM, REQUEST], 7, commands=COMMANDS)

    assert fired == [ARM, REQUEST]
    assert connector.calls == [([(ARM, 7), (REQUEST, 7)], {"confirm": False})]


async def test_optional_mode_with_no_limits_file_fires_the_commands(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional mode and no database path: the product builds an enforcing empty validator."""
    from osprey_connectors.workspace import reset_config_cache

    (repo / "config.yml").write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "mock",
                    "limits_checking": {
                        "enabled": True,
                        "mode": "optional",
                        "on_violation": "error",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_FILE", str(repo / "config.yml"))
    reset_config_cache()
    try:
        validator = LimitsValidator.from_config()
    finally:
        reset_config_cache()
    assert validator is not None
    assert validator.failsafe_reason is None
    connector = _Connector(validator=validator)

    with lock(None, context=ALICE_LIVE):
        assert await command_write(connector, [ARM, REQUEST], 1, commands=COMMANDS) == [
            ARM,
            REQUEST,
        ]
    assert connector.calls == [([(ARM, 1), (REQUEST, 1)], {"confirm": False})]


async def test_exclusive_mode_refuses_a_command_with_no_limits_record_whole() -> None:
    connector = _Connector(
        validator=_mode_validator("exclusive", {ARM: {"min_value": 0, "max_value": 1}})
    )

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelLimitsViolationError) as caught:
        await command_write(connector, [ARM, REQUEST], 1, commands=COMMANDS)

    assert caught.value.channel_address == REQUEST
    assert connector.calls == [], "the listed command before it is not written either"


async def test_optional_mode_applies_a_listed_commands_bounds() -> None:
    connector = _Connector(
        validator=_mode_validator("optional", {ARM: {"min_value": 0, "max_value": 1}})
    )

    with lock(None, context=ALICE_LIVE), pytest.raises(ChannelLimitsViolationError) as caught:
        # REQUEST is unlisted and would pass; ARM's band refuses 5.
        await command_write(connector, [REQUEST, ARM], 5, commands=COMMANDS)

    assert caught.value.channel_address == ARM
    assert connector.calls == []


async def test_a_connector_with_no_validator_fires_when_the_run_says_limits_are_off() -> None:
    connector = _Connector(validator=None)
    off = ActionContext(target="live", owner="alice", approved_target="live", limits_off=True)

    with lock(None, context=off):
        assert await command_write(connector, [ARM], 1, commands=COMMANDS) == [ARM]
    assert connector.calls == [([(ARM, 1)], {"confirm": False})]


async def test_the_failsafe_validator_refuses_even_when_the_run_says_limits_are_off() -> None:
    connector = _Connector(validator=_validator(failsafe_reason="the limits file is unreadable"))
    off = ActionContext(target="live", owner="alice", approved_target="live", limits_off=True)

    with lock(None, context=off), pytest.raises(OspreyWriteRefused) as caught:
        await command_write(connector, [ARM], 1, commands=COMMANDS)

    assert "the limits file is unreadable" in caught.value.reason
    assert connector.calls == []


async def test_limits_off_never_admits_an_address_that_is_not_a_command() -> None:
    connector = _Connector(validator=None)
    off = ActionContext(target="live", owner="alice", approved_target="live", limits_off=True)

    with lock(None, context=off), pytest.raises(OspreyWriteRefused) as caught:
        await command_write(connector, [ARM, CORRECTOR], 1, commands=COMMANDS)

    assert caught.value.channel_address == CORRECTOR
    assert connector.calls == []


#: The shipped example facility, whose ``limits.yaml`` lists no shot command.
EXAMPLE = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
#: The example facility's shot commands: one BPM's arm and the injection request.
EXAMPLE_SHOT = ("SR:DIAG:BPM:01:TBT:ARM", "SR:DIAG:INJECTION:REQUEST")


async def test_a_shots_commands_fire_under_the_shipped_example_limits_in_optional_mode(
    tmp_path: Path,
) -> None:
    import json

    records = yaml.safe_load((EXAMPLE / "limits.yaml").read_text(encoding="utf-8"))
    models = (EXAMPLE / "models.yaml").read_text(encoding="utf-8")
    for address in EXAMPLE_SHOT:
        assert address in models, f"{address} is no longer an example shot command"
    channels = [{"id": record["address"], "role": "setpoint"} for record in records["records"]]
    database = limits_document({"channels": channels, "limits": records})
    assert not set(EXAMPLE_SHOT) & set(database), "the example file lists no shot command"
    path = tmp_path / "channel_limits.json"
    path.write_text(json.dumps(database), encoding="utf-8")
    limits, raw = LimitsValidator._load_limits_database(str(path))
    validator = LimitsValidator(limits, {"mode": "optional", "on_violation": "error"}, raw)
    connector = _Connector(validator=validator)

    with lock(None, context=ALICE_LIVE):
        fired = await command_write(
            connector, list(EXAMPLE_SHOT), 1, commands=frozenset(EXAMPLE_SHOT)
        )

    assert fired == list(EXAMPLE_SHOT)
    assert connector.calls == [([(address, 1) for address in EXAMPLE_SHOT], {"confirm": False})]


def test_command_write_is_public_on_the_guarded_run_module() -> None:
    import osprey.runtime.guarded_run as guarded_run

    assert "command_write" in guarded_run.__all__

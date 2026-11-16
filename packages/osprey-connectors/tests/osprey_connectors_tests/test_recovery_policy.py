"""The one recovery policy: which failure restarts what, and how often."""

import asyncio
import gc

import pytest

from osprey_connectors import recovery
from osprey_connectors.errors import (
    ChannelError,
    ChannelReadFailedError,
    ChannelUnreachableError,
    ClientLibraryRefusedError,
    ConnectorInternalError,
    ConnectorUnavailableError,
    RecoveryAction,
    RecoveryRecord,
)
from osprey_connectors.recovery import RecoveryPolicy, Timing

_Step = recovery._Step


class _Incarnation:
    """Stands in for a proxy or a connector instance: hashable and weakly referable."""


class _Target:
    """A supervised connector that records every act the policy performs on it."""

    def __init__(self, *, in_child=True, answers_ping=True, respawns=None, gate=None):
        self.incarnation = _Incarnation()
        self.in_child = in_child
        self.answers_ping = answers_ping
        self.gate = gate
        self.pings = 0
        self.retires = 0
        self.respawns_called = 0
        if respawns is not None:
            self.respawn_answers = list(respawns)
            self.respawn = self._respawn

    async def ping(self):
        self.pings += 1
        if self.gate is not None:
            await self.gate.wait()
        return self.answers_ping

    async def retire(self):
        self.retires += 1
        if self.gate is not None:
            await self.gate.wait()

    async def _respawn(self):
        self.respawns_called += 1
        if self.gate is not None:
            await self.gate.wait()
        return self.respawn_answers.pop(0)


def _from_child(exc):
    setattr(exc, recovery._RAISED_BY_CHILD, True)
    return exc


def _instance(failure):
    """One real instance of a row's failure class."""
    if issubclass(failure, ChannelError):
        return failure("SR:X", "channel failed")
    if failure is ChannelReadFailedError:
        return failure(["SR:X"])
    if failure is ClientLibraryRefusedError:
        return failure(package="epics")
    return failure("failed")


def _rows(*, catch_all):
    return [row for row in recovery._STEPS if (row.failure is BaseException) is catch_all]


# -- the table ---------------------------------------------------------------


def test_the_catch_all_is_the_last_row_and_the_only_one():
    assert recovery._STEPS[-1].failure is BaseException
    assert len(_rows(catch_all=True)) == 1


def test_no_failure_reaches_two_steps():
    for row in _rows(catch_all=False):
        exc = _instance(row.failure)
        for origin in recovery._Origin:
            for write_timeout in (False, True):
                steps = {
                    other.step
                    for other in _rows(catch_all=False)
                    if recovery._matches(other, exc, origin, write_timeout=write_timeout)
                }
                assert len(steps) <= 1, (row.failure, origin, write_timeout, steps)


@pytest.mark.parametrize(
    ("exc", "origin", "write_timeout", "step"),
    [
        (ConnectorInternalError("bug"), recovery._Origin.CHILD, False, _Step.NONE),
        (ConnectorInternalError("bug"), recovery._Origin.LINK, False, _Step.NONE),
        (ChannelUnreachableError("SR:X"), recovery._Origin.CHILD, False, _Step.NONE),
        (ChannelUnreachableError("SR:X"), recovery._Origin.IN_PROCESS, False, _Step.NONE),
        (ChannelReadFailedError(["SR:X"]), recovery._Origin.CHILD, False, _Step.NONE),
        (ClientLibraryRefusedError(package="epics"), recovery._Origin.CHILD, False, _Step.NONE),
        (TimeoutError("deadline"), recovery._Origin.LINK, True, _Step.REPLACE),
        (TimeoutError("deadline"), recovery._Origin.LINK, False, _Step.PROBE),
        (TimeoutError("control system"), recovery._Origin.CHILD, False, _Step.NONE),
        (TimeoutError("control system"), recovery._Origin.IN_PROCESS, False, _Step.NONE),
        (ConnectorUnavailableError("gone"), recovery._Origin.CHILD, False, _Step.REPLACE),
        (ConnectorUnavailableError("gone"), recovery._Origin.IN_PROCESS, False, _Step.REPLACE),
        (ConnectionError("pipe"), recovery._Origin.LINK, False, _Step.REPLACE),
        (ConnectionError("transport"), recovery._Origin.CHILD, False, _Step.REPLACE),
        (ValueError("anything else"), recovery._Origin.CHILD, False, _Step.NONE),
    ],
)
def test_every_failure_class_has_one_step(exc, origin, write_timeout, step):
    assert recovery._step_of(exc, origin, write_timeout=write_timeout) is step


_N, _P, _R = _Step.NONE, _Step.PROBE, _Step.REPLACE

#: The step each IPC failure class takes, per origin (child, link, in process)
#: and write timeout (off, on), as it was before the classes took the package
#: base. A link timeout is probed or replaced; a lost connector is replaced from
#: anywhere; a codec error and a timeout outside the link restart nothing.
_IPC_STEPS = {
    "FrameDecodeError": (_N, _N, _N, _N, _N, _N),
    "FrameEncodeError": (_N, _N, _N, _N, _N, _N),
    "ChildUnresponsiveError": (_N, _N, _P, _R, _N, _N),
    "ConnectorHostSlowCallError": (_N, _N, _P, _R, _N, _N),
    "ConnectorHostError": (_R, _R, _R, _R, _R, _R),
    "ConnectorHostStartError": (_R, _R, _R, _R, _R, _R),
    "ConnectorHostLostError": (_R, _R, _R, _R, _R, _R),
    "ConnectorHostUnresponsiveError": (_R, _R, _R, _R, _R, _R),
    "ConnectorHostWriteTimeoutError": (_R, _R, _R, _R, _R, _R),
}


def _ipc_class(name):
    from osprey_connectors.ipc import _frames, _pool, _proxy

    return next(
        getattr(module, name) for module in (_frames, _proxy, _pool) if hasattr(module, name)
    )


@pytest.mark.parametrize("name", list(_IPC_STEPS))
def test_every_ipc_error_keeps_its_classification(name):
    from osprey_connectors.errors import ConnectorsError

    cls = _ipc_class(name)
    exc = cls.__new__(cls)
    assert isinstance(exc, ConnectorsError)
    steps = tuple(
        recovery._step_of(exc, origin, write_timeout=write_timeout)
        for origin in (recovery._Origin.CHILD, recovery._Origin.LINK, recovery._Origin.IN_PROCESS)
        for write_timeout in (False, True)
    )
    assert steps == _IPC_STEPS[name]


def test_a_slow_call_is_never_a_lost_connection():
    cls = _ipc_class("ConnectorHostSlowCallError")
    assert issubclass(cls, TimeoutError)
    assert not issubclass(cls, ConnectionError)
    assert not issubclass(_ipc_class("ChildUnresponsiveError"), ConnectionError)


def test_a_connect_refusal_is_answered_as_a_bare_connection_error():
    from osprey_connectors.errors import ConnectRefusedError

    for origin in recovery._Origin:
        for write_timeout in (False, True):
            assert recovery._step_of(
                ConnectRefusedError("refused"), origin, write_timeout=write_timeout
            ) is recovery._step_of(ConnectionError("x"), origin, write_timeout=write_timeout)


def test_a_lost_connector_that_is_also_a_timeout_is_decided_by_its_root():
    class Unanswered(ConnectorUnavailableError, TimeoutError):
        pass

    for write_timeout in (False, True):
        step = recovery._step_of(
            Unanswered("gone"), recovery._Origin.LINK, write_timeout=write_timeout
        )
        assert step is _Step.REPLACE


def test_the_policy_derives_the_origin_from_the_error_and_the_target():
    in_child = _Target()
    in_process = _Target(in_child=False)

    assert recovery._origin(ConnectionError("pipe"), in_child) is recovery._Origin.LINK
    assert recovery._origin(_from_child(ConnectionError("x")), in_child) is recovery._Origin.CHILD
    assert recovery._origin(ConnectionError("x"), in_process) is recovery._Origin.IN_PROCESS


# -- what recovery does --------------------------------------------------------


async def test_an_unreachable_channel_recovers_nothing():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target()
    exc = _from_child(ChannelUnreachableError("SR:X"))

    assert not isinstance(exc, ConnectionError)
    assert await policy.recover(exc, target, ticket=policy.begin()) is None
    assert (target.pings, target.retires) == (0, 0)


async def test_a_connector_bug_recovers_nothing():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target()

    record = await policy.recover(
        _from_child(ConnectorInternalError("KeyError('x')")), target, ticket=policy.begin()
    )

    assert record is None
    assert (target.pings, target.retires) == (0, 0)


async def test_a_lost_connector_is_replaced_once_for_twenty_failures():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    gate = asyncio.Event()
    target = _Target(gate=gate)
    calls = [
        asyncio.create_task(
            policy.recover(
                _from_child(ConnectorUnavailableError("gone")), target, ticket=policy.begin()
            )
        )
        for _ in range(20)
    ]
    await asyncio.sleep(0)
    gate.set()

    records = await asyncio.gather(*calls)

    assert target.retires == 1
    assert {record.action for record in records} == {RecoveryAction.CHILD_REPLACED}
    assert len({id(record) for record in records}) == 1


async def test_a_failure_after_the_replacement_does_not_replace_again():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target()

    first = await policy.recover(ConnectionError("pipe"), target, ticket=policy.begin())
    again = await policy.recover(ConnectionError("pipe"), target, ticket=policy.begin())

    assert target.retires == 1
    assert again is first


async def test_a_kept_child_is_pinged_again_next_time():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target(answers_ping=True)

    first = await policy.recover(TimeoutError("deadline"), target, ticket=policy.begin())
    second = await policy.recover(TimeoutError("deadline"), target, ticket=policy.begin())

    assert first.action is second.action is RecoveryAction.CHILD_KEPT
    assert target.pings == 2
    assert target.retires == 0


async def test_a_child_that_misses_the_ping_is_replaced():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target(answers_ping=False)

    record = await policy.recover(TimeoutError("deadline"), target, ticket=policy.begin())

    assert record.action is RecoveryAction.CHILD_REPLACED
    assert (target.pings, target.retires) == (1, 1)


async def test_a_write_timeout_replaces_without_a_ping_when_the_policy_says_so():
    policy = RecoveryPolicy(timing=Timing.LAZY, replace_on_write_timeout=True)
    target = _Target(answers_ping=True)

    write = await policy.recover(
        TimeoutError("deadline"), target, ticket=policy.begin(), write=True
    )

    assert write.action is RecoveryAction.CHILD_REPLACED
    assert (target.pings, target.retires) == (0, 1)


async def test_a_read_timeout_keeps_the_ping_rule_when_writes_replace():
    policy = RecoveryPolicy(timing=Timing.LAZY, replace_on_write_timeout=True)
    target = _Target(answers_ping=True)

    read = await policy.recover(TimeoutError("deadline"), target, ticket=policy.begin())

    assert read.action is RecoveryAction.CHILD_KEPT


async def test_an_in_process_connector_is_dropped():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target(in_child=False)

    record = await policy.recover(ConnectorUnavailableError("gone"), target, ticket=policy.begin())

    assert record.action is RecoveryAction.CONNECTOR_DROPPED
    assert target.retires == 1


async def test_an_eager_policy_respawns_instead_of_retiring():
    policy = RecoveryPolicy(timing=Timing.EAGER)
    target = _Target(respawns=[True])

    record = await policy.recover(ConnectorUnavailableError("gone"), target, ticket=policy.begin())

    assert record.action is RecoveryAction.CHILD_REPLACED
    assert (target.respawns_called, target.retires) == (1, 0)


async def test_a_refused_respawn_answers_the_calls_in_flight_only():
    policy = RecoveryPolicy(timing=Timing.EAGER)
    target = _Target(respawns=[False, True])
    in_flight = policy.begin()

    refused = await policy.recover(ConnectionError("gone"), target, ticket=policy.begin())
    earlier = await policy.recover(ConnectionError("gone"), target, ticket=in_flight)
    later = await policy.recover(ConnectionError("gone"), target, ticket=policy.begin())

    assert refused.action is RecoveryAction.RESPAWN_REFUSED
    assert earlier is refused
    assert later.action is RecoveryAction.CHILD_REPLACED
    assert target.respawns_called == 2


async def test_an_eager_policy_refuses_an_in_process_target():
    policy = RecoveryPolicy(timing=Timing.EAGER)

    with pytest.raises(ValueError, match="in-process"):
        await policy.recover(
            ConnectionError("gone"), _Target(in_child=False, respawns=[True]), ticket=policy.begin()
        )


async def test_an_eager_policy_refuses_a_target_that_cannot_respawn():
    policy = RecoveryPolicy(timing=Timing.EAGER)

    with pytest.raises(ValueError, match="respawn"):
        await policy.recover(ConnectionError("gone"), _Target(), ticket=policy.begin())


async def test_a_lazy_policy_never_calls_respawn():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target()
    assert not hasattr(target, "respawn")

    record = await policy.recover(ConnectionError("gone"), target, ticket=policy.begin())

    assert record.action is RecoveryAction.CHILD_REPLACED


@pytest.mark.usefixtures("earlier_garbage_frozen")
async def test_a_policy_holds_no_replaced_incarnation():
    policy = RecoveryPolicy(timing=Timing.LAZY)
    target = _Target()

    await policy.recover(ConnectionError("gone"), target, ticket=policy.begin())
    assert len(policy._memo) == 1
    del target
    gc.collect()

    assert len(policy._memo) == 0


def test_tickets_are_monotonic():
    policy = RecoveryPolicy(timing=Timing.LAZY)

    tickets = [policy.begin() for _ in range(3)]

    assert tickets == sorted(set(tickets))


# -- the records -------------------------------------------------------------


def test_an_unknown_action_word_fails_at_construction():
    with pytest.raises(ValueError):
        RecoveryRecord(action="child_rebooted", text="Something happened.")


def test_an_empty_sentence_fails_at_construction():
    with pytest.raises(ValueError):
        RecoveryRecord(action=RecoveryAction.CHILD_KEPT, text="")


def test_a_record_coerces_its_action_word_and_dumps_it_as_the_word():
    record = RecoveryRecord(action="child_kept", text="Kept.")

    assert record.action is RecoveryAction.CHILD_KEPT
    assert record == RecoveryRecord(action=RecoveryAction.CHILD_KEPT, text="Kept.")
    assert record.model_dump(mode="json") == {"action": "child_kept", "text": "Kept."}


def test_an_action_of_another_type_fails_at_construction():
    with pytest.raises(ValueError):
        RecoveryRecord(action=1, text="Something happened.")


def test_every_sentence_names_only_what_the_policy_knows():
    for (action, timing, _failure), text in recovery._SENTENCES.items():
        RecoveryRecord(action=action, text=text)
        assert "{" not in text
        if action is RecoveryAction.RESPAWN_REFUSED:
            assert timing is Timing.EAGER
        if action is RecoveryAction.CONNECTOR_DROPPED:
            assert timing is Timing.LAZY


def test_every_record_the_policy_can_make_has_a_sentence():
    for timing in Timing:
        for failure in recovery._Failure:
            actions = (
                [RecoveryAction.CHILD_REPLACED, RecoveryAction.RESPAWN_REFUSED]
                if timing is Timing.EAGER
                else [RecoveryAction.CHILD_REPLACED, RecoveryAction.CONNECTOR_DROPPED]
            )
            for action in actions:
                assert (action, timing, failure) in recovery._SENTENCES
        assert (RecoveryAction.CHILD_KEPT, timing, recovery._Failure.UNRESPONSIVE) in (
            recovery._SENTENCES
        )


def test_the_eager_sentences_tell_a_missed_ping_from_a_lost_connector():
    unavailable = recovery._SENTENCES[
        (RecoveryAction.CHILD_REPLACED, Timing.EAGER, recovery._Failure.UNAVAILABLE)
    ]
    unresponsive = recovery._SENTENCES[
        (RecoveryAction.CHILD_REPLACED, Timing.EAGER, recovery._Failure.UNRESPONSIVE)
    ]

    assert unavailable != unresponsive
    assert "ping" in unresponsive

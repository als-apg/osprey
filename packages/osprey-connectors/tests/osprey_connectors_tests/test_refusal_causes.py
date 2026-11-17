"""The refusal vocabulary's types: a verdict is allowed or refused, and a refusal says why.

A refusal carries its reason word and, where the word covers several cases, a
cause: a record a program reads by its fields. The words an operator reads are
the cause's sentence, pinned here kind by kind against the sentences the
verdict spoke before the causes existed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import pytest
from pydantic import ValidationError

from osprey_connectors import posture_store, session_hold
from osprey_connectors.control_system import ChannelWriteResult
from osprey_connectors.errors import ChannelWriteBlockedError
from osprey_connectors.posture_store import (
    Allowed,
    BudgetSpent,
    ChannelNotListed,
    FixedCause,
    OverlapHeld,
    Refused,
    StoppedBy,
    ValueOutOfRange,
    WrongMachine,
    _SessionHeld,
)
from osprey_connectors.refusals import REASONS, RefusalReason, VerdictReason
from osprey_connectors.write_authorization import AuthorizationRefusedError, ChannelRange

STOPPED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
ADDRESS = "SR:CORRECTOR:H1"

#: Every cause kind, and the sentence the verdict spoke for it before it had a
#: cause: each one an f-string or a literal of the earlier code, rendered here.
SENTENCES = [
    (ChannelNotListed(address=ADDRESS), f"Channel: {ADDRESS}."),
    (
        ValueOutOfRange(address=ADDRESS, min=-0.5, max=0.5),
        f"Channel: {ADDRESS}; allowed range [{-0.5}, {0.5}].",
    ),
    # An int bound is spoken as the int it is, as the range holds it.
    (ValueOutOfRange(address=ADDRESS, min=0, max=2), f"Channel: {ADDRESS}; allowed range [0, 2]."),
    (WrongMachine(covers="va"), "The authorization covers va, not this machine."),
    (StoppedBy(by="alice", revoked_at=STOPPED_AT), "Stopped by alice."),
    (BudgetSpent(allowed=41), "41 of 41 writes used."),
    (
        OverlapHeld(holder="optimizer", authorization="a" * 32, label="injector tuning 7"),
        f"Held by optimizer under {'a' * 32} (injector tuning 7).",
    ),
    (_SessionHeld(sources=("watchdog:a", "watchdog:b")), "Raised by: watchdog:a, watchdog:b."),
    (FixedCause(kind="not_an_id"), "The authorization this process names is not an id."),
    (FixedCause(kind="empty_batch"), "The write names no channel."),
    (FixedCause(kind="epoch_moved"), "The switch record has moved past its approval."),
    (FixedCause(kind="switch_left_allowed"), "The switch is no longer at allowed."),
    (FixedCause(kind="switch_not_at_allowed"), "The switch is not at allowed."),
    (FixedCause(kind="stop_during_issue"), "A stop came during the issue."),
    (FixedCause(kind="no_epoch"), "The switch has no epoch."),
    (FixedCause(kind="no_such_authorization"), "No such authorization."),
    (FixedCause(kind="no_ledger_directory"), "This process has no write ledger directory."),
    (
        FixedCause(kind="no_usable_ledger_directory"),
        "This process has no usable write ledger directory.",
    ),
    (FixedCause(kind="ledger_unreadable"), "The write ledger cannot be read."),
    (FixedCause(kind="lease_never_started"), "The holder never started its lease."),
    (FixedCause(kind="lease_already_started"), "A lease was already started."),
    (FixedCause(kind="request_expired"), "The request's expiry has passed."),
    # New with the causes: a renewal that is refused raises rather than answers False.
    (FixedCause(kind="lease_held_elsewhere"), "Another process holds the lease."),
    (FixedCause(kind="renewal_failed"), "The lease could not be renewed."),
]


@pytest.mark.parametrize(("cause", "sentence"), SENTENCES, ids=lambda value: str(value)[:40])
def test_each_cause_speaks_its_sentence_on_the_verdict_and_the_error(cause, sentence):
    assert cause.sentence() == sentence
    refused = Refused(reason="auth_channel", target="va", kind=None, cause=cause)
    assert refused.detail == sentence
    error = AuthorizationRefusedError(VerdictReason.AUTH_CHANNEL, cause)
    assert error.detail == sentence
    assert str(error) == f"{VerdictReason.AUTH_CHANNEL.phrase} {sentence}"


def test_the_table_names_every_kind():
    kinds = {cause.kind for cause, _sentence in SENTENCES}
    payload_kinds = {
        get_args(model.model_fields["kind"].annotation)[0]
        for model in (
            ChannelNotListed,
            ValueOutOfRange,
            WrongMachine,
            StoppedBy,
            BudgetSpent,
            OverlapHeld,
            _SessionHeld,
        )
    }
    fixed_kinds = set(get_args(FixedCause.model_fields["kind"].annotation))
    assert kinds == payload_kinds | fixed_kinds


def test_a_range_cause_holds_the_bounds_the_range_holds():
    bound = ChannelRange(min=0, max=1.5)
    cause = ValueOutOfRange(address=ADDRESS, min=bound.min, max=bound.max)
    assert cause.sentence() == f"Channel: {ADDRESS}; allowed range [{bound.min}, {bound.max}]."


def test_a_cause_is_not_part_of_the_decision():
    held = Refused(
        reason="session_held", target=None, kind=None, cause=_SessionHeld(sources=("x",))
    )
    bare = Refused(reason="session_held", target=None, kind=None)
    assert held == bare
    assert hash(held) == hash(bare)
    assert bare.detail is None


def test_a_cause_is_read_by_its_kind():
    refused = Refused.model_validate(
        {
            "reason": "auth_range",
            "target": "va",
            "kind": None,
            "cause": ValueOutOfRange(address=ADDRESS, min=0.0, max=1.0),
        }
    )
    assert isinstance(refused.cause, ValueOutOfRange)
    with pytest.raises(ValidationError):
        FixedCause(kind="not_a_kind")


# -- the verdict: allowed, or refused for one reason ---------------------------


def test_an_allowed_verdict_carries_no_reason():
    allowed = Allowed(target="va", kind="simulator", position="asked")
    assert (allowed.allowed, allowed.reason, allowed.cause, allowed.detail) == (
        True,
        None,
        None,
        None,
    )
    with pytest.raises(ValidationError):
        Allowed(target="va", kind=None, position=None, reason="switch_off")
    with pytest.raises(ValidationError):
        Allowed(target="va", kind=None, position=None, allowed=False)


def test_a_refused_verdict_needs_a_reason_word():
    refused = Refused(reason="switch_off", target="va", kind=None)
    assert refused.reason is VerdictReason.SWITCH_OFF
    assert (refused.allowed, refused.position) == (False, None)
    for missing in ({}, {"reason": None}, {"reason": "not_a_reason"}):
        with pytest.raises(ValidationError):
            Refused(target="va", kind=None, **missing)
    with pytest.raises(ValidationError):
        Refused(reason="switch_off", target="va", kind=None, allowed=True)
    with pytest.raises(ValidationError):
        Refused(reason="switch_off", target="va", kind=None, position="asked")


def test_an_allowed_and_a_refused_verdict_never_compare_equal():
    allowed = Allowed(target=None, kind=None, position=None)
    refused = Refused(reason="switch_off", target=None, kind=None)
    assert allowed != refused
    assert isinstance(allowed, posture_store.WriteVerdict)
    assert isinstance(refused, posture_store.WriteVerdict)


# -- the vocabulary -------------------------------------------------------------


def test_the_reasons_are_the_members_in_their_order():
    assert REASONS == tuple(VerdictReason)
    assert [reason.value for reason in REASONS][:3] == [
        "readonly_run",
        "tier_never",
        "session_held",
    ]
    assert all(reason == reason.value for reason in REASONS)


def test_a_write_result_carries_a_verdict_reason_only_on_writes_disabled():
    blocked = ChannelWriteResult.refused(
        ADDRESS, 1.0, RefusalReason.WRITES_DISABLED, "blocked", verdict_reason="auth_range"
    )
    assert blocked.verdict_reason is VerdictReason.AUTH_RANGE
    with pytest.raises(ValidationError, match="verdict_reason must be None"):
        ChannelWriteResult.refused(
            ADDRESS, 1.0, RefusalReason.VALIDATION_ERROR, "bad", verdict_reason="auth_range"
        )
    with pytest.raises(ValidationError, match="unknown refusal reason"):
        ChannelWriteResult.refused(
            ADDRESS, 1.0, RefusalReason.WRITES_DISABLED, "blocked", verdict_reason="off"
        )


def test_a_blocked_write_error_takes_the_word_of_its_verdict_reason():
    error = ChannelWriteBlockedError(ADDRESS, "WRITES_DISABLED", verdict_reason="switch_off")
    assert error.verdict_reason is VerdictReason.SWITCH_OFF
    assert ChannelWriteBlockedError(ADDRESS, "LIMITS").verdict_reason is None
    with pytest.raises(ValueError):
        ChannelWriteBlockedError(ADDRESS, "WRITES_DISABLED", verdict_reason="off")


# -- open_flag says what it did -------------------------------------------------


def _open(root, flag_id: str = "f1", prompt_ref: str = "p1") -> str:
    return session_hold.open_flag(
        "session-1",
        flag_id=flag_id,
        source="watchdog:a",
        reason="r",
        prompt_ref=prompt_ref,
        prompt_excerpt="e",
        root=root,
    )


def test_open_flag_answers_each_outcome(tmp_path, monkeypatch):
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert _open(None) == "no_store"
    root = tmp_path / "holds"
    assert _open(root) == "opened"
    assert _open(root, flag_id="f2") == "duplicate"
    assert _open(root, prompt_ref="p2") == "duplicate"
    (record,) = root.rglob(session_hold.RECORD_NAME)
    record.write_text("garbage", encoding="utf-8")
    assert _open(root, flag_id="f3", prompt_ref="p3") == "record_unreadable"
    assert record.read_text(encoding="utf-8") == "garbage"

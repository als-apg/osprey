The refusal reasons are `osprey_connectors.refusals.VerdictReason`, a `StrEnum`
whose members carry `.phrase`; `REASONS` is `tuple(VerdictReason)` in the same
order and `_PHRASES` is gone (read `reason.phrase`). A member equals its word,
so `verdict.reason == "switch_off"` still holds; `refusal_phrase` and
`refusal_text` take a member (a word is still accepted, an unknown one raises
`ValueError`). `RefusalReason` moved from `osprey_connectors.errors` to
`osprey_connectors.refusals`: import it from there.
`ChannelWriteResult.verdict_reason`, `ChannelWriteBlockedError.verdict_reason`,
`AuthorizationRefusedError.reason` and `StoppedSinceError.reason` are
`VerdictReason` members, and a result that sets `verdict_reason` on any refusal
but `WRITES_DISABLED` raises `ValueError`.

`WriteVerdict` is `Allowed | Refused` (both in `osprey_connectors.posture_store`)
instead of one record: build `Allowed(target=..., kind=..., position=...)` or
`Refused(reason=..., target=..., kind=...)`. `Refused` requires its reason and
`Allowed` cannot carry one; both still answer `allowed`, `reason`, `detail`,
`position`, `target`, `kind` and `authorization`. A refusal's `detail` is now
derived from its `cause`, a typed record (`ChannelNotListed`, `ValueOutOfRange`,
`WrongMachine`, `StoppedBy`, `BudgetSpent`, `OverlapHeld`, `FixedCause`, and a
session hold's own) that says which case the reason covers; pass `cause=`
instead of `detail=`. `AuthorizationRefusedError(reason, cause=None)` takes a cause in
place of its detail string, and its `detail` is the cause's sentence.

`authorized_write_verdict` and `authorized_execution_verdict` take a required
`authorization: str`; a process that names none asks `write_verdict` or
`execution_verdict` itself. `authorized_execution_verdict(..., spend=True)` is
`spend_authorized_write(...)`. `write_authorization.named_authorization()` is
gone: the control-system connectors read `OSPREY_WRITE_AUTHORIZATION` and
nothing else does.

`renew_lease` returns the renewed `Ledger` and raises
`AuthorizationRefusedError` instead of returning `False`: the grant's own
reason when it no longer stands, otherwise `auth_lease_lapsed` with its cause.
Write `try: renew_lease(id) except AuthorizationRefusedError: stop`.
`session_hold.open_flag` returns `"opened"`, `"no_store"`, `"duplicate"` or
`"record_unreadable"` instead of a bool; compare with
`"opened"`.

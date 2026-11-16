`ControlSystemConnector.write_channel`, `write_multiple_channels` and
`write_channel_checked` are final: a subclass that defines one raises
`TypeError` when its class is created. Implement
`async def _send(self, address, value, *, timeout_s, wait) -> SendReport`
instead (`osprey_connectors.control_system.SendReport`: no `fault` when the
value went out, else `not_sent`, `not_taken`, `unacknowledged`, `cs_refused` or
`unwritable` with the transport's words in `detail`); the base class runs the
write guard, the limits, the confirming read and the comparison, and builds
every result. The optional hooks are `_precheck` (was `_write_precheck`),
`_confirming_read(address, sent, *, timeout_s)`, `_run_check`, `_bound`,
`_current_values` and `_send_batch`. `get_metadata` and `validate_channel` are
no longer abstract (defaults: a read's metadata, a read that does not raise).
`subscribe` must take `on_connection`: the deprecated wrapper for one that does
not is gone.

A write the limits refuse is a `refused` result with reason `LIMITS` and the new
`limits_violation` (`osprey_connectors.errors.LimitsViolation`: `kind`,
`reason`, `min_value`, `max_value`, `max_step`, `current_value`) on a single write
as in a batch; `write_channel` no longer raises `ChannelLimitsViolationError`.
`write_channel_checked` raises `ChannelWriteBlockedError` whose `violation` is
that record; read it instead of `__cause__`.

`ChannelWriteBlockedError(address, reason, *, message=None, verdict_reason=None, violation=None)`:
`reason` is a required `RefusalReason` (its word is accepted; any other raises
`ValueError`) and `message` is keyword-only. Use `exc.reason.refuser` and
`exc.reason.remedy`; the `refuser`/`remedy` classmethods are gone.
`ChannelWriteFailedError(address, outcome, *, message=None, value_written=None, observed_value=None)`:
`outcome` is the `WriteOutcome` (`failed`, `mismatch` or `unconfirmed`) and the
uppercase `reason` is gone; compare `exc.outcome is WriteOutcome.FAILED`.
`ChannelNotFoundError.reason` and `ClientLibraryRefusedError.reason` are gone;
the refusal reason of an unknown channel is `RefusalReason.CHANNEL_NOT_FOUND`.

A connector that never loaded a limits validator is now checked against the one
the loaded config gives its type. A DOOCS or Tango limits check that does not
finish within the write's timeout refuses the write with nothing sent (it used
to share the set's bound and report `unconfirmed`), and a single Tango write to
an address that names no device and attribute is refused (`VALIDATION_ERROR`)
instead of raising `ValueError`. Write results are worded the same on every
connector, so some `error_message` and `notes` texts changed. The connector-host
frame protocol is version 3; a parent and child must both run this release.

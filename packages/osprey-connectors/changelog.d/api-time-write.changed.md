Every instant on the write path is a timezone-aware `datetime` in UTC, named
`*_at`. In `osprey_connectors.write_authorization`: `Approver.approved_at`,
`Revocation.revoked_at`, `LedgerStep.spent_at`, `Grant.issued_at` and
`expires_at`, `WriteRequest.requested_at` and `expires_at`, and
`Ledger.lease_expires_at` replace the `*_at_ns` / `lease_until_ns` integers.
`write_request()` and `self_issue()` take `expires_at=` (a `datetime`, for
example `datetime.now(UTC) + timedelta(hours=2)`) instead of `expires_at_ns=`,
and `grant_standing()`, `prune_grants()`, `Grant.live_at()`,
`control_context.blocking_pids()` and `converged()` take `now=` as a
`datetime` instead of `now_ns=` nanoseconds or epoch seconds.
`session_hold.HoldFlag.opened_at` (`None` for the unreadable-record flag),
`control_context.ServerReport.updated_at` and the `requested_at` that
`control_context.terminus()` takes are `datetime` too.

A naive `datetime` is refused with the new
`osprey_connectors.errors.NaiveDatetimeError` (a `ValueError`); no zone is
guessed. `osprey_connectors.model.Instant` is the field type and
`parse_instant()` the reader (`Z` or a numeric offset).

Files store ISO-8601 with `+00:00`. A grant, ledger or request written by an
earlier version (integer nanoseconds) can no longer be read, so it refuses
(`auth_unreadable`): ask for and approve a new authorization after upgrading.
A session-hold record whose stamps are not instants holds the session until an
operator answers it. A switch bound (`expires_at`) with no UTC offset is no
longer read as UTC: it is no bound, and the swap keeps blocking.

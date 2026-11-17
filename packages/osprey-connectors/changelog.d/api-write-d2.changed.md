**Breaking change:** the write-authorization records validate when they are
built, not only when a file is read: a `Grant`, `WriteRequest`, `Ledger`,
`LedgerStep`, `Approver`, `Revocation` or `ChannelRange` that breaks a rule a
reader applies (a blank or space-padded login, a non-hex id, `steps` below one,
a range with a non-finite bound or `min` above `max`, an expiry not after the
issue) raises `ValidationError` at construction. `write_request` and
`self_issue` take `channels: Mapping[str, ChannelRange]`; build a range with
`ChannelRange.of(lo, hi)` instead of passing a `(lo, hi)` tuple.
`revoke_grants(by=...)` with a name that is not a login raises instead of writing
a grant nobody can read.

Renamed, with the stored files unchanged: `Approver.identity` is `Approver.by`
(stored as `identity`), `Ledger.steps` is `Ledger.spent` (stored as `steps`),
and `GrantStanding.used` is `GrantStanding.steps_spent`. `WriteRequest` and
`Grant` share their terms; `WriteRequest.grant()` copies them.

One read shape: `control_context.read_record_status()` returns a
`ReadResult` (`status`, `value`, `path`) instead of a `(record, status)` tuple;
write `read.value` and `read.status`. `list_requests()` returns a `Listing`
(`items`, `unreadable`). `listed_grants` is gone: `list_grants(*directories,
self_issued=())` lists grants directories and the self-issued grants of ledger
directories, each once, as a `Listing` of `ListedGrant` (`grant`, `directory`),
and reports a directory it cannot list in `unreadable` instead of raising
`OSError`. `ReadStatus`, `ReadResult` and `Listing` are in
`osprey_connectors.model`; `identity.is_path_segment` is the one rule for a name
used as a directory, and `write_authorization.Login` the type of a grant's,
request's, approval's and stop's login.

Every read-only service method answers its value or `Unavailable`, and
`Unavailable` now requires a `cause` beside its `reason`: one of
`UnavailableCause` (`unreachable`, `timeout`, `bad_reply`, `refused`,
`misconfigured`, `not_ready`, `unsupported`, `failed`). Branch on `cause`, never
on the `reason` text, which is unchanged. Write `Unavailable(cause="timeout",
reason="...")` where you wrote `Unavailable(reason="...")`.

Pages no longer carry an unavailable outcome. `ChannelDirectory.search` returns
`DirectoryPage | Unavailable`, and `AlarmLog.history` and `PutLog.history`
return `LogPage[...] | Unavailable`; check `isinstance(answer, Unavailable)`
instead of `page.outcome is Outcome.UNAVAILABLE`. `Outcome` is gone: a page's
`outcome` is the `PageOutcome` word `"found"` or `"no_match"`, and the text that
says why a page has no rows is `note` (was `reason`). `HistoryPage` is now
`Page` (the shared fields) and `LogPage` (adds `retention_known`), and
`DirectoryPage` is a `Page[DirectoryEntry]` with `searched`, `request` and
`total: int | Unavailable`, which replaces `total: int | None` and the count
entry of `partial`; `partial`, `facets` and `facets_reason` are no longer
contract fields. `history` takes `start_at` and `limit` as keywords:
`history(address, start_at=..., limit=..., timeout_s=...)`.

An exact lookup that finds nothing answers `NotListed(address=...)`:
`describe` returns it instead of `Outcome.NOT_LISTED`, and
`AlarmState.state_for` instead of `None`.

`AlarmSeverity` (`OK` < `MINOR` < `MAJOR` < `INVALID` < `UNDEFINED`, with
`.rank` and `AlarmSeverity.of_recorded(word)`) replaces `severity_rank`.
`AlarmCondition.severity` and `AlarmLogEntry.severity` hold its members,
built from the member or its word; a word that is no severity now reads as
`UNDEFINED` instead of being kept. `AlarmLogEntry` gains `acknowledged`: a
logged `MAJOR_ACK` is `severity=MAJOR, acknowledged=True`, as on
`AlarmCondition`.

`ServiceKind` gains `"archive"`, and every backend has the `start()` and
`aclose()` pair, doing nothing unless the backend overrides it.

The framework reads the connectors' typed refusal vocabulary: a write verdict is
`Allowed` or `Refused`, and the framework's own fallback refusals are built as
`Refused`. The words, phrases and `[osprey-refusal: <reason>]` trailer every
surface serves are unchanged. The optimizer ends a run whose lease cannot be
renewed on the reason the renewal raises; when a ledger directory or lock
failure coincides with a grant that no longer stands, that reason now reads
`auth_lease_lapsed` instead of the grant's own.

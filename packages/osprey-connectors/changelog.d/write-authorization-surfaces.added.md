A stop through `control_context.set_switch()` (named machines, or `"all"`) now
marks every program authorization on the stopped machines revoked inside the
record lock, so the holder's next write refuses `auth_revoked`; `set_switch()`
takes `by=` to name whose stop it was. `osprey_connectors.write_authorization`
gains the holder's request (`write_request()`, `parse_request()`,
`list_requests()`, `withdraw_request()`, `OSPREY_WRITE_REQUESTS_DIR`), the
person's approval (`approve_request()`, which issues the grant under the
request's id, and `approved_grant()`, which a holder waits on), `prune_grants()`,
`grant_in_force()`, `read_ledger()`, and `listed_grants()` and
`grant_standing()`, which every listing shares: a grant whose lease lapsed or
whose steps are spent is shown ended, with its steps used. An approval or a
self-issue refuses `auth_overlap` against another holder's live grant in the
owner's own directory and the holders' self-issued grants, seen by an
approver through its read-only view of their ledger directory
(`OSPREY_WRITE_LEDGER_VIEW_DIR`, `self_issued_directories()`,
`self_issued_grants()`); a grant file there that cannot be read is logged and
left out, never a veto. `grant_in_force()` holds a self-issued grant in force
only while its switch stands at `allowed`. `control_context.held_epoch()`
holds the record lock around a writer that stamps the record's epoch. A stop
whose revocation marks cannot be written moves the record's epoch instead,
ending every grant of that owner, and the authorizations locks are created
group-writable (0660) so every uid sharing a grants directory can take them.

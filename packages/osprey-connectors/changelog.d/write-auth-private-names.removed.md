`osprey_connectors.write_authorization` no longer lists the helpers nothing
outside the module calls. `authorization_verdict()`, `spend_step()` and
`read_ledger()` are gone: a write path takes `authorized_execution_verdict()`,
and a listing takes `grant_standing()`. `auth_overlap()`, `grant_in_force()`,
`issue_grant()`, `parse_grant()`, `parse_ledger()`, `parse_request()`,
`pid_namespace()`, `process_start_time()`, `self_issued_grants()`,
`SCHEMA_VERSION`, `SURFACES`, `APPROVAL_SURFACES`, `COMPLETIONS` and
`PRUNE_AFTER_S` are private. No framework release uses any of them.

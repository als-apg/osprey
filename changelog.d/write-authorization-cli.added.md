`osprey write-authorization approve <request> | list | revoke <id>` approves,
lists and revokes programs' bounded write authorizations without a web
terminal: inside a deployment repo against the control-context tree (any
owner), outside one against this account's own state directory. An approval
checks the request's target against the configured control targets, the
owner's tier (judged as the program's own writes will be: the staged login map
for a program in a shared container, the rendered config for one on the host)
and the limits database as the web terminal's card does, and
records `approver.identity` as the acting identity with `surface: "cli"`;
`list` prunes grants that ended more than an hour ago, shows each one's steps
used of its total (one whose steps are spent or whose lease lapsed is not in
force) and the programs' self-issued grants from `var/write_ledger`, which
`revoke` also reaches. `osprey status` adds a table of the programs'
authorizations in force under Writes: owner, holder (program), target, steps
used of total, time to expiry and lease. Every surface also finds the request and the ledger of
a program run on the host with no ledger bound, which keeps them in its
owner's own `authorizations` directory.

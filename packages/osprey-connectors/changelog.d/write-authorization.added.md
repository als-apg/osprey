`osprey_connectors.write_authorization` defines a program's bounded write
authorization: a grant naming one machine, its owner and holder, a range per
channel, a step budget, an expiry and a renewed lease, with a ledger the
holder and its connector child keep, one step per write call. Its readers fail
closed, and `authorization_verdict()`, `spend_step()` and
`authorized_execution_verdict()` judge a whole batch against it. `REASONS`
gains `auth_unreadable`, `auth_holder`, `auth_channel`, `auth_range`,
`auth_exhausted`, `auth_lease_lapsed`, `auth_expired`, `auth_revoked` and
`auth_overlap`, and `WriteVerdict` gains `authorization`.

A process that names an authorization (`OSPREY_WRITE_AUTHORIZATION`) has every
write call checked against it by the connector's write guard: listed channels,
values in range, a step left, a live lease, no stop. One call is one step,
spent before anything is put; a batch — the base class's, which a connector's
own `write_multiple_channels` gives way to under an authorization — is checked
whole first, so a policy refusal writes nothing, and its per-channel writes
are not checked again. The machine's switch may stay off, and turning it on
does not lift the bounds. A process that names no authorization is checked
exactly as before. The bounds hold a program that runs as written: the budget
and the lease are the holder's own record, and the gateway is the boundary.

`ConnectorHostPool.connector()` takes `owner=` and `authorization=`; a key that
names either is `(target, execution_mode, owner, authorization)`, its child gets
`OSPREY_CONTROL_OWNER` and `OSPREY_WRITE_AUTHORIZATION`, selects the write
gateway when the authorization allows, and reports it as `authorization`. A
child of such a key that came up before its authorization could write refuses
`switching`, spending nothing, and that refusal comes back like any other.
`authorized_write_verdict()` is the answer a gateway is chosen on.

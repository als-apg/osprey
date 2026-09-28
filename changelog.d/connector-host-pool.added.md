`osprey_connectors.ipc.pool.ConnectorHostPool` lets one process drive several
control-system targets at once, including targets that serve the same channel
names. It runs one connector-host child per `(target, execution_mode)`, so
`live`, a read-only `live` and `standin` can sit side by side. Each child is
started on first use. Before any call reaches it, the pool checks the child's
endpoint and write posture against what the config derives.

The pool refuses some targets before it spawns anything
(`ConnectorHostStartError`, stage `"config"`):
- a Channel Access block (`epics`, `virtual_accelerator`, `live_standin`) with
  no gateway for the role the run selects, whose child would search by
  broadcast;
- a `standin` or `va` target whose selected gateway has the same address and
  port as one of the `live` target's gateways.

Failed calls are never retried:
- A child that dies with a call in flight fails that call with
  `ConnectorHostLostError`, and the next call gets a fresh child.
- A child that misses a call's deadline is pinged first. One that still
  answers is left running, and the call raises `TimeoutError`.
- A child that doesn't answer the ping is killed, and the call raises
  `ConnectorHostUnresponsiveError`, which is also a `TimeoutError`.
- With `kill_on_write_timeout=True`, a write that misses its deadline kills
  its child even if it answers the ping, and raises
  `ConnectorHostWriteTimeoutError` (a `TimeoutError`). The write may or may
  not have landed, but the child can no longer send it after a newer write.
  With the flag off (the default), a timed-out write may still land later.

The pool never loads a Channel Access or pvAccess client itself.

The connector-host proxy and child now also serve `validate_channel`, and the
proxy offers `write_channel_checked`. The endpoint derivation and post-connect
verification behind the target switch have moved to
`osprey_connectors.ipc.verification`, so the switch and the pool share one
check.

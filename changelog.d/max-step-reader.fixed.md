`max_step` now works on every control system, not only on EPICS Channel
Access. The check reads a channel's present value through the connector doing
the write — the simulator's store, DOOCS, TANGO, or the EPICS client the
deployment configured — instead of a direct pyepics call that ignored which
control system the write was bound for. On a Channel Access deployment the
step read now follows the writing connector's own client rather than the
process-wide `EPICS_CA_*` environment. A caller with no way to read the
channel still has its write refused.

The same read reaches the other write paths: `osprey.runtime.write_channel`
asks its connector for it, and a script writing through the sandbox measures
the step over its own Tango proxy, `doocs4py` or caproto client. The
`channel_write` approval hook, which holds no client of its own, applies every
other limit and leaves the step to the connector that performs the write.

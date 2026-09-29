**Breaking change:** raw client puts from the Python executor and from
notebook kernels are now refused in readwrite runs too, raising
`ChannelWriteBlockedError` with reason `RAW_CLIENT_WRITE`. This covers
`caput`, `PV.put`, caproto, DOOCS and PyTango writes, ophyd-async signals written
directly in a kernel, and `PV(..., monitor_delta=…)`, which writes `.MDEL`.
A RunEngine driving ophyd or ophyd-async devices inside a notebook or execute
script is refused the same way; submit the plan to a Bluesky lane queue
instead, where plans run unchanged. Replace raw puts with
`osprey.runtime.write_channel` or `write_channels`, as described in
`docs/source/architecture/python-executor.rst` and
`docs/source/how-to/web-terminal/notebooks.rst`. Reads are unchanged.
PVAccess puts (p4p `Context.put`, pvaPy `Channel.put`) are the exception: the
connector does not write PVAccess yet, so they keep their approval ask and,
with limits checking on, their limits check, as before.

`osprey.runtime` gains `read_channels`, which reads several channels in one
connector call, `channel_limits`, which returns a channel's limits record, and
`execution_deadline`, which returns the time the executor kills the run. A
batched read that leaves any channel without a value raises
`ChannelReadFailedError` naming every failed channel and the error each one
raised. A cancelled `execute` or `execute_file` run gets `SIGINT` and the time
left to its deadline to finish before it is killed, and both tools run one
shared gate sequence. A read-write run is reported as write activity even when
no write pattern is detected. The EPICS connector re-types a channel whose
request type was fixed before it connected. A channel limits record must state
`writable`.

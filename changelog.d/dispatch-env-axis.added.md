The event dispatcher and its workers honour the `env:` passthrough every other
service already did, declared once as `dispatch.env` and written into both
halves. A deployment whose channels sit behind a gateway can now name
`EPICS_CA_ADDR_LIST` and `EPICS_CA_NAME_SERVERS` there and have the pair reach
it; before, the two containers were the only ones the axis could not reach.

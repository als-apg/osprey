The approval hook now refuses an `execute` call in a readwrite run whose
code spells a raw client put — `caput(`, `caput_many(`, `epics.caput(`,
`PV(...).put(`, `aioca.caput(`, `write_door` or `open_door` —
before any human is asked to approve it. The hook answers with a deny that
names `osprey.runtime.write_channel`, and files a refused audit record with
reason `raw_client_write`. Comments and string literals do not count.
`write_channel`, `write_channels`, a bare `.put(`, the PVAccess puts (p4p
`ctxt.put(`, pvaPy `Channel.put`) and `execute_file` are asked about as
before.

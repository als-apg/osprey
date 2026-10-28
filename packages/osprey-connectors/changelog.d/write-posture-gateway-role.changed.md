The EPICS connector and the connector-host pool choose the write gateway from
the write verdict: a connector whose writes are allowed routes through
`gateways.write_access`, otherwise it stays on `read_only`. A write allowed
after the gateway was chosen is refused as `switching` until the connector
reconnects. A machine with only a write gateway reads through it while its
writes are off, refused in software; a read-only run on such a machine
refuses to connect.

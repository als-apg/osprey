A TANGO Controls connector now ships in-tree: `control_system.type: tango`
reads and writes TANGO device attributes through PyTango, addressed as
`domain/family/member/attribute`. Attribute quality is reported as the
channel's alarm state, `DevEnum` attributes read with their state labels, and
writes confirm with one fresh read like every other connector. PyTango is
imported at `connect()`, so the type registers everywhere and fails with a
clear `ImportError` only where a TANGO environment is genuinely absent.

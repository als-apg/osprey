The channel roster's build-side package neither pairs a setpoint with a
readback by address grammar nor reads a channel-finder database: `pairing.py`
and its `assign_readbacks`, `osprey.channel_roster.database` and its
`read_database_roster`, `RosterSourceKind.DATABASE` and the `NO_SOURCE`,
`GRAPH_NO_TTL`, `GRAPH_MALFORMED` and `DIRECTION_UNDERIVABLE` absence reasons
are deleted, with the virtual-accelerator manifest builder's
`va_graph_deferred` and `GRAPH_SOURCE_PARADIGM`. The roster's graph reader
stays. The retired-name guard holds the deleted names out of the tree.

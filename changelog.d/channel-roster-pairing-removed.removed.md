The channel roster no longer pairs a setpoint with a readback by address
grammar: a setpoint's readback is the one its facility-file channel record
states in `pair`. `osprey.channel_roster.pairing` and its `assign_readbacks`
are removed.

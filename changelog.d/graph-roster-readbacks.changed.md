Setpoint/readback pairing in the channel roster now keeps a readback the
source itself stated and applies the `:SP` -> `:RB` address grammar only to
the records the source paired nothing with. Before, the graph reader
assigned no readbacks at all and the grammar was the only pairing, so a
facility whose addresses carry no `:SP`/`:RB` suffix had every plan device
read its own setpoint back. On such a facility the bluesky plan devices now
carry the corpus-stated readback.

The graph reader also yields one record per address: bindings sharing a
`fullPv` (a channel the corpus hangs under several devices) are one channel,
with the direction their bindings agree on and no direction when they
disagree. Before, each binding was a record of its own, and a derived
plan-device document for such a corpus named the same device several times,
which the build refuses.

`osprey channel-finder preview --db-path` now recognises a middle-layer
database by its shape — nested groups ending in a `ChannelNames` list — instead
of by a fixed list of the demo machine's system codes. A middle-layer database
whose top-level groups are named for your own machine previews as one rather
than being rendered as an in-context file.

`osprey channel-finder build-database` now honours the CSV's `address` column
as the family's address pattern whenever it holds a placeholder, so a machine
whose addresses are not `<family><NN><suffix>` keeps its own shape instead of
having one synthesised from the family name. Rows of one family that give
different addresses, and patterns naming a placeholder the expander cannot fill
in, stop the build and name the family, rather than being written into the
database as the literal pattern text. The `instances` column now also accepts an explicit
range (`4-11`) for device numbering that does not start at one.

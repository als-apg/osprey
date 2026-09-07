`osprey channel-finder preview --db-path` now recognises a middle-layer
database by its shape — nested groups ending in a `ChannelNames` list — instead
of by a fixed list of the demo machine's system codes. A middle-layer database
whose top-level groups are named for your own machine previews as one rather
than being rendered as an in-context file.

`osprey build` writes `data/channel_finder/middle_layer.json` from the facility
file when a render selects the middle-layer pipeline, under
`"schema": "osprey.facility.channel_finder/1"`, and writes the DuckDB copy
`run_sql` queries beside it as `data/channel_finder/middle_layer.duckdb`. A
System is a top place and a Family is a group that carries `signals`, named by
its id less a leading `<System>/`; each Field lists one channel per member in
`CommonNames` order, so `ChannelNames` lines up with `DeviceList`, and carries
the family's sentence for that field. A channel whose field some member lacks
or holds twice is its own Field keyed by its address; a channel of no family is
left out, and the build names both counts in one note. A facility with no group
carrying `signals` stops the build with `view-unsupported`, as does a host where
DuckDB cannot load its full-text-search extension.
`channel_finder.pipelines.middle_layer.database.path` and `duckdb_path` render
to the two files, and `channel_finder.pipelines.middle_layer.database.type` is
gone. A hierarchical selection now stops with `view-unsupported` when a tree
key begins with `_`.
The middle-layer terminology table names each family as the index files and
names it, with its System where a class spans several.

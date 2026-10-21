Every channel-finder index `osprey build` writes (`data/channel_finder/in_context.json`,
`hierarchical.json`, `middle_layer.json`) states its row count as `count` beside `schema`: the
rows of an in_context index, the leaves of a hierarchical tree, the distinct addresses a
middle-layer index lists. `osprey channel-finder benchmark` reads that key for the run's
`channel_count` instead of re-parsing the index, and takes the count before it sends a query: an
index that is missing or states no `count` stops the run with an error naming the file, where it
used to record 0. A top place whose id is `count` stops a middle-layer build with
`view-unsupported`, as `schema` does.

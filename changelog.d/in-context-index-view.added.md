`osprey build` writes `data/channel_finder/in_context.json` from the facility
file when a render selects the in_context pipeline: one row per channel tagged
`in_context`, sorted by address, named by the channel's first name (else its
address), with its address and description, under
`"schema": "osprey.facility.channel_finder/1"`. The in_context pipeline loads
that file with the flat loader, and `channel_finder.pipelines.in_context.database.path`
renders to it. The `channel_finder.pipelines.in_context.database.type` key is
gone. A render that selects in_context while no channel is tagged stops with
`facility: view-unsupported`.
A render that selects another pipeline writes no in_context index and prints no
line about it.

`osprey build` writes `data/channel_finder/hierarchical.json` from the facility
file when a render selects the hierarchical pipeline, under
`"schema": "osprey.facility.channel_finder/1"`. Its levels are the facility's
place level words, then class, device and leaf; every channel sits at the same
depth, and an absent place, class or device is the node `-`. Each leaf spells
its channel's full address and carries the sentence its device's family group
keeps for that kind of signal, else the channel's own description. The
hierarchical pipeline loads that file, and
`channel_finder.pipelines.hierarchical.database.path` renders to it. The keys
`channel_finder.pipelines.hierarchical.database.type` and
`channel_finder.pipelines.in_context.database.presentation_mode` are gone. A
render that selects another pipeline writes no hierarchical index and prints no
line about it.

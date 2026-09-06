Graph-mode deployments now answer channel searches from a search index built
from the facility corpus at build time, rather than by parsing the corpus at
runtime, so the channel explorer, the channel roster and the OSPREY agent's new
`search_channels` tool answer in milliseconds at any corpus size. `osprey
build` writes the index to `services.graphdb.index_path`
(`data/channel_databases/graph.duckdb` by default); rebuild it after editing
the corpus with `osprey knowledge build-index`. `osprey health` gains
`channel_finder_seed` and `channel_finder_search_index`, which report the
corpus the store was seeded from and the index built beside it, and warn when
the two have drifted apart.

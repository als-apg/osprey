Every render with a `services.graphdb` block seeds its store from the graph
view the build writes, `./data/graph/facility.ttl`, and ships the search index
`data/channel_databases/graph.duckdb` derived from it, whatever the
channel-finder mode. The index's bindings carry each device's place path,
s position in metres and ordinal in its place (index schema version 2), and
the deploy refuses a rendered graph store block that names no corpus.

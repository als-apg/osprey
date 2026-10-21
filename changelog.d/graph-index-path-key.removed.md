`services.graphdb.index_path` is removed: the search index always sits at
`data/channel_databases/graph.duckdb` under the render. The roster's graph
reader and the index's `channels` table are deleted with it; the channel
roster reads the facility file only.

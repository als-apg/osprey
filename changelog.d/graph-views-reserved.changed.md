The graph view under `data/graph/` and the graph index at
`data/channel_databases/graph.duckdb` are build output: a `project/` mirror file
at either path stops the build with a `profile-invalid` line, and no agent-side
writer may write either path.

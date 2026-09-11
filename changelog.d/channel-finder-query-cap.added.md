The number of rows the middle-layer `run_sql` tool hands back is now a config
key, `channel_finder.query_max_rows` (default 500, the previous fixed cap). A
truncated answer now says which key cut it and at what number, so the agent
narrows the query rather than presenting a partial list as complete.

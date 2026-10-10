The middle-layer channel-finder import now loads its DuckDB tables in bulk
instead of one row at a time, so building a channel-finder index for a large
middle layer takes seconds rather than minutes. The imported rows and the
reported counts are unchanged.

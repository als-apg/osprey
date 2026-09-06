ARIEL's `sql_query` tool now refuses a query that reads no allowlisted table.
The table allowlist only ever ran over `FROM` and `JOIN` targets, so a query
with neither — `SELECT pg_read_file(...)`, say — had nothing to check and
passed; the read-only transaction around it stops writes, not server-side file
reads.

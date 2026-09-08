ARIEL's `sql_query` tool now refuses a `FROM`/`JOIN` shape its table allowlist
cannot resolve: a comma-separated FROM list (`FROM enhanced_entries, pg_shadow`),
a parenthesised join in place of a subquery, and any quoted identifier, comment,
dollar quoting or backslash anywhere in the query — each of which reached a
table the allowlist never checked. A `TABLE name` command inside a `WITH`
body, a set operation or a subquery is checked against the allowlist like a
`FROM` target. A multi-CTE `WITH a AS (...), b AS (...)` query, which was
refused by mistake, is now accepted.

A name counts as a CTE only where the query really declares one, and only where
Postgres would see it: inside a `WITH` list, outside any string literal, after
its body closes, and for references in the query that declared it. A literal
spelling a CTE declaration, an alias in a `WINDOW` list, or a `WITH` inside a
subquery no longer hides a joined table from the allowlist.

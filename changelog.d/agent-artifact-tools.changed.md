**Breaking change:** `artifact_list` lists the current session's artifacts
(`scope="all"` for every session), drops its `source_agent` and `artifact_type`
filters, and returns 20 rows unless `last_n` (1–200) says otherwise.
`artifact_delete_all("everything")` deletes only the current session's
artifacts. `artifact_read` returns a JSON artifact's content parsed.

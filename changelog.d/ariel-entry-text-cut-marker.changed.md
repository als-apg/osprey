An ARIEL entry whose text a search, `browse` or `entries_by_ids` result cuts
short now says so: it carries `raw_text_truncated: true` and its full length
in `raw_text_length`, and `entry_get` returns the whole entry.

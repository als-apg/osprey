The `session_log` tool now compares its `since` and `before` bounds with event
times as instants, so a bound written with a UTC offset selects exactly the
events at or after (or at or before) that moment. A bound without an offset is
read as UTC.

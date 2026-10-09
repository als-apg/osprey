Asked to open "the entry" after a logbook subagent read the answer off one of
its pictures, the agent now opens it with that picture enlarged. The logbook
subagents hand back the entry id and `attachment_id` of every picture they
viewed; `entry_open` takes `attachment_id` as a required field that may be
null, and lists the entry's viewable pictures when opened without one. The
logbook guidance no longer offers an entry id as a value the entry records, and
views an uncaptioned picture before reporting an answer missing.

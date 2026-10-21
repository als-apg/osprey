The Bluesky `get_run_data` tool now bounds an answer by size as well as by
rows. A row is as wide as the run, so a few hundred rows of a facility-scale
response matrix came back as megabytes of JSON, more than an agent session
accepts in one message. A window that would exceed 64 kB now returns the
rows that fit, with `truncated: true` and a note naming how many rows were
left out and how to read the rest.

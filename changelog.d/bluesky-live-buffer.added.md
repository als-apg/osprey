Two new build-profile keys bound the Bluesky bridge's in-memory run data:
`bluesky.live_max_runs` (default 50) is how many finished runs stay readable,
and `bluesky.live_max_rows_per_run` (default 10000) is how many rows of one run
are stored. Both were fixed in the code before. Rows past the cap are still
counted, so a long run reports its true length over a truncated buffer.

The dispatcher dashboard can clear its run history. A **Clear history** button
beside the Activity filters deletes every finished run — both the persisted
record and the in-memory entry, so the list stays empty after a reload — while
runs still in flight are kept. The button asks twice before it fires, and it
uses the same "is this run finished" rule as the optional `RETENTION_DAYS`
sweep, so the two can never disagree. Artifacts the cleared runs produced are
left alone; ageing those out remains the retention sweep's job.

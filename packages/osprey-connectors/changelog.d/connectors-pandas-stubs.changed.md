The archivers' time-range helpers raise `TypeError` naming the channel when
given a non-empty series that is not indexed by timestamps. An Archiver
Appliance sample whose `secs` or `nanos` is not a number raises the archiver's
`ValueError`, and a MongoDB document whose `date` is not a datetime is skipped
with a warning, as `stored_extent` already documented.

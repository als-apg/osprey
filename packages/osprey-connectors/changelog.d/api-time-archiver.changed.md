Archiver reads and channel readings take and give aware instants. `get_data`
takes its window by keyword, `get_data(addresses, *, start_at=..., end_at=...,
precision_ms=..., timeout_s=..., processing=...)`, instead of the positional
`start_date`/`end_date`. A naive bound is refused with the new
`osprey_connectors.errors.NaiveDatetimeError` (a `ValueError` naming the
parameter); it used to be read as facility wall-clock. Pass an aware datetime,
for example `datetime(2026, 7, 30, 10, tzinfo=ZoneInfo("America/Los_Angeles"))`.
`osprey_connectors.errors.require_aware` is the same check for your own code.

`ArchiverMetadata.archival_start`/`archival_end` are `first_sample_at`/
`last_sample_at`, and `ArchiveExtent.oldest`/`newest` are `oldest_at`/
`newest_at`; all four are aware UTC datetimes (the mock archiver's used to be in
the facility zone, and MongoDB's were naive).

`ChannelValue.timestamp` is `ChannelValue.sampled_at`, an aware datetime, and
every built-in connector (EPICS CA and PVA, DOOCS, Tango, the in-process virtual
accelerator) stamps it in UTC instead of the facility zone. Convert with
`.astimezone(get_facility_timezone())` where you show it. `ChannelMetadata.timestamp`
is gone: it repeated the reading's stamp, so read `ChannelValue.sampled_at`.

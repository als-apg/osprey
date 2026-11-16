The framework calls the connectors' aware-time archiver and reading API:
`archiver_read` and the archiver freshness check pass the window as
`start_at`/`end_at`, and the PV directory reads the archive's
`first_sample_at`/`last_sample_at` and `oldest_at`/`newest_at`. What the
framework serves is unchanged: `channel_read`, the channel finder's live values
and the `channel_write` audit note still show a reading's stamp in the facility
zone, and `archiver_read` still reads an operator's bare wall-clock time as
facility-local. The mock archiver's first and last sample in the PV directory
are now reported in UTC (`+00:00`), as the page always described them.

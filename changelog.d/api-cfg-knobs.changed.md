**Breaking change:** the `va_archiver:` profile keys carry their unit as `_h`
or `_s`: write `hot_span_h`, `hot_cadence_s`, `tail_cadence_s`,
`recorder_cadence_s`, `recorder_tail_cadence_s`, `recorder_poll_s` and
`timeout_s` in place of `hot_span_hours`, `hot_cadence_sec`,
`tail_cadence_sec`, `recorder_cadence_sec`, `recorder_tail_cadence_sec`,
`recorder_poll_sec` and `timeout_sec`; values and units are unchanged. A
profile that still writes an old key, in the block or under `config:`, is
refused at build naming the new key. Rebuild after upgrading: the first deploy
of an existing archive store lists `hot_cadence_s`, `hot_span_h` and
`tail_cadence_s` as changed (`None -> …`) and rebuilds the base series,
discarding recorded history unless `--keep-archiver-base` is passed. The
directory panel's archive configuration is served under its existing names.

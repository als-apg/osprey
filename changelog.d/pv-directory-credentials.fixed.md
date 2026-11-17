A web terminal is handed the credentials its PV directory services name: each
variable under `auth:` in `pv_directory.settings` and in the `alarm_log`,
`alarm_state` and `put_log` blocks, and each CA file under `tls.ca_bundle`.
Before, none reached a deployed terminal, so a directory service behind a
login answered "not set" on every read.

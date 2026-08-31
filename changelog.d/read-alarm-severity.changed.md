`channel_read` now reports `alarm_severity` beside `alarm_status` when metadata
is requested (0 healthy, higher is worse, absent when the control system reports
none) — the same alarm fields a write already carried, so alarm state can be
judged before a write, not only after it.

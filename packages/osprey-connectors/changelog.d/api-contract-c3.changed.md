**Breaking change:** a reading carries its own alarm, label and address, and a
write result carries its confirming re-read.

- `ChannelValue` gains a required `address`, `alarm: Alarm | None` and
  `enum_label`. Read `value.alarm.status` / `value.alarm.severity` instead of
  `value.metadata.alarm_status` / `alarm_severity`, and `value.enum_label`
  instead of `value.metadata.enum_label`. `alarm` is `None` when the control
  system reports no alarm state (DOOCS, a healthy virtual-accelerator channel);
  either field of an `Alarm` is `None` when it was not reported.
- `ChannelMetadata` holds only what describes the channel (units, precision,
  description, the limit pairs, `enum_labels`, `writable`, `raw_metadata`), so
  `get_metadata` no longer reports an alarm.
- `ChannelWriteResult.observed: ChannelValue | None` replaces `observed_value`,
  `alarm_status` and `alarm_severity`: read `result.observed.value` and
  `result.observed.alarm`. Only a `confirmed` or `mismatch` result carries it,
  and building any other result with one raises `ValueError`.
  `observed_number` stays. `ChannelWriteFailedError.observed_value` is unchanged.
- A connector builds readings with the address it was asked for and fills
  `alarm` from its own protocol fields. A TANGO reading no longer repeats its
  severity in `raw_metadata`.
- The connector-host frames carry the new `Alarm` record and a nested
  `observed` reading (protocol version 3).

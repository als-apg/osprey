The seed knobs carry their unit: `SeedKnobs` and `SeedFingerprint` fields
`hot_span_hours`, `hot_cadence_sec` and `tail_cadence_sec` are `hot_span_h`,
`hot_cadence_s` and `tail_cadence_s`, and so are the seed manifest's stored
fingerprint keys. A store seeded before this release compares as `mismatch`
(the three new keys read as `None`) and is reseeded on its next deploy.
`SeedKnobs.from_config` refuses a `va_archiver` block that still writes an old
key with a `ConfigurationError` (a `ValueError`; `key` is the old key) naming the
new one, instead of seeding its default.

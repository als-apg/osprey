`osprey-connectors` now depends on `typing-extensions>=4.4.0`. A simulator
archive manifest whose `seeded_at` is not a timestamp reads as unseeded when
its fingerprint is compared, as the framework already reads it.

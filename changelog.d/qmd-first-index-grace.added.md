How long the qmd sidecar's health check waits out its first full index build is
now a config key, `services.qmd.first_index_grace` (seconds, default 3600 — the
previous fixed value). The sidecar does not open its port until the index
exists and is non-empty, so a corpus that takes longer than the grace period to
index would otherwise report a working container as failed.

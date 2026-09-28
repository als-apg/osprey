`health.disk.min_free_gb` and `health.disk.max_used_percent` set when the
`disk_space` health row warns (defaults 1.0 GB free and 90 % full), so a large
shared volume that runs full by design no longer leaves `osprey health`
degraded. An invalid value is refused at load time and names the key.

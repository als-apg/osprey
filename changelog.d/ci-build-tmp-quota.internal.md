CI's unit and Tier 1 lanes keep their scratch files on the runner's disk
(`TMPDIR=${{ runner.temp }}`) instead of `/tmp`, which on the Ubuntu 26.04
image (`ubuntu-latest` since 2026-10-22) is a RAM-backed tmpfs with a ~6.3 GB
per-user quota that the real-build tests' project venvs overran with
`[Errno 122] Disk quota exceeded`.

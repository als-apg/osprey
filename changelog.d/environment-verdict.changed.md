`scripts/check.py` re-syncs a stale editable install before it runs, and a
machine that cannot judge (too little free disk, Docker storage that fails)
stops with `check: environment: <probe>: <fact>` and exit status 3.

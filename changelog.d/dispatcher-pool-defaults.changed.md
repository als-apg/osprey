A dispatcher run by hand from a `triggers.yml` that omits `max_concurrent_runs`
and `max_queue_depth` now starts at 2 concurrent runs and a queue of 50 — the
pair the documentation describes and the pair a built deployment already
writes. A deployment built by `osprey build` is unaffected; it writes both keys
from the profile.

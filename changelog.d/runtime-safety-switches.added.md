An admin-tier web terminal can turn the safety watchdog off for its own sessions
without a restart, in Settings > Safety (or `POST /api/safety-switches/watchdog`).
The move is audited and names who made it; while it is off, prompts there are not
judged and no new hold opens; a restart turns it back on.

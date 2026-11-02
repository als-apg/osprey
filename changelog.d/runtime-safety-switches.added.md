An admin-tier web terminal can turn the safety watchdog off without a restart,
in Settings > Safety (or `POST /api/safety-switches/watchdog`). Every operator
sees the switch and who turned it off. Each move is audited; while it is off,
prompts are not judged and no new hold opens; a restart turns it back on.

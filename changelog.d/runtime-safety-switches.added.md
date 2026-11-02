An admin-tier web terminal can turn the safety watchdog off at runtime with
`POST /api/safety-switches/watchdog` (`{"state": "off"}` or `"on"`); every operator
can read the state at `GET /api/safety-switches`. Each move is audited, prompts
are not judged and no new hold opens while it is off, and a restart returns to
the profile.

Panel and proxy routes drop branches only test doubles reached: the proxy's retry
for a client without per-request redirect control, the activity routes' ring-less
mode, a discovered-panel URL fallback and a missing-project guard in panel discovery.
`GET /api/panels` reads an absent Config-panel or scaffold-write flag as off, as the gated
routes do.

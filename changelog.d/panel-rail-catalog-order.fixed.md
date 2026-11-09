Web terminal: the panel rail keeps the same order across restarts. `/api/panels`
lists `enabled` and `visible` in the built-in catalog's order instead of a
per-process order that changed on every redeploy.

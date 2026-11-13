The `ariel-standalone` and `channel-finder-standalone` presets, and the
per-card `control-assistant-<name>` presets: the logbook and the channel finder
run as shared logins inside a deployment, and the bundled logins replace the
per-card presets. The `ariel.demo_narrative` key goes with them; `osprey up`
still seeds an empty logbook with the active scenarios' entries.

CI holds `osprey-connectors` to the framework releases it must keep working
under. `connectors-compat` runs the connectors built at HEAD under each release
listed in `[tool.osprey-connectors.compat] frameworks`: it scans the release for
connectors names that no longer resolve, imports every module of it, renders a
deployment with its CLI and opens the simulator view, and reads the
control-context record across its hook and golden files.
`framework-at-connectors-floor` runs the framework at HEAD with the connectors
release its floor names, and a nightly workflow runs each listed release's own
connector tests against HEAD and opens an issue when one fails. Both per-PR
lanes stand down until the cut-over release is listed and the floor reaches the
first independent connectors release.

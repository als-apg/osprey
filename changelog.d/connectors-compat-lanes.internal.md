CI holds `osprey-connectors` to the framework releases it must keep working
under. `connectors-compat` runs the connectors built at HEAD under each release
listed in `[tool.osprey-connectors.compat] frameworks`: it scans the release and
the pyAML backend it depends on for connectors names that no longer resolve
(attribute reads off module aliases included), imports every module of them,
renders a deployment with its CLI and opens the simulator view, and reads the
control-context record across its hook and golden files.
`framework-at-connectors-floor` runs the framework at HEAD with the connectors
release its floor names, held to that release's installed copy. A nightly
workflow runs the same checks and each listed release's own connector tests
against HEAD and opens an issue per failed job; dispatched, it runs both
against a named release with the HEAD or a published connectors as a dry run.
The plan holds both lists to the tags reachable from HEAD: a tagged release
must be listed or retired once a newer one follows it, and the floor must name
a tagged connectors release. `pyaml-cs-osprey` now declares the connectors
floor it imports against.

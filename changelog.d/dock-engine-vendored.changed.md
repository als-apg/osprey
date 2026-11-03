The dock layout engine is CANOPY's: `packages/dock-engine` leaves OSPREY, and
the web terminal serves the screen runtime's built `dock-engine.mjs` from
`/static/vendor/` as a tracked byte copy pinned to one runtime commit, with the
server's layout rules in `osprey.interfaces.web_terminal.dock_layout` replaying
the engine's own cases. The retired `slot` and key-leaf layout shapes are now
refused as `not_layout`.

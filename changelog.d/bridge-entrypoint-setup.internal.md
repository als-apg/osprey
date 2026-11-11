The Teams, Nextcloud Talk, Google Chat and Slack bridge entrypoints share one
`configure_bridge_logging` and `install_signal_handlers` from the bridge core instead
of four identical copies; the unused `DedupStore.seen` is removed.

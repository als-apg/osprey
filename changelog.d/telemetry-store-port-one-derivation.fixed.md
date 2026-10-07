`osprey health` and the telemetry store's ingest-account provisioning read the store's host
port the same way the agent's exporter does. A `services.openobserve.port` that is not an
integer is reported by name instead of producing an address that cannot be dialed.

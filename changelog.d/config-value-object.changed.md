Malformed configuration now fails with the documented error or falls back to
the documented default instead of crashing further in: a non-mapping `ariel:`
or `services.postgresql` raises `ConfigurationError` (and `osprey ariel`
commands exit naming the key), a non-string `facility_knowledge.bundle_path`
is a usage error, a non-string logbook composition provider answers 503, a
non-mapping optimizer `control_system` raises `InterfaceError`, and a
non-string `cli.theme` falls back to the default theme. `web.chat_turn_timeout_s`,
`web.chat_idle_timeout_s` and `web.chat_max_sessions` no longer read `true` or
`false` as 1 or 0; a boolean there falls back to the chat-pool defaults with a
warning.

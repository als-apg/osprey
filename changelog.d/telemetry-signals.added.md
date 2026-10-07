`claude_code.telemetry.signals` chooses which of metrics, logs and traces the
agent exports; unset, it exports all three. Every launch follows it: the web
terminals, the dispatch worker, `osprey chat` and SDK agent runs. A signal left
out is exported as `none`, so a shell export or a `.env` line cannot turn it
back on, and an empty list or an unknown name stops `osprey build`.

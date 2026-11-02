`osprey chat` no longer paints its companion servers' log records over the
agent's screen. Once the agent is launched, what the companion threads log --
a health check that passes late, a database pool that times out on a slow
host -- goes to `var/logs/chat-<timestamp>.log` in the deployment repo, named
on the line before the launch; errors still reach the terminal. Nothing is
dropped: every record is emitted as before and reaches the file.

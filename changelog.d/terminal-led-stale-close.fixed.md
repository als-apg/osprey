The terminal's connection light no longer goes dark on a connected terminal.
When the terminal is asked to resume a session that has no transcript yet, it
drops that socket and starts a new session; if the new socket opened before
the old one finished closing, the old close turned the light and the
terminal glow off, and they stayed off. Only the live connection's close
turns them off now.

The terminal no longer loses its connected glow while it is connected. When the
terminal is asked to resume a session that has no transcript yet, it drops that
socket and starts a new session; if the new socket opened before the old one
finished closing, the old close turned the glow off, and it stayed off. Only
the live connection's close turns it off now.

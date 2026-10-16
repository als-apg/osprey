Stopping a chat or terminal session now ends the commands its agent started
even when the process table cannot be read on the first try. The read is
retried while the agent is alive, and a warning names the session when it
still cannot look.

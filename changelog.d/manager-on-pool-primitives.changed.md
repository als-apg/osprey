The controls server's target switch now launches and retires its connector-host child through
the same handshake and teardown as the connector pool. A connector-host child now exits when its
parent process changes, including when it is handed to a subreaper rather than to init. A
restarting controls server no longer signals PIDs recorded by a dead predecessor.

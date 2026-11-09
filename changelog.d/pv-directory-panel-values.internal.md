Internal: with the `pv_directory` section configured, the Channels panel reads current
values through a read-only child process its lifespan supervises by exit code, over a
private Unix socket; the panel process itself opens no control session.

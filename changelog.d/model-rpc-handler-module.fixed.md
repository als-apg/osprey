The model-RPC handler is its own module, free of the Channel Access server, and
its exactly-once and timeout behaviour is tested on every host. A reply the
server could not deliver is followed by the timeout answer instead of silence.

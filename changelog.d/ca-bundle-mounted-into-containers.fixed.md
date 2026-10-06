The CA file a connection block names under `tls.ca_bundle` is now mounted
read-only at the same path into every web terminal, dispatch worker and archive
recorder that reads the block, so the key works in a container as it does on
the host. The path must be absolute: a `~` path is refused, because each process
would expand it against its own home.

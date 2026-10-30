`osprey web --detach --port 0` reads the port only from a whole startup line in
the child's log, so a line still being written can no longer be read as the
wrong port or crash the launch.

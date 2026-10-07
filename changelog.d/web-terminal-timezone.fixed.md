Web terminals, their auth sidecar and nginx now run in `system.timezone`, like
every other deployed service; they ran in UTC. On a deployment whose zone is
not UTC, the next `osprey up` re-renders `.env.users` and recreates those
containers.

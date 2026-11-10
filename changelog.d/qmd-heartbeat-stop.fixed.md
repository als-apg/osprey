The qmd sidecar no longer leaves a stray `sleep` behind, or stalls for two
minutes, when a long stage finishes the moment it starts: stopping the
progress heartbeat now takes its sleep along wherever the stop signal lands.

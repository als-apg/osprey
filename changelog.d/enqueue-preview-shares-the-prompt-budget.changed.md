The plan preview on a `queue_add` prompt now spends that prompt's own time
budget rather than a fixed 15 seconds, so a slow bridge costs the trajectory
and says so, never the prompt itself.

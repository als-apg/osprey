The Bluesky bridge, queueserver and sidecar containers now load the project
config through OSPREY's own loader. `bluesky.plan_module`, `bluesky.plan_dirs`,
`bluesky.excluded_plans` and `python_executor.child_env_passthrough` set in
the config now take effect there, the sidecar lists every plan lane a two-lane
deployment renders, and pre-flight and sidecar audit records reach the
project's agent record instead of staying inside the container.

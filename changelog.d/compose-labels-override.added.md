Every render writes `build/osprey-labels.override.yml`, which gives each rendered service the
`osprey.project.name`, `com.osprey.repo-id`, `osprey.project.root` and `osprey.config.digest`
labels, and every compose command OSPREY runs passes it last. A facility service whose template
leaves these labels out is now listed by `osprey status`, found by `osprey down` and
`osprey reset`, and recreated on a config change. Bluesky's per-lane Redis and Tiled containers
gain the same labels, so they are also recreated on a config change.

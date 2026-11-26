`osprey build` no longer warns about an unrecognized top-level entry for a
directory the profile names under a key whose path the build protects from
agent code, such as `optimizer.environments_dir`, `bluesky.plan_dir`,
`bluesky.external.parameter_schemas`, `bluesky.plan_dirs` or a
`services.qmd.corpora` source. Any other unrecognized entry beside them is
still reported.

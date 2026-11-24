`osprey build` no longer warns about an unrecognized top-level entry for a
directory the profile names in `optimizer.environments_dir`,
`bluesky.plan_dir` or `bluesky.external.parameter_schemas`. Any other
unrecognized entry beside them is still reported.

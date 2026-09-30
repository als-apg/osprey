`osprey facility validate` runs every check `osprey build` makes of
`data/facility/` and renders the facility file against the repo's main profile
in a temporary directory, writing nothing into the repo. A clean tree exits 0
and prints nothing; otherwise every error of the first failing stage prints on
stderr, one line each and sorted, and the command exits 1. A stale fix's line
carries the block to paste in its place.

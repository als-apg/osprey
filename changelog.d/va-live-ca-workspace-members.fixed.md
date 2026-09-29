The live Channel Access venue image (`scripts/va/live_ca/run_live_ca.sh`)
builds again: its build context now carries the `packages/` workspace members
the dependency sync installs from source, and a change to a member changes the
image tag.

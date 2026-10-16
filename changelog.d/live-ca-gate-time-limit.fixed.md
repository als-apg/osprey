The live Channel Access gate (`scripts/va/live_ca/gate.py`) now gives each
module a wall-clock limit (300 s by default; `--module-timeout=SECONDS` or
`OSPREY_LIVE_CA_MODULE_TIMEOUT` overrides it). A module that hangs is killed
together with every process it started and reported as a failure that names
it, instead of stalling the gate until the CI job times out. Output is read
from files, so a server process left running can no longer hold the gate open
either.

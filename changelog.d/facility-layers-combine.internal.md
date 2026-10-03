Add the facility build's first two stages: `osprey.facility.sources` reads each
source file under `data/facility/`, and `osprey.facility.combine` merges the
layers per field, applies `fixes.yaml` and records each record's provenance. No
command runs them yet.

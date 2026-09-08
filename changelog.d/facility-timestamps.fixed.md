Operator-facing timestamps now carry the facility timezone. The header of the
notebook written for an executed script shows the facility offset instead of a
UTC literal, and a Phoebus Data Browser plot opened without an explicit time
range spans the last 24 hours of facility-local wall clock rather than the
container's.

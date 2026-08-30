The hardware-write approval prompt no longer appears for a target the operator
has sandboxed from the control-target chip. The approval hook now reads the
same per-(session, target) posture store as the write gate, so a narrowed
target gets the write refusal alone instead of an approval dialog for a write
that would be refused anyway.

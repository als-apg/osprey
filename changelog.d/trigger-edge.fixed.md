An `epics_ca` dispatch trigger with an unrecognized `edge` value is now skipped
with a warning naming the trigger, instead of being armed as if it had said
`both`. Only `rising`, `falling` and `both` arm a monitor; sibling triggers in
the same set are unaffected.

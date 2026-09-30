An `epics_ca` trigger whose `threshold` or `cool_down_sec` is not a number, is
not finite, or (for `cool_down_sec`) is negative is skipped with a warning
naming it, and the other triggers arm. Before, a non-numeric value stopped the
dispatcher from starting, and `true` or `.nan` was accepted silently.

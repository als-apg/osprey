Virtual accelerator: a `trajectory` measurement kind reads the beam position at
each monitor over a chosen number of turns, and the virtual accelerator can
acquire turn-by-turn data and fire single shots through the model. A model can
now carry a degraded state, such as a ring that has lost its closed orbit, and
the trajectory response-matrix files are produced for the example ring. The
`first-turn-errors` scenario exercises this path end to end. One behaviour
changes: the example storage ring now accepts a write that leaves it without a
closed orbit and reports that state, where before the write was refused.

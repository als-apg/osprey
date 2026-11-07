`osprey.runtime.budget` computes one write time budget from the connector
blocks: the call bound T and step-read bound S (maxima over every block,
clamped so a write's bound `S + 2T` stays within 24 s), the stop grace and the
relay timeout, plus `bound_budget()` for the block a session is bound to. A
non-finite bound (NaN or ±infinity) in a block's `step_read_timeout_s`, or in
the budget's `T` or `S`, is refused with a `ValueError` that names it.

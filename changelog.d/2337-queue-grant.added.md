A queued Bluesky plan at the *asked* switch position now waits for a person
to approve its writes from the write-authorization chip, or with
`osprey write-authorization approve`, before it moves anything. While it
waits, the queue reads *awaiting approval* and the running plan names who has
to approve it and where. It asks again when it needs more writes, a wider
range or more time than was approved. At *allowed* the plan approves itself;
at *off* nothing changes. Turning writes off while a plan waits ends the plan
and withdraws its request, so a later approval moves nothing; turning them to
*allowed* lets it go on without asking. A plan that sets a channel to
`True` or `False` is approved and written as `1` or `0`. A request left by a
queue worker that stopped while it waited is withdrawn when the worker starts
again.

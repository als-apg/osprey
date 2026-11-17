A queued Bluesky plan at the *asked* switch position now waits for a person
to approve its writes from the write-authorization chip, or with
`osprey write-authorization approve`, before it moves anything. While it
waits, the queue reads *awaiting approval* and the running plan names who has
to approve it and where. It asks again when it needs more writes, a wider
range or more time than was approved. At *allowed* the plan approves itself;
at *off* nothing changes.

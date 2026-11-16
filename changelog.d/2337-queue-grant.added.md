A queued Bluesky plan at the *asked* switch position now waits for a person
to approve its writes from the write-authorization chip, or with
`osprey write-authorization approve`, before it moves anything. It asks again
when it needs more writes, a wider range or more time than was approved. At
*allowed* the plan approves itself; at *off* nothing changes.

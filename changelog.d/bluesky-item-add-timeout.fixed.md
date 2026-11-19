A Bluesky queue add that the queue server does not answer in time is checked
against the queue before its draft revision is freed: an item that carries the
add's run id counts as the add, so a retry cannot queue the plan a second time.
When the queue cannot be read either, the add answers `enqueue_outcome_unknown`
(not retryable) and keeps the revision reserved: check the queue, then edit the
draft to add the plan again.

The per-reader concurrency e2e test widens the reader pool to leave a slot
spare, so its third query meets the per-reader check at once instead of
queueing for a pool slot, where the outcome raced the release of a held
query's slot against its per-reader count. The test now also asserts the
third query did not wait.

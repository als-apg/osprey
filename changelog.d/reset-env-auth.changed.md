`osprey reset` (and `osprey init --reset`) now removes `.env.auth`, so a reset
deployment cannot carry the previous generation's login passwords into a
freshly seeded `.env`. The removal is disclosed in the reset plan; the next
deploy re-mints every hash from `.env`.

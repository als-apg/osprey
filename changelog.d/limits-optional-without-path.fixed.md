Limits checking in `optional` mode with no limits path configured no longer
blocks every write or refuses the deploy: there is nothing to load, so every
channel is unlisted and written with no limits. A configured path that fails to
load stays fail-safe in every mode.

A `compose up` that waits forever on a dependency no longer leaves `osprey up`
showing only a running timer. Every 30 seconds it names the containers it is
still waiting on and the last line compose wrote; after a time limit
(`compose_up_timeout`, default 30 minutes) it stops and fails with that account
plus the last log lines of every container that exited or is unhealthy, and
Ctrl-C prints the same account. A failed run's replayed output prints a line
compose repeated once, with its count.

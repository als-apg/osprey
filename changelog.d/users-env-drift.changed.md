A provider key rotated in `.env` now reaches the web terminals. `osprey up`
re-renders a `.env.users` it generated itself whenever the env chain has
changed, instead of leaving every terminal on the old value; a `.env.users`
you edited by hand is still never rewritten, but the deploy refuses to start
when it disagrees with `.env` on a provider key, naming the variable and the
fix (`osprey users env --output .env.users`). `osprey health` reports the same
drift as an error on a new `users_env` row, and a failing check of the
deployment's own agent provider (`claude_code.provider`) is now an error
rather than a warning, so a rejected key exits 2 instead of hiding among
advisories.

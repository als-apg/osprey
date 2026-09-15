A deploy whose env chain does not set a gateway endpoint its web terminals
need is now refused by name even when `.env.users` already exists, instead of
starting containers that restart forever; `osprey health` reports the same gap.
The `.env` seed `osprey up` offers also harvests that endpoint from your shell
alongside the provider's key.

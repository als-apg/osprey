Per-user web terminals no longer restart forever on a deployment whose provider
fronts a gateway with no default endpoint. The variable naming that endpoint now
crosses into `.env.users` with the provider's key, and a deploy whose env chain
sets neither is refused by name instead of generating the file.

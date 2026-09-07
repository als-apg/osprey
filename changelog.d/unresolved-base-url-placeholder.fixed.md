A provider's `base_url` left as an unresolved `${VAR}` — the variable is not
exported — is now read as "no endpoint configured" instead of being sent to the
HTTP client as a hostname, so the failure names the provider and the missing
value. `osprey health` reports that missing endpoint by name too, instead of
probing another vendor's API with the gateway's key, and the channel-finder
benchmark's ReAct backend expands the reference — from the shell or the
deployment's `.env` — rather than dialling it.

The `als-apg` provider no longer ships a built-in endpoint. Its gateway is a
site's own host, so the URL now comes from `api.providers.als-apg.base_url` in
`providers.yml` (shipped as `${ALS_APG_BASE_URL}`) or from that variable
directly; naming a provider with no endpoint anywhere is refused instead of
falling back on someone else's host. Existing deployments that relied on the
old default must export `ALS_APG_BASE_URL`.

The generated `.env.example` now lists the endpoint variables that have no
default, beside the provider API keys, so a deployment sees both halves of what
`als-apg` needs before its first launch rather than after it.

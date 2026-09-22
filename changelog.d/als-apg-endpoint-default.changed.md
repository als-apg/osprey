The `als-apg` provider now ships the gateway endpoint it fronts,
`https://llm.als.lbl.gov`, so a deployment that names none reaches it without
configuration. `ALS_APG_BASE_URL` and `api.providers.als-apg.base_url` remain
overrides for a site whose gateway is elsewhere, and the generated
`.env.example` no longer lists the endpoint as a value you have to supply.

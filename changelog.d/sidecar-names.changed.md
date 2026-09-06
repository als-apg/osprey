The sidecar-metadata step now takes its filenames from the ingestion adapter
(`metadata_sidecar_names`, default `("metadata.json",)`) instead of matching
that one name everywhere, so a logbook whose sidecar is spelled differently
gets its metadata merged. The fetch also honours the ingestion block's
`verify_ssl`, `ca_bundle` and `proxy_url`; it previously opened a plain
connection that ignored all three.

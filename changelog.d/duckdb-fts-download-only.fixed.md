The channel finder's full-text index no longer looks for a bundled DuckDB
extension file nobody ships; it downloads the extension (through
`http_proxy`/`HTTP_PROXY` when set), and a failed download says the host needs
network access or a proxy.

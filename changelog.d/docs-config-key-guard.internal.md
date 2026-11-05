Every YAML block in the docs is now checked against the config-key manifest, and
the configuration-reference key sweep reads its section list off the manifest and
covers the CLI, ports and contract pages. The manifest declares 72 keys the docs and
code already used (ARIEL watch settings, `health.*` suite and dashboard keys,
`modules.web_terminals` TLS, OIDC and claim keys, `claude_code.hooks`, the rest of
`claude_code.permissions`, and the top-level deploy switches).

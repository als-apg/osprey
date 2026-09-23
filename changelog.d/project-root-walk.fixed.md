A process started without a config file now anchors agent data and the audit
ledger on the deployment repo it stands in, found by walking up to
`profile.yml` as the hooks already do, instead of on its working directory.

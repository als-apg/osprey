`exclude_providers` in an application registry now removes the provider from
the registry that model calls resolve through, so a config that names an
excluded provider fails with `Unknown provider` instead of reaching it, and
`osprey registry` shows what the runtime can use. An exclusion that names no
built-in provider is reported with a warning.

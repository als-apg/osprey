A build profile refuses an environment-variable name that ends in a newline — in `env.required`,
`env.pinned`, `services.<name>.env`, `dispatch.env`, `bind_env`, `deploy.registry.token_env_var`,
`va_archiver.password_env` and the `bluesky.external` key variables — as every other config
surface already did. The `deploy` block's refusal no longer says a name must be uppercase.

A provider class registered through `ProviderRegistration` names the variable
its API key arrives in with `api_key_env_var`. In a process that has loaded
the application registry, the launch, `.env.users`, `.env.example` and the
build's key summary read that variable instead of `<NAME>_API_KEY`. The
built-in providers and config-only `providers.yml` entries keep the variables
they use today.

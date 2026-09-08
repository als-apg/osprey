Provider facts now come from one producer each, so a deployment that configures
its own endpoints is reported and run as it is configured.

- `osprey health` reports a provider it has no adapter class to probe as
  skipped, including the agent's own, instead of failing the deployment.
- `osprey audit` runs its reviewer on the audited deployment's provider and
  default model, and refuses a bare profile with no build to resolve one from.
- A provider registered in an application registry under a built-in name
  replaces that built-in instead of being dropped without a word.
- A keyless adapter passes the health check on its own `requires_api_key`
  declaration rather than by being named in a list.
- A `gateway:` key in a provider entry overrides the built-in default for
  spend-attribution headers.
- A misspelled `api_protocol` in `providers.yml` is refused at load.
- The `osprey status` remedy and the no-provider error name the keys that
  actually set a provider, and `registry_path` is documented.

`api_protocol` in a `providers.yml` entry now overrides the provider's own
protocol in both directions: `api_protocol: openai` on `anthropic`, `cborg` or
`als-apg` routes the launch through the translation proxy instead of being
ignored. A provider class registered through `ProviderRegistration` that
declares `api_protocol = "anthropic"` launches without the proxy.

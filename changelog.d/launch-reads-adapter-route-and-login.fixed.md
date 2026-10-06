`osprey web` stops before launch when a gateway that speaks the Anthropic API
natively has no API key (`cborg`, `als-apg`, or a `providers.yml` entry with
`api_protocol: anthropic` and no provider class), naming the variable. Before,
it warned about a subscription login these gateways do not offer and launched
into an authentication error. Direct `anthropic` without `ANTHROPIC_API_KEY`
still launches with a warning, since it offers an interactive login.

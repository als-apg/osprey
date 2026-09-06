`api.providers.<name>.api_protocol` now accepts only `anthropic` and `openai`.
Anything else — a capitalised `Anthropic`, a typo — is refused when the
provider is resolved, naming the provider and the two accepted values. It
previously read as "not Anthropic" and silently inserted the OpenAI
translation proxy in front of a gateway that speaks Anthropic natively. An
absent key still means OpenAI.

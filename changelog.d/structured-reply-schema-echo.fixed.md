A structured request through a provider without native structured output (the
prompt fallback, Ollama, AskSage) no longer comes back silently empty when the
model answers with the schema's shape instead of an instance of it. The request
now asks for an instance by field name, a reply with a top-level `properties`
or `type`/`title` the model does not define is asked again, and a second such
reply raises `StructuredReplySchemaEchoError`.

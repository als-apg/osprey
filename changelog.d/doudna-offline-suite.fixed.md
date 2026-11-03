Importing an OSPREY model provider no longer stalls on an air-gapped host.
LiteLLM fetched its model map from GitHub at import, so every process that
loaded a provider waited for that fetch to time out; it now uses the map the
installed LiteLLM ships, and an operator who set `LITELLM_LOCAL_MODEL_COST_MAP`
keeps their value. LiteLLM also no longer loads a `.env` found by searching
upward from the process. OpenAI model ids are routed with an explicit `openai`
prefix, so a model newer than the installed LiteLLM is no longer refused. The
web interfaces serve their static assets with the same content types on every
host instead of reading the host's mime table, which on RHEL served `.js` as
`application/javascript`.

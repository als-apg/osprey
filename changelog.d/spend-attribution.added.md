Requests to a LiteLLM-fronted provider (`als-apg`, `cborg`, or a custom
provider with `gateway: litellm`) now carry the acting identity for spend
attribution: the agent sends `x-litellm-end-user-id` (the terminal's roster
user, the dispatch worker, or the local account) and `x-litellm-tags`
(`osprey,surface:<terminal|dispatch|service|local>`) via
`ANTHROPIC_CUSTOM_HEADERS`, merged into any operator headers already there,
and the LiteLLM SDK path sets the same identity as the OpenAI `user` field.
The gateway's spend logs then book each call to a person and a surface instead
of to the deployment's shared key. The translation proxy forwards `x-litellm-*`
headers to an OpenAI-protocol gateway so both protocol paths attribute alike.

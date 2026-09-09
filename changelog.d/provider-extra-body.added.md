A provider entry in `providers.yml` may carry an `extra_body` mapping. The
mapping is sent in the request body of every completion, which is how a
LiteLLM gateway that uses client-side auth receives a per-user upstream key
alongside the gateway key in `api_key`. (#930)

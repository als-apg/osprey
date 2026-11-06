The provider catalog ships a keyless `vllm` entry for a self-hosted vLLM server
(default `http://localhost:8000/v1`, or `VLLM_BASE_URL`), and the `als-apg`
entry lists `claude-sonnet-5-5`. On `als-apg` the `sonnet` alias now resolves
to `claude-sonnet-5-5` in every deployment, as `haiku` and `opus` already
resolve to their 5.5 models; the default model stays `claude-sonnet-5`.

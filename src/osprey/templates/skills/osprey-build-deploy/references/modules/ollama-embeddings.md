# Module: ollama

Local Ollama server providing text embeddings (and optionally chat completion) to other components in the deploy — most commonly the wiki/paper search stack and the ARIEL embeddings provider. **Ollama is typically NOT containerized as part of this deploy.** The expected pattern is that a facility has a separate GPU host already running Ollama as a systemd service; this module just records its URL + model names so other modules can reach it. If you really need to run Ollama on the deploy server, see the "Running Ollama on the deploy server" section below — it's almost never the right call.

**Enabled when**: `modules.ollama.enabled: true`

## Configuration

```yaml
modules:
  ollama:
    enabled: true
    url: "http://gpu-host.example.org:11434"   # full URL incl. port; reachable from the deploy server
    embedding_model: "nomic-embed-text"        # model used for vector embeddings
    chat_model: null                           # optional — if set, available for local LLM calls
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `url` | URL | yes | Full URL with scheme + port; default Ollama port is `11434` |
| `embedding_model` | string | yes | Model tag as installed on the Ollama host (run `ollama list` there to confirm) |
| `chat_model` | string or null | no | Only set if some module is configured to use Ollama for chat completion (rarely the right choice when the facility has hosted-LLM gateway access via `${config.llm.api_key_env_var}`) |

## What scaffolding adds when this module is enabled

- compose: nothing by default (Ollama is external). If you opt into the on-server pattern below, a single `${config.facility.prefix}-ollama` service is added with a GPU device mapping.
- .gitlab-ci.yml: nothing — Ollama is not built or shipped through CI.
- scripts/deploy.sh: nothing — Ollama runs out-of-band.
- scripts/verify.sh: a check that does `curl -s ${config.modules.ollama.url}/api/tags` and confirms the response is non-empty JSON listing at least one model. Advisory only.
- .env.template: nothing required — the URL lives in `facility-config.yml` and is passed to consuming services via compose env vars.

If `modules.ariel.enabled: true` AND `modules.ariel.embeddings_provider: "ollama"`, the ARIEL service block in compose gets `OLLAMA_URL=${config.modules.ollama.url}` and `OLLAMA_EMBEDDING_MODEL=${config.modules.ollama.embedding_model}` injected automatically.

If `modules.wiki_search` or `modules.custom_mcp_servers` define a server that needs embeddings, they should list `${config.modules.ollama.url}` in their service env block via the templates.

## Operating the module

### Verify connectivity from the deploy server

```bash
ssh ${config.deploy.host} "curl -s ${config.modules.ollama.url}/api/tags | python3 -m json.tool"
```

Expected: a JSON object with a `"models"` array containing at least `${config.modules.ollama.embedding_model}`.

### Verify connectivity from inside a container

Service-to-service traffic must bypass the proxy. Confirm:

```bash
${config.runtime.engine} exec ${config.facility.prefix}-mcp-<some-server> sh -c \
  'curl -s ${config.modules.ollama.url}/api/tags >/dev/null && echo OK || echo FAIL'
```

If `FAIL`, the most common cause is the Ollama host not being in `network.no_proxy` in `facility-config.yml`. Add the host (and re-scaffold so it lands in `.env.template`).

### Pull a new model on the Ollama host

Ollama models are pulled and stored on the Ollama host, not on the deploy server. SSH directly to the Ollama host:

```bash
ssh <ollama-host> "ollama pull <model>"
ssh <ollama-host> "ollama list"   # verify it's there
```

After pulling, update `modules.ollama.embedding_model` (or `chat_model`) in `facility-config.yml` if you're switching models, then re-scaffold so consuming services pick up the new value.

### Restart Ollama (when it stops responding)

Ollama is typically managed by systemd on the GPU host:

```bash
ssh <ollama-host> "sudo systemctl restart ollama"
```

After restart, the first request to a given model triggers a cold load (~10–60s depending on model size). Warm it with a dummy request before pointing live traffic at it:

```bash
ssh <ollama-host> "curl -s http://localhost:11434/api/embeddings \
  -d '{\"model\": \"${config.modules.ollama.embedding_model}\", \"prompt\": \"warmup\"}' >/dev/null"
```

## Common embedding models

| Model | Dim | Notes |
|-------|-----|-------|
| `nomic-embed-text` | 768 | Default. Small, fast, good general-purpose quality. |
| `mxbai-embed-large` | 1024 | Larger, better quality on technical text. ~3× the index size. |
| `bge-large` | 1024 | BGE family, strong on long-form documents. |

Match the model in `facility-config.yml` to whatever was used to build the existing index — switching models means re-indexing every collection that consumes the embeddings.

## Optional: chat models

If `chat_model` is set, the most common choices are `llama3.x` or `qwen2.5`. **This is rarely worth the operational cost** when the facility has gateway access to a hosted LLM (`${config.llm.provider}` = `cborg`, `anthropic`, `openai`, etc.). Reasons to actually use it:

- The facility cannot reach external LLM gateways from the control network and has no proxy budget for them.
- The use case is high-volume or background and the per-token cost of a hosted LLM is prohibitive.
- The use case is summarization or simple structured extraction, where a smaller local model is sufficient.

If none of these apply, leave `chat_model: null` and let `${config.llm.model}` handle chat traffic.

## Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ConnectionRefused` from the verify check | Wrong URL, or Ollama down on the host | Confirm `${config.modules.ollama.url}` is reachable; SSH to the host and check `systemctl status ollama` |
| `404 model not found` in consumer logs | Model not pulled on the Ollama host | `ssh <ollama-host> "ollama pull <model>"` |
| First request is slow (~30s) then subsequent are fast | Cold model load | Expected after restart; warm with a dummy request |
| All requests routed through the squid proxy and fail silently | Ollama host not in `network.no_proxy` | Add it to the list in `facility-config.yml` and re-scaffold |
| Embeddings work but vector dim mismatch in consumer index | Model changed without re-indexing | Re-index the consumer collection with the new model, or switch back |
| `502 Bad Gateway` intermittently | GPU OOM or Ollama crash-restart loop | Check Ollama host's GPU memory (`nvidia-smi`); reduce concurrent model loads or switch to a smaller model |

## Cross-references

- If `modules.ariel.enabled` is also true and `modules.ariel.embeddings_provider == "ollama"`, the ARIEL ingestion pipeline calls `${config.modules.ollama.url}` to embed every entry. Pulling a new embedding model means re-running the ARIEL backfill.
- If `modules.custom_mcp_servers` includes a server that does its own RAG-style search (e.g., a Typesense-backed paper index), that server typically calls Ollama for both index-time and query-time embeddings. Confirm the URL is in the server's env block in compose.

## Running Ollama on the deploy server (NOT recommended)

If the deploy server has a GPU and you genuinely cannot use an external Ollama host, a containerized Ollama service can be added. This is **rarely the right call**:

- CPU-only inference of even small embedding models is 10–100× slower than GPU; latencies in seconds-per-query are typical.
- A GPU on the deploy server competes for memory and PCIe bandwidth with whatever else lives there.
- Ollama's container image is large (~5GB) and updates frequently.

If you accept those tradeoffs, set `modules.ollama.url: "http://${config.facility.prefix}-ollama:11434"` and add a service block to `docker-compose.local.yml` (NOT the scaffolded compose — that's regenerated). Mount a persistent volume for the model store so pulls survive restarts.

## Disabling

Set `modules.ollama.enabled: false` (or remove the block) and re-scaffold. Then audit:

- Any module that referenced `modules.ollama.url` (typically ARIEL or custom MCP servers) must be reconfigured to use a different embeddings provider, or disabled.
- The verify.sh Ollama check is removed.
- No data needs to be cleaned up — Ollama lives elsewhere.

If you were running the on-server containerized variant, also `${config.runtime.compose_command} down ${config.facility.prefix}-ollama` and remove the volume.

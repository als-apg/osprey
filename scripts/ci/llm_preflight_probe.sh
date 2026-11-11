#!/usr/bin/env bash
# Pre-flight check that the ALS-APG gateway answers one tiny Messages call, so a
# model-spending lane reds in seconds on a missing key, a rotated key or an
# outage instead of failing deep inside its first agent turn.
#
# Usage: scripts/ci/llm_preflight_probe.sh
#   ALS_APG_API_KEY          the gateway key (required; the lane's step-level secret).
#   ALS_APG_BASE_URL         the gateway base, with or without a trailing / or /v1
#                            (ci.yml exports the repository variable at workflow level).
#   OSPREY_E2E_FORCE_MODEL   the model the probe asks for (the lanes' gateway id).
set -euo pipefail

if [ -z "${ALS_APG_API_KEY:-}" ]; then
  echo "❌ ALS_APG_API_KEY secret is not set on this runner"
  echo "   (Check repo secrets; every LLM-calling lane needs it.)"
  exit 1
fi
: "${ALS_APG_BASE_URL:?ALS_APG_BASE_URL is not set (ci.yml exports vars.ALS_APG_BASE_URL)}"
: "${OSPREY_E2E_FORCE_MODEL:?OSPREY_E2E_FORCE_MODEL is not set}"

base="${ALS_APG_BASE_URL%/}"
base="${base%/v1}"
body="$(mktemp)"
trap 'rm -f "$body"' EXIT

code=$(curl -sS -o "$body" -w "%{http_code}" --max-time 15 \
  -X POST "$base/v1/messages" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -H "Authorization: Bearer $ALS_APG_API_KEY" \
  -d "{\"model\":\"$OSPREY_E2E_FORCE_MODEL\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}") || true
if [ "$code" != "200" ]; then
  echo "❌ ALS-APG endpoint probe failed (HTTP ${code:-none})"
  echo "   Body (first 500 chars):"
  head -c 500 "$body"; echo
  echo "   If 401/403: rotate ALS_APG_API_KEY or check IP allowlist."
  echo "   If 5xx: ALS-APG outage — retry, or contact the proxy admin."
  exit 1
fi
echo "✅ ALS-APG endpoint reachable (HTTP 200)"

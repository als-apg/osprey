// @ts-check
/*
 * System Health dashboard — the panel's HTTP surface.
 *
 * GET /checks (polled), GET /mcp (only when the MCP tile opens: it spawns the
 * servers), POST /mcp/probe, /reprobe and /ping (explicit clicks).
 */

/**
 * @param {string} path
 * @param {any} [body]
 */
export async function call(path, body) {
  const init =
    body === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        };
  const r = await fetch(path, init);
  if (r.status === 409) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.detail || 'refresh suppressed');
  }
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}

export function makeApi() {
  return {
    checks: () => call('/checks'),
    mcp: () => call('/mcp'),
    /** @param {string} [server] */
    mcpProbe: (server) => call('/mcp/probe', server ? { server } : {}),
    /** @param {any} body */
    reprobe: (body) => call('/reprobe', body),
    /** @param {string} provider */
    ping: (provider) => call('/ping', { provider }),
  };
}

/** @typedef {ReturnType<typeof makeApi>} Api */

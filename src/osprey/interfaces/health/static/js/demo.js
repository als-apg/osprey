// @ts-check
/*
 * System Health dashboard — `?demo` fixture.
 *
 * A small, self-consistent envelope in the /checks shape (results + layers +
 * mcp) and an /mcp introspection payload, plus a fake API answering /mcp,
 * /mcp/probe, /reprobe and /ping from them, so the page can be reviewed with
 * no backend. Loaded by dashboard.js only when the URL carries `?demo`.
 */

import { NOW, comp, demoRows } from './demo-data.js';
import { DEMO_MCP, mcpOf } from './demo-mcp.js';

/** @returns {any} */
export function demoEnvelope() {
  const rows = demoRows();
  const layers = [
    {
      id: 'op',
      label: 'Operator interface',
      components: [
        comp(rows, 'web-terminal', 'web terminal', '2 logins', { rows: [0, 1] }),
        comp(rows, 'panels', 'panels', '2 / 2', {
          groups: [
            { id: 'logbook', name: 'Logbook', rows: [2, 3] },
            { id: 'knowledge', name: 'Knowledge', rows: [4] },
          ],
        }),
      ],
    },
    {
      id: 'agent',
      label: 'Agent',
      components: [
        comp(rows, 'claude-code', 'Claude Code', '2.1.300', { rows: [5] }),
        comp(rows, 'model-provider', 'model provider', 'als-apg', { rows: [6], unused: [7, 8] }),
        comp(rows, 'embed-vision', 'embedding · vision', 'ollama', { rows: [9] }),
      ],
    },
    {
      id: 'chain',
      label: 'Safety chain',
      components: [
        comp(rows, 'hooks', 'hooks', '3 / 3 = render', { rows: [10, 11, 12] }),
        comp(rows, 'write-switch', 'write switch', 'asked', { rows: [13] }),
        comp(rows, 'limits', 'limits file', '1,284 channels', { rows: [14] }),
      ],
    },
    {
      id: 'mcp',
      label: 'MCP servers',
      components: [
        {
          id: 'mcp-servers',
          name: 'MCP servers',
          status: 'ok',
          fact: '',
          issue: null,
          header: null,
          used_by: [],
          rows: [],
          groups: [],
          unused: [],
        },
      ],
    },
    {
      id: 'data',
      label: 'Data services',
      components: [
        comp(rows, 'graph-store', 'graph store', 'bolt :10802', {
          rows: [15, 16],
          used_by: ['channel finder', 'graph MCP server', 'Channels panel'],
        }),
        comp(rows, 'agent-record', 'telemetry store', 'clickhouse · 1 / 2 roots readable', {
          issue: '1 / 2 transcript roots readable',
          header: 'agent record · clickhouse',
          groups: [
            { id: 'store', name: 'store', summary: 'clickhouse · 76 entries', rows: [17, 18] },
            {
              id: 'transcripts',
              name: 'transcripts',
              summary: '1 / 2 readable',
              rows: [19, 20],
              fix: 'docker compose exec agent-record ls /data/transcripts',
            },
          ],
        }),
      ],
    },
    {
      id: 'out',
      label: 'Connections out',
      components: [
        comp(rows, 'cs-connector', 'control-system connector', 'reachable', { rows: [21] }),
      ],
    },
    {
      id: 'cfg',
      label: 'Configuration & build',
      components: [comp(rows, 'config-yml', 'config.yml', 'valid', { rows: [22, 23] })],
    },
    {
      id: 'host',
      label: 'Host',
      components: [
        comp(rows, 'disk', 'disk', '30 % used', { rows: [24] }),
        comp(rows, 'container-runtime', 'container runtime', 'n/a in container', { rows: [25] }),
      ],
    },
  ].map((l) => ({ ...l, status: worstOfComps(l.components) }));
  const minor = rows.filter((r) => r.status === 'warning').length;
  const errors = rows.filter((r) => r.status === 'error').length;
  const skips = rows.filter((r) => r.status === 'skip').length;
  const ok = rows.length - minor - errors - skips;
  return {
    summary: ok + '/' + rows.length + ' checks passed',
    ok,
    warnings: minor,
    errors,
    skips,
    total: rows.length,
    elapsed_ms: 1736,
    deadline_hit: false,
    stale: false,
    warming: false,
    interval_s: 60,
    title: 'System Health',
    scanned_at: NOW() - 14,
    results: rows,
    layers,
    mcp: {
      configured: DEMO_MCP.servers.map((s) => s.name),
      answered: null,
      tools: null,
      probed_at: null,
    },
  };
}

/** @param {{status: string}[]} comps */
function worstOfComps(comps) {
  const rank = { error: 3, warning: 2, ok: 1, skip: 0 };
  let w = 'skip';
  for (const c of comps)
    if (rank[/** @type {'ok'} */ (c.status)] > rank[/** @type {'ok'} */ (w)]) w = c.status;
  return w;
}

/** Sleep helper for the fake API. @param {number} ms */
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * A fake backend with the same surface dashboard.js calls.
 *
 * @returns {{checks: () => Promise<any>, mcp: () => Promise<any>, mcpProbe: (server?: string) => Promise<any>,
 *            reprobe: (body: any) => Promise<any>, ping: (provider: string) => Promise<any>}}
 */
export function demoApi() {
  let env = demoEnvelope();
  /** @type {any} */
  let mcp = null;
  const jit = (/** @type {number} */ n) =>
    Math.max(1, Math.round(n * (0.85 + Math.random() * 0.3)));
  return {
    checks: async () => env,
    mcp: async () => {
      if (!mcp) {
        await wait(900);
        mcp = { ...DEMO_MCP, probed_at: NOW() };
        env = { ...env, mcp: mcpOf(mcp) };
      }
      return mcp;
    },
    mcpProbe: async (server) => {
      await wait(700);
      const base = mcp || DEMO_MCP;
      mcp = {
        probed_at: NOW(),
        servers: base.servers.map((/** @type {any} */ s) =>
          !server || s.name === server
            ? { ...s, init_ms: s.init_ms && jit(s.init_ms), list_ms: s.list_ms && jit(s.list_ms) }
            : s
        ),
      };
      env = { ...env, mcp: mcpOf(mcp) };
      return mcp;
    },
    reprobe: async (body) => {
      await wait(600);
      const names = new Set(body && body.checks ? body.checks : []);
      const comp = body && body.component;
      const t = NOW();
      env = {
        ...env,
        scanned_at: names.size || comp ? env.scanned_at : t,
        results: env.results.map((/** @type {any} */ r) =>
          !names.size && !comp
            ? { ...r, probed_at: t }
            : names.has(r.name) || r.component === comp
              ? { ...r, probed_at: t }
              : r
        ),
      };
      return env;
    },
    ping: async (provider) => {
      await wait(800);
      if (provider === 'ollama') {
        return {
          provider,
          status: 'ok',
          kind: 'completion',
          model: 'qwen3-vl:4b',
          latency_ms: jit(420),
          input_tokens: 11,
          output_tokens: 1,
          message: '',
        };
      }
      if (provider === 'als-apg') {
        return {
          provider,
          status: 'ok',
          kind: 'completion',
          model: 'claude-haiku-4-5',
          latency_ms: jit(1200),
          input_tokens: 14,
          output_tokens: 1,
          message: '',
        };
      }
      return {
        provider,
        status: 'skip',
        kind: 'completion',
        model: '',
        latency_ms: null,
        input_tokens: null,
        output_tokens: null,
        message: 'API key not set',
      };
    },
  };
}

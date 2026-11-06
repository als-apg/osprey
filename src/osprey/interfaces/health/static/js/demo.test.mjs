// @ts-check
/**
 * The `?demo` fixture keeps the /checks + /mcp contract the dashboard renders:
 *   npx vitest run src/osprey/interfaces/health/static/js/demo.test.mjs
 */

import { test, expect, describe } from 'vitest';

import { demoEnvelope, demoApi } from './demo.js';

describe('demoEnvelope', () => {
  test('carries the eight layers in order', () => {
    const env = demoEnvelope();
    expect(env.layers.map((/** @type {any} */ l) => l.id)).toEqual([
      'op',
      'agent',
      'chain',
      'mcp',
      'data',
      'out',
      'cfg',
      'host',
    ]);
  });

  test('every row index resolves and no result sits in two components', () => {
    const env = demoEnvelope();
    /** @type {number[]} */
    const seen = [];
    for (const l of env.layers) {
      for (const c of l.components) {
        const idx = [...c.rows, ...c.groups.flatMap((/** @type {any} */ g) => g.rows), ...c.unused];
        for (const i of idx) {
          expect(env.results[i]).toBeDefined();
          expect(env.results[i].component).toBe(c.id);
        }
        seen.push(...idx);
      }
    }
    expect(new Set(seen).size).toBe(seen.length);
  });

  test('statuses use the wire vocabulary and the minor group carries its fix', () => {
    const env = demoEnvelope();
    for (const r of env.results) expect(['ok', 'warning', 'error', 'skip']).toContain(r.status);
    const tel = env.layers
      .flatMap((/** @type {any} */ l) => l.components)
      .find((/** @type {any} */ c) => c.id === 'telemetry-store');
    expect(tel.status).toBe('warning');
    expect(tel.groups.find((/** @type {any} */ g) => g.id === 'transcripts').fix).toMatch(
      /^docker compose/
    );
  });

  test('MCP is unprobed until the servers are introspected', () => {
    const env = demoEnvelope();
    expect(env.mcp.answered).toBeNull();
    expect(env.mcp.configured.length).toBe(3);
  });
});

describe('demoApi', () => {
  test('mcp() introspects once and the envelope then counts the answers', async () => {
    const api = demoApi();
    const data = await api.mcp();
    expect(data.servers.filter((/** @type {any} */ s) => Array.isArray(s.tools)).length).toBe(2);
    const env = await api.checks();
    expect(env.mcp.answered).toBe(2);
    expect(env.mcp.tools).toBe(5);
  });

  test('reprobe of named checks re-stamps only those rows', async () => {
    const api = demoApi();
    const before = await api.checks();
    const after = await api.reprobe({ checks: ['yaml_valid'] });
    const pick = (/** @type {any} */ e, /** @type {string} */ n) =>
      e.results.find((/** @type {any} */ r) => r.name === n).probed_at;
    expect(pick(after, 'yaml_valid')).toBeGreaterThan(pick(before, 'yaml_valid'));
    expect(pick(after, 'disk_space')).toBe(pick(before, 'disk_space'));
  });

  test('ping answers a completion for a used provider and a reason otherwise', async () => {
    const api = demoApi();
    expect((await api.ping('als-apg')).status).toBe('ok');
    expect((await api.ping('cborg')).message).toBe('API key not set');
  });
});

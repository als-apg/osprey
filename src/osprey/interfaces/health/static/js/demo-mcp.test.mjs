// @ts-check
/**
 * Demo MCP payload and its summary (demo-mcp.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/demo-mcp.test.mjs
 */

import { test, expect, describe } from 'vitest';

import { DEMO_MCP, mcpOf } from './demo-mcp.js';

describe('DEMO_MCP', () => {
  test('has stdio servers with tools and one http server without', () => {
    const http = DEMO_MCP.servers.filter((s) => s.transport === 'http');
    expect(http.length).toBe(1);
    expect(http[0].tools).toBeNull();
    for (const s of DEMO_MCP.servers.filter((x) => x.transport === 'stdio')) {
      expect(Array.isArray(s.tools)).toBe(true);
      for (const t of s.tools || []) expect(t.inputSchema).toBeDefined();
    }
  });
});

describe('mcpOf', () => {
  test('counts answered stdio servers and their tools', () => {
    expect(mcpOf({ ...DEMO_MCP, probed_at: 7 })).toEqual({
      configured: ['controls', 'workspace', 'canopy'],
      stdio: 2,
      answered: 2,
      tools: 5,
      probed_at: 7,
    });
  });

  test('a stdio server without tools is not answered', () => {
    const data = { probed_at: 1, servers: [{ name: 'a', transport: 'stdio', tools: null }] };
    expect(mcpOf(data)).toMatchObject({ stdio: 1, answered: 0, tools: 0 });
  });
});

// @ts-check
/* global document, setTimeout -- vitest runs these in happy-dom */
/**
 * MCP servers drawer and tool detail (mcp.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/mcp.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { demoEnvelope, demoApi } from './demo.js';
import { DEMO_MCP } from './demo-mcp.js';
import { store, mcp, S, DS, MCP_ID, indexModel, indexMcp } from './state.js';
import { renderBands } from './overview.js';
import { openTile } from './drawer.js';
import { serverTable, serverInfo, toolDetail, loadMcp, probeServer } from './mcp.js';

beforeEach(() => {
  document.body.textContent = '';
  for (const id of ['ring', 'headline', 'bands']) {
    const n = document.createElement('div');
    n.id = id;
    document.body.append(n);
  }
  const d = demoEnvelope();
  store.env = d;
  store.R = d.results;
  store.LAYERS = d.layers;
  store.api = demoApi();
  mcp.data = null;
  mcp.loading = false;
  mcp.error = '';
  S.band = null;
  S.key = null;
  for (const k of Object.keys(DS)) delete DS[k];
  indexModel();
  renderBands();
});

describe('server table and info', () => {
  test('stdio rows show tools, startup and list; http reads not probed', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    const rows = [...serverTable(false).querySelectorAll('.srow')].map((e) => e.textContent);
    expect(rows[0]).toBe('›controls34.6 s166 ms');
    expect(rows[2]).toBe('canopyhttp · not probed———');
    const info = serverInfo(store.NODES[MCP_ID + '/controls']);
    expect(info.textContent).toContain('3 tools · startup 4.6 s · list 166 ms');
    expect(info.textContent).toContain('3.4.11 · protocol 2025-11-25 · stdio');
  });
});

describe('toolDetail', () => {
  test('arguments mark required ones and nest object items behind a toggle', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    const d = toolDetail(store.NODES[MCP_ID + '/controls/channel_write']);
    const sub = /** @type {HTMLElement} */ (d.querySelector('.argsub'));
    expect(sub.hidden).toBe(true);
    /** @type {HTMLElement} */ (d.querySelector('.argt')).click();
    expect(sub.hidden).toBe(false);
    expect([...d.querySelectorAll('.req')].length).toBe(3);
    expect(d.querySelector('pre.raw')?.textContent).toContain('"writes"');
  });

  test('description markdown-lite becomes paragraphs, lists and code, never markup', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    const d = toolDetail(store.NODES[MCP_ID + '/controls/channel_read']);
    expect(d.querySelectorAll('.desc li').length).toBe(2);
    expect(d.querySelector('.desc code')?.textContent).toBe('channels');
  });
});

describe('introspection', () => {
  test('opening the tile loads the servers once; probe re-stamps them', async () => {
    openTile(MCP_ID);
    expect(mcp.loading).toBe(true);
    await loadMcp(false);
    await new Promise((r) => setTimeout(r, 1000));
    expect(mcp.data.servers.length).toBe(3);
    const before = mcp.data.probed_at;
    await probeServer('controls');
    expect(mcp.data.probed_at).toBeGreaterThanOrEqual(before);
    expect(mcp.busy.size).toBe(0);
    expect(document.querySelector('#tile-' + MCP_ID + ' .f')?.textContent).toBe(
      '2 / 2 answered · 5 tools'
    );
  });
});

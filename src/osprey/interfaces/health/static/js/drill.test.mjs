// @ts-check
/* global KeyboardEvent, document -- vitest runs these in happy-dom */
/**
 * Miller-column drill-down (drill.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/drill.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { demoEnvelope } from './demo.js';
import { DEMO_MCP } from './demo-mcp.js';
import { store, mcp, S, DS, MCP_ID, indexModel, indexMcp, ds } from './state.js';
import { renderBands } from './overview.js';
import { openTile } from './drawer.js';
import { renderDrill, millKey, detailOf } from './drill.js';

beforeEach(() => {
  document.body.textContent = '';
  const b = document.createElement('section');
  b.id = 'bands';
  document.body.append(b);
  const d = demoEnvelope();
  store.env = d;
  store.R = d.results;
  store.LAYERS = d.layers;
  mcp.data = null;
  S.band = null;
  S.key = null;
  for (const k of Object.keys(DS)) delete DS[k];
  indexModel();
  renderBands();
});

describe('group columns', () => {
  test('the groups form a column; the selected group shows its rows with the fix', () => {
    openTile('agent-record');
    const items = [...document.querySelectorAll('#drill-agent-record .mcol .it')];
    expect(items.map((e) => e.querySelector('.nm')?.textContent)).toEqual(['store', 'transcripts']);
    expect(items[1].classList.contains('sel')).toBe(true);
    const detail = /** @type {HTMLElement} */ (
      document.querySelector('#drill-agent-record .mcol.detail')
    );
    expect(detail.querySelector('.dhd .dn')?.textContent).toBe('transcripts');
    expect(detail.querySelector('.fix')).not.toBeNull();
  });

  test('ArrowUp moves the selection within the column', () => {
    openTile('agent-record');
    millKey(new KeyboardEvent('keydown', { key: 'ArrowUp' }), 'agent-record');
    expect(ds('agent-record').path).toEqual(['agent-record/store']);
  });
});

describe('MCP columns', () => {
  test('servers › tools › detail; ArrowRight enters the tools column', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    openTile(MCP_ID, [MCP_ID + '/controls']);
    renderDrill(MCP_ID);
    millKey(new KeyboardEvent('keydown', { key: 'ArrowRight' }), MCP_ID);
    expect(ds(MCP_ID).path).toEqual([MCP_ID + '/controls', MCP_ID + '/controls/channel_read']);
    expect(document.querySelector('.tooldet h3')?.textContent).toBe('channel_read');
  });

  test('with nothing selected the detail column shows the server table', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    openTile(MCP_ID);
    expect(document.querySelectorAll('.mcol.detail .srow').length).toBe(3);
  });

  test('a tool detail carries its write tag', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    const d = detailOf(store.NODES[MCP_ID + '/controls/channel_write']);
    expect(d.querySelector('.tag')?.textContent).toBe('write · gated');
  });
});

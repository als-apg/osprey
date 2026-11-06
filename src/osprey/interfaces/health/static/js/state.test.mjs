// @ts-check
/**
 * Shared view state and the model derived from it (state.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/state.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { demoEnvelope } from './demo.js';
import { DEMO_MCP } from './demo-mcp.js';
import {
  store,
  mcp,
  pings,
  DS,
  MCP_ID,
  UNUSED_ID,
  indexModel,
  indexMcp,
  tileFact,
  tileSev,
  mcpTileState,
  ds,
  kidsOf,
  chainOf,
  compRows,
  layerRows,
  rowView,
} from './state.js';

function load() {
  const d = demoEnvelope();
  store.env = d;
  store.R = d.results;
  store.LAYERS = d.layers;
  mcp.data = null;
  pings.clear();
  for (const k of Object.keys(DS)) delete DS[k];
  indexModel();
  return d;
}

beforeEach(load);

describe('indexModel', () => {
  test('indexes every component and builds drill nodes for grouped ones', () => {
    expect(store.TILES['telemetry-store'].band).toBe('data');
    expect(store.NODES['telemetry-store'].children.map((n) => n.name)).toEqual([
      'store',
      'transcripts',
    ]);
    expect(store.NODES['telemetry-store/transcripts'].sev).toBe('MINOR');
    expect(store.NODES['disk']).toBeUndefined();
  });

  test('unused rows of a grouped component become a DISABLED pseudo-group', () => {
    const c = store.TILES['panels'].comp;
    c.unused = [7];
    indexModel();
    const n = store.NODES['panels/' + UNUSED_ID];
    expect(n.sev).toBe('DISABLED');
    expect(n.name).toBe('not used by this deployment');
  });
});

describe('tile facts', () => {
  test('a faulty tile shows its issue, a calm one its fact', () => {
    expect(tileFact(store.TILES['telemetry-store'].comp)).toBe('1 / 2 transcript roots readable');
    expect(tileFact(store.TILES['disk'].comp)).toBe('30 % used');
    expect(compRows(store.TILES['telemetry-store'].comp).length).toBe(4);
  });

  test('MCP tile reads the envelope until introspected, then the cache', () => {
    expect(mcpTileState().fact).toBe('3 servers');
    mcp.data = {
      probed_at: 1,
      servers: [...DEMO_MCP.servers, { name: 'dead', transport: 'stdio', tools: null }],
    };
    indexMcp();
    const st = mcpTileState();
    expect(st.sev).toBe('MAJOR');
    expect(st.issue).toBe('2 / 3 answered');
    expect(tileSev(store.TILES[MCP_ID].comp)).toBe('MAJOR');
  });
});

describe('drill lookups', () => {
  test('ds opens the first faulty group by default; MCP opens nothing', () => {
    expect(ds('telemetry-store').path).toEqual(['telemetry-store/transcripts']);
    mcp.data = DEMO_MCP;
    indexMcp();
    expect(ds(MCP_ID).path).toEqual([]);
  });

  test('kidsOf filters servers and tools by the query', () => {
    mcp.data = DEMO_MCP;
    indexMcp();
    expect(kidsOf(store.NODES[MCP_ID], 'archiver').map((n) => n.name)).toEqual(['controls']);
    expect(kidsOf(store.NODES[MCP_ID + '/controls'], 'archiver').map((n) => n.name)).toEqual([
      'archiver_read',
    ]);
    expect(chainOf(MCP_ID + '/controls/archiver_read')).toEqual([
      MCP_ID + '/controls',
      MCP_ID + '/controls/archiver_read',
    ]);
  });

  test('a pending deep link resolves when the servers arrive', () => {
    store.pendingMcpPath = [MCP_ID + '/workspace'];
    mcp.data = DEMO_MCP;
    indexMcp();
    expect(ds(MCP_ID).path).toEqual([MCP_ID + '/workspace']);
    expect(store.pendingMcpPath).toBeNull();
  });
});

describe('rows', () => {
  test('layerRows covers every referenced result', () => {
    expect(layerRows().length).toBe(store.R.length);
  });

  test('rowView prefers display, and a ping result over the poll', () => {
    const i = store.R.findIndex((r) => r.name === 'als-apg');
    store.R[i] = { ...store.R[i], display: 'authenticated · 1662 ms' };
    expect(rowView(i).value).toBe('authenticated · 1662 ms');
    pings.set('als-apg', { value: 'm · 9 ms', method: '1-token completion', status: 'ok', t: 5 });
    expect(rowView(i)).toMatchObject({
      value: 'm · 9 ms',
      method: '1-token completion',
      sev: 'OK',
      t: 5,
    });
  });
});

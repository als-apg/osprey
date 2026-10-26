// @ts-check
/**
 * Unit tests for the corpus strip's provenance line (stats-badges.js
 * renderProvenance): on the graph paradigm the strip names the store its
 * numbers came from and the tools the assistant reads that store with.
 *   npx vitest run tests/interfaces/channel_finder/corpus-provenance.test.mjs
 */

import { test, expect, beforeEach, afterEach } from 'vitest';

import { renderProvenance } from '../../../src/osprey/interfaces/channel_finder/static/js/stats-badges.js';
import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';

/** @returns {HTMLElement} */
function slot() {
  return /** @type {HTMLElement} */ (document.getElementById('corpus-prov'));
}

beforeEach(() => {
  document.body.innerHTML = '<div class="corpus-strip"><span id="corpus-prov"></span></div>';
  state.pipelineType = 'graph';
});

afterEach(() => {
  state.pipelineType = null;
  state.setGraphInfo([], null);
});

test('names the TTL file and the store, and lists the tools behind the info tip', () => {
  state.setGraphInfo(['read_cypher', 'get_schema'], { uri: 'bolt://localhost:7687', ttl_filename: 'als.ttl' });
  renderProvenance();

  expect(slot().querySelector('.graph-store-badge code')?.textContent).toBe('als.ttl @ bolt://localhost:7687');
  expect(slot().textContent).toContain('2 assistant tools');
  const tip = slot().querySelector('.osprey-info .osprey-tip');
  expect([...(tip?.querySelectorAll('.graph-tool-chip') ?? [])].map((c) => c.textContent)).toEqual(['read_cypher', 'get_schema']);
  expect(slot().querySelector('.osprey-info')?.getAttribute('aria-label')).toBeTruthy();
});

test('a store known by URI alone is named by its URI, and no tools means no tip', () => {
  state.setGraphInfo([], { uri: 'bolt://graph:7687', ttl_filename: null });
  renderProvenance();

  expect(slot().querySelector('code')?.textContent).toBe('bolt://graph:7687');
  expect(slot().querySelector('.osprey-info')).toBeNull();
});

test('the file-backed paradigms, and a graph with no store named, show nothing', () => {
  state.setGraphInfo(['read_cypher'], { uri: 'bolt://x', ttl_filename: null });
  state.pipelineType = 'hierarchical';
  renderProvenance();
  expect(slot().innerHTML).toBe('');

  state.pipelineType = 'graph';
  state.setGraphInfo(['read_cypher'], null);
  renderProvenance();
  expect(slot().innerHTML).toBe('');
});

test('store and tool names are inert text', () => {
  const payload = '<img src=x onerror=alert(1)>';
  state.setGraphInfo([payload], { uri: payload, ttl_filename: null });
  renderProvenance();
  expect(slot().querySelector('img')).toBeNull();
  expect(slot().querySelector('code')?.textContent).toBe(payload);
});

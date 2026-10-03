// @ts-check
/**
 * The Channel Finder's Explore views are read-only.
 *
 * No explore renderer imports a write helper from api.js, none renders an add,
 * edit or delete control, and each file-backed view names the corrections route:
 * data/facility/fixes.yaml, applied by `osprey build`. The pipeline switch in
 * app.js is not a database write and keeps its call.
 *   npx vitest run tests/interfaces/channel_finder/explore-read-only.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). app.js is mocked so its boot does not
 * run; fetch answers each read route with a small fixture.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import { test, expect, vi, afterEach } from 'vitest';

vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/app.js', () => ({
  showToast: vi.fn(),
}));

import { mountInContext, unmountInContext } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-in-context.js';
import { mountHierarchical, unmountHierarchical } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-hierarchical.js';
import { mountMiddleLayer, unmountMiddleLayer } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-middle-layer.js';

// `import.meta.dirname` is a plain path string, so it sidesteps happy-dom's URL.
const JS_DIR = join(import.meta.dirname, '../../../src/osprey/interfaces/channel_finder/static/js');

/** Selectors of every control that once changed a channel database. */
const EDIT_CONTROLS = [
  '.item-action-btn',
  '.column-add-btn',
  '#ic-add-channel',
  '#ml-add-family',
  '.ic-inline-input',
];

/** Read-route fixtures, matched by path prefix. */
const ROUTES = {
  '/api/channels': { channels: [{ channel: 'CH:A', address: 'CH:A', description: 'first' }], total: 1 },
  '/api/explore/hierarchy-info': {
    hierarchy_levels: ['system', 'device'],
    hierarchy_config: { levels: { system: { type: 'tree' }, device: { type: 'instances' } } },
  },
  '/api/explore/options': { options: [{ name: 'SYS', count: 2, description: 'a system' }] },
  '/api/explore/systems': { systems: [{ name: 'SYS', description: 'a system' }] },
  '/api/explore/families': { families: [{ name: 'FAM', description: 'a family' }] },
};

/** @type {string[]} */
let requested = [];

function stubReadRoutes() {
  requested = [];
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
    requested.push(url);
    const path = url.split('?')[0];
    const body = /** @type {Record<string, any>} */ (ROUTES)[path];
    if (body === undefined) return { ok: false, status: 404, statusText: 'Not Found', json: async () => ({ detail: 'Not Found' }) };
    return { ok: true, json: async () => body };
  }));
}

/** @returns {HTMLElement} */
function freshContainer() {
  document.body.innerHTML = '<div id="explore-content"></div>';
  return /** @type {HTMLElement} */ (document.getElementById('explore-content'));
}

/** @param {HTMLElement} container */
function expectReadOnly(container) {
  const text = container.textContent || '';
  expect(text).toContain('data/facility/fixes.yaml');
  expect(text).toContain('osprey build');
  for (const sel of EDIT_CONTROLS) {
    expect(container.querySelectorAll(sel), sel).toHaveLength(0);
  }
}

afterEach(() => {
  unmountInContext();
  unmountHierarchical();
  unmountMiddleLayer();
  vi.unstubAllGlobals();
});

test('no explore renderer imports a write helper from api.js', () => {
  const files = readdirSync(JS_DIR).filter(f => f.startsWith('explore') && f.endsWith('.js'));
  expect(files.length).toBeGreaterThan(0);
  for (const file of files) {
    const src = readFileSync(join(JS_DIR, file), 'utf8');
    expect(src, file).not.toMatch(/\b(postJSON|putJSON|deleteJSON)\b/);
  }
});

test('app.js keeps its pipeline switch call', () => {
  const src = readFileSync(join(JS_DIR, 'app.js'), 'utf8');
  expect(src).toMatch(/putJSON\('\/api\/pipeline'/);
});

test('in-context view names the corrections route and renders no edit control', async () => {
  stubReadRoutes();
  const container = freshContainer();
  await mountInContext(container);
  expect(container.textContent).toContain('CH:A');
  expectReadOnly(container);
});

test('hierarchical view names the corrections route and renders no edit control', async () => {
  stubReadRoutes();
  const container = freshContainer();
  await mountHierarchical(container);
  expect(container.textContent).toContain('SYS');
  expectReadOnly(container);
  expect(requested.some(u => u.startsWith('/api/tree/'))).toBe(false);
});

test('middle-layer view names the corrections route and renders no edit control', async () => {
  stubReadRoutes();
  const container = freshContainer();
  await mountMiddleLayer(container);
  const system = /** @type {HTMLElement} */ (container.querySelector('[data-system="SYS"]'));
  system.click();
  await vi.waitFor(() => expect(container.querySelector('[data-family="FAM"]')).not.toBeNull());
  expectReadOnly(container);
});

/**
 * Unit tests for the Lattice Dashboard's model handling: the header model
 * selector fed by GET /api/models and the POST /api/models/select round trip.
 *
 * happy-dom environment (configured globally), fetch stubbed with the
 * server's response bodies. header-contrib.js is mocked as in
 * header.test.mjs, so the tile-bar contribution's shape can be asserted:
 *   npx vitest run tests/interfaces/lattice_dashboard/models.test.mjs
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { byId } from '../_support/dom.mjs';

vi.mock('/design-system/js/header-contrib.js', () => ({
  contributeHeader: vi.fn(),
  onHeaderAction: vi.fn(),
}));

import {
  contributeHeader,
  onHeaderAction,
} from '/design-system/js/header-contrib.js';
import { createHeader } from '../../../src/osprey/interfaces/lattice_dashboard/static/js/header.js';
import { createNetClient } from '../../../src/osprey/interfaces/lattice_dashboard/static/js/net.js';
import {
  bindModelSelect,
  renderModelSelect,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/models.js';

/** GET /api/models as the server lists it: served models first. */
const MODELS = [
  { name: 'main', served: true, solve: 'periodic', selected: true },
  { name: 'transfer', served: true, solve: 'single_pass', selected: false },
  { name: 'spare', served: false, solve: 'periodic', selected: false },
];

/** A GET /api/state body for a selected periodic, served model. */
const PERIODIC_STATE = {
  base_lattice: '/decks/main.m',
  model: 'main',
  solve: 'periodic',
  served: true,
  fast_figures: ['optics', 'resonance', 'chromaticity', 'footprint'],
  notice: null,
  figures: {},
  families: {},
  summary: {},
};

function mountFixture() {
  document.body.innerHTML = `
    <select id="model-select" style="display:none"></select>
    <button id="btn-refresh"></button>
    <button id="btn-verify"></button>
    <button id="btn-baseline"><span class="baseline-label">Baseline</span></button>
  `;
}

/** @param {any} body @param {number} [status] */
function response(body, status = 200) {
  return {
    ok: status < 400,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

/** Route stubbed fetches by path. @param {Record<string, any>} routes */
function stubFetch(routes) {
  const fetchMock = vi.fn((/** @type {string} */ path, /** @type {RequestInit} */ init) => {
    if (!(path in routes)) throw new Error(`unexpected fetch ${path} ${init?.method ?? 'GET'}`);
    return Promise.resolve(routes[path]);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function makeNetCallbacks() {
  return {
    onState: vi.fn(),
    onModels: vi.fn(),
    onParamSet: vi.fn(),
    onFigureData: vi.fn(),
    onFigureUnavailable: vi.fn(),
    onFigureStatus: vi.fn(),
    onFigureReady: vi.fn(),
    onFigureError: vi.fn(),
    onSettingsUpdated: vi.fn(),
    onBaselineSet: vi.fn(),
  };
}

function makeHeaderCallbacks() {
  return {
    onRefresh: vi.fn(),
    onVerify: vi.fn(),
    onBaseline: vi.fn(),
    onSelectModel: vi.fn(),
  };
}

/** The items of the most recent contribution. @returns {any[]} */
function lastContribution() {
  const calls = vi.mocked(contributeHeader).mock.calls;
  return calls[calls.length - 1][0];
}

beforeEach(() => {
  mountFixture();
  vi.mocked(contributeHeader).mockClear();
  vi.mocked(onHeaderAction).mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('model selector', () => {
  test('the standalone selector lists the models served first and marks the unserved one', () => {
    renderModelSelect(MODELS);

    const select = /** @type {HTMLSelectElement} */ (byId('model-select'));
    expect(select.style.display).toBe('');
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      'main',
      'transfer',
      'spare (not served: optics only)',
    ]);
    expect(select.value).toBe('main');
  });

  test('picking a model in the standalone selector reports its name', () => {
    const onSelect = vi.fn();
    renderModelSelect(MODELS);
    bindModelSelect(onSelect);

    const select = /** @type {HTMLSelectElement} */ (byId('model-select'));
    select.value = 'spare';
    select.dispatchEvent(new Event('change'));

    expect(onSelect).toHaveBeenCalledWith('spare');
  });

  test('the tile bar carries the same choice as a menu, checked on the selected model', () => {
    const header = createHeader(makeHeaderCallbacks());
    header.init();
    header.syncModels(MODELS);

    const menu = lastContribution().find((/** @type {any} */ i) => i.id === 'model');
    expect(menu).toMatchObject({ kind: 'menu', label: 'main' });
    expect(menu.items).toEqual([
      { id: 'main', label: 'main', checked: true },
      { id: 'transfer', label: 'transfer', checked: false },
      { id: 'spare', label: 'spare (not served: optics only)', checked: false },
    ]);
  });

  test('no model menu is contributed while the build lists no model', () => {
    const header = createHeader(makeHeaderCallbacks());
    header.init();
    header.syncModels([]);

    expect(lastContribution().map((/** @type {any} */ i) => i.id)).not.toContain('model');
  });

  test('a tile-bar menu pick reaches the model-select handler with the entry id', () => {
    const cb = makeHeaderCallbacks();
    createHeader(cb).init();

    vi.mocked(onHeaderAction).mock.calls[0][0]('model', 'transfer');

    expect(cb.onSelectModel).toHaveBeenCalledWith('transfer');
  });

  test('selecting a model posts its name, then re-reads the state and the model list', async () => {
    const fetchMock = stubFetch({
      '/api/models/select': response(PERIODIC_STATE),
      '/api/state': response({ ...PERIODIC_STATE, model: 'spare', served: false }),
      '/api/models': response(MODELS),
    });
    const cb = makeNetCallbacks();
    const net = createNetClient(cb);

    await net.selectModel('spare');

    const select = fetchMock.mock.calls.find(([path]) => path === '/api/models/select');
    expect(select?.[1]).toMatchObject({ method: 'POST', body: JSON.stringify({ name: 'spare' }) });
    expect(cb.onState).toHaveBeenCalledWith(expect.objectContaining({ model: 'spare' }));
    expect(net.getState().model).toBe('spare');
    expect(cb.onModels).toHaveBeenCalledWith(MODELS);
  });

  test('a model the server does not know leaves the state alone and resets the selectors', async () => {
    stubFetch({
      '/api/models/select': response({ detail: 'Unknown model: gone' }, 404),
      '/api/models': response(MODELS),
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const cb = makeNetCallbacks();

    await createNetClient(cb).selectModel('gone');

    expect(cb.onState).not.toHaveBeenCalled();
    expect(cb.onModels).toHaveBeenCalledWith(MODELS);
  });
});

describe('fast figures', () => {
  test('Refresh marks computing exactly the fast figures the state names', async () => {
    stubFetch({
      '/api/state': response({ ...PERIODIC_STATE, solve: 'single_pass', fast_figures: ['optics'] }),
      '/api/refresh': response({ status: 'ok' }),
    });
    const cb = makeNetCallbacks();
    const net = createNetClient(cb);
    await net.fetchState();

    await net.refresh();

    expect(cb.onFigureStatus.mock.calls).toEqual([['optics', 'computing']]);
    expect(fetch).toHaveBeenCalledWith('/api/refresh', expect.objectContaining({ method: 'POST' }));
  });
});

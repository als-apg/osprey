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
  renderNotice,
  showFigureUnavailable,
  syncAvailability,
  unavailableFigures,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/models.js';

const ALL_FIGURES = ['optics', 'resonance', 'chromaticity', 'footprint', 'da', 'lma'];

/** The refusal body every non-optics figure route gives a single-pass model. */
const SINGLE_PASS_409 = { detail: 'not available for a single-pass model' };


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

/** One figure cell per figure, shaped like index.html's. */
function figureCells() {
  return ALL_FIGURES.map((name) => `
    <div class="figure-cell" id="cell-${name}" data-figure="${name}">
      <div class="figure-plot" id="plot-${name}">
        <div class="figure-placeholder">Waiting for lattice...</div>
      </div>
    </div>`).join('');
}

/** The text a figure panel shows. @param {string} name */
function panelText(name) {
  return byId(`plot-${name}`).textContent?.trim();
}

function mountFixture() {
  document.body.innerHTML = `
    <div id="model-notice" role="status" style="display:none"></div>
    ${figureCells()}
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
      'spare (not served)',
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
      { id: 'spare', label: 'spare (not served)', checked: false },
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

describe('figures the selected model cannot draw', () => {
  test('a periodic, served model hides no figure', () => {
    expect(unavailableFigures(PERIODIC_STATE, ALL_FIGURES)).toEqual([]);
  });

  test('a single-pass model hides every figure but optics, verification pair included', () => {
    const state = { ...PERIODIC_STATE, solve: 'single_pass', fast_figures: ['optics'] };
    expect(unavailableFigures(state, ALL_FIGURES)).toEqual([
      'resonance', 'chromaticity', 'footprint', 'da', 'lma',
    ]);
  });


  test("a figure route's 409 reaches the panel as its detail, not as a failure", async () => {
    stubFetch({ '/api/figures/resonance': response(SINGLE_PASS_409, 409) });
    const cb = makeNetCallbacks();

    await createNetClient(cb).fetchAndRenderFigure('resonance');

    expect(cb.onFigureUnavailable).toHaveBeenCalledWith(
      'resonance', 'not available for a single-pass model'
    );
    expect(cb.onFigureData).not.toHaveBeenCalled();
  });

  test('a single-pass model shows the 409 body in each hidden panel and leaves optics', async () => {
    const routes = Object.fromEntries(
      ALL_FIGURES.filter((n) => n !== 'optics')
        .map((n) => [`/api/figures/${n}`, response(SINGLE_PASS_409, 409)])
    );
    stubFetch(routes);
    const net = createNetClient({
      ...makeNetCallbacks(),
      onFigureUnavailable: showFigureUnavailable,
    });
    const state = { ...PERIODIC_STATE, solve: 'single_pass', fast_figures: ['optics'] };

    syncAvailability(state, ALL_FIGURES, net.fetchAndRenderFigure);
    await vi.waitFor(() => expect(panelText('lma')).toBe('not available for a single-pass model'));

    for (const name of ['resonance', 'chromaticity', 'footprint', 'da', 'lma']) {
      expect(panelText(name)).toBe('not available for a single-pass model');
      expect(byId(`cell-${name}`).dataset.available).toBe('false');
    }
    expect(panelText('optics')).toBe('Waiting for lattice...');
    expect(fetch).not.toHaveBeenCalledWith('/api/figures/optics', expect.anything());
  });

  test('an unserved model hides no figure', () => {
    const fetchFigure = vi.fn();
    const state = { ...PERIODIC_STATE, model: 'spare', served: false };

    syncAvailability(state, ALL_FIGURES, fetchFigure);

    expect(unavailableFigures(state, ALL_FIGURES)).toEqual([]);
    for (const name of ALL_FIGURES) {
      expect(byId(`plot-${name}`).querySelector('.figure-unavailable')).toBeNull();
      expect(panelText(name)).toBe('Waiting for lattice...');
    }
    expect(fetchFigure).not.toHaveBeenCalled();
  });

  test('switching back to a periodic model restores the hidden panels', () => {
    syncAvailability({ ...PERIODIC_STATE, solve: 'single_pass' }, ALL_FIGURES, (name) =>
      showFigureUnavailable(name, 'not available for a single-pass model'));

    syncAvailability(PERIODIC_STATE, ALL_FIGURES, vi.fn());

    for (const name of ALL_FIGURES) {
      expect(panelText(name)).toBe('Waiting for lattice...');
      expect(byId(`cell-${name}`).dataset.available).toBeUndefined();
    }
  });
});

describe('notice banner', () => {
  test.each([
    ['no lattice model is served'],
    ['no simulator view in this build'],
  ])('shows the state notice %j verbatim', (notice) => {
    renderNotice(notice);

    const banner = byId('model-notice');
    expect(banner.textContent).toBe(notice);
    expect(banner.style.display).toBe('');
  });

  test('a null notice empties and hides the banner', () => {
    renderNotice('no lattice model is served');

    renderNotice(null);

    const banner = byId('model-notice');
    expect(banner.textContent).toBe('');
    expect(banner.style.display).toBe('none');
  });
});

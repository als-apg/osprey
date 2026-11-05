/**
 * Unit tests for the Lattice Dashboard's model handling: the toolbar's model
 * switcher fed by GET /api/models, the POST /api/models/select round trip,
 * and the panels a selected model cannot draw.
 *
 * happy-dom environment (configured globally), fetch stubbed with the
 * server's response bodies:
 *   npx vitest run tests/interfaces/lattice_dashboard/models.test.mjs
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

import { createNetClient } from '../../../src/osprey/interfaces/lattice_dashboard/static/js/net.js';
import {
  bindModelSwitch,
  MAX_SEGMENT_MODELS,
  renderModelSwitch,
  renderNotice,
  selectionNotice,
  showFigureUnavailable,
  syncAvailability,
  unavailableFigures,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/models.js';

const ALL_FIGURES = ['optics', 'resonance', 'chromaticity', 'footprint', 'da', 'lma'];

/** The refusal body every non-optics figure route gives a single-pass model. */
const SINGLE_PASS_409 = { status: 'unavailable', reason: 'not available for a single-pass model' };

/**
 * GET /api/models as the server lists it: served models first.
 * @type {import('../../../src/osprey/interfaces/lattice_dashboard/static/js/models.js').ModelEntry[]}
 */
const MODELS = [
  { name: 'main', served: true, solve: 'periodic', selected: true, error: null },
  { name: 'transfer', served: true, solve: 'single_pass', selected: false, error: null },
  { name: 'spare', served: false, solve: 'periodic', selected: false, error: null },
];

/**
 * The selection of a GET /api/state body.
 * @param {string} model @param {string[]} figures @param {string} [status]
 */
function selection(model, figures, status = 'ready') {
  return {
    model,
    deck_sha256: 'd'.repeat(64),
    status,
    error: null,
    capabilities: {
      figures,
      fast_figures: figures.filter((n) => !['da', 'lma'].includes(n)),
      verify: figures.includes('da'),
    },
  };
}

/** A GET /api/state body for a selected periodic model. */
const PERIODIC_STATE = {
  selection: selection('main', ALL_FIGURES),
  notice: null,
  figures: {},
  families: {},
  summary: {},
};

/** A GET /api/state body for a selected single-pass model. */
const SINGLE_PASS_STATE = { ...PERIODIC_STATE, selection: selection('transfer', ['optics']) };

/** One figure cell per figure, shaped like index.html's. */
function figureCells() {
  return ALL_FIGURES.map(
    (name) => `
    <div class="figure-cell" id="cell-${name}" data-figure="${name}">
      <div class="figure-plot" id="plot-${name}">
        <div class="figure-placeholder">Waiting for lattice...</div>
      </div>
    </div>`
  ).join('');
}

/** The text a figure panel shows. @param {string} name */
function panelText(name) {
  return byId(`plot-${name}`).textContent?.trim();
}

function mountFixture() {
  document.body.innerHTML = `
    <div id="model-switch" hidden></div>
    <div id="model-notice" role="status" style="display:none"></div>
    ${figureCells()}
  `;
}

/** The switcher's segment buttons. @returns {HTMLButtonElement[]} */
function segments() {
  return Array.from(document.querySelectorAll('#model-switch .osprey-seg > button'));
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

/** The single-pass selection of MODELS: `transfer` is selected. */
const SINGLE_PASS_MODELS = MODELS.map((model) => ({
  ...model,
  selected: model.name === 'transfer',
}));

beforeEach(() => {
  mountFixture();
  // The module keeps the last list and handler; start each test from none.
  renderModelSwitch([]);
  bindModelSwitch(vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('model switcher', () => {
  test('one segment per model, served first, the selected one pressed', () => {
    renderModelSwitch(MODELS);

    expect(byId('model-switch').hidden).toBe(false);
    expect(segments().map((b) => b.textContent)).toEqual([
      'main',
      'transfer',
      'spare (not served)',
    ]);
    expect(segments().map((b) => b.getAttribute('aria-pressed'))).toEqual([
      'true',
      'false',
      'false',
    ]);
  });

  test('the switcher is hidden while the build lists no model', () => {
    renderModelSwitch([]);
    expect(byId('model-switch').hidden).toBe(true);
  });

  test('pressing another model reports its name; the pressed one does nothing', () => {
    const onSelect = vi.fn();
    renderModelSwitch(MODELS);
    bindModelSwitch(onSelect);

    segments()[0].click();
    segments()[2].click();

    expect(onSelect.mock.calls).toEqual([['spare']]);
  });

  test(`more than ${MAX_SEGMENT_MODELS} models become a select`, () => {
    const many = Array.from({ length: MAX_SEGMENT_MODELS + 1 }, (_, i) => ({
      name: `M${i}`,
      served: true,
      solve: 'periodic',
      selected: i === 2,
      error: null,
    }));
    const onSelect = vi.fn();
    renderModelSwitch(many);
    bindModelSwitch(onSelect);

    expect(segments()).toHaveLength(0);
    const select = qs(document, '#model-switch select', HTMLSelectElement);
    expect(select.options).toHaveLength(MAX_SEGMENT_MODELS + 1);
    expect(select.value).toBe('M2');

    select.value = 'M4';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onSelect).toHaveBeenCalledWith('M4');
  });

  test('selecting a model posts its name, then re-reads the state and the model list', async () => {
    const fetchMock = stubFetch({
      '/api/models/select': response(PERIODIC_STATE),
      '/api/state': response({ ...PERIODIC_STATE, selection: selection('spare', ALL_FIGURES) }),
      '/api/models': response(MODELS),
    });
    const cb = makeNetCallbacks();
    const net = createNetClient(cb);

    await net.selectModel('spare');

    const select = fetchMock.mock.calls.find(([path]) => path === '/api/models/select');
    expect(select?.[1]).toMatchObject({ method: 'POST', body: JSON.stringify({ name: 'spare' }) });
    expect(cb.onState).toHaveBeenCalledWith(
      expect.objectContaining({ selection: expect.objectContaining({ model: 'spare' }) })
    );
    expect(net.getState().selection.model).toBe('spare');
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
  test('Refresh marks computing exactly the fast figures the capabilities name', async () => {
    stubFetch({
      '/api/state': response(SINGLE_PASS_STATE),
      '/api/models': response(MODELS),
      '/api/refresh': response({ status: 'ok', launched: ['optics'] }),
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
  test('a periodic model hides no figure', () => {
    expect(unavailableFigures(PERIODIC_STATE, ALL_FIGURES)).toEqual([]);
  });

  test('a selection that is not ready hides no figure', () => {
    const state = { ...PERIODIC_STATE, selection: selection('main', [], 'loading') };
    expect(unavailableFigures(state, ALL_FIGURES)).toEqual([]);
  });

  test('the capabilities alone decide, whatever the model is called or served', () => {
    const state = { ...PERIODIC_STATE, selection: selection('main', ['optics', 'resonance']) };
    expect(unavailableFigures(state, ALL_FIGURES)).toEqual([
      'chromaticity',
      'footprint',
      'da',
      'lma',
    ]);
  });

  test('a single-pass model hides every figure but optics, verification pair included', () => {
    expect(unavailableFigures(SINGLE_PASS_STATE, ALL_FIGURES)).toEqual([
      'resonance',
      'chromaticity',
      'footprint',
      'da',
      'lma',
    ]);
  });

  test("a figure route's 409 reaches the panel as its detail, not as a failure", async () => {
    stubFetch({ '/api/figures/resonance': response(SINGLE_PASS_409, 409) });
    const cb = makeNetCallbacks();

    await createNetClient(cb).fetchAndRenderFigure('resonance');

    expect(cb.onFigureUnavailable).toHaveBeenCalledWith(
      'resonance',
      'not available for a single-pass model'
    );
    expect(cb.onFigureData).not.toHaveBeenCalled();
  });

  test('a single-pass model shows the 409 body in each hidden panel and leaves optics', async () => {
    const routes = Object.fromEntries(
      ALL_FIGURES.filter((n) => n !== 'optics').map((n) => [
        `/api/figures/${n}`,
        response(SINGLE_PASS_409, 409),
      ])
    );
    stubFetch(routes);
    const net = createNetClient({
      ...makeNetCallbacks(),
      onFigureUnavailable: showFigureUnavailable,
    });
    syncAvailability(SINGLE_PASS_STATE, ALL_FIGURES, net.fetchAndRenderFigure);
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
    const state = { ...PERIODIC_STATE, selection: selection('spare', ALL_FIGURES) };

    syncAvailability(state, ALL_FIGURES, fetchFigure);

    expect(unavailableFigures(state, ALL_FIGURES)).toEqual([]);
    for (const name of ALL_FIGURES) {
      expect(byId(`plot-${name}`).querySelector('.figure-unavailable')).toBeNull();
      expect(panelText(name)).toBe('Waiting for lattice...');
    }
    expect(fetchFigure).not.toHaveBeenCalled();
  });

  test('switching back to a periodic model restores the hidden panels', () => {
    syncAvailability(SINGLE_PASS_STATE, ALL_FIGURES, (name) =>
      showFigureUnavailable(name, 'not available for a single-pass model')
    );

    syncAvailability(PERIODIC_STATE, ALL_FIGURES, vi.fn());

    for (const name of ALL_FIGURES) {
      expect(panelText(name)).toBe('Waiting for lattice...');
      expect(byId(`cell-${name}`).dataset.available).toBeUndefined();
    }
  });

  test('a hidden panel shows the refusal in the centred state, with no action', () => {
    renderModelSwitch(SINGLE_PASS_MODELS);
    showFigureUnavailable('chromaticity', 'not available for a single-pass model');

    const state = qs(byId('plot-chromaticity'), '.figure-unavailable .osprey-state');
    expect(qs(state, '.osprey-state-message').textContent).toBe(
      'not available for a single-pass model'
    );
    expect(state.querySelector('svg')).not.toBeNull();
    expect(state.querySelector('button')).toBeNull();
  });

  test('the Resonance panel offers a switch to a periodic model, and it selects it', () => {
    const onSelect = vi.fn();
    bindModelSwitch(onSelect);
    renderModelSwitch(SINGLE_PASS_MODELS);

    showFigureUnavailable('resonance', 'not available for a single-pass model');

    const action = qs(byId('plot-resonance'), '.osprey-state button', HTMLButtonElement);
    expect(action.textContent).toBe('Switch to main');
    action.click();
    expect(onSelect).toHaveBeenCalledWith('main');
  });

  test('the switch prefers a model named SR', () => {
    renderModelSwitch([
      ...SINGLE_PASS_MODELS,
      { name: 'SR', served: true, solve: 'periodic', selected: false, error: null },
    ]);

    showFigureUnavailable('resonance', 'not available for a single-pass model');

    expect(qs(byId('plot-resonance'), '.osprey-state button').textContent).toBe('Switch to SR');
  });

  test('with no periodic model to switch to, the Resonance panel has no action', () => {
    renderModelSwitch(SINGLE_PASS_MODELS.filter((model) => model.solve === 'single_pass'));

    showFigureUnavailable('resonance', 'not available for a single-pass model');

    expect(byId('plot-resonance').querySelector('button')).toBeNull();
  });

  test('a model list that arrives after the refusal adds the switch', () => {
    showFigureUnavailable('resonance', 'not available for a single-pass model');
    expect(byId('plot-resonance').querySelector('button')).toBeNull();

    renderModelSwitch(SINGLE_PASS_MODELS);

    expect(qs(byId('plot-resonance'), '.osprey-state button').textContent).toBe('Switch to main');
  });
});

describe('notice banner', () => {
  test.each([['no lattice model is served'], ['no simulator view in this build']])(
    'shows the state notice %j verbatim',
    (notice) => {
      renderNotice(notice);

      const banner = byId('model-notice');
      expect(banner.textContent).toBe(notice);
      expect(banner.style.display).toBe('');
    }
  );

  test('a null notice empties and hides the banner', () => {
    renderNotice('no lattice model is served');

    renderNotice(null);

    const banner = byId('model-notice');
    expect(banner.textContent).toBe('');
    expect(banner.style.display).toBe('none');
  });

  test('a model that did not load puts its error in the banner', () => {
    const state = {
      notice: null,
      selection: { model: 'main', status: 'failed', error: 'ValueError: no deck' },
    };

    expect(selectionNotice(state)).toBe('main did not load: ValueError: no deck');
  });

  test('a model that failed with no known error says only that it did not load', () => {
    const state = { notice: null, selection: { model: 'main', status: 'failed', error: null } };

    expect(selectionNotice(state)).toBe('main did not load');
  });

  test('a failed pick with no model and no error says no model loaded', () => {
    const state = { notice: null, selection: { model: null, status: 'failed', error: null } };

    expect(selectionNotice(state)).toBe('No model loaded');
  });

  test('the build notice wins over a selection error', () => {
    const state = {
      notice: 'no lattice model is served',
      selection: { model: null, status: 'failed', error: 'no model' },
    };

    expect(selectionNotice(state)).toBe('no lattice model is served');
  });

  test('a loaded or loading selection adds nothing to the banner', () => {
    for (const status of ['ready', 'loading']) {
      expect(selectionNotice({ notice: null, selection: { model: 'main', status } })).toBeNull();
    }
  });

  test('a pick the build does not list puts its error in the banner', () => {
    const state = {
      notice: null,
      selection: { model: null, status: 'none', error: 'Unknown model: GHOST' },
    };

    expect(selectionNotice(state)).toBe('No model loaded: Unknown model: GHOST');
  });

  test('no selection without an error adds nothing to the banner', () => {
    const state = { notice: null, selection: { model: null, status: 'none', error: null } };

    expect(selectionNotice(state)).toBeNull();
  });
});

// @ts-check
/**
 * Unit tests for the graph finder's mount (explore-graph.js).
 *
 * The module owns everything between the graph endpoints and the DOM: the panel
 * shell (title, tool chips, store badge), the search box and its debounce, the
 * facet rail, the result page and its selection, the device card, the copy and
 * send actions, and the informational pane the empty and unreachable stores
 * share. The state and markup modules it drives are pure and tested on their
 * own, so what is asserted here is the wiring: which request goes out when,
 * which reply is allowed to land, and what reaches the page. Run with:
 *   npx vitest run tests/interfaces/channel_finder/explore-graph-renderer.test.mjs
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  showToast: vi.fn(),
  refreshStatsBadges: vi.fn(),
  copyText: vi.fn(async () => true),
  isEmbedded: vi.fn(() => false),
}));

// app.js boots the whole panel on import and stats-badges.js talks to
// /api/statistics; both are mocked so the mount is exercised on its own and the
// toast and the retry's badge refresh can be asserted directly.
vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/app.js', () => ({
  showToast: mocks.showToast,
}));

vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/stats-badges.js', () => ({
  refreshStatsBadges: mocks.refreshStatsBadges,
}));

vi.mock('/design-system/js/clipboard.js', () => ({ copyText: mocks.copyText }));

vi.mock('/design-system/js/frame-params.js', () => ({ isEmbedded: mocks.isEmbedded }));

import { mountGraph, unmountGraph } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-graph.js';
import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';

const ONTOLOGY_PATH = '/api/graph/ontology';
const SEARCH_PATH = '/api/graph/search';
const DEVICE_PATH = '/api/graph/device';
const SEM = 'https://narad.example.org/schema/shared_semantics/';
const DEV = 'https://narad.example.org/data/device/';

/** The relationship vocabulary the demo corpus seeds. */
const DEMO_RELATIONSHIPS = ['HASBINDING', 'READSSIGNAL', 'SUBCLASSOF', 'TYPE', 'WRITESSIGNAL'];

/**
 * Demo-shaped ontology payload: a taxonomy under one root, exactly as
 * `GET /api/graph/ontology` answers for the seeded demo corpus.
 *
 * @param {Record<string, any>} [overrides] - Payload fields to replace.
 * @returns {Record<string, any>} The response body.
 */
function ontologyPayload(overrides = {}) {
  // The three groupings hold no devices of their own; the two leaves do.
  /** @type {[string, string[], boolean][]} */
  const spec = [
    ['AcceleratorDevice', [], true],
    ['Magnet', ['AcceleratorDevice'], true],
    ['Diagnostic', ['AcceleratorDevice'], true],
    ['Quadrupole', ['Magnet'], false],
    ['BeamPositionMonitor', ['Diagnostic'], false],
  ];
  return {
    classes: spec.map(([name, parents, abstract], i) => ({
      uri: SEM + name,
      name,
      altLabel: [],
      parents: parents.map((p) => SEM + p),
      rollup: (i + 1) * 8,
      abstract,
    })),
    relationship_types: DEMO_RELATIONSHIPS.slice(),
    truncated: false,
    empty: false,
    suggestions: [],
    ...overrides,
  };
}

/**
 * One page row shaped as `GRAPH_SEARCH_CYPHER` projects it, matching the
 * Python fixture's `DEMO_SEARCH_ROW` rows.
 *
 * @param {string} device - Device name.
 * @param {string} suffix - Address suffix appended to the device name.
 * @param {string|null} signal - Semantic signal name, or null for none.
 * @param {string[]} edges - Signal edges the address carries.
 * @returns {Record<string, any>} The row.
 */
function searchRow(device, suffix, signal, edges) {
  return {
    fullPv: `${device}${suffix}`,
    description: `${signal ?? 'status'} on ${device}`,
    device,
    device_uri: DEV + device,
    section: device.slice(0, 5),
    system: 'MG',
    edges: edges.slice(),
    signals: signal ? [{ uri: `${SEM}${signal}`, name: signal }] : [],
  };
}

const DEMO_ROWS = [
  searchRow('SR01C___QFA____', 'AM00', 'current', ['READSSIGNAL']),
  searchRow('SR01C___QFA____', 'SP00', 'current', ['WRITESSIGNAL']),
  searchRow('SR02C___BPM1___', 'AM00', 'beamPositionX', ['READSSIGNAL']),
];

/**
 * Demo-shaped search payload.
 *
 * @param {Record<string, any>} [overrides] - Payload fields to replace.
 * @returns {Record<string, any>} The response body.
 */
function searchPayload(overrides = {}) {
  return {
    total: 128,
    devices: 64,
    page: 1,
    pages: 3,
    page_size: 50,
    truncated: false,
    rows: DEMO_ROWS.map((row) => ({ ...row })),
    facets: {
      section: [{ value: 'SR01C', count: 2 }, { value: 'SR02C', count: 1 }],
      system: [{ value: 'MG', count: 3 }],
      class: [{ value: `${SEM}Quadrupole`, count: 1 }, { value: `${SEM}Magnet`, count: 1 }],
      signal: [{ value: 'current', count: 2 }, { value: 'beamPositionX', count: 1 }],
      dir: [{ value: 'R', count: 2 }, { value: 'W', count: 1 }],
    },
    empty: false,
    suggestions: [],
    ...overrides,
  };
}

/**
 * One device shaped as `GET /api/graph/device` answers it: channels grouped
 * under their signal, each binding carrying the graph edges it was found by.
 *
 * @param {Record<string, any>} [overrides] - Payload fields to replace.
 * @returns {Record<string, any>} The response body.
 */
function devicePayload(overrides = {}) {
  return {
    uri: `${DEV}SR01C___QFA____`,
    device: 'SR01C___QFA____',
    class: 'Quadrupole',
    class_uri: `${SEM}Quadrupole`,
    rawType: 'QFA',
    section: 'SR01C',
    system: 'MG',
    sPositionM: 12.734,
    ordinalInSection: 1,
    systemDescription: 'Storage ring magnets',
    familyDescription: 'Focusing quadrupole, family A',
    signals: [{
      uri: `${SEM}current`,
      name: 'current',
      bindings: [
        {
          fullPv: 'SR01C___QFA____AM00',
          edges: ['READSSIGNAL'],
          description: 'current readback',
          fieldDescription: 'Current',
          subfieldDescription: 'Readback',
        },
        {
          fullPv: 'SR01C___QFA____SP00',
          edges: ['WRITESSIGNAL'],
          description: 'current setpoint',
          fieldDescription: 'Current',
          subfieldDescription: 'Setpoint',
        },
      ],
    }],
    ...overrides,
  };
}

/**
 * Stub `fetch` with a per-path router. A path the router has no answer for
 * replies 200 with an empty body rather than throwing.
 *
 * @param {(path: string) => {ok: boolean, status?: number, body: any}|Promise<any>} route
 *   Reply for a path, or a promise the caller settles itself.
 * @returns {any} The vi mock, for call assertions.
 */
function stubFetch(route) {
  const fn = vi.fn(async (/** @type {any} */ input) => {
    const reply = await route(String(input));
    return {
      ok: reply.ok !== false,
      status: reply.status ?? (reply.ok === false ? 503 : 200),
      statusText: '',
      json: async () => reply.body,
    };
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

/**
 * Stub `fetch` so the demo ontology and one search payload answer.
 *
 * @param {Record<string, any>} [search] - The search body; the demo page by default.
 * @param {Record<string, any>} [ontology] - The ontology body.
 * @returns {any} The vi mock.
 */
function stubDemo(search = searchPayload(), ontology = ontologyPayload()) {
  return stubFetch((path) => {
    if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontology };
    if (path.startsWith(SEARCH_PATH)) return { ok: true, body: search };
    if (path.startsWith(DEVICE_PATH)) return { ok: true, body: devicePayload() };
    return { ok: true, body: {} };
  });
}

/**
 * Every request made to one path, in order.
 *
 * @param {any} fetchMock - The mock returned by `stubFetch`.
 * @param {string} path - The path prefix to keep.
 * @returns {string[]} The full urls.
 */
function callsTo(fetchMock, path) {
  return fetchMock.mock.calls
    .map((/** @type {any[]} */ call) => String(call[0]))
    .filter((/** @type {string} */ url) => url.startsWith(path));
}

/**
 * The abort signal the mount passed with its first request to `path`.
 *
 * @param {any} fetchMock - The mock returned by `stubFetch`.
 * @param {string} path - The path prefix to look for.
 * @param {number} [index] - Which request to that path, 0-based.
 * @returns {AbortSignal} The signal that request was given.
 */
function signalFor(fetchMock, path, index = 0) {
  const calls = fetchMock.mock.calls
    .filter((/** @type {any[]} */ call) => String(call[0]).startsWith(path));
  const call = calls[index];
  if (!call) throw new Error(`no request ${index} to ${path}`);
  return call[1].signal;
}

/** @returns {HTMLElement} The freshly mounted host element. */
function host() {
  const el = document.getElementById('explore-content');
  if (!el) throw new Error('no host element');
  return /** @type {HTMLElement} */ (el);
}

/**
 * Click one element of the mounted panel.
 * @param {string} selector - CSS selector inside the host.
 * @returns {void}
 */
function click(selector) {
  const el = /** @type {HTMLElement|null} */ (host().querySelector(selector));
  if (!el) throw new Error(`no element for ${selector}`);
  el.click();
}

/**
 * Tick a checkbox and fire the change the mount listens for.
 * @param {string} selector - CSS selector inside the host.
 * @param {boolean} checked - The new checked state.
 * @returns {void}
 */
function check(selector, checked) {
  const el = /** @type {HTMLInputElement|null} */ (host().querySelector(selector));
  if (!el) throw new Error(`no checkbox for ${selector}`);
  el.checked = checked;
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

/**
 * Type into the query box and fire the input event the mount debounces.
 * @param {string} value - The new query.
 * @returns {void}
 */
function type(value) {
  const input = /** @type {HTMLInputElement|null} */ (host().querySelector('#graph-finder-q'));
  if (!input) throw new Error('no query box');
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

/** @returns {string[]} The addresses of the rows currently drawn. */
function drawnPvs() {
  return [...host().querySelectorAll('.result-table tbody tr[data-pv]')]
    .map((row) => row.getAttribute('data-pv') ?? '');
}

beforeEach(() => {
  document.body.innerHTML = '<div id="explore-content"></div>';
  state.setGraphInfo(['read_cypher', 'get_schema'], {
    uri: 'bolt://localhost:7687',
    ttl_filename: 'als_corpus.ttl',
  });
  mocks.isEmbedded.mockReturnValue(false);
  mocks.copyText.mockResolvedValue(true);
});

afterEach(() => {
  unmountGraph();
  state.setGraphInfo([], null);
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('the first render', () => {
  test('mounting asks for the ontology and the first page, and draws both', async () => {
    const fetchMock = stubDemo();

    await mountGraph(host());
    const container = host();

    // Both reads go out, and the search carries the empty state's parameters.
    expect(callsTo(fetchMock, ONTOLOGY_PATH)).toHaveLength(1);
    const search = callsTo(fetchMock, SEARCH_PATH);
    expect(search).toHaveLength(1);
    expect(search[0]).toBe(`${SEARCH_PATH}?q=&page=1`);

    // The rail carries every facet group, with the class tree nested by depth.
    expect(container.querySelectorAll('.facet').length).toBe(5);
    expect(container.querySelector('.facet-item[data-facet="section"][data-value="SR01C"]'))
      .not.toBeNull();
    const root = container.querySelector(`.facet-item[data-value="${SEM}AcceleratorDevice"]`);
    expect(root?.getAttribute('data-depth')).toBe('0');
    expect(container.querySelector(`.facet-item[data-value="${SEM}Quadrupole"]`)
      ?.getAttribute('data-depth')).toBe('2');

    // The page of rows, the counts, and the pager.
    expect(drawnPvs()).toEqual(DEMO_ROWS.map((row) => row.fullPv));
    expect(container.querySelector('.finder-count')?.textContent)
      .toBe(`${(128).toLocaleString()} channels`);
    expect(container.querySelector('.result-counts')?.textContent)
      .toContain(`${(64).toLocaleString()} devices`);
    expect(container.querySelector('.pager-pos')?.textContent).toBe('1 / 3');

    // Panel chrome: tool chips and the store badge, and no failure pane.
    expect([...container.querySelectorAll('.graph-tool-chip')].map((c) => c.textContent))
      .toEqual(['read_cypher', 'get_schema']);
    const badge = container.querySelector('.graph-store-badge');
    expect(badge?.textContent).toContain('als_corpus.ttl');
    expect(badge?.textContent).toContain('bolt://localhost:7687');
    expect(container.querySelector('.explore-unknown')).toBeNull();
    expect(container.querySelector('.graph-truncated')).toBeNull();
  });

  test('either read reporting a clipped list carries the caution', async () => {
    stubDemo(searchPayload(), ontologyPayload({ truncated: true }));
    await mountGraph(host());
    expect(host().querySelector('.graph-truncated')).not.toBeNull();
    unmountGraph();

    document.body.innerHTML = '<div id="explore-content"></div>';
    stubDemo(searchPayload({ truncated: true }));
    await mountGraph(host());

    expect(host().querySelector('.graph-truncated')).not.toBeNull();
    // A caution, not a failure: the rows are still drawn.
    expect(drawnPvs()).toHaveLength(3);
  });

  test('a parent the payload never sent is ignored, and a second parent is not drawn twice', async () => {
    // Multi declares two parents that are both present, so it hangs under the
    // one whose uri sorts first and appears once. Orphan's only parent is not
    // in the payload at all, which leaves it a root rather than dropping it.
    stubDemo(searchPayload(), ontologyPayload({
      classes: [
        { uri: `${SEM}Alpha`, name: 'Alpha', altLabel: [], parents: [], rollup: 8 },
        { uri: `${SEM}Zed`, name: 'Zed', altLabel: [], parents: [], rollup: 8 },
        {
          uri: `${SEM}Multi`,
          name: 'Multi',
          altLabel: [],
          parents: [`${SEM}Zed`, `${SEM}Alpha`],
          rollup: 4,
        },
        { uri: `${SEM}Orphan`, name: 'Orphan', altLabel: [], parents: [`${SEM}Absent`], rollup: 2 },
      ],
    }));

    await mountGraph(host());

    // Pre-order, roots by name: Alpha, its child Multi, then Orphan and Zed.
    const drawn = [...host().querySelectorAll('.facet-item[data-facet="cls"]')].map((item) => [
      (item.getAttribute('data-value') ?? '').slice(SEM.length),
      item.getAttribute('data-depth'),
    ]);
    expect(drawn).toEqual([['Alpha', '0'], ['Multi', '1'], ['Orphan', '0'], ['Zed', '0']]);
  });

  test('a cyclic taxonomy still draws, and a class below the cycle keeps its parent', async () => {
    // Alpha and Beta declare each other a parent, and Ex hangs under Beta. The
    // cycle has to be cut somewhere, but Ex is not part of it and must not be
    // promoted to a root just for descending from one.
    stubDemo(searchPayload(), ontologyPayload({
      classes: [
        { uri: `${SEM}Alpha`, name: 'Alpha', altLabel: [], parents: [`${SEM}Beta`], rollup: 4 },
        { uri: `${SEM}Beta`, name: 'Beta', altLabel: [], parents: [`${SEM}Alpha`], rollup: 4 },
        { uri: `${SEM}Ex`, name: 'Ex', altLabel: [], parents: [`${SEM}Beta`], rollup: 2 },
      ],
    }));

    await mountGraph(host());
    const container = host();

    /** @param {string} name @returns {string|null} */
    const depthOf = (name) =>
      container.querySelector(`.facet-item[data-value="${SEM}${name}"]`)?.getAttribute('data-depth')
      ?? null;

    // Exactly one class is cut loose, and it is one of the two on the cycle.
    const roots = [...container.querySelectorAll('.facet-item[data-depth="0"]')];
    expect(roots).toHaveLength(1);
    expect(roots[0].getAttribute('data-value')).toBe(`${SEM}Alpha`);
    expect(depthOf('Beta')).toBe('1');
    // Ex still hangs under Beta rather than becoming a second root.
    expect(depthOf('Ex')).toBe('2');
  });

  test('the rail marks a class that holds no devices of its own', async () => {
    stubDemo();

    await mountGraph(host());
    const container = host();

    /** @param {string} name @returns {boolean} */
    const isAbstract = (name) => Boolean(
      container.querySelector(`.facet-item[data-value="${SEM}${name}"]`)
        ?.classList.contains('abstract'),
    );

    expect(isAbstract('Magnet')).toBe(true);
    expect(isAbstract('Quadrupole')).toBe(false);
  });

  test('store-sourced strings are inert, never markup', async () => {
    const attack = '<img src=x onerror=alert(1)>';
    stubDemo(
      searchPayload({
        rows: [{ ...DEMO_ROWS[0], description: attack, device: attack }],
        facets: { ...searchPayload().facets, section: [{ value: attack, count: 1 }] },
      }),
      ontologyPayload({
        classes: [{ uri: `${SEM}Magnet`, name: attack, altLabel: [], parents: [], rollup: 4 }],
      }),
    );

    await mountGraph(host());
    const container = host();

    expect(container.querySelectorAll('img, script')).toHaveLength(0);
    const hasOnAttr = [...container.querySelectorAll('*')].some((el) =>
      [...el.attributes].some((attr) => attr.name.startsWith('on')));
    expect(hasOnAttr, 'no on* event-handler attribute').toBe(false);
    // The value still reaches the reader, as inert text...
    expect(container.textContent).toContain(attack);
    // ...and travels back to the server as data: the facet the mount sends is
    // the string the store gave it, not a fragment the parser rewrote.
    expect(container.querySelector('.facet-item[data-facet="section"]')
      ?.getAttribute('data-value')).toBe(attack);
  });
});

describe('searching', () => {
  test('typing costs one search after the debounce, not one per keystroke', async () => {
    const fetchMock = stubDemo();
    await mountGraph(host());
    vi.useFakeTimers();

    type('qu');
    type('qua');
    type('quad');
    // Nothing has gone out yet: the query is still settling.
    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(200);

    const searches = callsTo(fetchMock, SEARCH_PATH);
    expect(searches).toHaveLength(2);
    expect(searches[1]).toBe(`${SEARCH_PATH}?q=quad&page=1`);
  });

  test('a late reply to an abandoned query never overwrites the current one', async () => {
    /** @type {(() => void)[]} */
    const pending = [];
    const bodies = [searchPayload(), searchPayload({ rows: [DEMO_ROWS[2]], total: 1, pages: 1 })];
    let searchIndex = 0;

    const fetchMock = stubFetch((path) => {
      if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontologyPayload() };
      if (!path.startsWith(SEARCH_PATH)) return { ok: true, body: {} };
      const index = searchIndex++;
      // The first two searches are held open so they can be settled in reverse.
      if (index >= 1 && index <= 2) {
        return new Promise((resolve) => {
          pending[index - 1] = () => resolve({ ok: true, body: bodies[index - 1] });
        });
      }
      return { ok: true, body: bodies[1] };
    });

    await mountGraph(host());
    vi.useFakeTimers();

    type('first');
    await vi.advanceTimersByTimeAsync(200);
    type('second');
    await vi.advanceTimersByTimeAsync(200);
    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(3);

    // The newer query answers first, then the abandoned one answers late.
    pending[1]();
    await vi.advanceTimersByTimeAsync(0);
    pending[0]();
    await vi.advanceTimersByTimeAsync(0);

    expect(drawnPvs()).toEqual([DEMO_ROWS[2].fullPv]);
    expect(host().querySelector('.pager-pos')?.textContent).toBe('1 / 1');
  });

  test('a facet click filters, and its chip removes the filter again', async () => {
    const fetchMock = stubDemo();
    await mountGraph(host());

    click('.facet-item[data-facet="section"][data-value="SR01C"]');
    await vi.waitFor(() =>
      expect(host().querySelector('.active-filter[data-chip="section:SR01C"]')).not.toBeNull());

    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(2);
    expect(callsTo(fetchMock, SEARCH_PATH)[1]).toBe(`${SEARCH_PATH}?q=&section=SR01C&page=1`);
    expect(host().querySelector('.active-filter[data-chip="section:SR01C"]')?.textContent)
      .toContain('Section: SR01C');

    click('.active-filter[data-chip="section:SR01C"]');
    await vi.waitFor(() => expect(host().querySelector('.active-filter')).toBeNull());

    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(3);
    expect(callsTo(fetchMock, SEARCH_PATH)[2]).toBe(`${SEARCH_PATH}?q=&page=1`);
  });

  test('the pager steps forward', async () => {
    const fetchMock = stubDemo();
    await mountGraph(host());

    click('[data-page="next"]');

    await vi.waitFor(() => expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(2));
    expect(callsTo(fetchMock, SEARCH_PATH)[1]).toBe(`${SEARCH_PATH}?q=&page=2`);
  });

  test('a page the store no longer has is clamped and asked for again', async () => {
    stubDemo();
    await mountGraph(host());

    // The corpus shrank under the open page: the store now answers one page,
    // so page 1 is fetched rather than page 2 being drawn from a stale answer.
    const narrowed = stubDemo(searchPayload({ pages: 1, total: 2, devices: 1 }));
    click('[data-page="next"]');

    await vi.waitFor(() => expect(callsTo(narrowed, SEARCH_PATH)).toHaveLength(2));
    expect(callsTo(narrowed, SEARCH_PATH)[0]).toBe(`${SEARCH_PATH}?q=&page=2`);
    expect(callsTo(narrowed, SEARCH_PATH)[1]).toBe(`${SEARCH_PATH}?q=&page=1`);
    expect(host().querySelector('.pager-pos')?.textContent).toBe('1 / 1');
  });

  test('unmounting mid-flight aborts the request and renders nothing', async () => {
    /** @type {() => void} */
    let settle = () => {};
    const fetchMock = stubFetch((path) => {
      if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontologyPayload() };
      if (path.startsWith(SEARCH_PATH)) {
        return new Promise((resolve) => {
          settle = () => resolve({ ok: true, body: searchPayload() });
        });
      }
      return { ok: true, body: {} };
    });

    const mounted = mountGraph(host());
    unmountGraph();

    // The request is actually cancelled, not merely ignored on arrival.
    expect(signalFor(fetchMock, SEARCH_PATH).aborted).toBe(true);

    settle();
    await mounted;

    expect(host().innerHTML).toBe('');
    expect(host().querySelector('.result-table')).toBeNull();
  });

  test('a query still settling when the panel goes away never leaves', async () => {
    const fetchMock = stubDemo();
    await mountGraph(host());
    vi.useFakeTimers();

    type('quad');
    unmountGraph();
    await vi.advanceTimersByTimeAsync(200);

    // Only the mount's own search was ever sent: the debounce died with the
    // panel rather than firing into a torn-down view.
    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(1);
  });
});

describe('the informational pane', () => {
  test('an unseeded store offers the seed command, not an error', async () => {
    stubDemo(
      searchPayload({
        total: 0,
        devices: 0,
        pages: 0,
        rows: [],
        empty: true,
        suggestions: ['Seed it with `osprey knowledge seed-graph`.'],
      }),
      ontologyPayload({ classes: [], relationship_types: [], empty: true }),
    );

    await mountGraph(host());
    const container = host();

    const pane = container.querySelector('.explore-unknown--info');
    expect(pane).not.toBeNull();
    expect(pane?.textContent).toContain('osprey knowledge seed-graph');
    expect(container.querySelector('#graph-retry')).not.toBeNull();
    // Nothing is drawn, and nothing claims to be broken.
    expect(container.querySelector('.result-table')).toBeNull();
    expect(pane?.getAttribute('role')).toBeNull();
  });

  test('an unreachable store shows its detail and remedies, and Retry re-asks', async () => {
    const fetchMock = stubFetch((path) => (
      path.startsWith(ONTOLOGY_PATH) || path.startsWith(SEARCH_PATH)
        ? {
          ok: false,
          status: 503,
          body: {
            detail: 'Graph store is not reachable at bolt://localhost:7687.',
            error_type: 'service_unavailable',
            suggestions: ['Start the graphdb service.', 'Check services.graphdb.uri.'],
          },
        }
        : { ok: true, body: {} }
    ));

    await mountGraph(host());
    const pane = host().querySelector('.explore-unknown--info');

    expect(pane?.textContent).toContain('Graph store is not reachable at bolt://localhost:7687.');
    expect(pane?.textContent).toContain('Start the graphdb service.');
    expect(pane?.textContent).toContain('Check services.graphdb.uri.');
    expect(callsTo(fetchMock, ONTOLOGY_PATH)).toHaveLength(1);
    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(1);

    click('#graph-retry');

    // Retry re-asks both reads and the header statistics, which fail together.
    await vi.waitFor(() => expect(callsTo(fetchMock, ONTOLOGY_PATH)).toHaveLength(2));
    expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(2);
    expect(mocks.refreshStatsBadges).toHaveBeenCalledTimes(1);
  });

  test("the pane's detail and remedies are inert, never markup", async () => {
    const attack = '<img src=x onerror=alert(1)>';
    stubFetch(() => ({
      ok: false,
      status: 503,
      body: { detail: `Store said: ${attack}`, suggestions: [`Try ${attack}`] },
    }));

    await mountGraph(host());
    const container = host();

    expect(container.querySelectorAll('img, script')).toHaveLength(0);
    const hasOnAttr = [...container.querySelectorAll('*')].some((el) =>
      [...el.attributes].some((attr) => attr.name.startsWith('on')));
    expect(hasOnAttr, 'no on* event-handler attribute').toBe(false);
    // Both halves still reach the reader, as text.
    expect(container.querySelector('.explore-unknown-title')?.textContent)
      .toBe(`Store said: ${attack}`);
    expect(container.querySelector('.explore-unknown-body')?.textContent).toContain(`Try ${attack}`);
  });

  test('a recovered store draws the finder on the retry', async () => {
    let down = true;
    stubFetch((path) => {
      if (down && (path.startsWith(ONTOLOGY_PATH) || path.startsWith(SEARCH_PATH))) {
        return { ok: false, status: 503, body: { detail: 'down', suggestions: [] } };
      }
      if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontologyPayload() };
      if (path.startsWith(SEARCH_PATH)) return { ok: true, body: searchPayload() };
      return { ok: true, body: {} };
    });

    await mountGraph(host());
    expect(host().querySelector('.explore-unknown--info')).not.toBeNull();

    down = false;
    click('#graph-retry');

    await vi.waitFor(() => expect(host().querySelector('.result-table')).not.toBeNull());
    expect(host().querySelector('.explore-unknown--info')).toBeNull();
    expect(drawnPvs()).toHaveLength(3);
  });
});

describe('the device card', () => {
  test('a device name opens its card, and the ✕ closes it', async () => {
    stubDemo();
    await mountGraph(host());

    click('.result-table td.dev button.dev');
    await vi.waitFor(() => expect(host().querySelector('.device-card')).not.toBeNull());

    const card = host().querySelector('.device-card');
    if (!card) throw new Error('no device card');
    expect(card.querySelector('.name')?.textContent).toBe('SR01C___QFA____');
    expect(card.querySelector('.meta')?.textContent).toContain('Quadrupole (QFA)');
    // The direction drawn for each binding is the one its graph edges mean.
    const dirs = [...card.querySelectorAll('.sig-table .dir')].map((d) => d.textContent);
    expect(dirs).toEqual(['R', 'W']);
    expect(card.textContent).toContain('SR01C___QFA____SP00');

    click('.device-card [data-action="close-card"]');
    expect(host().querySelector('.device-card')).toBeNull();
    // Closing the card leaves the finder standing.
    expect(drawnPvs()).toHaveLength(3);
  });

  test('a device the store no longer holds is a miss, not a broken panel', async () => {
    stubFetch((path) => {
      if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontologyPayload() };
      if (path.startsWith(SEARCH_PATH)) return { ok: true, body: searchPayload() };
      return {
        ok: false,
        status: 404,
        body: {
          detail: `No device at ${DEV}SR01C___QFA____`,
          error_type: 'not_found',
          suggestions: ['Search for the device by name.'],
        },
      };
    });

    await mountGraph(host());
    click('.result-table td.dev button.dev');

    await vi.waitFor(() => expect(host().querySelector('.device-card-error')).not.toBeNull());
    expect(host().querySelector('.card-error')?.textContent)
      .toBe(`No device at ${DEV}SR01C___QFA____`);
    // The finder underneath is untouched, and the miss closes like a hit.
    expect(drawnPvs()).toHaveLength(3);
    click('.device-card [data-action="close-card"]');
    expect(host().querySelector('.device-card')).toBeNull();
  });

  test('closing the card abandons a lookup still in flight', async () => {
    /** @type {() => void} */
    let settle = () => {};
    let devices = 0;
    const fetchMock = stubFetch((path) => {
      if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontologyPayload() };
      if (path.startsWith(SEARCH_PATH)) return { ok: true, body: searchPayload() };
      // The first lookup answers; the second is held open.
      if (devices++ === 0) return { ok: true, body: devicePayload() };
      return new Promise((resolve) => {
        settle = () => resolve({ ok: true, body: devicePayload({ device: 'LATE' }) });
      });
    });

    await mountGraph(host());
    click('.result-table td.dev button.dev');
    await vi.waitFor(() => expect(host().querySelector('.device-card')).not.toBeNull());

    // A second device is asked for while the first card is still on screen, so
    // the ✕ is there to close it while that lookup is in flight.
    click('.result-table tbody tr:last-child td.dev button.dev');
    click('.device-card [data-action="close-card"]');

    expect(signalFor(fetchMock, DEVICE_PATH, 1).aborted).toBe(true);
    expect(host().querySelector('.device-card')).toBeNull();

    settle();
    await vi.waitFor(() => expect(host().querySelector('.result-table')).not.toBeNull());
    // The abandoned answer never draws a card behind the operator's back.
    expect(host().querySelector('.device-card')).toBeNull();
  });

  test('a store that cannot serve the read reports it with its own remedy', async () => {
    stubFetch((path) => {
      if (path.startsWith(ONTOLOGY_PATH)) return { ok: true, body: ontologyPayload() };
      if (path.startsWith(SEARCH_PATH)) return { ok: true, body: searchPayload() };
      return {
        ok: false,
        status: 503,
        body: {
          detail: 'Graph store is not reachable at bolt://localhost:7687.',
          error_type: 'service_unavailable',
          suggestions: ['Start the graphdb service.'],
        },
      };
    });

    await mountGraph(host());
    click('.result-table td.dev button.dev');

    // A 503 is not a missing device: it goes to the panel's own pane, which is
    // the one place with room for the remedy the store sent.
    await vi.waitFor(() =>
      expect(host().querySelector('.explore-unknown--info')).not.toBeNull());
    const pane = host().querySelector('.explore-unknown--info');
    expect(pane?.textContent).toContain('Graph store is not reachable at bolt://localhost:7687.');
    expect(pane?.textContent).toContain('Start the graphdb service.');
    expect(host().querySelector('.device-card')).toBeNull();
    expect(host().querySelector('#graph-retry')).not.toBeNull();
  });
});

describe('the selection', () => {
  test('rows and the page checkbox both feed the selection, which survives paging', async () => {
    const fetchMock = stubDemo();
    await mountGraph(host());

    check(`tbody input[data-pv="${DEMO_ROWS[0].fullPv}"]`, true);
    expect(host().querySelector('.sel-count')?.textContent).toBe('1 selected');

    check('thead input[data-select-page]', true);
    expect(host().querySelector('.sel-count')?.textContent).toBe('3 selected');

    // A new page redraws the rows; the selection is the operator's, not the
    // page's, so it is still there.
    click('[data-page="next"]');
    await vi.waitFor(() => expect(callsTo(fetchMock, SEARCH_PATH)).toHaveLength(2));
    expect(host().querySelector('.sel-count')?.textContent).toBe('3 selected');

    click('[data-action="clear"]');
    expect(host().querySelector('.sel-count')?.textContent).toBe('0 selected');
    expect(host().querySelector('[data-action="copy"]')?.hasAttribute('disabled')).toBe(true);
  });

  test('Copy hands the clipboard the addresses, newline-joined', async () => {
    stubDemo();
    await mountGraph(host());
    check('thead input[data-select-page]', true);

    click('[data-action="copy"]');

    await vi.waitFor(() => expect(mocks.copyText).toHaveBeenCalledTimes(1));
    expect(mocks.copyText).toHaveBeenCalledWith(DEMO_ROWS.map((row) => row.fullPv).join('\n'));
    expect(mocks.showToast).toHaveBeenCalledWith('Copied 3 addresses', 'success');
  });

  test('a refused clipboard says what to do instead', async () => {
    stubDemo();
    mocks.copyText.mockResolvedValue(false);
    await mountGraph(host());
    check('thead input[data-select-page]', true);

    click('[data-action="copy"]');

    await vi.waitFor(() => expect(mocks.showToast).toHaveBeenCalledTimes(1));
    expect(mocks.showToast).toHaveBeenCalledWith(
      'Copy failed — select the table and copy manually', 'error');
  });

  test("a row's copy button copies that one address", async () => {
    stubDemo();
    await mountGraph(host());

    click(`tbody .copy-btn[data-copy="${DEMO_ROWS[1].fullPv}"]`);

    await vi.waitFor(() => expect(mocks.copyText).toHaveBeenCalledTimes(1));
    expect(mocks.copyText).toHaveBeenCalledWith(DEMO_ROWS[1].fullPv);
  });
});

describe('sending to the assistant', () => {
  test('embedded, Send posts one message to this page\'s own origin', async () => {
    mocks.isEmbedded.mockReturnValue(true);
    const post = vi.fn();
    vi.stubGlobal('parent', { postMessage: post });
    stubDemo();

    await mountGraph(host());
    check('thead input[data-select-page]', true);
    click('[data-action="send"]');

    expect(post).toHaveBeenCalledTimes(1);
    expect(post).toHaveBeenCalledWith(
      { type: 'osprey-paste-to-terminal', text: DEMO_ROWS.map((r) => r.fullPv).join(' ') },
      window.location.origin,
    );
    expect(mocks.showToast).toHaveBeenCalledWith('Posted 3 addresses to the prompt', 'success');
  });

  test('standalone there is nothing to send to, so no Send button is drawn', async () => {
    mocks.isEmbedded.mockReturnValue(false);
    stubDemo();

    await mountGraph(host());

    expect(host().querySelector('[data-action="send"]')).toBeNull();
    // Copy and Clear are still offered.
    expect(host().querySelector('[data-action="copy"]')).not.toBeNull();
    expect(host().querySelector('[data-action="clear"]')).not.toBeNull();
  });
});

test('unmounting clears the pane', async () => {
  stubDemo();

  await mountGraph(host());
  expect(host().querySelectorAll('.result-table tbody tr').length).toBeGreaterThan(0);

  unmountGraph();

  expect(host().innerHTML).toBe('');
});

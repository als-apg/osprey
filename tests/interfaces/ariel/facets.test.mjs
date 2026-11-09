// @ts-check
/**
 * Rail facets (facets.js):
 *   npx vitest run tests/interfaces/ariel/facets.test.mjs
 *
 * facets.js renders value counts per filter dimension, a 365-day histogram and
 * the Enhancements section into the rail's facets mount, fetching
 * GET /api/facets with the list's scope. Only api.js is mocked (the network
 * boundary). Module state is per import, so every case resets modules.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return { ...actual, facetsApi: { ...actual.facetsApi, get: vi.fn() } };
});

const API = '../../../src/osprey/interfaces/ariel/static/js/api.js';
const STORE = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const FACETS = '../../../src/osprey/interfaces/ariel/static/js/facets.js';
const RAIL = '../../../src/osprey/interfaces/ariel/static/js/rail.js';
const TIMEZONE = '/design-system/js/facility-time.js';

const CAPABILITIES = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [
        {
          name: 'keyword',
          label: 'Keyword',
          parameters: [{ name: 'fuzzy', label: 'Fuzzy', type: 'bool' }],
        },
      ],
    },
  },
  default_mode: 'keyword',
  shared_parameters: [{ name: 'author', label: 'Author', type: 'text' }],
  filter_dimensions: [
    { name: 'author', label: 'Author', kind: 'builtin' },
    { name: 'tags', label: 'Tags', kind: 'builtin' },
    { name: 'time', label: 'Time', kind: 'builtin' },
    { name: 'enhancement', label: 'Enhancement', kind: 'enhancement' },
  ],
  enhancement_modules: [
    { name: 'text_embedding', label: 'Text embedding' },
    { name: 'image_caption', label: 'Image caption' },
  ],
});

/** @param {string} day */
function payload(day, extra = {}) {
  return {
    dimensions: [
      {
        name: 'author',
        label: 'Author',
        kind: 'builtin',
        values: [
          { value: 'alice', count: 4 },
          { value: 'Bob', count: 2 },
        ],
      },
      {
        name: 'tags',
        label: 'Tags',
        kind: 'builtin',
        values: [{ value: '<img src=x onerror=alert(1)>', count: 1 }],
      },
      {
        name: 'enhancement',
        label: 'Enhancement',
        kind: 'enhancement',
        values: [
          { value: 'text_embedding:failed', count: 3 },
          { value: 'image_caption:stale', count: 1 },
          { value: 'image_caption:complete', count: 9 },
        ],
      },
    ],
    histogram: [{ day, count: 5 }],
    exact: true,
    tier: null,
    ...extra,
  };
}

/** @type {any} */
let store;
/** @type {any} */
let facets;
/** @type {any} */
let get;
/** @type {string} */
let today;
/** @type {HTMLElement} */
let root;

async function settle() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Fresh modules; the store patched before mount; facets mounted and settled.
 * @param {Record<string, any>} [patch]
 * @param {any} [response]
 */
async function boot(patch = {}, response) {
  vi.resetModules();
  const api = await import(API);
  get = api.facetsApi.get;
  get.mockReset();
  store = await import(STORE);
  facets = await import(FACETS);
  const tz = await import(TIMEZONE);
  today = tz.facilityDayKey(Date.now());
  get.mockResolvedValue(response === undefined ? payload(today) : response);
  if (Object.keys(patch).length) store.setState(patch);
  facets.initFacets(root, { capabilities: CAPABILITIES });
  await settle();
}

/**
 * @param {string} dim
 * @param {string} value
 * @returns {HTMLButtonElement}
 */
function row(dim, value) {
  const el = root.querySelector(
    `[data-facet-dim="${dim}"][data-facet-value="${CSS.escape(value)}"]`
  );
  if (!(el instanceof HTMLButtonElement)) throw new Error(`no row ${dim}=${value}`);
  return el;
}

function sections() {
  const el = root.querySelector('.ariel-facets-sections');
  if (!(el instanceof HTMLElement)) throw new Error('no sections container');
  return el;
}

beforeEach(() => {
  document.body.replaceChildren();
  root = document.createElement('div');
  root.className = 'ariel-rail-facets';
  document.body.append(root);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('value sections', () => {
  test('one section per dimension, rows are nav rows with count badges', async () => {
    await boot();
    expect(root.querySelector('[data-facet-section="author"]')).not.toBeNull();
    expect(root.querySelector('[data-facet-section="tags"]')).not.toBeNull();
    const alice = row('author', 'alice');
    expect(alice.classList.contains('osprey-nav-row')).toBe(true);
    expect(alice.getAttribute('aria-pressed')).toBe('false');
    expect(alice.querySelector('.osprey-nav-badge')?.textContent).toBe('4');
    expect(alice.querySelector('.osprey-nav-label')?.textContent).toBe('alice');
  });

  test('values are text, never markup', async () => {
    await boot();
    const tag = row('tags', '<img src=x onerror=alert(1)>');
    expect(tag.querySelector('img')).toBeNull();
    expect(tag.querySelector('.osprey-nav-label')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
  });

  test('a selected value is aria-pressed, ignoring case', async () => {
    await boot({ filter: { author: ['BOB'] } });
    expect(row('author', 'Bob').getAttribute('aria-pressed')).toBe('true');
    expect(row('author', 'alice').getAttribute('aria-pressed')).toBe('false');
  });

  test('inexact counts are labelled with ≈', async () => {
    await boot({}, payload('2026-01-01', { exact: false }));
    expect(row('author', 'alice').querySelector('.osprey-nav-badge')?.textContent).toBe('≈4');
  });

  test('a click toggles the value in store filter', async () => {
    await boot();
    row('author', 'alice').click();
    expect(store.getState().filter).toEqual({ author: ['alice'] });
    await settle();
    expect(row('author', 'alice').getAttribute('aria-pressed')).toBe('true');
    row('author', 'alice').click();
    expect(store.getState().filter).toEqual({});
  });

  test('under Mine the author section shows only the Mine name, and an author click switches to All', async () => {
    await boot({ mailbox: 'mine', mineAuthor: 'Alice' });
    expect(root.querySelectorAll('[data-facet-dim="author"]')).toHaveLength(1);
    expect(row('author', 'alice').getAttribute('aria-pressed')).toBe('true');
    row('author', 'alice').click();
    expect(store.getState().mailbox).toBe('all');
    expect(store.getState().filter).toEqual({ author: ['alice'] });
  });

  test('a non-author click under Mine stays in Mine', async () => {
    await boot({ mailbox: 'mine', mineAuthor: 'Alice' });
    row('tags', '<img src=x onerror=alert(1)>').click();
    expect(store.getState().mailbox).toBe('mine');
  });
});

describe('histogram', () => {
  test('covers the last 365 days ending today, bar per day', async () => {
    await boot();
    const bars = root.querySelectorAll('.ariel-facet-day');
    expect(bars).toHaveLength(365);
    expect(bars[bars.length - 1].getAttribute('data-day')).toBe(today);
    expect(bars[bars.length - 1].getAttribute('aria-label')).toBe(`${today}: 5`);
  });

  test('highlights the selected window', async () => {
    await boot({ filter: { time: { start: today, end: today } } });
    const selected = root.querySelectorAll('.ariel-facet-day-selected');
    expect(selected).toHaveLength(1);
    expect(selected[0].getAttribute('data-day')).toBe(today);
    expect(selected[0].getAttribute('aria-pressed')).toBe('true');
  });

  test('no highlight without a window', async () => {
    await boot();
    expect(root.querySelectorAll('.ariel-facet-day-selected')).toHaveLength(0);
  });

  test('clicking a day sets time to that day, keeping other dimensions', async () => {
    await boot({ filter: { author: ['alice'] } });
    const bars = root.querySelectorAll('.ariel-facet-day');
    const day = /** @type {HTMLButtonElement} */ (bars[100]);
    const key = day.getAttribute('data-day');
    day.click();
    expect(store.getState().filter).toEqual({ author: ['alice'], time: { start: key, end: key } });
  });
});

describe('enhancements', () => {
  test('lists failed and stale values only, with module labels and the CLI hint', async () => {
    await boot();
    const sec = root.querySelector('[data-facet-section="enhancement"]');
    expect(sec?.querySelector('.osprey-section-label')?.textContent).toBe('Enhancements');
    const rows = sec?.querySelectorAll('[data-facet-dim="enhancement"]') ?? [];
    expect([...rows].map((r) => r.getAttribute('data-facet-value'))).toEqual([
      'text_embedding:failed',
      'image_caption:stale',
    ]);
    expect(
      row('enhancement', 'text_embedding:failed').querySelector('.osprey-nav-label')?.textContent
    ).toBe('Text embedding: failed');
    expect(sec?.querySelector('.ariel-facets-note')).toBeNull();
    expect(sec?.querySelector('.ariel-facets-hint')?.textContent).toContain('--retry-failed');
  });

  test('a clean logbook shows the empty note and the CLI hint', async () => {
    const clean = payload('2026-01-01');
    clean.dimensions[2].values = [];
    await boot({}, clean);
    const sec = root.querySelector('[data-facet-section="enhancement"]');
    expect(sec?.querySelector('.ariel-facets-note')?.textContent).toMatch(/no failed or stale/i);
    expect(sec?.querySelector('.ariel-facets-hint')?.textContent).toContain('osprey ariel enhance');
  });

  test('an enhancement click toggles it in filter', async () => {
    await boot();
    row('enhancement', 'image_caption:stale').click();
    expect(store.getState().filter).toEqual({ enhancement: ['image_caption:stale'] });
  });
});

describe('degraded and hidden states', () => {
  test('a failed fetch hides counts and the note, shows no error, and clears busy', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    await boot({}, null);
    get.mockRejectedValue(new Error('boom'));
    store.setState({ query: 'beam' });
    await settle();
    expect(root.querySelectorAll('.osprey-nav-badge')).toHaveLength(0);
    expect(root.querySelector('.ariel-facets-note')).toBeNull();
    expect(root.querySelector('.ariel-facets-hint')).toBeNull();
    expect(root.textContent).not.toMatch(/error|boom|fail/i);
    expect(root.getAttribute('aria-busy')).toBe('false');
  });

  test('an available:false payload is treated like a failure', async () => {
    await boot({}, { available: false, exact: false, dimensions: [], histogram: [] });
    expect(root.querySelectorAll('.osprey-nav-badge')).toHaveLength(0);
    expect(root.querySelector('.ariel-facets-note')).toBeNull();
    expect(root.getAttribute('aria-busy')).toBe('false');
  });

  test('under Drafts the sections are hidden, nothing is fetched, busy is false', async () => {
    await boot({ mailbox: 'drafts' });
    expect(get).not.toHaveBeenCalled();
    expect(sections().hidden).toBe(true);
    expect(root.getAttribute('aria-busy')).toBe('false');
    store.setState({ mailbox: 'all' });
    await settle();
    expect(get).toHaveBeenCalledTimes(1);
    expect(sections().hidden).toBe(false);
  });

  test('the icon strip hides the sections and clears busy', async () => {
    await boot();
    /** @type {(v: any) => void} */
    let resolve = () => {};
    get.mockReturnValue(new Promise((r) => (resolve = r)));
    store.setState({ query: 'beam' });
    await settle();
    expect(root.getAttribute('aria-busy')).toBe('true');
    store.setState({ railCollapsed: true });
    await settle();
    expect(sections().hidden).toBe(true);
    expect(root.getAttribute('aria-busy')).toBe('false');
    resolve(payload(today));
  });

  test('aria-busy is true only while a fetch is in flight', async () => {
    await boot();
    expect(root.getAttribute('aria-busy')).toBe('false');
    /** @type {(v: any) => void} */
    let resolve = () => {};
    get.mockReturnValue(new Promise((r) => (resolve = r)));
    store.setState({ query: 'beam' });
    await settle();
    expect(root.getAttribute('aria-busy')).toBe('true');
    resolve(payload(today));
    await settle();
    expect(root.getAttribute('aria-busy')).toBe('false');
  });
});

describe('requests', () => {
  test('sends the request filter, query, mode and the mode knobs', async () => {
    await boot({
      query: ' beam ',
      mode: 'keyword',
      filters: { fuzzy: true, author: 'carol', bogus: 1 },
      filter: { tags: ['rf'] },
      mailbox: 'mine',
      mineAuthor: 'Alice',
    });
    expect(get).toHaveBeenCalledTimes(1);
    const args = get.mock.calls[0][0];
    expect(args.query).toBe('beam');
    expect(args.mode).toBe('keyword');
    expect(args.filters).toEqual({ author: ['Alice'], tags: ['rf'] });
    expect(args.advancedParams).toEqual({ fuzzy: true });
    expect(args.signal).toBeInstanceOf(AbortSignal);
  });

  test('a burst of changes refetches once', async () => {
    await boot();
    expect(get).toHaveBeenCalledTimes(1);
    store.setState({ query: 'a' });
    store.setState({ filter: { author: ['alice'] } });
    store.setState({ mode: 'keyword' });
    await settle();
    expect(get).toHaveBeenCalledTimes(2);
  });

  test('refetches on each settled change of query, mode, knobs, filter and mailbox', async () => {
    await boot();
    const patches = [
      { query: 'x' },
      { mode: 'keyword' },
      { filters: { fuzzy: true } },
      { filter: { tags: ['rf'] } },
      { mailbox: 'mine', mineAuthor: 'Alice' },
    ];
    for (const patch of patches) {
      store.setState(patch);
      await settle();
    }
    expect(get).toHaveBeenCalledTimes(1 + patches.length);
  });

  test('a change that leaves the request alone does not refetch', async () => {
    await boot();
    store.setState({ mineAuthor: 'Zed' }); // not under Mine: no effect on the request
    store.setState({ sel: 'e1' });
    await settle();
    expect(get).toHaveBeenCalledTimes(1);
  });

  test('a newer fetch aborts the superseded one, whose late answer is ignored', async () => {
    await boot();
    /** @type {(v: any) => void} */
    let first = () => {};
    get.mockReturnValueOnce(new Promise((r) => (first = r)));
    store.setState({ query: 'one' });
    await settle();
    const firstSignal = get.mock.calls[1][0].signal;
    get.mockResolvedValueOnce(payload(today, { exact: false }));
    store.setState({ query: 'two' });
    await settle();
    expect(firstSignal.aborted).toBe(true);
    first(payload(today));
    await settle();
    expect(row('author', 'alice').querySelector('.osprey-nav-badge')?.textContent).toBe('≈4');
  });
});

describe('rail mount', () => {
  test('the rail always renders the facets mount below the saved searches', async () => {
    vi.resetModules();
    const rail = await import(RAIL);
    const railRoot = document.createElement('nav');
    document.body.append(railRoot);
    rail.initRail(railRoot);
    const mount = railRoot.querySelector('.ariel-rail-facets');
    expect(mount).not.toBeNull();
    expect(mount?.getAttribute('aria-busy')).toBe('false');
    expect(railRoot.lastElementChild).toBe(mount);
    expect(mount?.previousElementSibling?.classList.contains('ariel-rail-saved')).toBe(true);
  });
});

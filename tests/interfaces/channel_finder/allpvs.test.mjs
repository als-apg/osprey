// @ts-check
/**
 * The All PVs view (allpvs.js): rows and marks, filters, the request line, the
 * states, current values, CSV, and the Plot and Send hand-offs under their gates.
 * The answers are the shared route fixture's; the toolbar is index.html's.
 *   npx vitest run tests/interfaces/channel_finder/allpvs.test.mjs
 */

import { test, expect, vi, afterEach, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  mountAllPvs,
  unmountAllPvs,
} from '../../../src/osprey/interfaces/channel_finder/static/js/allpvs.js';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);

const PANEL_HTML = (() => {
  const html = readFileSync(
    join(import.meta.dirname, '../../../src/osprey/interfaces/channel_finder/static/index.html'),
    'utf8'
  );
  const match = html.match(/<div class="cf-panel"[\s\S]*<\/main>/);
  if (!match) throw new Error('index.html carries no panel root');
  return `${match[0]}</div>`;
})();

const DESCRIPTOR = { label: 'All PVs', configured_services: [], canopy: true, ariel: true };

/** The Contracts' No source form. */
const NO_SOURCE = {
  outcome: 'unavailable',
  reason: 'not available for this target',
  source: { target: 'live', service: 'none', label: 'All PVs', rehearsal: false },
  searched: '',
  request: '',
  total: null,
  rows: [],
  truncated: false,
  partial: [],
  facets: null,
  facets_reason: null,
};

/** @type {(() => void) | null} */
let unmount = null;

afterEach(() => {
  unmount?.();
  unmount = null;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

test('the view exports what the router mounts and unmounts', () => {
  expect(mountAllPvs).toBeTypeOf('function');
  expect(unmountAllPvs).toBeTypeOf('function');
});

/**
 * A fetch answer.
 * @param {number} status
 * @param {any} body
 */
function answer(status, body) {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: '',
    json: async () => JSON.parse(text),
    text: async () => text,
  };
}

/**
 * @typedef {{
 *   path?: string,
 *   framed?: boolean,
 *   trace?: number,
 *   search?: any,
 *   values?: any,
 *   csv?: { status: number, body: string },
 *   query?: string,
 *   params?: Record<string, string|string[]>,
 * }} Setup
 */

/**
 * Mount All PVs in a fresh module registry at `path`, framed by a host (or a
 * pop-out, or nothing), against a fetch that answers the directory routes.
 * @param {Setup} [setup]
 */
async function mount(setup = {}) {
  const {
    path = '/u/a/panel/channel-finder/?embedded=true',
    framed = true,
    trace = 200,
    search = ROUTES.search,
    values = null,
    csv = { status: 200, body: 'name\nDEMO:BPM01:X\n' },
    query = '',
    params = {},
  } = setup;
  window.history.replaceState(null, '', path);
  const postMessage = vi.fn();
  vi.stubGlobal('parent', framed ? { postMessage } : window);
  document.body.innerHTML = PANEL_HTML;

  /** @type {string[]} */
  const searches = [];
  const fetchMock = vi.fn(async (/** @type {any} */ url, /** @type {any} */ opts = {}) => {
    const target = String(url);
    if (opts.method === 'HEAD') return answer(trace, '');
    if (target.startsWith('/api/directory/search')) {
      searches.push(target);
      return answer(200, typeof search === 'function' ? search(target) : search);
    }
    if (target.startsWith('/api/directory/values')) {
      const names = new URL(target, 'http://x').searchParams.getAll('name');
      const rows = values ?? ROUTES.values.filter((/** @type {any} */ v) => names.includes(v.name));
      return answer(200, rows);
    }
    if (target.startsWith('/api/directory/export.csv')) return answer(csv.status, csv.body);
    return answer(404, { detail: 'Not Found' });
  });
  vi.stubGlobal('fetch', fetchMock);

  vi.resetModules();
  const { state } = await import(`${JS}/state.js`);
  state.directory = { ...DESCRIPTOR };
  state.scopeQuery = query;
  const labels = vi.fn();
  state.on('directoryLabel', labels);
  const view = await import(`${JS}/allpvs.js`);
  const container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
  view.mountAllPvs(container, params);
  unmount = view.unmountAllPvs;
  return { state, view, container, fetchMock, searches, postMessage, labels };
}

/** @param {HTMLElement} root */
const rows = (root) => [
  .../** @type {NodeListOf<HTMLElement>} */ (root.querySelectorAll('tbody tr')),
];

/** @param {HTMLElement} root @param {string} label */
function button(root, label) {
  return /** @type {HTMLButtonElement|undefined} */ (
    [...root.querySelectorAll('button')].find((b) => b.textContent?.trim() === label)
  );
}

/** @param {HTMLElement} root */
const plot = (root) =>
  /** @type {HTMLButtonElement|null} */ (root.querySelector('[data-action="plot"]'));

/** @param {HTMLElement} root */
const send = (root) =>
  /** @type {HTMLButtonElement|null} */ (root.querySelector('[data-action="send"]'));

/** @param {HTMLElement} root @param {number} index */
function select(root, index) {
  const box = /** @type {HTMLInputElement} */ (rows(root)[index].querySelector('input'));
  box.click();
}

/**
 * A found page of `n` rows named `PV:<i>`.
 * @param {number} n
 */
function pageOf(n) {
  const base = ROUTES.search.rows[0];
  return {
    ...ROUTES.search,
    total: n,
    rows: Array.from({ length: n }, (_, i) => ({ ...base, name: `PV:${i}` })),
  };
}

describe('rows and states', () => {
  test('rows carry their marks, status, values and the directory label', async () => {
    const { container, labels } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const [curated, directory] = rows(container);
    expect(curated.textContent).toContain('◆');
    expect(curated.textContent).toContain('DEMO:BPM01:X');
    expect(curated.textContent).toContain('Active');
    expect(directory.textContent).toContain('◇');
    expect(directory.textContent).toContain('Inactive');
    expect(directory.classList.contains('inactive')).toBe(true);
    await vi.waitFor(() => expect(curated.textContent).toContain('0.125'));
    expect(directory.textContent).toContain('no answer');
    expect(document.getElementById('cf-result-count')?.textContent).toBe('2 PVs');
    expect(labels).toHaveBeenCalledWith('All PVs');
  });

  test('the plain-words line and the request behind its toggle', async () => {
    const { container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(container.textContent).toContain(ROUTES.search.searched);
    const request = /** @type {HTMLElement} */ (container.querySelector('.allpvs-request'));
    expect(request.hidden).toBe(true);
    button(container, 'show request')?.click();
    expect(request.hidden).toBe(false);
    expect(request.textContent).toContain(ROUTES.search.request);
    expect(request.querySelector('input, textarea, [contenteditable]')).toBeNull();
    expect(button(container, 'hide request')).toBeDefined();
  });

  test('the no-source form shows its reason and no table', async () => {
    const { container, labels } = await mount({ search: NO_SOURCE });
    await vi.waitFor(() =>
      expect(container.querySelector('.osprey-state')?.textContent).toContain(
        'not available for this target'
      )
    );
    expect(container.querySelector('table')).toBeNull();
    expect(button(container, 'Retry')).toBeUndefined();
    expect(labels).toHaveBeenCalledWith('All PVs');
  });

  test('no match never says the PV does not exist', async () => {
    const { container } = await mount({
      search: { ...ROUTES.search, outcome: 'no_match', total: 0, rows: [], facets: null },
    });
    await vi.waitFor(() =>
      expect(container.querySelector('.osprey-state')?.textContent).toContain('No match')
    );
    expect(container.textContent).not.toMatch(/does not exist/i);
  });

  test('a partial answer is labelled', async () => {
    const { container } = await mount({
      search: { ...ROUTES.search, partial: [{ part: 'count', reason: 'no answer within 6 s' }] },
    });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const chip = container.querySelector('.osprey-chip--warn');
    expect(chip?.textContent).toBe('partial');
  });

  test('Refresh reads the visible page again', async () => {
    const { container, fetchMock } = await mount();
    await vi.waitFor(() => expect(rows(container)[0].textContent).toContain('0.125'));
    const reads = () =>
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/directory/values'));
    expect(reads()).toHaveLength(1);
    /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="refresh"]')).click();
    await vi.waitFor(() => expect(reads()).toHaveLength(2));
    expect(new URL(reads()[1][0], 'http://x').searchParams.getAll('name')).toEqual([
      'DEMO:BPM01:X',
      'DEMO:BPM01:HEARTBEAT',
    ]);
  });
});

describe('search and filters', () => {
  test('the toolbar search is seeded from scopeQuery and writes it back', async () => {
    const { container, state, searches } = await mount({ query: 'BPM01' });
    const input = /** @type {HTMLInputElement} */ (document.getElementById('cf-search'));
    expect(input.value).toBe('BPM01');
    expect(input.disabled).toBe(false);
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(searches[0]).toContain('q=BPM01');

    input.value = 'BPM02';
    input.dispatchEvent(new Event('input'));
    expect(state.scopeQuery).toBe('BPM02');
    await vi.waitFor(() => expect(searches.at(-1)).toContain('q=BPM02'));
  });

  test('the route words win over scopeQuery', async () => {
    const { state, searches } = await mount({ query: 'old', params: { q: 'new words' } });
    expect(state.scopeQuery).toBe('new words');
    await vi.waitFor(() => expect(searches[0]).toContain('q=new+words'));
  });

  test('the route filters seed the facets', async () => {
    const { searches } = await mount({ params: { ioc: ['IOC1'], record_type: ['ai'] } });
    await vi.waitFor(() => expect(searches).toHaveLength(1));
    const first = new URL(searches[0], 'http://x').searchParams;
    expect(first.getAll('ioc')).toEqual(['IOC1']);
    expect(first.getAll('record_type')).toEqual(['ai']);
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    await vi.waitFor(() =>
      expect(
        sidebar.querySelector('[data-facet="ioc"][data-value="IOC1"]')?.getAttribute('aria-pressed')
      ).toBe('true')
    );
  });

  test('facets list origin, IOC and record type with counts and filter the search', async () => {
    const { searches } = await mount();
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    await vi.waitFor(() => expect(sidebar.querySelectorAll('.osprey-list-row')).toHaveLength(5));
    const labels = [...sidebar.querySelectorAll('.osprey-section-label')].map((l) => l.textContent);
    expect(labels).toEqual(['Origin', 'IOC', 'Record type']);
    const curated = /** @type {HTMLButtonElement} */ (
      sidebar.querySelector('[data-facet="origin"][data-value="curated"]')
    );
    expect(curated.textContent).toContain('◆ Curated');
    expect(curated.textContent).toContain('1');
    expect(sidebar.textContent).toContain('◇ Control system only');
    curated.click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('origin=curated'));
  });

  test('above the filter cap the sidebar says why there are no filters', async () => {
    const reason = 'too many results to filter (9000); narrow the search';
    await mount({ search: { ...ROUTES.search, facets: null, facets_reason: reason } });
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    await vi.waitFor(() => expect(sidebar.textContent).toContain(reason));
    expect(sidebar.querySelector('.osprey-list-row')).toBeNull();
  });

  test('unmount gives the toolbar back', async () => {
    const { container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    unmount?.();
    unmount = null;
    const input = /** @type {HTMLInputElement} */ (document.getElementById('cf-search'));
    expect(input.disabled).toBe(true);
    expect(document.getElementById('cf-result-count')?.hidden).toBe(true);
  });
});

describe('Plot and Send', () => {
  test('Plot posts the Canopy url with pv in selection order', async () => {
    const { container, postMessage } = await mount();
    await vi.waitFor(() => expect(plot(container)).not.toBeNull());
    select(container, 1);
    select(container, 0);
    plot(container)?.click();
    expect(postMessage).toHaveBeenCalledWith(
      {
        type: 'osprey:navigate',
        panel: 'canopy',
        url: '/u/a/panel/canopy/embed/@trace?pv=DEMO%3ABPM01%3AHEARTBEAT&pv=DEMO%3ABPM01%3AX&range=-1h',
      },
      window.location.origin
    );
  });

  test('Plot is disabled past 12 selected', async () => {
    const { container } = await mount({ search: pageOf(13) });
    await vi.waitFor(() => expect(plot(container)).not.toBeNull());
    for (let i = 0; i < 12; i++) select(container, i);
    expect(plot(container)?.disabled).toBe(false);
    select(container, 12);
    expect(plot(container)?.disabled).toBe(true);
    expect(plot(container)?.textContent).toBe('Plot (max 12)');
    expect(plot(container)?.title).toBe('Data Browser takes at most 12 PVs');
  });

  test('Send pastes the selection in selection order', async () => {
    const { container, postMessage } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    select(container, 1);
    select(container, 0);
    send(container)?.click();
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'osprey-paste-to-terminal', text: 'DEMO:BPM01:HEARTBEAT DEMO:BPM01:X' },
      window.location.origin
    );
  });

  test('Send pastes the selection to the assistant', async () => {
    const { container, postMessage } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    select(container, 0);
    send(container)?.click();
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'osprey-paste-to-terminal', text: 'DEMO:BPM01:X' },
      window.location.origin
    );
  });

  test('a pop-out shows no Plot and no Send', async () => {
    const { container, fetchMock } = await mount({ framed: false });
    await vi.waitFor(() => expect(rows(container)[0].textContent).toContain('0.125'));
    expect(plot(container)).toBeNull();
    expect(send(container)).toBeNull();
    expect(fetchMock.mock.calls.some(([, opts]) => opts?.method === 'HEAD')).toBe(false);
  });

  test('no Plot when Trace does not answer 200', async () => {
    const { container, fetchMock } = await mount({ trace: 404 });
    await vi.waitFor(() =>
      expect(fetchMock.mock.calls.some(([, opts]) => opts?.method === 'HEAD')).toBe(true)
    );
    await vi.waitFor(() => expect(rows(container)[0].textContent).toContain('0.125'));
    expect(plot(container)).toBeNull();
    expect(send(container)).not.toBeNull();
  });
});

describe('the selection basket', () => {
  /**
   * A found page of rows named after the searched words, `<words>:<i>`.
   * @param {string} url
   */
  function byWords(url) {
    const q = new URL(url, 'http://x').searchParams.get('q') || 'PV';
    const page = pageOf(2);
    return {
      ...page,
      rows: page.rows.map((/** @type {any} */ row, /** @type {number} */ i) => ({
        ...row,
        name: `${q}:${i}`,
      })),
    };
  }

  /** @param {HTMLElement} root */
  const selection = (root) => root.querySelector('.allpvs-actions')?.textContent ?? '';

  test('new words keep it, and Plot posts it in selection order', async () => {
    const { container, postMessage, searches } = await mount({ search: byWords, query: 'A' });
    await vi.waitFor(() => expect(rows(container)[0]?.dataset.name).toBe('A:0'));
    select(container, 0);

    const input = /** @type {HTMLInputElement} */ (document.getElementById('cf-search'));
    input.value = 'B';
    input.dispatchEvent(new Event('input'));
    await vi.waitFor(() => expect(rows(container)[0]?.dataset.name).toBe('B:0'));
    expect(searches.at(-1)).toContain('q=B');
    expect(selection(container)).toContain('1 selected');
    select(container, 1);
    expect(selection(container)).toContain('2 selected');

    plot(container)?.click();
    expect(postMessage).toHaveBeenCalledWith(
      {
        type: 'osprey:navigate',
        panel: 'canopy',
        url: '/u/a/panel/canopy/embed/@trace?pv=A%3A0&pv=B%3A1&range=-1h',
      },
      window.location.origin
    );
  });

  test('a facet change and paging keep it; clear empties it', async () => {
    const { container, searches } = await mount({ search: pageOf(60) });
    await vi.waitFor(() => expect(rows(container).length).toBeGreaterThan(1));
    select(container, 0);
    select(container, 1);

    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    /** @type {HTMLButtonElement} */ (
      sidebar.querySelector('[data-facet="origin"][data-value="curated"]')
    ).click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('origin=curated'));
    await vi.waitFor(() => expect(rows(container).length).toBeGreaterThan(1));
    expect(selection(container)).toContain('2 selected');

    /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="next"]')).click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('page=1'));
    await vi.waitFor(() => expect(selection(container)).toContain('page 2'));
    expect(selection(container)).toContain('2 selected');

    button(container, 'clear')?.click();
    expect(selection(container)).not.toContain('selected');
  });

  test('the select-page box adds the page in row order after what is selected', async () => {
    const { container, postMessage } = await mount({ search: pageOf(3) });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(3));
    select(container, 2);
    const head = () =>
      /** @type {HTMLInputElement} */ (container.querySelector('[data-action="select-page"]'));
    head().click();
    expect(selection(container)).toContain('3 selected');
    send(container)?.click();
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'osprey-paste-to-terminal', text: 'PV:2 PV:0 PV:1' },
      window.location.origin
    );
    head().click();
    expect(selection(container)).not.toContain('selected');
  });

  test('a new mount starts with nothing selected', async () => {
    const { view, container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    select(container, 0);
    expect(selection(container)).toContain('1 selected');
    view.unmountAllPvs();
    view.mountAllPvs(container, {});
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(selection(container)).not.toContain('selected');
  });
});

describe('CSV', () => {
  /** @param {HTMLElement} container */
  async function exportCsv(container) {
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    /** @type {Blob[]} */
    const blobs = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
      blobs.push(/** @type {Blob} */ (blob));
      return 'blob:csv';
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const clicks = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="csv"]')).click();
    return { blobs, clicks };
  }

  test('a 200 downloads the answer under the searched words', async () => {
    const { container, fetchMock } = await mount({ query: 'BPM01' });
    const { blobs, clicks } = await exportCsv(container);
    await vi.waitFor(() => expect(clicks).toHaveBeenCalledTimes(1));
    expect(await blobs[0].text()).toBe('name\nDEMO:BPM01:X\n');
    const link = /** @type {HTMLAnchorElement} */ (clicks.mock.contexts[0]);
    expect(link.download).toBe('BPM01.csv');
    const call = fetchMock.mock.calls.find(([url]) => String(url).includes('export.csv'));
    expect(String(call?.[0])).toContain('q=BPM01');
  });

  test('a 503 shows its detail and downloads nothing', async () => {
    const { container } = await mount({
      csv: { status: 503, body: JSON.stringify({ detail: 'no answer within 6 s' }) },
    });
    const { clicks } = await exportCsv(container);
    await vi.waitFor(() => expect(container.textContent).toContain('CSV: no answer within 6 s'));
    expect(clicks).not.toHaveBeenCalled();
  });

  test('a failure that is not JSON names its status', async () => {
    const { container } = await mount({ csv: { status: 502, body: '<html>' } });
    const { clicks } = await exportCsv(container);
    await vi.waitFor(() =>
      expect(container.textContent).toContain('CSV: export did not answer (502)')
    );
    expect(clicks).not.toHaveBeenCalled();
  });
});

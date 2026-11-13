// @ts-check
/**
 * The All PVs view (allpvs.js): rows and marks, the alarm column, filters and
 * their chips, the request line, the states, current values, CSV, and the
 * selection bar's Plot, Send and Copy names under their gates.
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
 *   searchStatus?: number,
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
    searchStatus = 200,
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
      return answer(searchStatus, typeof search === 'function' ? search(target) : search);
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
  const routes = vi.fn();
  state.on('replaceRoute', routes);
  const view = await import(`${JS}/allpvs.js`);
  const container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
  view.mountAllPvs(container, params);
  unmount = view.unmountAllPvs;
  return { state, view, container, fetchMock, searches, postMessage, labels, routes };
}

/**
 * A directory row as a target with no answer from the alarm feed serves it.
 * @param {Record<string, unknown>} row
 */
function withoutAlarm(row) {
  const copy = { ...row };
  delete copy.alarm;
  return copy;
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
  test('rows carry their marks, alarm, values and the directory label', async () => {
    const { container, labels } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const [curated, directory] = rows(container);
    expect(curated.textContent).toContain('◆');
    expect(curated.textContent).toContain('DEMO:BPM01:X');
    expect(curated.textContent).not.toContain('Active');
    expect(curated.querySelector('.allpvs-alarm .cf-sev--major')).not.toBeNull();
    expect(directory.textContent).toContain('◇');
    expect(directory.querySelector('.allpvs-inactive')?.textContent).toBe('Inactive');
    expect(directory.classList.contains('inactive')).toBe(true);
    expect(directory.querySelector('.allpvs-alarm')?.textContent).toBe('·');
    await vi.waitFor(() => expect(curated.textContent).toContain('0.125 mm'));
    expect(directory.textContent).toContain('no answer');
    expect(document.getElementById('cf-result-count')?.textContent).toBe('2 PVs');
    expect(labels).toHaveBeenCalledWith('All PVs');
  });

  test('the plain-words line and the request behind its toggle', async () => {
    const { container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(container.querySelector('.allpvs-searched')?.textContent).toBe('every PV');
    expect(container.querySelector('.allpvs-searched')?.getAttribute('title')).toBe(
      ROUTES.search.searched
    );
    const request = /** @type {HTMLElement} */ (container.querySelector('.allpvs-request'));
    expect(request.hidden).toBe(true);
    button(container, 'show request')?.click();
    expect(request.hidden).toBe(false);
    expect(request.textContent).toContain(ROUTES.search.request);
    expect(request.querySelector('input, textarea, [contenteditable]')).toBeNull();
    expect(button(container, 'hide request')).toBeDefined();
  });

  test('the no-source form shows its reason and no table', async () => {
    const { container, labels } = await mount({ search: ROUTES['search no source'] });
    await vi.waitFor(() =>
      expect(container.querySelector('.osprey-state')?.textContent).toContain(
        'All PVs: not available for this target'
      )
    );
    expect(container.querySelector('table')).toBeNull();
    expect(button(container, 'Retry')).toBeUndefined();
    expect(labels).toHaveBeenCalledWith('All PVs');
    expect(document.getElementById('cf-result-count')?.textContent).toBe('');
  });

  test('an unavailable page replaces the Searching state', async () => {
    const { container, searches } = await mount({ search: ROUTES['search no source'] });
    await vi.waitFor(() => expect(searches).toHaveLength(1));
    await vi.waitFor(() =>
      expect(container.querySelector('.osprey-state')?.textContent).not.toContain('Searching')
    );
  });

  test('a 503 search shows the failure with Retry', async () => {
    const { container, searches } = await mount({
      search: { detail: 'no answer within 6 s' },
      searchStatus: 503,
    });
    await vi.waitFor(() =>
      expect(container.querySelector('.osprey-state')?.textContent).toContain(
        'Directory did not answer:'
      )
    );
    const retry = button(container, 'Retry');
    expect(retry).toBeDefined();
    retry?.click();
    await vi.waitFor(() => expect(searches).toHaveLength(2));
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

  test('an alarm feed without an answer dashes the column and names it in partial', async () => {
    const reason = 'alarm feed still loading';
    const search = {
      ...ROUTES.search,
      rows: ROUTES.search.rows.map(withoutAlarm),
      partial: [{ part: 'alarm', reason }],
    };
    const { container } = await mount({ search });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const cells = [...container.querySelectorAll('td.allpvs-alarm')];
    expect(cells.map((td) => [td.textContent, td.getAttribute('title')])).toEqual([
      ['—', reason],
      ['—', reason],
    ]);
    expect(container.querySelector('.osprey-chip--warn')?.textContent).toBe('partial');
  });

  test('no alarm feed, no Alarm column', async () => {
    const search = {
      ...ROUTES.search,
      rows: ROUTES.search.rows.map(withoutAlarm),
    };
    const { container } = await mount({ search });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(container.querySelector('td.allpvs-alarm')).toBeNull();
    const heads = [...container.querySelectorAll('th')].map((th) => th.textContent);
    expect(heads).not.toContain('Alarm');
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

  test('the head states the words of the answer shown, not the words being typed', async () => {
    const { container, searches } = await mount();
    const input = /** @type {HTMLInputElement} */ (document.getElementById('cf-search'));
    input.value = 'BPM02';
    input.dispatchEvent(new Event('input'));
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(searches).toHaveLength(1);
    expect(container.querySelector('.allpvs-searched')?.textContent).toBe('every PV');

    await vi.waitFor(() => expect(searches).toHaveLength(2));
    await vi.waitFor(() =>
      expect(container.querySelector('.allpvs-searched')?.textContent).toBe('matches BPM02')
    );
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
    await vi.waitFor(() => expect(sidebar.querySelectorAll('.osprey-list-row')).toHaveLength(9));
    const labels = [...sidebar.querySelectorAll('.osprey-section-label')].map((l) => l.textContent);
    expect(labels).toEqual(['Alarm', 'Origin', 'IOC', 'Record type', 'Archive']);
    const curated = /** @type {HTMLButtonElement} */ (
      sidebar.querySelector('[data-facet="origin"][data-value="curated"]')
    );
    expect(curated.textContent).toContain('◆ Curated');
    expect(curated.textContent).toContain('1');
    expect(sidebar.textContent).toContain('◇ Control system only');
    curated.click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('origin=curated'));
  });

  test('the Alarm facet filters the search and writes the route', async () => {
    const { searches, routes } = await mount({ params: { q: 'BPM' } });
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    const major = () =>
      /** @type {HTMLButtonElement|null} */ (
        sidebar.querySelector('[data-facet="alarm"][data-value="MAJOR"]')
      );
    await vi.waitFor(() => expect(major()).not.toBeNull());
    expect(major()?.querySelector('.cf-sev--major')).not.toBeNull();
    expect(sidebar.querySelector('[data-facet="alarm"][data-value="none"]')?.textContent).toContain(
      'No alarm'
    );
    major()?.click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('alarm=MAJOR'));
    expect(routes.mock.calls.at(-1)?.[0].params).toEqual({ q: 'BPM', alarm: ['MAJOR'] });
  });

  test('each row carries its archive mark, and the Archive facet filters the search', async () => {
    const { container, searches, routes } = await mount({ params: { q: 'BPM' } });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const marks = [...container.querySelectorAll('td.allpvs-archived')].map((td) =>
      td.getAttribute('title')
    );
    expect(marks).toEqual(['Archived', 'Not archived']);
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    const no = /** @type {HTMLButtonElement} */ (
      sidebar.querySelector('[data-facet="archived"][data-value="no"]')
    );
    no.click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('archived=no'));
    expect(routes.mock.calls.at(-1)?.[0].params).toEqual({ q: 'BPM', archived: ['no'] });
  });

  test('an archive that cannot list its channels dots every row and hides the facet', async () => {
    const reason = 'this archiver does not report what it holds';
    const search = {
      ...ROUTES.search,
      rows: ROUTES.search.rows.map((/** @type {any} */ row) => ({ ...row, archived: null })),
      facets: { ...ROUTES.search.facets, archived: [{ value: 'unknown', count: 2 }] },
      archived_reason: reason,
    };
    const { container } = await mount({ search });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const cells = [...container.querySelectorAll('td.allpvs-archived')];
    expect(cells.map((td) => td.textContent)).toEqual(['·', '·']);
    expect(cells[0].getAttribute('title')).toBe(`unknown: ${reason}`);
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    expect(sidebar.querySelector('[data-facet="archived"]')).toBeNull();
  });

  test('without an archive on the target there is no Arch. column', async () => {
    const search = {
      ...ROUTES.search,
      rows: ROUTES.search.rows.map((/** @type {any} */ row) =>
        Object.fromEntries(Object.entries(row).filter(([key]) => key !== 'archived'))
      ),
    };
    const { container } = await mount({ search });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(container.querySelector('th.col-arch')).toBeNull();
  });

  test("a chip's × removes that one filter and writes the route", async () => {
    const { container, searches, routes } = await mount({
      params: { q: 'BPM', ioc: ['IOC1'], alarm: ['MAJOR'] },
    });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    const chips = () =>
      [...container.querySelectorAll('.allpvs-chip')].map((c) => c.firstChild?.textContent);
    expect(chips()).toEqual(['Alarm: MAJOR', 'IOC: IOC1']);
    /** @type {HTMLButtonElement} */ (
      container.querySelector('.allpvs-chip [data-facet="ioc"][data-value="IOC1"]')
    ).click();
    expect(routes.mock.calls.at(-1)?.[0].params).toEqual({ q: 'BPM', alarm: ['MAJOR'] });
    await vi.waitFor(() => expect(searches.at(-1)).not.toContain('ioc='));
    expect(searches.at(-1)).toContain('alarm=MAJOR');
    await vi.waitFor(() => expect(chips()).toEqual(['Alarm: MAJOR']));
  });

  test('clear all empties the words and the filters', async () => {
    const { container, state, searches, routes } = await mount({
      params: { q: 'BPM', origin: ['curated'] },
    });
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    button(container, 'clear all')?.click();
    expect(state.scopeQuery).toBe('');
    expect(/** @type {HTMLInputElement} */ (document.getElementById('cf-search')).value).toBe('');
    expect(routes.mock.calls.at(-1)?.[0].params).toEqual({ q: '' });
    await vi.waitFor(() => expect(searches.at(-1)).not.toContain('origin='));
    expect(searches.at(-1)).not.toContain('q=BPM');
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

describe('the address bar', () => {
  /** @param {import('vitest').Mock} routes */
  const lastParams = (routes) => routes.mock.calls.at(-1)?.[0].params;

  test('mount states the route words', async () => {
    const { routes } = await mount({ params: { q: 'BPM' } });
    expect(routes).toHaveBeenCalledTimes(1);
    expect(routes).toHaveBeenCalledWith({ view: 'allpvs', params: { q: 'BPM' } });
  });

  test('a facet click states the facet beside the words', async () => {
    const { routes } = await mount({ params: { q: 'BPM' } });
    const sidebar = /** @type {HTMLElement} */ (document.querySelector('.osprey-panel-sidebar'));
    await vi.waitFor(() =>
      expect(sidebar.querySelector('[data-facet="origin"][data-value="curated"]')).not.toBeNull()
    );
    /** @type {HTMLButtonElement} */ (
      sidebar.querySelector('[data-facet="origin"][data-value="curated"]')
    ).click();
    expect(lastParams(routes)).toEqual({ q: 'BPM', origin: ['curated'] });
  });

  test('new words state the words with the facets cleared', async () => {
    const { routes } = await mount({ params: { q: 'BPM', ioc: ['IOC1'] } });
    expect(lastParams(routes)).toEqual({ q: 'BPM', ioc: ['IOC1'] });
    const input = /** @type {HTMLInputElement} */ (document.getElementById('cf-search'));
    input.value = 'QF';
    input.dispatchEvent(new Event('input'));
    expect(routes).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(lastParams(routes)).toEqual({ q: 'QF' }));
  });

  test('blank words are stated as an empty q', async () => {
    const { routes } = await mount();
    expect(routes).toHaveBeenCalledWith({ view: 'allpvs', params: { q: '' } });
  });

  test('paging states nothing', async () => {
    const { container, searches, routes } = await mount({ search: pageOf(60) });
    await vi.waitFor(() => expect(rows(container).length).toBeGreaterThan(1));
    const before = routes.mock.calls.length;
    /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="next"]')).click();
    await vi.waitFor(() => expect(searches.at(-1)).toContain('page=1'));
    expect(routes).toHaveBeenCalledTimes(before);
  });
});

describe('Plot and Send', () => {
  test('the selection bar shows only while something is selected', async () => {
    const { container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    expect(container.querySelector('.allpvs-selection-bar')).toBeNull();
    select(container, 0);
    expect(container.querySelector('.allpvs-selection-bar')?.textContent).toContain('1 selected');
    button(container, 'Clear')?.click();
    expect(container.querySelector('.allpvs-selection-bar')).toBeNull();
  });

  test('Plot posts the Canopy url with pv in selection order', async () => {
    const { container, postMessage } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    select(container, 1);
    select(container, 0);
    await vi.waitFor(() => expect(plot(container)).not.toBeNull());
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
    await vi.waitFor(() => expect(rows(container)).toHaveLength(13));
    for (let i = 0; i < 12; i++) select(container, i);
    await vi.waitFor(() => expect(plot(container)).not.toBeNull());
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

  test('Copy names copies the selection in selection order, one per line', async () => {
    const writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    const { container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    select(container, 1);
    select(container, 0);
    button(container, 'Copy names')?.click();
    await vi.waitFor(() =>
      expect(writeText).toHaveBeenCalledWith('DEMO:BPM01:HEARTBEAT\nDEMO:BPM01:X')
    );
  });

  test('a pop-out shows no Plot and no Send', async () => {
    const { container, fetchMock } = await mount({ framed: false });
    await vi.waitFor(() => expect(rows(container)[0].textContent).toContain('0.125'));
    select(container, 0);
    expect(container.querySelector('.allpvs-selection-bar')).not.toBeNull();
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
    select(container, 0);
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
  const selection = (root) => root.querySelector('.allpvs-selection')?.textContent ?? '';

  /** @param {HTMLElement} root */
  const pager = (root) => root.querySelector('.allpvs-pager')?.textContent ?? '';

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
    await vi.waitFor(() => expect(pager(container)).toContain('page 2'));
    expect(selection(container)).toContain('2 selected');

    button(container, 'Clear')?.click();
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

  /**
   * Select a row, then leave and come back the way app.js does: the active
   * view changes before each mount.
   * @param {string} away
   */
  async function visit(away) {
    const { state, view, container } = await mount();
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    select(container, 0);
    expect(selection(container)).toContain('1 selected');
    view.unmountAllPvs();
    state.setActiveView(away);
    state.setActiveView('allpvs');
    view.mountAllPvs(container, {});
    await vi.waitFor(() => expect(rows(container)).toHaveLength(2));
    return container;
  }

  test.each([
    ['a PV', 'pv'],
    ['an IOC', 'ioc'],
  ])('%s page and Back keep it', async (_, away) => {
    expect(selection(await visit(away))).toContain('1 selected');
  });

  test('leaving All PVs empties it', async () => {
    expect(selection(await visit('explore'))).not.toContain('selected');
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

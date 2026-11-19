// @ts-check
/**
 * The IOCs view (servers.js, servers-render.js): the header, the tiles and the
 * text filter, grouping, the outcome statements, the links, and the poll.
 * The answers are the shared route fixture's; the toolbar is index.html's.
 *   npx vitest run tests/interfaces/channel_finder/iocs.test.mjs
 */

import { epicsDirectory } from './epics-words.mjs';
import { test, expect, vi, afterEach, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);

/** The fixture's fleet answer: demo-ioc-rf silent and expected running, then demo-ioc-bpm alive. */
const FLEET = ROUTES.fleet;

const PANEL_HTML = (() => {
  const html = readFileSync(
    join(import.meta.dirname, '../../../src/osprey/interfaces/channel_finder/static/index.html'),
    'utf8'
  );
  const match = html.match(/<div class="cf-panel"[\s\S]*<\/main>/);
  if (!match) throw new Error('index.html carries no panel root');
  return `${match[0]}</div>`;
})();

const DESCRIPTOR = epicsDirectory({
  label: 'All PVs',
  configured_services: ['fleet'],
  canopy: true,
  ariel: true,
});

/**
 * A fetch answer.
 * @param {number} status
 * @param {any} body
 */
function answer(status, body) {
  const text = JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: '',
    json: async () => JSON.parse(text),
    text: async () => text,
  };
}

/** @type {(() => void) | null} */
let unmount = null;

afterEach(() => {
  unmount?.();
  unmount = null;
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

/**
 * Mount the IOCs view in a fresh module registry against a fetch that
 * answers the fleet route with `fleet` (or what it returns).
 * @param {{ fleet?: any | (() => any), status?: number }} [setup]
 */
async function mount({ fleet = FLEET, status = 200 } = {}) {
  window.history.replaceState(null, '', '/?embedded=true#/servers');
  document.body.innerHTML = PANEL_HTML;
  /** @type {string[]} */
  const reads = [];
  const fetchMock = vi.fn(async (/** @type {any} */ url) => {
    const target = String(url);
    reads.push(target);
    if (target.startsWith('/api/directory/fleet')) {
      return answer(status, typeof fleet === 'function' ? await fleet() : fleet);
    }
    return answer(404, { detail: 'Not Found' });
  });
  vi.stubGlobal('fetch', fetchMock);

  vi.resetModules();
  const { state } = await import(`${JS}/state.js`);
  state.directory = { ...DESCRIPTOR };
  const view = await import(`${JS}/servers.js`);
  const container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
  view.mountServers(container);
  unmount = view.unmountServers;
  return { view, container, fetchMock, reads };
}

/** @param {HTMLElement} root */
const rows = (root) => [
  .../** @type {NodeListOf<HTMLElement>} */ (root.querySelectorAll('.servers-list tr[data-name]')),
];

/** @param {HTMLElement} root */
const names = (root) => rows(root).map((row) => row.dataset.name);

/** @param {HTMLElement} root @param {string} key */
const tile = (root, key) =>
  /** @type {HTMLButtonElement|null} */ (root.querySelector(`[data-tile="${key}"]`));

/** @param {HTMLElement} root @param {string} key */
const tileCount = (root, key) => tile(root, key)?.querySelector('.servers-tile-n')?.textContent;

/** Wait for the first answer to be drawn. @param {HTMLElement} root */
const drawn = (root) => vi.waitFor(() => expect(rows(root).length).toBeGreaterThan(0));

/** @returns {string} */
const resultCount = () => document.getElementById('cf-result-count')?.textContent ?? '';

/** @param {HTMLElement} root */
const statement = (root) => /** @type {HTMLElement} */ (root.querySelector('.servers-state'));

/** @param {string} words */
function type(words) {
  const input = /** @type {HTMLInputElement} */ (document.getElementById('cf-search'));
  input.value = words;
  input.dispatchEvent(new Event('input'));
}

/** @param {Record<string, any>} snapshot */
const withSnapshot = (snapshot) => ({
  ...FLEET,
  snapshot: { ...FLEET.snapshot, ...snapshot },
});

describe('an ok answer', () => {
  test('lists the IOCs in the served order with their state, findings and links', async () => {
    const { container } = await mount();
    await drawn(container);
    expect(names(container)).toEqual(['demo-ioc-rf', 'demo-ioc-bpm']);
    const [rf] = rows(container);
    const cells = [...rf.querySelectorAll('td')].map((cell) => cell.textContent?.trim());
    expect(cells[0]).toBe('demo-ioc-rf');
    expect(cells[1]).toBe('ioc-host-02');
    expect(cells[2]).toBe('silent');
    expect(cells[4]).toBe('Expected running, not alive');
    expect(cells[5]).toBe('Active');
    expect(cells[6]).toBe('directory');
    expect(rf.querySelector('a')?.getAttribute('href')).toBe('#/server/demo-ioc-rf');
    expect(rf.querySelector('[data-state]')?.getAttribute('title')).toBe(
      'heartbeat did not answer'
    );
  });

  test('time in state is how long ago the state began', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date((FLEET.snapshot.generated_at + 60) * 1000));
    const { container } = await mount();
    await drawn(container);
    const [rf, bpm] = rows(container);
    // rf since generated_at - 600 → 660 s ago; bpm since generated_at - 100 → 160 s ago.
    expect(rf.querySelector('td.num')?.textContent).toBe('11 min');
    expect(bpm.querySelector('td.num')?.textContent).toBe('2 min');
  });

  test('the header states the snapshot time, the read path and each source', async () => {
    const { container } = await mount();
    await drawn(container);
    const header = /** @type {HTMLElement} */ (container.querySelector('.servers-header'));
    expect(header.hidden).toBe(false);
    expect(header.textContent).toContain('2023-11-14 22:13:20 UTC');
    expect(header.textContent).toContain('live');
    expect(header.querySelector('[data-role="read-path"]')?.textContent).toBe('read path ok');
    expect(header.querySelector('[data-source="directory"]')?.textContent).toBe('directory: ok');
  });

  test('a down read path says every IOC is unknown; none says it is unverified', async () => {
    const { container } = await mount({
      fleet: withSnapshot({ read_path: { status: 'down', channel: 'SR:PROBE' } }),
    });
    await drawn(container);
    const chip = container.querySelector('[data-role="read-path"]');
    expect(chip?.textContent).toBe('read path down: every IOC unknown');
    expect(chip?.className).toContain('osprey-chip--err');
    unmount?.();
    const again = await mount({
      fleet: withSnapshot({ read_path: { status: 'unverified', channel: null } }),
    });
    await drawn(again.container);
    expect(again.container.querySelector('[data-role="read-path"]')?.textContent).toBe(
      'read path unverified'
    );
  });

  test('a source that failed shows its outcome and reason', async () => {
    const { container } = await mount({
      fleet: withSnapshot({
        sources: [
          { name: 'directory', outcome: 'ok', reason: null },
          { name: 'roster', outcome: 'unavailable', reason: 'roster file missing' },
        ],
      }),
    });
    await drawn(container);
    const roster = container.querySelector('[data-source="roster"]');
    expect(roster?.textContent).toBe('roster: unavailable · roster file missing');
    expect(roster?.className).toContain('osprey-chip--warn');
  });

  test('the toolbar counts the IOCs', async () => {
    const { container } = await mount();
    await drawn(container);
    expect(resultCount()).toBe('2 IOCs');
  });

  test('unreached and unknown take the neutral chip, never a red one', async () => {
    const [rf, bpm] = FLEET.snapshot.servers;
    const { container } = await mount({
      fleet: withSnapshot({
        servers: [
          { ...rf, state: 'unreached', findings: [] },
          { ...bpm, state: 'unknown', findings: [] },
        ],
      }),
    });
    await drawn(container);
    for (const chip of container.querySelectorAll('[data-state]')) {
      expect(chip.className).toBe('osprey-chip');
    }
  });

  test('every value is text, never markup', async () => {
    const [rf] = FLEET.snapshot.servers;
    const { container } = await mount({
      fleet: withSnapshot({
        servers: [{ ...rf, name: '<img src=x onerror=alert(1)>', host: '<b>h</b>' }],
      }),
    });
    await drawn(container);
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('.servers-list b')).toBeNull();
    expect(rows(container)[0].dataset.name).toBe('<img src=x onerror=alert(1)>');
  });
});

describe('the tiles and the text filter', () => {
  test('a tile per state and per finding carries the snapshot counts', async () => {
    const { container } = await mount();
    await drawn(container);
    expect(tileCount(container, 'state:alive')).toBe('1');
    expect(tileCount(container, 'state:silent')).toBe('1');
    expect(tileCount(container, 'state:unknown')).toBe('0');
    expect(tile(container, 'state:unknown')?.classList.contains('is-zero')).toBe(true);
    expect(tileCount(container, 'finding:expected_running_not_alive')).toBe('1');
    expect(tileCount(container, 'finding:degraded')).toBe('0');
  });

  test('a state tile keeps only its rows, and a second press clears it', async () => {
    const { container } = await mount();
    await drawn(container);
    tile(container, 'state:alive')?.click();
    expect(names(container)).toEqual(['demo-ioc-bpm']);
    expect(tile(container, 'state:alive')?.getAttribute('aria-pressed')).toBe('true');
    expect(resultCount()).toBe('1 of 2 IOCs');
    tile(container, 'state:alive')?.click();
    expect(names(container)).toEqual(['demo-ioc-rf', 'demo-ioc-bpm']);
  });

  test('a finding tile keeps the rows carrying it', async () => {
    const { container } = await mount();
    await drawn(container);
    tile(container, 'finding:expected_running_not_alive')?.click();
    expect(names(container)).toEqual(['demo-ioc-rf']);
  });

  test('the toolbar text filters by name or host, case blind', async () => {
    const { container } = await mount();
    await drawn(container);
    type('BPM');
    expect(names(container)).toEqual(['demo-ioc-bpm']);
    type('host-02');
    expect(names(container)).toEqual(['demo-ioc-rf']);
  });

  test('a filter that keeps nothing says so and offers to clear it', async () => {
    const { container } = await mount();
    await drawn(container);
    type('nothing-matches');
    expect(names(container)).toEqual([]);
    const list = /** @type {HTMLElement} */ (container.querySelector('.servers-list'));
    expect(list.textContent).toContain('No IOC for this filter');
    /** @type {HTMLButtonElement} */ (list.querySelector('button')).click();
    expect(names(container)).toEqual(['demo-ioc-rf', 'demo-ioc-bpm']);
    expect(/** @type {HTMLInputElement} */ (document.getElementById('cf-search')).value).toBe('');
  });
});

describe('grouping', () => {
  test('groups by the snapshot key, in the order the groups first appear', async () => {
    const { container } = await mount();
    await drawn(container);
    const toggle = /** @type {HTMLButtonElement} */ (
      container.querySelector('[data-action="group"]')
    );
    expect(toggle.textContent).toBe('Group by host');
    toggle.click();
    const groups = [...container.querySelectorAll('tbody[data-group]')].map(
      (body) => /** @type {HTMLElement} */ (body).dataset.group
    );
    expect(groups).toEqual(['ioc-host-02', 'ioc-host-01']);
    expect(names(container)).toEqual(['demo-ioc-rf', 'demo-ioc-bpm']);
    /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="group"]')).click();
    expect(container.querySelectorAll('tbody[data-group]')).toHaveLength(0);
  });

  test('a snapshot with no grouping key offers no toggle', async () => {
    const { container } = await mount({ fleet: withSnapshot({ group_by: null }) });
    await drawn(container);
    expect(container.querySelector('[data-action="group"]')).toBeNull();
  });
});

describe('every non-ok outcome is a statement with no rows', () => {
  const NON_OK = [
    ['no_snapshot', 'No snapshot yet'],
    ['not_refreshed', 'Target not refreshed'],
    ['cannot_list', 'Cannot list the IOCs'],
    ['unavailable', 'server fleet unavailable'],
    ['stale', 'Snapshot stale'],
  ];

  test.each(NON_OK)('%s', async (outcome, message) => {
    const fleet = {
      outcome,
      target: 'live',
      reason: outcome === 'no_snapshot' || outcome === 'not_refreshed' ? null : 'the reason',
      generated_at: outcome === 'stale' ? FLEET.generated_at : null,
      snapshot: outcome === 'stale' ? FLEET.snapshot : null,
    };
    const { container } = await mount({ fleet });
    await vi.waitFor(() => expect(statement(container).textContent).toContain(message));
    expect(rows(container)).toHaveLength(0);
    expect(/** @type {HTMLElement} */ (container.querySelector('.servers-band')).hidden).toBe(true);
    expect(/** @type {HTMLElement} */ (container.querySelector('.servers-header')).hidden).toBe(
      true
    );
    expect(resultCount()).toBe('');
    const text = statement(container).textContent ?? '';
    expect(text).toContain('live');
    expect(text).not.toMatch(/\bdown\b/i);
    if (outcome === 'stale') expect(text).toContain('2023-11-14 22:13:20 UTC');
    else expect(text).toContain('This says nothing about whether any IOC runs.');
  });

  test('a route that does not answer is a statement with Retry', async () => {
    let calls = 0;
    const { container } = await mount({
      fleet: () => (calls++ === 0 ? { detail: 'boom' } : FLEET),
      status: 200,
    });
    // The first answer is no fleet answer at all: shown as unavailable.
    await vi.waitFor(() =>
      expect(statement(container).textContent).toContain('server fleet unavailable')
    );
    /** @type {HTMLButtonElement} */ (statement(container).querySelector('button')).click();
    await drawn(container);
    expect(statement(container).hidden).toBe(true);
  });

  test('an HTTP failure is a statement, never rows', async () => {
    const { container } = await mount({ fleet: { detail: 'Not Found' }, status: 404 });
    await vi.waitFor(() =>
      expect(statement(container).textContent).toContain('The server fleet route did not answer')
    );
    expect(rows(container)).toHaveLength(0);
  });

  test('an ok answer after a statement brings the rows back', async () => {
    let calls = 0;
    const none = {
      outcome: 'no_snapshot',
      target: 'live',
      reason: null,
      generated_at: null,
      snapshot: null,
    };
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, view } = await mount({ fleet: () => (calls++ === 0 ? none : FLEET) });
    await vi.waitFor(() => expect(statement(container).textContent).toContain('No snapshot yet'));
    vi.advanceTimersByTime(view.POLL_MS);
    await drawn(container);
    expect(statement(container).hidden).toBe(true);
  });
});

describe('the poll', () => {
  test('reads every 5 s while mounted and visible, and stops on unmount', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, view, reads } = await mount();
    await drawn(container);
    const fleetReads = () => reads.filter((url) => url.startsWith('/api/directory/fleet'));
    expect(fleetReads()).toHaveLength(1);
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(fleetReads()).toHaveLength(2));
    view.unmountServers();
    unmount = null;
    vi.advanceTimersByTime(view.POLL_MS * 4);
    await Promise.resolve();
    expect(fleetReads()).toHaveLength(2);
    expect(resultCount()).toBe('');
  });

  test('a hidden document is not polled', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, view, reads } = await mount();
    await drawn(container);
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    vi.advanceTimersByTime(view.POLL_MS * 3);
    await Promise.resolve();
    expect(reads.filter((url) => url.startsWith('/api/directory/fleet'))).toHaveLength(1);
  });

  test('only the fleet route is read: no PV value and no directory query', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, view, reads } = await mount();
    await drawn(container);
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(reads).toHaveLength(2));
    expect(reads.every((url) => url === '/api/directory/fleet')).toBe(true);
  });

  test('an identical answer leaves the table and the focus as they were', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, view, reads } = await mount();
    await drawn(container);
    const alive = /** @type {HTMLButtonElement} */ (tile(container, 'state:alive'));
    alive.focus();
    const firstRow = rows(container)[0];
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(reads).toHaveLength(2));
    await Promise.resolve();
    expect(rows(container)[0]).toBe(firstRow);
    expect(tile(container, 'state:alive')).toBe(alive);
    expect(document.activeElement).toBe(alive);
  });
});

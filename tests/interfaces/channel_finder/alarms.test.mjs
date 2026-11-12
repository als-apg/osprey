// @ts-check
/**
 * The Alarms view (alarms.js, alarms-render.js): the summary band and its
 * filters, the alarm tree, the stale feed, the list in served order, the
 * inspector, the states, and the poll. The answers are the shared route
 * fixture's; the toolbar is index.html's.
 *   npx vitest run tests/interfaces/channel_finder/alarms.test.mjs
 */

import { test, expect, vi, afterEach, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);

/** The fixture's alarm answer: a latched MAJOR row, then an acknowledged MINOR row. */
const ALARMS = ROUTES.alarms;
const [MAJOR_ROW, MINOR_ROW] = ALARMS.rows;

const PANEL_HTML = (() => {
  const html = readFileSync(
    join(import.meta.dirname, '../../../src/osprey/interfaces/channel_finder/static/index.html'),
    'utf8'
  );
  const match = html.match(/<div class="cf-panel"[\s\S]*<\/main>/);
  if (!match) throw new Error('index.html carries no panel root');
  return `${match[0]}</div>`;
})();

const DESCRIPTOR = {
  label: 'All PVs',
  configured_services: ['alarm_state'],
  canopy: true,
  ariel: true,
};

const GUIDANCE = [
  { title: 'Response', details: 'Check cavity 1 cooling water flow and the tuner position.' },
];

/**
 * The PV facts of `name`: the fixture's, its alarm configuration carrying the
 * demo replay's guidance list, with or without an alarm history section.
 * @param {string} name
 * @param {{ history?: boolean }} [options]
 */
function factsOf(name, { history = true } = {}) {
  const facts = structuredClone(ROUTES['pv/{name}']);
  facts.name = name;
  for (const section of facts.sections) {
    if (section.section === 'alarm_config') {
      section.record.guidance = GUIDANCE;
      section.record.displays = [{ title: 'RF cavity 1', details: 'rf/cavity_01.bob' }];
    }
    if (section.section === 'alarm_history') {
      section.outcome = 'found';
      section.rows = [
        {
          time: '2026-11-06T11:58:40.000Z',
          severity: 'MAJOR',
          status: 'HIHI_ALARM',
          message: 'HIHI_ALARM',
          user: null,
          kind: 'state',
        },
      ];
      delete section.reason;
    }
  }
  if (!history) {
    facts.sections = facts.sections.filter(
      (/** @type {any} */ section) => section.section !== 'alarm_history'
    );
  }
  return facts;
}

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
 * @typedef {{
 *   alarms?: any | (() => any),
 *   alarmsStatus?: number,
 *   facts?: (name: string) => any,
 *   framed?: boolean,
 * }} Setup
 */

/**
 * Mount the Alarms view in a fresh module registry against a fetch that
 * answers the directory routes.
 * @param {Setup} [setup]
 */
async function mount(setup = {}) {
  const { alarms = ALARMS, alarmsStatus = 200, facts = factsOf, framed = true } = setup;
  window.history.replaceState(null, '', '/u/a/panel/channel-finder/?embedded=true');
  const postMessage = vi.fn();
  vi.stubGlobal('parent', framed ? { postMessage } : window);
  document.body.innerHTML = PANEL_HTML;

  /** @type {string[]} */
  const reads = [];
  const fetchMock = vi.fn(async (/** @type {any} */ url, /** @type {any} */ opts = {}) => {
    const target = String(url);
    if (opts.method === 'HEAD') return answer(200, '');
    if (target.startsWith('/api/directory/alarms')) {
      reads.push(target);
      return answer(alarmsStatus, typeof alarms === 'function' ? alarms() : alarms);
    }
    if (target.startsWith('/api/directory/pv/')) {
      const name = decodeURIComponent(target.slice('/api/directory/pv/'.length));
      return answer(200, facts(name));
    }
    if (target.startsWith('/api/directory/values')) {
      const names = new URL(target, 'http://x').searchParams.getAll('name');
      return answer(
        200,
        names.map((name) => ({
          name,
          outcome: 'found',
          value: 46.2000001,
          timestamp: null,
          severity: 0,
          status: 'NO_ALARM',
        }))
      );
    }
    return answer(404, { detail: 'Not Found' });
  });
  vi.stubGlobal('fetch', fetchMock);

  vi.resetModules();
  const { state } = await import(`${JS}/state.js`);
  state.directory = { ...DESCRIPTOR };
  const counts = vi.fn();
  state.on('alarmCounts', counts);
  const view = await import(`${JS}/alarms.js`);
  const container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
  view.mountAlarms(container);
  unmount = view.unmountAlarms;
  return { state, view, container, fetchMock, reads, counts, postMessage };
}

/** @param {HTMLElement} root */
const rows = (root) => [
  .../** @type {NodeListOf<HTMLElement>} */ (root.querySelectorAll('.alarms-list tbody tr')),
];

/** @param {HTMLElement} root */
const names = (root) => rows(root).map((row) => row.dataset.name);

/** @param {HTMLElement} root @param {string} key */
const tile = (root, key) =>
  /** @type {HTMLButtonElement|null} */ (root.querySelector(`[data-tile="${key}"]`));

/** @param {HTMLElement} root @param {string} key */
const tileCount = (root, key) => tile(root, key)?.querySelector('.alarms-tile-n')?.textContent;

/** @param {HTMLElement} root */
const inspector = (root) => /** @type {HTMLElement} */ (root.querySelector('.alarms-inspector'));

/** @param {HTMLElement} root */
async function loaded(root) {
  await vi.waitFor(() => expect(rows(root).length).toBeGreaterThan(0));
}

/** The first row's value and the inspector's facts have answered. @param {HTMLElement} root */
async function settled(root) {
  await loaded(root);
  await vi.waitFor(() => expect(inspector(root).textContent).toContain('Response'));
  await vi.waitFor(() =>
    expect(rows(root)[0].querySelector('.alarms-value')?.textContent).toContain('46.2')
  );
}

/** @returns {HTMLElement|null} the page's info popover */
const popover = () => document.querySelector('.osprey-info-popover');

test('the view exports what the router mounts and unmounts', async () => {
  const view = await import(`${JS}/alarms.js`);
  expect(view.mountAlarms).toBeTypeOf('function');
  expect(view.unmountAlarms).toBeTypeOf('function');
});

describe('the summary band', () => {
  test('a tile per severity with its count, then the flags', async () => {
    const { container, counts } = await mount();
    await loaded(container);
    expect(tileCount(container, 'MAJOR')).toBe('1');
    expect(tileCount(container, 'INVALID')).toBe('0');
    expect(tile(container, 'INVALID')?.classList.contains('is-zero')).toBe(true);
    expect(tileCount(container, 'MINOR')).toBe('1');
    expect(tile(container, 'UNDEFINED')).toBeNull();
    expect(tileCount(container, 'unack')).toBe('1');
    expect(tileCount(container, 'latched')).toBe('1');
    const order = [...container.querySelectorAll('[data-tile]')].map(
      (el) => /** @type {HTMLElement} */ (el).dataset.tile
    );
    expect(order).toEqual(['MAJOR', 'INVALID', 'MINOR', 'unack', 'latched']);
    expect(document.getElementById('cf-result-count')?.textContent).toBe('2 active');
    expect(counts).toHaveBeenCalledWith(expect.objectContaining({ counts: ALARMS.counts }));
  });

  test('UNDEFINED gets a tile when its count is not zero', async () => {
    const { container } = await mount({
      alarms: { ...ALARMS, counts: { ...ALARMS.counts, UNDEFINED: 2 } },
    });
    await loaded(container);
    expect(tileCount(container, 'UNDEFINED')).toBe('2');
  });

  test('a tile filters and a second press clears', async () => {
    const { container } = await mount();
    await loaded(container);
    tile(container, 'MAJOR')?.click();
    expect(names(container)).toEqual([MAJOR_ROW.pv]);
    expect(tile(container, 'MAJOR')?.getAttribute('aria-pressed')).toBe('true');
    expect(document.getElementById('cf-result-count')?.textContent).toBe('1 active');
    tile(container, 'MAJOR')?.click();
    expect(names(container)).toEqual([MAJOR_ROW.pv, MINOR_ROW.pv]);
    expect(tile(container, 'MAJOR')?.getAttribute('aria-pressed')).toBe('false');
  });

  test('Unacknowledged and Latched filter by flag', async () => {
    const { container } = await mount();
    await loaded(container);
    tile(container, 'unack')?.click();
    expect(names(container)).toEqual([MAJOR_ROW.pv]);
    tile(container, 'latched')?.click();
    expect(names(container)).toEqual([MAJOR_ROW.pv]);
    expect(tile(container, 'unack')?.getAttribute('aria-pressed')).toBe('false');
  });

  test('a filter that keeps nothing offers Clear filter', async () => {
    const { container } = await mount();
    await loaded(container);
    tile(container, 'INVALID')?.click();
    const state = /** @type {HTMLElement} */ (
      container.querySelector('.alarms-list .osprey-state')
    );
    expect(state.textContent).toContain('No active alarms for this filter');
    expect(inspector(container).hidden).toBe(true);
    /** @type {HTMLButtonElement} */ (state.querySelector('button')).click();
    expect(names(container)).toEqual([MAJOR_ROW.pv, MINOR_ROW.pv]);
  });

  test('the feed chip says live, and opens the source', async () => {
    const { container } = await mount();
    await loaded(container);
    const chip = /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="feed"]'));
    expect(chip.classList.contains('osprey-chip--ok')).toBe(true);
    expect(chip.textContent).toMatch(/^feed live · last contact \d+ [a-z]+.* ago$/);
    chip.click();
    const pop = /** @type {HTMLElement} */ (document.querySelector('.osprey-info-popover'));
    expect(pop.textContent).toContain('alarm_replay');
    expect(pop.textContent).toContain('PVs in tree');
    expect(pop.textContent).toContain('3');
  });
});

describe('the list', () => {
  test('rows are drawn in the served order, never sorted', async () => {
    const served = ['INVALID', 'MAJOR', 'MINOR'].map((severity, i) => ({
      ...MAJOR_ROW,
      pv: `PV:${i}`,
      severity,
      path: null,
    }));
    const { container } = await mount({ alarms: { ...ALARMS, rows: served } });
    await loaded(container);
    expect(names(container)).toEqual(['PV:0', 'PV:1', 'PV:2']);
    expect(inspector(container).textContent).toContain('PV:0');
  });

  test('a row reads severity, PV link, status, value, duration and flags', async () => {
    const { container } = await mount();
    await loaded(container);
    const [major, minor] = rows(container);
    expect(major.querySelector('.cf-sev--major')?.textContent).toBe('MAJOR');
    expect(major.querySelector('a')?.getAttribute('href')).toBe(
      '#/pv/SR%3ARF%3ACAVITY%3A01%3ATEMPERATURE%3ARB'
    );
    expect(major.textContent).toContain('NO_ALARM');
    expect(major.querySelector('.alarms-flags')?.textContent).toContain('latched');
    expect(major.querySelector('.alarms-flags')?.textContent).toContain('unacknowledged');
    expect(minor.querySelector('.alarms-flags')?.textContent).toBe('acknowledged');
    await vi.waitFor(() => expect(major.ownerDocument.body.textContent).toContain('46.2'));
    expect(rows(container)[0].querySelector('.alarms-for')?.getAttribute('title')).not.toBe('');
  });

  test('a cut answer names the served and the active counts', async () => {
    const { container } = await mount({
      alarms: { ...ALARMS, truncated: true, counts: { MAJOR: 40, MINOR: 2, OK: 3 } },
    });
    await loaded(container);
    expect(container.querySelector('.alarms-cut')?.textContent).toBe(
      'Worst 2 of 42 active alarms shown'
    );
    expect(document.getElementById('cf-result-count')?.textContent).toBe('2 active');
  });

  test('an answer that is not cut has no such line', async () => {
    const { container } = await mount();
    await loaded(container);
    expect(container.querySelector('.alarms-cut')).toBeNull();
  });

  test('a PV name is text, never markup', async () => {
    const evil = { ...MAJOR_ROW, pv: '<img src=x onerror=alert(1)>', path: null };
    const { container } = await mount({ alarms: { ...ALARMS, rows: [evil] } });
    await loaded(container);
    expect(container.querySelector('img')).toBeNull();
    expect(names(container)).toEqual([evil.pv]);
  });

  test('no active rows: the centred state with the size of the feed', async () => {
    const { container } = await mount({ alarms: { ...ALARMS, counts: { OK: 3 }, rows: [] } });
    await vi.waitFor(() =>
      expect(container.querySelector('.alarms-list .osprey-state')?.textContent).toContain(
        'No active alarms'
      )
    );
    expect(container.textContent).toContain('The alarm feed holds 3 PVs, all OK.');
  });

  test('a stale feed shows the banner and dims the rows', async () => {
    const { container } = await mount({ alarms: { ...ALARMS, stale: true } });
    await loaded(container);
    const banner = /** @type {HTMLElement} */ (container.querySelector('.alarms-banner'));
    expect(banner.hidden).toBe(false);
    expect(banner.textContent).toContain('Alarm feed not updating.');
    expect(banner.textContent).toContain('No contact since');
    expect(banner.textContent).toContain('may be out of date');
    expect(rows(container).every((row) => row.classList.contains('is-stale'))).toBe(true);
    const chip = container.querySelector('[data-action="feed"]');
    expect(chip?.classList.contains('osprey-chip--warn')).toBe(true);
    expect(chip?.textContent).toBe('feed stale');
  });

  test('a live feed has no banner', async () => {
    const { container } = await mount();
    await loaded(container);
    expect(/** @type {HTMLElement} */ (container.querySelector('.alarms-banner')).hidden).toBe(
      true
    );
  });
});

describe('the alarm tree', () => {
  test('the paths become nested nodes with the worst severity and count', async () => {
    const { container } = await mount();
    await loaded(container);
    const sidebar = /** @type {HTMLElement} */ (container.querySelector('.alarms-tree'));
    expect(sidebar.hidden).toBe(false);
    const nodes = [...sidebar.querySelectorAll('[data-node]')].map(
      (el) => /** @type {HTMLElement} */ (el).dataset.node
    );
    expect(nodes).toEqual(['/Demo', '/Demo/SR', '/Demo/SR/RF', '/Demo/SR/Vacuum']);
    const demo = /** @type {HTMLElement} */ (sidebar.querySelector('[data-node="/Demo"]'));
    expect(demo.querySelector('.cf-sev--major')).not.toBeNull();
    expect(demo.querySelector('.osprey-nav-badge')?.textContent).toBe('2');
    const vacuum = /** @type {HTMLElement} */ (
      sidebar.querySelector('[data-node="/Demo/SR/Vacuum"]')
    );
    expect(vacuum.querySelector('.cf-sev--minor')).not.toBeNull();
    const toggle = /** @type {HTMLButtonElement} */ (document.getElementById('cf-sidebar-toggle'));
    expect(toggle.disabled).toBe(false);
  });

  test('a node filters the list to its subtree, and a second press clears', async () => {
    const { container } = await mount();
    await loaded(container);
    const node = () =>
      /** @type {HTMLElement} */ (container.querySelector('[data-node="/Demo/SR/Vacuum"]'));
    node().click();
    expect(names(container)).toEqual([MINOR_ROW.pv]);
    expect(node().getAttribute('aria-pressed')).toBe('true');
    node().click();
    expect(names(container)).toEqual([MAJOR_ROW.pv, MINOR_ROW.pv]);
  });

  test('a node and a tile filter together', async () => {
    const { container } = await mount();
    await loaded(container);
    /** @type {HTMLElement} */ (container.querySelector('[data-node="/Demo/SR/RF"]')).click();
    tile(container, 'MINOR')?.click();
    expect(names(container)).toEqual([]);
  });

  test('with no path on any row there is no sidebar, and the toggle says why', async () => {
    const flat = ALARMS.rows.map((/** @type {any} */ row) => ({ ...row, path: null }));
    const { container } = await mount({ alarms: { ...ALARMS, rows: flat } });
    await loaded(container);
    expect(/** @type {HTMLElement} */ (container.querySelector('.alarms-tree')).hidden).toBe(true);
    const toggle = /** @type {HTMLButtonElement} */ (document.getElementById('cf-sidebar-toggle'));
    expect(toggle.disabled).toBe(true);
    expect(toggle.title).toBe('No alarm tree on this target');
  });

  test('the toggle keeps its reason when the toolbar refreshes', async () => {
    const flat = ALARMS.rows.map((/** @type {any} */ row) => ({ ...row, path: null }));
    const { container, state, view } = await mount({ alarms: { ...ALARMS, rows: flat } });
    // At boot the toolbar is wired before any view mounts.
    const { initToolbar } = await import(`${JS}/panel-toolbar.js`);
    initToolbar();
    view.mountAlarms(container);
    await loaded(container);
    state.emit('pipelineChanged', 'hierarchical');
    const toggle = /** @type {HTMLButtonElement} */ (document.getElementById('cf-sidebar-toggle'));
    expect(toggle.title).toBe('No alarm tree on this target');
  });
});

describe('the inspector', () => {
  test('shows the first row, guidance before the facts, and transitions', async () => {
    const { container } = await mount();
    await loaded(container);
    const insp = inspector(container);
    await vi.waitFor(() => expect(insp.textContent).toContain('Response'));
    expect(insp.textContent).toContain(MAJOR_ROW.pv);
    expect(insp.querySelector('.cf-sev--major')?.textContent).toBe('MAJOR · NO_ALARM');
    expect(insp.textContent).toContain('BPM 01 horizontal position');
    expect(insp.textContent).toContain(GUIDANCE[0].details);
    expect(insp.textContent).not.toContain('[object Object]');
    const labels = [...insp.querySelectorAll('.osprey-section-label')].map((el) => el.textContent);
    expect(labels.indexOf('Guidance')).toBeGreaterThan(-1);
    expect(labels.indexOf('Guidance')).toBeLessThan(labels.indexOf('Facts'));
    expect(labels).toContain('Displays');
    expect(labels.at(-1)).toBe('Transitions');
    expect(insp.textContent).toContain(
      'The record is back to NO_ALARM. The alarm holds until it is acknowledged.'
    );
    expect(insp.textContent).toContain("Read-only. Acknowledge in the alarm server's own client.");
    expect(insp.querySelector('a.osprey-btn')?.getAttribute('href')).toBe(
      '#/pv/SR%3ARF%3ACAVITY%3A01%3ATEMPERATURE%3ARB'
    );
    expect(insp.querySelector('[data-action="send"]')).not.toBeNull();
  });

  test('without an alarm history section, the one line says so', async () => {
    const { container } = await mount({ facts: (name) => factsOf(name, { history: false }) });
    await loaded(container);
    const insp = inspector(container);
    await vi.waitFor(() => expect(insp.textContent).toContain('Response'));
    expect(insp.textContent).toContain(
      'No alarm log on this target. Only the current state and its start time are known.'
    );
    const labels = [...insp.querySelectorAll('.osprey-section-label')].map((el) => el.textContent);
    expect(labels).not.toContain('Transitions');
  });

  test('a click on a row selects it', async () => {
    const { container } = await mount();
    await loaded(container);
    /** @type {HTMLElement} */ (rows(container)[1].querySelector('.alarms-status')).click();
    expect(rows(container)[1].getAttribute('aria-selected')).toBe('true');
    expect(inspector(container).textContent).toContain(MINOR_ROW.pv);
    expect(inspector(container).textContent).not.toContain('The record is back to');
  });

  test('on a narrow panel a row opens its PV page', async () => {
    const { container } = await mount();
    await loaded(container);
    const real = window.getComputedStyle.bind(window);
    vi.spyOn(window, 'getComputedStyle').mockImplementation((el) =>
      el === inspector(container)
        ? /** @type {CSSStyleDeclaration} */ (/** @type {unknown} */ ({ display: 'none' }))
        : real(el)
    );
    /** @type {HTMLElement} */ (rows(container)[1].querySelector('.alarms-status')).click();
    expect(location.hash).toBe('#/pv/SR%3AVAC%3AGAUGE%3ASR03%3APRESSURE%3ARB');
  });

  test('Send to assistant posts the selected PV to the host', async () => {
    const { container, postMessage } = await mount();
    await loaded(container);
    /** @type {HTMLElement} */ (inspector(container).querySelector('[data-action="send"]')).click();
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'osprey-paste-to-terminal', text: MAJOR_ROW.pv },
      window.location.origin
    );
    expect(document.querySelector('.osprey-toast-host')?.textContent).toContain(
      `Posted ${MAJOR_ROW.pv} to the prompt`
    );
  });

  test('a selected row beyond the values read gets its value read', async () => {
    const many = Array.from({ length: 60 }, (_, i) => ({ ...MINOR_ROW, pv: `X:${i}` }));
    const { container, fetchMock } = await mount({ alarms: { ...ALARMS, rows: many } });
    await loaded(container);
    await vi.waitFor(() =>
      expect(rows(container)[0].querySelector('.alarms-value')?.textContent).toContain('46.2')
    );
    /** @type {HTMLElement} */ (rows(container)[55].querySelector('.alarms-status')).click();
    await vi.waitFor(() =>
      expect(inspector(container).querySelector('.alarms-insp-value')?.textContent).toContain(
        '46.2'
      )
    );
    const valueReads = fetchMock.mock.calls
      .map(([url]) => String(url))
      .filter((url) => url.startsWith('/api/directory/values'));
    expect(valueReads.at(-1)).toContain('X%3A55');
  });

  test('without a host there is no Plot and no Send', async () => {
    const { container } = await mount({ framed: false });
    await loaded(container);
    expect(inspector(container).querySelector('[data-action="send"]')).toBeNull();
    expect(inspector(container).querySelector('[data-action="plot"]')).toBeNull();
  });
});

describe('states', () => {
  test('an unavailable feed shows its reason with Retry', async () => {
    let calls = 0;
    const unavailable = {
      outcome: 'unavailable',
      source: ALARMS.source,
      caveats: [],
      reason: 'no answer within 6 s',
    };
    const { container, reads } = await mount({
      alarms: () => (calls++ === 0 ? unavailable : ALARMS),
    });
    const state = () => container.querySelector('.alarms-state .osprey-state');
    await vi.waitFor(() =>
      expect(state()?.textContent).toContain('Alarm feed unavailable: no answer within 6 s')
    );
    expect(/** @type {HTMLElement} */ (container.querySelector('.alarms-band')).hidden).toBe(true);
    /** @type {HTMLButtonElement} */ (container.querySelector('.alarms-state button')).click();
    await loaded(container);
    expect(reads).toHaveLength(2);
    expect(state()).toBeNull();
  });

  test('a route that does not answer shows the error with Retry', async () => {
    const { container } = await mount({ alarmsStatus: 503, alarms: { detail: 'down' } });
    await vi.waitFor(() =>
      expect(container.querySelector('.alarms-state .osprey-state')?.textContent).toContain(
        'Alarm feed did not answer: down'
      )
    );
    expect(container.querySelector('.alarms-state button')?.textContent).toBe('Retry');
  });

  test('a target with no alarm-state source', async () => {
    const none = {
      outcome: 'unavailable',
      source: { target: 'live', service: 'none', label: 'none', rehearsal: false },
      caveats: [],
      reason: 'not available for this target',
    };
    const { container } = await mount({ alarms: none });
    await vi.waitFor(() =>
      expect(container.querySelector('.alarms-state .osprey-state')?.textContent).toBe(
        'Not available for this target'
      )
    );
    expect(container.querySelector('.alarms-state button')).toBeNull();
  });

  test('a failed service with no source shows its reason with Retry', async () => {
    const failed = {
      outcome: 'unavailable',
      source: { target: 'live', service: 'none', label: 'none', rehearsal: false },
      caveats: [],
      reason: 'the service failed: x',
    };
    const { container } = await mount({ alarms: failed });
    await vi.waitFor(() =>
      expect(container.querySelector('.alarms-state .osprey-state')?.textContent).toContain(
        'Alarm feed unavailable: the service failed: x'
      )
    );
    expect(container.querySelector('.alarms-state button')?.textContent).toBe('Retry');
  });
});

describe('the poll', () => {
  test('reads every 5 s while mounted, and stops on unmount', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, reads, view, counts } = await mount();
    await loaded(container);
    expect(reads).toHaveLength(1);
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(reads).toHaveLength(2));
    await vi.waitFor(() => expect(counts).toHaveBeenCalledTimes(2));
    view.unmountAlarms();
    unmount = null;
    vi.advanceTimersByTime(view.POLL_MS * 4);
    await Promise.resolve();
    expect(reads).toHaveLength(2);
    expect(document.getElementById('cf-result-count')?.textContent).toBe('');
  });

  test('an identical answer leaves every part, and the focus, as it was', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const { container, view, counts } = await mount();
    await settled(container);
    const major = /** @type {HTMLButtonElement} */ (tile(container, 'MAJOR'));
    major.focus();
    const firstRow = rows(container)[0];
    const inspectorChild = inspector(container).firstElementChild;
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(counts).toHaveBeenCalledTimes(2));
    expect(tile(container, 'MAJOR')).toBe(major);
    expect(rows(container)[0]).toBe(firstRow);
    expect(inspector(container).firstElementChild).toBe(inspectorChild);
    expect(document.activeElement).toBe(major);
  });

  test('an answer with other counts keeps the focus on the same tile', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    let calls = 0;
    const changed = { ...ALARMS, counts: { ...ALARMS.counts, MINOR: 2 } };
    const { container, view, counts } = await mount({
      alarms: () => (calls++ === 0 ? ALARMS : changed),
    });
    await settled(container);
    const major = /** @type {HTMLButtonElement} */ (tile(container, 'MAJOR'));
    major.focus();
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(counts).toHaveBeenCalledTimes(2));
    expect(tileCount(container, 'MINOR')).toBe('2');
    expect(tile(container, 'MAJOR')).not.toBe(major);
    expect(document.activeElement).toBe(tile(container, 'MAJOR'));
  });

  test('the For cell ticks across a poll in the same row', async () => {
    vi.useFakeTimers({
      toFake: ['setInterval', 'clearInterval', 'Date'],
      now: Date.parse(MAJOR_ROW.time) + 30_000,
    });
    const { container, view, counts } = await mount();
    await settled(container);
    const firstRow = rows(container)[0];
    const forCell = () => firstRow.querySelector('.alarms-for')?.textContent;
    expect(forCell()).toBe('30 s');
    expect(inspector(container).textContent).toContain('(30 s ago)');
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(counts).toHaveBeenCalledTimes(2));
    expect(rows(container)[0]).toBe(firstRow);
    expect(forCell()).toBe('35 s');
    expect(inspector(container).textContent).toContain('(35 s ago)');
  });

  test('the feed popover stays open across an identical poll, and closes when the band changes', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    let calls = 0;
    const stale = { ...ALARMS, stale: true };
    const { container, view, counts } = await mount({
      alarms: () => (calls++ < 2 ? ALARMS : stale),
    });
    await settled(container);
    const chip = () =>
      /** @type {HTMLButtonElement} */ (container.querySelector('[data-action="feed"]'));
    chip().click();
    expect(popover()?.hidden).toBe(false);
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(counts).toHaveBeenCalledTimes(2));
    expect(popover()?.hidden).toBe(false);
    chip().click();
    expect(popover()?.hidden).toBe(true);
    chip().click();
    expect(popover()?.hidden).toBe(false);
    vi.advanceTimersByTime(view.POLL_MS);
    await vi.waitFor(() => expect(counts).toHaveBeenCalledTimes(3));
    expect(chip().textContent).toBe('feed stale');
    expect(popover()?.hidden).toBe(true);
  });

  test('a late answer for a view that is gone is dropped', async () => {
    /** @type {(value: any) => void} */
    let release = () => {};
    const pending = new Promise((resolve) => {
      release = resolve;
    });
    const { container, view, counts } = await mount({ alarms: () => pending });
    view.unmountAlarms();
    unmount = null;
    release(ALARMS);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(counts).not.toHaveBeenCalled();
    expect(container.querySelector('.alarms-list tbody')).toBeNull();
  });

  test('Refresh reads the list and the values again', async () => {
    const { container, reads, fetchMock } = await mount();
    await loaded(container);
    const valueReads = () =>
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/directory/values'))
        .length;
    await vi.waitFor(() => expect(valueReads()).toBe(1));
    /** @type {HTMLElement} */ (container.querySelector('[data-action="refresh"]')).click();
    await vi.waitFor(() => expect(reads).toHaveLength(2));
    await vi.waitFor(() => expect(valueReads()).toBe(2));
  });
});

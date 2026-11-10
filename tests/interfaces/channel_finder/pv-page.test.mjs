// @ts-check
/**
 * The PV page (pv-page.js), rendered from the shared route fixture.
 *   npx vitest run tests/interfaces/channel_finder/pv-page.test.mjs
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test, expect, vi, afterEach, describe } from 'vitest';
import {
  mountPvPage,
  unmountPvPage,
} from '../../../src/osprey/interfaces/channel_finder/static/js/pv-page.js';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

/** @type {Record<string, any>} */
const FIXTURE = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);
const NAME = FIXTURE['pv/{name}'].name;
const MOUNT = '/u/a/panel/channel-finder/';
const ARIEL_SEARCH = '/u/a/panel/ariel/api/search';
const DESCRIPTOR = {
  label: 'All PVs',
  configured_services: ['alarm_log', 'alarm_state', 'put_log', 'logbook'],
  canopy: true,
  ariel: true,
};

/** An ARIEL `view: 'row'` answer: RowResponse entries. */
const ROW_ANSWER = {
  entries: [
    {
      entry_id: 'e 1',
      timestamp: '2026-11-04T09:00:00Z',
      author: 'operator',
      source_system: 'demo',
      subject: 'BPM01 drift',
      snippet: 'saw <b>DEMO:BPM01:X</b> drift',
      entry_url: null,
    },
    {
      entry_id: 'e2',
      timestamp: '2026-11-03T09:00:00Z',
      author: 'physicist',
      source_system: 'demo',
      subject: null,
      snippet: 'orbit check on the BPM row after the restart',
      entry_url: null,
    },
  ],
  total: 2,
};

/**
 * @typedef {{
 *   facts?: any,
 *   availability?: any,
 *   values?: any,
 *   ariel?: () => any,
 *   trace?: number,
 * }} Answers
 */

/** @param {any} value */
const clone = (value) => JSON.parse(JSON.stringify(value));

/** @param {any} body @param {number} [status] */
function jsonAnswer(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    type: 'basic',
    redirected: false,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
  };
}

/**
 * Stub fetch with the fixture's answers; `ariel` answers ARIEL's search.
 * @param {Answers} [answers]
 */
function stubRoutes(answers = {}) {
  const fetchMock = vi.fn(async (/** @type {any} */ input, /** @type {any} */ opts = {}) => {
    const url = String(input);
    if (url.startsWith('/api/directory/pv/'))
      return jsonAnswer(answers.facts ?? clone(FIXTURE['pv/{name}']));
    if (url === '/api/directory/logbook/availability')
      return jsonAnswer(answers.availability ?? clone(FIXTURE['logbook/availability']));
    if (url.startsWith('/api/directory/values'))
      return jsonAnswer(answers.values ?? clone(FIXTURE.values));
    if (url === ARIEL_SEARCH) return answers.ariel ? answers.ariel() : jsonAnswer(ROW_ANSWER);
    if (url.endsWith('/panel/canopy/trace/') && opts.method === 'HEAD')
      return { status: answers.trace ?? 200 };
    throw new Error(`unexpected fetch ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/**
 * Mount the page in a frame: `path` is the page's url, `framed` whether its
 * parent is another window.
 * @param {{ path?: string, framed?: boolean, directory?: any, answers?: Answers }} [frame]
 */
async function mountAt({
  path = `${MOUNT}?embedded=true`,
  framed = true,
  directory = DESCRIPTOR,
  answers = {},
} = {}) {
  window.history.replaceState(null, '', path);
  const parent = { postMessage: vi.fn() };
  vi.stubGlobal('parent', framed ? parent : window);
  const fetchMock = stubRoutes(answers);
  vi.resetModules();
  const { state } = await import(`${JS}/state.js`);
  state.directory = directory;
  const page = await import(`${JS}/pv-page.js`);
  const container = document.createElement('div');
  document.body.append(container);
  await page.mountPvPage(container, { name: NAME });
  return { container, fetchMock, parent, page };
}

/** @param {HTMLElement} container @param {string} id */
const section = (container, id) =>
  /** @type {HTMLElement|null} */ (container.querySelector(`[data-section="${id}"]`));

/** @param {any} fetchMock */
const arielCalls = (fetchMock) =>
  fetchMock.mock.calls.filter((/** @type {any[]} */ [url]) =>
    String(url).includes('/panel/ariel/')
  );

/**
 * The fixture's facts with one section replaced (or removed with null).
 * @param {string} id
 * @param {any} replacement
 */
function factsWith(id, replacement) {
  const facts = clone(FIXTURE['pv/{name}']);
  facts.sections = facts.sections
    .map((/** @type {any} */ s) => (s.section === id ? replacement : s))
    .filter(Boolean);
  return facts;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.replaceChildren();
  window.history.replaceState(null, '', '/');
});

test('the view exports what the router mounts and unmounts', () => {
  expect(mountPvPage).toBeTypeOf('function');
  expect(unmountPvPage).toBeTypeOf('function');
});

describe('sections', () => {
  test('every section the facts carry, in the page order, then logbook and value', async () => {
    const { container, page } = await mountAt();
    const order = [...container.querySelectorAll('[data-section]')].map(
      (el) => /** @type {HTMLElement} */ (el).dataset.section
    );
    expect(order).toEqual([
      'directory',
      'alarm_state',
      'alarm_history',
      'alarm_config',
      'put_history',
      'logbook',
      'value',
    ]);
    page.unmountPvPage();
  });

  test('a section the facts do not carry is not shown, nor the logbook without ARIEL', async () => {
    const { container, fetchMock } = await mountAt({
      directory: { ...DESCRIPTOR, ariel: false },
      answers: { facts: factsWith('put_history', null) },
    });
    expect(section(container, 'put_history')).toBeNull();
    expect(section(container, 'logbook')).toBeNull();
    expect(section(container, 'alarm_history')).not.toBeNull();
    expect(fetchMock.mock.calls.map(([url]) => String(url))).not.toContain(
      '/api/directory/logbook/availability'
    );
  });

  test('the header carries the mark and the listing status', async () => {
    const { container } = await mountAt();
    const head = /** @type {HTMLElement} */ (container.querySelector('.pvp-head'));
    expect(head.querySelector('.pvp-name')?.textContent).toBe(NAME);
    expect(head.querySelector('.pvp-mark')?.textContent).toBe('◆');
    const chip = head.querySelector('[data-slot="listing"]');
    expect(chip?.textContent).toBe('Active');
    expect(chip?.classList.contains('osprey-chip--ok')).toBe(true);
    expect(head.querySelector('.pvp-stale')).toBeNull();
  });

  test('detail properties first, the rest behind All properties', async () => {
    const { container } = await mountAt();
    const directory = /** @type {HTMLElement} */ (section(container, 'directory'));
    const first = [...directory.querySelectorAll(':scope > .pvp-facts dt')].map(
      (dt) => dt.textContent
    );
    expect(first.slice(0, 2)).toEqual(['iocName', 'recordType']);
    const more = /** @type {HTMLElement} */ (directory.querySelector('details.pvp-more'));
    expect(more.querySelector('summary')?.textContent).toBe('All properties');
    const rest = [...more.querySelectorAll('dt')].map((dt) => dt.textContent);
    expect(rest).toEqual(['hostName', 'pvStatus', 'recordDesc', 'time']);
  });

  test('a section without an answer says why; one without a source says so', async () => {
    const noTarget = factsWith('alarm_config', {
      section: 'alarm_config',
      outcome: 'unavailable',
      source: { target: 'va', service: 'none', label: 'All PVs', rehearsal: false },
      record: null,
      reason: 'not available for this target',
    });
    const { container } = await mountAt({ answers: { facts: noTarget } });
    expect(section(container, 'alarm_history')?.textContent).toContain(
      'Unavailable: no answer within 6 s'
    );
    expect(section(container, 'alarm_config')?.textContent).toContain(
      'Not available for this target'
    );
  });

  test('alarm configuration with its label', async () => {
    const { container } = await mountAt();
    const config = /** @type {HTMLElement} */ (section(container, 'alarm_config'));
    expect(config.querySelector('h3')?.textContent).toContain('live configuration');
    expect(config.textContent).toContain('/Demo/BPM/DEMO:BPM01:X');
  });
});

describe('alarm state', () => {
  test('live state beside the record’s own alarm from the value read', async () => {
    const { container } = await mountAt();
    const alarm = /** @type {HTMLElement} */ (section(container, 'alarm_state'));
    expect(alarm.textContent).toContain('MINOR HIGH');
    expect(alarm.querySelector('[data-slot="record-alarm"]')?.textContent).toBe('MINOR HIGH');
  });

  test('a stale record reads stale, never no alarm, with a warn chip', async () => {
    const facts = factsWith('alarm_state', {
      ...clone(FIXTURE['pv/{name}'].sections[1]),
      record: {
        severity: 'OK',
        status: 'OK',
        latched: false,
        acknowledged: false,
        message: '',
        time: '2026-11-05T08:10:02.000Z',
        stale: true,
      },
    });
    const { container } = await mountAt({ answers: { facts } });
    const alarm = /** @type {HTMLElement} */ (section(container, 'alarm_state'));
    expect(alarm.textContent).toContain('stale (the alarm feed is not updating)');
    expect(alarm.textContent?.toLowerCase()).not.toContain('no alarm');
    const chip = container.querySelector('.pvp-head .pvp-stale');
    expect(chip?.classList.contains('osprey-chip--warn')).toBe(true);
    expect(chip?.querySelector('svg')).not.toBeNull();
  });

  test('a feed still loading says so and never reads no alarm', async () => {
    const facts = factsWith('alarm_state', {
      section: 'alarm_state',
      outcome: 'unavailable',
      source: FIXTURE['pv/{name}'].sections[1].source,
      record: null,
      reason: 'alarm feed still loading',
    });
    const { container } = await mountAt({ answers: { facts } });
    const alarm = /** @type {HTMLElement} */ (section(container, 'alarm_state'));
    expect(alarm.textContent).toContain('alarm feed still loading');
    expect(alarm.textContent?.toLowerCase()).not.toContain('no alarm');
  });
});

describe('histories', () => {
  test('each row shows the recorded user', async () => {
    const facts = factsWith('alarm_history', {
      section: 'alarm_history',
      outcome: 'found',
      source: FIXTURE['pv/{name}'].sections[3].source,
      rows: [
        {
          time: '2026-11-04T10:00:00Z',
          severity: 'MINOR',
          status: 'HIGH',
          message: 'acknowledged',
          user: 'shift-lead',
          kind: 'state',
        },
      ],
      truncated: false,
      retention_known: true,
      caveats: ['the logger keeps 30 days'],
    });
    const { container } = await mountAt({ answers: { facts } });
    for (const [id, user] of [
      ['put_history', 'operator'],
      ['alarm_history', 'shift-lead'],
    ]) {
      const tableEl = /** @type {HTMLTableElement} */ (
        section(container, id)?.querySelector('table')
      );
      const heads = [...tableEl.querySelectorAll('th')].map((th) => th.textContent);
      const column = heads.indexOf('User');
      expect(column).toBeGreaterThan(-1);
      expect(tableEl.querySelectorAll('tbody td')[column].textContent).toBe(user);
    }
    expect(section(container, 'alarm_history')?.textContent).toContain('the logger keeps 30 days');
  });

  test('a cut history says only the newest rows are shown', async () => {
    const put = clone(FIXTURE['pv/{name}'].sections[4]);
    const truncated = await mountAt({
      answers: { facts: factsWith('put_history', { ...put, truncated: true }) },
    });
    expect(section(truncated.container, 'put_history')?.textContent).toContain(
      'Newest 20 rows shown'
    );
    const whole = await mountAt({
      answers: { facts: factsWith('put_history', { ...put, truncated: false }) },
    });
    expect(section(whole.container, 'put_history')?.textContent).not.toContain('rows shown');
  });
});

describe('logbook', () => {
  test('the one ARIEL request is the pinned POST at the ARIEL mount', async () => {
    const { fetchMock } = await mountAt();
    const calls = arielCalls(fetchMock);
    expect(calls).toHaveLength(1);
    const [url, opts] = calls[0];
    expect(url).toBe(ARIEL_SEARCH);
    expect(opts.method).toBe('POST');
    expect(JSON.parse(opts.body)).toEqual({
      query: `"${NAME}"`,
      mode: 'keyword',
      view: 'row',
      max_results: 10,
    });
  });

  test('rows carry title, time, author and an entry url inside the ARIEL mount', async () => {
    const { container, page } = await mountAt();
    const logbook = await page.readLogbook(NAME);
    expect(logbook.outcome).toBe('found');
    expect(logbook.rows[0]).toMatchObject({
      title: 'BPM01 drift',
      time: '2026-11-04T09:00:00Z',
      author: 'operator',
      excerpt: 'saw DEMO:BPM01:X drift',
      url: '/u/a/panel/ariel/#entry?id=e%201',
    });
    expect(logbook.rows[1].title).toBe('orbit check on the BPM row after the');
    const links = [...container.querySelectorAll('[data-section="logbook"] a')];
    expect(links.map((a) => a.getAttribute('href'))).toEqual([
      '/u/a/panel/ariel/#entry?id=e%201',
      '/u/a/panel/ariel/#entry?id=e2',
    ]);
  });

  test('a row opens through the host, or is an ordinary link without one', async () => {
    const framed = await mountAt();
    const link = /** @type {HTMLAnchorElement} */ (
      framed.container.querySelector('[data-section="logbook"] a')
    );
    const click = new MouseEvent('click', { bubbles: true, cancelable: true });
    link.dispatchEvent(click);
    expect(click.defaultPrevented).toBe(true);
    expect(framed.parent.postMessage).toHaveBeenCalledWith(
      { type: 'osprey:navigate', panel: 'ariel', url: '/u/a/panel/ariel/#entry?id=e%201' },
      window.location.origin
    );

    const popout = await mountAt({ framed: false });
    const plain = /** @type {HTMLAnchorElement} */ (
      popout.container.querySelector('[data-section="logbook"] a')
    );
    const plainClick = new MouseEvent('click', { bubbles: true, cancelable: true });
    plain.addEventListener('click', (event) => event.preventDefault());
    plain.dispatchEvent(plainClick);
    expect(plain.getAttribute('href')).toBe('/u/a/panel/ariel/#entry?id=e%201');
    expect(popout.parent.postMessage).not.toHaveBeenCalled();
  });

  test('a snippet carrying <b> and <script> is rendered as text', async () => {
    const answer = {
      entries: [
        {
          ...ROW_ANSWER.entries[0],
          subject: null,
          snippet: 'before <b>hit</b> <script>alert(1)</script> after',
        },
      ],
    };
    const { container } = await mountAt({ answers: { ariel: () => jsonAnswer(answer) } });
    const logbook = /** @type {HTMLElement} */ (section(container, 'logbook'));
    expect(logbook.querySelector('script')).toBeNull();
    expect(logbook.querySelector('b')).toBeNull();
    expect(logbook.querySelector('.pvp-logbook-excerpt')?.textContent).toBe(
      'before hit <script>alert(1)</script> after'
    );
  });

  test('a login answer reads logbook login required', async () => {
    const answers = [
      () => ({ ...jsonAnswer({}, 401) }),
      () => ({ ...jsonAnswer({}, 403) }),
      () => ({ ...jsonAnswer({}, 0), type: 'opaqueredirect' }),
      () => ({
        ...jsonAnswer({}, 200),
        headers: new Headers({ 'content-type': 'text/html; charset=utf-8' }),
      }),
    ];
    for (const ariel of answers) {
      const { container } = await mountAt({ answers: { ariel } });
      expect(section(container, 'logbook')?.textContent).toContain(
        'Unavailable: logbook login required'
      );
    }
  });

  test('a 503 answer reads logbook not available', async () => {
    const { container } = await mountAt({ answers: { ariel: () => jsonAnswer({}, 503) } });
    expect(section(container, 'logbook')?.textContent).toContain(
      'Unavailable: logbook not available'
    );
  });

  test('on va the logbook is not available for this target and ARIEL is not asked', async () => {
    const { container, fetchMock } = await mountAt({
      answers: {
        availability: { available: false, source: null, reason: 'not available for this target' },
      },
    });
    expect(section(container, 'logbook')?.textContent).toContain('Not available for this target');
    expect(arielCalls(fetchMock)).toHaveLength(0);
  });

  test('standalone, the logbook needs the web terminal and ARIEL is not asked', async () => {
    const { container, fetchMock } = await mountAt({ path: '/', framed: false });
    expect(section(container, 'logbook')?.textContent).toContain(
      'Unavailable: logbook needs the web terminal'
    );
    expect(arielCalls(fetchMock)).toHaveLength(0);
  });
});

describe('hand-off', () => {
  /** @param {HTMLElement} container */
  const buttons = (container) =>
    [...container.querySelectorAll('[data-slot="actions"] button')].map((b) => b.textContent);

  test('in the host, Plot and Send to assistant post to it', async () => {
    const { container, parent } = await mountAt();
    expect(buttons(container)).toEqual(['Plot', 'Send to assistant']);
    const [plot, send] = /** @type {HTMLButtonElement[]} */ ([
      ...container.querySelectorAll('[data-slot="actions"] button'),
    ]);
    plot.click();
    expect(parent.postMessage).toHaveBeenCalledWith(
      {
        type: 'osprey:navigate',
        panel: 'canopy',
        url: '/u/a/panel/canopy/embed/@trace?pv=DEMO%3ABPM01%3AX&range=-1h',
      },
      window.location.origin
    );
    send.click();
    expect(parent.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-paste-to-terminal', text: NAME },
      window.location.origin
    );
  });

  test('no Plot without CANOPY or when Trace does not answer', async () => {
    const noCanopy = await mountAt({ directory: { ...DESCRIPTOR, canopy: false } });
    expect(buttons(noCanopy.container)).toEqual(['Send to assistant']);
    const noTrace = await mountAt({ answers: { trace: 404 } });
    expect(buttons(noTrace.container)).toEqual(['Send to assistant']);
  });

  test('a pop-out shows no Plot and no Send but still reads the logbook', async () => {
    const { container, fetchMock } = await mountAt({ framed: false });
    expect(buttons(container)).toEqual([]);
    expect(arielCalls(fetchMock)).toHaveLength(1);
    expect(container.querySelectorAll('[data-section="logbook"] a')).toHaveLength(2);
  });
});

describe('current value and states', () => {
  test('the current value reads on mount and again on Refresh', async () => {
    const { container, fetchMock } = await mountAt();
    const value = /** @type {HTMLElement} */ (section(container, 'value'));
    expect(value.textContent).toContain('0.125');
    const reads = () =>
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/directory/values'))
        .length;
    expect(reads()).toBe(1);
    /** @type {HTMLButtonElement} */ (value.querySelector('button')).click();
    await vi.waitFor(() => expect(reads()).toBe(2));
  });

  test('a value that does not answer says no answer', async () => {
    const values = [{ ...FIXTURE.values[1], name: NAME }];
    const { container } = await mountAt({ answers: { values } });
    expect(section(container, 'value')?.textContent).toContain('no answer');
  });

  test('a PV nothing knows of is a state with a search of the directory', async () => {
    const facts = {
      name: NAME,
      sections: [
        {
          section: 'directory',
          outcome: 'not_listed',
          source: FIXTURE['pv/{name}'].sections[0].source,
          origin: 'unknown',
          detail_properties: [],
          record: null,
        },
        { ...clone(FIXTURE['pv/{name}'].sections[3]) },
      ],
    };
    const { container } = await mountAt({ answers: { facts } });
    const stateEl = /** @type {HTMLElement} */ (container.querySelector('.osprey-state'));
    expect(stateEl.textContent).toContain(`${NAME} is not listed in All PVs`);
    /** @type {HTMLButtonElement} */ (stateEl.querySelector('button')).click();
    expect(location.hash).toBe(`#/allpvs?q=${encodeURIComponent(NAME)}`);
  });
});

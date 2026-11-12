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

/**
 * The fixture's PV facts section `id`.
 * @param {string} id
 * @returns {any}
 */
const fixtureSection = (id) =>
  FIXTURE['pv/{name}'].sections.find((/** @type {any} */ s) => s.section === id);
const MOUNT = '/u/a/panel/channel-finder/';
const ARIEL_SEARCH = '/u/a/panel/ariel/api/search';
const DESCRIPTOR = {
  label: 'All PVs',
  configured_services: ['alarm_log', 'alarm_state', 'put_log', 'archive', 'logbook'],
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

describe('layout', () => {
  /** @param {HTMLElement} container @param {number} column */
  const cards = (container, column) =>
    [...container.querySelectorAll(`.pvp-grid > .pvp-col:nth-child(${column}) > *`)].map(
      (el) =>
        /** @type {HTMLElement} */ (el).dataset.section ??
        `off:${/** @type {HTMLElement} */ (el).dataset.off}`
    );

  test('breadcrumb, header, banner, then Value, Alarm, Put history | Directory, Archive, Logbook', async () => {
    const { container, page } = await mountAt();
    const top = [...container.querySelectorAll('.pv-page > *')].map((el) => el.className);
    expect(top).toEqual(['pvp-crumb', 'pvp-head', 'pvp-banner pvp-banner--minor', 'pvp-grid']);
    expect(cards(container, 1)).toEqual(['value', 'alarm', 'put_history']);
    expect(cards(container, 2)).toEqual(['directory', 'archive', 'logbook']);
    page.unmountPvPage();
  });

  test('the breadcrumb names All PVs, the IOC and the PV', async () => {
    const { container } = await mountAt();
    const crumb = /** @type {HTMLElement} */ (container.querySelector('.pvp-crumb'));
    expect([...crumb.querySelectorAll('a')].map((a) => a.getAttribute('href'))).toEqual([
      '#/allpvs',
      '#/ioc/demo-ioc-bpm',
    ]);
    expect(crumb.textContent).toBe(`All PVs/demo-ioc-bpm/${NAME}`);
  });

  test('a card whose service is not configured is one dashed line', async () => {
    const { container, fetchMock } = await mountAt({
      directory: { ...DESCRIPTOR, configured_services: [], ariel: false },
      answers: {
        facts: factsWith('put_history', null),
      },
    });
    const off = [...container.querySelectorAll('.pvp-off')].map((el) => el.textContent);
    expect(off).toEqual([
      'Alarms · not configured',
      'Put history · not configured',
      'Archive · not configured',
      'Logbook · not configured',
    ]);
    expect(section(container, 'archive')).toBeNull();
    expect(section(container, 'alarm')).toBeNull();
    expect(section(container, 'put_history')).toBeNull();
    expect(section(container, 'logbook')).toBeNull();
    expect(fetchMock.mock.calls.map(([url]) => String(url))).not.toContain(
      '/api/directory/logbook/availability'
    );
  });

  test('only the unconfigured kinds are dashed', async () => {
    const { container } = await mountAt({
      directory: { ...DESCRIPTOR, configured_services: ['alarm_state', 'logbook'] },
      answers: { facts: factsWith('put_history', null) },
    });
    expect([...container.querySelectorAll('.pvp-off')].map((el) => el.textContent)).toEqual([
      'Put history · not configured',
      'Archive · not configured',
    ]);
    expect(section(container, 'alarm')).not.toBeNull();
    expect(section(container, 'logbook')).not.toBeNull();
  });

  test('a configured section with no source on this target stays a card with its one line', async () => {
    const put = clone(fixtureSection('put_history'));
    const facts = factsWith('put_history', {
      ...put,
      outcome: 'unavailable',
      source: { target: 'va', service: 'none', label: 'All PVs', rehearsal: false },
      rows: [],
      reason: 'not available for this target',
    });
    const { container } = await mountAt({ answers: { facts } });
    const card = /** @type {HTMLElement} */ (section(container, 'put_history'));
    expect([...card.querySelectorAll('.pvp-note')].map((p) => p.textContent)).toEqual([
      'Not available for this target',
    ]);
  });

  test('the header carries the mark, the listing status and the description', async () => {
    const { container } = await mountAt();
    const head = /** @type {HTMLElement} */ (container.querySelector('.pvp-head'));
    expect(head.querySelector('.pvp-name')?.textContent).toBe(NAME);
    expect(head.querySelector('.pvp-mark')?.textContent).toBe('◆');
    const chip = head.querySelector('[data-slot="listing"]');
    expect(chip?.textContent).toBe('Active');
    expect(chip?.classList.contains('osprey-chip--ok')).toBe(true);
    expect(head.querySelector('.pvp-stale')).toBeNull();
    expect(head.querySelector('.pvp-sub')?.textContent).toBe('BPM 01 horizontal position');
  });
});

describe('alarm', () => {
  test('one card: server state beside the record’s own alarm, configuration and its label', async () => {
    const { container } = await mountAt();
    const alarm = /** @type {HTMLElement} */ (section(container, 'alarm'));
    expect(alarm.querySelector('.cf-sev')?.textContent).toBe('MINOR · HIGH');
    expect(alarm.querySelector('[data-slot="record-alarm"]')?.textContent).toBe('MINOR HIGH');
    expect(alarm.querySelector('h3')?.textContent).toContain('live configuration');
    expect(alarm.textContent).toContain('/Demo/BPM/DEMO:BPM01:X');
    expect(alarm.textContent).toContain('Unavailable: no answer within 6 s');
    expect(alarm.textContent).not.toContain('[object Object]');
  });

  test('the banner only when not OK, with the first guidance entry', async () => {
    const { container } = await mountAt();
    const banner = /** @type {HTMLElement} */ (container.querySelector('.pvp-banner'));
    expect(banner.querySelector('.pvp-banner-alarm')?.textContent).toBe('MINOR · HIGH');
    expect(banner.textContent).toContain('latched (record now MINOR HIGH)');
    expect(banner.textContent).toContain('not acknowledged');
    expect(banner.querySelector('.pvp-banner-guidance')?.textContent).toContain('Response.');

    const ok = factsWith('alarm_state', {
      ...clone(fixtureSection('alarm_state')),
      record: { ...fixtureSection('alarm_state').record, severity: 'OK', status: 'NO_ALARM' },
    });
    const clear = await mountAt({ answers: { facts: ok } });
    expect(clear.container.querySelector('.pvp-banner')).toBeNull();
  });

  test('a stale feed shows the chip and reads stale, never no alarm', async () => {
    const facts = factsWith('alarm_state', {
      ...clone(fixtureSection('alarm_state')),
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
    const alarm = /** @type {HTMLElement} */ (section(container, 'alarm'));
    expect(alarm.textContent).toContain('stale (the alarm feed is not updating)');
    expect(alarm.textContent?.toLowerCase()).not.toContain('no alarm');
    const chip = container.querySelector('.pvp-head .pvp-stale');
    expect(chip?.classList.contains('osprey-chip--warn')).toBe(true);
    expect(chip?.querySelector('svg')).not.toBeNull();
  });

  test('History rows only with an alarm-log section', async () => {
    const facts = factsWith('alarm_history', {
      section: 'alarm_history',
      outcome: 'found',
      source: fixtureSection('alarm_history').source,
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
    const history = /** @type {HTMLElement} */ (
      section(container, 'alarm')?.querySelector('[data-part="history"]')
    );
    const heads = [...history.querySelectorAll('th')].map((th) => th.textContent);
    expect(history.querySelectorAll('tbody td')[heads.indexOf('User')].textContent).toBe(
      'shift-lead'
    );
    expect(section(container, 'alarm')?.textContent).toContain('the logger keeps 30 days');

    const without = await mountAt({
      directory: { ...DESCRIPTOR, configured_services: ['alarm_state', 'put_log', 'logbook'] },
      answers: { facts: factsWith('alarm_history', null) },
    });
    const alarm = /** @type {HTMLElement} */ (section(without.container, 'alarm'));
    expect(alarm.querySelector('[data-part="history"]')).toBeNull();
  });
});

describe('put history', () => {
  test('rows show the recorded user; a cut history says only the newest are shown', async () => {
    const put = clone(fixtureSection('put_history'));
    const { container } = await mountAt({
      answers: { facts: factsWith('put_history', { ...put, truncated: true }) },
    });
    const card = /** @type {HTMLElement} */ (section(container, 'put_history'));
    const heads = [...card.querySelectorAll('th')].map((th) => th.textContent);
    expect(card.querySelectorAll('tbody td')[heads.indexOf('User')].textContent).toBe('operator');
    expect(card.textContent).toContain('Newest 20 rows shown');
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
    [...container.querySelectorAll('[data-slot="actions"] button')].map(
      (b) => b.textContent || b.getAttribute('aria-label')
    );

  test('in the host, Copy name, Send to assistant and Plot; Send and Plot post to it', async () => {
    const { container, parent } = await mountAt();
    expect(buttons(container)).toEqual(['Copy name', 'Send to assistant', 'Plot']);
    const [, send, plot] = /** @type {HTMLButtonElement[]} */ ([
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

  test('Copy name copies the name', async () => {
    const writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    const { container } = await mountAt();
    /** @type {HTMLButtonElement} */ (
      container.querySelector('[data-slot="actions"] button[aria-label="Copy name"]')
    ).click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith(NAME));
  });

  test('no Plot without CANOPY or when Trace does not answer', async () => {
    const noCanopy = await mountAt({ directory: { ...DESCRIPTOR, canopy: false } });
    expect(buttons(noCanopy.container)).toEqual(['Copy name', 'Send to assistant']);
    const noTrace = await mountAt({ answers: { trace: 404 } });
    expect(buttons(noTrace.container)).toEqual(['Copy name', 'Send to assistant']);
  });

  test('a pop-out shows no Plot and no Send but still reads the logbook', async () => {
    const { container, fetchMock } = await mountAt({ framed: false });
    expect(buttons(container)).toEqual(['Copy name']);
    expect(arielCalls(fetchMock)).toHaveLength(1);
    expect(container.querySelectorAll('[data-section="logbook"] a')).toHaveLength(2);
  });
});

describe('current value and states', () => {
  test('the current value reads on mount, with its unit, and again on Refresh', async () => {
    const { container, fetchMock } = await mountAt();
    const value = /** @type {HTMLElement} */ (section(container, 'value'));
    expect(value.querySelector('.pvp-bigval')?.textContent).toBe('0.125mm');
    expect(value.querySelector('.pvp-unit')?.textContent).toBe('mm');
    expect(value.textContent).toContain('record MINOR HIGH');
    const reads = () =>
      fetchMock.mock.calls.filter(([url]) => String(url).startsWith('/api/directory/values'))
        .length;
    expect(reads()).toBe(1);
    /** @type {HTMLButtonElement} */ (value.querySelector('button[aria-label="Refresh"]')).click();
    await vi.waitFor(() => expect(reads()).toBe(2));
  });

  test('a value that does not answer says no answer', async () => {
    const values = [{ ...FIXTURE.values[1], name: NAME }];
    const { container } = await mountAt({ answers: { values } });
    expect(section(container, 'value')?.textContent).toContain('no answer');
  });

  /** @param {string} outcome */
  const unlisted = (outcome) => ({
    name: NAME,
    outcome,
    sections: [
      {
        section: 'directory',
        outcome: 'not_listed',
        source: fixtureSection('directory').source,
        caveats: [],
        origin: 'unknown',
        detail_properties: [],
        record: null,
      },
      { ...clone(fixtureSection('put_history')) },
    ],
  });

  test('a PV the facts call unknown is a state with a search of the directory', async () => {
    const { container } = await mountAt({ answers: { facts: unlisted('unknown') } });
    const stateEl = /** @type {HTMLElement} */ (container.querySelector('.osprey-state'));
    expect(stateEl.textContent).toContain(`${NAME} is not listed in All PVs`);
    /** @type {HTMLButtonElement} */ (stateEl.querySelector('button')).click();
    expect(location.hash).toBe(`#/allpvs?q=${encodeURIComponent(NAME)}`);
  });

  test('a PV not listed that another section found shows the page', async () => {
    const { container } = await mountAt({ answers: { facts: unlisted('not_listed') } });
    expect(container.querySelector('.osprey-state')).toBeNull();
    const chip = container.querySelector('.pvp-head [data-slot="listing"]');
    expect(chip?.textContent).toBe('not listed');
    expect(section(container, 'put_history')?.querySelector('tbody td')).not.toBeNull();
    expect(section(container, 'put_history')?.textContent).toContain('operator');
  });
});

describe('the Archive card', () => {
  /** @param {HTMLElement} card */
  const pairs = (card) =>
    Object.fromEntries(
      [...card.querySelectorAll('dt')].map((dt) => [
        dt.textContent,
        dt.nextElementSibling?.textContent,
      ])
    );

  /** @param {any} record */
  const withRecord = (record) =>
    factsWith('archive', { ...clone(fixtureSection('archive')), record });

  test('an archiver that cannot say reads unknown with its reason', async () => {
    const { container } = await mountAt();
    const card = /** @type {HTMLElement} */ (section(container, 'archive'));
    expect(card.querySelector('h3')?.textContent).toBe('Archive');
    expect(pairs(card)).toEqual({
      Archived:
        'unknown (the EPICS Archiver Appliance connector does not ask the appliance whether this channel is archived)',
    });
  });

  test('an archived PV shows its first and last sample, the age and the period', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-11-12T08:00:12Z'));
    const facts = withRecord({
      archived: true,
      first_sample: '2026-10-13T08:00:00+00:00',
      last_sample: '2026-11-12T08:00:04+00:00',
      sampling_period_s: 10,
      description: null,
    });
    const { container } = await mountAt({ answers: { facts } });
    const shown = pairs(/** @type {HTMLElement} */ (section(container, 'archive')));
    vi.useRealTimers();
    expect(shown.Archived).toBe('yes');
    expect(Object.keys(shown)).toEqual([
      'Archived',
      'First sample',
      'Last sample',
      'Sampling period',
    ]);
    expect(shown['Last sample']).toMatch(/ · 8 s ago$/);
    expect(shown['Sampling period']).toBe('10 s');
  });

  test('a PV the archive does not record reads no', async () => {
    const facts = withRecord({
      archived: false,
      first_sample: null,
      last_sample: null,
      sampling_period_s: null,
      description: null,
    });
    const { container } = await mountAt({ answers: { facts } });
    expect(pairs(/** @type {HTMLElement} */ (section(container, 'archive')))).toEqual({
      Archived: 'no',
    });
  });

  test('an archive serving another target says so', async () => {
    const facts = factsWith('archive', {
      section: 'archive',
      outcome: 'unavailable',
      source: { target: 'va', service: 'none', label: 'All PVs', rehearsal: false },
      caveats: [],
      record: null,
      reason: 'not available for this target',
    });
    const { container } = await mountAt({ answers: { facts } });
    const card = /** @type {HTMLElement} */ (section(container, 'archive'));
    expect(card.querySelector('.pvp-note')?.textContent).toBe('Not available for this target');
  });
});

// @ts-check
/**
 * The IOC page (ioc-page.js) over the shared fixture's `ioc/{name}` and
 * `values` answers. directory-api.js is replaced, so the page is judged by the
 * names it hands to readValues and what it shows for the rows answered.
 *   npx vitest run tests/interfaces/channel_finder/ioc-page.test.mjs
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test, expect, vi, beforeEach, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

/** @type {Record<string, any>} */
const FIXTURE = JSON.parse(
  readFileSync(
    resolve(dirname(fileURLToPath(import.meta.url)), 'fixtures/directory_routes.json'),
    'utf8'
  )
);
const LIVE = FIXTURE['ioc/{name}'];
const VALUES = FIXTURE.values;

const VA = {
  ...LIVE,
  ioc: 'sim-SR',
  source: { ...LIVE.source, target: 'va' },
  ports: null,
  engineer: 'ca:SIM:SR:ENGINEER',
  engineer_source: 'ENGINEER PV',
  health_pvs: [
    { name: 'ca:SIM:SR:ENGINEER', role: 'engineer' },
    { name: 'ca:SIM:SR:HEARTBEAT', role: 'heartbeat' },
    { name: 'ca:SIM:SR:UPTIME', role: 'uptime' },
  ],
};

/**
 * A ValueRow for `name`.
 * @param {string} name
 * @param {unknown} value
 */
const found = (name, value) => ({
  name,
  outcome: 'found',
  value,
  timestamp: '2026-11-05T08:12:41.250000+00:00',
  severity: 0,
  status: 'NO_ALARM',
});

/** @param {string} name */
const silent = (name) => ({
  name,
  outcome: 'no_answer',
  value: null,
  timestamp: null,
  severity: null,
  status: null,
  reason: 'no answer within 2 s',
});

const fetchIocFacts = vi.fn();
const readValues = vi.fn();

/** @type {HTMLElement} */
let container;

beforeEach(() => {
  vi.useFakeTimers();
  document.body.innerHTML = '<div id="view-container"></div>';
  container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
  fetchIocFacts.mockReset();
  readValues.mockReset();
  vi.resetModules();
  vi.doMock(`${JS}/directory-api.js`, () => ({ fetchIocFacts, readValues }));
});

let page = /** @type {any} */ (null);

afterEach(() => {
  page?.unmountIocPage();
  page = null;
  vi.doUnmock(`${JS}/directory-api.js`);
  vi.useRealTimers();
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

/**
 * Mount the page for `facts`, with readValues answering each call from
 * `answers` in turn, and let the facts and the first read settle.
 * @param {Record<string, any>} facts
 * @param {Record<string, any>[][]} answers
 */
async function mount(facts, answers) {
  fetchIocFacts.mockResolvedValue(facts);
  for (const rows of answers) readValues.mockResolvedValueOnce(rows);
  page = await import(`${JS}/ioc-page.js`);
  page.mountIocPage(container, { name: facts.ioc });
  await vi.advanceTimersByTimeAsync(0);
}

/** @returns {Record<string, string>} the Network card, label to value. */
function factList() {
  const out = /** @type {Record<string, string>} */ ({});
  for (const dt of container.querySelectorAll('.cf-ioc-facts dt')) {
    out[dt.textContent ?? ''] = dt.nextElementSibling?.textContent ?? '';
  }
  return out;
}

/** @returns {Record<string, [string, string]>} the tiles, role to [value, line under it]. */
function tiles() {
  const out = /** @type {Record<string, [string, string]>} */ ({});
  for (const box of container.querySelectorAll('[data-tile]')) {
    const role = /** @type {HTMLElement} */ (box).dataset.tile ?? '';
    out[role] = [
      box.querySelector('.cf-ioc-tile-value')?.textContent ?? '',
      box.querySelector('.cf-ioc-tile-sub')?.textContent ?? '',
    ];
  }
  return out;
}

/** @returns {string} the engineer in the header line. */
const engineer = () => container.querySelector('.cf-ioc-engineer')?.textContent ?? '';

/** @returns {Record<string, string>} the health PV table, name to value. */
function healthTable() {
  const out = /** @type {Record<string, string>} */ ({});
  for (const tr of container.querySelectorAll('.cf-ioc-health-table tr')) {
    const [name, value] = [...tr.querySelectorAll('td')].map((td) => td.textContent ?? '');
    out[name] = value;
  }
  return out;
}

/** @returns {HTMLButtonElement} */
const chip = () =>
  /** @type {HTMLButtonElement} */ (container.querySelector('[data-role="alive"]'));

test('the view exports what the router mounts and unmounts', async () => {
  page = await import(`${JS}/ioc-page.js`);
  expect(page.mountIocPage).toBeTypeOf('function');
  expect(page.unmountIocPage).toBeTypeOf('function');
});

test('the fixture payload renders its header, tiles, cards and the IOC link', async () => {
  await mount(LIVE, [VALUES]);

  expect(fetchIocFacts).toHaveBeenCalledWith('demo-ioc-bpm');
  expect(readValues).toHaveBeenCalledWith(['DEMO:BPM01:HEARTBEAT']);
  expect(container.querySelector('.cf-ioc-crumb')?.textContent).toBe('All PVs/demo-ioc-bpm');
  expect(container.querySelector('.cf-ioc-name')?.textContent).toBe('demo-ioc-bpm');
  expect(container.querySelector('.cf-ioc-sub')?.textContent).toBe(
    'Host ioc-host-01 · engineer operator'
  );
  expect(tiles()).toEqual({
    pvs: ['2', '1 inactive'],
    curated: ['1', '1 control system only'],
    alarm: ['1', '1 major'],
    heartbeat: ['unknown', 'DEMO:BPM01:HEARTBEAT: second read pending'],
  });
  expect(factList()).toEqual({ Host: 'ioc-host-01', Ports: '5064, 5075' });
  expect(healthTable()).toEqual({ 'DEMO:BPM01:HEARTBEAT': 'no answer' });
  const alarms = /** @type {HTMLElement} */ (container.querySelector('[data-card="alarms"]'));
  const cells = [...alarms.querySelectorAll('td')].map((td) => td.textContent);
  expect(cells.slice(0, 3)).toEqual(['MAJOR', 'DEMO:BPM01:X', 'HIGH_ALARM']);
  expect(alarms.querySelector('a')?.getAttribute('href')).toBe('#/pv/DEMO%3ABPM01%3AX');
  const link = container.querySelector('a.cf-ioc-link');
  expect(link?.getAttribute('href')).toBe('#/allpvs?ioc=demo-ioc-bpm');
  expect(link?.classList).toContain('primary');
});

test('missing ports, an engineer not recorded and an uncountable curated count say so', async () => {
  await mount(
    {
      ...LIVE,
      ports: null,
      engineer: null,
      engineer_source: null,
      curated_count: null,
      curated_reason: 'no curated roster for this target',
    },
    [VALUES]
  );
  expect(factList()).toMatchObject({ Ports: 'not recorded' });
  expect(engineer()).toBe('not recorded');
  expect(tiles().curated).toEqual(['—', 'no curated roster for this target']);
});

describe('alarms', () => {
  test('without an alarm feed there is no In alarm tile and no alarms card', async () => {
    const facts = { ...LIVE };
    delete facts.alarms;
    await mount(facts, [VALUES]);
    expect(tiles().alarm).toBeUndefined();
    expect(container.querySelector('[data-card="alarms"]')).toBeNull();
  });

  test('a feed that did not answer shows its reason, never zero', async () => {
    const reason = 'cannot filter by alarm: alarm feed still loading';
    await mount({ ...LIVE, alarms: { outcome: 'unavailable', reason } }, [VALUES]);
    expect(tiles().alarm).toEqual(['—', reason]);
    expect(container.querySelector('[data-card="alarms"]')?.textContent).toContain(reason);
  });

  test('an IOC with nothing in alarm says so', async () => {
    await mount({ ...LIVE, alarms: { outcome: 'no_match', total: 0, rows: [] } }, [VALUES]);
    expect(tiles().alarm).toEqual(['0', 'none']);
    expect(container.querySelector('[data-card="alarms"]')?.textContent).toContain(
      'No active alarms.'
    );
  });
});

describe('uptime', () => {
  test('a number of seconds reads as a duration, with the value under it', async () => {
    await mount(VA, [
      [
        found('ca:SIM:SR:ENGINEER', 'x'),
        found('ca:SIM:SR:HEARTBEAT', 41),
        { ...found('ca:SIM:SR:UPTIME', 6377.2), units: 's' },
      ],
    ]);
    expect(tiles().uptime).toEqual(['1 h 46 min', '6377.2 s']);
  });

  test('a formatted string reads as read', async () => {
    await mount(VA, [
      [
        found('ca:SIM:SR:ENGINEER', 'x'),
        found('ca:SIM:SR:HEARTBEAT', 41),
        found('ca:SIM:SR:UPTIME', '3 days, 04:12:09'),
      ],
    ]);
    expect(tiles().uptime[0]).toBe('3 days, 04:12:09');
  });

  test('no Uptime tile without an UPTIME health PV', async () => {
    await mount(LIVE, [VALUES]);
    expect(tiles().uptime).toBeUndefined();
  });
});

test('on the VA the engineer line shows the ENGINEER PV value from the one value read', async () => {
  const rows = [
    found('ca:SIM:SR:ENGINEER', 'operator on shift'),
    found('ca:SIM:SR:HEARTBEAT', 41),
    found('ca:SIM:SR:UPTIME', 3600),
  ];
  await mount(VA, [rows, [found('ca:SIM:SR:HEARTBEAT', 43)]]);

  expect(readValues).toHaveBeenCalledTimes(1);
  expect(readValues).toHaveBeenCalledWith(VA.health_pvs.map((/** @type {any} */ pv) => pv.name));
  expect(factList()).toMatchObject({ Ports: 'not recorded' });
  expect(engineer()).toBe('operator on shift');
  expect(healthTable()).toEqual({
    'ca:SIM:SR:ENGINEER': 'operator on shift',
    'ca:SIM:SR:HEARTBEAT': '41',
    'ca:SIM:SR:UPTIME': '3600',
  });
});

test('a silent ENGINEER PV reads no answer, never the PV name', async () => {
  await mount(VA, [
    [silent('ca:SIM:SR:ENGINEER'), found('ca:SIM:SR:HEARTBEAT', 1), found('ca:SIM:SR:UPTIME', 1)],
  ]);
  expect(engineer()).toBe('no answer');
});

test('the Heartbeat tile follows aliveness, with the two reads under it', async () => {
  await mount(VA, [
    [
      found('ca:SIM:SR:ENGINEER', 'x'),
      found('ca:SIM:SR:HEARTBEAT', 41),
      found('ca:SIM:SR:UPTIME', 1),
    ],
    [found('ca:SIM:SR:HEARTBEAT', 43)],
  ]);
  expect(tiles().heartbeat[0]).toBe('unknown');
  await vi.advanceTimersByTimeAsync(2000);
  expect(tiles().heartbeat).toEqual(['advancing', '41 → 43 in 2 s']);
  expect(chip().dataset.alive).toBe('yes');

  readValues
    .mockResolvedValueOnce([found('ca:SIM:SR:HEARTBEAT', 43)])
    .mockResolvedValueOnce([found('ca:SIM:SR:HEARTBEAT', 43)]);
  /** @type {HTMLButtonElement} */ (container.querySelector('button.cf-ioc-refresh')).click();
  await vi.advanceTimersByTimeAsync(2000);
  expect(tiles().heartbeat).toEqual(['not advancing', '43 → 43 in 2 s']);
});

describe('alive', () => {
  test('a HEARTBEAT that advances in 2 s is yes', async () => {
    await mount(VA, [
      [
        found('ca:SIM:SR:ENGINEER', 'x'),
        found('ca:SIM:SR:HEARTBEAT', 41),
        found('ca:SIM:SR:UPTIME', 1),
      ],
      [found('ca:SIM:SR:HEARTBEAT', 43)],
    ]);
    expect(chip().dataset.alive).toBe('unknown');
    expect(chip().classList).toContain('osprey-chip--info');

    await vi.advanceTimersByTimeAsync(1999);
    expect(readValues).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    expect(readValues).toHaveBeenLastCalledWith(['ca:SIM:SR:HEARTBEAT']);
    expect(chip().dataset.alive).toBe('yes');
    expect(chip().textContent).toBe('alive: yes');
    expect(chip().classList).toContain('osprey-chip--ok');
    expect(healthTable()['ca:SIM:SR:HEARTBEAT']).toBe('43');
  });

  test('a frozen HEARTBEAT is no', async () => {
    await mount(VA, [
      [
        found('ca:SIM:SR:ENGINEER', 'x'),
        found('ca:SIM:SR:HEARTBEAT', 41),
        found('ca:SIM:SR:UPTIME', 1),
      ],
      [found('ca:SIM:SR:HEARTBEAT', 41)],
    ]);
    await vi.advanceTimersByTimeAsync(2000);
    expect(chip().dataset.alive).toBe('no');
    expect(chip().classList).toContain('osprey-chip--err');
  });

  test('a HEARTBEAT with no answer is no', async () => {
    await mount(LIVE, [VALUES, [silent('DEMO:BPM01:HEARTBEAT')]]);
    await vi.advanceTimersByTimeAsync(2000);
    expect(chip().dataset.alive).toBe('no');
  });

  test('before the second read it is unknown, and the chip opens the reason', async () => {
    await mount(VA, [
      [
        found('ca:SIM:SR:ENGINEER', 'x'),
        found('ca:SIM:SR:HEARTBEAT', 41),
        found('ca:SIM:SR:UPTIME', 1),
      ],
    ]);
    expect(chip().dataset.alive).toBe('unknown');
    chip().click();
    const popover = document.querySelector('.osprey-info-popover');
    expect(popover?.textContent).toContain('ca:SIM:SR:HEARTBEAT: second read pending');
  });

  test('an IOC with no health PVs listed is unknown and shows the reason', async () => {
    await mount(
      { ...LIVE, health_pvs: [], health_pvs_reason: 'no IOC health PVs listed for this IOC' },
      []
    );
    expect(readValues).not.toHaveBeenCalled();
    expect(container.querySelector('.cf-ioc-reason')?.textContent).toBe(
      'no IOC health PVs listed for this IOC'
    );
    await vi.advanceTimersByTimeAsync(2000);
    expect(chip().dataset.alive).toBe('unknown');
  });

  test('health PVs without a heartbeat role are unknown, whatever their names end in', async () => {
    const pvs = [
      { name: 'ca:SIM:SR:UPTIME', role: 'uptime' },
      { name: 'ca:SIM:SR:X:HEARTBEAT', role: 'uptime' },
    ];
    await mount({ ...VA, health_pvs: pvs }, [
      [found('ca:SIM:SR:UPTIME', 5), found('ca:SIM:SR:X:HEARTBEAT', 7)],
    ]);
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(1);
    expect(chip().dataset.alive).toBe('unknown');
  });

  test('Refresh reads the values and the two HEARTBEAT reads again', async () => {
    const first = [
      found('ca:SIM:SR:ENGINEER', 'x'),
      found('ca:SIM:SR:HEARTBEAT', 41),
      found('ca:SIM:SR:UPTIME', 1),
    ];
    await mount(VA, [first, [found('ca:SIM:SR:HEARTBEAT', 41)]]);
    await vi.advanceTimersByTimeAsync(2000);
    expect(chip().dataset.alive).toBe('no');

    readValues
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce([found('ca:SIM:SR:HEARTBEAT', 42)]);
    /** @type {HTMLButtonElement} */ (container.querySelector('button.cf-ioc-refresh')).click();
    await vi.advanceTimersByTimeAsync(0);
    expect(chip().dataset.alive).toBe('unknown');
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(4);
    expect(chip().dataset.alive).toBe('yes');
  });

  test('Refresh drops the answers of reads still in flight', async () => {
    const rows = [
      found('ca:SIM:SR:ENGINEER', 'x'),
      found('ca:SIM:SR:HEARTBEAT', 41),
      found('ca:SIM:SR:UPTIME', 1),
    ];
    /** @type {(rows: any[]) => void} */
    let answerOldFirst = () => {};
    /** @type {(rows: any[]) => void} */
    let answerOldSecond = () => {};
    const refresh = () =>
      /** @type {HTMLButtonElement} */ (container.querySelector('button.cf-ioc-refresh')).click();

    await mount(VA, [rows]);
    readValues.mockImplementationOnce(() => new Promise((done) => (answerOldSecond = done)));
    await vi.advanceTimersByTimeAsync(2000);
    readValues.mockResolvedValueOnce(rows);
    refresh();
    await vi.advanceTimersByTimeAsync(0);
    answerOldSecond([found('ca:SIM:SR:HEARTBEAT', 41)]);
    await vi.advanceTimersByTimeAsync(0);
    expect(chip().dataset.alive).toBe('unknown');

    readValues.mockImplementationOnce(() => new Promise((done) => (answerOldFirst = done)));
    refresh();
    readValues
      .mockResolvedValueOnce(rows)
      .mockResolvedValueOnce([found('ca:SIM:SR:HEARTBEAT', 45)]);
    refresh();
    await vi.advanceTimersByTimeAsync(0);
    answerOldFirst(rows);
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(6);
    expect(chip().dataset.alive).toBe('yes');
  });

  test('unmount cancels the second read', async () => {
    await mount(LIVE, [VALUES]);
    page.unmountIocPage();
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(1);
  });
});

test('an IOC not in the directory and a directory without an answer show a state', async () => {
  await mount({ ...LIVE, outcome: 'not_listed' }, []);
  expect(container.querySelector('.osprey-state-message')?.textContent).toBe(
    'IOC demo-ioc-bpm is not listed in the directory'
  );
  page.unmountIocPage();

  fetchIocFacts.mockResolvedValue({
    ...LIVE,
    outcome: 'unavailable',
    reason: 'timed out after 5 s',
  });
  page.mountIocPage(container, { name: 'demo-ioc-bpm' });
  await vi.advanceTimersByTimeAsync(0);
  expect(container.querySelector('.osprey-state-detail')?.textContent).toBe('timed out after 5 s');
});

test('facts that could not be read show a state whose Retry reads them again', async () => {
  fetchIocFacts.mockRejectedValueOnce(new Error('HTTP 502'));
  page = await import(`${JS}/ioc-page.js`);
  page.mountIocPage(container, { name: 'demo-ioc-bpm' });
  await vi.advanceTimersByTimeAsync(0);
  expect(container.querySelector('.osprey-state-message')?.textContent).toBe(
    'The IOC facts could not be read'
  );
  const retry = /** @type {HTMLButtonElement} */ (container.querySelector('button'));
  expect(retry.textContent).toBe('Retry');

  fetchIocFacts.mockResolvedValue(LIVE);
  readValues.mockResolvedValueOnce(VALUES);
  retry.click();
  await vi.advanceTimersByTimeAsync(0);
  expect(fetchIocFacts).toHaveBeenCalledTimes(2);
  expect(container.querySelector('.cf-ioc-name')?.textContent).toBe(LIVE.ioc);
  expect(container.querySelector('button.cf-ioc-refresh .osprey-btn-label')?.textContent).toBe(
    'Refresh'
  );
});

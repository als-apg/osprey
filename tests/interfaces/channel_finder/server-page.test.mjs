// @ts-check
/**
 * The IOC page (server-page.js) over the shared fixture's `ioc/{name}` and
 * `values` answers. directory-api.js is replaced, so the page is judged by the
 * names it hands to readValues and what it shows for the rows answered.
 *   npx vitest run tests/interfaces/channel_finder/server-page.test.mjs
 */

import { useEpicsWords } from './epics-words.mjs';
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
  server: 'sim-SR',
  source: { ...LIVE.source, target: 'va' },
  ports: null,
  engineer: 'ca:SIM:SR:ENGINEER',
  engineer_source: 'ENGINEER PV',
  health_channels: [
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

const fetchServerFacts = vi.fn();
const readValues = vi.fn();

/** @type {HTMLElement} */
let container;

beforeEach(async () => {
  vi.useFakeTimers();
  document.body.innerHTML = '<div id="view-container"></div>';
  container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
  fetchServerFacts.mockReset();
  readValues.mockReset();
  vi.resetModules();
  vi.doMock(`${JS}/directory-api.js`, () => ({ fetchServerFacts, readValues }));
  await useEpicsWords();
});

let page = /** @type {any} */ (null);

afterEach(() => {
  page?.unmountServerPage();
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
  fetchServerFacts.mockResolvedValue(facts);
  for (const rows of answers) readValues.mockResolvedValueOnce(rows);
  page = await import(`${JS}/server-page.js`);
  page.mountServerPage(container, { name: facts.server });
  await vi.advanceTimersByTimeAsync(0);
}

/** @returns {Record<string, string>} the Network card, label to value. */
function factList() {
  const out = /** @type {Record<string, string>} */ ({});
  for (const dt of container.querySelectorAll('.cf-server-facts dt')) {
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
      box.querySelector('.cf-server-tile-value')?.textContent ?? '',
      box.querySelector('.cf-server-tile-sub')?.textContent ?? '',
    ];
  }
  return out;
}

/** @returns {string} the engineer in the header line. */
const engineer = () => container.querySelector('.cf-server-engineer')?.textContent ?? '';

/** @returns {Record<string, string>} the health PV table, name to value. */
function healthTable() {
  const out = /** @type {Record<string, string>} */ ({});
  for (const tr of container.querySelectorAll('.cf-server-health-table tr')) {
    const [name, value] = [...tr.querySelectorAll('td')].map((td) => td.textContent ?? '');
    out[name] = value;
  }
  return out;
}

/** @returns {HTMLButtonElement} */
const chip = () =>
  /** @type {HTMLButtonElement} */ (container.querySelector('[data-role="alive"]'));

test('the view exports what the router mounts and unmounts', async () => {
  page = await import(`${JS}/server-page.js`);
  expect(page.mountServerPage).toBeTypeOf('function');
  expect(page.unmountServerPage).toBeTypeOf('function');
});

test('the fixture payload renders its header, tiles, cards and the IOC link', async () => {
  await mount(LIVE, [VALUES]);

  expect(fetchServerFacts).toHaveBeenCalledWith('demo-ioc-bpm');
  expect(readValues).toHaveBeenCalledWith(['DEMO:BPM01:HEARTBEAT']);
  expect(container.querySelector('.cf-server-crumb')?.textContent).toBe('All PVs/demo-ioc-bpm');
  expect(container.querySelector('.cf-server-name')?.textContent).toBe('demo-ioc-bpm');
  expect(container.querySelector('.cf-server-sub')?.textContent).toBe(
    'Host ioc-host-01 · engineer operator'
  );
  expect(tiles()).toEqual({
    channels: ['2', '1 inactive'],
    curated: ['1', '1 control system only'],
    alarm: ['1', '1 major'],
    heartbeat: ['unknown', 'DEMO:BPM01:HEARTBEAT: second read pending'],
  });
  expect(factList()).toEqual({ Host: 'ioc-host-01', Ports: '5064, 5075' });
  expect(healthTable()).toEqual({ 'DEMO:BPM01:HEARTBEAT': 'no answer' });
  const alarms = /** @type {HTMLElement} */ (container.querySelector('[data-card="alarms"]'));
  const cells = [...alarms.querySelectorAll('td')].map((td) => td.textContent);
  expect(cells.slice(0, 3)).toEqual(['MAJOR', 'DEMO:BPM01:X', 'HIGH_ALARM']);
  expect(alarms.querySelector('a')?.getAttribute('href')).toBe('#/channel/DEMO%3ABPM01%3AX');
  const link = container.querySelector('a.cf-server-link');
  expect(link?.getAttribute('href')).toBe('#/channels?server=demo-ioc-bpm');
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

  test('the major line reads the counts of the whole IOC, not the rows shown', async () => {
    const alarms = { ...LIVE.alarms, total: 60, counts: { MAJOR: 12, MINOR: 48 } };
    await mount({ ...LIVE, alarms }, [VALUES]);
    expect(tiles().alarm).toEqual(['60', '12 major']);
  });

  test('without counts there is no major line', async () => {
    const alarms = { ...LIVE.alarms };
    delete alarms.counts;
    await mount({ ...LIVE, alarms }, [VALUES]);
    expect(tiles().alarm).toEqual(['1', '']);
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
  expect(readValues).toHaveBeenCalledWith(
    VA.health_channels.map((/** @type {any} */ pv) => pv.name)
  );
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
  /** @type {HTMLButtonElement} */ (container.querySelector('button.cf-server-refresh')).click();
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
      {
        ...LIVE,
        health_channels: [],
        health_channels_reason: 'no IOC health PVs listed for this IOC',
      },
      []
    );
    expect(readValues).not.toHaveBeenCalled();
    expect(container.querySelector('.cf-server-reason')?.textContent).toBe(
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
    await mount({ ...VA, health_channels: pvs }, [
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
    /** @type {HTMLButtonElement} */ (container.querySelector('button.cf-server-refresh')).click();
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
      /** @type {HTMLButtonElement} */ (
        container.querySelector('button.cf-server-refresh')
      ).click();

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
    page.unmountServerPage();
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(1);
  });
});

test('an IOC not in the directory and a directory without an answer show a state', async () => {
  await mount({ ...LIVE, outcome: 'not_listed' }, []);
  expect(container.querySelector('.osprey-state-message')?.textContent).toBe(
    'IOC demo-ioc-bpm is not listed in the directory'
  );
  page.unmountServerPage();

  fetchServerFacts.mockResolvedValue({
    ...LIVE,
    outcome: 'unavailable',
    reason: 'timed out after 5 s',
  });
  page.mountServerPage(container, { name: 'demo-ioc-bpm' });
  await vi.advanceTimersByTimeAsync(0);
  expect(container.querySelector('.osprey-state-detail')?.textContent).toBe('timed out after 5 s');
});

test('facts that could not be read show a state whose Retry reads them again', async () => {
  fetchServerFacts.mockRejectedValueOnce(new Error('HTTP 502'));
  page = await import(`${JS}/server-page.js`);
  page.mountServerPage(container, { name: 'demo-ioc-bpm' });
  await vi.advanceTimersByTimeAsync(0);
  expect(container.querySelector('.osprey-state-message')?.textContent).toBe(
    'The IOC facts could not be read'
  );
  const retry = /** @type {HTMLButtonElement} */ (container.querySelector('button'));
  expect(retry.textContent).toBe('Retry');

  fetchServerFacts.mockResolvedValue(LIVE);
  readValues.mockResolvedValueOnce(VALUES);
  retry.click();
  await vi.advanceTimersByTimeAsync(0);
  expect(fetchServerFacts).toHaveBeenCalledTimes(2);
  expect(container.querySelector('.cf-server-name')?.textContent).toBe(LIVE.server);
  expect(container.querySelector('button.cf-server-refresh .osprey-btn-label')?.textContent).toBe(
    'Refresh'
  );
});

describe('fleet', () => {
  const SNAPSHOT = FIXTURE.fleet.snapshot;
  const VA_ROWS = [
    found('ca:SIM:SR:ENGINEER', 'operator on shift'),
    { ...found('ca:SIM:SR:UPTIME', 3600), units: 's' },
  ];
  /** The VA facts' IOC row in the fixture snapshot, as the facts carry it. */
  const ROW = { ...SNAPSHOT.servers[1], name: 'sim-SR' };

  /**
   * The `fleet` field of IOC facts.
   * @param {Record<string, any>} fields
   */
  const entry = (fields) => ({
    outcome: 'ok',
    target: 'va',
    reason: null,
    generated_at: SNAPSHOT.generated_at,
    row: null,
    ...fields,
  });

  /** @param {number} epoch */
  const at = async (epoch) => {
    const { timeText } = await import(`${JS}/channel-page-sections.js`);
    return timeText(new Date(epoch * 1000).toISOString());
  };

  test('a row shows its state, findings and read time, and the heartbeat is not read', async () => {
    await mount({ ...VA, fleet: entry({ row: ROW }) }, [VA_ROWS]);
    await vi.advanceTimersByTimeAsync(2000);

    expect(readValues).toHaveBeenCalledTimes(1);
    expect(readValues).toHaveBeenCalledWith(['ca:SIM:SR:ENGINEER', 'ca:SIM:SR:UPTIME']);
    expect(tiles().heartbeat).toBeUndefined();
    expect(tiles().state).toEqual(['alive', `read ${await at(ROW.liveness.read_at)}`]);
    expect(chip().textContent).toBe('state: alive');
    expect(chip().dataset.state).toBe('alive');
    expect(chip().classList).toContain('osprey-chip--ok');
    expect(tiles().uptime).toEqual(['1 h', '3600 s']);
    expect(engineer()).toBe('operator on shift');
    expect(healthTable()['ca:SIM:SR:HEARTBEAT']).toBe('read by the server fleet service');

    chip().click();
    const popover = document.querySelector('.osprey-info-popover');
    expect(popover?.textContent).toContain('counter moved');
  });

  test('a row with findings names them, and a quiet state is not ok', async () => {
    const row = { ...SNAPSHOT.servers[0], name: 'sim-SR' };
    await mount({ ...VA, fleet: entry({ row }) }, [VA_ROWS]);
    expect(tiles().state).toEqual([
      'silent',
      `expected running, not alive · read ${await at(row.liveness.read_at)}`,
    ]);
    expect(chip().classList).toContain('osprey-chip--err');
  });

  test('a row without a liveness read gives the snapshot time', async () => {
    const row = { ...ROW, state: 'unknown', reason: 'no liveness PV', liveness: null };
    await mount({ ...VA, fleet: entry({ row }) }, [VA_ROWS]);
    expect(tiles().state).toEqual(['unknown', `read ${await at(SNAPSHOT.generated_at)}`]);
    expect(chip().classList).toContain('osprey-chip--info');
  });

  const OUTCOMES = /** @type {[string, string, string][]} */ ([
    ['no_snapshot', 'no snapshot', 'has no snapshot of target va yet'],
    ['not_refreshed', 'not refreshed', 'does not refresh target va'],
    ['cannot_list', 'cannot list', 'cannot list the IOCs of target va'],
    ['unavailable', 'unavailable', 'gave no usable answer for target va'],
    ['stale', 'stale', 'The snapshot of target va is stale'],
    ['not_in_snapshot', 'not in snapshot', 'sim-SR is not in the snapshot of target va'],
  ]);

  test.each(OUTCOMES)(
    'the %s outcome is a statement, never a state',
    async (outcome, word, says) => {
      const reason = `reason for ${outcome}`;
      await mount({ ...VA, fleet: entry({ outcome, reason }) }, [VA_ROWS]);
      await vi.advanceTimersByTimeAsync(2000);

      expect(readValues).toHaveBeenCalledTimes(1);
      expect(readValues).toHaveBeenCalledWith(['ca:SIM:SR:ENGINEER', 'ca:SIM:SR:UPTIME']);
      const [value, sub] = tiles().state;
      expect(value).toBe(word);
      expect(sub).toContain(says);
      expect(sub).toContain('This says nothing about whether this IOC runs.');
      expect(chip().textContent).toBe('state: not known');
      expect(chip().dataset.outcome).toBe(outcome);
      expect(chip().classList).toContain('osprey-chip--info');
      chip().click();
      expect(document.querySelector('.osprey-info-popover')?.textContent).toContain(reason);
      expect(tiles().uptime[0]).toBe('1 h');
    }
  );

  test('a stale outcome gives the time of the last snapshot', async () => {
    await mount({ ...VA, fleet: entry({ outcome: 'stale', reason: 'old' }) }, [VA_ROWS]);
    expect(tiles().state[1]).toContain(`Last snapshot ${await at(SNAPSHOT.generated_at)}.`);
  });

  test('Refresh reads the other values again and never the heartbeat', async () => {
    await mount({ ...VA, fleet: entry({ row: ROW }) }, [VA_ROWS]);
    readValues.mockResolvedValueOnce(VA_ROWS);
    /** @type {HTMLButtonElement} */ (container.querySelector('button.cf-server-refresh')).click();
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(2);
    for (const [names] of readValues.mock.calls) {
      expect(names).not.toContain('ca:SIM:SR:HEARTBEAT');
    }
    expect(chip().dataset.state).toBe('alive');
  });

  test('without the fleet field the page reads the heartbeat twice as before', async () => {
    expect('fleet' in VA).toBe(false);
    await mount(VA, [
      [
        found('ca:SIM:SR:ENGINEER', 'x'),
        found('ca:SIM:SR:HEARTBEAT', 1),
        found('ca:SIM:SR:UPTIME', 1),
      ],
      [found('ca:SIM:SR:HEARTBEAT', 2)],
    ]);
    await vi.advanceTimersByTimeAsync(2000);
    expect(readValues).toHaveBeenCalledTimes(2);
    expect(tiles().state).toBeUndefined();
    expect(tiles().heartbeat).toEqual(['advancing', '1 → 2 in 2 s']);
    expect(chip().textContent).toBe('alive: yes');
  });
});

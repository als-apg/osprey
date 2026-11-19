/**
 * Unit tests for the runs list (runs-list.js): the rows it shows for a given
 * state, the selection a click sends, and its times: a stored run's start as
 * the service host wrote it, an open job's on the facility clock, and today
 * and yesterday on the facility's calendar.
 */

import { afterEach, describe, test, expect, vi } from 'vitest';

import { FACILITY_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';
import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import {
  createRunsList,
  dayLabel,
  pastRuns,
  runTime,
  runsModel,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/runs-list.js';
import {
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent */

/** Noon, 17 Nov 2026, local time. */
const NOW = new Date(2026, 10, 17, 12, 0, 0).getTime();

/** @param {StoreEvent[]} events */
const run = (events) => events.reduce(reduce, initialState());

/** @param {string} timestamp @param {Record<string, unknown>} [overrides] */
const stored = (timestamp, overrides = {}) => ({
  run_id: `injector/${timestamp}`,
  environment: 'injector',
  timestamp,
  objective_name: 'charge',
  has_model: false,
  job_id: `job-${timestamp}`,
  ...overrides,
});

const RESULTS = [
  stored('2026-11-16_21-13-01'),
  stored('2026-11-17_09-15-00'),
  stored('2026-11-14_17-44-10', { objective_name: 'loss' }),
];

/** @type {StoreEvent[]} */
const LISTS = [
  { type: 'poll/job', job: null, at: 0 },
  { type: 'environments', list: [{ name: 'injector', display_name: 'Injector' }], error: null },
  { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
  { type: 'results', list: RESULTS, error: null },
];

describe('runTime and dayLabel', () => {
  test("a stored run's directory name gives its day and clock time", () => {
    expect(runTime('2026-11-16_21-15-04')).toEqual({ day: '2026-11-16', clock: '21:15' });
    expect(runTime('2026-11-16_21-15-04-ab12')).toEqual({ day: '2026-11-16', clock: '21:15' });
    expect(runTime('legacy')).toBeNull();
  });

  test('today, yesterday, a date this year, a date of another year', () => {
    expect(dayLabel('2026-11-17', NOW)).toBe('Today');
    expect(dayLabel('2026-11-16', NOW)).toBe('Yesterday');
    expect(dayLabel('2026-11-14', NOW)).toBe('14 Nov');
    expect(dayLabel('2025-11-14', NOW)).toBe('14 Nov 2025');
  });
});

describe('the times of the runs list on the facility clock', () => {
  afterEach(() => stampFacilityZone(null));

  test("today and yesterday are the facility's days", () => {
    // 15:30 UTC on 19 Sep is 00:30 on the 20th in Tokyo and 11:30 on the 19th in New York:
    // at least one of the two is not the day of the zone the tests run in.
    const now = Date.parse('2026-09-19T15:30:00Z');
    stampFacilityZone('Asia/Tokyo');
    expect(['2026-09-20', '2026-09-19', '2026-09-18'].map((d) => dayLabel(d, now))).toEqual([
      'Today',
      'Yesterday',
      '18 Sep',
    ]);
    stampFacilityZone('America/New_York');
    expect(['2026-09-20', '2026-09-19', '2026-09-18'].map((d) => dayLabel(d, now))).toEqual([
      '20 Sep',
      'Today',
      'Yesterday',
    ]);
  });

  test("a date carries its year when it is not the facility's year now", () => {
    // 15:30 UTC on 31 Dec 2026 is already 1 Jan 2027 in Tokyo.
    const now = Date.parse('2026-12-31T15:30:00Z');
    stampFacilityZone('Asia/Tokyo');
    expect(dayLabel('2026-12-31', now)).toBe('Yesterday');
    expect(dayLabel('2026-12-30', now)).toBe('30 Dec 2026');
    stampFacilityZone('America/New_York');
    expect(dayLabel('2026-12-30', now)).toBe('Yesterday');
    expect(dayLabel('2026-12-28', now)).toBe('28 Dec');
  });

  test('a stored run shows its start as the service host wrote it, under the day heading of that date', () => {
    stampFacilityZone('Asia/Tokyo');
    // 00:30 on 20 Sep in Tokyo.
    const now = Date.parse('2026-09-19T15:30:00Z');
    const state = run([
      ...LISTS.slice(0, 3),
      { type: 'results', list: [stored('2026-09-19_23-40-00')], error: null },
    ]);
    const model = runsModel(state, now);
    expect(model.past.map((g) => [g.label, g.rows.map((r) => r.time)])).toEqual([
      ['Yesterday', ['23:40']],
    ]);
  });

  test('an open job shows when it was made on the facility clock, with its date when that is not today', () => {
    stampFacilityZone(FACILITY_ZONE);
    // Both instants fall on one calendar day in every candidate facility zone.
    const now = Date.parse('2026-09-19T21:30:00Z');
    /** @param {Intl.DateTimeFormatOptions} options @param {string} iso */
    const facility = (options, iso) =>
      new Intl.DateTimeFormat(undefined, { ...options, timeZone: FACILITY_ZONE }).format(
        new Date(iso)
      );
    /** @type {Intl.DateTimeFormatOptions} */
    const clock = { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' };
    const state = run([
      ...LISTS,
      {
        type: 'jobs',
        list: [
          { job_id: 'j5', status: 'running', created_at: '2026-09-19T21:05:00+00:00' },
          { job_id: 'j6', status: 'created', created_at: '2026-09-17T21:05:00+00:00' },
          { job_id: 'j7', status: 'created', created_at: null },
        ],
        error: null,
      },
    ]);
    expect(runsModel(state, now).open.map((r) => r.time)).toEqual([
      facility(clock, '2026-09-19T21:05:00Z'),
      facility({ day: 'numeric', month: 'short', ...clock }, '2026-09-17T21:05:00Z'),
      null,
    ]);
  });
});

describe('runsModel', () => {
  test('stored runs newest first, by day, named by the objective the service describes', () => {
    const model = runsModel(run(LISTS), NOW);
    expect(model.past.map((g) => g.label)).toEqual(['Today', 'Yesterday', '14 Nov']);
    const rows = model.past.flatMap((g) => g.rows);
    expect(rows.map((r) => [r.title, r.meta, r.time])).toEqual([
      ['Charge', 'Injector', '09:15'],
      ['Charge', 'Injector', '21:13'],
      ['loss', 'Injector', '17:44'],
    ]);
    expect(rows.every((r) => r.dot === null && r.chip === null)).toBe(true);
    expect(model).toMatchObject({ total: 3, shown: 3, note: null });
  });

  test("a stored run's status shows once the service answered for its job", () => {
    const runId = RESULTS[1].run_id;
    const model = runsModel(
      run([
        ...LISTS,
        {
          type: 'past/loaded',
          runId,
          details: null,
          job: { job_id: 'x', status: 'error' },
          error: null,
        },
      ]),
      NOW
    );
    expect(model.past[0].rows[0].dot).toEqual({ tone: 'err', label: 'Failed' });
  });

  test('the filter narrows the stored runs and the count says so', () => {
    const model = runsModel(run([...LISTS, { type: 'filter', text: 'LOSS' }]), NOW);
    expect(model.past.flatMap((g) => g.rows).map((r) => r.title)).toEqual(['loss']);
    expect(model).toMatchObject({ total: 3, shown: 1 });
    expect(runsModel(run([...LISTS, { type: 'filter', text: 'nothing' }]), NOW).note).toBe(
      'No past run matches.'
    );
  });

  test('before the list answers it says it is reading; a failed list says why', () => {
    expect(runsModel(initialState(), NOW).note).toBe('Reading past runs…');
    const failed = run([{ type: 'results', list: null, error: 'down' }]);
    expect(runsModel(failed, NOW).note).toBe('Past runs could not be read: down');
  });

  test('no job of now: what the panel knows about the service', () => {
    expect(runsModel(initialState(), NOW)).toMatchObject({
      now: [],
      nowNote: 'Reading the optimizer…',
    });
    expect(runsModel(run(LISTS), NOW).nowNote).toBe('No draft, nothing running.');
    expect(runsModel(run([{ type: 'poll/failed', error: 'x' }]), NOW).nowNote).toBe(
      'The optimizer could not be read.'
    );
  });

  test('a draft of now: ready, with its objective and its knobs', () => {
    const model = runsModel(
      run([
        ...LISTS,
        { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 1 },
        { type: 'poll/config', jobId: 'j1', revision: 'r', config: storedConfig() },
      ]),
      NOW
    );
    expect(model.now).toHaveLength(1);
    expect(model.now[0]).toMatchObject({
      title: 'Draft',
      chip: { label: 'ready' },
      meta: 'Charge · 2 knobs',
      selected: true,
      selection: { kind: 'present' },
    });
  });

  test('a run of now: its status and its progress', () => {
    const model = runsModel(
      run([...LISTS, { type: 'poll/job', job: runningJob({ job_id: 'j1' }), at: 1 }]),
      NOW
    );
    expect(model.now[0]).toMatchObject({
      title: 'Run',
      chip: { label: 'running' },
      aside: '3 of 10',
    });
  });

  test('a new run being set up is a row of now', () => {
    const model = runsModel(
      run([...LISTS, { type: 'setup/open', jobId: null, environment: 'injector', target: 'va' }]),
      NOW
    );
    expect(model.now).toHaveLength(1);
    expect(model.now[0]).toMatchObject({
      key: 'setup',
      chip: { label: 'not saved' },
      meta: 'Injector',
      selected: true,
    });
  });

  test('beside an open set-up, the current run is its own row and picks the run itself', () => {
    const model = runsModel(
      run([
        ...LISTS,
        { type: 'setup/open', jobId: null, environment: 'injector', target: 'va' },
        { type: 'poll/job', job: runningJob({ job_id: 'j7' }), at: 1 },
      ]),
      NOW
    );
    expect(model.now.map((r) => [r.key, r.selected])).toEqual([
      ['setup', true],
      ['job:j7', false],
    ]);
    expect(model.now[1].selection).toEqual({ kind: 'job', jobId: 'j7' });
  });

  test('other open jobs are listed; the current one is not listed twice', () => {
    const model = runsModel(
      run([
        ...LISTS,
        { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 1 },
        {
          type: 'jobs',
          list: [
            { job_id: 'j1', status: 'created' },
            { job_id: 'j5-abcdefgh', status: 'running', created_at: null },
          ],
          error: null,
        },
        { type: 'select', selection: { kind: 'job', jobId: 'j5-abcdefgh' } },
      ]),
      NOW
    );
    expect(model.open.map((r) => [r.title, r.meta, r.selected])).toEqual([
      ['Run', 'running · j5-abcde', true],
    ]);
    expect(model.now[0].selected).toBe(false);
  });

  test('a run active beside a newer draft comes first and is "now"; the draft is picked by its id', () => {
    const model = runsModel(
      run([
        ...LISTS,
        { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 1 },
        { type: 'poll/config', jobId: 'j1', revision: 'r', config: storedConfig() },
        {
          type: 'jobs',
          list: [
            { job_id: 'j1', status: 'created' },
            { job_id: 'r9', status: 'running' },
          ],
          error: null,
        },
        { type: 'poll/active', job: runningJob({ job_id: 'r9' }), config: storedConfig(), at: 2 },
      ]),
      NOW
    );
    expect(model.now.map((r) => [r.key, r.chip?.label, r.selected, r.selection])).toEqual([
      ['job:r9', 'running', true, { kind: 'present' }],
      ['job:j1', 'ready', false, { kind: 'job', jobId: 'j1' }],
    ]);
    expect(model.open).toEqual([]);
  });
});

describe('pastRuns', () => {
  test('newest first, narrowed by the filter as the runs list is', () => {
    expect(pastRuns(run(LISTS)).map((r) => r.timestamp)).toEqual([
      '2026-11-17_09-15-00',
      '2026-11-16_21-13-01',
      '2026-11-14_17-44-10',
    ]);
    // The display name of the objective matches as well as its identifier.
    const state = run([...LISTS, { type: 'filter', text: 'charge' }]);
    expect(pastRuns(state).map((r) => r.objective_name)).toEqual(['charge', 'charge']);
    expect(runsModel(state, NOW).shown).toBe(pastRuns(state).length);
    expect(pastRuns(run([...LISTS, { type: 'filter', text: 'injector' }]))).toHaveLength(3);
    expect(pastRuns(initialState())).toEqual([]);
  });
});

describe('createRunsList', () => {
  function mount() {
    const host = document.createElement('div');
    document.body.replaceChildren(host);
    const actions = /** @type {any} */ ({ select: vi.fn(), newRun: vi.fn() });
    const onPicked = vi.fn();
    const list = createRunsList({ host, actions, onPicked, now: () => NOW });
    return { host, actions, onPicked, list };
  }

  test('a click on a stored run selects it and closes an overlay', () => {
    const { host, actions, onPicked, list } = mount();
    list.update(run(LISTS));
    /** @type {HTMLElement} */ (
      host.querySelector(`[data-row="run:${RESULTS[0].run_id}"]`)
    ).click();
    expect(actions.select).toHaveBeenCalledWith({ kind: 'run', runId: RESULTS[0].run_id });
    expect(onPicked).toHaveBeenCalledTimes(1);
  });

  test('the table button selects the history; with nothing now, New run opens a set-up', () => {
    const { host, actions, list } = mount();
    list.update(run(LISTS));
    /** @type {HTMLElement} */ (
      host.querySelector('[aria-label="All past runs in a table"]')
    ).click();
    expect(actions.select).toHaveBeenCalledWith({ kind: 'history' });
    /** @type {HTMLElement} */ (host.querySelector('.osprey-nav-row.add')).click();
    expect(actions.newRun).toHaveBeenCalledTimes(1);
  });

  test('a selected row is marked; an unchanged list is not redrawn; focus survives a redraw', () => {
    const { host, list } = mount();
    const state = run([
      ...LISTS,
      { type: 'select', selection: { kind: 'run', runId: RESULTS[1].run_id } },
    ]);
    list.update(state);
    const row = /** @type {HTMLElement} */ (
      host.querySelector(`[data-row="run:${RESULTS[1].run_id}"]`)
    );
    expect(row.getAttribute('aria-selected')).toBe('true');
    list.update(state);
    expect(host.querySelector(`[data-row="run:${RESULTS[1].run_id}"]`)).toBe(row);
    row.focus();
    list.update(reduce(state, { type: 'filter', text: 'charge' }));
    expect(host.querySelectorAll('[data-row^="run:"]')).toHaveLength(2);
    expect(host.querySelector(`[data-row="run:${RESULTS[1].run_id}"]`)).toBe(row);
    expect(document.activeElement).toBe(row);
  });

  test('a row that changes keeps its element, so a click across an update still selects', () => {
    const { host, actions, list } = mount();
    const job = runningJob({ job_id: 'j1' });
    const state = run([...LISTS, { type: 'poll/job', job, at: 1 }]);
    list.update(state);
    const row = /** @type {HTMLElement} */ (host.querySelector('[data-row="job:j1"]'));
    // Where a mousedown before the update lands.
    const aside = /** @type {HTMLElement} */ (row.querySelector('.opt-run-aside'));
    const before = aside.textContent;

    // One more evaluation arrives between mousedown and mouseup.
    const more = { ...job, data: [...job.data, { phase_a: 1, sol_b: 3, charge: 3.0 }] };
    const next = reduce(state, { type: 'poll/job', job: more, at: 2 });
    list.update(next);
    expect(row.isConnected).toBe(true);
    expect(host.querySelector('[data-row="job:j1"]')).toBe(row);
    expect(row.querySelector('.opt-run-aside')).toBe(aside);
    expect(aside.isConnected).toBe(true);
    expect(aside.textContent).not.toBe(before);
    expect(aside.textContent).toContain('4');
    aside.click();
    expect(actions.select).toHaveBeenCalledWith({ kind: 'present' });

    // A row that is no longer listed is removed; the others keep their element.
    const gone = host.querySelector(`[data-row="run:${RESULTS[0].run_id}"]`);
    const kept = host.querySelector(`[data-row="run:${RESULTS[1].run_id}"]`);
    list.update(reduce(next, { type: 'results', list: RESULTS.slice(1), error: null }));
    expect(gone?.isConnected).toBe(false);
    expect(host.querySelector(`[data-row="run:${RESULTS[0].run_id}"]`)).toBeNull();
    expect(host.querySelector(`[data-row="run:${RESULTS[1].run_id}"]`)).toBe(kept);
    expect(row.isConnected).toBe(true);
  });
});

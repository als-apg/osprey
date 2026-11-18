// @ts-check
/**
 * The PV page's card builders (pv-page-sections.js), from the shared fixture.
 *   npx vitest run tests/interfaces/channel_finder/pv-page-sections.test.mjs
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test, expect, describe } from 'vitest';
import {
  alarmBanner,
  alarmCard,
  archiveCard,
  directoryCard,
  logbookCard,
  putHistoryCard,
  recordAlarmText,
  unavailableLine,
  unconfiguredLine,
  valueCard,
} from '../../../src/osprey/interfaces/channel_finder/static/js/pv-page-sections.js';

/** @type {Record<string, any>} */
const FIXTURE = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);
/** @param {any} value */
const clone = (value) => JSON.parse(JSON.stringify(value));
/** @param {string} id @returns {any} */
const section = (id) =>
  clone(FIXTURE['pv/{name}'].sections.find((/** @type {any} */ s) => s.section === id));

const SOURCE = { target: 'live', service: 'put_log', label: 'put_log', rehearsal: false };
const RETENTION = 'Retention unknown: how far back the service keeps rows is not known';

/** @param {HTMLElement} frame */
const footer = (frame) =>
  [...frame.querySelectorAll('footer.pvp-footer p')].map((line) => line.textContent);

test('an answer without a source reads not available for this target', () => {
  expect(unavailableLine('not available for this target').textContent).toBe(
    'Not available for this target'
  );
  expect(unavailableLine('no answer within 6 s').textContent).toBe(
    'Unavailable: no answer within 6 s'
  );
});

test('the record alarm reads its severity through severity.js', () => {
  expect(recordAlarmText(/** @type {any} */ ({ severity: 1, status: 'HIGH' }))).toBe(
    'MINOR · HIGH'
  );
  expect(recordAlarmText(/** @type {any} */ ({ severity: 2, status: 'HIHI' }))).toBe(
    'MAJOR · HIHI'
  );
  expect(recordAlarmText(/** @type {any} */ ({ severity: 0, status: 'OK' }))).toBe('OK');
  expect(recordAlarmText(/** @type {any} */ ({ severity: 7, status: '' }))).toBe('UNDEFINED');
  expect(recordAlarmText(/** @type {any} */ ({ severity: null, status: 'HIGH' }))).toBe('HIGH');
});

test('an unconfigured card is one dashed line', () => {
  const line = unconfiguredLine('Put history');
  expect(line.textContent).toBe('Put history · not configured');
  expect(line.className).toBe('pvp-off');
  expect(line.dataset.off).toBe('Put history');
});

test('the value card waits for the read and its Refresh calls back', () => {
  let refreshed = 0;
  const frame = valueCard(() => refreshed++);
  expect(frame.dataset.section).toBe('value');
  expect(frame.classList).toContain('osprey-card');
  expect(frame.querySelector('[data-slot="value"]')?.textContent).toBe('Reading…');
  /** @type {HTMLButtonElement} */ (frame.querySelector('button[aria-label="Refresh"]')).click();
  expect(refreshed).toBe(1);
});

describe('directory card', () => {
  test('IOC, host, record type and origin, then detail properties and the rest', () => {
    const frame = directoryCard(section('directory'));
    const [main, detail] = [...frame.querySelectorAll(':scope > .pvp-facts')];
    const pairs = [...main.querySelectorAll('dt')].map((dt) => [
      dt.textContent,
      dt.nextElementSibling?.textContent,
    ]);
    expect(pairs).toEqual([
      ['IOC', 'demo-ioc-bpm'],
      ['Host', 'ioc-host-01'],
      ['Record type', 'ai'],
      ['Origin', 'Curated'],
    ]);
    expect(main.querySelector('a')?.getAttribute('href')).toBe('#/ioc/demo-ioc-bpm');
    expect([...detail.querySelectorAll('dt')].map((dt) => dt.textContent)).toEqual([
      'iocName',
      'recordType',
    ]);
    const more = /** @type {HTMLElement} */ (frame.querySelector('details.pvp-more'));
    expect(more.querySelector('summary')?.textContent).toBe('All properties');
    expect([...more.querySelectorAll('dt')].map((dt) => dt.textContent)).toEqual([
      'hostName',
      'pvStatus',
      'recordDesc',
      'time',
    ]);
    expect(frame.textContent).toContain('bpm');
  });

  test('a directory without an answer says why, with its caveats', () => {
    const frame = directoryCard({
      ...section('directory'),
      outcome: 'unavailable',
      record: null,
      reason: 'no answer within 6 s',
      caveats: ['covers the SR tree only'],
    });
    expect(frame.textContent).toContain('Unavailable: no answer within 6 s');
    expect(footer(frame)).toEqual(['covers the SR tree only']);
  });
});

describe('alarm card', () => {
  const sections = () => ({
    state: section('alarm_state'),
    config: section('alarm_config'),
    history: section('alarm_history'),
  });

  test('state, configuration and history in one card, the label beside the title', () => {
    const frame = alarmCard(sections());
    expect(frame.dataset.section).toBe('alarm');
    expect(frame.querySelector('h3 .pvp-card-aside')?.textContent).toBe('live configuration');
    expect(frame.querySelector('.cf-sev--minor')?.textContent).toBe('MINOR · HIGH');
    expect(frame.querySelector('[data-slot="record-alarm"]')).not.toBeNull();
    expect(frame.textContent).toContain('/Demo/BPM/DEMO:BPM01:X');
    const history = /** @type {HTMLElement} */ (frame.querySelector('[data-part="history"]'));
    expect(history.textContent).toContain('Unavailable: no answer within 6 s');
  });

  test('guidance and displays render as title and details, never [object Object]', () => {
    const frame = alarmCard(sections());
    const text = frame.textContent ?? '';
    expect(text).not.toContain('[object Object]');
    const guidance = /** @type {HTMLElement} */ (frame.querySelector('.pvp-guidance'));
    expect(guidance.querySelector('.pvp-entry-title')?.textContent).toBe('Response');
    expect(guidance.querySelector('.pvp-entry-details')?.textContent).toBe(
      section('alarm_config').record.guidance[0].details
    );
    const displays = [...frame.querySelectorAll('dt')].find((dt) => dt.textContent === 'Displays');
    expect(displays?.nextElementSibling?.textContent).toBe('RF cavity 1rf/cavity_01.bob');
  });

  test('no History without an alarm history section', () => {
    const { state, config } = sections();
    expect(alarmCard({ state, config }).querySelector('[data-part="history"]')).toBeNull();
  });

  test('a PV the feed does not hold is not in the alarm tree', () => {
    const frame = alarmCard({
      ...sections(),
      state: { ...section('alarm_state'), outcome: 'not_listed', record: null },
    });
    expect(frame.querySelector('.cf-sev--none')?.textContent).toBe('Not in the alarm tree');
  });

  test('a feed that did not answer leaves the configuration and history standing', () => {
    const state = {
      ...section('alarm_state'),
      outcome: 'unavailable',
      record: null,
      reason: 'alarm feed still loading',
    };
    const frame = alarmCard({ ...sections(), state });
    expect(frame.textContent).toContain('Unavailable: alarm feed still loading');
    expect(frame.textContent?.toLowerCase()).not.toContain('no alarm');
    expect(frame.textContent).toContain('/Demo/BPM/DEMO:BPM01:X');
  });

  test('a stale state reads stale, never no alarm', () => {
    const state = section('alarm_state');
    state.record = { ...state.record, severity: 'OK', status: 'OK', stale: true };
    const frame = alarmCard({ ...sections(), state });
    expect(frame.textContent).toContain('stale (the alarm feed is not updating)');
    expect(frame.textContent?.toLowerCase()).not.toContain('no alarm');
  });

  test('on a target with no alarm source, one line', () => {
    /** @param {any} s */
    const none = (s) => ({ ...s, outcome: 'unavailable', reason: 'not available for this target' });
    const { state, config, history } = sections();
    const frame = alarmCard({ state: none(state), config: none(config), history: none(history) });
    expect([...frame.querySelectorAll('.pvp-note')].map((p) => p.textContent)).toEqual([
      'Not available for this target',
    ]);
  });

  test('every section shows its caveats once in the footer', () => {
    const caveats = ['covers the SR tree only'];
    const { state, config } = sections();
    const frame = alarmCard({ state: { ...state, caveats }, config: { ...config, caveats } });
    expect(footer(frame)).toEqual(caveats);
  });
});

describe('archive card', () => {
  test('the age of the last sample is counted from now, the caveats in the footer', () => {
    const archive = section('archive');
    archive.caveats = ['The appliance records on change.'];
    archive.record = {
      archived: true,
      first_sample: null,
      last_sample: '2026-11-12T08:00:00+00:00',
      sampling_period_s: null,
      description: null,
    };
    const frame = archiveCard(archive, Date.parse('2026-11-12T08:02:00Z'));
    const dd = [...frame.querySelectorAll('dd')].map((cell) => cell.textContent);
    expect(dd[0]).toBe('yes');
    expect(dd[1]).toMatch(/ · 2 min ago$/);
    expect(footer(frame)).toEqual(['The appliance records on change.']);
  });

  test('an archive without an answer says why', () => {
    const frame = archiveCard({
      ...section('archive'),
      outcome: 'unavailable',
      record: null,
      reason: 'store down',
    });
    expect(frame.querySelector('.pvp-note')?.textContent).toBe('Unavailable: store down');
  });
});

describe('put history card', () => {
  /** @param {any} fields */
  const history = (fields) =>
    putHistoryCard({
      section: 'put_history',
      outcome: 'found',
      source: /** @type {any} */ (SOURCE),
      rows: [],
      truncated: false,
      caveats: [],
      ...fields,
    });
  const row = { at: '2026-11-04T14:20:00.000Z', pv: 'A', old: '1', new: '2' };

  test('the fixture rows as a table with the recorded user', () => {
    const frame = putHistoryCard(section('put_history'));
    const heads = [...frame.querySelectorAll('th')].map((th) => th.textContent);
    expect(frame.querySelectorAll('tbody td')[heads.indexOf('User')].textContent).toBe('operator');
  });

  test('cells are text and an empty history says so', () => {
    const frame = history({ rows: [{ ...row, field: '<i>VAL</i>' }] });
    expect(frame.querySelector('i')).toBeNull();
    expect(frame.querySelector('tbody td:nth-child(2)')?.textContent).toBe('<i>VAL</i>');
    expect(history({}).textContent).toContain('No rows recorded');
  });

  test('a cut history and an unknown retention say so', () => {
    expect(footer(history({ rows: [row], truncated: true }))).toEqual(['Newest 20 rows shown']);
    expect(footer(history({ rows: [row], retention_known: false }))).toEqual([RETENTION]);
    expect(history({ rows: [row], retention_known: true }).querySelector('footer')).toBeNull();
    const down = history({ outcome: 'unavailable', reason: 'no answer within 6 s' });
    expect(down.querySelector('footer')).toBeNull();
    expect(down.textContent).toContain('Unavailable: no answer within 6 s');
  });
});

test('the logbook card waits for the page to fill it', () => {
  const frame = logbookCard();
  expect(frame.dataset.section).toBe('logbook');
  expect(frame.querySelector('[data-slot="logbook"]')?.textContent).toBe('Reading…');
});

describe('alarm banner', () => {
  test('a PV not OK: severity and alarm, since, latched with the record now, the first guidance', () => {
    const state = section('alarm_state');
    state.record = {
      ...state.record,
      severity: 'MAJOR',
      status: 'NO_ALARM',
      message: 'HIHI_ALARM',
    };
    const banner = /** @type {HTMLElement} */ (
      alarmBanner(state, section('alarm_config'), '38 min')
    );
    expect(banner.classList).toContain('pvp-banner--major');
    expect(banner.querySelector('.pvp-banner-alarm')?.textContent).toBe('MAJOR · HIHI_ALARM');
    const rest = banner.querySelector('.pvp-banner-rest')?.textContent ?? '';
    expect(rest).toMatch(
      /^for 38 min, since .+ · latched \(record now NO_ALARM\) · not acknowledged$/
    );
    expect(banner.querySelector('[data-slot="record-alarm"]')?.textContent).toBe('NO_ALARM');
    expect(banner.querySelector('.pvp-banner-guidance')?.textContent).toBe(
      `Response. ${section('alarm_config').record.guidance[0].details}`
    );
  });

  test('no banner when OK, without a state or when the feed did not answer', () => {
    const state = section('alarm_state');
    expect(
      alarmBanner({ ...state, record: { ...state.record, severity: 'OK' } }, undefined, '')
    ).toBeNull();
    expect(alarmBanner(undefined, undefined, '')).toBeNull();
    expect(
      alarmBanner({ ...state, outcome: 'not_listed', record: null }, undefined, '')
    ).toBeNull();
    expect(
      alarmBanner({ ...state, outcome: 'unavailable', record: null, reason: 'x' }, undefined, '')
    ).toBeNull();
  });
});

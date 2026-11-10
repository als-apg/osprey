// @ts-check
/**
 * The PV page's section renderers (pv-page-sections.js).
 *   npx vitest run tests/interfaces/channel_finder/pv-page-sections.test.mjs
 */

import { test, expect } from 'vitest';
import {
  SECTION_ORDER,
  factsSection,
  severityText,
  unavailableLine,
  valueText,
} from '../../../src/osprey/interfaces/channel_finder/static/js/pv-page-sections.js';

const SOURCE = { target: 'live', service: 'put_log', label: 'put_log', rehearsal: false };

test('the page order of the facts sections', () => {
  expect(SECTION_ORDER).toEqual([
    'directory',
    'alarm_state',
    'alarm_history',
    'alarm_config',
    'put_history',
  ]);
});

test('an answer without a source reads not available for this target', () => {
  expect(unavailableLine('not available for this target').textContent).toBe(
    'Not available for this target'
  );
  expect(unavailableLine('no answer within 6 s').textContent).toBe(
    'Unavailable: no answer within 6 s'
  );
});

test('severities and values as text', () => {
  expect(severityText(0)).toBe('NO_ALARM');
  expect(severityText(2)).toBe('MAJOR');
  expect(severityText('MINOR')).toBe('MINOR');
  expect(severityText(null)).toBe('');
  expect(valueText([1, 2])).toBe('[1,2]');
  expect(valueText(null)).toBe('');
});

test('a history renders its cells as text and an empty one says so', () => {
  const rows = factsSection({
    section: 'put_history',
    outcome: 'found',
    source: /** @type {any} */ (SOURCE),
    rows: [
      {
        at: '2026-11-04T14:20:00.000Z',
        pv: 'A',
        field: '<i>VAL</i>',
        old: '1',
        new: '2',
        user: 'operator',
        client: 'host',
      },
    ],
    truncated: false,
    caveats: [],
  });
  expect(rows.querySelector('i')).toBeNull();
  expect(rows.querySelector('tbody td:nth-child(2)')?.textContent).toBe('<i>VAL</i>');

  const empty = factsSection({
    section: 'alarm_history',
    outcome: 'found',
    source: /** @type {any} */ (SOURCE),
    rows: [],
    truncated: false,
    caveats: [],
  });
  expect(empty.textContent).toContain('No rows recorded');
});

const FEED = { target: 'live', service: 'kafka', label: 'kafka', rehearsal: false };

/** @param {HTMLElement} frame */
const footer = (frame) =>
  [...frame.querySelectorAll('footer.pvp-footer p')].map((line) => line.textContent);

test('every section shows its caveats in its footer', () => {
  const caveats = ['covers the SR tree only'];
  const sections = [
    {
      section: 'directory',
      outcome: 'not_listed',
      source: /** @type {any} */ ({ ...SOURCE, label: 'All PVs' }),
      record: null,
      caveats,
    },
    {
      section: 'alarm_state',
      outcome: 'found',
      source: /** @type {any} */ (FEED),
      record: /** @type {any} */ ({ severity: 'MINOR', status: 'HIGH', stale: false }),
      caveats,
    },
    {
      section: 'alarm_config',
      outcome: 'unavailable',
      source: /** @type {any} */ (FEED),
      record: null,
      reason: 'alarm feed still loading',
      caveats,
    },
  ];
  for (const payload of sections) {
    const frame = factsSection(/** @type {any} */ (payload));
    expect(footer(frame), payload.section).toEqual(caveats);
  }
  const bare = factsSection(/** @type {any} */ ({ ...sections[1], caveats: [] }));
  expect(bare.querySelector('footer')).toBeNull();
});

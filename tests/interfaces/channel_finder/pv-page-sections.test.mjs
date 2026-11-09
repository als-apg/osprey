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

// @ts-check
/**
 * Guidance and displays as title + details (alarm-text.js configEntries).
 *   npx vitest run tests/interfaces/channel_finder/alarm-text.test.mjs
 */

import { test, expect, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { configEntries } from '../../../src/osprey/interfaces/channel_finder/static/js/alarm-text.js';

/** The configuration records of the demo alarm replay. */
const REPLAY = readFileSync(
  join(import.meta.dirname, '../../../src/osprey/services/pv_directory/demo/example-alarms.ndjson'),
  'utf8'
)
  .split('\n')
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .filter((entry) => entry.key.startsWith('config:'));

describe('configEntries', () => {
  test('the demo replay guidance renders its title and details', () => {
    expect(REPLAY.length).toBeGreaterThan(0);
    for (const { value } of REPLAY) {
      const entries = configEntries(value.guidance);
      expect(entries.length).toBe(value.guidance.length);
      expect(entries[0].title).toBe('Response');
      expect(entries[0].details).toBe(value.guidance[0].details);
      for (const entry of entries) {
        expect(`${entry.title} ${entry.details}`).not.toContain('[object Object]');
      }
    }
  });

  test('the demo replay displays render their title and file', () => {
    const entries = configEntries(REPLAY[0].value.displays);
    expect(entries).toEqual([{ title: 'RF cavity 1', details: 'rf/cavity_01.bob' }]);
  });

  test('a string is one entry with that text as details', () => {
    expect(configEntries('Call the RF on-call.')).toEqual([
      { title: '', details: 'Call the RF on-call.' },
    ]);
  });

  test.each([
    [42, '42'],
    [true, 'true'],
    [{ a: 1 }, '{"a":1}'],
  ])('%j is one entry with its JSON as details', (value, details) => {
    expect(configEntries(value)).toEqual([{ title: '', details }]);
  });

  test('null and undefined give none', () => {
    expect(configEntries(null)).toEqual([]);
    expect(configEntries(undefined)).toEqual([]);
  });

  test('a list entry that is not a mapping, or a field that is not text, is text', () => {
    expect(configEntries(['plain', 7, { title: 'T', details: { x: 1 } }, { title: null }])).toEqual(
      [
        { title: '', details: 'plain' },
        { title: '', details: '7' },
        { title: 'T', details: '{"x":1}' },
        { title: '', details: '' },
      ]
    );
  });
});

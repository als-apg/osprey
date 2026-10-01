// @ts-check
/**
 * ARIEL entry times read in the facility zone.
 *
 * The API sends each entry time as a facility-local ISO string with its
 * offset, the same wall time ARIEL's agent tools quote. The panel learns the
 * zone from `/api/capabilities`, stamps it on `<html>`, and both entry
 * formatters render through the shared facility-time formatter: the stamped
 * zone decides the wall time and the viewer's locale decides the date
 * convention. Without a stamp the time reads in the viewer's zone, named.
 *
 *   npx vitest run tests/interfaces/ariel/entry-times.test.mjs
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { test, expect, describe, beforeEach } from 'vitest';

import {
  applyFacilityTimezone,
  formatTimestamp,
  formatHumanTimestamp,
  formatRelativeTime,
} from '../../../src/osprey/interfaces/ariel/static/js/components.js';

/** An entry time as the API and the MCP tool send it (Asia/Tokyo facility). */
const ENTRY_AT = '2026-06-01T09:00:00+09:00';

/** @type {Intl.DateTimeFormatOptions} */
const DISPLAY = {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
};

/** @type {Intl.DateTimeFormatOptions} */
const SIMPLE = {
  weekday: 'long',
  month: 'long',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
};

/**
 * @param {string} value
 * @param {Intl.DateTimeFormatOptions} options
 * @param {string} timeZone
 */
function intl(value, options, timeZone) {
  return new Intl.DateTimeFormat(undefined, { ...options, timeZone }).format(new Date(value));
}

beforeEach(() => {
  document.documentElement.removeAttribute('data-facility-timezone');
});

describe('applyFacilityTimezone', () => {
  test('stamps the zone the capabilities name', () => {
    applyFacilityTimezone({ facility_timezone: 'Asia/Tokyo' });
    expect(document.documentElement.getAttribute('data-facility-timezone')).toBe('Asia/Tokyo');
  });

  test('leaves the page unstamped without a zone', () => {
    for (const capabilities of [null, undefined, {}, { facility_timezone: '' }]) {
      applyFacilityTimezone(capabilities);
      expect(document.documentElement.hasAttribute('data-facility-timezone')).toBe(false);
    }
  });
});

describe('entry times in the facility zone', () => {
  test('formatTimestamp reads the wire time in the stamped zone', () => {
    applyFacilityTimezone({ facility_timezone: 'Asia/Tokyo' });
    const expected = intl(ENTRY_AT, DISPLAY, 'Asia/Tokyo');
    expect(formatTimestamp(ENTRY_AT)).toBe(expected);
    const hour = new Intl.DateTimeFormat(undefined, { ...DISPLAY, timeZone: 'Asia/Tokyo' })
      .formatToParts(new Date(ENTRY_AT))
      .find((p) => p.type === 'hour');
    expect(hour?.value).toBe('09');
  });

  test('formatHumanTimestamp reads the wire time in the stamped zone', () => {
    applyFacilityTimezone({ facility_timezone: 'Asia/Tokyo' });
    expect(formatHumanTimestamp(ENTRY_AT)).toBe(intl(ENTRY_AT, SIMPLE, 'Asia/Tokyo'));
  });

  test('the stamp decides the wall time, not the viewer', () => {
    applyFacilityTimezone({ facility_timezone: 'Asia/Tokyo' });
    const tokyo = formatTimestamp(ENTRY_AT);
    applyFacilityTimezone({ facility_timezone: 'America/Los_Angeles' });
    const losAngeles = formatTimestamp(ENTRY_AT);
    expect(losAngeles).toBe(intl(ENTRY_AT, DISPLAY, 'America/Los_Angeles'));
    expect(losAngeles).not.toBe(tokyo);
  });

  test('without a stamp the time names the viewer zone', () => {
    const viewer = new Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(formatTimestamp(ENTRY_AT)).toBe(
      intl(ENTRY_AT, { ...DISPLAY, timeZoneName: 'short' }, viewer),
    );
    expect(formatHumanTimestamp(ENTRY_AT)).toBe(
      intl(ENTRY_AT, { ...SIMPLE, timeZoneName: 'short' }, viewer),
    );
  });

  test('an unparsable timestamp is shown as sent', () => {
    expect(formatTimestamp('not-a-time')).toBe('not-a-time');
    expect(formatHumanTimestamp('not-a-time')).toBe('not-a-time');
    expect(formatTimestamp('')).toBe('');
    expect(formatHumanTimestamp('')).toBe('');
    expect(formatTimestamp(undefined)).toBe('');
    expect(formatHumanTimestamp(undefined)).toBe('');
  });

  test('an old entry falls back from relative to the facility-zone time', () => {
    applyFacilityTimezone({ facility_timezone: 'Asia/Tokyo' });
    const old = '2020-01-01T09:00:00+09:00';
    expect(formatRelativeTime(old)).toBe(intl(old, DISPLAY, 'Asia/Tokyo'));
  });

  test('no locale is pinned in the ARIEL formatters', () => {
    const source = readFileSync(
      join(import.meta.dirname, '../../../src/osprey/interfaces/ariel/static/js/components.js'),
      'utf8',
    );
    expect(source).not.toMatch(/['"]en-US['"]/);
  });
});

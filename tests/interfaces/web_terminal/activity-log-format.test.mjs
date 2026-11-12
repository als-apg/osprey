// @ts-check
/**
 * Unit tests for the Activity page's formatters
 * (web_terminal/static/js/activity-log-format.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-format.test.mjs
 */
import { afterEach, describe, expect, test } from 'vitest';

import {
  argSummary,
  clock,
  dayLabel,
  formatBytes,
  formatDuration,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-format.js';
import { FACILITY_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';

describe('formatDuration', () => {
  test('nothing measured renders nothing', () => {
    expect(formatDuration(null)).toBe('');
    expect(formatDuration(undefined)).toBe('');
    expect(formatDuration(Number.NaN)).toBe('');
  });

  test('milliseconds, seconds, minutes, hours', () => {
    expect(formatDuration(0)).toBe('0 ms');
    expect(formatDuration(840)).toBe('840 ms');
    expect(formatDuration(839.6)).toBe('840 ms');
    expect(formatDuration(17400)).toBe('17.4 s');
    expect(formatDuration(125000)).toBe('2 min 5 s');
    expect(formatDuration(3_900_000)).toBe('1 h 5 min');
  });

  test('a unit never shows its own upper bound', () => {
    expect(formatDuration(999.7)).toBe('1.0 s');
    expect(formatDuration(59_980)).toBe('1 min 0 s');
    expect(formatDuration(119_600)).toBe('2 min 0 s');
    expect(formatDuration(3_599_700)).toBe('1 h 0 min');
  });
});

describe('formatBytes', () => {
  test('bytes, kilobytes, megabytes', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(null)).toBe('0 B');
    expect(formatBytes(undefined)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2117)).toBe('2.1 KB');
    expect(formatBytes(3 * 1048576)).toBe('3.0 MB');
  });
});

describe('argSummary', () => {
  test('one key=value per argument, two spaces apart', () => {
    expect(argSummary({ channel: 'SR:DCCT', limit: 5, raw: true })).toBe(
      'channel=SR:DCCT  limit=5  raw=true'
    );
  });

  test('a value is cut at 44 characters', () => {
    const summary = argSummary({ code: 'x'.repeat(60) });
    expect(summary).toBe(`code=${'x'.repeat(43)}…`);
    expect(argSummary({ code: 'x'.repeat(44) })).toBe(`code=${'x'.repeat(44)}`);
  });

  test('a nested value is its JSON; line breaks fold to one space', () => {
    expect(argSummary({ channels: ['A', 'B'], code: 'a = 1\n\nb = 2' })).toBe(
      'channels=["A","B"]  code=a = 1 b = 2'
    );
  });

  test('no arguments, or arguments that are no object, summarize to nothing', () => {
    expect(argSummary({})).toBe('');
    expect(argSummary(null)).toBe('');
    expect(argSummary(undefined)).toBe('');
    expect(argSummary('text')).toBe('');
    expect(argSummary(['a'])).toBe('');
  });
});

describe('clock', () => {
  afterEach(() => stampFacilityZone(null));

  /** @param {Intl.DateTimeFormatOptions} extra */
  const expected = (extra) =>
    new Intl.DateTimeFormat(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hourCycle: 'h23',
      timeZone: FACILITY_ZONE,
      ...extra,
    }).format(new Date('2026-01-15T20:04:05.678Z'));

  test('reads the facility clock, 24-hour with seconds', () => {
    stampFacilityZone(FACILITY_ZONE);
    expect(clock('2026-01-15T20:04:05.678Z')).toBe(expected({}));
    expect(clock('2026-01-15T20:04:05.678Z')).toMatch(/^\d{2}:04:05$/);
  });

  test('ms adds the milliseconds', () => {
    stampFacilityZone(FACILITY_ZONE);
    const text = clock('2026-01-15T20:04:05.678Z', { ms: true });
    expect(text).toBe(expected({ fractionalSecondDigits: 3 }));
    expect(text).toMatch(/^\d{2}:04:05.678$/);
  });

  test('empty and unparseable input render nothing', () => {
    stampFacilityZone(FACILITY_ZONE);
    expect(clock('')).toBe('');
    expect(clock(null)).toBe('');
    expect(clock('not-a-time')).toBe('');
  });
});

describe('dayLabel', () => {
  afterEach(() => stampFacilityZone(null));

  const NOW = Date.parse('2026-11-12T12:00:00Z');

  test('today and yesterday by name', () => {
    stampFacilityZone('UTC');
    expect(dayLabel('2026-11-12T00:00:01Z', NOW)).toBe('Today');
    expect(dayLabel('2026-11-11T23:59:59Z', NOW)).toBe('Yesterday');
  });

  test('an older day by date, in the viewer locale', () => {
    stampFacilityZone('UTC');
    const want = new Intl.DateTimeFormat(undefined, {
      day: 'numeric',
      month: 'short',
      timeZone: 'UTC',
    }).format(new Date('2026-11-10T08:00:00Z'));
    expect(dayLabel('2026-11-10T08:00:00Z', NOW)).toBe(want);
  });

  test('a day of another year carries the year', () => {
    stampFacilityZone('UTC');
    const want = new Intl.DateTimeFormat(undefined, {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      timeZone: 'UTC',
    }).format(new Date('2025-12-31T08:00:00Z'));
    expect(dayLabel('2025-12-31T08:00:00Z', NOW)).toBe(want);
  });

  test('the day is the facility day, not the UTC day', () => {
    // 23:30 UTC on the 11th is already the 12th in Tokyo.
    stampFacilityZone('Asia/Tokyo');
    expect(dayLabel('2026-11-11T23:30:00Z', NOW)).toBe('Today');
    expect(dayLabel('2026-11-11T10:00:00Z', NOW)).toBe('Yesterday');
  });

  test('yesterday across a month boundary', () => {
    stampFacilityZone('UTC');
    expect(dayLabel('2026-10-31T10:00:00Z', Date.parse('2026-11-01T01:00:00Z'))).toBe('Yesterday');
  });

  test('empty and unparseable input render nothing', () => {
    expect(dayLabel('', NOW)).toBe('');
    expect(dayLabel(null, NOW)).toBe('');
    expect(dayLabel('not-a-time', NOW)).toBe('');
  });
});

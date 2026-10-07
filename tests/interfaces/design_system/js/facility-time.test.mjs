/**
 * Unit tests for facility-time.js — the one browser module that renders an
 * instant in the facility zone stamped on `<html data-facility-timezone>`.
 *
 * Expected display strings are computed with Intl in the runner's locale, so
 * the suite holds in any locale and any viewer zone:
 *   npx vitest run tests/interfaces/design_system/js/facility-time.test.mjs
 *   TZ=UTC npx vitest run tests/interfaces/design_system/js/facility-time.test.mjs
 *   TZ=Asia/Tokyo npx vitest run tests/interfaces/design_system/js/facility-time.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  facilityZone,
  formatFacilityTime,
  facilityZoneLabel,
  facilityDayKey,
  facilityWallClock,
  viewerSharesFacilityClock,
} from '/design-system/js/facility-time.js';

const ATTR = 'data-facility-timezone';
const INSTANT = '2026-01-15T20:04:05Z';
/** @type {Intl.DateTimeFormatOptions} */
const HM = { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' };
const VIEWER = new Intl.DateTimeFormat().resolvedOptions().timeZone;

/** @param {string} zone */
function stamp(zone) {
  document.documentElement.setAttribute(ATTR, zone);
}

/**
 * @param {string} zone
 * @param {Intl.DateTimeFormatOptions} options
 */
function intl(zone, options) {
  return new Intl.DateTimeFormat(undefined, { ...options, timeZone: zone }).format(
    new Date(INSTANT),
  );
}

describe('facility-time.js', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute(ATTR);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('facilityZone', () => {
    test('returns the stamped zone', () => {
      stamp('Asia/Tokyo');
      expect(facilityZone()).toEqual({ id: 'Asia/Tokyo', facility: true });
    });

    test("absent or empty means the viewer's zone", () => {
      expect(facilityZone()).toEqual({ id: VIEWER, facility: false });
      stamp('');
      expect(facilityZone()).toEqual({ id: VIEWER, facility: false });
    });

    test('an id this browser does not know falls back and warns once', () => {
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      stamp('Not/AZone');
      expect(facilityZone()).toEqual({ id: VIEWER, facility: false });
      expect(facilityZone()).toEqual({ id: VIEWER, facility: false });
      expect(warn).toHaveBeenCalledTimes(1);
    });
  });

  describe('formatFacilityTime', () => {
    test('renders in the stamped zone', () => {
      stamp('Asia/Tokyo');
      expect(formatFacilityTime(INSTANT, HM)).toBe('05:04');
      stamp('UTC');
      expect(formatFacilityTime(INSTANT, HM)).toBe('20:04');
    });

    test('a stamped zone is not labelled', () => {
      stamp('Asia/Tokyo');
      expect(formatFacilityTime(INSTANT, HM)).toBe(intl('Asia/Tokyo', HM));
    });

    test("the fallback names the viewer's zone on a time", () => {
      expect(formatFacilityTime(INSTANT, HM)).toBe(
        intl(VIEWER, { ...HM, timeZoneName: 'short' }),
      );
    });

    test('a date-only fallback carries no zone name', () => {
      /** @type {Intl.DateTimeFormatOptions} */
      const dateOnly = { month: 'short', day: 'numeric' };
      expect(formatFacilityTime(INSTANT, dateOnly)).toBe(intl(VIEWER, dateOnly));
    });

    test('reads the stamp on every call', () => {
      const zone = VIEWER === 'Asia/Tokyo' ? 'Europe/Berlin' : 'Asia/Tokyo';
      const before = formatFacilityTime(INSTANT, HM);
      stamp(zone);
      const after = formatFacilityTime(INSTANT, HM);
      expect(after).not.toBe(before);
      expect(after).toBe(intl(zone, HM));
    });

    test('bad input is the empty string', () => {
      stamp('UTC');
      for (const bad of ['', null, undefined, 'garbage', new Date(NaN)]) {
        expect(formatFacilityTime(bad, HM)).toBe('');
      }
      const iso = formatFacilityTime(INSTANT, HM);
      expect(formatFacilityTime(Date.parse(INSTANT), HM)).toBe(iso);
      expect(formatFacilityTime(new Date(INSTANT), HM)).toBe(iso);
    });
  });

  describe('facilityZoneLabel', () => {
    test('the short name of the stamped zone', () => {
      stamp('Asia/Tokyo');
      const part = new Intl.DateTimeFormat(undefined, {
        timeZone: 'Asia/Tokyo',
        timeZoneName: 'short',
      })
        .formatToParts(new Date(INSTANT))
        .find((p) => p.type === 'timeZoneName');
      expect(facilityZoneLabel(Date.parse(INSTANT))).toBe(part ? part.value : 'Asia/Tokyo');
    });
  });

  describe('facilityDayKey', () => {
    test('the calendar day in the stamped zone', () => {
      stamp('Asia/Tokyo');
      expect(facilityDayKey(INSTANT)).toBe('2026-01-16');
      stamp('UTC');
      expect(facilityDayKey(INSTANT)).toBe('2026-01-15');
      expect(facilityDayKey('garbage')).toBe('');
    });
  });

  describe('facilityWallClock', () => {
    test('the wall-clock digits in the stamped zone, to the millisecond', () => {
      stamp('Asia/Tokyo');
      expect(facilityWallClock('2026-01-15T20:04:05.123Z')).toBe('2026-01-16 05:04:05.123');
      stamp('UTC');
      expect(facilityWallClock(INSTANT)).toBe('2026-01-15 20:04:05.000');
    });

    test('midnight is 00, never 24', () => {
      stamp('UTC');
      expect(facilityWallClock('2026-01-15T00:00:00Z')).toBe('2026-01-15 00:00:00.000');
      stamp('Asia/Tokyo');
      expect(facilityWallClock('2026-01-15T15:00:00Z')).toBe('2026-01-16 00:00:00.000');
    });

    test('bad input is the empty string', () => {
      stamp('Asia/Tokyo');
      for (const bad of ['', null, undefined, 'garbage', new Date(NaN)]) {
        expect(facilityWallClock(bad)).toBe('');
      }
    });
  });

  describe('viewerSharesFacilityClock', () => {
    test('false when the stamp differs from the viewer', () => {
      stamp(VIEWER === 'Asia/Tokyo' ? 'Europe/Berlin' : 'Asia/Tokyo');
      expect(viewerSharesFacilityClock(Date.parse(INSTANT))).toBe(false);
    });

    test('true when the stamp is the viewer\'s zone and when absent', () => {
      expect(viewerSharesFacilityClock(Date.parse(INSTANT))).toBe(true);
      stamp(VIEWER);
      expect(viewerSharesFacilityClock(Date.parse(INSTANT))).toBe(true);
    });
  });
});

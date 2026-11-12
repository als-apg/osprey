// @ts-check
/**
 * The panel's one way of writing a span of time (duration.js).
 *   npx vitest run tests/interfaces/channel_finder/duration.test.mjs
 */

import { test, expect, describe } from 'vitest';
import {
  durationText,
  sinceText,
} from '../../../src/osprey/interfaces/channel_finder/static/js/duration.js';

describe('sinceText', () => {
  const now = Date.parse('2026-11-06T12:40:00.000Z');

  test.each([
    ['2026-11-06T12:39:55.000Z', '5 s'],
    ['2026-11-06T12:02:00.000Z', '38 min'],
    ['2026-11-06T10:40:00.000Z', '2 h'],
    ['2026-11-06T10:35:00.000Z', '2 h 5 min'],
    ['2026-11-04T09:40:00.000Z', '2 d 3 h'],
    ['2026-11-06T12:40:05.000Z', '0 s'],
    [null, ''],
    ['not a time', ''],
  ])('%s is %s ago', (time, text) => {
    expect(sinceText(time, now)).toBe(text);
  });
});

describe('durationText', () => {
  test('a number of seconds in the largest two units', () => {
    expect(durationText(3 * 86400 + 4 * 3600 + 59)).toBe('3 d 4 h');
  });
});

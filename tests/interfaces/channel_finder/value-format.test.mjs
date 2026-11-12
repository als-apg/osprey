// @ts-check
/**
 * A value read as text (value-format.js formatValue).
 *   npx vitest run tests/interfaces/channel_finder/value-format.test.mjs
 */

import { test, expect, describe } from 'vitest';
import { formatValue } from '../../../src/osprey/interfaces/channel_finder/static/js/value-format.js';

/**
 * A values-route row that answered `value`.
 * @param {unknown} value
 * @param {Record<string, unknown>} [extra]
 * @returns {any}
 */
const found = (value, extra = {}) => ({
  name: 'X',
  outcome: 'found',
  value,
  timestamp: null,
  severity: 0,
  status: 'NO_ALARM',
  ...extra,
});

describe('formatValue', () => {
  test('a PV that did not answer says so, with the reason on hover', () => {
    const row = { ...found(null), outcome: 'no_answer', reason: 'no answer within 2 s' };
    expect(formatValue(row)).toEqual({
      text: 'no answer',
      unit: '',
      title: 'no answer within 2 s',
    });
  });

  test('an unavailable read shows its reason', () => {
    const row = { ...found(null), outcome: 'unavailable', reason: 'not available for this target' };
    expect(formatValue(row).text).toBe('not available for this target');
  });

  test.each([
    [0.125, '0.125'],
    [46.2000001, '46.2'],
    [1.23456789, '1.23457'],
    [3.1e-9, '3.1e-9'],
    [-0.5, '-0.5'],
    [6379, '6379'],
    [1234567, '1234567'],
  ])('the number %s reads %s', (value, text) => {
    expect(formatValue(found(value)).text).toBe(text);
  });

  test('a rounded number keeps the full value in the title', () => {
    expect(formatValue(found(1.23456789)).title).toBe('1.23456789');
  });

  test('text is as is, an array or object is JSON', () => {
    expect(formatValue(found('RUNNING')).text).toBe('RUNNING');
    expect(formatValue(found([1, 2])).text).toBe('[1,2]');
    expect(formatValue(found({ a: 'b' })).text).toBe('{"a":"b"}');
  });

  test('units and precision are read when the row carries them', () => {
    expect(formatValue(found(46.234, { units: 'degC', precision: 1 }))).toEqual({
      text: '46.2',
      unit: 'degC',
      title: '46.234',
    });
    expect(formatValue(found(46.234)).unit).toBe('');
  });

  test('an enum shows its label, with the index on hover', () => {
    expect(formatValue(found(1, { enum_label: 'On' }))).toEqual({
      text: 'On',
      unit: '',
      title: '1',
    });
    expect(formatValue(found(1, { enum_label: null })).text).toBe('1');
  });

  test('a precision that is not a count is ignored', () => {
    expect(formatValue(found(1.23456789, { precision: -1 })).text).toBe('1.23457');
    expect(formatValue(found(1.5, { precision: 'x' })).text).toBe('1.5');
  });
});

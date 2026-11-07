// @ts-check
/**
 * Unit tests for the System Health `?demo` fixtures (demo.js):
 *   npx vitest run tests/interfaces/health/demo.test.mjs
 */

import { test, expect, describe } from 'vitest';

import { demoData } from '../../../src/osprey/interfaces/health/static/js/demo.js';

describe('demoData', () => {
  test('the default snapshot counts its own results', () => {
    const d = demoData('');
    const count = (/** @type {string} */ s) => d.results.filter((r) => r.status === s).length;
    expect(d.total).toBe(d.results.length);
    expect([d.ok, d.warnings, d.errors, d.skips]).toEqual([
      count('ok'),
      count('warning'),
      count('error'),
      count('skip'),
    ]);
    expect(d.summary.startsWith(d.ok + '/' + d.total + ' checks passed')).toBe(true);
    expect(d.warming).toBe(false);
    expect(d.stale).toBe(false);
  });

  test('stale and warming exercise the envelope states', () => {
    const stale = demoData('stale');
    expect(stale.stale).toBe(true);
    expect(stale.deadline_hit).toBe(true);
    const warming = demoData('warming');
    expect(warming.warming).toBe(true);
    expect(warming.results).toEqual([]);
  });
});

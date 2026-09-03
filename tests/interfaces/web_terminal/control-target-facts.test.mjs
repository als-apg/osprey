// @ts-check
/**
 * Unit tests for the control-target derived facts (control-target-facts.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-facts.test.mjs
 *
 * The module is the pure half of the control-target vocabulary — what a row,
 * a confirm and the first-contact copy SAY about a machine, derived from the
 * state the chip publishes and nothing else. The popover suite drives the same
 * wording through a rendered panel; this file pins the tables themselves, so a
 * phrase can be changed (or drift into a wrong claim) with a failing test
 * naming the phrase rather than a DOM assertion three layers away.
 */
import { describe, expect, test } from 'vitest';

import {
  KIND_READ_PHRASES,
  KIND_WORDS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/control-target-facts.js';

describe('KIND_READ_PHRASES', () => {
  test('names where the values come from, per kind', () => {
    expect(KIND_READ_PHRASES.live).toBe('read live machine values');
    expect(KIND_READ_PHRASES.standin).toBe('read values from the rehearsal copy');
    expect(KIND_READ_PHRASES.va).toBe('read values from the simulator');
    expect(KIND_READ_PHRASES.simulated).toBe('read demo data');
  });

  test('covers exactly the kinds KIND_WORDS names', () => {
    // The two tables are read for the same chip state. A kind in one and not
    // the other renders a machine with a name and no capability sentence — or
    // worse, falls back to another kind's promise about the values.
    expect(Object.keys(KIND_READ_PHRASES).sort()).toEqual(Object.keys(KIND_WORDS).sort());
  });
});

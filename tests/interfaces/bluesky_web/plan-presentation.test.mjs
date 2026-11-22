/**
 * Unit tests for the plan form's live readout (plan-presentation.js).
 *
 *   npx vitest run tests/interfaces/bluesky_web/plan-presentation.test.mjs
 *
 * Pure functions over a catalog entry and collected `plan_args`: no DOM.
 */

import { test, expect } from 'vitest';

import {
  fillSummary,
  summarizePlanArgs,
} from '../../../src/osprey/interfaces/bluesky_web/panels/bluesky/plan-presentation.js';

const GAP_SUMMARY = ['gap {gap}', '{start} to {stop}', '{num} points', '{monitors} monitors'];

/**
 * A catalog entry as `GET /plans` serves it.
 *
 * @param {string} name
 * @param {string[]|null} summary
 */
function entry(name, summary) {
  return {
    name,
    provenance: 'facility',
    metadata: { name, description: 'A plan.', writes: true, layout: null, summary },
  };
}

test('a part shows when every parameter it names has a value', () => {
  expect(fillSummary(GAP_SUMMARY, { gap: 'ID:GAP:SP' })).toBe('gap ID:GAP:SP');
  expect(fillSummary(GAP_SUMMARY, { gap: 'ID:GAP:SP', start: 12, num: 25 })).toBe(
    'gap ID:GAP:SP · 25 points'
  );
  expect(fillSummary(GAP_SUMMARY, { gap: 'ID:GAP:SP', start: 12, stop: 18, num: 25 })).toBe(
    'gap ID:GAP:SP · 12 to 18 · 25 points'
  );
  expect(fillSummary(['{gap}', '{num} points'], { gap: null, num: { a: 1 } })).toBe('');
  expect(fillSummary(['fast {fast}'], { fast: false })).toBe('fast false');
});

test('a list parameter fills in as its count', () => {
  const args = { gap: 'ID:GAP:SP', start: 12, stop: 18, num: 25, monitors: ['FLUX:1', 'FLUX:2'] };
  expect(fillSummary(GAP_SUMMARY, args)).toBe('gap ID:GAP:SP · 12 to 18 · 25 points · 2 monitors');
  expect(fillSummary(['{monitors} monitors'], { monitors: [] })).toBe('');
});

test('doubled braces are literal', () => {
  expect(fillSummary(['{{gap}} {gap}', 'set {{}}'], { gap: 'G' })).toBe('{gap} G · set {}');
});

test('a plan with no summary and no table entry reads the count', () => {
  const plan = entry('facility_plan', null);
  expect(summarizePlanArgs(plan, { a: 1, b: 2 })).toBe('2 parameters set');
  expect(summarizePlanArgs(plan, { a: 1 })).toBe('1 parameter set');
  expect(summarizePlanArgs(plan, {})).toBe('');
  expect(summarizePlanArgs(undefined, { a: 1 })).toBe('1 parameter set');
  // A stated summary with nothing to show yet falls back the same way.
  expect(summarizePlanArgs(entry('facility_plan', ['{z} zones']), { a: 1 })).toBe(
    '1 parameter set'
  );
});

test('a stated summary wins over the shipped table', () => {
  const plan = entry('orm', ['{num} kicks']);
  expect(summarizePlanArgs(plan, { correctors: ['HCM1', 'HCM2'], num: 7 })).toBe('7 kicks');
});

test('the shipped orm readout is unchanged', () => {
  const plan = entry('orm', null);
  const args = {
    correctors: ['HCM1', 'HCM2'],
    readbacks: ['BPM1', 'BPM2', 'BPM3'],
    span: 5,
    num: 7,
  };
  expect(summarizePlanArgs(plan, args)).toBe(
    '2 correctors · 3 BPMs · ±5 · 2 × 7 = 14 sweep points'
  );
  expect(summarizePlanArgs(plan, { ...args, sweep: 'monodirectional' })).toBe(
    '2 correctors · 3 BPMs · 0…5 · 2 × 7 = 14 sweep points'
  );
});

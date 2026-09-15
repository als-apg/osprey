/**
 * The shared severity order for health-check statuses (check-status.js).
 *
 * Two surfaces reduce a list of checks to the single worst status present:
 * the System Health dashboard's own bundle and the web terminal's health bar
 * item. The design system is the only mount both pages hold, so the order and
 * the reducer that reads it live here and both surfaces import them.
 *
 * Imported through the absolute `/design-system/js/*` specifier the panels
 * themselves use (aliased in vitest.config.js), so this exercises the same
 * module graph the browser loads.
 *
 *   npx vitest run tests/interfaces/design_system/js/check-status.test.mjs
 */

import { test, expect } from 'vitest';

import { STATUS_RANK, worstStatus } from '/design-system/js/check-status.js';

test('severity runs error over warning over skip over ok', () => {
  expect(STATUS_RANK.error).toBeGreaterThan(STATUS_RANK.warning);
  expect(STATUS_RANK.warning).toBeGreaterThan(STATUS_RANK.skip);
  expect(STATUS_RANK.skip).toBeGreaterThan(STATUS_RANK.ok);
});

test('the worst status wins wherever it sits in the list', () => {
  expect(worstStatus([{ status: 'ok' }, { status: 'warning' }, { status: 'ok' }])).toBe('warning');
  expect(worstStatus([{ status: 'error' }, { status: 'warning' }, { status: 'skip' }])).toBe('error');
  expect(worstStatus([{ status: 'skip' }, { status: 'warning' }, { status: 'error' }])).toBe('error');
  expect(worstStatus([{ status: 'ok' }, { status: 'skip' }])).toBe('skip');
});

test('an empty list is ok', () => {
  expect(worstStatus([])).toBe('ok');
});

test('a status the table does not name neither wins nor displaces a real one', () => {
  expect(worstStatus([{ status: 'ok' }, { status: 'mystery' }])).toBe('ok');
  expect(worstStatus([{ status: 'mystery' }, { status: 'warning' }])).toBe('warning');
  expect(worstStatus([{ status: 'mystery' }])).toBe('ok');
});

test('the table is frozen, so no importer can extend it for everyone else', () => {
  expect(Object.isFrozen(STATUS_RANK)).toBe(true);
});

// @ts-check
/**
 * Unit tests for latest-request.js — the design system's newest-request-wins
 * primitive. Only the newest request of one kind is current: starting one
 * aborts and retires the previous, and a cancel retires without starting.
 *
 * Pure module, no DOM:
 *   npx vitest run tests/interfaces/design_system/latest-request.test.mjs
 */

import { test, expect } from 'vitest';

import { createLatestRequest } from '/design-system/js/latest-request.js';

test('a token is current until the next begin', () => {
  const latest = createLatestRequest();
  const first = latest.begin();
  expect(first.isCurrent()).toBe(true);

  const second = latest.begin();
  expect(first.isCurrent()).toBe(false);
  expect(second.isCurrent()).toBe(true);
});

test("begin aborts the previous token's signal", () => {
  const latest = createLatestRequest();
  const first = latest.begin();
  expect(first.signal.aborted).toBe(false);

  const second = latest.begin();
  expect(first.signal.aborted).toBe(true);
  expect(second.signal.aborted).toBe(false);
});

test('cancel aborts and retires the current token', () => {
  const latest = createLatestRequest();
  const token = latest.begin();

  latest.cancel();
  expect(token.signal.aborted).toBe(true);
  expect(token.isCurrent()).toBe(false);
});

test('a cancel with nothing in flight is harmless and a later begin is current', () => {
  const latest = createLatestRequest();
  expect(() => latest.cancel()).not.toThrow();

  const token = latest.begin();
  expect(token.isCurrent()).toBe(true);
  expect(token.signal.aborted).toBe(false);
});

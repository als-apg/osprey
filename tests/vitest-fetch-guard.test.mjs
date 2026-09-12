// @ts-check
/**
 * Unit tests for the unstubbed-request guard in vitest.setup.mjs:
 *   npx vitest run tests/vitest-fetch-guard.test.mjs
 *
 * Nothing serves the environment's origin, so a request no test stubbed is a
 * missing stub. The guard refuses it at the call site and records it, and the
 * setup file's own teardown fails the test that made it. That record is
 * published on `globalThis` precisely so this file can make such a request on
 * purpose: it reads the record, clears it, and the teardown then sees nothing.
 */

import { test, expect } from 'vitest';

/** The key vitest.setup.mjs publishes its record of refused requests under. */
const RECORD_KEY = '__OSPREY_UNSTUBBED_FETCHES__';

/** @returns {string[]} */
function record() {
  return /** @type {string[]} */ (Reflect.get(globalThis, RECORD_KEY));
}

test('an unstubbed request is refused at the call site and recorded', async () => {
  const url = 'http://localhost:3000/api/nothing-stubs-this';

  await expect(fetch(url)).rejects.toThrow(url);
  expect(record()).toContain(url);

  // Clearing it is what leaves this file's own teardown green — the drain the
  // setup file runs after every test finds nothing left to report.
  record().length = 0;
});

test('a cleared record leaves nothing for the next test', () => {
  expect(record()).toEqual([]);
});

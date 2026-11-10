// @ts-check
/**
 * Contract tests for the hub's start-up tail (panel-startup.js):
 *   npx vitest run tests/interfaces/web_terminal/panel-startup.test.mjs
 *
 * finishStartup(panelConfig) hands the /api/panels payload to the six hub
 * surfaces outside the panel stack, in order, and sets html[data-panels-ready]
 * only after the last of them. A failed fetch (null) still reaches every
 * consumer, and first contact is told the facts are in with nothing in them.
 *
 * Every consumer is a stub at its module boundary; each one's own rule is
 * pinned in its own test file.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

const { calls, spies } = vi.hoisted(() => {
  /** @type {{ name: string, arg: any, ready: string | undefined }[]} */
  const calls = [];
  const record = (/** @type {string} */ name) => (/** @type {any} */ arg) => {
    calls.push({ name, arg, ready: document.documentElement.dataset.panelsReady });
  };
  return {
    calls,
    spies: {
      applyConfigTabGate: record('applyConfigTabGate'),
      applyConfigUnreadableNotice: record('applyConfigUnreadableNotice'),
      applyScaffoldWriteGate: record('applyScaffoldWriteGate'),
      applyTourConfig: record('applyTourConfig'),
      applyLoginNudge: record('applyLoginNudge'),
      setFacts: record('setFacts'),
    },
  };
});

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/config-tab.js', () => ({
  applyConfigTabGate: spies.applyConfigTabGate,
  applyConfigUnreadableNotice: spies.applyConfigUnreadableNotice,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/write-gate.js', () => ({
  applyScaffoldWriteGate: spies.applyScaffoldWriteGate,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tour.js', () => ({
  applyTourConfig: spies.applyTourConfig,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/login-nudge.js', () => ({
  applyLoginNudge: spies.applyLoginNudge,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/first-contact.js', () => ({
  setFacts: spies.setFacts,
}));

const { finishStartup } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/panel-startup.js');

const ORDER = [
  'applyConfigTabGate',
  'applyConfigUnreadableNotice',
  'applyScaffoldWriteGate',
  'applyTourConfig',
  'applyLoginNudge',
  'setFacts',
];

beforeEach(() => {
  calls.length = 0;
  delete document.documentElement.dataset.panelsReady;
});

describe('finishStartup', () => {
  test('hands the payload to the six consumers in order', () => {
    const tour = { invite: 'first-visit', capabilities: [] };
    const panelConfig = { panels: [], tour };

    finishStartup(panelConfig);

    expect(calls.map((c) => c.name)).toEqual(ORDER);
    for (const call of calls.slice(0, -1)) expect(call.arg).toBe(panelConfig);
    expect(calls.at(-1)?.arg).toBe(tour);
  });

  test('sets the ready mark last', () => {
    finishStartup({ tour: {} });

    expect(calls.every((c) => c.ready === undefined)).toBe(true);
    expect(document.documentElement.dataset.panelsReady).toBe('true');
  });

  test('a failed fetch still reaches every consumer and marks the hub ready', () => {
    finishStartup(null);

    expect(calls.map((c) => c.name)).toEqual(ORDER);
    expect(calls.every((c) => c.arg === null)).toBe(true);
    expect(document.documentElement.dataset.panelsReady).toBe('true');
  });
});

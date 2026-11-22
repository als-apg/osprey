// @ts-check
/**
 * Unit tests for activity-history.js — the recent-activity popover that opens
 * from the activity strip:
 *   npx vitest run tests/interfaces/web_terminal/activity-history.test.mjs
 *
 * The history is read from the server on every open. Each read here waits for
 * the test to answer it, so an earlier open's read can land after a later one.
 * What is asserted is the list the operator sees.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import { createActivityHistory } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-history.js';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js').AgentActivityEvent} AgentActivityFrame */

/** @type {Array<(events: AgentActivityFrame[]) => void>} */
let answers;
/** @type {HTMLElement} */
let mount;

/**
 * A channel write as the server's buffer holds it.
 * @param {string} channel
 * @returns {AgentActivityFrame}
 */
function written(channel) {
  return {
    type: 'agent_activity',
    tool: 'write_channel',
    target: { kind: 'channel', detail: channel },
    ts: 1234,
  };
}

/** Let an answered read reach the popover. */
async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
}

/** The popover's rows, top to bottom. */
function rowTexts() {
  return [...document.querySelectorAll('.activity-history-row')].map((r) => r.textContent);
}

beforeEach(() => {
  answers = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(
      () =>
        new Promise((resolve) => {
          answers.push((events) =>
            resolve({ ok: true, status: 200, json: async () => ({ events }) })
          );
        })
    )
  );
  mount = document.createElement('div');
  document.body.append(mount);
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

test("a reopen shows its own read's history when the earlier open's read lands last", async () => {
  const history = createActivityHistory({ mount });
  const first = history.open();
  history.close();
  const second = history.open();
  expect(answers).toHaveLength(2);

  answers[1]([written('SR01:HCM2:SP')]);
  await settle();
  answers[0]([written('SR01:HCM1:SP')]);
  await Promise.all([first, second]);

  expect(history.isOpen()).toBe(true);
  expect(rowTexts()).toHaveLength(1);
  expect(rowTexts()[0]).toContain('SR01:HCM2:SP');
});

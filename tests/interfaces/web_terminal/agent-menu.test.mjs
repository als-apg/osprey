// @ts-check
/**
 * Unit tests for the Agent pane's ⋯ rows (agent-menu.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-menu.test.mjs
 *
 * The rows reach the hub's menu through setServiceMenuExtras, a double here:
 * how the hub lays the tile ⋯ out (sections, headings, danger last) is
 * panel-menu-policy's and tile-menu's, tested beside them.
 */

import { test, expect, beforeEach, vi } from 'vitest';

import { TERMINAL_RAIL_ID } from '../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js';

const hub = vi.hoisted(() => ({ setServiceMenuExtras: vi.fn() }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => hub);

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/agent-menu.js')} */
let menu;

beforeEach(async () => {
  vi.resetModules();
  hub.setServiceMenuExtras.mockReset();
  menu = await import('../../../src/osprey/interfaces/web_terminal/static/js/agent-menu.js');
});

/** The last registration the hub received: the panel id and the rows function. */
const registered = () => /** @type {[string, any]} */ (hub.setServiceMenuExtras.mock.lastCall);

/** The labels the hub's ⋯ shows now, read through what it was handed. */
const shown = () => registered()[1]().map((/** @type {any} */ row) => row.label);

test('the rows reach the hub under the Agent pane id', () => {
  menu.setAgentMenuRow('export', () => ({ label: 'Export conversation (Markdown)', run() {} }));
  expect(registered()[0]).toBe(TERMINAL_RAIL_ID);
  expect(shown()).toEqual(['Export conversation (Markdown)']);
});

test('Export comes first, whatever order the features started in', () => {
  menu.setAgentMenuRow('notifications', () => ({ label: 'Desktop notifications', run() {} }));
  menu.setAgentMenuRow('export', () => ({ label: 'Export conversation (Markdown)', run() {} }));
  expect(shown()).toEqual(['Export conversation (Markdown)', 'Desktop notifications']);
});

test('a row is asked at every open, so it follows the pane', () => {
  let on = false;
  menu.setAgentMenuRow('notifications', () => ({ label: 'Desktop notifications', checked: on }));
  const rows = registered()[1];
  expect(rows()[0].checked).toBe(false);
  on = true;
  expect(rows()[0].checked).toBe(true);
});

test('a row that answers null is left out', () => {
  menu.setAgentMenuRow('notifications', () => null);
  menu.setAgentMenuRow('export', () => ({ label: 'Export conversation (Markdown)', run() {} }));
  expect(shown()).toEqual(['Export conversation (Markdown)']);
});

test('a row taken away leaves the menu; the last one takes the pane out of the hub', () => {
  menu.setAgentMenuRow('export', () => ({ label: 'Export conversation (Markdown)', run() {} }));
  menu.setAgentMenuRow('notifications', () => ({ label: 'Desktop notifications', run() {} }));
  menu.setAgentMenuRow('export', null);
  expect(shown()).toEqual(['Desktop notifications']);
  menu.setAgentMenuRow('notifications', null);
  expect(registered()).toEqual([TERMINAL_RAIL_ID, null]);
  expect(menu.agentMenuRows()).toEqual([]);
});

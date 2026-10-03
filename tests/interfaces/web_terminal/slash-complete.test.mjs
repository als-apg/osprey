// @ts-check
/**
 * Unit tests for the Simple view's slash-command suggestions (slash-complete.js):
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/slash-complete.test.mjs
 *
 * The keyboard contract is what is pinned: the list opens only on a single
 * leading `/` token, arrows move the armed row with wrap, Tab or Enter fills
 * the box with the armed command and never sends, Escape closes until the
 * text changes, and a key the list does not take is left to the caller.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { attachSlashComplete } from '../../../src/osprey/interfaces/web_terminal/static/js/slash-complete.js';

const COMMANDS = [
  { name: 'diagnose', description: 'Investigate failures', argument_hint: '', kind: 'skill' },
  { name: 'session-report', description: 'Write a report', argument_hint: '', kind: 'skill' },
  { name: 'sim-scenarios', description: 'Run scenarios', argument_hint: '', kind: 'skill' },
  { name: 'ops:nested', description: 'Nested command', argument_hint: '<pv>', kind: 'command' },
];

/** @type {HTMLTextAreaElement} */
let textarea;
/** @type {HTMLElement} */
let mount;
/** @type {import('vitest').Mock<() => Promise<any[]>>} */
let load;
/** @type {ReturnType<typeof attachSlashComplete>} */
let slash;

const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

/** @param {string} value */
function type(value) {
  textarea.value = value;
  textarea.dispatchEvent(new Event('input'));
}

/**
 * @param {string} key
 * @param {KeyboardEventInit} [init]
 */
function key(key, init = {}) {
  const event = new KeyboardEvent('keydown', { key, cancelable: true, ...init });
  const handled = slash.handleKeydown(event);
  return { handled, event };
}

const popup = () => /** @type {HTMLElement} */ (mount.querySelector('.op-slash-popup'));
const rows = () => [...mount.querySelectorAll('[role=option]')];
const rowNames = () => rows().map((r) => r.querySelector('.op-slash-name')?.textContent);
const armed = () => mount.querySelector('[aria-selected="true"]');

beforeEach(() => {
  document.body.replaceChildren();
  mount = document.createElement('div');
  textarea = document.createElement('textarea');
  mount.append(textarea);
  document.body.append(mount);
  load = vi.fn(() => Promise.resolve(/** @type {any[]} */ (COMMANDS)));
  slash = attachSlashComplete(textarea, { load, mount });
});

describe('attachSlashComplete', () => {
  test("'/' opens the list with every command, first row armed", async () => {
    type('/');
    await flush();
    expect(popup().hidden).toBe(false);
    expect(slash.isOpen()).toBe(true);
    expect(textarea.getAttribute('aria-expanded')).toBe('true');
    expect(textarea.getAttribute('role')).toBe('combobox');
    expect(textarea.getAttribute('aria-controls')).toBe(popup().id);
    expect(rowNames()).toEqual(['/diagnose', '/ops:nested', '/session-report', '/sim-scenarios']);
    expect(textarea.getAttribute('aria-activedescendant')).toBe(rows()[0].id);
    expect(rows()[0].getAttribute('aria-selected')).toBe('true');
    expect(rows()[1].querySelector('.op-slash-hint')?.textContent).toBe('<pv>');
    expect(rows()[0].querySelector('.op-slash-hint')).toBeNull();
  });

  test('typing narrows by fuzzy match', async () => {
    type('/sim');
    await flush();
    expect(rowNames()[0]).toBe('/sim-scenarios');
  });

  test('the list closes off a single leading slash token', async () => {
    for (const value of ['/diagnose x', 'diagnose', 'check /diag']) {
      type('/');
      await flush();
      expect(slash.isOpen()).toBe(true);
      type(value);
      await flush();
      expect(slash.isOpen()).toBe(false);
      expect(popup().hidden).toBe(true);
      expect(textarea.getAttribute('aria-expanded')).toBe('false');
    }
  });

  test('arrows move the armed row and wrap', async () => {
    type('/');
    await flush();
    const up = key('ArrowUp');
    expect(up.handled).toBe(true);
    expect(up.event.defaultPrevented).toBe(true);
    expect(armed()).toBe(rows()[3]);
    expect(textarea.getAttribute('aria-activedescendant')).toBe(rows()[3].id);
    const down = key('ArrowDown');
    expect(down.handled).toBe(true);
    expect(down.event.defaultPrevented).toBe(true);
    expect(armed()).toBe(rows()[0]);
  });

  test.each(['Enter', 'Tab'])('%s fills the box with the armed command', async (name) => {
    type('/');
    await flush();
    const { handled, event } = key(name);
    expect(handled).toBe(true);
    expect(event.defaultPrevented).toBe(true);
    expect(textarea.value).toBe('/diagnose ');
    expect(textarea.selectionStart).toBe('/diagnose '.length);
    expect(textarea.selectionEnd).toBe('/diagnose '.length);
    expect(slash.isOpen()).toBe(false);
  });

  test('Shift+Enter is left to the caller', async () => {
    type('/');
    await flush();
    expect(key('Enter', { shiftKey: true }).handled).toBe(false);
  });

  test('Escape closes until the text changes', async () => {
    type('/di');
    await flush();
    const esc = key('Escape');
    expect(esc.handled).toBe(true);
    expect(esc.event.defaultPrevented).toBe(true);
    expect(slash.isOpen()).toBe(false);
    type('/di');
    await flush();
    expect(slash.isOpen()).toBe(false);
    type('/dia');
    await flush();
    expect(slash.isOpen()).toBe(true);
  });

  test('no match leaves Enter to the caller', async () => {
    type('/zzz');
    await flush();
    expect(slash.isOpen()).toBe(false);
    const { handled, event } = key('Enter');
    expect(handled).toBe(false);
    expect(event.defaultPrevented).toBe(false);
  });

  test('an IME composition is never taken', async () => {
    type('/');
    await flush();
    expect(key('Enter', { isComposing: true }).handled).toBe(false);
    expect(textarea.value).toBe('/');
  });

  test('a click picks a row and keeps focus', async () => {
    type('/');
    await flush();
    const row = rows()[2];
    const down = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
    row.dispatchEvent(down);
    expect(down.defaultPrevented).toBe(true);
    row.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(textarea.value).toBe('/session-report ');
  });

  test('a failed load stays closed and is retried', async () => {
    load.mockReset();
    load.mockRejectedValueOnce(new Error('down')).mockResolvedValue(COMMANDS);
    type('/');
    await flush();
    expect(slash.isOpen()).toBe(false);
    type('');
    type('/');
    await flush();
    expect(load).toHaveBeenCalledTimes(2);
    expect(slash.isOpen()).toBe(true);
    expect(rows()).toHaveLength(4);
  });

  test('each fresh open refetches', async () => {
    type('/');
    await flush();
    expect(load).toHaveBeenCalledTimes(1);
    slash.close();
    type('');
    /** @type {(v: any) => void} */
    let settle = () => {};
    load.mockReturnValueOnce(new Promise((resolve) => { settle = resolve; }));
    type('/');
    expect(load).toHaveBeenCalledTimes(2);
    // The previous list is shown while the new one is on its way.
    expect(slash.isOpen()).toBe(true);
    expect(rows()).toHaveLength(4);
    settle([COMMANDS[0]]);
    await flush();
    expect(rowNames()).toEqual(['/diagnose']);
  });
});

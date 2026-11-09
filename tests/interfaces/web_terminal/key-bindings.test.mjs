// @ts-check
/**
 * Unit tests for key-bindings.js, the one definition of the shortcuts a menu
 * row shows:
 *   npx vitest run tests/interfaces/web_terminal/key-bindings.test.mjs
 *
 * Pins the hint each platform prints, that every non-native hint is a chord
 * its own matcher claims, and that the matchers claim exactly what
 * terminal.js and collapsible.js have always claimed.
 */

import { test, expect, describe, afterEach } from 'vitest';

import {
  KEY_BINDINGS,
  bindingFor,
  hintFor,
  keyFor,
  matcherFor,
} from '../../../src/osprey/interfaces/web_terminal/static/js/key-bindings.js';

/** @param {string} value */
function stubPlatform(value) {
  Object.defineProperty(navigator, 'platform', { value, configurable: true });
  Object.defineProperty(navigator, 'userAgentData', { value: undefined, configurable: true });
}

/**
 * @param {Partial<KeyboardEventInit> & {key: string, type?: string}} init
 * @returns {KeyboardEvent}
 */
function keyEvent(init) {
  const { type = 'keydown', ...rest } = init;
  return new KeyboardEvent(type, { bubbles: true, cancelable: true, ...rest });
}

afterEach(() => stubPlatform(''));

describe('hintFor', () => {
  test.each([
    ['terminal-copy', 'MacIntel', '⌘C'],
    ['terminal-copy', 'Linux x86_64', 'Ctrl+Shift+C'],
    ['terminal-copy', 'Win32', 'Ctrl+Shift+C'],
    ['sidebar-toggle', 'MacIntel', '⌘\\'],
    ['sidebar-toggle', 'Linux x86_64', 'Ctrl+\\'],
  ])('%s on %s reads %s', (id, platform, hint) => {
    stubPlatform(platform);
    expect(hintFor(id)).toBe(hint);
  });

  test('reads the platform at call time, not at import', () => {
    stubPlatform('MacIntel');
    expect(hintFor('sidebar-toggle')).toBe('⌘\\');
    stubPlatform('Linux x86_64');
    expect(hintFor('sidebar-toggle')).toBe('Ctrl+\\');
  });

  test('an unknown id throws rather than printing nothing', () => {
    expect(() => hintFor('no-such-binding')).toThrow(/no-such-binding/);
  });
});

describe('platform descriptors', () => {
  test('the bindings are exactly terminal-copy and sidebar-toggle (no new bindings)', () => {
    expect(Object.keys(KEY_BINDINGS).sort()).toEqual(['sidebar-toggle', 'terminal-copy']);
  });

  test('terminal-copy is native on macOS (the browser copy) and claimed elsewhere', () => {
    expect(bindingFor('terminal-copy', true)).toEqual({ hint: '⌘C', native: true });
    const other = bindingFor('terminal-copy', false);
    expect('matcher' in other && other.matcher).toBe(matcherFor('terminal-copy'));
  });

  // A hint a row prints must be a chord something answers to: the browser
  // (native) or the binding's own matcher.
  test.each([
    ['terminal-copy', false, { key: 'C', ctrlKey: true, shiftKey: true }],
    ['sidebar-toggle', true, { key: '\\', metaKey: true }],
    ['sidebar-toggle', false, { key: '\\', ctrlKey: true }],
  ])('%s (mac=%s) claims the chord it hints', (id, mac, init) => {
    const descriptor = bindingFor(id, mac);
    expect('matcher' in descriptor).toBe(true);
    if ('matcher' in descriptor) expect(descriptor.matcher(keyEvent(init))).toBe(true);
  });

  test('keyFor gives collapsible.js the bare sidebar key', () => {
    expect(keyFor('sidebar-toggle')).toBe('\\');
    expect(() => keyFor('terminal-copy')).toThrow();
  });
});

describe('terminal-copy matcher', () => {
  const matches = matcherFor('terminal-copy');

  test.each([['c'], ['C']])('claims Ctrl+Shift+C (key %s) on every platform', (key) => {
    for (const platform of ['MacIntel', 'Linux x86_64']) {
      stubPlatform(platform);
      expect(matches(keyEvent({ key, ctrlKey: true, shiftKey: true }))).toBe(true);
    }
  });

  test.each([
    ['plain Ctrl+C', { key: 'c', ctrlKey: true }],
    ['Cmd+C', { key: 'c', metaKey: true }],
    ['Cmd+Ctrl+Shift+C', { key: 'c', metaKey: true, ctrlKey: true, shiftKey: true }],
    ['Ctrl+Alt+Shift+C', { key: 'c', altKey: true, ctrlKey: true, shiftKey: true }],
    ['the keyup', { type: 'keyup', key: 'C', ctrlKey: true, shiftKey: true }],
    ['Ctrl+Shift+V', { key: 'v', ctrlKey: true, shiftKey: true }],
  ])('leaves %s alone', (_name, init) => {
    expect(matches(keyEvent(init))).toBe(false);
  });
});

describe('sidebar-toggle matcher', () => {
  const matches = matcherFor('sidebar-toggle');

  test.each([
    ['Cmd+\\', { key: '\\', metaKey: true }],
    ['Ctrl+\\', { key: '\\', ctrlKey: true }],
  ])('claims %s', (_name, init) => {
    expect(matches(keyEvent(init))).toBe(true);
  });

  test.each([
    ['bare \\', { key: '\\' }],
    ['Ctrl+Shift+\\', { key: '\\', ctrlKey: true, shiftKey: true }],
    ['Cmd+Alt+\\', { key: '\\', metaKey: true, altKey: true }],
    ['Ctrl+/', { key: '/', ctrlKey: true }],
  ])('leaves %s alone', (_name, init) => {
    expect(matches(keyEvent(init))).toBe(false);
  });
});

// @ts-check
/**
 * Unit tests for chat-voice-shortcut.js -- the key held to talk with voice
 * input's `key` input (`web.voice.key`):
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/chat-voice-shortcut.test.mjs
 *
 * What is pinned: the combinations a deployment may name are understood, with
 * the label and ARIA name the page shows; ones the chat cannot use are refused;
 * a key event matches only with exactly the combination's modifiers; and
 * letting go of the key or one of its modifiers ends a hold.
 */

import { describe, expect, test } from 'vitest';

import {
  DEFAULT_SHORTCUT,
  matchesShortcut,
  parseShortcut,
  releasesShortcut,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-voice-shortcut.js';

describe('parseShortcut', () => {
  test.each([
    ['Ctrl+Space', { key: ' ', ctrl: true, label: 'Ctrl + Space', aria: 'Control+Space' }],
    ['ctrl + space', { key: ' ', ctrl: true, label: 'Ctrl + Space' }],
    ['Alt+M', { key: 'M', alt: true, label: 'Alt + M', aria: 'Alt+M' }],
    ['F8', { key: 'F8', ctrl: false, label: 'F8', aria: 'F8' }],
    ['Shift+F2', { key: 'F2', shift: true, label: 'Shift + F2' }],
  ])('%j is understood', (text, expected) => {
    expect(parseShortcut(text)).toMatchObject(expected);
  });

  test.each([[''], [null], ['Ctrl'], ['Ctrl+A+B'], ['M'], ['Enter'], ['Ctrl+Escape'], ['Tab']])(
    '%j is refused',
    (text) => {
      expect(parseShortcut(text)).toBeNull();
    }
  );

  test('the default is Ctrl + Space', () => {
    expect(DEFAULT_SHORTCUT).toMatchObject({ key: ' ', ctrl: true, label: 'Ctrl + Space' });
  });
});

describe('matchesShortcut', () => {
  test.each([
    [{ key: ' ', ctrlKey: true }, true],
    [{ key: ' ' }, false],
    [{ key: ' ', ctrlKey: true, shiftKey: true }, false],
    [{ key: ' ', ctrlKey: true, altKey: true }, false],
    [{ key: 'k', ctrlKey: true }, false],
  ])('%j against Ctrl + Space is %s', (init, matches) => {
    expect(matchesShortcut(new KeyboardEvent('keydown', init), DEFAULT_SHORTCUT)).toBe(matches);
  });

  test('a letter matches whatever its case', () => {
    const altM =
      /** @type {import('../../../src/osprey/interfaces/web_terminal/static/js/chat-voice-shortcut.js').Shortcut} */ (
        parseShortcut('Alt+M')
      );
    expect(matchesShortcut(new KeyboardEvent('keydown', { key: 'm', altKey: true }), altM)).toBe(
      true
    );
  });
});

describe('releasesShortcut', () => {
  test.each([
    [' ', true],
    ['Control', true],
    ['Shift', false],
    ['a', false],
  ])('letting go of %j ends a Ctrl + Space hold: %s', (key, ends) => {
    expect(releasesShortcut(key, DEFAULT_SHORTCUT)).toBe(ends);
  });
});

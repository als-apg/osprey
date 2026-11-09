// @ts-check
/**
 * Unit tests for the terminal pane's boot-failure notice (terminal.js):
 *   npx vitest run tests/interfaces/web_terminal/terminal-boot-failure.test.mjs
 *
 * xterm.js and its addons are page globals loaded by <script> tags. When any
 * of them is missing, initTerminal() must say so in the pane and return, not
 * throw and leave the pane empty. The sentence names where the assets were
 * meant to come from: the CDN when the page carries data-vendor="cdn", the
 * image otherwise.
 *
 * Same isolation scheme as terminal-copy.test.mjs: fresh module per test,
 * xterm stubbed from _terminal-fake.mjs.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import { FakeTerminal, stubTerminalGlobals } from './_terminal-fake.mjs';

const XTERM_GLOBALS = ['Terminal', 'FitAddon', 'WebLinksAddon', 'ClipboardAddon'];

/** @returns {Promise<typeof import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js')>} */
async function loadTerminal() {
  return import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
}

/** Make every xterm global absent, as on a page whose scripts failed to load. */
function stubNoXterm() {
  for (const name of XTERM_GLOBALS) vi.stubGlobal(name, undefined);
}

/** @returns {string} */
function noticeText() {
  const notice = document.querySelector('#terminal-container .terminal-boot-failure');
  if (!notice) throw new Error('no .terminal-boot-failure in the pane');
  return notice.textContent ?? '';
}

beforeEach(() => {
  vi.resetModules();
  localStorage.clear();
  delete document.documentElement.dataset.vendor;
  document.body.innerHTML =
    '<div><div id="terminal-container"><span class="stale">stale</span></div></div>';
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  delete document.documentElement.dataset.vendor;
});

test('a CDN page names the CDN and offline: true', async () => {
  stubNoXterm();
  document.documentElement.dataset.vendor = 'cdn';
  const terminal = await loadTerminal();
  terminal.initTerminal('terminal-container');
  const text = noticeText();
  expect(text).toContain('did not load from the CDN');
  expect(text).toContain('offline: true');
  expect(document.querySelector('#terminal-container .stale')).toBeNull();
});

test('an offline page names the image', async () => {
  stubNoXterm();
  const terminal = await loadTerminal();
  terminal.initTerminal('terminal-container');
  expect(noticeText()).toContain('missing from this image');
});

test('a missing addon global renders the notice and constructs no terminal', async () => {
  stubTerminalGlobals();
  vi.stubGlobal('ClipboardAddon', undefined);
  const terminal = await loadTerminal();
  expect(() => terminal.initTerminal('terminal-container')).not.toThrow();
  expect(noticeText()).not.toBe('');
  expect(FakeTerminal.last).toBeNull();
});

test('all globals present renders no notice', async () => {
  stubTerminalGlobals();
  const terminal = await loadTerminal();
  terminal.initTerminal('terminal-container');
  expect(document.querySelector('#terminal-container .terminal-boot-failure')).toBeNull();
});

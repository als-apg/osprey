// @ts-check
/**
 * Unit tests for the Agent tile's pre-paint boot (agent-surface-boot.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-surface-boot.test.mjs
 *
 * The script is a classic <head> script, not a module, so each test runs its
 * shipped source against a prepared <html> and storage, the way the browser
 * does on load.
 */

import { test, expect, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const SOURCE = readFileSync(
  join(
    import.meta.dirname,
    '../../../src/osprey/interfaces/web_terminal/static/js/agent-surface-boot.js'
  ),
  'utf8'
);

const root = document.documentElement;

function runBoot() {
  new Function(SOURCE)();
}

beforeEach(() => {
  sessionStorage.clear();
  for (const name of ['data-agent-surface', 'data-terminal', 'data-osprey-storage-scope']) {
    root.removeAttribute(name);
  }
});

afterEach(() => {
  sessionStorage.clear();
});

test('a tab with no choice opens on the chat', () => {
  runBoot();
  expect(root.hasAttribute('data-agent-surface')).toBe(false);
});

test('a tab that opened the terminal reopens on it', () => {
  sessionStorage.setItem('osprey-agent-surface', 'terminal');
  runBoot();
  expect(root.getAttribute('data-agent-surface')).toBe('terminal');
});

test('a deployment without the terminal stays in chat whatever the tab stored', () => {
  sessionStorage.setItem('osprey-agent-surface', 'terminal');
  root.setAttribute('data-terminal', 'off');
  runBoot();
  expect(root.hasAttribute('data-agent-surface')).toBe(false);
});

test('a scoped mount reads only its own scope', () => {
  root.setAttribute('data-osprey-storage-scope', 'alice');
  sessionStorage.setItem('osprey-agent-surface', 'terminal');
  runBoot();
  expect(root.hasAttribute('data-agent-surface')).toBe(false);

  sessionStorage.setItem('osprey-agent-surface--alice', 'terminal');
  runBoot();
  expect(root.getAttribute('data-agent-surface')).toBe('terminal');
});

test('blocked storage opens on the chat instead of throwing', () => {
  const real = Object.getOwnPropertyDescriptor(window, 'sessionStorage');
  Object.defineProperty(window, 'sessionStorage', {
    configurable: true,
    get() {
      throw new Error('blocked');
    },
  });
  try {
    expect(runBoot).not.toThrow();
    expect(root.hasAttribute('data-agent-surface')).toBe(false);
  } finally {
    if (real) Object.defineProperty(window, 'sessionStorage', real);
  }
});

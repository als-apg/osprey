// @ts-check
/**
 * Unit tests for the Agent tile's surface: the state (agent-surface.js) and its
 * pre-paint boot script (agent-surface-boot.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-surface.test.mjs
 *
 * The tile shows the chat unless this tab opted into the terminal. The choice
 * lives in sessionStorage, so a reload keeps it and a new tab starts in chat.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const BOOT = readFileSync(join(import.meta.dirname, JS, 'agent-surface-boot.js'), 'utf8');

const html = document.documentElement;

/** Run the boot script the way a page load does. */
function reload() {
  html.removeAttribute('data-agent-surface');
  new Function(BOOT)();
}

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js')} */
let surface;

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  surface = await import(`${JS}/agent-surface.js`);
});

afterEach(() => {
  html.removeAttribute('data-agent-surface');
  html.removeAttribute('data-terminal');
  html.removeAttribute('data-osprey-storage-scope');
  sessionStorage.clear();
  vi.restoreAllMocks();
});

describe('the surface', () => {
  test('is the chat on a fresh page', () => {
    reload();
    expect(surface.isTerminalSurface()).toBe(false);
    expect(html.hasAttribute('data-agent-surface')).toBe(false);
  });

  test('opening the terminal stamps <html> and tells the listeners', () => {
    const listener = vi.fn();
    surface.onSurfaceChange(listener);

    surface.showTerminalSurface();

    expect(html.getAttribute('data-agent-surface')).toBe('terminal');
    expect(surface.isTerminalSurface()).toBe(true);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  test('going back clears both, and tells the listeners', () => {
    surface.showTerminalSurface();
    const listener = vi.fn();
    surface.onSurfaceChange(listener);

    surface.showChatSurface();

    expect(html.hasAttribute('data-agent-surface')).toBe(false);
    expect(sessionStorage.getItem('osprey-agent-surface')).toBeNull();
    expect(listener).toHaveBeenCalledTimes(1);
  });

  test('asking for the surface already showing says nothing', () => {
    const listener = vi.fn();
    surface.onSurfaceChange(listener);

    surface.showChatSurface();
    surface.showTerminalSurface();
    surface.showTerminalSurface();

    expect(listener).toHaveBeenCalledTimes(1);
  });
});

describe('across page loads', () => {
  test('a reload keeps the terminal', () => {
    surface.showTerminalSurface();
    reload();
    expect(html.getAttribute('data-agent-surface')).toBe('terminal');
  });

  test('a reload after going back stays in chat', () => {
    surface.showTerminalSurface();
    surface.showChatSurface();
    reload();
    expect(html.hasAttribute('data-agent-surface')).toBe(false);
  });

  test('a new tab starts in chat: its sessionStorage is empty', () => {
    surface.showTerminalSurface();
    sessionStorage.clear();
    reload();
    expect(html.hasAttribute('data-agent-surface')).toBe(false);
  });

  test('the choice is per render on a multi-user mount', () => {
    html.setAttribute('data-osprey-storage-scope', 'alice');
    surface.showTerminalSurface();
    expect(sessionStorage.getItem('osprey-agent-surface--alice')).toBe('terminal');
    reload();
    expect(html.getAttribute('data-agent-surface')).toBe('terminal');

    html.setAttribute('data-osprey-storage-scope', 'bob');
    reload();
    expect(html.hasAttribute('data-agent-surface')).toBe(false);
  });
});

describe('web.terminal: false', () => {
  beforeEach(() => {
    html.setAttribute('data-terminal', 'off');
  });

  test('opening the terminal does nothing', () => {
    const listener = vi.fn();
    surface.onSurfaceChange(listener);

    surface.showTerminalSurface();

    expect(html.hasAttribute('data-agent-surface')).toBe(false);
    expect(sessionStorage.length).toBe(0);
    expect(listener).not.toHaveBeenCalled();
  });

  test('a kept choice is not applied on load', () => {
    sessionStorage.setItem('osprey-agent-surface', 'terminal');
    reload();
    expect(html.hasAttribute('data-agent-surface')).toBe(false);
    expect(surface.isTerminalSurface()).toBe(false);
  });

  test('a stamped attribute does not count as the terminal surface', () => {
    html.setAttribute('data-agent-surface', 'terminal');
    expect(surface.isTerminalSurface()).toBe(false);
  });
});

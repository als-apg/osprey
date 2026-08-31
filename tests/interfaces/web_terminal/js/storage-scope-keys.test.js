// @ts-check
/**
 * Per-persona scoping of the web terminal's remaining shared-origin
 * localStorage keys:
 *   npx vitest run tests/interfaces/web_terminal/js/storage-scope-keys.test.js
 *
 * localStorage is origin-scoped, not path-scoped, so on a multi-user mount
 * (`/u/alice/`, `/u/bob/`) every persona shares one slot per bare key. The
 * server stamps `data-osprey-storage-scope` on <html>; each reader and writer
 * derives its key through storage-scope.js's `scopedStorageKey()`, and a scoped
 * read NEVER falls back to the bare key (that polluted slot is the whole bug).
 *
 * One test file per behaviour would repeat the same three-line arrangement five
 * times, so the modules whose storage paths are reachable without mocking the
 * API layer share this file. Two things they do NOT share are the observable:
 * each block asserts through the module's own public surface, never by reaching
 * into a private key helper.
 *
 * settings.js needs a stubbed `/health`, so it lives in the sibling
 * storage-scope-settings.test.js. dock-workspace.js's layout key is exercised
 * only by the Playwright suite (its storage path runs behind a live dockview
 * instance); its derivation is the same one-line `scopedStorageKey()` call as
 * every module here.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  setRailPosition,
  followThemeFamily,
  initRailThemeCoupling,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/rail-position.js';
import { maybeShowRailHint } from '../../../../src/osprey/interfaces/web_terminal/static/js/rail-hint.js';
import { clearStoredSessionId } from '../../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';
import {
  initAgentAttention,
  badgePanelActivity,
  clearBadge,
  restoreAgentBadges,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/panel-agent-attention.js';
import {
  openPalette,
  closePalette,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/palette.js';

const SCOPE_ATTR = 'data-osprey-storage-scope';

/** Serve this document as `user` would be served on a multi-user mount.
 * @param {string} user */
function serveAs(user) {
  document.documentElement.setAttribute(SCOPE_ATTR, user);
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute(SCOPE_ATTR);
  document.documentElement.removeAttribute('data-rail-position');
  document.body.innerHTML = '';
  document.body.className = '';
  window.history.replaceState(null, '', '/');
  // happy-dom does not implement scrollIntoView — the rail badge path calls it.
  Element.prototype.scrollIntoView = () => {};
});

afterEach(() => {
  document.documentElement.removeAttribute(SCOPE_ATTR);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('rail-position.js (the writer half of the rail-boot pair)', () => {
  const BASE = 'osprey-rail-position';

  test('a scoped flip pins bob, and only bob', () => {
    serveAs('bob');

    setRailPosition('top');

    expect(localStorage.getItem(`${BASE}--bob`)).toBe('top');
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test('an unscoped flip writes the legacy key unchanged', () => {
    setRailPosition('top');

    expect(localStorage.getItem(BASE)).toBe('top');
  });

  test("another persona's pin does not pin bob's rail against the theme coupling", () => {
    // The shared slot, written by whoever flipped the rail last. Read as bob's
    // own pin it would freeze his rail wherever that persona left it.
    localStorage.setItem(BASE, 'top');
    serveAs('bob');
    initRailThemeCoupling({ family_rail_defaults: { retro: 'top' } });
    document.documentElement.setAttribute('data-rail-position', 'left');

    followThemeFamily('retro');

    // Moved: bob has no pin of his own, so the family coupling applies.
    expect(document.documentElement.getAttribute('data-rail-position')).toBe('top');
  });

  test("bob's own pin still outranks the theme coupling", () => {
    localStorage.setItem(`${BASE}--bob`, 'left');
    serveAs('bob');
    initRailThemeCoupling({ family_rail_defaults: { retro: 'top' } });
    document.documentElement.setAttribute('data-rail-position', 'left');

    followThemeFamily('retro');

    expect(document.documentElement.getAttribute('data-rail-position')).toBe('left');
  });
});

describe('rail-hint.js', () => {
  const BASE = 'osprey-rail-hint-dismissed-v1';

  /** Mount the anchor the hint attaches to, rail in its (new) left column. */
  function mountRail() {
    document.body.innerHTML = '<div class="shell-body"><div class="panel-rail-region"></div></div>';
    document.documentElement.setAttribute('data-rail-position', 'left');
  }

  /** @param {string} selector */
  function click(selector) {
    const el = document.querySelector(selector);
    if (!(el instanceof HTMLElement)) throw new Error(`expected ${selector}`);
    el.click();
  }

  test('a persona still gets the one-time hint another persona dismissed', () => {
    localStorage.setItem(BASE, '1');
    serveAs('bob');
    mountRail();

    expect(maybeShowRailHint()).toBe(true);
  });

  test('dismissing marks only this persona as having seen it', () => {
    serveAs('bob');
    mountRail();
    maybeShowRailHint();

    click('.rail-hint-dismiss');

    expect(localStorage.getItem(`${BASE}--bob`)).toBe('1');
    expect(localStorage.getItem(BASE)).toBe(null);
    expect(maybeShowRailHint()).toBe(false);
  });

  test('unscoped serving keeps writing the legacy flag', () => {
    mountRail();
    maybeShowRailHint();

    click('.rail-hint-dismiss');

    expect(localStorage.getItem(BASE)).toBe('1');
  });
});

describe('terminal.js PTY session pointer', () => {
  const BASE = 'osprey-pty-session';

  // Read, write and clear all resolve the key through one module-private
  // helper, so pinning the clear path pins the derivation all three use. The
  // stakes on this key are the highest in the file: a session id replayed into
  // another persona's container would attach that persona's terminal to a PTY
  // that is not theirs. The READ half — a bare stale id must not become this
  // persona's auto-resume — needs initTerminal()'s xterm/WebSocket harness and
  // is pinned where that harness lives: terminal-resume.test.mjs ('per-persona
  // pointer scope'). test_logout_resume_browser.py proves the same round trip
  // in a real multi-user page.
  test('clearing removes this persona\'s pointer and leaves the shared slot alone', () => {
    localStorage.setItem(BASE, 'someone-elses-session');
    localStorage.setItem(`${BASE}--bob`, 'bobs-session');
    serveAs('bob');

    clearStoredSessionId();

    expect(localStorage.getItem(`${BASE}--bob`)).toBe(null);
    expect(localStorage.getItem(BASE)).toBe('someone-elses-session');
  });

  test('unscoped serving clears the legacy key unchanged', () => {
    localStorage.setItem(BASE, 'my-session');

    clearStoredSessionId();

    expect(localStorage.getItem(BASE)).toBe(null);
  });
});

describe('panel-agent-attention.js acknowledgements', () => {
  const PANEL = 'okf';

  /** Mount a rail carrying one entry the badge can anchor to. */
  function mountRail() {
    document.body.innerHTML =
      `<div id="rail"><button class="panel-rail-button" data-panel-id="${PANEL}"></button></div>`;
    const rail = document.getElementById('rail');
    if (!rail) throw new Error('expected the rail fixture');
    initAgentAttention(rail);
  }

  test('an ack is stored per persona — panel ids are shared, personas are not', () => {
    serveAs('bob');
    mountRail();
    expect(badgePanelActivity(PANEL, 42)).toBe(true);

    clearBadge(PANEL);

    expect(localStorage.getItem(`agent-ack:${PANEL}--bob`)).toBe('42');
    expect(localStorage.getItem(`agent-ack:${PANEL}`)).toBe(null);
  });

  test('unscoped serving keeps the legacy per-panel key', () => {
    mountRail();
    expect(badgePanelActivity(PANEL, 42)).toBe(true);

    clearBadge(PANEL);

    expect(localStorage.getItem(`agent-ack:${PANEL}`)).toBe('42');
  });

  // The read side lives in restoreAgentBadges(): an event at or below the
  // stored ack is skipped, anything newer is re-badged on reload.
  /** Stub the recent-activity endpoint with one panel event at `ts`.
   * @param {number} ts */
  function serveRecent(ts) {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({ events: [{ target: { kind: 'panel', panel: PANEL }, ts }] }),
    })));
  }

  function isBadged() {
    return document.querySelector('.panel-rail-button')?.classList.contains('agent-attention') ?? false;
  }

  test("another persona's acknowledgement does not silence bob's badge on reload", async () => {
    localStorage.setItem(`agent-ack:${PANEL}`, '42');
    serveAs('bob');
    mountRail();
    serveRecent(42);

    await restoreAgentBadges();

    expect(isBadged()).toBe(true);
  });

  test("bob's own acknowledgement does", async () => {
    localStorage.setItem(`agent-ack:${PANEL}--bob`, '42');
    serveAs('bob');
    mountRail();
    serveRecent(42);

    await restoreAgentBadges();

    expect(isBadged()).toBe(false);
  });
});

describe('palette.js recent commands', () => {
  const BASE = 'osprey-palette-recent-v1';
  /** The stable key of the fixture's one action row (group + label). */
  const RESTART = 'Actions\x1fRestart terminal';

  /** Flush pending microtasks (the palette's config fetch). */
  function flushMicro() {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  /** A deps bundle with one action and a config fetch that hits no network. */
  function makeDeps() {
    return {
      getHiddenPanels: () => [],
      getVisiblePanels: () => [],
      getPresets: () => [],
      showPanel: vi.fn(),
      focusPanel: vi.fn(),
      applyPreset: vi.fn(),
      revealSetting: vi.fn(),
      actions: [{ label: 'Restart terminal', run: vi.fn() }],
      fetchConfig: () => Promise.resolve({ sections: {} }),
    };
  }

  /** @param {string} value */
  function typeQuery(value) {
    const el = /** @type {HTMLInputElement} */ (document.querySelector('.command-palette-input'));
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function pressEnter() {
    document
      .querySelector('.command-palette-input')
      ?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
  }

  /** @returns {(string|null)[]} the visible section headings, in DOM order. */
  function headings() {
    return [...document.querySelectorAll('.command-palette-group-heading')].map((h) => h.textContent);
  }

  afterEach(() => {
    closePalette();
    document.body.innerHTML = '';
  });

  test('executing a command records it under this persona\'s key', async () => {
    serveAs('bob');
    openPalette(makeDeps());
    await flushMicro();

    typeQuery('restart terminal');
    pressEnter();

    expect(JSON.parse(localStorage.getItem(`${BASE}--bob`) || 'null')).toEqual([RESTART]);
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test("another persona's Recent list is never read", async () => {
    localStorage.setItem(BASE, JSON.stringify([RESTART]));
    serveAs('bob');

    openPalette(makeDeps());
    await flushMicro();

    expect(headings()).not.toContain('Recent');
  });

  test('this persona\'s own Recent list renders', async () => {
    localStorage.setItem(`${BASE}--bob`, JSON.stringify([RESTART]));
    serveAs('bob');

    openPalette(makeDeps());
    await flushMicro();

    expect(headings()[0]).toBe('Recent');
  });

  test('unscoped serving still reads and writes the legacy key', async () => {
    openPalette(makeDeps());
    await flushMicro();

    typeQuery('restart terminal');
    pressEnter();

    expect(JSON.parse(localStorage.getItem(BASE) || 'null')).toEqual([RESTART]);
  });
});

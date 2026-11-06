// @ts-check
/**
 * Per-user scoping of the web terminal's remaining shared-origin
 * localStorage keys:
 *   npx vitest run tests/interfaces/web_terminal/js/storage-scope-keys.test.js
 *
 * localStorage is origin-scoped, not path-scoped, so on a multi-user mount
 * (`/u/alice/`, `/u/bob/`) every user shares one slot per bare key. The
 * server stamps `data-osprey-storage-scope` on <html>; each reader and writer
 * derives its key through storage-scope.js's `scopedStorageKey()`, and a scoped
 * read NEVER falls back to the bare key (that polluted slot is the whole bug).
 *
 * One test file per behaviour would repeat the same three-line arrangement
 * many times, so the modules whose storage keys are pinned for scoping share
 * this file. Two things they do NOT share are the observable: each block
 * asserts through the module's own public surface, never by reaching into a
 * private key helper. sidebar-artifacts.js is the one block that needs its
 * collaborators stubbed (the hub's API, the session, the panel manager); it
 * mocks them for that block alone (vi.doMock + a fresh import).
 *
 * settings.js needs a stubbed `/health`, so it lives in the sibling
 * storage-scope-settings.test.js. The session pointer's key belongs to
 * session-pointer.js and is pinned in session-pointer.test.mjs. dock-workspace.js's
 * layout key is exercised only by the Playwright suite (its storage path runs
 * behind a live dockview instance); its derivation is the same one-line
 * `scopedStorageKey()` call as every module here.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { initCollapsible } from '/design-system/js/collapsible.js';
import {
  initSidebar,
  OPEN_LISTS_STORAGE_KEY,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/sidebar.js';
import {
  applyTourConfig,
  INVITE_DELAY_MS,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/tour.js';
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
  document.documentElement.removeAttribute('data-sidebar');
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

describe('collapsible.js (the writer half of the sidebar-boot pair)', () => {
  const BASE = 'osprey-sidebar';
  // sidebar-boot.js is a classic pre-paint script, not a module: run its
  // source as the page would, against this document.
  const BOOT = readFileSync(
    join(
      import.meta.dirname,
      '../../../../src/osprey/interfaces/design_system/static/js/sidebar-boot.js'
    ),
    'utf8'
  );

  /** Collapse the sidebar through its labels toggle. */
  function collapse() {
    document.body.innerHTML = '<button id="sidebar-toggle" type="button"></button>';
    const toggle = document.getElementById('sidebar-toggle');
    initCollapsible({ storageKey: BASE, toggle });
    toggle?.click();
  }

  /** Reload: what sidebar-boot.js stamps on a fresh <html>.
   * @returns {string | null} */
  function bootedWidth() {
    document.documentElement.removeAttribute('data-sidebar');
    new Function(BOOT)();
    return document.documentElement.getAttribute('data-sidebar');
  }

  test('a scoped collapse is stored for bob, and only bob', () => {
    serveAs('bob');

    collapse();

    expect(localStorage.getItem(`${BASE}--bob`)).toBe('collapsed');
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test('an unscoped collapse writes the legacy key unchanged', () => {
    collapse();

    expect(localStorage.getItem(BASE)).toBe('collapsed');
  });

  test("sidebar-boot.js reads bob's own width back before first paint", () => {
    serveAs('bob');
    collapse();

    expect(bootedWidth()).toBe('collapsed');
  });

  test("another user's collapse does not collapse bob's sidebar", () => {
    // The shared slot, written by whoever collapsed last. Read as bob's own
    // it would hand him that user's width on every reload.
    localStorage.setItem(BASE, 'collapsed');
    serveAs('bob');

    expect(bootedWidth()).toBe(null);
  });
});

describe('sidebar.js open lists', () => {
  const BASE = 'osprey-sidebar-open';

  /** Mount the Agent row with its session tree and wire the sidebar. */
  function mountSidebar() {
    document.body.innerHTML = `
      <nav id="sidebar">
        <div class="sidebar-row" id="sidebar-agent-wrap">
          <button class="osprey-nav-row parent" id="sidebar-agent-row" type="button"></button>
          <div class="sidebar-kids" id="sidebar-session-tree" data-session-tree></div>
        </div>
      </nav>`;
    initSidebar();
  }

  /** @returns {boolean} whether the session tree shows */
  function isOpen() {
    const tree = document.getElementById('sidebar-session-tree');
    if (!tree) throw new Error('expected #sidebar-session-tree');
    return !tree.hidden;
  }

  /** Toggle the tree from the Agent row's keyboard: ArrowLeft closes it, ArrowRight opens it. */
  function toggleTree() {
    const key = isOpen() ? 'ArrowLeft' : 'ArrowRight';
    document
      .querySelector('#sidebar-agent-row')
      ?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
  }

  test('the key is the one this block pins', () => {
    expect(OPEN_LISTS_STORAGE_KEY).toBe(BASE);
  });

  test('closing the tree is stored for bob, and only bob', () => {
    serveAs('bob');
    mountSidebar();

    toggleTree();

    expect(JSON.parse(localStorage.getItem(`${BASE}--bob`) || 'null')).toEqual([]);
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test("another user's open lists are never read", () => {
    localStorage.setItem(BASE, JSON.stringify([]));
    serveAs('bob');

    mountSidebar();

    // bob has none of his own, so he gets the first-visit default.
    expect(isOpen()).toBe(true);
  });

  test("bob's own open lists are", () => {
    localStorage.setItem(`${BASE}--bob`, JSON.stringify([]));
    serveAs('bob');

    mountSidebar();

    expect(isOpen()).toBe(false);
  });

  test('unscoped serving still reads and writes the legacy key', () => {
    localStorage.setItem(BASE, JSON.stringify([]));
    mountSidebar();
    expect(isOpen()).toBe(false);

    toggleTree();

    expect(JSON.parse(localStorage.getItem(BASE) || 'null')).toEqual(['agent']);
  });
});

describe('sidebar-artifacts.js tree mode', () => {
  const BASE = 'osprey-sidebar-artifacts-view';
  const JS = '../../../../src/osprey/interfaces/web_terminal/static/js';
  /** The collaborators stubbed for this block only. */
  const STUBBED = {
    [`${JS}/api.js`]: () => ({
      withPrefix: (/** @type {string} */ path) => path,
      createEventSource: () => ({ stop() {} }),
      apiRequest: async () => null,
    }),
    [`${JS}/session-model.js`]: () => ({
      initSessionModel: async () => {},
      resetSessionModel: () => {},
      subscribe: () => () => {},
      rows: () => [],
      activeRow: () => null,
      pick: () => null,
      setPick: () => {},
      hasMore: () => false,
      recordState: () => 'complete',
      refresh: () => {},
    }),
    [`${JS}/dock-workspace.js`]: () => ({
      isTerminalPanelOpen: () => true,
      openTerminalPanel: () => {},
    }),
    [`${JS}/tile-header-contrib.js`]: () => ({ contributeLocal: () => {} }),
    [`${JS}/panel-iframe-sync.js`]: () => ({
      sendArtifactSelectToIframe: () => {},
      sendArtifactActionToIframe: () => {},
      sendArtifactChangedToIframe: () => {},
    }),
    [`${JS}/panel-manager.js`]: () => ({ activateTab: () => {} }),
    [`${JS}/panel-placement.js`]: () => ({ hasDockedTile: () => true }),
    [`${JS}/panel-menu-policy.js`]: () => ({ setServiceMenuExtras: () => {} }),
    '/design-system/js/theme-manager.js': () => ({ getTheme: () => 'dark' }),
  };

  /** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js')} */
  let mod;

  beforeEach(async () => {
    vi.resetModules();
    for (const [path, factory] of Object.entries(STUBBED)) vi.doMock(path, factory);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ artifacts: [], total: 0, next_cursor: null }),
      }))
    );
    document.body.innerHTML = `<div data-session-tree></div>`;
    mod = await import(`${JS}/sidebar-artifacts.js`);
  });

  afterEach(() => {
    mod.resetSidebarArtifacts();
    for (const path of Object.keys(STUBBED)) vi.doUnmock(path);
    vi.unstubAllGlobals();
  });

  test('a mode is stored for bob, and only bob', () => {
    serveAs('bob');
    mod.initSidebarArtifacts();

    mod.setTreeView('kind');

    expect(mod.treeView()).toBe('kind');
    expect(localStorage.getItem(`${BASE}--bob`)).toBe('kind');
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test("another user's mode is never read", () => {
    localStorage.setItem(BASE, 'kind');
    serveAs('bob');

    mod.initSidebarArtifacts();

    expect(mod.treeView()).toBe('activity');
  });

  test("bob's own mode is", () => {
    localStorage.setItem(`${BASE}--bob`, 'kind');
    serveAs('bob');

    mod.initSidebarArtifacts();

    expect(mod.treeView()).toBe('kind');
  });

  test('unscoped serving still writes the legacy key', () => {
    mod.initSidebarArtifacts();

    mod.setTreeView('kind');

    expect(localStorage.getItem(BASE)).toBe('kind');
  });
});

describe('tour.js', () => {
  const BASE = 'osprey-tour-dismissed-v2';

  /** Mount the one anchor the tour minimally needs (the terminal card). */
  function mountShell() {
    document.body.innerHTML = '<div class="terminal-card"></div>';
  }

  /** Arm the automatic invite and let its delay elapse. */
  function inviteNow() {
    applyTourConfig({ tour: { policy: 'once' } });
    vi.advanceTimersByTime(INVITE_DELAY_MS + 1);
  }

  /** @param {string} selector */
  function click(selector) {
    const el = document.querySelector(selector);
    if (!(el instanceof HTMLElement)) throw new Error(`expected ${selector}`);
    el.click();
  }

  /** Close whatever the tour has open so the next test starts clean. */
  function escapeTour() {
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
  }

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    escapeTour();
    vi.useRealTimers();
  });

  /** Dismiss via the invite's "Don't show this again" checkbox + Skip. */
  function dismissInvite() {
    const box = document.querySelector('.tour-remember input');
    if (!(box instanceof HTMLInputElement)) throw new Error('expected the remember checkbox');
    box.checked = true;
    click('.tour-invite-actions .tour-btn:not(.primary)');
  }

  test('a user still gets the invite another user dismissed', () => {
    localStorage.setItem(BASE, '1');
    serveAs('bob');
    mountShell();

    inviteNow();

    expect(document.querySelector('.tour-card')).not.toBe(null);
  });

  test('dismissing marks only this user as having seen it', () => {
    serveAs('bob');
    mountShell();
    inviteNow();

    dismissInvite();

    expect(localStorage.getItem(`${BASE}--bob`)).toBe('1');
    expect(localStorage.getItem(BASE)).toBe(null);
    inviteNow();
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('unscoped serving keeps writing the legacy flag', () => {
    mountShell();
    inviteNow();

    dismissInvite();

    expect(localStorage.getItem(BASE)).toBe('1');
  });
});

describe('panel-agent-attention.js acknowledgements', () => {
  const PANEL = 'okf';

  /** Mount a rail carrying one entry the badge can anchor to. */
  function mountRail() {
    document.body.innerHTML = `<div id="rail"><button class="panel-rail-button" data-panel-id="${PANEL}"></button></div>`;
    const rail = document.getElementById('rail');
    if (!rail) throw new Error('expected the rail fixture');
    initAgentAttention(rail);
  }

  test('an ack is stored per user — panel ids are shared, users are not', () => {
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
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ events: [{ target: { kind: 'panel', panel: PANEL }, ts }] }),
      }))
    );
  }

  function isBadged() {
    return (
      document.querySelector('.panel-rail-button')?.classList.contains('agent-attention') ?? false
    );
  }

  test("another user's acknowledgement does not silence bob's badge on reload", async () => {
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

  // The palette's one network read, answered here so nothing dials out.
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (!url.endsWith('/api/config')) throw new Error(`unstubbed fetch: ${url}`);
        return { ok: true, status: 200, json: async () => ({ sections: {} }) };
      })
    );
  });

  /** A deps bundle with one action. */
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
      ?.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })
      );
  }

  /** @returns {(string|null)[]} the visible section headings, in DOM order. */
  function headings() {
    return [...document.querySelectorAll('.command-palette-group-heading')].map(
      (h) => h.textContent
    );
  }

  afterEach(() => {
    closePalette();
    document.body.innerHTML = '';
  });

  test("executing a command records it under this user's key", async () => {
    serveAs('bob');
    openPalette(makeDeps());
    await flushMicro();

    typeQuery('restart terminal');
    pressEnter();

    expect(JSON.parse(localStorage.getItem(`${BASE}--bob`) || 'null')).toEqual([RESTART]);
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test("another user's Recent list is never read", async () => {
    localStorage.setItem(BASE, JSON.stringify([RESTART]));
    serveAs('bob');

    openPalette(makeDeps());
    await flushMicro();

    expect(headings()).not.toContain('Recent');
  });

  test("this user's own Recent list renders", async () => {
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

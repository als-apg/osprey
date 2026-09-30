/**
 * Unit tests for the command-palette overlay (palette.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/palette.test.mjs
 *
 * happy-dom weakly supports scrollIntoView/focus, so these tests assert only
 * CLASS/ATTRIBUTE state and never scroll position. document.activeElement is
 * asserted in one place only — the focus-on-open block, which checks WHEN
 * focus() is called relative to the reveal frame, not browser focus fidelity.
 * scrollIntoView is stubbed to a no-op and the one network call the palette
 * makes, GET /api/config, is answered by the fetch stub below.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply to it.
 */

import { test, expect, describe, beforeEach, beforeAll, afterEach, vi } from 'vitest';

import {
  openPalette,
  closePalette,
  isOpen,
} from '../../../src/osprey/interfaces/web_terminal/static/js/palette.js';

// palette-boot.js's Cmd/Ctrl+K handler must bail while the feedback modal is
// open (see the "keyboard arbitration" describe block at the bottom of this
// file). feedback-modal.js is mocked rather than imported for real — the
// arbitration palette-boot owes it is "is the feedback modal open right now",
// and driving the real modal would drag its focus-trap/render graph into this
// suite. Mirrors the same pattern feedback-modal.test.mjs uses for palette.js.
const feedbackModalState = vi.hoisted(() => ({ open: false }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/feedback-modal.js', () => ({
  isFeedbackModalOpen: () => feedbackModalState.open,
}));

/** A sections fixture: three leaf dot-keys under two sections. */
const SECTIONS = {
  control_system: { type: 'epics', writes_enabled: true },
  ui: { theme: 'dark' },
};

/** @param {any} body */
function configResponse(body) {
  return { ok: true, status: 200, statusText: 'OK', json: async () => body };
}

/** The palette's own config read, answered with SECTIONS unless a test holds or fails it. */
const defaultConfigAnswer = () => Promise.resolve(configResponse({ sections: SECTIONS }));
/** @type {() => Promise<any>} */
let answerConfig = defaultConfigAnswer;

/**
 * Hold the next config reads until the test resolves them.
 * @returns {(body: any) => void} resolves every held read with `body`
 */
function holdConfig() {
  /** @type {(value: any) => void} */
  let release = () => {};
  const pending = new Promise((r) => {
    release = r;
  });
  answerConfig = () => pending.then(configResponse);
  return release;
}

// Two requests reach this environment: the palette's GET /api/config, and the
// operator's bar layout, which bar-sync.js (reached through palette-boot.js's
// import graph) GETs at import time — before the dynamic import below is
// evaluated. Any other URL is a dependency this file has not declared, and
// fails loudly.
vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
  if (url === '/api/config') return answerConfig();
  if (url !== '/api/bar-items') throw new Error(`unstubbed fetch: ${url}`);
  return {
    ok: true,
    status: 200,
    json: async () => ({
      version: 1,
      rev: 0,
      header: [],
      status: [],
      header_visible: true,
      status_visible: true,
    }),
  };
}));

const { initCommandPalette } = await import(
  '../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js'
);

/** Flush the requestAnimationFrame that adds the `.visible` class. */
function flushRaf() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
}

/** Flush pending microtasks (config-fetch resolution). */
function flushMicro() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Build a deps bundle with spied navigation callbacks. `revealSetting` is
 * present, so the palette reads /api/config (see `answerConfig`).
 * @param {Record<string, any>} [over]
 */
function makeDeps(over = {}) {
  return {
    getHiddenPanels: () => [{ id: 'ariel', label: 'ARIEL' }],
    getVisiblePanels: () => [{ id: 'lattice', label: 'Lattice' }],
    getPresets: () => [{ name: 'Focus', panels: ['ariel'] }],
    showPanel: vi.fn(),
    focusPanel: vi.fn(),
    applyPreset: vi.fn(),
    revealSetting: vi.fn(),
    actions: [{ label: 'Restart terminal', run: vi.fn() }],
    ...over,
  };
}

/** @returns {HTMLElement} */
function overlay() {
  return /** @type {HTMLElement} */ (document.querySelector('.command-palette-overlay'));
}

/** @returns {HTMLInputElement} */
function input() {
  return /** @type {HTMLInputElement} */ (document.querySelector('.command-palette-input'));
}

/** Type into the palette input and fire the input event.
 * @param {string} value */
function typeQuery(value) {
  const el = input();
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Dispatch a bubbling keydown from the input (so capture-phase nav sees it).
 * @param {string} key */
function pressKey(key) {
  input().dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

/** localStorage key holding the Recent list (mirrors palette.js). */
const RECENT_KEY = 'osprey-palette-recent-v1';

/** @returns {string[]} the stored Recent keys, most-recent first. */
function storedRecent() {
  return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
}

/** @returns {string[]} the visible section headings, in DOM order. */
function headings() {
  return [...document.querySelectorAll('.command-palette-group-heading')].map((h) => h.textContent);
}

/** @returns {string[]} the visible row labels, in DOM order. */
function rowLabels() {
  return [...document.querySelectorAll('.command-palette-item-label')].map((l) => l.textContent);
}

beforeEach(() => {
  answerConfig = defaultConfigAnswer;
  closePalette();
  document.body.innerHTML = '';
  localStorage.clear();
  // happy-dom does not implement scrollIntoView — stub to a no-op.
  Element.prototype.scrollIntoView = () => {};
});

afterEach(() => {
  closePalette();
  vi.restoreAllMocks();
});

describe('open / close lifecycle', () => {
  test('open creates a visible overlay; close tears it down', async () => {
    openPalette(makeDeps());
    expect(isOpen()).toBe(true);
    expect(overlay()).toBeTruthy();
    await flushRaf();
    expect(overlay().classList.contains('visible')).toBe(true);

    closePalette();
    expect(isOpen()).toBe(false);
    expect(overlay().classList.contains('visible')).toBe(false);
    // transitionend drives node removal (no real transitions in happy-dom).
    overlay().dispatchEvent(new Event('transitionend'));
    expect(document.querySelector('.command-palette-overlay')).toBeNull();
  });

  test('opening twice does not create a second overlay', () => {
    openPalette(makeDeps());
    openPalette(makeDeps());
    expect(document.querySelectorAll('.command-palette-overlay').length).toBe(1);
  });
});

describe('filtering + highlight', () => {
  test("the highlight marks the query's characters inside the label", async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('control');
    const row = [...document.querySelectorAll('.command-palette-item')].find(
      (r) => r.querySelector('.command-palette-item-label')?.textContent === 'control_system.type',
    );
    if (!row) throw new Error('no control_system.type row');
    const marks = [...row.querySelectorAll('.command-palette-match')];
    expect(marks.map((m) => m.textContent).join('')).toBe('control');
    expect(marks.every((m) => m.closest('.command-palette-item-label'))).toBe(true);
  });

  test('a row found through its searchText alone renders its label un-highlighted', async () => {
    // "optics" is a Lattice synonym carried only in searchText: it ranks the
    // row, but no character of "Focus Lattice" is marked for it.
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('optics');
    const labels = [...document.querySelectorAll('.command-palette-item-label')];
    const lattice = labels.find((l) => l.textContent === 'Focus Lattice');
    if (!lattice) throw new Error('no Focus Lattice row');
    expect(lattice.querySelectorAll('.command-palette-match')).toHaveLength(0);
  });

  test('a non-matching query renders the empty state', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('zzzzznope');
    expect(document.querySelector('.command-palette-empty')).toBeTruthy();
    expect(document.querySelectorAll('.command-palette-item').length).toBe(0);
  });

  test('group order Settings -> Panels -> Layouts -> Actions is preserved', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('');
    const headings = [...document.querySelectorAll('.command-palette-group-heading')].map(
      (h) => h.textContent,
    );
    expect(headings).toEqual(['Settings', 'Panels', 'Layouts', 'Actions']);
  });
});

describe('keyboard navigation', () => {
  test('Arrow keys move the active row and skip status rows', async () => {
    // Loading state: a status decoration coexists with panel/layout/action rows.
    const resolveConfig = holdConfig();
    openPalette(makeDeps());
    typeQuery('');

    const status = /** @type {HTMLElement} */ (document.querySelector('.command-palette-status'));
    expect(status).toBeTruthy();
    expect(status.getAttribute('role')).toBeNull();

    const options = () => [...document.querySelectorAll('[role="option"]')];
    expect(options().length).toBeGreaterThan(0);

    pressKey('ArrowDown');
    const active1 = /** @type {HTMLElement} */ (document.querySelector('.command-palette-item--active'));
    expect(active1).toBeTruthy();
    expect(active1.getAttribute('aria-selected')).toBe('true');
    // aria-activedescendant always points at a real option id (never the status row).
    const ad1 = input().getAttribute('aria-activedescendant');
    expect(options().some((o) => o.id === ad1)).toBe(true);

    pressKey('ArrowUp');
    const ad2 = input().getAttribute('aria-activedescendant');
    expect(options().some((o) => o.id === ad2)).toBe(true);
    expect(document.querySelectorAll('.command-palette-item--active').length).toBe(1);

    resolveConfig({ sections: SECTIONS });
  });

  test('Enter runs the active item and closes', async () => {
    const run = vi.fn();
    openPalette(makeDeps({ actions: [{ label: 'Restart terminal', run }] }));
    await flushMicro();
    typeQuery('restart terminal');
    pressKey('Enter');
    expect(run).toHaveBeenCalledTimes(1);
    expect(isOpen()).toBe(false);
  });

  test('Escape closes and is handled in capture phase (stops propagation)', async () => {
    const bubbleSpy = vi.fn();
    document.addEventListener('keydown', bubbleSpy); // bubble-phase document listener
    try {
      openPalette(makeDeps());
      await flushMicro();
      pressKey('Escape');
      expect(isOpen()).toBe(false);
      // Capture-phase palette handler stopped propagation before the bubble listener.
      expect(bubbleSpy).not.toHaveBeenCalled();
    } finally {
      document.removeEventListener('keydown', bubbleSpy);
    }
  });
});

describe('concurrent config fetch', () => {
  test('pending -> loading status; resolved -> settings items appear', async () => {
    const resolveConfig = holdConfig();
    openPalette(makeDeps());
    typeQuery('control');

    // While pending, the Settings group shows the loading decoration.
    const status = /** @type {HTMLElement} */ (document.querySelector('.command-palette-status'));
    expect(status).toBeTruthy();
    expect(status.textContent).toMatch(/loading/i);

    resolveConfig({ sections: SECTIONS });
    await flushMicro();

    // Status row gone; real settings items now match the query.
    expect(document.querySelector('.command-palette-status')).toBeNull();
    const labels = [...document.querySelectorAll('.command-palette-item-label')].map(
      (l) => l.textContent,
    );
    expect(labels.some((t) => t.includes('control_system'))).toBe(true);
  });

  test('a rejecting fetch yields the unavailable status row', async () => {
    answerConfig = () => Promise.reject(new Error('boom'));
    openPalette(makeDeps());
    typeQuery('control');
    await flushMicro();
    const status = /** @type {HTMLElement} */ (document.querySelector('.command-palette-status'));
    expect(status).toBeTruthy();
    expect(status.textContent).toMatch(/unavailable/i);
  });
});

describe('focus on open', () => {
  test('the search input is focused once the overlay becomes visible', async () => {
    openPalette(makeDeps());
    // Focus is deferred to the reveal frame: a real browser makes focus() a
    // no-op while the overlay subtree is still `visibility: hidden`.
    expect(document.activeElement).not.toBe(input());

    await flushRaf();

    expect(overlay().classList.contains('visible')).toBe(true);
    expect(document.activeElement).toBe(input());
  });

  test('open then close within one frame leaves the overlay hidden and focus put', async () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();

    openPalette(makeDeps());
    closePalette();
    await flushRaf();

    expect(isOpen()).toBe(false);
    expect(overlay().classList.contains('visible')).toBe(false);
    expect(document.activeElement).toBe(trigger);
  });
});

describe('recent commands', () => {
  /** The stable keys of two fixture rows (group + label, \x1f-joined). */
  const RESTART = 'Actions\x1fRestart terminal';
  const SHOW_ARIEL = 'Panels\x1fShow ARIEL';

  /** Click the row whose label reads exactly `label`.
   * @param {string} label */
  function clickRow(label) {
    const row = [...document.querySelectorAll('.command-palette-item')].find(
      (r) => r.querySelector('.command-palette-item-label')?.textContent === label,
    );
    if (!row) throw new Error(`no palette row labelled "${label}"`);
    /** @type {HTMLElement} */ (row).click();
  }

  /** Seed the stored Recent list directly.
   * @param {string[]} keys */
  function seedRecent(keys) {
    localStorage.setItem(RECENT_KEY, JSON.stringify(keys));
  }

  test('the Enter path records the executed item key', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('restart terminal');
    pressKey('Enter');
    expect(storedRecent()).toEqual([RESTART]);
  });

  test('the click path records the same plain key', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('show ariel');
    clickRow('Show ARIEL');
    expect(storedRecent()).toEqual([SHOW_ARIEL]);
  });

  test('the list is capped at 5, most-recent first', async () => {
    seedRecent(['k1', 'k2', 'k3', 'k4', 'k5']);
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('restart terminal');
    pressKey('Enter');
    expect(storedRecent()).toEqual([RESTART, 'k1', 'k2', 'k3', 'k4']);
  });

  test('re-executing an item moves it to the front instead of duplicating it', async () => {
    seedRecent(['k1', RESTART, 'k2']);
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('restart terminal');
    pressKey('Enter');
    expect(storedRecent()).toEqual([RESTART, 'k1', 'k2']);
  });

  test('an empty query renders Recent first, in stored order', async () => {
    seedRecent([RESTART, SHOW_ARIEL]);
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('');

    expect(headings()).toEqual(['Recent', 'Settings', 'Panels', 'Layouts', 'Actions']);
    expect(rowLabels().slice(0, 2)).toEqual(['Restart terminal', 'Show ARIEL']);
  });

  test('Recent is hidden as soon as the query is non-empty', async () => {
    seedRecent([RESTART]);
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('restart');
    expect(headings()).not.toContain('Recent');
  });

  test('stored keys with no live registry match are skipped silently', async () => {
    seedRecent(['Panels\x1fShow GONE', RESTART]);
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('');

    expect(headings()[0]).toBe('Recent');
    // Only the surviving key renders, so the label appears exactly twice: once
    // in Recent, once in its home group.
    expect(rowLabels()[0]).toBe('Restart terminal');
    expect(rowLabels().filter((l) => l === 'Restart terminal').length).toBe(2);
  });

  test('an all-stale list renders no Recent section at all', async () => {
    seedRecent(['Panels\x1fShow GONE', 'Actions\x1fVanished']);
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('');
    expect(headings()).toEqual(['Settings', 'Panels', 'Layouts', 'Actions']);
  });

  test('selection stays on the home-group row when Recent duplicates it', async () => {
    seedRecent([RESTART]);
    const resolveConfig = holdConfig();
    openPalette(makeDeps());
    typeQuery('');

    const options = () => [...document.querySelectorAll('[role="option"]')];
    const activeIdx = () => options().findIndex(
      (o) => o.classList.contains('command-palette-item--active'),
    );

    // Wrap backwards from the first row onto the LAST one — the Actions group's
    // own "Restart terminal", the twin of the Recent row at index 0.
    pressKey('ArrowUp');
    expect(activeIdx()).toBe(options().length - 1);

    resolveConfig({ sections: SECTIONS });
    await flushMicro();

    // The mid-open re-render must not snap the selection up to the Recent
    // duplicate: namespaced nav keys keep the two rows distinct.
    expect(activeIdx()).toBe(options().length - 1);
    expect(activeIdx()).not.toBe(0);
    const active = options()[activeIdx()];
    expect(active.querySelector('.command-palette-item-label')?.textContent).toBe(
      'Restart terminal',
    );
  });

  test('blocked storage disables Recent without breaking execution', async () => {
    // A private-mode-style storage that throws on every access. Swapping the
    // global (rather than spying on Storage.prototype, which happy-dom's
    // instances bypass) is what actually reaches the module's bare
    // `localStorage` reference.
    const blocked = {
      getItem() {
        throw new Error('storage blocked');
      },
      setItem() {
        throw new Error('storage blocked');
      },
    };
    const real = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    Object.defineProperty(globalThis, 'localStorage', {
      value: blocked,
      configurable: true,
      writable: true,
    });

    try {
      const run = vi.fn();
      openPalette(makeDeps({ actions: [{ label: 'Restart terminal', run }] }));
      await flushMicro();
      typeQuery('');
      expect(headings()).toEqual(['Settings', 'Panels', 'Layouts', 'Actions']);

      typeQuery('restart terminal');
      pressKey('Enter');
      expect(run).toHaveBeenCalledTimes(1);
      expect(isOpen()).toBe(false);
    } finally {
      if (real) Object.defineProperty(globalThis, 'localStorage', real);
    }
  });
});

describe('keyboard arbitration (palette-boot.js)', () => {
  /**
   * Dispatch a keydown from a temporary `<body>` child and report whether it
   * reached that node.
   *
   * The witness has to sit DOWNSTREAM of `document`: palette-boot's hotkey
   * handler is a capture-phase listener on `document`, and `stopPropagation()`
   * stops the event reaching other NODES, never other listeners on the same
   * node — so a second listener on `document` would fire regardless of
   * whether the handler consumed the event. A listener on a body child is the
   * honest stand-in (same pattern as feedback-modal.test.mjs's `pressKey`).
   *
   * @param {string} key
   * @param {KeyboardEventInit} [init]
   * @returns {{stopped: boolean, defaultPrevented: boolean}}
   */
  function pressGlobalKey(key, init = {}) {
    const target = document.createElement('div');
    document.body.appendChild(target);
    let seen = false;
    target.addEventListener('keydown', () => {
      seen = true;
    });
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    target.dispatchEvent(event);
    target.remove();
    return { stopped: !seen, defaultPrevented: event.defaultPrevented };
  }

  beforeAll(() => {
    // Wires the capture-phase document listener once. Calling it per-test
    // would stack duplicate listeners on `document` for the rest of the file.
    initCommandPalette();
  });

  beforeEach(() => {
    closePalette();
    document.body.innerHTML = '';
    feedbackModalState.open = false;
  });

  afterEach(() => {
    closePalette();
    feedbackModalState.open = false;
  });

  test('arbitration: Ctrl+K is ignored while the feedback modal is open', () => {
    feedbackModalState.open = true;

    const result = pressGlobalKey('k', { ctrlKey: true });

    expect(result.defaultPrevented).toBe(false);
    expect(result.stopped).toBe(false);
    expect(isOpen()).toBe(false);
  });

  test('arbitration: Ctrl+K opens the palette again once the feedback modal closes', () => {
    feedbackModalState.open = false;

    const result = pressGlobalKey('k', { ctrlKey: true });

    expect(result.defaultPrevented).toBe(true);
    expect(result.stopped).toBe(true);
    expect(isOpen()).toBe(true);
  });
});

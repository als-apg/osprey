// @ts-check
/**
 * The terminal body's place menu (terminal-menu.js):
 *   npx vitest run tests/interfaces/web_terminal/terminal-menu.test.mjs
 *
 * Covered: the row set and order (Copy · Paste · Select all · Clear terminal ·
 * Switch to chat · Restart terminal, destructive and last); Copy disabled with
 * its reason while xterm has no selection, enabled and copying through the
 * event window's clipboard when it has one, with the key-bindings hint; Paste
 * through `term.paste`, disabled with its reason without the async clipboard
 * API or outside a secure context, and a notice when the browser refuses at
 * click time; Select all, Clear terminal, Switch to chat, Restart terminal;
 * Shift+right-click left to the browser; and the keyboard route — the menu
 * key composed into terminal.js's ONE xterm key handler opens exactly one
 * menu, is swallowed (not sent to the agent), and leaves Ctrl+Shift+C copying.
 *
 * context-menu.js and place-menu.js are the real modules. Module state is
 * isolated by vi.resetModules() + dynamic import per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { FakeTerminal, stubTerminalGlobals } from './_terminal-fake.mjs';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/terminal-menu.js';
const TERMINAL = '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';
const PLACES = '../../../src/osprey/interfaces/web_terminal/static/js/place-menu.js';
const MENU = '../../../src/osprey/interfaces/design_system/static/js/context-menu.js';
const PLATFORM = '../../../src/osprey/interfaces/design_system/static/js/platform.js';

/** @type {any} */ let terminalMenu;
/** @type {any} */ let pm;
/** @type {any} */ let cm;
/** @type {{ handleKey: (e: KeyboardEvent) => boolean, uninstall: () => void } | null} */
let installed = null;
/** @type {any} */ let term;
/** @type {{ onRestart: any, onSwitchToChat: any }} */ let spies;
/** @type {{ writeText: any, readText: any }} */ let clipboard;

/** Make a fake xterm with the API terminal-menu.js uses. */
function fakeTerm() {
  return {
    selection: '',
    hasSelection() {
      return this.selection !== '';
    },
    getSelection() {
      return this.selection;
    },
    selectAll: vi.fn(),
    clear: vi.fn(),
    paste: vi.fn(),
    focus: vi.fn(),
  };
}

/**
 * Give the page a clipboard (or none) and a secure-context flag.
 * @param {any} value
 * @param {boolean} [secure]
 */
function setClipboard(value, secure = true) {
  Object.defineProperty(window.navigator, 'clipboard', { value, configurable: true });
  Object.defineProperty(window, 'isSecureContext', { value: secure, configurable: true });
}

beforeEach(async () => {
  vi.resetModules();
  cm = await import(MENU);
  pm = await import(PLACES);
  terminalMenu = await import(MODULE);
  document.body.innerHTML = `
    <div id="terminal-container"><div class="xterm"><div class="xterm-screen"><canvas></canvas></div>
      <textarea class="xterm-helper-textarea"></textarea></div></div>
    <div class="elsewhere"></div>`;
  clipboard = {
    writeText: vi.fn(() => Promise.resolve()),
    readText: vi.fn(() => Promise.resolve('pasted text')),
  };
  setClipboard(clipboard);
  term = fakeTerm();
  spies = { onRestart: vi.fn(), onSwitchToChat: vi.fn() };
  installed = terminalMenu.installTerminalMenu({
    container: el('#terminal-container'),
    term,
    ...spies,
  });
  pm.installPlaceMenus(document);
});

afterEach(() => {
  installed?.uninstall();
  installed = null;
  cm.closeContextMenu();
  pm.resetPlaceMenus([document]);
  document.getElementById('undo-notice')?.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** @param {string} sel @returns {HTMLElement} */
function el(sel) {
  const node = document.querySelector(sel);
  if (!node) throw new Error(`no ${sel}`);
  return /** @type {HTMLElement} */ (node);
}

/**
 * Right-click `target`.
 * @param {Element} target
 * @param {MouseEventInit} [init]
 */
function rightClick(target, init = {}) {
  const e = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    button: 2,
    clientX: 40,
    clientY: 50,
    ...init,
  });
  target.dispatchEvent(e);
  return e;
}

function menus() {
  return document.querySelectorAll('.osprey-context-menu[role="menu"]');
}

function rowLabels() {
  return [...document.querySelectorAll('.osprey-context-menu [role="menuitem"]')].map(
    (b) => b.querySelector('.osprey-context-label')?.textContent
  );
}

/** @param {string} label @returns {HTMLElement} */
function row(label) {
  const found = [...document.querySelectorAll('.osprey-context-menu [role="menuitem"]')].find(
    (b) => b.querySelector('.osprey-context-label')?.textContent === label
  );
  if (!found) throw new Error(`no row ${label}; rows: ${rowLabels().join(', ')}`);
  return /** @type {HTMLElement} */ (found);
}

/** @param {string} label */
function clickRow(label) {
  row(label).click();
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('the terminal body menu', () => {
  test('a right-click inside xterm opens the Terminal menu with the rows in order', () => {
    const e = rightClick(el('canvas'));
    expect(e.defaultPrevented).toBe(true);
    expect(menus()).toHaveLength(1);
    expect(menus()[0].getAttribute('aria-label')).toBe('Terminal');
    expect(rowLabels()).toEqual([
      'Copy',
      'Paste',
      'Select all',
      'Clear terminal',
      'Switch to chat',
      'Restart terminal',
    ]);
  });

  test('Restart terminal is destructive and the only destructive row', () => {
    rightClick(el('canvas'));
    expect(row('Restart terminal').classList.contains('danger')).toBe(true);
    expect(document.querySelectorAll('.osprey-context-menu .danger')).toHaveLength(1);
  });

  test('a right-click outside the container opens nothing', () => {
    const e = rightClick(el('.elsewhere'));
    expect(e.defaultPrevented).toBe(false);
    expect(menus()).toHaveLength(0);
  });

  test('Shift+right-click leaves the browser its own menu', () => {
    const e = rightClick(el('canvas'), { shiftKey: true });
    expect(e.defaultPrevented).toBe(false);
    expect(menus()).toHaveLength(0);
  });

  test('a right-click on the xterm helper textarea still opens the menu', () => {
    const e = rightClick(el('.xterm-helper-textarea'));
    expect(e.defaultPrevented).toBe(true);
    expect(menus()).toHaveLength(1);
  });
});

describe('Copy', () => {
  test('is disabled with the reason while xterm has no selection', () => {
    rightClick(el('canvas'));
    const copy = row('Copy');
    expect(copy.getAttribute('aria-disabled')).toBe('true');
    expect(copy.title).toBe(terminalMenu.COPY_DISABLED_TIP);
    expect(copy.title).toMatch(/plain drag is the agent's selection/);
    expect(copy.title).toMatch(/Shift-drag \(Option-drag on macOS\)/);
    copy.click();
    expect(clipboard.writeText).not.toHaveBeenCalled();
  });

  test('copies the xterm selection through the clipboard when there is one', async () => {
    term.selection = 'ls -la';
    rightClick(el('canvas'));
    const copy = row('Copy');
    expect(copy.getAttribute('aria-disabled')).toBeNull();
    expect(copy.title).toBe('');
    copy.click();
    await flush();
    expect(clipboard.writeText).toHaveBeenCalledWith('ls -la');
  });

  test('carries the terminal-copy hint for the platform', async () => {
    const platform = await import(PLATFORM);
    const mac = platform.isMacPlatform();
    rightClick(el('canvas'));
    const hint = row('Copy').querySelector('.osprey-context-hint')?.textContent;
    expect(hint).toBe(mac ? '⌘C' : 'Ctrl+Shift+C');
  });

  test('a refused copy shows a notice and does not throw', async () => {
    term.selection = 'x';
    clipboard.writeText = vi.fn(() => Promise.reject(new Error('denied')));
    rightClick(el('canvas'));
    clickRow('Copy');
    await flush();
    expect(document.getElementById('undo-notice')?.textContent).toContain(
      terminalMenu.COPY_REFUSED_NOTICE
    );
  });
});

describe('Paste', () => {
  test('reads the clipboard and pastes it through term.paste', async () => {
    rightClick(el('canvas'));
    expect(row('Paste').getAttribute('aria-disabled')).toBeNull();
    clickRow('Paste');
    await flush();
    expect(clipboard.readText).toHaveBeenCalled();
    expect(term.paste).toHaveBeenCalledWith('pasted text');
  });

  test('is disabled with the reason when the async clipboard API is absent', () => {
    setClipboard(undefined);
    rightClick(el('canvas'));
    const paste = row('Paste');
    expect(paste.getAttribute('aria-disabled')).toBe('true');
    expect(paste.title).toBe(terminalMenu.PASTE_DISABLED_TIP);
  });

  test('is disabled with the reason outside a secure context', () => {
    setClipboard(clipboard, false);
    rightClick(el('canvas'));
    const paste = row('Paste');
    expect(paste.getAttribute('aria-disabled')).toBe('true');
    expect(paste.title).toBe(terminalMenu.PASTE_DISABLED_TIP);
    paste.click();
    expect(clipboard.readText).not.toHaveBeenCalled();
  });

  test('a refusal at click time shows a notice', async () => {
    clipboard.readText = vi.fn(() => Promise.reject(new Error('NotAllowedError')));
    rightClick(el('canvas'));
    clickRow('Paste');
    await flush();
    expect(term.paste).not.toHaveBeenCalled();
    const notice = document.getElementById('undo-notice');
    expect(notice?.textContent).toContain(terminalMenu.PASTE_REFUSED_NOTICE);
  });
});

describe('the other rows', () => {
  test('Select all selects everything in xterm', () => {
    rightClick(el('canvas'));
    clickRow('Select all');
    expect(term.selectAll).toHaveBeenCalledTimes(1);
  });

  test('Clear terminal clears xterm', () => {
    rightClick(el('canvas'));
    clickRow('Clear terminal');
    expect(term.clear).toHaveBeenCalledTimes(1);
  });

  test('Switch to chat shows the chat surface', () => {
    rightClick(el('canvas'));
    clickRow('Switch to chat');
    expect(spies.onSwitchToChat).toHaveBeenCalledTimes(1);
  });

  test('Restart terminal restarts the session', () => {
    rightClick(el('canvas'));
    clickRow('Restart terminal');
    expect(spies.onRestart).toHaveBeenCalledTimes(1);
  });

  test('uninstall removes the place', () => {
    installed?.uninstall();
    installed = null;
    const e = rightClick(el('canvas'));
    expect(e.defaultPrevented).toBe(false);
    expect(menus()).toHaveLength(0);
  });
});

describe('handleKey', () => {
  /** @param {KeyboardEventInit & { type?: string }} init */
  function key(init) {
    const { type = 'keydown', ...rest } = init;
    return new KeyboardEvent(type, { bubbles: true, cancelable: true, ...rest });
  }

  test('Shift+F10 opens the menu, marks the event handled and answers true', () => {
    const e = key({ key: 'F10', shiftKey: true });
    expect(installed?.handleKey(e)).toBe(true);
    expect(e.defaultPrevented).toBe(true);
    expect(menus()).toHaveLength(1);
    expect(rowLabels()[0]).toBe('Copy');
  });

  test('the ContextMenu key opens the menu', () => {
    expect(installed?.handleKey(key({ key: 'ContextMenu' }))).toBe(true);
    expect(menus()).toHaveLength(1);
  });

  test('keyup of the menu key is swallowed without opening a menu', () => {
    expect(installed?.handleKey(key({ type: 'keyup', key: 'F10', shiftKey: true }))).toBe(true);
    expect(menus()).toHaveLength(0);
  });

  test('other keys are not taken', () => {
    for (const init of [
      { key: 'F10' },
      { key: 'a' },
      { key: 'c', ctrlKey: true, shiftKey: true },
    ]) {
      const e = key(init);
      expect(installed?.handleKey(e)).toBe(false);
      expect(e.defaultPrevented).toBe(false);
    }
    expect(menus()).toHaveLength(0);
  });
});

describe('composed into terminal.js', () => {
  /** @type {any} */ let terminal;

  beforeEach(async () => {
    installed?.uninstall();
    installed = null;
    pm.resetPlaceMenus([document]);
    vi.resetModules();
    stubTerminalGlobals();
    vi.stubGlobal('requestAnimationFrame', () => 0);
    if (!document.fonts) {
      Object.defineProperty(document, 'fonts', {
        value: { ready: Promise.resolve() },
        configurable: true,
      });
    }
    cm = await import(MENU);
    pm = await import(PLACES);
    terminal = await import(TERMINAL);
    terminal.initTerminal('terminal-container');
    pm.installPlaceMenus(document);
  });

  /** @returns {FakeTerminal} */
  function xterm() {
    const t = FakeTerminal.last;
    if (!t) throw new Error('initTerminal() built no terminal');
    return t;
  }

  test('one key handler is attached and the menu key opens exactly one menu', () => {
    const handler = xterm().keyHandler;
    expect(handler).toBeTypeOf('function');
    const helper = el('#terminal-container');
    const e = new KeyboardEvent('keydown', {
      key: 'F10',
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    // xterm runs its custom handler from its own keydown listener, before the
    // event bubbles to the window where the dispatcher's keyboard route sits.
    helper.addEventListener('keydown', (ev) => {
      expect(handler?.(/** @type {KeyboardEvent} */ (ev))).toBe(false);
    });
    helper.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(true);
    expect(menus()).toHaveLength(1);
  });

  test('a right-click in the container opens the Terminal menu', () => {
    const e = rightClick(el('#terminal-container'));
    expect(e.defaultPrevented).toBe(true);
    expect(menus()[0].getAttribute('aria-label')).toBe('Terminal');
  });

  test('Ctrl+Shift+C still copies through the composed handler', async () => {
    xterm().selection = 'hello';
    const e = new KeyboardEvent('keydown', {
      key: 'C',
      code: 'KeyC',
      ctrlKey: true,
      shiftKey: true,
    });
    expect(xterm().keyHandler?.(e)).toBe(false);
    await flush();
    expect(clipboard.writeText).toHaveBeenCalledWith('hello');
    expect(menus()).toHaveLength(0);
  });

  test('plain keys pass through to xterm', () => {
    const e = new KeyboardEvent('keydown', { key: 'a' });
    expect(xterm().keyHandler?.(e)).toBe(true);
  });
});

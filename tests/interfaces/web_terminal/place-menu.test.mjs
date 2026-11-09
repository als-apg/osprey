// @ts-check
/**
 * The shell's place-menu dispatcher (place-menu.js):
 *   npx vitest run tests/interfaces/web_terminal/place-menu.test.mjs
 *
 * Places are registered with registerPlace and a document is wired with
 * installPlaceMenus. Covered: the pointer route (innermost place, declines,
 * already-handled events, zero rows), the keyboard route (Shift+F10 and the
 * menu key, never declining for Shift), per-document idempotency and a second
 * document standing for a pop-out window.
 *
 * context-menu.js is the real module. Module state is isolated by
 * vi.resetModules() + dynamic import per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/place-menu.js';
const MENU = '../../../src/osprey/interfaces/design_system/static/js/context-menu.js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/place-menu.js')} */
let pm;
/** @type {typeof import('../../../src/osprey/interfaces/design_system/static/js/context-menu.js')} */
let cm;
/** Scopes the listeners a test adds to the shared document. */
let scope = new AbortController();

beforeEach(async () => {
  scope = new AbortController();
  vi.resetModules();
  cm = await import(MENU);
  pm = await import(MODULE);
  document.body.innerHTML = `
    <div class="outer" tabindex="0">
      <div class="inner" tabindex="0"><span class="leaf">text</span></div>
      <input class="field" type="text">
    </div>
    <div class="elsewhere" tabindex="0"></div>`;
});

afterEach(() => {
  scope.abort();
  cm.closeContextMenu();
  pm.resetPlaceMenus([document]);
  document.body.innerHTML = '';
});

/** @param {string} sel */
const el = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));

/** @param {Document} [doc] */
const openMenu = (doc = document) => doc.querySelector('.osprey-context-menu[role="menu"]');

/**
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

/**
 * @param {Element} target
 * @param {KeyboardEventInit} init
 */
function press(target, init) {
  const e = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(e);
  return e;
}

/** @param {string} label */
const rows = (label) => [{ label, run: () => {} }];

describe('pointer route', () => {
  test('a right-click in a registered place opens its menu', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer actions' });
    pm.installPlaceMenus(document);

    const e = rightClick(el('.leaf'));

    expect(e.defaultPrevented).toBe(true);
    expect(build).toHaveBeenCalledWith(el('.outer'), e);
    const menu = openMenu();
    expect(menu?.getAttribute('aria-label')).toBe('Outer actions');
    expect(menu?.textContent).toContain('Copy');
  });

  test('the innermost place wins', () => {
    const outer = vi.fn(() => rows('Outer'));
    const inner = vi.fn(() => rows('Inner'));
    pm.registerPlace({ match: '.outer', build: outer, ariaLabel: 'Outer' });
    pm.registerPlace({
      match: (n) => n.classList.contains('inner'),
      build: inner,
      ariaLabel: (n) => `Menu ${n.className}`,
    });
    pm.installPlaceMenus(document);

    rightClick(el('.leaf'));

    expect(inner).toHaveBeenCalledTimes(1);
    expect(outer).not.toHaveBeenCalled();
    expect(openMenu()?.getAttribute('aria-label')).toBe('Menu inner');
  });

  test('zero rows open nothing and leave the browser its menu', () => {
    pm.registerPlace({ match: '.outer', build: () => [], ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);

    const e = rightClick(el('.leaf'));

    expect(e.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('a target outside every place is left alone', () => {
    pm.registerPlace({ match: '.outer', build: () => rows('Copy'), ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);

    const e = rightClick(el('.elsewhere'));

    expect(e.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('an event a surface already handled is ignored', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);
    // A surface's own listener — on the element or on the document itself.
    document.addEventListener('contextmenu', (e) => e.preventDefault(), { signal: scope.signal });

    rightClick(el('.leaf'));

    expect(build).not.toHaveBeenCalled();
    expect(openMenu()).toBeNull();
  });

  test('Shift+right-click declines to the browser', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);

    const e = rightClick(el('.leaf'), { shiftKey: true });

    expect(e.defaultPrevented).toBe(false);
    expect(build).not.toHaveBeenCalled();
  });

  test('a right-click in a text field declines to the browser', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);

    const e = rightClick(el('.field'));

    expect(e.defaultPrevented).toBe(false);
    expect(build).not.toHaveBeenCalled();
  });

  test('a removed place no longer opens', () => {
    const remove = pm.registerPlace({
      match: '.outer',
      build: () => rows('Copy'),
      ariaLabel: 'Outer',
    });
    pm.installPlaceMenus(document);
    remove();

    const e = rightClick(el('.leaf'));

    expect(e.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });
});

describe('keyboard route', () => {
  test('Shift+F10 on a registered place opens its menu at the element', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.inner', build, ariaLabel: 'Inner' });
    pm.installPlaceMenus(document);
    el('.inner').focus();

    const e = press(el('.inner'), { key: 'F10', shiftKey: true });

    expect(e.defaultPrevented).toBe(true);
    expect(build).toHaveBeenCalledWith(el('.inner'), e);
    expect(openMenu()?.getAttribute('aria-label')).toBe('Inner');
  });

  test('the ContextMenu key opens the focused element’s place menu', () => {
    pm.registerPlace({ match: '.outer', build: () => rows('Copy'), ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);

    const e = press(el('.inner'), { key: 'ContextMenu' });

    expect(e.defaultPrevented).toBe(true);
    expect(openMenu()?.getAttribute('aria-label')).toBe('Outer');
  });

  test('a keydown already handled is skipped', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.inner', build, ariaLabel: 'Inner' });
    pm.installPlaceMenus(document);
    el('.inner').addEventListener('keydown', (e) => e.preventDefault());

    press(el('.inner'), { key: 'F10', shiftKey: true });

    expect(build).not.toHaveBeenCalled();
    expect(openMenu()).toBeNull();
  });

  test('other keys and plain F10 do nothing', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.inner', build, ariaLabel: 'Inner' });
    pm.installPlaceMenus(document);

    expect(press(el('.inner'), { key: 'F10' }).defaultPrevented).toBe(false);
    expect(press(el('.inner'), { key: 'Enter', shiftKey: true }).defaultPrevented).toBe(false);
    expect(build).not.toHaveBeenCalled();
  });

  test('Shift+F10 in a text field declines to the browser', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);

    const e = press(el('.field'), { key: 'F10', shiftKey: true });

    expect(e.defaultPrevented).toBe(false);
    expect(build).not.toHaveBeenCalled();
  });

  test('zero rows from the keyboard leave the event alone', () => {
    pm.registerPlace({ match: '.inner', build: () => [], ariaLabel: 'Inner' });
    pm.installPlaceMenus(document);

    const e = press(el('.inner'), { key: 'F10', shiftKey: true });

    expect(e.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });
});

describe('installation', () => {
  test('installing twice on one document wires it once', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);
    pm.installPlaceMenus(document);

    rightClick(el('.leaf'));

    expect(build).toHaveBeenCalledTimes(1);
  });

  test('an uninstalled document opens nothing', () => {
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });

    const e = rightClick(el('.leaf'));

    expect(e.defaultPrevented).toBe(false);
    expect(build).not.toHaveBeenCalled();
  });

  test('a second document (a pop-out) gets its own menus, mounted in it', () => {
    const popout = document.implementation.createHTMLDocument('pop-out');
    popout.body.innerHTML = '<div class="outer" tabindex="0"><span class="leaf">x</span></div>';
    const build = vi.fn(() => rows('Copy'));
    pm.registerPlace({ match: '.outer', build, ariaLabel: 'Outer' });
    pm.installPlaceMenus(document);
    pm.installPlaceMenus(popout);
    pm.installPlaceMenus(popout);

    const leaf = /** @type {Element} */ (popout.querySelector('.leaf'));
    const e = rightClick(leaf);

    expect(e.defaultPrevented).toBe(true);
    expect(build).toHaveBeenCalledTimes(1);
    expect(openMenu(popout)).not.toBeNull();
    expect(openMenu(document)).toBeNull();

    cm.closeContextMenu();
    const k = press(/** @type {Element} */ (popout.querySelector('.outer')), {
      key: 'F10',
      shiftKey: true,
    });
    expect(k.defaultPrevented).toBe(true);
    expect(openMenu(popout)).not.toBeNull();
    pm.resetPlaceMenus([popout]);
  });
});

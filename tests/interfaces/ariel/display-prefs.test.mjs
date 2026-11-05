// @ts-check
/**
 * display-prefs.js: the badge-strip preference — defaults per frame kind,
 * persistence under a per-frame scoped key, change notification, and the
 * standalone switch wiring.
 *
 *   npx vitest run tests/interfaces/ariel/display-prefs.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/display-prefs.js')} PrefsMod */
/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} StoreMod */

/** @type {PrefsMod} */
let prefs;
/** @type {StoreMod} */
let store;

/** @param {string} url */
async function boot(url = '/') {
  history.replaceState(null, '', url);
  vi.resetModules();
  store = await import('../../../src/osprey/interfaces/ariel/static/js/store.js');
  prefs = await import('../../../src/osprey/interfaces/ariel/static/js/display-prefs.js');
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

afterEach(() => {
  history.replaceState(null, '', '/');
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

describe('defaults per frame', () => {
  test('standalone: on', async () => {
    await boot('/');
    expect(prefs.getStripPref()).toBe(true);
  });

  test('embedded and wide: on', async () => {
    await boot('/?embedded=true');
    store.setState({ stacked: false });
    expect(prefs.getStripPref()).toBe(true);
  });

  test('embedded and narrow (stacked): off', async () => {
    await boot('/?embedded=true');
    store.setState({ stacked: true });
    expect(prefs.getStripPref()).toBe(false);
  });

  test('standalone stays on when stacked', async () => {
    await boot('/');
    store.setState({ stacked: true });
    expect(prefs.getStripPref()).toBe(true);
  });
});

describe('persistence', () => {
  test('a choice is stored per frame kind and survives a reload', async () => {
    await boot('/');
    prefs.setStripPref(false);
    expect(localStorage.getItem('ariel-strip-standalone')).toBe('false');
    await boot('/');
    expect(prefs.getStripPref()).toBe(false);

    await boot('/?embedded=true');
    expect(prefs.getStripPref(), 'the embedded frame keeps its own default').toBe(true);
    prefs.setStripPref(false);
    store.setState({ stacked: false });
    expect(localStorage.getItem('ariel-strip-embedded')).toBe('false');
    expect(prefs.getStripPref()).toBe(false);
  });

  test('a stored choice beats the stacked default', async () => {
    await boot('/?embedded=true');
    prefs.setStripPref(true);
    store.setState({ stacked: true });
    expect(prefs.getStripPref()).toBe(true);
  });

  test('the key carries the storage scope', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'alice');
    await boot('/');
    prefs.setStripPref(false);
    expect(prefs.stripStorageKey()).toBe('ariel-strip-standalone--alice');
    expect(localStorage.getItem('ariel-strip-standalone--alice')).toBe('false');
    expect(localStorage.getItem('ariel-strip-standalone')).toBeNull();
  });

  test('a garbage stored value falls back to the default', async () => {
    localStorage.setItem('ariel-strip-standalone', 'maybe');
    await boot('/');
    expect(prefs.getStripPref()).toBe(true);
  });
});

describe('change notification', () => {
  test('setStripPref notifies subscribers until they unsubscribe', async () => {
    await boot('/');
    const cb = vi.fn();
    const off = prefs.onStripPrefChange(cb);
    prefs.setStripPref(false);
    expect(cb).toHaveBeenLastCalledWith(false);
    off();
    prefs.setStripPref(true);
    expect(cb).toHaveBeenCalledTimes(1);
  });

  test('a stacked flip notifies while the embedded default applies, not once a choice is stored', async () => {
    await boot('/?embedded=true');
    const cb = vi.fn();
    prefs.onStripPrefChange(cb);
    store.setState({ stacked: true });
    expect(cb).toHaveBeenLastCalledWith(false);
    prefs.setStripPref(true);
    cb.mockClear();
    store.setState({ stacked: false });
    expect(cb).not.toHaveBeenCalled();
  });
});

describe('standalone switch', () => {
  test('aria-checked follows the preference and a click flips it', async () => {
    await boot('/');
    const button = document.createElement('button');
    button.setAttribute('role', 'switch');
    document.body.appendChild(button);
    const unbind = prefs.bindStripSwitch(button);
    expect(button.getAttribute('aria-checked')).toBe('true');
    button.click();
    expect(button.getAttribute('aria-checked')).toBe('false');
    expect(prefs.getStripPref()).toBe(false);
    prefs.setStripPref(true);
    expect(button.getAttribute('aria-checked')).toBe('true');
    unbind();
    button.click();
    expect(prefs.getStripPref(), 'unbound: clicks do nothing').toBe(true);
    button.remove();
  });

  test('binding nothing is a no-op', async () => {
    await boot('/');
    expect(() => prefs.bindStripSwitch(null)()).not.toThrow();
  });
});

// @ts-check
/**
 * display-prefs.js: the badge-strip preference — its default, persistence
 * under a scoped key, and change notification.
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

describe('default', () => {
  test('wide: on', async () => {
    await boot('/?embedded=true');
    store.setState({ stacked: false });
    expect(prefs.getStripPref()).toBe(true);
  });

  test('narrow (stacked): off', async () => {
    await boot('/?embedded=true');
    store.setState({ stacked: true });
    expect(prefs.getStripPref()).toBe(false);
  });
});

describe('persistence', () => {
  test('a choice is stored and survives a reload', async () => {
    await boot('/?embedded=true');
    prefs.setStripPref(false);
    expect(localStorage.getItem('ariel-strip-embedded')).toBe('false');
    await boot('/?embedded=true');
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
    expect(prefs.stripStorageKey()).toBe('ariel-strip-embedded--alice');
    expect(localStorage.getItem('ariel-strip-embedded--alice')).toBe('false');
    expect(localStorage.getItem('ariel-strip-embedded')).toBeNull();
  });

  test('a garbage stored value falls back to the default', async () => {
    localStorage.setItem('ariel-strip-embedded', 'maybe');
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

  test('a stacked flip notifies while the default applies, not once a choice is stored', async () => {
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
